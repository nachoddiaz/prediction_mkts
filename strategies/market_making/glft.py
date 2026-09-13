"""
strategies/market_making/glft.py
─────────────────────────────────
GLFT quoter in logit space — Avellaneda-Stoikov approximation (MATH.md §5.4).

Operates entirely on X = logit(p) ∈ ℝ, where the classical models (AS, GLFT,
Cartea-Jaimungal) apply with their proofs intact.

Quotes are computed in X and mapped back to price via σ(X) = 1/(1+e^{-X}) at
the very end, so the quoted spread compresses near the boundaries automatically
— no clamping hack required.

Why the AS approximation and not the exact GLFT ODE system:
  Exact GLFT requires integrating 2Q+1 coupled ODEs backwards from T, one per
  inventory level. For Q=10 that is 21 equations and milliseconds per quote.
  The AS approximation in logit space is adequate until we have enough real
  data to measure the difference; the exact solver would land as glft_exact().

Mapping to MATH.md v2.2:
  §5.4 (2.1) → reservation log-odds
  §5.4 (2.2) → optimal half-spread, exact rent term
  §5.4 (2.3) → tick floor
  §6.4       → near-resolution regime multiplier on γ
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime

from features.resolution import (
    TAU_1H,
    NearResolutionRegime,
    effective_gamma_for_regime,
)
from normalizer.price_grid import PriceLadder, ladder_for_venue
from normalizer.schema import MarketId

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fallback tick. This is NOT the real tick of either target venue: Kalshi steps
# 0.001 in its main band and Polymarket is uniformly 0.001. It survives only as
# the default for markets without ladder metadata (Manifold, test fixtures).
# The source of truth is normalizer/price_grid.PriceLadder.
TICK = 0.01

# Log-odds of the quotable boundaries: σ(LOGIT_MIN) ≈ 0.01, σ(LOGIT_MAX) ≈ 0.99
LOGIT_MIN = math.log(TICK / (1 - TICK))  # ≈ -4.595
LOGIT_MAX = math.log((1 - TICK) / TICK)  # ≈  4.595

# Ladder used when the caller supplies none: one-cent steps.
_DEFAULT_LADDER = PriceLadder.uniform(TICK)

# Upper bound on the half-spread in logit space.
#
# Why it exists: δ_X enters σ(r̃ ± δ_X). A large δ_X saturates the sigmoid at 0
# and 1 — a quote of "bid at zero, ask at one", which is not a quote but the
# absence of one. Quoting wider than σ(±MAX_HALF_SPREAD_X) adds no information
# and hides an upstream problem (σ_b blown up, τ miscomputed, κ_x absurd).
#
# 6.0 nats → bid ≈ 0.0025, ask ≈ 0.9975 at zero inventory. Anything beyond is
# clamped AND the quote is marked invalid, so the symptom shows up in the trace
# instead of propagating as a plausible-looking number.
MAX_HALF_SPREAD_X = 6.0

# ---------------------------------------------------------------------------
# Default parameters — PLACEHOLDERS until calibrated on real venue data.
#
# Why these differ from the previous (0.1, 0.8):
#   Those values were implicitly tuned to an incorrect rent term. Under the
#   exact form (1/γ)·ln(1+γ/κ), κ_x = 0.8 yields a half-spread of 1.18 nats —
#   a 53-cent spread at p=0.5, which is not a quote but a refusal to quote.
#
#   κ_x is now chosen by the spread it implies, which is the observable
#   quantity: rent(γ=0.1, κ_x=12) = 0.083 nats → ≈ 4.1c spread at p=0.5,
#   consistent with a liquid prediction market.
#
# This does not replace calibration: GLFTCalibrator should estimate κ_x per
# market. These are the values to start from before any fills are observed.
DEFAULT_GAMMA_I = 0.1
DEFAULT_KAPPA_X = 12.0


# ---------------------------------------------------------------------------
# Logit primitives
# ---------------------------------------------------------------------------


def logit(p: float) -> float:
    """X = logit(p) = ln(p/(1-p)). Clamps p into (ε, 1-ε)."""
    p = max(1e-9, min(1 - 1e-9, p))
    return math.log(p / (1.0 - p))


def sigma(x: float) -> float:
    """
    p = σ(X) = 1/(1+e^{-X}). Inverse logit, numerically stable.

    Why two branches:
      The direct form 1/(1+exp(-x)) overflows with OverflowError for x < -709
      (math.exp(710) > DBL_MAX). That happened on real data: a mis-estimated
      σ_b blew up the half-spread and the backtester aborted with "math range
      error" instead of producing a PnL curve.

      For x ≥ 0 use 1/(1+e^{-x}) — the exponent is negative, no overflow.
      For x < 0 use e^{x}/(1+e^{x}) — the exponent is negative again.
      The two branches are algebraically the same function; neither overflows.
    """
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def sigma_prime(x: float) -> float:
    """σ'(X) = σ(X)·(1-σ(X)) = p(1-p). Jacobian of the logit transform."""
    p = sigma(x)
    return p * (1.0 - p)


# ---------------------------------------------------------------------------
# Quote dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Quote:
    """
    Complete result of one quoting cycle.

    Carries inputs, intermediate quantities and outputs — everything visible
    for debugging, backtesting and logging. The execution engine only reads
    bid_p, ask_p and is_valid.

    Why frozen:
      A Quote is a decision taken at time t. It must not mutate after
      construction; if conditions change, a new Quote is produced.
    """

    # --- Identity ---
    market_id: MarketId
    timestamp: datetime
    model: str  # "glft" | "cartea_jaimungal"

    # --- Model inputs ---
    mid_price_p: float  # p_t in price space ∈ (0,1)
    mid_price_X: float  # X_t = logit(p_t) ∈ ℝ
    inventory: float  # q_t — current maker position
    tau_years: float  # τ = T - t, in years
    regime: NearResolutionRegime

    # --- Parameters in force ---
    gamma_I: float  # γ_I — CARA inventory risk aversion
    kappa_x: float  # κ_x — fill-curve decay in logit space
    belief_vol: float  # σ_b — instantaneous volatility of logit(p)

    # --- Intermediate quantities ---
    sigma_bar_sq: float  # σ̄²_b·τ — integrated variance
    reservation_X: float  # r̃_X = X - q·γ_eff·σ̄²_b·τ
    half_spread_X: float  # δ*/2 in logit space
    signal_skew: float  # signal component (0.0 for pure GLFT)

    # --- Quotes in logit space ---
    bid_X: float  # r̃_X - δ*/2
    ask_X: float  # r̃_X + δ*/2

    # --- Quotes in price space — what reaches the market ---
    bid_p: float  # σ(bid_X), snapped to the price ladder
    ask_p: float  # σ(ask_X), snapped to the price ladder

    # --- Validity ---
    is_valid: bool  # False → the execution engine does not quote
    invalid_reason: str  # "" when is_valid is True

    @property
    def spread_p(self) -> float:
        """Spread in price space."""
        return self.ask_p - self.bid_p

    @property
    def spread_X(self) -> float:
        """Spread in logit space."""
        return self.ask_X - self.bid_X

    @property
    def mid_quoted_p(self) -> float:
        """Midpoint of our own quotes, in price space."""
        return (self.bid_p + self.ask_p) / 2.0

    def log_line(self) -> str:
        """Single-line log summary of the most relevant fields."""
        return (
            f"model={self.model} "
            f"p={self.mid_price_p:.4f} q={self.inventory:+.0f} "
            f"τ={self.tau_years * 365:.1f}d "
            f"σ_b={self.belief_vol:.4f} "
            f"r̃_p={sigma(self.reservation_X):.4f} "
            f"bid={self.bid_p:.4f} ask={self.ask_p:.4f} "
            f"spread={self.spread_p:.4f} "
            f"valid={self.is_valid}"
        )


def apply_tick_floor(
    bid_p: float,
    ask_p: float,
    reservation_X: float,
    tau_years: float,
    ladder: PriceLadder | None = None,
) -> tuple[float, float, bool, str]:
    """
    Snap quotes onto the market's quotable price grid (MATH.md §5.4 eq 2.3).

    Logic:
      Both sides inside the quotable range → returned snapped to the grid.
      Either side outside:
        τ < TAU_1H → near resolution, do not quote (is_valid=False)
        τ ≥ TAU_1H → anchor the pair at the minimum spread around σ(r̃_X)

    Why a ladder rather than a constant:
      TICK = 0.01 was wrong on both venues. Kalshi publishes per-market
      `price_ranges` — 0.001 in the main band, 0.0001 in the tails — and
      Polymarket declares minimum_tick_size = 0.001. With a fixed cent, a
      market quoting 0.0030/0.0040 received a bid of 0.0100: buying at a cent
      what the book offers at four tenths of one.

    Args:
        ladder: the market's grid. None falls back to a one-cent ladder, which
                remains correct for Manifold and for venue-agnostic tests.

    Returns:
        (bid_p, ask_p, is_valid, reason)
    """
    grid = ladder if ladder is not None else _DEFAULT_LADDER

    lo, hi = grid.min_quotable, grid.max_quotable

    # Snap to the grid ALWAYS, not only when the range is violated.
    #
    # σ(r̃ ± δ) returns an arbitrary real — 0.003202952327837194 — which no
    # venue accepts as a price. Previously snapping happened only inside the
    # tick-floor branch, so the normal path emitted unquotable orders.
    #
    # The bid snaps DOWN and the ask snaps UP: rounding can only widen the
    # spread, never tighten it. Rounding inward would quote more aggressively
    # than the model actually calls for.
    bid_snapped = grid.floor_to_grid(bid_p)
    ask_snapped = grid.ceil_to_grid(ask_p)

    if lo <= bid_snapped and ask_snapped <= hi:
        # Normal case — both sides inside the quotable range
        return bid_snapped, ask_snapped, True, ""

    # Range violated — the response depends on τ
    if tau_years < TAU_1H:
        # Near resolution — stop quoting rather than invent a spread
        return (
            bid_p,
            ask_p,
            False,
            (
                f"tick_floor_near_resolution: "
                f"bid={bid_p:.4f} ask={ask_p:.4f} tau_hours={tau_years * 8760:.1f}"
            ),
        )

    # Far from expiry — anchor at the minimum quotable spread.
    #
    # The original adjustment was `bid = max(TICK, r-TICK/2)`, `ask =
    # min(1-TICK, r+TICK/2)` with a guard that only fired when bid >= ask. Near
    # the boundaries it produced SUB-TICK spreads: at r ≈ 0.0119 it gave
    # bid=0.0100, ask=0.0169 — a 0.0069 spread that no venue accepts.
    #
    # Now the bid is anchored to the grid and the ask one tick above, shifting
    # the pair if it leaves the range. The resulting spread is always exactly
    # one tick and both sides land on the grid.
    reservation_p = sigma(reservation_X)
    tick = grid.tick_at(reservation_p)

    bid_p_adj = grid.floor_to_grid(reservation_p - tick / 2.0)
    bid_p_adj = min(max(bid_p_adj, lo), hi - tick)
    ask_p_adj = grid.ceil_to_grid(bid_p_adj + tick)

    return bid_p_adj, ask_p_adj, True, "tick_floor_adjusted"


# ---------------------------------------------------------------------------
# GLFTQuoter
# ---------------------------------------------------------------------------


class GLFTQuoter:
    """
    GLFT market maker in logit space — AS approximation (MATH.md §5.4).

    Equations:

      Integrated variance:
        σ̄²_b = σ²_b (constant under this approximation)

      Effective risk aversion (§6.4):
        γ_eff = γ_I · regime_multiplier(τ)

      Reservation log-odds:
        r̃_X(t,q) = X_t − q·γ_eff·σ̄²_b·τ                        (2.1)

      Optimal half-spread in logit space:
        δ*/2 = γ_eff·σ̄²_b·τ/2 + (1/γ_eff)·ln(1 + γ_eff/κ_x)     (2.2)

      Quotes in price space:
        bid_p = σ(r̃_X − δ*/2)
        ask_p = σ(r̃_X + δ*/2)

      Tick floor (eq 2.3): both sides snapped to the market's price ladder;
      outside the quotable range, τ decides whether to widen or stop quoting.

    Usage:
        quoter = GLFTQuoter(gamma_I=DEFAULT_GAMMA_I, kappa_x=DEFAULT_KAPPA_X)
        quote  = quoter.quote(
            market_id=mid,
            mid_p=0.45,
            inventory=3,
            tau_years=0.19,
            belief_vol=0.12,
            regime=NearResolutionRegime.NORMAL,
        )
        if quote.is_valid:
            execution_engine.submit(quote.bid_p, quote.ask_p)
    """

    def __init__(
        self,
        gamma_I: float,
        kappa_x: float,
    ) -> None:
        """
        Args:
            gamma_I: CARA inventory risk aversion (1/$). Higher γ_I widens
                     spreads and shrinks the inventory the maker will carry.

            kappa_x: fill-curve decay in logit space (dimensionless), estimated
                     by GLFTCalibrator. Higher κ_x means fills are more
                     sensitive to the quoted spread.
        """
        if gamma_I <= 0:
            raise ValueError(f"gamma_I must be positive, got {gamma_I}")
        if kappa_x <= 0:
            raise ValueError(f"kappa_x must be positive, got {kappa_x}")

        self.gamma_I = gamma_I
        self.kappa_x = kappa_x

    def quote(
        self,
        market_id: MarketId,
        mid_p: float,
        inventory: float,
        tau_years: float,
        belief_vol: float,
        regime: NearResolutionRegime,
        timestamp: datetime | None = None,
        ladder: PriceLadder | None = None,
    ) -> Quote:
        """
        Compute the optimal quotes for the current market state.

        Args:
            market_id:  canonical market identifier
            mid_p:      current mid price ∈ (0, 1)
            inventory:  current signed maker position, in contracts
            tau_years:  time to resolution, in years
            belief_vol: σ_b — instantaneous volatility of logit(p), produced by
                        belief_vol_from_ticks() in features/microstructure.py
            regime:     NearResolutionRegime from features/resolution.py
            ladder:     the market's quotable price grid. None falls back to the
                        venue default, which is correct unless Kalshi publishes
                        a per-market ladder — in which case pass the one carried
                        on Market.price_ladder.

        Returns:
            A fully populated Quote. The execution engine only needs is_valid,
            bid_p and ask_p.
        """
        ts = timestamp or datetime.now(tz=UTC)
        grid = ladder if ladder is not None else ladder_for_venue(market_id.venue)

        # --- Immediate halt when the regime demands it ---
        if regime in (NearResolutionRegime.HALT, NearResolutionRegime.RESOLVED):
            return self._invalid_quote(
                market_id=market_id,
                timestamp=ts,
                mid_p=mid_p,
                inventory=inventory,
                tau_years=tau_years,
                belief_vol=belief_vol,
                regime=regime,
                reason=f"regime={regime.value}",
            )

        # --- State in logit space ---
        X_t = logit(mid_p)

        # --- Integrated variance σ̄²_b·τ ---
        # Under this approximation σ_b is constant, so σ̄²_b = σ²_b.
        # A later version would integrate σ²_b(u) over [t,T].
        sigma_bar_sq = belief_vol**2 * tau_years

        # --- Effective risk aversion (§6.4) ---
        # γ_eff = γ_I · regime multiplier: ×1 NORMAL, ×2 WARNING, ×4 CRITICAL.
        #
        # Through v2.1 this multiplier was computed in resolution.py and ignored
        # here, which made the whole near-resolution extension of MATH.md §6
        # decorative: the system neither widened spreads nor reduced inventory
        # appetite as resolution approached — it only stopped quoting under five
        # minutes.
        gamma_eff = effective_gamma_for_regime(self.gamma_I, regime)

        # --- Reservation log-odds (2.1) ---
        # r̃_X = X_t − q·γ_eff·σ̄²_b·τ
        # Inventory skew shifts the centre of the quotes:
        #   q > 0 (long)  → r̃_X < X_t → quote lower to sell down
        #   q < 0 (short) → r̃_X > X_t → quote higher to buy back
        reservation_X = X_t - inventory * gamma_eff * sigma_bar_sq

        # --- Optimal half-spread in logit space (2.2) ---
        # δ*/2 = γ_eff·σ̄²_b·τ/2 + (1/γ_eff)·ln(1 + γ_eff/κ_x)
        #
        # First term:  compensation for inventory risk.
        # Second term: the microstructure rent — a lower bound on the spread,
        #              asymptotically independent of inventory and of time.
        #
        # The rent term is the EXACT first-order condition from A-S (§2.4), not
        # the (1/κ_x)·ln(1+γ/κ_x) approximation used through v2.1. That
        # expression is neither the exact form nor its γ≪κ limit (which is
        # 1/κ_x); it sits below both. At γ_I=0.1, κ_x=0.8 it understated the
        # rent eightfold.
        inventory_term = gamma_eff * sigma_bar_sq / 2.0
        rent_term = (1.0 / gamma_eff) * math.log1p(gamma_eff / self.kappa_x)
        half_spread_X = inventory_term + rent_term

        # --- Sanity bound on the half-spread ---
        # A runaway δ_X does not produce a wide quote, it produces σ(±δ) = {0,1},
        # which is not quoting at all. Clamp AND mark the quote invalid so the
        # cause (σ_b blown up, absurd τ, uncalibrated κ_x) is visible in the
        # trace rather than propagating as a plausible-looking number.
        half_spread_clamped = False
        if not math.isfinite(half_spread_X) or half_spread_X > MAX_HALF_SPREAD_X:
            log.warning(
                "half_spread_clamped: market=%s raw=%.4g max=%.2f "
                "belief_vol=%.4g tau_years=%.4g gamma_eff=%.4g kappa_x=%.4g",
                market_id,
                half_spread_X,
                MAX_HALF_SPREAD_X,
                belief_vol,
                tau_years,
                gamma_eff,
                self.kappa_x,
            )
            half_spread_X = MAX_HALF_SPREAD_X
            half_spread_clamped = True

        # --- Quotes in logit space ---
        bid_X = reservation_X - half_spread_X
        ask_X = reservation_X + half_spread_X

        # --- Map back to price space via σ(X) ---
        bid_p_raw = sigma(bid_X)
        ask_p_raw = sigma(ask_X)

        # --- Tick floor (§5.4 eq 2.3) ---
        bid_p, ask_p, is_valid, reason = apply_tick_floor(
            bid_p=bid_p_raw,
            ask_p=ask_p_raw,
            reservation_X=reservation_X,
            tau_years=tau_years,
            ladder=grid,
        )

        # --- Coherence guard: the quote must straddle the reservation price ---
        #
        # The correct invariant is bid ≤ σ(r̃_X) ≤ ask, NOT bid ≤ mid ≤ ask.
        # A maker carrying inventory deliberately shifts BOTH quotes to the same
        # side of the mid in order to flatten; requiring them to straddle the
        # mid rejected exactly the inventory skew the model exists to produce.
        #
        # (An earlier version did check against the mid. It was a patch for the
        # global 0.01 tick, which pushed the bid to 0.0100 in markets quoting in
        # thousandths. Per-venue ladders removed that cause, at which point the
        # patch became actively harmful.)
        reservation_p = sigma(reservation_X)
        if not (bid_p <= reservation_p <= ask_p):
            log.warning(
                "quote_does_not_straddle_reservation: market=%s r=%.4f bid=%.4f ask=%.4f",
                market_id,
                reservation_p,
                bid_p,
                ask_p,
            )
            is_valid = False
            reason = (
                f"quote_does_not_straddle_reservation: r={reservation_p:.4f} "
                f"bid={bid_p:.4f} ask={ask_p:.4f}"
            )

        # A clamped half-spread means the model left its range of validity. The
        # quote is still returned, but marked invalid so the router will not
        # send it and the symptom stays visible in the trace.
        if half_spread_clamped:
            is_valid = False
            reason = (
                f"half_spread_clamped: belief_vol={belief_vol:.4g} "
                f"tau_years={tau_years:.4g} kappa_x={self.kappa_x:.4g}"
            )

        return Quote(
            market_id=market_id,
            timestamp=ts,
            model="glft",
            mid_price_p=mid_p,
            mid_price_X=X_t,
            inventory=inventory,
            tau_years=tau_years,
            regime=regime,
            gamma_I=self.gamma_I,
            kappa_x=self.kappa_x,
            belief_vol=belief_vol,
            sigma_bar_sq=sigma_bar_sq,
            reservation_X=reservation_X,
            half_spread_X=half_spread_X,
            signal_skew=0.0,  # GLFT puro — sin señal direccional
            bid_X=bid_X,
            ask_X=ask_X,
            bid_p=bid_p,
            ask_p=ask_p,
            is_valid=is_valid,
            invalid_reason=reason,
        )

    def _invalid_quote(
        self,
        market_id: MarketId,
        timestamp: datetime,
        mid_p: float,
        inventory: float,
        tau_years: float,
        belief_vol: float,
        regime: NearResolutionRegime,
        reason: str,
    ) -> Quote:
        """Build an invalid Quote — for halt and near-resolution cases."""
        X_t = logit(mid_p)
        return Quote(
            market_id=market_id,
            timestamp=timestamp,
            model="glft",
            mid_price_p=mid_p,
            mid_price_X=X_t,
            inventory=inventory,
            tau_years=tau_years,
            regime=regime,
            gamma_I=self.gamma_I,
            kappa_x=self.kappa_x,
            belief_vol=belief_vol,
            sigma_bar_sq=0.0,
            reservation_X=X_t,
            half_spread_X=0.0,
            signal_skew=0.0,
            bid_X=X_t,
            ask_X=X_t,
            bid_p=mid_p,
            ask_p=mid_p,
            is_valid=False,
            invalid_reason=reason,
        )
