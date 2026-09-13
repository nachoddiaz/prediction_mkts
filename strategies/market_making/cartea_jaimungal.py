"""
strategies/market_making/cartea_jaimungal.py
─────────────────────────────────────────────
Cartea-Jaimungal quoter in logit space — AS approximation (MATH.md §4.5).

Adds a short-horizon directional signal (the latent drift mu_t) that skews the
reservation price away from the mid.

Mapping to MATH.md v2.2:
  §4.4 → ODEs for phi_1 and phi_2
  §4.5 → reservation price with a directional signal
  §5.4 → reservation price and spread in logit space
  §6.4 → effective gamma under the near-resolution regimes
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

from features.resolution import (
    NearResolutionRegime,
    effective_gamma_for_regime,
)
from normalizer.price_grid import PriceLadder, ladder_for_venue
from normalizer.schema import MarketId
from strategies.market_making.glft import (
    MAX_HALF_SPREAD_X,
    Quote,
    apply_tick_floor,
    logit,
    sigma,
)

log = logging.getLogger(__name__)

# Measure-change discount ρ_μ ∈ (0,1] — §4.4.
# 0.5 means "trust half the signal". An honest placeholder: with no resolved
# markets there is no way to calibrate what μ̂ is worth out of sample, and an
# unvalidated signal does not deserve full confidence.
DEFAULT_RHO_MU = 0.5


class CarteaJaimungalQuoter:
    """
    Cartea-Jaimungal quoter in logit space.

    Equations (MATH.md §4.5 and §5.4):

      Integrated variance:
        σ̄²_b = σ²_b (constant under this approximation)

      Alpha-capture factor (phi_1) — §4.4:
        φ_1(τ) = ρ_μ · (1 − e^{−φτ}) / φ

        This is precisely the accumulated drift one unit of inventory can still
        capture over the remaining horizon:
            (1/μ_t)·∫_t^T E[μ_s|μ_t] ds = ∫_0^τ e^{−φu} du = (1−e^{−φτ})/φ

        v2.1 wrote φ_1 = (ρ·σ_b·η/φ)(1−e^{−φτ}), which is DIMENSIONALLY
        IMPOSSIBLE: with [μ̂] = nats/year, φ_1·μ̂ must come out in nats, so
        [φ_1] = years. But [ρ·σ_b·η/φ] = nats/year, making the product
        nats²/year². The ρ·σ_b·η factor belongs to the q² term (the cross
        variation d⟨S,μ⟩), not to the μq term, whose HJB coefficient is simply
        1. η does NOT enter φ_1.

      Reservation log-odds:
        r̃_X = X_t − q·γ_eff·σ̄²_b·τ + φ_1(τ)·μ̂_t

        The sign of the signal term is POSITIVE: a positive expected drift means
        the maker wants to accumulate long inventory, so it lifts both quotes.
        This follows from r̃ = S + ∂_q g — the midpoint between ask = S + δ^a and
        bid = S − δ^b, with δ^{a,b} = 1/κ ± ∂_q g from §4.3.

      Optimal half-spread in logit space (identical to GLFT):
        δ*/2 = γ_eff·σ̄²_b·τ/2 + (1/γ_eff)·ln(1 + γ_eff/κ_x)

      Quotes:
        bid_p = σ(r̃_X − δ*/2)
        ask_p = σ(r̃_X + δ*/2)
    """

    def __init__(
        self,
        gamma_I: float,
        kappa_x: float,
        phi: float,
        eta: float,
        rho: float,
    ) -> None:
        """
        Args:
            gamma_I: CARA inventory risk aversion.
            kappa_x: fill-curve decay in logit space.
            phi: mean-reversion speed of the latent drift (phi > 0), in 1/year.
                 A large phi means a signal that decays quickly.
            eta: volatility of the latent drift (eta > 0), in nats/year^{3/2}.
                 It does NOT enter phi_1 (see §4.4). It is retained because it
                 is part of the OU model calibrated by CJCalibrator and because
                 it prices the option to trade on future alpha, which lives in
                 phi_0 and does not affect the quotes. Reported as a diagnostic.
            rho: measure-change discount rho_mu ∈ (0, 1]. The signal is
                 estimated under P while the quotes live under Q; rho_mu shrinks
                 confidence in mu_hat, with 1 meaning "trust the signal fully".
                 Through v2.1 this parameter was used as if it were the
                 price-signal correlation inside phi_1, which not only broke the
                 units but permitted negative values that inverted the signal.
        """
        if gamma_I <= 0:
            raise ValueError(f"gamma_I must be positive, got {gamma_I}")
        if kappa_x <= 0:
            raise ValueError(f"kappa_x must be positive, got {kappa_x}")
        if phi <= 0:
            raise ValueError(f"phi must be positive, got {phi}")
        if eta <= 0:
            raise ValueError(f"eta must be positive, got {eta}")
        if not (0.0 < rho <= 1.0):
            raise ValueError(f"rho (rho_mu, measure-change discount) must be in (0, 1], got {rho}")

        self.gamma_I = gamma_I
        self.kappa_x = kappa_x
        self.phi = phi
        self.eta = eta
        self.rho = rho

    def quote(
        self,
        market_id: MarketId,
        mid_p: float,
        inventory: float,
        tau_years: float,
        belief_vol: float,
        regime: NearResolutionRegime,
        mu_hat: float = 0.0,
        timestamp: datetime | None = None,
        ladder: PriceLadder | None = None,
    ) -> Quote:
        """
        Compute Cartea-Jaimungal quotes, incorporating the mu_hat signal.
        """
        ts = timestamp or datetime.now(tz=UTC)
        grid = ladder if ladder is not None else ladder_for_venue(market_id.venue)

        # Immediate halt when the regime demands it
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

        # State in logit space
        X_t = logit(mid_p)

        # Integrated variance
        sigma_bar_sq = belief_vol**2 * tau_years

        # Effective risk aversion under the near-resolution regime (§6.4)
        gamma_eff = effective_gamma_for_regime(self.gamma_I, regime)

        # Signal skew: φ_1(τ)·μ̂_t, with φ_1(τ) = ρ_μ·(1 − e^{−φτ})/φ  (§4.4)
        # τ → 0 ⟹ φ_1 → 0: no horizon left over which to capture the drift.
        if tau_years > 0:
            phi_1 = (self.rho / self.phi) * (-math.expm1(-self.phi * tau_years))
            signal_skew = phi_1 * mu_hat
        else:
            phi_1 = 0.0
            signal_skew = 0.0

        # Reservation log-odds (§4.5): r̃ = S + ∂_q g = X - q·γ_eff·σ̄² + φ_1·μ̂
        reservation_X = X_t - inventory * gamma_eff * sigma_bar_sq + signal_skew

        # Optimal half-spread in logit — exact first-order condition (§2.4),
        # identical to GLFT
        inventory_term = gamma_eff * sigma_bar_sq / 2.0
        rent_term = (1.0 / gamma_eff) * math.log1p(gamma_eff / self.kappa_x)
        half_spread_X = inventory_term + rent_term

        # Sanity bound — see GLFTQuoter.quote
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

        # Quotes in logit space
        bid_X = reservation_X - half_spread_X
        ask_X = reservation_X + half_spread_X

        # Map back to price space
        bid_p_raw = sigma(bid_X)
        ask_p_raw = sigma(ask_X)

        # Aplicar tick floor
        bid_p, ask_p, is_valid, reason = apply_tick_floor(
            bid_p=bid_p_raw,
            ask_p=ask_p_raw,
            reservation_X=reservation_X,
            tau_years=tau_years,
            ladder=grid,
        )

        # The same two guards as GLFT — see GLFTQuoter.quote for the reasoning.
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

        if half_spread_clamped:
            is_valid = False
            reason = (
                f"half_spread_clamped: belief_vol={belief_vol:.4g} "
                f"tau_years={tau_years:.4g} kappa_x={self.kappa_x:.4g}"
            )

        return Quote(
            market_id=market_id,
            timestamp=ts,
            model="cartea_jaimungal",
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
            signal_skew=signal_skew,
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
        X_t = logit(mid_p)
        return Quote(
            market_id=market_id,
            timestamp=timestamp,
            model="cartea_jaimungal",
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
