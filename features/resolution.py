"""
features/resolution.py
───────────────────────
Computation of tau and the near-resolution flags.

Why this file is separate from microstructure.py:
  tau is an input to nearly every formula in MATH.md — GLFT,
  Cartea-Jaimungal, belief volatility, Kelly. It is fundamental enough to
  deserve its own module, separate from the microstructure signals that
  depend on the order book.

  The near-resolution flags are business rules that belong alongside it: they
  drive the execution engine (circuit breakers) rather than being trading
  features. Keeping them here makes that distinction explicit.

Relation to MATH.md:
  - tau              → τ = T - t in years, appearing in every formula
  - σ_B(p, τ)        → uses tau directly
  - reservation price → p̃ = p - q·γ·p(1-p) [τ cancels against σ_B]
  - signal skew      → φ₁(t) = (ρση/φ)(1 - e^{-φτ})
  - near-resolution  → §6.4: quoting-halt rules
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

# ---------------------------------------------------------------------------
# Near-resolution thresholds — from MATH.md §6.4
#
# Why these specific values:
#   24h: inventory risk starts to become significant.
#        γ_effective = 2γ and Q_max = Q/2.
#   1h:  Bernoulli volatility starts to diverge quickly.
#        At most 1 contract of inventory is admitted.
#   5 min: extreme near resolution. The book is dominated by informed flow
#        (α → 1 in Glosten-Milgrom). Quoting halts entirely.
# ---------------------------------------------------------------------------
TAU_24H = 24.0 / (365.25 * 24)  # 24 hours, in years
TAU_1H = 1.0 / (365.25 * 24)  # 1 hour, in years
TAU_5MIN = 5.0 / (365.25 * 24 * 60)  # 5 minutes, in years


class NearResolutionRegime(str, Enum):
    """
    Near-resolution regime, per MATH.md §6.4.

    NORMAL      → τ ≥ 24h. The system runs with normal parameters.
    WARNING     → 1h ≤ τ < 24h. Reduce max inventory, double γ.
    CRITICAL    → 5min ≤ τ < 1h. Max inventory = 1, halt one side.
    HALT        → τ < 5min. Quoting halts entirely.
    RESOLVED    → τ ≤ 0. The market has already resolved.
    """

    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    HALT = "halt"
    RESOLVED = "resolved"


# Per-regime γ multipliers — MATH.md §6.4.
# A single table: compute_resolution_features() and effective_gamma_for_regime()
# read from here, so the two cannot drift apart.
GAMMA_MULTIPLIER_BY_REGIME: dict[NearResolutionRegime, float] = {
    NearResolutionRegime.NORMAL: 1.0,
    NearResolutionRegime.WARNING: 2.0,
    NearResolutionRegime.CRITICAL: 4.0,
    # HALT and RESOLVED do not quote; the value reflects the extreme risk and
    # keeps the function total over the enum.
    NearResolutionRegime.HALT: 4.0,
    NearResolutionRegime.RESOLVED: 1.0,
}


@dataclass(frozen=True)
class ResolutionFeatures:
    """
    Every feature derived from the time remaining to resolution.

    Why a frozen dataclass:
      Like the domain objects — immutable once computed. If you need new
      features, build a new object.

    Fields:
      tau_years          → τ in years — the direct input to every formula
      tau_days           → τ in days, for logging and dashboards
      tau_hours          → τ in hours, for near-resolution decisions
      tau_minutes        → τ in minutes, for halt decisions
      regime             → NearResolutionRegime per §6.4
      gamma_multiplier   → factor by which to scale γ in the model
      q_max_fraction     → fraction of Q_max permitted (1.0 = normal)
      should_halt        → True when all quoting must stop
      should_halt_side   → True when one side must stop quoting
    """

    tau_years: float
    tau_days: float
    tau_hours: float
    tau_minutes: float
    regime: NearResolutionRegime
    gamma_multiplier: float
    q_max_fraction: float
    should_halt: bool
    should_halt_side: bool


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_resolution_features(
    resolution_date: datetime,
    now: datetime | None = None,
) -> ResolutionFeatures:
    """
    Compute every resolution feature from the closing date.

    Why `now` is an optional parameter:
      Tests need to control time in order to verify that the regime and the
      flags switch at the right thresholds. With now=None it uses
      datetime.now(UTC) in production; with now=<datetime> in tests we can
      simulate any instant without mocks.

    Args:
        resolution_date: contract resolution date and time (UTC)
        now:             the current instant. None = datetime.now(UTC)

    Returns:
        A ResolutionFeatures with tau and every flag computed.
    """
    if now is None:
        now = datetime.now(tz=UTC)

    # Ensure both datetimes are UTC-aware
    if resolution_date.tzinfo is None:
        resolution_date = resolution_date.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    # --- Calcular tau en distintas unidades ---
    delta_seconds = (resolution_date - now).total_seconds()

    # Once the market has resolved, tau is 0 in every unit
    if delta_seconds <= 0:
        return ResolutionFeatures(
            tau_years=0.0,
            tau_days=0.0,
            tau_hours=0.0,
            tau_minutes=0.0,
            regime=NearResolutionRegime.RESOLVED,
            gamma_multiplier=1.0,  # no active trading
            q_max_fraction=0.0,  # no abrir nuevas posiciones
            should_halt=True,
            should_halt_side=True,
        )

    tau_years = delta_seconds / (365.25 * 24 * 3600)
    tau_days = delta_seconds / (24 * 3600)
    tau_hours = delta_seconds / 3600
    tau_minutes = delta_seconds / 60

    # --- Determine the regime per MATH.md §6.4 ---
    # Parameters for the exponential q_max (MATH.md §6.3)
    # q_max = Q_0 * exp(-r * psi(tau)) where psi(tau) = exp(-tau/tau_star)
    # r: oracle reversal probability (typically 0.001-0.01)
    # tau_star: characteristic time scale (typically 24h)
    r_oracle = 0.001  # TODO: move to config
    tau_star = TAU_24H  # 24 hours
    psi_tau = math.exp(-tau_years / tau_star)
    q_max_fraction_exp = math.exp(-r_oracle * psi_tau)

    if tau_years < TAU_5MIN:
        # τ < 5min: full halt
        # Informed traders dominate the book (α → 1 in Glosten-Milgrom)
        # σ_b diverges — no optimal spread is computable
        regime = NearResolutionRegime.HALT
        gamma_multiplier = GAMMA_MULTIPLIER_BY_REGIME[regime]
        q_max_fraction = 0.0  # open no new positions (overrides the exponential)
        should_halt = True
        should_halt_side = True

    elif tau_years < TAU_1H:
        # 5min ≤ τ < 1h: critical
        # Max inventory reduced exponentially
        regime = NearResolutionRegime.CRITICAL
        gamma_multiplier = GAMMA_MULTIPLIER_BY_REGIME[regime]
        q_max_fraction = min(0.1, q_max_fraction_exp)  # exponential capped at 10%
        should_halt = False
        should_halt_side = True  # halt the side carrying inventory

    elif tau_years < TAU_24H:
        # 1h ≤ τ < 24h: warning
        # Exponential decay of Q_max per §6.3
        regime = NearResolutionRegime.WARNING
        gamma_multiplier = GAMMA_MULTIPLIER_BY_REGIME[regime]
        q_max_fraction = min(0.5, q_max_fraction_exp)  # exponencial limitado al 50%
        should_halt = False
        should_halt_side = False

    else:
        # τ ≥ 24h: normal operation with a smooth exponential decay
        regime = NearResolutionRegime.NORMAL
        gamma_multiplier = GAMMA_MULTIPLIER_BY_REGIME[regime]
        q_max_fraction = q_max_fraction_exp  # the pure exponential formula
        should_halt = False
        should_halt_side = False

    return ResolutionFeatures(
        tau_years=tau_years,
        tau_days=tau_days,
        tau_hours=tau_hours,
        tau_minutes=tau_minutes,
        regime=regime,
        gamma_multiplier=gamma_multiplier,
        q_max_fraction=q_max_fraction,
        should_halt=should_halt,
        should_halt_side=should_halt_side,
    )


# ---------------------------------------------------------------------------
# Helpers for direct use from the execution engine
# ---------------------------------------------------------------------------


def effective_gamma_for_regime(gamma: float, regime: NearResolutionRegime) -> float:
    """
    γ_effective = γ · regime multiplier — MATH.md §6.4.

    Why this variant in addition to effective_gamma(gamma, rf):
      The quoters receive the resolved regime, not the full ResolutionFeatures.
      Without this function they would have to recompute the features (and
      with them a `now` different from the one that produced the regime), or —
      as happened through v2.1 — ignore the multiplier entirely.

    Args:
        gamma:  the base risk aversion γ_I
        regime: the near-resolution regime in force

    Returns:
        γ scaled: ×1 NORMAL, ×2 WARNING, ×4 CRITICAL.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    return gamma * GAMMA_MULTIPLIER_BY_REGIME[regime]


def effective_gamma(gamma: float, rf: ResolutionFeatures) -> float:
    """
    Return γ_effective = γ · gamma_multiplier.

    Used by GLFT and Cartea-Jaimungal to scale risk aversion according to the
    near-resolution regime.

    Args:
        gamma: the base risk-aversion coefficient
        rf:    previously computed ResolutionFeatures

    Returns:
        γ adjusted for the current regime
    """
    return gamma * rf.gamma_multiplier


def effective_q_max(q_max: float, rf: ResolutionFeatures) -> float:
    """
    Return Q_max_effective = Q_max · q_max_fraction.

    Used by the risk manager to cap inventory according to the
    near-resolution regime.

    Args:
        q_max: the maximum inventory limit under normal conditions
        rf:    previously computed ResolutionFeatures

    Returns:
        Q_max adjusted for the current regime
    """
    return q_max * rf.q_max_fraction


# REMOVED: bernoulli_vol_safe — obsolete under MATH.md v2.1.
# Volatility is now σ_b, calibrated from the quadratic variation of logit(p),
# rather than computed analytically from p and τ.
# The execution engine must use belief_vol_from_ticks() from microstructure.py
# and respect the should_halt and should_halt_side flags instead of relying on
# an arbitrary sentinel value (100.0).
