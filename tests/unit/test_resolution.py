"""
tests/unit/test_resolution.py
───────────────────────────────
Tests de features/resolution.py.

Why 'now' is controlled in every test:
  If the tests used real time they would be non-deterministic — the same test
  could pass today and fail tomorrow once resolution_date fell into the past.
  Passing `now` explicitly makes the tests fully deterministic regardless of
  when they run.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from features.resolution import (
    NearResolutionRegime,
    compute_resolution_features,
    effective_gamma,
    effective_q_max,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

NOW = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)


def resolution_in(
    days: float = 0,
    hours: float = 0,
    minutes: float = 0,
) -> datetime:
    """Build a resolution date N time units from now."""
    delta = timedelta(days=days, hours=hours, minutes=minutes)
    return NOW + delta


# ---------------------------------------------------------------------------
# Tests — NORMAL regime (τ ≥ 24h)
# ---------------------------------------------------------------------------


class TestNormal:
    def test_regime_normal(self) -> None:
        rf = compute_resolution_features(resolution_in(days=30), now=NOW)
        assert rf.regime == NearResolutionRegime.NORMAL

    def test_tau_years_is_correct(self) -> None:
        rf = compute_resolution_features(resolution_in(days=365), now=NOW)
        assert rf.tau_years == pytest.approx(1.0, rel=1e-2)

    def test_tau_days_is_correct(self) -> None:
        rf = compute_resolution_features(resolution_in(days=7), now=NOW)
        assert rf.tau_days == pytest.approx(7.0, rel=1e-3)

    def test_tau_hours_is_correct(self) -> None:
        rf = compute_resolution_features(resolution_in(hours=48), now=NOW)
        assert rf.tau_hours == pytest.approx(48.0, rel=1e-3)

    def test_no_halt(self) -> None:
        rf = compute_resolution_features(resolution_in(days=30), now=NOW)
        assert not rf.should_halt
        assert not rf.should_halt_side

    def test_gamma_multiplier_1(self) -> None:
        rf = compute_resolution_features(resolution_in(days=30), now=NOW)
        assert rf.gamma_multiplier == pytest.approx(1.0)

    def test_q_max_fraction_1(self) -> None:
        rf = compute_resolution_features(resolution_in(days=30), now=NOW)
        assert rf.q_max_fraction == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests — WARNING regime (1h ≤ τ < 24h)
# ---------------------------------------------------------------------------


class TestWarning:
    def test_regime_warning_a_12h(self) -> None:
        rf = compute_resolution_features(resolution_in(hours=12), now=NOW)
        assert rf.regime == NearResolutionRegime.WARNING

    def test_regime_warning_exactly_at_threshold(self) -> None:
        """Exactly at 24h it must be WARNING, not NORMAL."""
        rf = compute_resolution_features(resolution_in(hours=23, minutes=59), now=NOW)
        assert rf.regime == NearResolutionRegime.WARNING

    def test_gamma_multiplier_2(self) -> None:
        """§6.4: γ_effective = 2γ en WARNING."""
        rf = compute_resolution_features(resolution_in(hours=12), now=NOW)
        assert rf.gamma_multiplier == pytest.approx(2.0)

    def test_q_max_fraction_05(self) -> None:
        """§6.4: Q_max = Q/2 en WARNING."""
        rf = compute_resolution_features(resolution_in(hours=12), now=NOW)
        assert rf.q_max_fraction == pytest.approx(0.5)

    def test_no_halt(self) -> None:
        rf = compute_resolution_features(resolution_in(hours=12), now=NOW)
        assert not rf.should_halt
        assert not rf.should_halt_side


# ---------------------------------------------------------------------------
# Tests — CRITICAL regime (5min ≤ τ < 1h)
# ---------------------------------------------------------------------------


class TestCritical:
    def test_regime_critical_a_30min(self) -> None:
        rf = compute_resolution_features(resolution_in(minutes=30), now=NOW)
        assert rf.regime == NearResolutionRegime.CRITICAL

    def test_halt_side_activo(self) -> None:
        """§6.4: under CRITICAL the inventory-adding side is halted."""
        rf = compute_resolution_features(resolution_in(minutes=30), now=NOW)
        assert not rf.should_halt
        assert rf.should_halt_side

    def test_gamma_multiplier_4(self) -> None:
        rf = compute_resolution_features(resolution_in(minutes=30), now=NOW)
        assert rf.gamma_multiplier == pytest.approx(4.0)

    def test_q_max_fraction_01(self) -> None:
        rf = compute_resolution_features(resolution_in(minutes=30), now=NOW)
        assert rf.q_max_fraction == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Tests — HALT regime (τ < 5min)
# ---------------------------------------------------------------------------


class TestHalt:
    def test_regime_halt_a_1min(self) -> None:
        rf = compute_resolution_features(resolution_in(minutes=1), now=NOW)
        assert rf.regime == NearResolutionRegime.HALT

    def test_halt_total(self) -> None:
        """§6.4: τ < 5min → halt all quoting."""
        rf = compute_resolution_features(resolution_in(minutes=1), now=NOW)
        assert rf.should_halt
        assert rf.should_halt_side

    def test_q_max_fraction_0(self) -> None:
        rf = compute_resolution_features(resolution_in(minutes=1), now=NOW)
        assert rf.q_max_fraction == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests — RESOLVED regime (τ ≤ 0)
# ---------------------------------------------------------------------------


class TestResolved:
    def test_regime_resolved_in_the_past(self) -> None:
        past = NOW - timedelta(days=1)
        rf = compute_resolution_features(past, now=NOW)
        assert rf.regime == NearResolutionRegime.RESOLVED

    def test_zero_tau_in_the_past(self) -> None:
        past = NOW - timedelta(hours=1)
        rf = compute_resolution_features(past, now=NOW)
        assert rf.tau_years == 0.0
        assert rf.tau_days == 0.0
        assert rf.tau_hours == 0.0
        assert rf.tau_minutes == 0.0

    def test_full_halt_when_resolved(self) -> None:
        past = NOW - timedelta(days=1)
        rf = compute_resolution_features(past, now=NOW)
        assert rf.should_halt
        assert rf.q_max_fraction == 0.0

    def test_exactly_at_resolution_date(self) -> None:
        """Exactly at the resolution instant → RESOLVED."""
        rf = compute_resolution_features(NOW, now=NOW)
        assert rf.regime == NearResolutionRegime.RESOLVED


# ---------------------------------------------------------------------------
# Tests — transitions between regimes
# ---------------------------------------------------------------------------


class TestTransiciones:
    def test_transicion_normal_warning(self) -> None:
        """One tick before 24h is WARNING, one tick after is NORMAL."""
        antes = compute_resolution_features(resolution_in(hours=23, minutes=59), now=NOW)
        despues = compute_resolution_features(resolution_in(hours=24, minutes=1), now=NOW)
        assert antes.regime == NearResolutionRegime.WARNING
        assert despues.regime == NearResolutionRegime.NORMAL

    def test_transicion_warning_critical(self) -> None:
        antes = compute_resolution_features(resolution_in(minutes=59), now=NOW)
        despues = compute_resolution_features(resolution_in(hours=1, minutes=1), now=NOW)
        assert antes.regime == NearResolutionRegime.CRITICAL
        assert despues.regime == NearResolutionRegime.WARNING

    def test_transicion_critical_halt(self) -> None:
        antes = compute_resolution_features(resolution_in(minutes=4), now=NOW)
        despues = compute_resolution_features(resolution_in(minutes=6), now=NOW)
        assert antes.regime == NearResolutionRegime.HALT
        assert despues.regime == NearResolutionRegime.CRITICAL


# ---------------------------------------------------------------------------
# Tests — helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_effective_gamma_normal(self) -> None:
        rf = compute_resolution_features(resolution_in(days=30), now=NOW)
        assert effective_gamma(0.1, rf) == pytest.approx(0.1)

    def test_effective_gamma_warning(self) -> None:
        """En WARNING γ_effective = 2 * γ_base."""
        rf = compute_resolution_features(resolution_in(hours=12), now=NOW)
        assert effective_gamma(0.1, rf) == pytest.approx(0.2)

    def test_effective_gamma_halt(self) -> None:
        rf = compute_resolution_features(resolution_in(minutes=1), now=NOW)
        assert effective_gamma(0.1, rf) == pytest.approx(0.4)

    def test_effective_q_max_normal(self) -> None:
        rf = compute_resolution_features(resolution_in(days=30), now=NOW)
        assert effective_q_max(100.0, rf) == pytest.approx(100.0)

    def test_effective_q_max_warning(self) -> None:
        """En WARNING Q_max_effective = Q_max / 2."""
        rf = compute_resolution_features(resolution_in(hours=12), now=NOW)
        assert effective_q_max(100.0, rf) == pytest.approx(50.0)

    def test_effective_q_max_halt(self) -> None:
        rf = compute_resolution_features(resolution_in(minutes=1), now=NOW)
        assert effective_q_max(100.0, rf) == pytest.approx(0.0)

    # REMOVED: bernoulli_vol_safe tests — obsolete under MATH.md v2.1.
    # Volatility is now computed from the quadratic variation of logit(p)
    # via belief_vol_from_ticks(), not analytically from p and τ.
