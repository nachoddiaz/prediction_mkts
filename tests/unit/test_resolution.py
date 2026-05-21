"""
tests/unit/test_resolution.py
───────────────────────────────
Tests de features/resolution.py.

Por qué controlamos 'now' en todos los tests:
  compute_resolution_features() usa datetime.now(UTC) por defecto.
  Si los tests usaran el tiempo real, serían no-deterministas —
  el mismo test podría pasar hoy y fallar mañana si resolution_date
  queda en el pasado. Pasando now explícitamente, los tests son
  completamente deterministas independientemente de cuándo se ejecuten.
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
    """Construye una fecha de resolución a N tiempo desde NOW."""
    delta = timedelta(days=days, hours=hours, minutes=minutes)
    return NOW + delta


# ---------------------------------------------------------------------------
# Tests — régimen NORMAL (τ ≥ 24h)
# ---------------------------------------------------------------------------


class TestNormal:
    def test_regime_normal(self) -> None:
        rf = compute_resolution_features(resolution_in(days=30), now=NOW)
        assert rf.regime == NearResolutionRegime.NORMAL

    def test_tau_years_correcto(self) -> None:
        rf = compute_resolution_features(resolution_in(days=365), now=NOW)
        assert rf.tau_years == pytest.approx(1.0, rel=1e-2)

    def test_tau_dias_correcto(self) -> None:
        rf = compute_resolution_features(resolution_in(days=7), now=NOW)
        assert rf.tau_days == pytest.approx(7.0, rel=1e-3)

    def test_tau_horas_correcto(self) -> None:
        rf = compute_resolution_features(resolution_in(hours=48), now=NOW)
        assert rf.tau_hours == pytest.approx(48.0, rel=1e-3)

    def test_sin_halt(self) -> None:
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
# Tests — régimen WARNING (1h ≤ τ < 24h)
# ---------------------------------------------------------------------------


class TestWarning:
    def test_regime_warning_a_12h(self) -> None:
        rf = compute_resolution_features(resolution_in(hours=12), now=NOW)
        assert rf.regime == NearResolutionRegime.WARNING

    def test_regime_warning_justo_en_umbral(self) -> None:
        """Exactamente en 24h debe ser WARNING, no NORMAL."""
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

    def test_sin_halt(self) -> None:
        rf = compute_resolution_features(resolution_in(hours=12), now=NOW)
        assert not rf.should_halt
        assert not rf.should_halt_side


# ---------------------------------------------------------------------------
# Tests — régimen CRITICAL (5min ≤ τ < 1h)
# ---------------------------------------------------------------------------


class TestCritical:
    def test_regime_critical_a_30min(self) -> None:
        rf = compute_resolution_features(resolution_in(minutes=30), now=NOW)
        assert rf.regime == NearResolutionRegime.CRITICAL

    def test_halt_side_activo(self) -> None:
        """§6.4: en CRITICAL se para el lado con inventario."""
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
# Tests — régimen HALT (τ < 5min)
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
# Tests — régimen RESOLVED (τ ≤ 0)
# ---------------------------------------------------------------------------


class TestResolved:
    def test_regime_resolved_en_pasado(self) -> None:
        past = NOW - timedelta(days=1)
        rf = compute_resolution_features(past, now=NOW)
        assert rf.regime == NearResolutionRegime.RESOLVED

    def test_tau_cero_en_pasado(self) -> None:
        past = NOW - timedelta(hours=1)
        rf = compute_resolution_features(past, now=NOW)
        assert rf.tau_years == 0.0
        assert rf.tau_days == 0.0
        assert rf.tau_hours == 0.0
        assert rf.tau_minutes == 0.0

    def test_halt_total_en_resolved(self) -> None:
        past = NOW - timedelta(days=1)
        rf = compute_resolution_features(past, now=NOW)
        assert rf.should_halt
        assert rf.q_max_fraction == 0.0

    def test_exactamente_en_resolution_date(self) -> None:
        """Exactamente en el momento de resolución → RESOLVED."""
        rf = compute_resolution_features(NOW, now=NOW)
        assert rf.regime == NearResolutionRegime.RESOLVED


# ---------------------------------------------------------------------------
# Tests — transiciones entre regímenes
# ---------------------------------------------------------------------------


class TestTransiciones:
    def test_transicion_normal_warning(self) -> None:
        """Un tick antes de 24h es WARNING, un tick después es NORMAL."""
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

    # ELIMINADO: tests de bernoulli_vol_safe — función obsoleta según MATH.md v2.1
    # La volatilidad ahora se calcula desde variación cuadrática de logit(p)
    # usando belief_vol_from_ticks(), no analíticamente desde p y τ.
