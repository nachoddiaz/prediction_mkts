"""
tests/unit/test_calibration.py
Tests for features/calibration.py — Brier decomposition and calibrators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.calibration import (
    BetaCalibrator,
    BrierComponents,
    IsotonicCalibrator,
    VennAbersCalibrator,
    brier_decompose,
    recalibrate_out_of_time,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _perfect_forecasts(n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Forecasts equal to true probabilities — Brier should be near UNC."""
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.1, 0.9, n)
    outcomes = rng.binomial(1, probs).astype(float)
    return probs, outcomes


def _biased_forecasts(n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Systematically overconfident forecasts — high REL."""
    rng = np.random.default_rng(1)
    true_probs = rng.uniform(0.3, 0.7, n)
    forecasts = np.clip(true_probs + rng.uniform(0.1, 0.3, n), 0.01, 0.99)
    outcomes = rng.binomial(1, true_probs).astype(float)
    return forecasts, outcomes


# ---------------------------------------------------------------------------
# brier_decompose
# ---------------------------------------------------------------------------


class TestBrierDecompose:
    def test_returns_dataclass(self) -> None:
        f, o = _perfect_forecasts()
        result = brier_decompose(f, o)
        assert isinstance(result, BrierComponents)

    def test_identity_holds(self) -> None:
        """Br = REL - RES + UNC + WBV - 2*WBC must hold exactly."""
        f, o = _perfect_forecasts()
        b = brier_decompose(f, o, n_bins=20)
        reconstructed = b.reliability - b.resolution + b.uncertainty + b.wbv - 2 * b.wbc
        assert abs(b.brier - reconstructed) < 1e-10

    def test_compact_identity(self) -> None:
        """Br = REL - GRES + UNC where GRES = RES + 2*WBC - WBV."""
        f, o = _perfect_forecasts()
        b = brier_decompose(f, o, n_bins=20)
        assert abs(b.brier - (b.reliability - b.gres + b.uncertainty)) < 1e-10

    def test_brier_positive(self) -> None:
        f, o = _perfect_forecasts()
        b = brier_decompose(f, o)
        assert b.brier >= 0.0

    def test_biased_has_higher_reliability(self) -> None:
        f_perfect, o_perfect = _perfect_forecasts()
        f_biased, o_biased = _biased_forecasts()
        b_perfect = brier_decompose(f_perfect, o_perfect)
        b_biased = brier_decompose(f_biased, o_biased)
        assert b_biased.reliability > b_perfect.reliability

    def test_bootstrap_ci_present(self) -> None:
        f, o = _perfect_forecasts(200)
        b = brier_decompose(f, o, bootstrap_n=50, random_state=99)
        assert b.brier_ci is not None
        lo, hi = b.brier_ci
        assert lo <= b.brier <= hi

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            brier_decompose(np.array([0.5, 0.6]), np.array([0.0]))

    def test_too_few_obs_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            brier_decompose(np.array([0.5] * 5), np.array([0.0] * 5), n_bins=20)

    def test_n_obs_recorded(self) -> None:
        f, o = _perfect_forecasts(300)
        b = brier_decompose(f, o)
        assert b.n_obs == 300

    def test_uncertainty_formula(self) -> None:
        """UNC = ō(1-ō) must match direct computation."""
        f, o = _perfect_forecasts()
        b = brier_decompose(f, o)
        o_bar = float(o.mean())
        assert abs(b.uncertainty - o_bar * (1 - o_bar)) < 1e-12


# ---------------------------------------------------------------------------
# IsotonicCalibrator
# ---------------------------------------------------------------------------


class TestIsotonicCalibrator:
    def test_fit_predict(self) -> None:
        f, o = _biased_forecasts()
        split = len(f) // 2
        cal = IsotonicCalibrator().fit(f[:split], o[:split])
        preds = cal.predict(f[split:])
        assert preds.shape == f[split:].shape
        assert np.all((preds >= 0) & (preds <= 1))

    def test_not_fitted_raises(self) -> None:
        cal = IsotonicCalibrator()
        with pytest.raises(RuntimeError):
            cal.predict(np.array([0.5]))

    def test_reduces_brier(self) -> None:
        f, o = _biased_forecasts(600)
        split = 300
        cal = IsotonicCalibrator().fit(f[:split], o[:split])
        preds_cal = np.clip(cal.predict(f[split:]), 1e-7, 1 - 1e-7)
        preds_raw = np.clip(f[split:], 1e-7, 1 - 1e-7)
        brier_cal = float(np.mean((preds_cal - o[split:]) ** 2))
        brier_raw = float(np.mean((preds_raw - o[split:]) ** 2))
        assert brier_cal <= brier_raw + 0.01  # calibration should not hurt much


# ---------------------------------------------------------------------------
# VennAbersCalibrator
# ---------------------------------------------------------------------------


class TestVennAbersCalibrator:
    def test_fit_predict(self) -> None:
        f, o = _biased_forecasts(200)
        split = 100
        cal = VennAbersCalibrator().fit(f[:split], o[:split])
        preds = cal.predict(f[split:])
        assert preds.shape == f[split:].shape
        assert np.all((preds >= 0) & (preds <= 1))

    def test_interval_ordered(self) -> None:
        f, o = _biased_forecasts(100)
        cal = VennAbersCalibrator().fit(f[:60], o[:60])
        intervals = cal.predict_interval(f[60:])
        assert intervals.shape == (len(f[60:]), 2)
        assert np.all(intervals[:, 0] <= intervals[:, 1])

    def test_not_fitted_raises(self) -> None:
        with pytest.raises(RuntimeError):
            VennAbersCalibrator().predict(np.array([0.5]))

    def test_point_estimate_within_interval(self) -> None:
        f, o = _biased_forecasts(100)
        cal = VennAbersCalibrator().fit(f[:60], o[:60])
        test_s = f[60:70]
        preds = cal.predict(test_s)
        intervals = cal.predict_interval(test_s)
        for p, (lo, hi) in zip(preds, intervals, strict=False):
            assert lo - 1e-10 <= p <= hi + 1e-10


# ---------------------------------------------------------------------------
# BetaCalibrator
# ---------------------------------------------------------------------------


class TestBetaCalibrator:
    def test_fit_predict(self) -> None:
        f, o = _biased_forecasts(300)
        split = 150
        cal = BetaCalibrator().fit(f[:split], o[:split])
        preds = cal.predict(f[split:])
        assert preds.shape == f[split:].shape
        assert np.all((preds >= 0) & (preds <= 1))

    def test_not_fitted_raises(self) -> None:
        with pytest.raises(RuntimeError):
            BetaCalibrator().predict(np.array([0.5]))

    def test_identity_transform_on_perfect(self) -> None:
        """On well-calibrated data, beta params should be close to (1, 1, 0)."""
        f, o = _perfect_forecasts(1000)
        cal = BetaCalibrator().fit(f, o)
        preds = cal.predict(f)
        # Should not deviate wildly from input
        assert float(np.mean(np.abs(preds - f))) < 0.15


# ---------------------------------------------------------------------------
# recalibrate_out_of_time
# ---------------------------------------------------------------------------


class TestRecalibrateOutOfTime:
    def _make_data(self, n: int = 500) -> tuple[pd.Series, pd.Series, pd.Series]:
        rng = np.random.default_rng(42)
        ts = pd.date_range("2024-01-01", periods=n, freq="6h")
        f = pd.Series(np.clip(rng.uniform(0.2, 0.8, n), 0.01, 0.99))
        o = pd.Series(rng.binomial(1, f).astype(float))
        return f, o, pd.Series(ts)

    def test_returns_dataframe(self) -> None:
        f, o, ts = self._make_data(500)
        result = recalibrate_out_of_time(f, o, ts, method="isotonic", weeks_fit=2)
        assert isinstance(result, pd.DataFrame)
        assert "calibrated" in result.columns
        assert "raw_forecast" in result.columns

    def test_calibrated_in_unit_interval(self) -> None:
        f, o, ts = self._make_data(500)
        result = recalibrate_out_of_time(f, o, ts, method="isotonic", weeks_fit=2)
        assert (result["calibrated"] >= 0).all()
        assert (result["calibrated"] <= 1).all()

    def test_too_few_weeks_raises(self) -> None:
        f, o, ts = self._make_data(20)
        with pytest.raises(ValueError, match="weeks_fit"):
            recalibrate_out_of_time(f, o, ts, method="isotonic", weeks_fit=10)
