"""
tests/unit/test_params.py
──────────────────────────
Tests de strategies/market_making/params.py

Why synthetic data rather than DuckDB:
  With synthetic data we know the true parameters, so we can verify the
  calibrator recovers the right ones. If we generate an AR(1) with a known φ,
  the calibrator must estimate φ close to that value.

  Tests against real data would only check the code runs, not that it computes
  the right answer — and a wrong estimator would still pass the asserts.

Run with -s to see the output:
    uv run pytest tests/unit/test_params.py -v -s
"""

from __future__ import annotations

import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from strategies.market_making.params import (
    MIN_TICKS_CJ,
    MIN_TICKS_GLFT,
    CJCalibrator,
    CJResult,
    GLFTCalibrator,
    GLFTResult,
)

# ---------------------------------------------------------------------------
# Helpers de display
# ---------------------------------------------------------------------------

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"


def section(title: str) -> None:
    print(f"\n  {CYAN}{'─' * 50}{RESET}")
    print(f"  {BOLD}{title}{RESET}")
    print(f"  {CYAN}{'─' * 50}{RESET}")


def input_field(key: str, value: object) -> None:
    print(f"  {DIM}IN   {key:<28}{RESET} {value}")


def output_field(key: str, value: object, ok: bool = True) -> None:
    color = GREEN if ok else YELLOW
    print(f"  {color}OUT  {key:<28}{RESET} {color}{value}{RESET}")


def divider() -> None:
    print(f"  {DIM}{'·' * 50}{RESET}")


# ---------------------------------------------------------------------------
# Helpers for generating synthetic data
# ---------------------------------------------------------------------------


def make_ticks_df(
    n: int = 300,
    mid_start: float = 0.5,
    drift: float = 0.0,
    noise: float = 0.005,
    spread: float = 0.02,
    spread_noise: float = 0.005,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = datetime(2026, 4, 1, tzinfo=UTC)

    increments = rng.normal(drift, noise, n)
    mids = np.cumsum(increments) + mid_start
    mids = np.clip(mids, 0.01, 0.99)

    spreads = np.abs(rng.normal(spread, spread_noise, n))
    spreads = np.clip(spreads, 0.001, 0.1)

    return pd.DataFrame(
        {
            "timestamp": [base + timedelta(minutes=i) for i in range(n)],
            "mid": mids,
            "yes_bid": mids - spreads / 2,
            "yes_ask": mids + spreads / 2,
            "spread": spreads,
            "tick_type": "quote",
            "volume": 0.0,
            "side": None,
        }
    )


def make_features_df(
    ticks_df: pd.DataFrame,
    obi_signal: float = 0.1,
    noise: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(ticks_df)
    mids = ticks_df["mid"].values
    returns = np.diff(mids, prepend=mids[0])
    obi = obi_signal * returns + rng.normal(0, noise, n)
    obi = np.clip(obi, -1, 1)

    return pd.DataFrame(
        {
            "timestamp": ticks_df["timestamp"],
            "obi": obi,
            "quoted_spread": ticks_df["spread"],
            "relative_spread": ticks_df["spread"] / ticks_df["mid"],
            "belief_vol": np.sqrt(ticks_df["mid"] * (1 - ticks_df["mid"]) / 0.5),  # placeholder
            "ewma_vol": 0.01 * np.ones(n),
            "tau_years": 0.5 * np.ones(n),
            "mu_hat": obi,
        }
    )


# ---------------------------------------------------------------------------
# Tests GLFTCalibrator
# ---------------------------------------------------------------------------


class TestGLFTCalibrator:
    def test_returns_glft_result(self) -> None:
        section("GLFT — devuelve GLFTResult")

        df = make_ticks_df(n=200)
        input_field("n_ticks", len(df))
        input_field("spread_mean", f"{df['spread'].mean():.4f}")
        input_field("spread_std", f"{df['spread'].std():.4f}")
        input_field("mid_range", f"[{df['mid'].min():.4f}, {df['mid'].max():.4f}]")

        result = GLFTCalibrator().fit(df)

        output_field("type", type(result).__name__)
        output_field("kappa_p", f"{result.kappa_p:.4f}")
        output_field("A", f"{result.A:.6f}")
        output_field("log_likelihood", f"{result.log_likelihood:.2f}")
        output_field("n_observations", result.n_observations)

        assert isinstance(result, GLFTResult)

    def test_kappa_is_positive(self) -> None:
        section("GLFT — κ siempre positivo")

        df = make_ticks_df(n=200)
        input_field("n_ticks", len(df))
        input_field("spread", f"{df['spread'].mean():.4f}")

        result = GLFTCalibrator().fit(df)

        output_field("kappa_p", f"{result.kappa_p:.6f}", ok=result.kappa_p > 0)
        output_field("kappa_p > 0", result.kappa_p > 0)

        assert result.kappa_p > 0

    def test_A_positivo(self) -> None:
        section("GLFT — A siempre positivo")

        df = make_ticks_df(n=200)
        input_field("n_ticks", len(df))

        result = GLFTCalibrator().fit(df)

        output_field("A", f"{result.A:.8f}", ok=result.A > 0)
        output_field("A > 0", result.A > 0)

        assert result.A > 0

    def test_n_observations_is_correct(self) -> None:
        section("GLFT — n_observations consistente")

        n = 200
        df = make_ticks_df(n=n)
        input_field("n_ticks_input", n)

        result = GLFTCalibrator().fit(df)

        output_field("n_observations", result.n_observations)
        output_field("n <= n_input", result.n_observations <= n)
        output_field("n > 0", result.n_observations > 0)

        assert result.n_observations <= n
        assert result.n_observations > 0

    def test_log_likelihood_is_float(self) -> None:
        section("GLFT — log-likelihood es float")

        df = make_ticks_df(n=200)
        result = GLFTCalibrator().fit(df)

        output_field("log_likelihood", f"{result.log_likelihood:.4f}")
        output_field("type", type(result.log_likelihood).__name__)
        output_field("is not nan", not math.isnan(result.log_likelihood))

        assert isinstance(result.log_likelihood, float)

    def test_to_dict_has_kappa_and_A(self) -> None:
        section("GLFT — to_dict() tiene κ y A")

        df = make_ticks_df(n=200)
        result = GLFTCalibrator().fit(df)
        d = result.to_dict()

        output_field("dict keys", list(d.keys()))
        output_field("kappa_p", f"{d['kappa_p']:.4f}")
        output_field("A", f"{d['A']:.6f}")

        assert "kappa_p" in d
        assert "A" in d

    def test_summary_is_a_string(self) -> None:
        section("GLFT — summary() es string legible")

        df = make_ticks_df(n=200)
        result = GLFTCalibrator().fit(df)
        s = result.summary()

        print(f"\n{s}\n")

        assert isinstance(s, str)
        assert "κ" in s

    def test_raises_with_too_few_ticks(self) -> None:
        section("GLFT — ValueError con pocos ticks")

        n = MIN_TICKS_GLFT - 1
        df = make_ticks_df(n=n)
        input_field("n_ticks", n)
        input_field("MIN_TICKS_GLFT", MIN_TICKS_GLFT)
        input_field("n < MIN", n < MIN_TICKS_GLFT)

        with pytest.raises(ValueError, match="at least") as exc_info:
            GLFTCalibrator().fit(df)

        output_field("exception", type(exc_info.value).__name__)
        output_field("message", str(exc_info.value)[:60])

    def test_narrow_versus_wide_spread(self) -> None:
        section("GLFT — spread estrecho vs ancho")

        df_narrow = make_ticks_df(n=300, spread=0.01, seed=1)
        df_wide = make_ticks_df(n=300, spread=0.05, seed=1)

        input_field("spread_narrow", 0.01)
        input_field("spread_wide", 0.05)
        input_field("n_ticks", 300)

        r_narrow = GLFTCalibrator().fit(df_narrow)
        r_wide = GLFTCalibrator().fit(df_wide)

        divider()
        output_field("kappa_p_narrow", f"{r_narrow.kappa_p:.4f}")
        output_field("A_narrow", f"{r_narrow.A:.6f}")
        output_field("kappa_p_wide", f"{r_wide.kappa_p:.4f}")
        output_field("A_wide", f"{r_wide.A:.6f}")

        assert r_narrow.kappa_p > 0
        assert r_wide.kappa_p > 0

    def test_raises_on_missing_column(self) -> None:
        section("GLFT — ValueError si falta columna spread")

        df = make_ticks_df(n=200).drop(columns=["spread"])
        input_field("columns", list(df.columns))
        input_field("spread?", "spread" in df.columns)

        with pytest.raises(ValueError, match="Column") as exc_info:
            GLFTCalibrator().fit(df, quoted_spread_col="spread")

        output_field("exception", type(exc_info.value).__name__)
        output_field("message", str(exc_info.value)[:60])

    def test_alternative_column(self) -> None:
        section("GLFT — columna spread alternativa")

        df = make_ticks_df(n=200).rename(columns={"spread": "quoted_spread"})
        input_field("columna_usada", "quoted_spread")

        result = GLFTCalibrator().fit(df, quoted_spread_col="quoted_spread")

        output_field("kappa_p", f"{result.kappa_p:.4f}")
        output_field("A", f"{result.A:.6f}")

        assert result.kappa_p > 0


# ---------------------------------------------------------------------------
# Tests CJCalibrator
# ---------------------------------------------------------------------------


class TestCJCalibrator:
    def test_returns_cj_result(self) -> None:
        section("CJ — devuelve CJResult")

        ticks = make_ticks_df(n=300)
        features = make_features_df(ticks)

        input_field("n_ticks", len(ticks))
        input_field("n_features", len(features))
        input_field("obi_mean", f"{features['obi'].mean():.4f}")
        input_field("obi_std", f"{features['obi'].std():.4f}")

        result = CJCalibrator().fit(ticks, features)

        output_field("type", type(result).__name__)
        output_field("phi", f"{result.phi:.4f}")
        output_field("eta", f"{result.eta:.6f}")
        output_field("rho", f"{result.rho:.4f}")
        output_field("w_obi", f"{result.w_obi:.6f}")
        output_field("w_news", f"{result.w_news:.6f}")
        output_field("w_onchain", f"{result.w_onchain:.6f}")
        output_field("signal_r2", f"{result.signal_r2:.4f}")
        output_field("ar1_alpha", f"{result.ar1_alpha:.4f}")

        assert isinstance(result, CJResult)

    def test_phi_is_positive(self) -> None:
        section("CJ — φ siempre positivo")

        ticks = make_ticks_df(n=300)
        features = make_features_df(ticks)
        result = CJCalibrator().fit(ticks, features)

        input_field("n_ticks", len(ticks))
        output_field("phi", f"{result.phi:.6f}", ok=result.phi > 0)
        output_field("phi > 0", result.phi > 0)

        assert result.phi > 0

    def test_eta_is_positive(self) -> None:
        section("CJ — η siempre positivo")

        ticks = make_ticks_df(n=300)
        features = make_features_df(ticks)
        result = CJCalibrator().fit(ticks, features)

        input_field("n_ticks", len(ticks))
        output_field("eta", f"{result.eta:.8f}", ok=result.eta > 0)
        output_field("eta > 0", result.eta > 0)

        assert result.eta > 0

    def test_rho_within_range(self) -> None:
        section("CJ — ρ ∈ [-0.99, 0.99]")

        ticks = make_ticks_df(n=300)
        features = make_features_df(ticks)
        result = CJCalibrator().fit(ticks, features)

        input_field("n_ticks", len(ticks))
        output_field("rho", f"{result.rho:.6f}", ok=-0.99 <= result.rho <= 0.99)
        output_field("en [-0.99, 0.99]", -0.99 <= result.rho <= 0.99)

        assert -0.99 <= result.rho <= 0.99

    def test_ar1_alpha_within_range(self) -> None:
        section("CJ — α AR(1) ∈ (0, 1)")

        ticks = make_ticks_df(n=300)
        features = make_features_df(ticks)
        result = CJCalibrator().fit(ticks, features)

        output_field("ar1_alpha", f"{result.ar1_alpha:.6f}", ok=0 < result.ar1_alpha < 1)
        output_field("en (0, 1)", 0 < result.ar1_alpha < 1)

        assert 0 < result.ar1_alpha < 1

    def test_signal_r2_within_range(self) -> None:
        section("CJ — R² Ridge ∈ [0, 1]")

        ticks = make_ticks_df(n=300)
        features = make_features_df(ticks, obi_signal=0.3)
        result = CJCalibrator().fit(ticks, features)

        input_field("obi_signal", 0.3)
        output_field("signal_r2", f"{result.signal_r2:.6f}", ok=0 <= result.signal_r2 <= 1)
        output_field("en [0, 1]", 0 <= result.signal_r2 <= 1)

        assert 0.0 <= result.signal_r2 <= 1.0

    def test_to_dict_has_every_field(self) -> None:
        section("CJ — to_dict() tiene todos los campos")

        ticks = make_ticks_df(n=300)
        features = make_features_df(ticks)
        result = CJCalibrator().fit(ticks, features)
        d = result.to_dict()

        output_field("dict keys", sorted(d.keys()))
        for key, val in d.items():
            output_field(key, f"{val:.6f}")

        required = {"phi", "eta", "rho", "w_obi", "w_news", "w_onchain"}
        assert required.issubset(d.keys())

    def test_summary_is_a_string(self) -> None:
        section("CJ — summary() legible")

        ticks = make_ticks_df(n=300)
        features = make_features_df(ticks)
        result = CJCalibrator().fit(ticks, features)
        s = result.summary()

        print(f"\n{s}\n")

        assert isinstance(s, str)
        assert "φ" in s and "η" in s and "ρ" in s

    def test_raises_with_too_few_ticks(self) -> None:
        section("CJ — ValueError con pocos ticks")

        n = MIN_TICKS_CJ - 1
        ticks = make_ticks_df(n=n)
        features = make_features_df(ticks)

        input_field("n_ticks", n)
        input_field("MIN_TICKS_CJ", MIN_TICKS_CJ)

        with pytest.raises(ValueError, match="at least") as exc_info:
            CJCalibrator().fit(ticks, features)

        output_field("exception", type(exc_info.value).__name__)
        output_field("message", str(exc_info.value)[:60])

    def test_stronger_signal_raises_w_obi(self) -> None:
        section("CJ — mayor señal OBI → mayor w_obi")

        ticks = make_ticks_df(n=500, seed=42)
        f_strong = make_features_df(ticks, obi_signal=0.5, noise=0.01, seed=42)
        f_weak = make_features_df(ticks, obi_signal=0.01, noise=0.1, seed=42)

        input_field("obi_signal_strong", 0.5)
        input_field("obi_signal_weak", 0.01)
        input_field("n_ticks", 500)

        r_strong = CJCalibrator().fit(ticks, f_strong)
        r_weak = CJCalibrator().fit(ticks, f_weak)

        divider()
        output_field("w_obi_strong", f"{r_strong.w_obi:.6f}")
        output_field("w_obi_weak", f"{r_weak.w_obi:.6f}")
        output_field("r2_strong", f"{r_strong.signal_r2:.4f}")
        output_field("r2_weak", f"{r_weak.signal_r2:.4f}")
        output_field("|w_strong| >= |w_weak|·0.5", abs(r_strong.w_obi) >= abs(r_weak.w_obi) * 0.5)

        assert abs(r_strong.w_obi) >= abs(r_weak.w_obi) * 0.5

    def test_constant_series_does_not_raise(self) -> None:
        section("CJ — señal constante no lanza excepción")

        ticks = make_ticks_df(n=300)
        features = make_features_df(ticks, obi_signal=0.0, noise=0.0)

        input_field("obi_signal", 0.0)
        input_field("noise", 0.0)
        input_field("obi_std", f"{features['obi'].std():.8f}")

        result = CJCalibrator().fit(ticks, features)

        output_field("phi", f"{result.phi:.4f}")
        output_field("eta", f"{result.eta:.8f}")
        output_field("w_obi", f"{result.w_obi:.6f}")
        output_field("no exception", True)

        assert isinstance(result, CJResult)
        assert result.phi > 0

    def test_without_obi_column(self) -> None:
        section("CJ — sin columna OBI → pesos = 0")

        ticks = make_ticks_df(n=300)
        features = make_features_df(ticks).drop(columns=["obi"])

        input_field("columnas_features", list(features.columns))
        input_field("obi presente", "obi" in features.columns)

        result = CJCalibrator().fit(ticks, features)

        output_field("w_obi", f"{result.w_obi:.6f}", ok=result.w_obi == 0.0)
        output_field("w_obi == 0", result.w_obi == 0.0)

        assert result.w_obi == 0.0

    def test_ar1_recovers_synthetic_alpha(self) -> None:
        section("CJ — AR(1) recupera α sintético conocido")

        rng = np.random.default_rng(0)
        alpha = 0.95
        n = 500
        base = datetime(2026, 1, 1, tzinfo=UTC)

        mu = np.zeros(n)
        for i in range(1, n):
            mu[i] = alpha * mu[i - 1] + rng.normal(0, 0.01)

        ticks = pd.DataFrame(
            {
                "timestamp": [base + timedelta(minutes=i) for i in range(n)],
                "mid": np.clip(0.5 + np.cumsum(rng.normal(0, 0.005, n)), 0.01, 0.99),
                "spread": np.abs(rng.normal(0.02, 0.005, n)),
            }
        )

        features = pd.DataFrame(
            {
                "timestamp": ticks["timestamp"],
                "obi": mu,
            }
        )

        input_field("alpha_sintetico", alpha)
        input_field("n_ticks", n)
        input_field("mu_std", f"{mu.std():.6f}")

        result = CJCalibrator().fit(ticks, features)

        error = abs(result.ar1_alpha - alpha)
        output_field("alpha_estimado", f"{result.ar1_alpha:.4f}")
        output_field("alpha_real", f"{alpha:.4f}")
        output_field("error_abs", f"{error:.4f}", ok=error < 0.1)
        output_field("error < 0.1", error < 0.1)
        output_field("phi_estimado", f"{result.phi:.4f}")
        output_field("eta_estimado", f"{result.eta:.6f}")

        assert error < 0.1, f"AR(1) alpha: expected ~{alpha}, got {result.ar1_alpha:.4f}"

    def test_ridge_alpha_afecta_pesos(self) -> None:
        section("CJ — mayor Ridge α → pesos más pequeños")

        ticks = make_ticks_df(n=500, seed=42)
        features = make_features_df(ticks, obi_signal=0.3, seed=42)

        input_field("ridge_alpha_low", 0.01)
        input_field("ridge_alpha_high", 100.0)
        input_field("obi_signal", 0.3)

        r_low = CJCalibrator(ridge_alpha=0.01).fit(ticks, features)
        r_high = CJCalibrator(ridge_alpha=100.0).fit(ticks, features)

        output_field("w_obi_low_penalty", f"{r_low.w_obi:.6f}")
        output_field("w_obi_high_penalty", f"{r_high.w_obi:.6f}")
        output_field("|w_high| <= |w_low|·1.1", abs(r_high.w_obi) <= abs(r_low.w_obi) * 1.1)

        assert abs(r_high.w_obi) <= abs(r_low.w_obi) * 1.1
