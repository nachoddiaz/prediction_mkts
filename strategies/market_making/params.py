"""
strategies/market_making/params.py
────────────────────────────────────
Parameter calibration for the GLFT and Cartea-Jaimungal models: κ, A, φ, η
and ρ, estimated from the tick and feature history stored in DuckDB.

Two independent calibrators:

  GLFTCalibrator  → estimates κ and A by MLE over proxied fill events
  CJCalibrator    → estimates φ, η, ρ and w_i via ridge regression + AR(1)

Both take DataFrames, so they work equally well from notebooks and scripts.

Typical notebook flow:
    reader = MarketDataReader()
    ticks_df    = reader.ticks("kalshi:KXBTC-TEST", start, end)
    features_df = reader.features("kalshi:KXBTC-TEST", start, end)

    glft = GLFTCalibrator().fit(ticks_df)
    cj   = CJCalibrator().fit(ticks_df, features_df)

    update_model_params("kalshi", {**glft, **cj})

Relation to MATH.md:
  §3.2 → MLE for κ, A
  §4.6 → ridge for w_i, AR(1) for φ, η, ρ
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from features.microstructure import epoch_seconds

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Minimum observations to calibrate — below this there is no convergence
MIN_TICKS_GLFT = 100
MIN_TICKS_CJ = 200

# Parameter bounds for the MLE optimiser.
# They keep the optimiser out of economically meaningless regions.
KAPPA_BOUNDS = (0.01, 20.0)
A_BOUNDS = (1e-6, 10.0)

# Ridge penalty — regularisation against overfitting the w_i
# Tune it in the notebook if the weights come out unstable.
RIDGE_ALPHA = 1.0


# ---------------------------------------------------------------------------
# Dataclasses de resultados
# ---------------------------------------------------------------------------


@dataclass
class GLFTResult:
    """
    Result of a GLFT calibration.

    kappa_p: arrival-rate decay in price space (1/$).
             Calibrated directly from the observed spread.
             High → the book is deep and the maker can quote wide spreads.
             Low  → the book is thin and needs tight spreads to get filled.

    kappa_x: decay in logit space (dimensionless) — §3 MATH.md v2.1
             κ_x = κ_p · p̄(1-p̄), where p̄ is the average price.
             This is the value glft.py uses in X = logit(p) space.

    A:       baseline arrival rate, in orders per unit time at zero spread.
             Proportional to market volume.

    log_likelihood: log-likelihood at the optimum.
                    Useful for comparing calibrations across periods.

    n_observations: number of observations used.
    """

    kappa_p: float
    kappa_x: float
    A: float
    log_likelihood: float
    n_observations: int

    def to_dict(self) -> dict[str, float]:
        """Shaped to be passed straight into update_model_params()."""
        return {"kappa_p": self.kappa_p, "kappa_x": self.kappa_x, "A": self.A}

    def summary(self) -> str:
        return (
            f"GLFT Calibration\n"
            f"  κ_p = {self.kappa_p:.4f}  (arrival rate decay, price space)\n"
            f"  κ_x = {self.kappa_x:.4f}  (arrival rate decay, logit space)\n"
            f"  A   = {self.A:.6f}  (baseline arrival rate)\n"
            f"  log-likelihood = {self.log_likelihood:.2f}\n"
            f"  n = {self.n_observations} observations"
        )


@dataclass
class CJResult:
    """
    Result of a Cartea-Jaimungal calibration.

    phi: mean-reversion speed of μ_t.
         Higher φ → the drift reverts faster → a less persistent signal.

    eta: volatility of the latent drift.
         Higher η → the drift fluctuates more → a noisier signal.

    rho: correlation between price and signal innovations.
         ρ > 0 → a positive signal precedes price increases
         ρ < 0 → a positive signal precedes price decreases

    w_obi, w_news, w_onchain: weights of the composite signal,
         calibrated by ridge regression on subsequent returns.

    ar1_alpha:    AR(1) coefficient of the μ̂_t series
    signal_r2:    R² of the ridge regression
    n_obs_signal: observations used in the ridge fit
    n_obs_ar1:    observations used in the AR(1) fit
    """

    phi: float
    eta: float
    rho: float
    w_obi: float
    w_news: float
    w_onchain: float

    # Quality metrics
    ar1_alpha: float
    signal_r2: float
    n_obs_signal: int
    n_obs_ar1: int

    def to_dict(self) -> dict[str, float]:
        """Para pasar directamente a update_model_params()."""
        return {
            "phi": self.phi,
            "eta": self.eta,
            "rho": self.rho,
            "w_obi": self.w_obi,
            "w_news": self.w_news,
            "w_onchain": self.w_onchain,
        }

    def summary(self) -> str:
        return (
            f"Cartea-Jaimungal Calibration\n"
            f"  φ = {self.phi:.4f}  (mean-reversion speed)\n"
            f"  η = {self.eta:.6f}  (drift volatility)\n"
            f"  ρ = {self.rho:.4f}  (price-signal correlation)\n"
            f"  Signal weights:\n"
            f"    w_obi     = {self.w_obi:.6f}\n"
            f"    w_news    = {self.w_news:.6f}\n"
            f"    w_onchain = {self.w_onchain:.6f}\n"
            f"  Signal R² = {self.signal_r2:.4f}\n"
            f"  AR(1) α   = {self.ar1_alpha:.4f}\n"
            f"  n_signal  = {self.n_obs_signal}\n"
            f"  n_ar1     = {self.n_obs_ar1}"
        )


# ---------------------------------------------------------------------------
# GLFTCalibrator
# ---------------------------------------------------------------------------


class GLFTCalibrator:
    """
    Calibrate GLFT's κ and A by MLE over proxied fill events.

    Why a proxy and not real fills:
      There is no execution layer yet, so no fills are observed. The proxy
      uses the mid-to-mid spread between consecutive ticks: if the spread
      compressed significantly relative to the spread we would have quoted,
      we assume a fill occurred.

      Once the execution layer lands, the proxy is replaced by real fills —
      the calibrator is unchanged, only its input data.

    Usage:
        calibrator = GLFTCalibrator()
        result     = calibrator.fit(ticks_df)
        print(result.summary())
    """

    def fit(
        self,
        ticks_df: pd.DataFrame,
        quoted_spread_col: str = "spread",
    ) -> GLFTResult:
        """
        Calibrate κ and A from a tick series.

        Args:
            ticks_df:          DataFrame with 'spread' and 'timestamp' columns,
                               sorted by timestamp ascending. This is the
                               direct output of reader.ticks().
            quoted_spread_col: column holding the observed spread.

        Returns:
            A GLFTResult with κ, A and quality metrics.

        Raises:
            ValueError: if there are fewer than MIN_TICKS_GLFT observations.
        """
        df = ticks_df.copy().sort_values("timestamp").reset_index(drop=True)

        if len(df) < MIN_TICKS_GLFT:
            raise ValueError(
                f"GLFTCalibrator needs at least {MIN_TICKS_GLFT} ticks, "
                f"got {len(df)}. Accumulate more data before calibrating."
            )

        # Construir fill events proxy
        spreads, fills = self._build_fill_events(df, quoted_spread_col)

        if len(spreads) < MIN_TICKS_GLFT // 2:
            raise ValueError(
                f"Too few valid spread observations: {len(spreads)}. "
                f"Check that the spread column is not all NaN."
            )

        # MLE
        kappa_p, A, log_lik = self._mle(spreads, fills)

        # Conversión a espacio logit — §3 MATH.md v2.1
        # κ_x = κ_p · p̄(1-p̄)
        p_bar = df["mid"].mean() if "mid" in df.columns else 0.5
        kappa_x = kappa_p * p_bar * (1.0 - p_bar)

        result = GLFTResult(
            kappa_p=kappa_p,
            kappa_x=kappa_x,
            A=A,
            log_likelihood=log_lik,
            n_observations=len(spreads),
        )

        log.info(
            "GLFT calibration complete: kappa_p=%.4f kappa_x=%.4f A=%.6f ll=%.2f n=%d",
            kappa_p,
            kappa_x,
            A,
            log_lik,
            len(spreads),
        )

        return result

    def _build_fill_events(
        self,
        df: pd.DataFrame,
        spread_col: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build (quoted_spread, fill_proxy) pairs from the tick series.

        Proxy logic:
          For each tick i the quoted spread is spread[i]. If
          spread[i+1] < spread[i] * threshold we assume a fill at i. The
          threshold is 0.5 — if the spread halved, an execution is likely.

        Why this proxy:
          In a market with a maker, the spread compresses when someone crosses
          the book. Compression on the following tick is the most readily
          observable evidence that a fill occurred on the previous one.

        Returns:
            (spreads, fills): numpy arrays of equal length.
        """
        if spread_col not in df.columns:
            raise ValueError(f"Column '{spread_col}' not found. Available: {list(df.columns)}")

        spread_series = df[spread_col].dropna().values.astype(float)

        if len(spread_series) < 2:
            return np.array([]), np.array([])

        # Only use positive spreads (valid book)
        valid_mask = spread_series > 1e-6
        spreads_raw = spread_series[valid_mask]

        if len(spreads_raw) < 2:
            return np.array([]), np.array([])

        # Next spread (the fill proxy)
        spreads_current = spreads_raw[:-1]
        spreads_next = spreads_raw[1:]

        # Fill proxy: the spread compressed by more than 50%
        fill_threshold = 0.5
        fills = (spreads_next < spreads_current * fill_threshold).astype(float)

        log.debug(
            "Fill events: %d total, %d fills (%.1f%%)",
            len(fills),
            fills.sum(),
            100 * fills.mean(),
        )

        return spreads_current, fills

    def _mle(
        self,
        spreads: np.ndarray,
        fills: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Maximum-likelihood estimation of κ and A.

        Log-likelihood (Poisson process):
          ℓ(κ, A) = Σ_i [ f_i·ln(A·e^{-κδ_i}) - A·e^{-κδ_i} ]
                  = Σ_i [ f_i·(ln A - κδ_i) - A·e^{-κδ_i} ]

        Why -ℓ is minimised with scipy:
          scipy.optimize.minimize works with minimisation, so we negate the
          log-likelihood to turn maximisation into minimisation.

        Returns:
            (kappa, A, log_likelihood)
        """

        def neg_log_likelihood(params: np.ndarray) -> float:
            kappa, log_A = params
            A = np.exp(log_A)  # A is always positive

            # Arrival intensity per observation
            lam = A * np.exp(-kappa * spreads)

            # Poisson-process log-likelihood
            #   f_i=1: an arrival occurred, contributing ln(λ_i) - λ_i·Δt
            #   f_i=0: no arrival, contributing -λ_i·Δt
            # Simplified (Δt = 1 by normalisation):
            ll = np.sum(fills * np.log(lam + 1e-12) - lam)

            return float(-ll)

        # Multiple starting points, to avoid local minima
        best_result = None
        best_nll = np.inf

        initial_points = [
            [1.0, np.log(0.1)],
            [2.0, np.log(0.05)],
            [0.5, np.log(0.2)],
            [3.0, np.log(0.01)],
        ]

        for x0 in initial_points:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    neg_log_likelihood,
                    x0=x0,
                    method="L-BFGS-B",
                    bounds=[
                        KAPPA_BOUNDS,
                        (np.log(A_BOUNDS[0]), np.log(A_BOUNDS[1])),
                    ],
                    options={"maxiter": 1000, "ftol": 1e-10},
                )

            if result.success and result.fun < best_nll:
                best_nll = result.fun
                best_result = result

        if best_result is None:
            log.warning("MLE did not converge — using fallback values")
            return 1.5, 0.1, -np.inf

        kappa = float(best_result.x[0])
        A = float(np.exp(best_result.x[1]))
        ll = float(-best_result.fun)

        return kappa, A, ll


# ---------------------------------------------------------------------------
# CJCalibrator
# ---------------------------------------------------------------------------


class CJCalibrator:
    """
    Calibrate the Cartea-Jaimungal parameters φ, η, ρ and w_i.

    Three sequential steps (MATH.md §4.6):
      1. Ridge regression → weights w_i → the μ̂_t series
      2. AR(1) on μ̂_t → φ, η
      3. Empirical correlation → ρ

    Usage:
        calibrator = CJCalibrator()
        result     = calibrator.fit(ticks_df, features_df)
        print(result.summary())
    """

    def __init__(self, ridge_alpha: float = RIDGE_ALPHA) -> None:
        """
        Args:
            ridge_alpha: ridge penalty.
                         Higher α → weights closer to 0 → less overfitting.
                         Tune it if the weights are unstable across periods.
        """
        self.ridge_alpha = ridge_alpha

    def fit(
        self,
        ticks_df: pd.DataFrame,
        features_df: pd.DataFrame,
    ) -> CJResult:
        """
        Calibrate the CJ parameters from historical ticks and features.

        Args:
            ticks_df:    DataFrame with 'timestamp' and 'mid' columns — the
                         direct output of reader.ticks().

            features_df: DataFrame with 'timestamp' and 'obi' columns, and
                         optionally 'ewma_vol' — the direct output of
                         reader.features().

        Returns:
            A CJResult with φ, η, ρ, w_i and quality metrics.

        Raises:
            ValueError: if there are fewer than MIN_TICKS_CJ observations.
        """
        ticks = ticks_df.copy().sort_values("timestamp").reset_index(drop=True)
        features = features_df.copy().sort_values("timestamp").reset_index(drop=True)

        if len(ticks) < MIN_TICKS_CJ:
            raise ValueError(f"CJCalibrator needs at least {MIN_TICKS_CJ} ticks, got {len(ticks)}.")

        # Step 1 — ridge regression → weights w_i → the μ̂_t series
        weights, mu_hat_series, r2, n_signal = self._calibrate_signal(ticks, features)

        # Step 2 — AR(1) on μ̂_t → φ, η
        phi, eta, ar1_alpha, n_ar1 = self._calibrate_ar1(mu_hat_series, ticks)

        # Step 3 — empirical correlation → ρ
        rho = self._calibrate_rho(ticks, mu_hat_series, eta)

        result = CJResult(
            phi=phi,
            eta=eta,
            rho=rho,
            w_obi=float(weights.get("obi", 0.0)),
            w_news=float(weights.get("news", 0.0)),
            w_onchain=float(weights.get("onchain", 0.0)),
            ar1_alpha=ar1_alpha,
            signal_r2=r2,
            n_obs_signal=n_signal,
            n_obs_ar1=n_ar1,
        )

        log.info(
            "CJ calibration: phi=%.4f eta=%.6f rho=%.4f w_obi=%.4f r2=%.4f n=%d",
            phi,
            eta,
            rho,
            result.w_obi,
            r2,
            n_signal,
        )

        return result

    def _calibrate_signal(
        self,
        ticks: pd.DataFrame,
        features: pd.DataFrame,
    ) -> tuple[dict[str, float], pd.Series, float, int]:
        """
        Step 1: ridge regression to estimate the weights w_i.

        Target: next-tick return Δp_{t+1} = p_{t+1} - p_t
        Features: OBI_t (and, in future, News_t and OnChain_t)

        Why ridge and not OLS:
          The features can be correlated (multicollinearity). Ridge penalises
          large coefficients, producing weights that are more stable across
          different calibration periods.

        Returns:
            (weights_dict, mu_hat_series, r2, n_obs)
        """
        # Build log-odds increments ΔX_{t+1} = logit(p_{t+1}) - logit(p_t).
        # §5 MATH.md v2.1: the OU process for μ_t operates on X = logit(p) under ℙ,
        # not on p. Near the boundaries dX/dp = 1/(p(1-p)) → ∞, so using Δp
        # instead of ΔX overstates the edge in extreme markets.
        mids = ticks["mid"].values.astype(float)
        mids_clipped = np.clip(mids, 1e-6, 1 - 1e-6)
        X_logit = np.log(mids_clipped / (1.0 - mids_clipped))
        returns = np.diff(X_logit)  # ΔX_{t+1} for t=0..N-2

        # Available features — currently OBI only.
        # In future: add the news and onchain columns.
        available_features: dict[str, np.ndarray] = {}

        if "obi" in features.columns:
            # Align features with returns by timestamp.
            # Returns run from tick t to t+1, so we use the features at t.
            obi_values = features["obi"].values.astype(float)
            n = min(len(returns), len(obi_values) - 1)
            available_features["obi"] = obi_values[:n]
            returns_aligned = returns[:n]
        else:
            log.warning("CJ: 'obi' column not found in features_df — using zero signal")
            return {"obi": 0.0, "news": 0.0, "onchain": 0.0}, pd.Series([0.0] * len(ticks)), 0.0, 0

        if len(returns_aligned) < 10:
            log.warning("CJ: too few aligned observations for Ridge")
            return {"obi": 0.0, "news": 0.0, "onchain": 0.0}, pd.Series([0.0] * len(ticks)), 0.0, 0

        # Construir matriz de features X
        feature_names = list(available_features.keys())
        X = np.column_stack([available_features[k] for k in feature_names])
        y = returns_aligned

        # Standardise features so the ridge penalty is applied fairly
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        # Ridge regression
        ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        ridge.fit(X_std, y)

        r2 = float(ridge.score(X_std, y))

        # Unscale the coefficients — we want weights on the original scale:
        # w_raw = w_std / std(feature)
        weights_raw = ridge.coef_ / scaler.scale_

        weights_dict = {name: float(w) for name, w in zip(feature_names, weights_raw, strict=False)}
        # Unavailable features carry a weight of 0
        for name in ("news", "onchain"):
            weights_dict.setdefault(name, 0.0)

        # Build the series μ̂_t = Σ w_i · f_i(t)
        # Apply to every available feature row, not only those used in the fit
        mu_hat_values = np.zeros(len(features))
        if "obi" in features.columns:
            mu_hat_values = weights_dict["obi"] * features["obi"].values.astype(float)

        mu_hat_series = pd.Series(
            mu_hat_values,
            index=features.index,
            name="mu_hat",
        )

        log.debug(
            "Ridge regression: r2=%.4f weights=%s n=%d",
            r2,
            weights_dict,
            len(returns_aligned),
        )

        return weights_dict, mu_hat_series, r2, len(returns_aligned)

    def _calibrate_ar1(
        self,
        mu_hat: pd.Series,
        ticks: pd.DataFrame,
    ) -> tuple[float, float, float, int]:
        """
        Step 2: AR(1) on the μ̂_t series to estimate φ and η.

        Model: μ̂_{t+1} = α·μ̂_t + ε_t, where α = e^{-φ·Δt}

        From α we estimate:
          φ = -ln(α) / Δt
          η = ν·√(2φ / (1 - α²))
        where ν² = Var(ε_t)

        Why OLS rather than MLE for the AR(1):
          For a Gaussian AR(1), OLS is equivalent to MLE, and it is simpler
          and numerically faster.

        Returns:
            (phi, eta, alpha, n_obs)
        """
        values = mu_hat.dropna().values.astype(float)

        if len(values) < 10:
            log.warning("AR(1): too few observations — using defaults")
            return 1.0, 0.05, 0.99, 0

        # OLS: regression of μ̂_{t+1} on μ̂_t
        y_lag = values[:-1]
        y_next = values[1:]

        # Avoid division by zero when the series is constant
        var_lag = np.var(y_lag)
        if var_lag < 1e-12:
            log.warning("AR(1): signal is constant — phi and eta undetermined")
            return 1.0, 0.05, 0.99, 0

        # AR(1) coefficient
        alpha = float(np.sum(y_lag * y_next) / np.sum(y_lag**2))
        # Clip to (0, 1) — the drift must be stationary
        alpha = np.clip(alpha, 0.01, 0.9999)

        # Residuals and their variance
        residuals = y_next - alpha * y_lag
        nu2 = float(np.var(residuals))

        # Estimate Δt in years from the tick DataFrame's timestamps
        if "timestamp" in ticks.columns and len(ticks) > 1:
            # epoch_seconds() rather than `.astype("int64")/1e9`: DuckDB returns
            # datetime64[us], so the naive pattern gave Δt 1000× too small and
            # φ = -ln(α)/Δt came out 1000× too large — that is, pinned to the
            # top of the [0.01, 100] clip for any real series.
            dt_s = np.diff(epoch_seconds(ticks["timestamp"]))
            positive = dt_s[dt_s > 0]
            dt_years = (
                float(np.median(positive)) / (365.25 * 24 * 3600)
                if len(positive) > 0
                else 1.0 / (365.25 * 24 * 60)
            )
        else:
            dt_years = 1.0 / (365.25 * 24 * 60)  # default: 1 minuto

        # φ from α: α = e^{-φ·Δt} → φ = -ln(α)/Δt
        phi = float(-np.log(alpha) / max(dt_years, 1e-10))
        phi = np.clip(phi, 0.01, 100.0)

        # η from the residual variance
        # η² = ν²·(2φ/(1-α²))
        denom = 1 - alpha**2
        if denom < 1e-10:
            eta = float(np.sqrt(nu2))
        else:
            eta = float(np.sqrt(nu2 * 2 * phi / denom))
        eta = max(eta, 1e-8)

        log.debug(
            "AR(1): alpha=%.6f phi=%.4f eta=%.6f nu2=%.8f dt_years=%.8f n=%d",
            alpha,
            phi,
            eta,
            nu2,
            dt_years,
            len(y_lag),
        )

        return phi, eta, alpha, len(y_lag)

    def _calibrate_rho(
        self,
        ticks: pd.DataFrame,
        mu_hat: pd.Series,
        eta: float,
    ) -> float:
        """
        Step 3: empirical correlation between Δp_t and Δμ̂_t.

        ρ = Cov(Δp_t, Δμ̂_t) / (σ̂·η·Δt)

        where σ̂ is the empirical standard deviation of Δp_t.

        Why this estimator:
          ρ is the correlation between the price Brownian motion and the
          signal Brownian motion in the continuous model. On discrete data,
          the correlation between the increments Δp and Δμ̂ is its most direct
          counterpart.

        Returns:
            rho ∈ [-0.99, 0.99]
        """
        if "mid" not in ticks.columns:
            return 0.0

        # ρ is the correlation between the X_t Brownian and the μ_t Brownian,
        # both in logit space (§5 MATH.md v2.1). Using Δp introduces a
        # p(1-p) factor that biases ρ, especially far from 50%.
        mids = ticks["mid"].values.astype(float)
        mids_clipped = np.clip(mids, 1e-6, 1 - 1e-6)
        X_logit = np.log(mids_clipped / (1.0 - mids_clipped))
        dX = np.diff(X_logit)
        mu_vals = mu_hat.values.astype(float)

        n = min(len(dX), len(mu_vals) - 1)
        if n < 5:
            return 0.0

        dp_aligned = dX[:n]
        dmu_aligned = np.diff(mu_vals[: n + 1])

        std_dp = np.std(dp_aligned)
        std_dmu = np.std(dmu_aligned)

        if std_dp < 1e-12 or std_dmu < 1e-12:
            return 0.0

        rho = float(np.corrcoef(dp_aligned, dmu_aligned)[0, 1])

        # Clip — a ρ very close to ±1 causes numerical instability
        rho = float(np.clip(rho, -0.99, 0.99))

        log.debug("rho calibration: rho=%.4f", rho)

        return rho
