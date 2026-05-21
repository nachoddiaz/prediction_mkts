"""
features/calibration.py
────────────────────────
Brier Score decomposition (§8.1 MATH.md v2.1) and recalibration (§8.2).

Two components:
  1. brier_decompose()   — 5-component decomposition with within-bin terms
  2. Calibrators         — VennAbersCalibrator (primary), BetaCalibrator (fallback),
                           IsotonicCalibrator (baseline comparison)

§8.1 v2.1 correction (defect Q):
  The standard identity Br = REL - RES + UNC holds only in expectation.
  With finite K-bin quantile binning, two within-bin terms appear:
    Br = REL - RES + UNC + WBV - 2·WBC
  where WBV = within-bin forecast variance, WBC = within-bin forecast-outcome covariance.

§8.2 correction (defect R):
  Isotonic regression is biased on autocorrelated series.
  Use Venn-Abers predictors + out-of-time split instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

# ---------------------------------------------------------------------------
# Brier decomposition
# ---------------------------------------------------------------------------


@dataclass
class BrierComponents:
    """
    5-component Brier decomposition per §8.1 MATH.md v2.1.

    Identity:   Br = REL - RES + UNC + WBV - 2·WBC
    Compact:    Br = REL - GRES + UNC   (bin-width invariant)
    where:      GRES = RES + 2·WBC - WBV
    """

    brier: float
    reliability: float  # REL: mean squared calibration error per bin
    resolution: float  # RES: bin mean outcome variance from climatology
    uncertainty: float  # UNC: climatological ō(1-ō)
    wbv: float  # WBV: within-bin forecast variance
    wbc: float  # WBC: within-bin forecast-outcome covariance
    gres: float  # GRES = RES + 2·WBC - WBV
    n_bins: int
    n_obs: int
    brier_ci: tuple[float, float] | None = field(default=None)
    reliability_ci: tuple[float, float] | None = field(default=None)
    resolution_ci: tuple[float, float] | None = field(default=None)

    def summary(self) -> str:
        lines = [
            f"Brier Decomposition (K={self.n_bins}, n={self.n_obs})",
            f"  Br   = {self.brier:.6f}",
            f"  REL  = {self.reliability:.6f}  (calibration error; lower is better)",
            f"  RES  = {self.resolution:.6f}  (discrimination; higher is better)",
            f"  UNC  = {self.uncertainty:.6f}  (climatological uncertainty)",
            f"  WBV  = {self.wbv:.6f}  (within-bin forecast variance)",
            f"  WBC  = {self.wbc:.6f}  (within-bin forecast-outcome covariance)",
            f"  GRES = {self.gres:.6f}  (generalised resolution; bin-width invariant)",
        ]
        if self.brier_ci:
            lines.append(f"  Br 95% CI = [{self.brier_ci[0]:.6f}, {self.brier_ci[1]:.6f}]")
        return "\n".join(lines)


def _compute_components(
    forecasts: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int,
) -> tuple[float, float, float, float, float, float]:
    """
    Inner computation — returns (brier, rel, res, wbv, wbc, unc).
    Uses quantile bins to guarantee non-empty bins.
    """
    n = len(forecasts)
    o_bar = float(outcomes.mean())
    unc = o_bar * (1.0 - o_bar)

    # Quantile-based bin edges (np.unique handles duplicate quantiles)
    bin_edges = np.unique(np.quantile(forecasts, np.linspace(0, 1, n_bins + 1)))
    bin_idx = np.digitize(forecasts, bin_edges[1:], right=False)
    n_actual_bins = len(bin_edges) - 1

    rel = res = wbv = wbc = 0.0

    for k in range(n_actual_bins):
        mask = bin_idx == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue

        f_k = forecasts[mask]
        o_k = outcomes[mask].astype(float)
        f_bar_k = float(f_k.mean())
        o_bar_k = float(o_k.mean())

        rel += n_k * (f_bar_k - o_bar_k) ** 2
        res += n_k * (o_bar_k - o_bar) ** 2

        if n_k > 1:
            # Population variance / covariance (ddof=0 — consistent with Brier identity)
            f_dev = f_k - f_bar_k
            o_dev = o_k - o_bar_k
            wbv += n_k * float(np.mean(f_dev**2))
            wbc += n_k * float(np.mean(f_dev * o_dev))

    rel /= n
    res /= n
    wbv /= n
    wbc /= n

    brier = rel - res + unc + wbv - 2.0 * wbc
    return brier, rel, res, wbv, wbc, unc


def brier_decompose(
    forecasts: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 20,
    bootstrap_n: int = 0,
    random_state: int = 42,
) -> BrierComponents:
    """
    5-component Brier decomposition per §8.1 MATH.md v2.1.

    Br = REL - RES + UNC + WBV - 2·WBC

    Args:
        forecasts:    predicted probabilities in [0, 1], shape (n,)
        outcomes:     binary outcomes {0, 1}, shape (n,)
        n_bins:       number of quantile bins (MATH.md specifies K=20)
        bootstrap_n:  bootstrap resamples for 95% CIs; 0 = skip
        random_state: RNG seed

    Returns:
        BrierComponents with all five decomposition terms.
    """
    forecasts = np.asarray(forecasts, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)

    if len(forecasts) != len(outcomes):
        raise ValueError(f"Length mismatch: forecasts={len(forecasts)}, outcomes={len(outcomes)}")
    if len(forecasts) < n_bins:
        raise ValueError(f"Need at least n_bins={n_bins} observations, got {len(forecasts)}")

    brier, rel, res, wbv, wbc, unc = _compute_components(forecasts, outcomes, n_bins)
    gres = res + 2.0 * wbc - wbv

    brier_ci = rel_ci = res_ci = None

    if bootstrap_n > 0:
        rng = np.random.default_rng(random_state)
        boot_brier = np.empty(bootstrap_n)
        boot_rel = np.empty(bootstrap_n)
        boot_res = np.empty(bootstrap_n)

        for i in range(bootstrap_n):
            idx = rng.integers(0, len(forecasts), size=len(forecasts))
            b, r, s, *_ = _compute_components(forecasts[idx], outcomes[idx], n_bins)
            boot_brier[i] = b
            boot_rel[i] = r
            boot_res[i] = s

        brier_ci = (
            float(np.percentile(boot_brier, 2.5)),
            float(np.percentile(boot_brier, 97.5)),
        )
        rel_ci = (
            float(np.percentile(boot_rel, 2.5)),
            float(np.percentile(boot_rel, 97.5)),
        )
        res_ci = (
            float(np.percentile(boot_res, 2.5)),
            float(np.percentile(boot_res, 97.5)),
        )

    return BrierComponents(
        brier=brier,
        reliability=rel,
        resolution=res,
        uncertainty=unc,
        wbv=wbv,
        wbc=wbc,
        gres=gres,
        n_bins=n_bins,
        n_obs=len(forecasts),
        brier_ci=brier_ci,
        reliability_ci=rel_ci,
        resolution_ci=res_ci,
    )


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------


class VennAbersCalibrator:
    """
    Venn-Abers predictors (Vovk-Petej 2014) — primary calibrator per §8.2.

    Valid coverage even without iid / exchangeability — essential for
    autocorrelated market tick data.

    Algorithm (inductive version):
      For each test score s*:
        1. Augment calibration set with (s*, 0), fit isotonic → f0(s*)
        2. Augment calibration set with (s*, 1), fit isotonic → f1(s*)
        3. Point estimate: p* = (f0 + f1) / 2
        4. Interval: [min(f0,f1), max(f0,f1)]
    """

    def __init__(self) -> None:
        self._cal_scores: np.ndarray | None = None
        self._cal_outcomes: np.ndarray | None = None

    def fit(self, scores: np.ndarray, outcomes: np.ndarray) -> VennAbersCalibrator:
        scores = np.asarray(scores, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        idx = np.argsort(scores)
        self._cal_scores = scores[idx]
        self._cal_outcomes = outcomes[idx]
        return self

    def _va_pair(self, s: float) -> tuple[float, float]:
        assert self._cal_scores is not None and self._cal_outcomes is not None
        pos = int(np.searchsorted(self._cal_scores, s))

        s_aug = np.insert(self._cal_scores, pos, s)

        o0 = np.insert(self._cal_outcomes, pos, 0.0)
        f0 = float(IsotonicRegression(out_of_bounds="clip").fit(s_aug, o0).predict([s])[0])

        o1 = np.insert(self._cal_outcomes, pos, 1.0)
        f1 = float(IsotonicRegression(out_of_bounds="clip").fit(s_aug, o1).predict([s])[0])

        return min(f0, f1), max(f0, f1)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self._cal_scores is None:
            raise RuntimeError("Call fit() first")
        return np.array([(lo + hi) / 2 for lo, hi in (self._va_pair(s) for s in scores)])

    def predict_interval(self, scores: np.ndarray) -> np.ndarray:
        """Returns (n, 2) array of [f0, f1] confidence intervals."""
        if self._cal_scores is None:
            raise RuntimeError("Call fit() first")
        return np.array([self._va_pair(s) for s in scores])


class IsotonicCalibrator:
    """Standard isotonic regression — baseline comparison for §8.2."""

    def __init__(self) -> None:
        self._ir = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(self, scores: np.ndarray, outcomes: np.ndarray) -> IsotonicCalibrator:
        self._ir.fit(np.asarray(scores, dtype=float), np.asarray(outcomes, dtype=float))
        self._fitted = True
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        return self._ir.predict(np.asarray(scores, dtype=float))


class BetaCalibrator:
    """
    Beta calibration (Kull et al. 2017) — fallback per §8.2 when n < 2000.

    p_cal = σ(a·log(s) + b·log(1-s) + c)

    Fitted by MLE. Valid for monotone, S-shaped and reverse-S-shaped distortions.
    """

    def __init__(self) -> None:
        self._params: np.ndarray | None = None

    def fit(self, scores: np.ndarray, outcomes: np.ndarray) -> BetaCalibrator:
        scores = np.clip(np.asarray(scores, dtype=float), 1e-7, 1 - 1e-7)
        outcomes = np.asarray(outcomes, dtype=float)

        log_s = np.log(scores)
        log_1ms = np.log(1.0 - scores)

        def neg_ll(params: np.ndarray) -> float:
            a, b, c = params
            p_cal = np.clip(expit(a * log_s + b * log_1ms + c), 1e-7, 1 - 1e-7)
            return -float(np.sum(outcomes * np.log(p_cal) + (1 - outcomes) * np.log(1 - p_cal)))

        result = minimize(neg_ll, x0=[1.0, 1.0, 0.0], method="L-BFGS-B")
        self._params = result.x
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self._params is None:
            raise RuntimeError("Call fit() first")
        scores = np.clip(np.asarray(scores, dtype=float), 1e-7, 1 - 1e-7)
        a, b, c = self._params
        return expit(a * np.log(scores) + b * np.log(1.0 - scores) + c)


# ---------------------------------------------------------------------------
# Out-of-time evaluation framework
# ---------------------------------------------------------------------------


def recalibrate_out_of_time(
    forecasts: pd.Series,
    outcomes: pd.Series,
    timestamps: pd.Series,
    method: str = "venn_abers",
    weeks_fit: int = 4,
    min_labelled: int = 2000,
) -> pd.DataFrame:
    """
    Rolling out-of-time recalibration per §8.2 MATH.md v2.1.

    Strategy: fit on weeks [w-weeks_fit, w-1], score on week w, roll forward.
    Falls back to beta calibration when n_train < min_labelled.

    Args:
        forecasts:     model predicted probabilities
        outcomes:      binary outcomes {0, 1}; NaN rows are scored but not used for fitting
        timestamps:    datetime index for week assignment
        method:        "venn_abers" | "isotonic" | "beta"
        weeks_fit:     rolling look-back window in weeks
        min_labelled:  threshold below which beta calibration is used instead of venn_abers

    Returns:
        DataFrame: timestamp, raw_forecast, calibrated, outcome, week, calibrator_used
    """
    df = (
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(timestamps.values),
                "forecast": forecasts.values.astype(float),
                "outcome": outcomes.values.astype(float),
            }
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    iso_week = df["timestamp"].dt.isocalendar()
    df["year_week"] = iso_week.year.astype(str) + "-" + iso_week.week.astype(str).str.zfill(2)

    all_weeks = np.sort(df["year_week"].unique())
    if len(all_weeks) <= weeks_fit:
        raise ValueError(
            f"Need more than weeks_fit={weeks_fit} distinct weeks of data, " f"got {len(all_weeks)}"
        )

    rows = []

    for i, test_week in enumerate(all_weeks[weeks_fit:], start=weeks_fit):
        train_weeks = all_weeks[max(0, i - weeks_fit) : i]
        train_mask = df["year_week"].isin(train_weeks) & df["outcome"].notna()
        test_mask = df["year_week"] == test_week

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        used_method = method
        if len(train_df) < min_labelled and method == "venn_abers":
            used_method = "beta"

        try:
            if used_method == "venn_abers":
                cal: VennAbersCalibrator | IsotonicCalibrator | BetaCalibrator = (
                    VennAbersCalibrator().fit(
                        train_df["forecast"].values,
                        train_df["outcome"].values,
                    )
                )
            elif used_method == "isotonic":
                cal = IsotonicCalibrator().fit(
                    train_df["forecast"].values, train_df["outcome"].values
                )
            else:
                cal = BetaCalibrator().fit(train_df["forecast"].values, train_df["outcome"].values)
        except Exception:
            cal = IsotonicCalibrator().fit(train_df["forecast"].values, train_df["outcome"].values)
            used_method = "isotonic_fallback"

        calibrated = cal.predict(test_df["forecast"].values)

        for row, cal_p in zip(test_df.itertuples(index=False), calibrated, strict=False):
            rows.append(
                {
                    "timestamp": row.timestamp,
                    "raw_forecast": row.forecast,
                    "calibrated": float(cal_p),
                    "outcome": row.outcome,
                    "week": test_week,
                    "calibrator_used": used_method,
                }
            )

    return pd.DataFrame(rows)
