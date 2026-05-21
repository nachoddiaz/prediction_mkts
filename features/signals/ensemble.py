"""
features/signals/ensemble.py
─────────────────────────────
SignalEnsemble — Cholesky-orthogonalized multi-signal combination.

§8.3 MATH.md v2.1:
  μ̂_t = w_1·OBI + w_2·News + w_3·OnChain

  OBI ⊥ News | Y is empirically violated (defect J): OBI is mechanically
  caused by the same news driving NewsSignal on the same horizon.
  We Cholesky-factor the empirical signal covariance and update only on
  the residuals, so each weight w_i can be interpreted causally.

Bayesian update in log-odds (§8.3):
  X^posterior = X^prior + Σ_k ln Λ_k(s_k)
  Valid only after orthogonalisation.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

SIGNAL_NAMES = ("obi", "news", "onchain")


class SignalEnsemble:
    """
    Combines OBI, News, and OnChain signals with Cholesky orthogonalization.

    Two-phase usage:
      Phase A (calibration): fit_orthogonalization(signal_matrix) on historical data.
      Phase B (runtime):     compute_mu_hat(obi, news, onchain) returns μ̂_t.

    When no orthogonalization has been fitted (e.g., news/onchain are stub zeros),
    the ensemble falls back to the raw weighted sum — equivalent to OBI proxy.

    Usage:
        ensemble = SignalEnsemble(weights={"obi": 0.8, "news": 0.15, "onchain": 0.05})
        ensemble.fit_orthogonalization(historical_signal_matrix)  # (n, 3)
        mu = ensemble.compute_mu_hat(obi=0.3, news=0.1, onchain=0.0)
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        w = weights or {}
        self._weights: dict[str, float] = {
            "obi": float(w.get("obi", 1.0)),
            "news": float(w.get("news", 0.0)),
            "onchain": float(w.get("onchain", 0.0)),
        }
        self._chol_inv: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def fit_orthogonalization(self, signal_matrix: np.ndarray) -> SignalEnsemble:
        """
        Fit Cholesky decomposition of empirical signal covariance.

        Args:
            signal_matrix: (n_obs, 3) array — columns are [OBI, News, OnChain].
                           Rows with all-zero news/onchain columns are fine; the
                           decomposition will reflect their zero variance and
                           the L^{-1} will leave them unchanged.

        Returns:
            self (for chaining)
        """
        if signal_matrix.ndim != 2 or signal_matrix.shape[1] != len(SIGNAL_NAMES):
            raise ValueError(
                f"signal_matrix must have shape (n_obs, {len(SIGNAL_NAMES)}), "
                f"got {signal_matrix.shape}"
            )

        cov = np.cov(signal_matrix.T)
        # Small regularization prevents singular matrices when a signal is constant (e.g., stub)
        cov = cov + 1e-8 * np.eye(cov.shape[0])

        try:
            L = np.linalg.cholesky(cov)
            self._chol_inv = np.linalg.inv(L)
            log.debug("SignalEnsemble: Cholesky orthogonalization fitted")
        except np.linalg.LinAlgError:
            self._chol_inv = None
            log.warning(
                "SignalEnsemble: Cholesky decomposition failed — "
                "using raw weighted sum (no orthogonalization)"
            )

        return self

    def update_weights(self, weights: dict[str, float]) -> None:
        """Update signal weights post-calibration (e.g., from CJCalibrator)."""
        for key in SIGNAL_NAMES:
            if key in weights:
                self._weights[key] = float(weights[key])
        log.debug("SignalEnsemble weights updated: %s", self._weights)

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def compute_mu_hat(
        self,
        obi: float,
        news: float = 0.0,
        onchain: float = 0.0,
    ) -> float:
        """
        Compute μ̂_t from raw signals with optional Cholesky orthogonalization.

        When news and onchain are 0.0 (stubs) and no orthogonalization is fitted,
        this reduces to w_obi · OBI — identical to the current proxy in store.py.

        Args:
            obi:     Order Book Imbalance ∈ [-1, 1]
            news:    normalized news sentiment ∈ [-1, 1]  (0.0 = stub)
            onchain: normalized on-chain signal ∈ [-1, 1] (0.0 = stub)

        Returns:
            μ̂_t as a float
        """
        signals = np.array([obi, news, onchain], dtype=float)

        if self._chol_inv is not None:
            signals = self._chol_inv @ signals

        w = np.array([self._weights["obi"], self._weights["news"], self._weights["onchain"]])
        return float(np.dot(w, signals))

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_cj_result(cls, cj_result: object) -> SignalEnsemble:
        """Build ensemble from a CJResult (strategies/market_making/params.py)."""
        return cls(
            weights={
                "obi": float(getattr(cj_result, "w_obi", 1.0)),
                "news": float(getattr(cj_result, "w_news", 0.0)),
                "onchain": float(getattr(cj_result, "w_onchain", 0.0)),
            }
        )

    @classmethod
    def obi_only(cls) -> SignalEnsemble:
        """OBI-only ensemble — matches the current proxy in store.py."""
        return cls(weights={"obi": 1.0, "news": 0.0, "onchain": 0.0})
