# Here we are going to calibrate κ, A, φ, η, ρ from DuckDB
# First we are  goint ti calibrate GLFT and then the Cartea-Jaimungal

"""
strategies/market_making/params.py
────────────────────────────────────
Calibración de parámetros para los modelos GLFT y Cartea-Jaimungal.

Dos calibradores independientes:

  GLFTCalibrator  → estima κ y A por MLE sobre fill events proxy
  CJCalibrator    → estima φ, η, ρ y w_i por Ridge + AR(1)

Ambos reciben DataFrames — compatibles con notebooks y scripts.

Flujo típico desde notebook:
    reader = MarketDataReader()
    ticks_df    = reader.ticks("kalshi:KXBTC-TEST", start, end)
    features_df = reader.features("kalshi:KXBTC-TEST", start, end)

    glft = GLFTCalibrator().fit(ticks_df)
    cj   = CJCalibrator().fit(ticks_df, features_df)

    update_model_params("kalshi", {**glft, **cj})

Relación con el MATH.md:
  §3.2 → MLE para κ, A
  §4.6 → Ridge para w_i, AR(1) para φ, η, ρ
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

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Mínimo de observaciones para calibrar — con menos no hay convergencia
MIN_TICKS_GLFT = 100
MIN_TICKS_CJ = 200

# Límites de los parámetros para la optimización MLE
# Evitan que el optimizador explore regiones sin sentido económico
KAPPA_BOUNDS = (0.01, 20.0)
A_BOUNDS = (1e-6, 10.0)

# Lambda de Ridge — penalización para evitar overfitting en w_i
# Ajustar en el notebook si los pesos son inestables
RIDGE_ALPHA = 1.0


# ---------------------------------------------------------------------------
# Dataclasses de resultados
# ---------------------------------------------------------------------------


@dataclass
class GLFTResult:
    """
    Resultado de la calibración GLFT.

    kappa_p: decay del arrival rate en espacio de precio (1/$)
             Calibrado directamente desde spread observado.
             Valor alto → el libro es profundo, el MM puede cotizar spreads anchos
             Valor bajo → el libro es fino, necesita spreads estrechos para tener fills

    kappa_x: decay en espacio logit (adimensional) — §3 MATH.md v2.1
             κ_x = κ_p · p̄(1-p̄) donde p̄ es el precio promedio
             Este es el valor que usa glft.py en espacio X = logit(p)

    A:       baseline arrival rate en órdenes/segundo con spread=0
             Proporcional al volumen del mercado

    log_likelihood: valor del log-likelihood en el óptimo
                    Útil para comparar calibraciones entre períodos

    n_observations: número de observaciones usadas
    """

    kappa_p: float
    kappa_x: float
    A: float
    log_likelihood: float
    n_observations: int

    def to_dict(self) -> dict[str, float]:
        """Para pasar directamente a update_model_params()."""
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
    Resultado de la calibración Cartea-Jaimungal.

    phi: mean-reversion speed de μ_t
         Mayor φ → el drift revierte más rápido → señal menos persistente

    eta: volatilidad del drift latente
         Mayor η → el drift fluctúa más → señal más ruidosa

    rho: correlación entre innovaciones de precio y señal
         ρ > 0 → señal positiva precede subidas de precio
         ρ < 0 → señal positiva precede bajadas

    w_obi, w_news, w_onchain: pesos de la señal compuesta
         Calibrados por Ridge regression sobre retornos siguientes

    ar1_alpha:    coeficiente AR(1) de la serie μ̂_t
    signal_r2:    R² de la Ridge regression
    n_obs_signal: observaciones usadas en Ridge
    n_obs_ar1:    observaciones usadas en AR(1)
    """

    phi: float
    eta: float
    rho: float
    w_obi: float
    w_news: float
    w_onchain: float

    # Métricas de calidad
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
    Calibra κ y A del modelo GLFT por MLE sobre fill events proxy.

    Por qué proxy y no fills reales:
      No tenemos execution layer todavía — no hay fills observados.
      El proxy usa el spread mid-to-mid entre ticks consecutivos:
      si el spread se comprimió significativamente respecto al spread
      que habríamos cotizado, asumimos que hubo un fill.

      Cuando implementemos el execution layer, sustituiremos el proxy
      por fills reales — el calibrador no cambia, solo los datos de entrada.

    Uso:
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
        Calibra κ y A desde una serie de ticks.

        Args:
            ticks_df:          DataFrame con columnas 'spread' y 'timestamp',
                               ordenado por timestamp ASC.
                               Output directo de reader.ticks().
            quoted_spread_col: columna con el spread observado.

        Returns:
            GLFTResult con κ, A y métricas de calidad.

        Raises:
            ValueError: si hay menos de MIN_TICKS_GLFT observaciones.
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
        Construye pares (spread_cotizado, fill_proxy) desde la serie de ticks.

        Lógica del proxy:
          Para cada tick i, el spread cotizado es spread[i].
          Si spread[i+1] < spread[i] * threshold, asumimos fill en i.
          El threshold es 0.5 — si el spread se comprimió a la mitad,
          es probable que hubiera una ejecución.

        Por qué este proxy:
          En un mercado con MM, el spread se comprime cuando alguien cruza
          el libro. La compresión del spread siguiente es la señal más
          fácilmente observable de que hubo un fill en el tick anterior.

        Returns:
            (spreads, fills): arrays numpy de igual longitud.
        """
        if spread_col not in df.columns:
            raise ValueError(f"Column '{spread_col}' not found. " f"Available: {list(df.columns)}")

        spread_series = df[spread_col].dropna().values.astype(float)

        if len(spread_series) < 2:
            return np.array([]), np.array([])

        # Solo usar spreads positivos (libro válido)
        valid_mask = spread_series > 1e-6
        spreads_raw = spread_series[valid_mask]

        if len(spreads_raw) < 2:
            return np.array([]), np.array([])

        # Spread siguiente (fill proxy)
        spreads_current = spreads_raw[:-1]
        spreads_next = spreads_raw[1:]

        # Fill proxy: el spread se comprimió más del 50%
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
        Estimación por máxima verosimilitud de κ y A.

        Log-likelihood (Poisson process):
          ℓ(κ, A) = Σ_i [ f_i·ln(A·e^{-κδ_i}) - A·e^{-κδ_i} ]
                  = Σ_i [ f_i·(ln A - κδ_i) - A·e^{-κδ_i} ]

        Por qué minimizar -ℓ con scipy:
          scipy.optimize.minimize trabaja con minimización.
          Negamos el log-likelihood para convertirlo en minimización.

        Returns:
            (kappa, A, log_likelihood)
        """

        def neg_log_likelihood(params: np.ndarray) -> float:
            kappa, log_A = params
            A = np.exp(log_A)  # A siempre positivo

            # Intensidad de llegada por observación
            lam = A * np.exp(-kappa * spreads)

            # Log-likelihood de proceso de Poisson
            # f_i=1: hubo llegada, contribuye ln(λ_i) - λ_i·Δt
            # f_i=0: no hubo llegada, contribuye -λ_i·Δt
            # Simplificado (Δt=1 por normalización):
            ll = np.sum(fills * np.log(lam + 1e-12) - lam)

            return -ll

        # Punto inicial — buscar en varias combinaciones para evitar mínimos locales
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
    Calibra φ, η, ρ y w_i del modelo Cartea-Jaimungal.

    Tres pasos secuenciales (§4.6 del MATH.md):
      1. Ridge regression → pesos w_i → serie μ̂_t
      2. AR(1) sobre μ̂_t → φ, η
      3. Correlación empírica → ρ

    Uso:
        calibrator = CJCalibrator()
        result     = calibrator.fit(ticks_df, features_df)
        print(result.summary())
    """

    def __init__(self, ridge_alpha: float = RIDGE_ALPHA) -> None:
        """
        Args:
            ridge_alpha: penalización de Ridge.
                         Mayor α → pesos más cercanos a 0 → menos overfitting
                         Ajustar si los pesos son inestables entre períodos.
        """
        self.ridge_alpha = ridge_alpha

    def fit(
        self,
        ticks_df: pd.DataFrame,
        features_df: pd.DataFrame,
    ) -> CJResult:
        """
        Calibra los parámetros CJ desde ticks y features históricos.

        Args:
            ticks_df:    DataFrame con columnas 'timestamp', 'mid'.
                         Output directo de reader.ticks().

            features_df: DataFrame con columnas 'timestamp', 'obi',
                         y opcionalmente 'ewma_vol'.
                         Output directo de reader.features().

        Returns:
            CJResult con φ, η, ρ, w_i y métricas de calidad.

        Raises:
            ValueError: si hay menos de MIN_TICKS_CJ observaciones.
        """
        ticks = ticks_df.copy().sort_values("timestamp").reset_index(drop=True)
        features = features_df.copy().sort_values("timestamp").reset_index(drop=True)

        if len(ticks) < MIN_TICKS_CJ:
            raise ValueError(
                f"CJCalibrator needs at least {MIN_TICKS_CJ} ticks, " f"got {len(ticks)}."
            )

        # Paso 1 — Ridge regression → pesos w_i → serie μ̂_t
        weights, mu_hat_series, r2, n_signal = self._calibrate_signal(ticks, features)

        # Paso 2 — AR(1) sobre μ̂_t → φ, η
        phi, eta, ar1_alpha, n_ar1 = self._calibrate_ar1(mu_hat_series, ticks)

        # Paso 3 — correlación → ρ
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
            "CJ calibration: phi=%.4f eta=%.6f rho=%.4f " "w_obi=%.4f r2=%.4f n=%d",
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
        Paso 1: Ridge regression para estimar los pesos w_i.

        Target: retorno del siguiente tick Δp_{t+1} = p_{t+1} - p_t
        Features: OBI_t (y en el futuro: News_t, OnChain_t)

        Por qué Ridge y no OLS:
          Los features pueden estar correlacionados (multicolinealidad).
          Ridge penaliza los coeficientes grandes, produciendo pesos
          más estables entre períodos de calibración distintos.

        Returns:
            (weights_dict, mu_hat_series, r2, n_obs)
        """
        # Construir incrementos de log-odds ΔX_{t+1} = logit(p_{t+1}) - logit(p_t)
        # §5 MATH.md v2.1: el proceso OU de μ_t opera sobre X = logit(p) bajo ℙ,
        # no sobre p. Cerca de las fronteras dX/dp = 1/(p(1-p)) → ∞, por lo que
        # usar Δp en lugar de ΔX sobreestima el edge en mercados atípicos.
        mids = ticks["mid"].values.astype(float)
        mids_clipped = np.clip(mids, 1e-6, 1 - 1e-6)
        X_logit = np.log(mids_clipped / (1.0 - mids_clipped))
        returns = np.diff(X_logit)  # ΔX_{t+1} para t=0..N-2

        # Features disponibles — por ahora solo OBI
        # En el futuro: añadir columnas de news y onchain
        available_features: dict[str, np.ndarray] = {}

        if "obi" in features.columns:
            # Alinear features con retornos por timestamp
            # Los retornos son del tick t al t+1, así que usamos features en t
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

        # Estandarizar features para que la penalización Ridge sea justa
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        # Ridge regression
        ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        ridge.fit(X_std, y)

        r2 = float(ridge.score(X_std, y))

        # Desescalar coeficientes — queremos pesos en escala original
        # w_raw = w_std / std(feature)
        weights_raw = ridge.coef_ / scaler.scale_

        weights_dict = {name: float(w) for name, w in zip(feature_names, weights_raw, strict=False)}
        # Features no disponibles tienen peso 0
        for name in ("news", "onchain"):
            weights_dict.setdefault(name, 0.0)

        # Construir serie μ̂_t = Σ w_i · f_i(t)
        # Aplicar a todos los features disponibles (no solo los usados en Ridge)
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
        Paso 2: AR(1) sobre la serie μ̂_t para estimar φ y η.

        Modelo: μ̂_{t+1} = α·μ̂_t + ε_t
        donde α = e^{-φ·Δt}

        De α estimamos:
          φ = -ln(α) / Δt
          η = ν·√(2φ / (1 - α²))
        donde ν² = Var(ε_t)

        Por qué OLS y no MLE para AR(1):
          Para un AR(1) gaussiano el OLS es equivalente al MLE.
          Es más simple y más rápido numéricamente.

        Returns:
            (phi, eta, alpha, n_obs)
        """
        values = mu_hat.dropna().values.astype(float)

        if len(values) < 10:
            log.warning("AR(1): too few observations — using defaults")
            return 1.0, 0.05, 0.99, 0

        # OLS: regresión de μ̂_{t+1} sobre μ̂_t
        y_lag = values[:-1]
        y_next = values[1:]

        # Evitar división por cero si la serie es constante
        var_lag = np.var(y_lag)
        if var_lag < 1e-12:
            log.warning("AR(1): signal is constant — phi and eta undetermined")
            return 1.0, 0.05, 0.99, 0

        # Coeficiente AR(1)
        alpha = float(np.sum(y_lag * y_next) / np.sum(y_lag**2))
        # Acotar en (0, 1) — el drift debe ser estacionario
        alpha = np.clip(alpha, 0.01, 0.9999)

        # Residuos y su varianza
        residuals = y_next - alpha * y_lag
        nu2 = float(np.var(residuals))

        # Estimar Δt en años — desde timestamps del DataFrame de ticks
        if "timestamp" in ticks.columns and len(ticks) > 1:
            ts = pd.to_datetime(ticks["timestamp"]).values
            dt_ns = np.diff(ts).astype("float64")
            dt_years = float(np.median(dt_ns[dt_ns > 0])) / 1e9 / (365.25 * 24 * 3600)
        else:
            dt_years = 1.0 / (365.25 * 24 * 60)  # default: 1 minuto

        # φ desde α: α = e^{-φ·Δt} → φ = -ln(α)/Δt
        phi = float(-np.log(alpha) / max(dt_years, 1e-10))
        phi = np.clip(phi, 0.01, 100.0)

        # η desde la varianza de los residuos
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
        Paso 3: correlación empírica entre Δp_t y Δμ̂_t.

        ρ = Cov(Δp_t, Δμ̂_t) / (σ̂·η·Δt)

        donde σ̂ es la desviación estándar empírica de Δp_t.

        Por qué este estimador:
          ρ es la correlación entre el Browniano del precio y el Browniano
          de la señal en el modelo continuo. En datos discretos, la
          correlación entre los incrementos Δp y Δμ̂ es el estimador
          más directo.

        Returns:
            rho ∈ [-0.99, 0.99]
        """
        if "mid" not in ticks.columns:
            return 0.0

        # ρ es la correlación entre el Browniano de X_t y el Browniano de μ_t,
        # ambos en espacio logit (§5 MATH.md v2.1). Usar Δp introduce un factor
        # p(1-p) que sesga ρ, especialmente lejos del 50%.
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

        # Acotar — ρ muy cercano a ±1 causa inestabilidad numérica
        rho = float(np.clip(rho, -0.99, 0.99))

        log.debug("rho calibration: rho=%.4f", rho)

        return rho
