"""
features/microstructure.py
───────────────────────────
Cálculo de features de microestructura del orderbook y series de ticks
Sólo se calculan señales observables en mercado

Tres secciones:

  1. SNAPSHOT  — operan sobre OrderBook o Tick individual.
                 O(1), usadas en el hot path del feature store.

  2. SERIES    — operan sobre DataFrames de ticks históricos.
                 Usadas para calibración y análisis en notebooks.

  3. PIPELINE  — función de alto nivel que lee de DuckDB,
                 calcula todo y devuelve un dict listo para
                 persistir en la tabla features.

Relación con el MATH.md:
  OBI            → w_1 en μ̂_t = w_1·OBI + w_2·News + w_3·OnChain (§4.6)
  quoted_spread  → comparación con δ* óptimo de GLFT (§3)
  bernoulli_vol  → σ_B(p,τ) = √(p(1-p)/τ) — entra en δ*/2 y p̃ (§5.3)
  ewma_vol       → validación empírica de σ_B; alarma de régimen jump (§6.1)
  mu_hat         → proxy de μ̂_t hasta calibrar señal completa (§4.6)
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from normalizer.schema import OrderBook, Tick
from storage.reader import MarketDataReader

# ---------------------------------------------------------------------------
# Constante EWMA
#
# λ = 0.94 es el estándar de RiskMetrics para datos diarios.
# Para tick data intraday valores más altos (0.97-0.99) reaccionan
# más rápido a cambios de régimen — ajustar por mercado en calibración.
# ---------------------------------------------------------------------------
EWMA_LAMBDA: float = 0.94


# ══════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — SNAPSHOT
# Operan sobre un objeto del dominio individual.
# O(1) — usadas en el hot path del feature store.
# ══════════════════════════════════════════════════════════════════════


def order_book_imbalance(ob: OrderBook, levels: int = 5) -> float:
    """
    Order Book Imbalance (OBI) ∈ [-1, 1].

    OBI = (V_bid - V_ask) / (V_bid + V_ask)

    Interpretación:
      +1 → toda la liquidez en el bid (presión compradora extrema)
      -1 → toda la liquidez en el ask (presión vendedora extrema)
       0 → libro equilibrado

    Relación con el MATH.md:
      Es el componente w_1·OBI_t de la señal μ̂_t en §4.6.
      OBI positivo → drift alcista esperado → reservation price sube
      via el signal skew φ_1(t)·μ̂_t.

    Por qué top N niveles y no el libro completo:
      Los niveles lejanos tienen poco impacto en el precio inmediato.
      Top 5 es el estándar en la literatura de microestructura.
      Además, en mercados ilíquidos el libro puede tener pocos niveles.

    Args:
        ob:     snapshot del orderbook (dominio canónico)
        levels: número de niveles a considerar

    Returns:
        float en [-1, 1], 0.0 si el libro está vacío
    """
    bid_vol = ob.bid_depth(levels)
    ask_vol = ob.ask_depth(levels)
    denom = bid_vol + ask_vol

    if denom < 1e-9:
        return 0.0

    return float((bid_vol - ask_vol) / denom)


def quoted_spread(ob: OrderBook) -> float | None:
    """
    Spread absoluto entre mejor bid y mejor ask.

    δ = r^a - r^b ∈ [0, 1]

    Es el coste inmediato de cruzar el libro para el taker.
    En el modelo GLFT δ* es el spread óptimo que el MM debería
    cotizar — el quoted_spread real nos dice si el MM actual
    es más estrecho o más ancho que el óptimo teórico.

    Returns:
        float en [0, 1], None si el libro está incompleto
    """
    return ob.spread


def relative_spread(ob: OrderBook) -> float | None:
    """
    Spread relativo al mid-price.

    δ_rel = δ / mid

    Normaliza el spread por el nivel de precio, permitiendo
    comparar liquidez entre mercados con distintas probabilidades.

    Ejemplo:
      Mercado al 50%, spread 0.02 → δ_rel = 4%
      Mercado al 5%,  spread 0.02 → δ_rel = 40% (mucho menos líquido)

    Returns:
        float ≥ 0, None si el libro está incompleto
    """
    mid = ob.mid
    spr = ob.spread

    if mid is None or spr is None or mid < 1e-9:
        return None

    return float(spr / mid)


def bernoulli_vol(p: float, tau_years: float) -> float:
    """
    Volatilidad endógena de un contrato binario — superficie de Bernoulli.

    σ_B(p, τ) = √(p(1-p) / τ)

    Esta es la volatilidad teórica que entra en GLFT (§5.3):
      - Reservation price: p̃ = p - q·γ·p(1-p)  [τ se cancela]
      - Optimal half-spread: δ*/2 = γ·p(1-p)/2 + (1/γ)·ln(1 + γ/κ)

    Por qué es endógena:
      No necesita calibración. p está en el precio del mercado
      y τ está en la fecha de resolución. Ambos son observables.

    A diferencia de Black-Scholes donde σ es exógeno y hay que
    calibrarlo desde opciones, aquí σ_B se deriva directamente
    de la estructura del contrato binario.

    Args:
        p:         probabilidad implícita ∈ (0, 1)
        tau_years: tiempo hasta resolución en años > 0

    Returns:
        float ≥ 0. inf si τ ≤ 0 (near-resolution).
        Usar bernoulli_vol_safe() de resolution.py para evitar inf.
    """
    if tau_years <= 1e-9:
        return math.inf

    if not 0.0 < p < 1.0:
        return 0.0

    return math.sqrt(p * (1.0 - p) / tau_years)


def bernoulli_vol_from_tick(tick: Tick, tau_years: float) -> float:
    """
    σ_B calculada desde el mid-price de un Tick.
    Convenience wrapper para uso en el feature store.
    """
    return bernoulli_vol(tick.mid, tau_years)


# ══════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — SERIES
# Operan sobre DataFrames de ticks históricos.
# Usadas en calibración y análisis en notebooks.
# ══════════════════════════════════════════════════════════════════════


def ewma_vol(
    ticks_df: pd.DataFrame,
    lam: float = EWMA_LAMBDA,
) -> float:
    """
    Volatilidad EWMA sobre una serie de ticks históricos.

    σ²_t = (1-λ)·σ²_{t-1} + λ·(Δp_t)²/Δt

    Por qué dividir por Δt:
      Normaliza la varianza a unidades anualizadas independientemente
      de la frecuencia de muestreo. Un tick cada segundo y un tick
      cada minuto producen estimaciones comparables.

    Complementa σ_B con información empírica:
      - Si ewma_vol >> bernoulli_vol → régimen de jump (§6.1)
        activar near-resolution rules aunque τ sea grande
      - Si ewma_vol ≈ bernoulli_vol → mercado se comporta como predice
        el modelo, σ_B es una buena estimación

    Args:
        ticks_df: DataFrame con columnas 'mid' y 'timestamp',
                  ordenado por timestamp ASC
        lam:      parámetro de decaimiento ∈ (0, 1)

    Returns:
        Volatilidad EWMA anualizada como float.
        0.0 si hay menos de 2 ticks.
    """
    if len(ticks_df) < 2:
        return 0.0

    df = ticks_df.sort_values("timestamp").reset_index(drop=True)
    mids = df["mid"].values.astype(float)
    ts = pd.to_datetime(df["timestamp"]).values

    dp = np.diff(mids)
    dt = np.diff(ts).astype("float64") / 1e9  # nanoseconds → seconds

    mask = dt > 0
    if not mask.any():
        return 0.0

    dp = dp[mask]
    dt = dt[mask]

    # Convertir dt a años para anualizar correctamente
    dt_years = dt / (365.25 * 24 * 3600)

    var = float(dp[0] ** 2 / dt_years[0]) if dt_years[0] > 0 else 0.0
    for r, t in zip(dp[1:], dt_years[1:], strict=False):
        if t > 0:
            var = (1 - lam) * var + lam * (r**2 / t)

    return float(math.sqrt(max(var, 0.0)))


def ewma_vol_series(
    ticks_df: pd.DataFrame,
    lam: float = EWMA_LAMBDA,
) -> pd.Series:
    """
    Serie temporal de volatilidad EWMA — una estimación por tick.

    Útil para visualizar la evolución de la volatilidad en notebooks
    y para detectar visualmente regímenes de jump.

    Returns:
        pd.Series con el mismo índice que ticks_df.
        NaN en el primer elemento (sin estimación previa).
    """
    if len(ticks_df) < 2:
        return pd.Series([float("nan")] * len(ticks_df), index=ticks_df.index)

    df = ticks_df.sort_values("timestamp").reset_index(drop=True)
    mids = df["mid"].values.astype(float)
    ts = pd.to_datetime(df["timestamp"]).values

    dp = np.diff(mids)
    dt_ns = np.diff(ts).astype("float64")
    dt_years = dt_ns / 1e9 / (365.25 * 24 * 3600)

    vols = [float("nan")]
    var = float("nan")

    for r, t in zip(dp, dt_years, strict=False):
        if t <= 0 or math.isnan(var):
            var = r**2 / t if t > 0 else float("nan")
        else:
            var = (1 - lam) * var + lam * (r**2 / t)
        vols.append(math.sqrt(max(var, 0.0)) if not math.isnan(var) else float("nan"))

    return pd.Series(vols, index=df.index).reindex(ticks_df.index)


def obi_series(
    ticks_df: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    OBI aproximado sobre una serie de ticks TRADE.

    Como no tenemos el orderbook completo en cada tick, aproximamos
    el OBI usando el flujo de trades:
      YES trade (compra agresiva) → +1 (presión compradora)
      NO  trade (venta agresiva)  → -1 (presión vendedora)
      QUOTE tick                  →  0 (sin información de dirección)

    Rolling sum normalizado sobre una ventana de N ticks.

    Args:
        ticks_df: DataFrame con columna 'side' ('yes'|'no'|None)
        window:   ventana rolling en número de ticks

    Returns:
        pd.Series con OBI aproximado ∈ [-1, 1]
    """
    if "side" not in ticks_df.columns:
        return pd.Series([0.0] * len(ticks_df), index=ticks_df.index)

    signal = ticks_df["side"].map({"yes": 1.0, "no": -1.0}).fillna(0.0)
    rolling_sum = signal.rolling(window, min_periods=1).sum()
    rolling_count = signal.abs().rolling(window, min_periods=1).sum()

    obi = rolling_sum / rolling_count.replace(0, float("nan"))
    return obi.fillna(0.0)


def spread_timeseries_from_df(ticks_df: pd.DataFrame) -> pd.Series:
    """
    Serie temporal de spreads desde un DataFrame de ticks.
    Requiere columnas 'yes_bid' y 'yes_ask'.
    """
    if "yes_bid" not in ticks_df.columns or "yes_ask" not in ticks_df.columns:
        raise ValueError("ticks_df must have columns yes_bid and yes_ask")
    return (ticks_df["yes_ask"] - ticks_df["yes_bid"]).rename("spread")


# ══════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — PIPELINE
# Lee de DuckDB, calcula todas las features, devuelve dict listo
# para persistir en la tabla features.
# ══════════════════════════════════════════════════════════════════════


def compute_features_from_db(
    market_id: str,
    reader: MarketDataReader,
    tau_years: float,
    ewma_window: int = 50,
    obi_levels: int = 5,
) -> dict[str, Any] | None:
    """
    Pipeline principal — lee de DuckDB y calcula todas las features.

    Llamado por features/store.py cada vez que llega un nuevo tick
    o snapshot. Devuelve un dict listo para writer.write_features_sync().

    Flujo:
      1. Lee último orderbook  → OBI, quoted_spread, relative_spread
      2. Lee últimos N ticks   → EWMA vol, mid actual
      3. Calcula σ_B           → con mid y tau
      4. μ̂ aproximado          → OBI como proxy hasta calibrar señal completa

    Por qué OBI como proxy de μ̂:
      Hasta calibrar Ridge regression + AR(1) (§4.6 del MATH.md),
      el OBI es la mejor señal disponible en tiempo real.
      OBI > 0 implica presión compradora → drift alcista → μ̂ > 0.
      Los pesos w_i se calibrarán en models/signals/ensemble.py.

    Args:
        market_id:   string canónico "venue:raw_id"
        reader:      instancia de MarketDataReader (conexión abierta)
        tau_years:   tiempo hasta resolución en años (de Resolution.tau)
        ewma_window: número de ticks históricos para EWMA
        obi_levels:  niveles del libro para OBI

    Returns:
        Dict con todas las features, None si no hay datos suficientes.
    """
    # --- 1. Último orderbook para OBI y spreads ---
    ob_df = reader.latest_orderbook(market_id)

    obi_val = 0.0
    q_spread = None
    r_spread = None
    mid_from_ob = None

    if not ob_df.empty:
        best_bid = ob_df["best_bid"].iloc[0]
        best_ask = ob_df["best_ask"].iloc[0]
        bid_depth = ob_df["bid_depth_5"].iloc[0]
        ask_depth = ob_df["ask_depth_5"].iloc[0]

        # OBI desde profundidades pre-computadas del libro
        denom = float(bid_depth + ask_depth)
        if denom > 1e-9:
            obi_val = float((bid_depth - ask_depth) / denom)

        # Spread desde best bid/ask
        if best_bid is not None and best_ask is not None:
            best_bid = float(best_bid)
            best_ask = float(best_ask)
            q_spread = best_ask - best_bid
            mid_from_ob = (best_bid + best_ask) / 2.0
            if mid_from_ob > 1e-9:
                r_spread = q_spread / mid_from_ob

    # --- 2. Últimos N ticks para EWMA y mid actual ---
    ticks_df = reader.latest_ticks(market_id, n=ewma_window)

    if ticks_df.empty:
        return None

    mid_current = float(ticks_df["mid"].iloc[0])
    timestamp = ticks_df["timestamp"].iloc[0]

    # Preferir mid del orderbook (más preciso que el tick)
    p = mid_from_ob if mid_from_ob is not None else mid_current

    # EWMA en orden ASC (latest_ticks devuelve DESC)
    ticks_asc = ticks_df.sort_values("timestamp").reset_index(drop=True)
    ewma = ewma_vol(ticks_asc)

    # --- 3. Bernoulli vol ---
    bvol = bernoulli_vol(p, tau_years)
    bvol_stored = float(bvol) if not math.isinf(bvol) else None

    # --- 4. μ̂ aproximado como OBI ---
    mu_hat = obi_val

    # --- Venue desde market_id ---
    venue = market_id.split(":")[0] if ":" in market_id else "unknown"

    # Convertir timestamp a datetime aware
    if hasattr(timestamp, "to_pydatetime"):
        ts = timestamp.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
    else:
        ts = datetime.now(tz=UTC)

    return {
        "market_id": market_id,
        "venue": venue,
        "timestamp": ts,
        "obi": round(obi_val, 6),
        "quoted_spread": round(float(q_spread), 6) if q_spread is not None else None,
        "relative_spread": round(float(r_spread), 6) if r_spread is not None else None,
        "bernoulli_vol": round(bvol_stored, 6) if bvol_stored is not None else None,
        "ewma_vol": round(ewma, 6),
        "tau_years": round(tau_years, 8),
        "mu_hat": round(mu_hat, 6),
    }


def compute_features_batch(
    market_ids: list[str],
    reader: MarketDataReader,
    tau_map: dict[str, float],
) -> list[dict[str, Any]]:
    """
    Calcula features para múltiples mercados de una vez.
    Excluye los None (mercados sin datos suficientes).

    Args:
        market_ids: lista de market_id strings
        reader:     instancia de MarketDataReader
        tau_map:    dict {market_id: tau_years}
    """
    results = []
    for mid in market_ids:
        tau = tau_map.get(mid, 0.0)
        row = compute_features_from_db(mid, reader, tau)
        if row is not None:
            results.append(row)
    return results
