"""
backtesting/metrics.py
──────────────────────
Cálculo de métricas de rendimiento y riesgo para el backtest.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def calculate_metrics(trace_df: pd.DataFrame, initial_cash: float) -> dict[str, Any]:
    """
    Calcula métricas clave a partir de la traza temporal del backtest.

    La traza debe contener las columnas:
      - timestamp (datetime)
      - cash (float)
      - position (float) (posición firmada de YES)
      - mid_price (float) (precio de referencia del mercado)
      - bid_p (float) (precio de compra cotizado, puede ser NaN)
      - ask_p (float) (precio de venta cotizado, puede ser NaN)
      - action (str, opcional) (ej. 'BUY', 'SELL', o vacía si es snapshot de estado)
      - order_status (str, opcional) (ej. 'FILLED')

    Métricas calculadas:
      - PnL Total (realizado + valorización de posición a mid-price)
      - Retorno Total (%)
      - Sharpe Ratio (anualizado sobre retornos diarios)
      - Max Drawdown (en valor absoluto de dólares y en porcentaje)
      - Ratio de Fills (porcentaje de órdenes completadas sobre colocadas)
      - Estadísticas de Inventario (min, max, promedio, desviación estándar)
      - Spread Promedio cotizado (en espacio precio)
    """
    if trace_df.empty:
        return {
            "total_pnl": 0.0,
            "total_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0,
            "trade_count": 0,
            "fill_rate": 0.0,
            "inv_min": 0.0,
            "inv_max": 0.0,
            "inv_mean": 0.0,
            "inv_std": 0.0,
            "avg_spread": 0.0,
        }

    # Copiar para evitar SideEffects
    df = trace_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Calcular Equity = Cash + Position * MidPrice
    df["equity"] = df["cash"] + df["position"] * df["mid_price"]

    # 1. PnL Total y Retorno
    final_equity = df["equity"].iloc[-1]
    total_pnl = final_equity - initial_cash
    total_return_pct = (total_pnl / initial_cash) * 100.0

    # 2. Sharpe Ratio Diario Anualizado
    # Resamplear a diario usando el último valor conocido de cada día (forward fill)
    df_daily = df.set_index("timestamp")["equity"].resample("D").last().ffill()
    daily_returns = df_daily.pct_change().dropna()

    # Si hay muy pocos datos o no hay variación, Sharpe es 0
    if len(daily_returns) > 1 and daily_returns.std() > 1e-8:
        # Sharpe = (mean / std) * sqrt(365) (los prediction markets operan 24/7/365)
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365.0)
    else:
        sharpe = 0.0

    # 3. Max Drawdown (Absolute USD & Percentage)
    # Roll peak
    df["peak"] = df["equity"].cummax()
    df["dd_usd"] = df["peak"] - df["equity"]
    df["dd_pct"] = (df["dd_usd"] / df["peak"]) * 100.0

    max_dd_usd = float(df["dd_usd"].max())
    max_dd_pct = float(df["dd_pct"].max())

    # 4. Trades & Fill Rate
    # Un trade es un fill confirmado
    trade_count = 0
    if "order_status" in df.columns:
        # Contar filas donde hubo un fill
        trade_count = int((df["order_status"] == "FILLED").sum())

    submitted_count = 0
    if "action" in df.columns:
        # Contar órdenes enviadas (acciones BUY o SELL)
        submitted_count = int(df["action"].isin(["BUY", "SELL"]).sum())

    fill_rate = (trade_count / submitted_count * 100.0) if submitted_count > 0 else 0.0

    # 5. Inventario
    inv_min = float(df["position"].min())
    inv_max = float(df["position"].max())
    inv_mean = float(df["position"].mean())
    inv_std = float(df["position"].std())

    # 6. Spreads
    avg_spread = 0.0
    if "bid_p" in df.columns and "ask_p" in df.columns:
        df["spread"] = df["ask_p"] - df["bid_p"]
        avg_spread = float(df["spread"].dropna().mean())

    return {
        "total_pnl": round(total_pnl, 4),
        "total_return_pct": round(total_return_pct, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_usd": round(max_dd_usd, 4),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "trade_count": trade_count,
        "fill_rate": round(fill_rate, 4),
        "inv_min": inv_min,
        "inv_max": inv_max,
        "inv_mean": round(inv_mean, 4),
        "inv_std": round(inv_std, 4),
        "avg_spread": round(avg_spread, 4),
    }
