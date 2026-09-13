"""
backtesting/metrics.py
──────────────────────
Performance and risk metrics for the backtester.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def calculate_metrics(trace_df: pd.DataFrame, initial_cash: float) -> dict[str, Any]:
    """
    Compute the key metrics from the backtest's time-ordered trace.

    The trace must contain the columns:
      - timestamp (datetime)
      - cash (float)
      - position (float) — the signed YES position
      - mid_price (float) — the market's reference price
      - bid_p (float) — our quoted bid, may be NaN
      - ask_p (float) — our quoted ask, may be NaN
      - action (str, optional) — e.g. 'BUY', 'SELL', or empty for a state snapshot
      - order_status (str, optional) — e.g. 'FILLED'

    Metrics computed:
      - Total PnL (realised + position marked to mid)
      - Total return (%)
      - Sharpe ratio (annualised over daily returns)
      - Max drawdown (in absolute dollars and as a percentage)
      - Fill rate (percentage of placed orders that filled)
      - Inventory statistics (min, max, mean, standard deviation)
      - Average quoted spread (in price space)
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

    # Copy to avoid side effects
    df = trace_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Calcular Equity = Cash + Position * MidPrice
    df["equity"] = df["cash"] + df["position"] * df["mid_price"]

    # 1. PnL Total y Retorno
    final_equity = df["equity"].iloc[-1]
    total_pnl = final_equity - initial_cash
    total_return_pct = (total_pnl / initial_cash) * 100.0

    # 2. Sharpe Ratio Diario Anualizado
    # Resample to daily using each day's last known value (forward fill)
    df_daily = df.set_index("timestamp")["equity"].resample("D").last().ffill()
    daily_returns = df_daily.pct_change().dropna()

    # With too little data or no variation, Sharpe is 0
    if len(daily_returns) > 1 and daily_returns.std() > 1e-8:
        # Sharpe = (mean / std) * sqrt(365) — prediction markets trade 24/7/365
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
    # A trade is a confirmed fill
    trade_count = 0
    if "order_status" in df.columns:
        # Count the rows where a fill occurred
        trade_count = int((df["order_status"] == "FILLED").sum())

    submitted_count = 0
    if "action" in df.columns:
        # Count submitted orders (BUY or SELL actions)
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
