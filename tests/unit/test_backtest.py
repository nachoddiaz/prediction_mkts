"""
tests/unit/test_backtest.py
───────────────────────────
Unit tests for the backtesting framework.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import pytest

from backtesting.engine import BacktestEngine
from backtesting.metrics import calculate_metrics
from backtesting.scenarios.base_scenario import BaseScenario
from backtesting.scenarios.resolution_spike import ResolutionSpikeScenario
from normalizer.schema import (
    MarketCategory,
    MarketId,
    MarketStatus,
    Price,
    Resolution,
    Side,
    Size,
    Tick,
    TickType,
    Venue,
)
from storage.writer import MarketDataWriter


def test_calculate_metrics_empty() -> None:
    """With empty input the metrics must come back zero-initialised."""
    res = calculate_metrics(pd.DataFrame(), 10000.0)
    assert res["total_pnl"] == 0.0
    assert res["total_return_pct"] == 0.0
    assert res["sharpe_ratio"] == 0.0
    assert res["max_drawdown_usd"] == 0.0
    assert res["max_drawdown_pct"] == 0.0
    assert res["trade_count"] == 0
    assert res["fill_rate"] == 0.0


def test_calculate_metrics_synthetic() -> None:
    """Return, drawdown and Sharpe computed over a synthetic trace."""
    # Simulate 3 days of activity with equity closes:
    #   Day 1: 10000 -> 10100
    #   Day 2: 10100 -> 9800  (drawdown from the 10100 peak)
    #   Day 3: 9800  -> 10500
    base_time = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
    records = [
        # t0
        {
            "timestamp": base_time,
            "cash": 10000.0,
            "position": 0.0,
            "mid_price": 0.5,
            "bid_p": 0.49,
            "ask_p": 0.51,
            "action": "BUY",
            "order_status": "FILLED",
        },
        # t1 (day 1)
        {
            "timestamp": base_time + timedelta(days=1),
            "cash": 9950.0,
            "position": 100.0,
            "mid_price": 10100.0 / 100.0 - 99.5,  # equity = 10100
            "bid_p": 0.50,
            "ask_p": 0.52,
            "action": None,
            "order_status": None,
        },
        # t2 (day 2)
        {
            "timestamp": base_time + timedelta(days=2),
            "cash": 9950.0,
            "position": 100.0,
            "mid_price": 9800.0 / 100.0 - 99.5,  # equity = 9800
            "bid_p": 0.48,
            "ask_p": 0.50,
            "action": None,
            "order_status": None,
        },
        # t3 (day 3)
        {
            "timestamp": base_time + timedelta(days=3),
            "cash": 10500.0,
            "position": 0.0,
            "mid_price": 0.55,
            "bid_p": 0.54,
            "ask_p": 0.56,
            "action": "SELL",
            "order_status": "FILLED",
        },
    ]
    df = pd.DataFrame(records)
    res = calculate_metrics(df, 10000.0)

    # Validar PnL y Retorno
    assert res["total_pnl"] == 500.0
    assert res["total_return_pct"] == 5.0

    # Maximum drawdown
    # Peak = 10100. Trough = 9800. USD DD = 300. Pct DD = 300 / 10100 = 2.9703%
    assert res["max_drawdown_usd"] == pytest.approx(300.0)
    assert res["max_drawdown_pct"] == pytest.approx(300.0 / 10100.0 * 100.0, abs=1e-3)

    # Validar contadores de trades
    assert res["trade_count"] == 2
    assert res["fill_rate"] == 100.0


@pytest.fixture
def temp_db() -> str:
    """Fixture creating a temporary DuckDB file populated with test data."""
    fd, path = tempfile.mkstemp(suffix=".duckdb")
    os.close(fd)
    if os.path.exists(path):
        os.remove(path)

    writer = MarketDataWriter(path)

    # 1. Create markets
    mid = MarketId(Venue.MANIFOLD, "test_market")
    from normalizer.schema import Market

    base_time = datetime.now(tz=UTC) - timedelta(hours=12)
    res_date = base_time + timedelta(hours=10)
    market = Market(
        market_id=mid,
        question="Is this a test?",
        category=MarketCategory.SCIENCE,
        resolution=Resolution(resolution_date=res_date),
        status=MarketStatus.OPEN,
    )
    writer.write_market_sync(market)

    # 2. Escribir ticks (QUOTE y TRADE)
    ticks = []
    # Generate a few ticks over 10 hours
    for i in range(20):
        t = base_time + timedelta(minutes=30 * i)
        # Algunos quotes
        ticks.append(
            from_tick_helper(
                mid=mid,
                ts=t,
                bid=0.45 + 0.002 * i,
                ask=0.47 + 0.002 * i,
                tick_type=TickType.QUOTE,
            )
        )
        if i % 5 == 0:
            # Insertar trade
            ticks.append(
                from_tick_helper(
                    mid=mid,
                    ts=t + timedelta(seconds=10),
                    bid=0.45 + 0.002 * i,
                    ask=0.47 + 0.002 * i,
                    tick_type=TickType.TRADE,
                    volume=5.0,
                    side=Side.YES,
                )
            )

    writer.write_ticks_sync(ticks)

    # 3. Escribir features alineadas
    features = []
    for i in range(20):
        t = base_time + timedelta(minutes=30 * i)
        features.append(
            {
                "market_id": str(mid),
                "venue": mid.venue.value,
                "timestamp": t,
                "obi": 0.1 * (i % 3 - 1),
                "quoted_spread": 0.02,
                "relative_spread": 4.5,
                "belief_vol": 0.12,
                "ewma_vol": 0.08,
                "tau_years": (2.0 - i * 0.02) / 365.25,
                "mu_hat": 0.005 * (i % 2 - 0.5),
            }
        )
    writer.write_features_sync(features)
    writer.close()

    yield path

    # Cleanup
    if os.path.exists(path):
        os.remove(path)


def from_tick_helper(
    mid: MarketId,
    ts: datetime,
    bid: float,
    ask: float,
    tick_type: TickType,
    volume: float = 0.0,
    side: Side | None = None,
) -> Tick:
    return Tick(
        market_id=mid,
        timestamp=ts,
        tick_type=tick_type,
        yes_bid=Price(bid),
        yes_ask=Price(ask),
        volume=Size(volume),
        side=side if tick_type == TickType.TRADE else None,
    )


def test_backtest_engine_runs_glft(temp_db: str) -> None:
    """The BacktestEngine runs correctly with the GLFT strategy."""
    engine = BacktestEngine(
        db_path=temp_db,
        market_id="manifold:test_market",
        strategy_name="glft",
        strategy_params={"gamma_I": 0.05, "kappa_x": 1.2},
        initial_cash=5000.0,
        order_size=2.0,
    )

    metrics, trace_df = engine.run()

    assert not trace_df.empty
    assert "equity" in trace_df.columns
    assert "position" in trace_df.columns
    assert metrics["trade_count"] >= 0
    assert metrics["avg_spread"] > 0.0


def test_backtest_engine_runs_cj(temp_db: str) -> None:
    """
    Verifies that the BacktestEngine runs correctly with the
    Cartea-Jaimungal.
    """
    engine = BacktestEngine(
        db_path=temp_db,
        market_id="manifold:test_market",
        strategy_name="cartea_jaimungal",
        strategy_params={
            "gamma_I": 0.05,
            "kappa_x": 10.0,
            "phi": 2.0,
            "eta": 0.02,
            # rho is ρ_μ ∈ (0,1] since v2.2 — the measure-change discount,
            # not the price-signal correlation.
            "rho": 0.3,
        },
        initial_cash=5000.0,
        order_size=1.0,
    )

    metrics, trace_df = engine.run()

    assert not trace_df.empty
    assert "equity" in trace_df.columns
    assert metrics["trade_count"] >= 0


def test_scenario_sweeps(temp_db: str) -> None:
    """BaseScenario can run parameter sweeps without error."""
    scenario = BaseScenario(db_path=temp_db, market_id="manifold:test_market")

    results = scenario.run_parameter_sweep(
        strategy_name="glft",
        sweep_param_name="gamma_I",
        sweep_values=[0.01, 0.05, 0.1],
        base_params={"kappa_x": 1.0},
    )

    assert len(results) == 3
    assert results[0]["sweep_value"] == 0.01
    assert results[1]["sweep_value"] == 0.05
    assert results[2]["sweep_value"] == 0.1
    # Each run must contain the expected metrics
    assert "total_pnl" in results[0]


def test_resolution_spike_scenario(temp_db: str) -> None:
    """ResolutionSpikeScenario analyses the closing period correctly."""
    scenario = ResolutionSpikeScenario(db_path=temp_db, market_id="manifold:test_market")

    # Simulate a period close to expiry (the last 12 hours, aligned to base_time)
    base_time = datetime.now(tz=UTC) - timedelta(hours=12)
    res_time = base_time + timedelta(hours=10)
    metrics, trace_df = scenario.analyze_resolution_period(
        strategy_name="glft",
        strategy_params={"gamma_I": 0.05, "kappa_x": 1.0},
        resolution_time=res_time,
        hours_before=12.0,
    )

    assert not trace_df.empty
    assert "equity" in trace_df.columns
