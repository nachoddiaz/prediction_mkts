"""
tests/unit/test_backtest.py
───────────────────────────
Tests unitarios para el framework de Backtesting.
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
    """Verifica que el cálculo con datos vacíos devuelva métricas inicializadas a cero."""
    res = calculate_metrics(pd.DataFrame(), 10000.0)
    assert res["total_pnl"] == 0.0
    assert res["total_return_pct"] == 0.0
    assert res["sharpe_ratio"] == 0.0
    assert res["max_drawdown_usd"] == 0.0
    assert res["max_drawdown_pct"] == 0.0
    assert res["trade_count"] == 0
    assert res["fill_rate"] == 0.0


def test_calculate_metrics_synthetic() -> None:
    """Verifica cálculos de rendimiento, drawdown y Sharpe con una traza sintética."""
    # Simular 3 días de actividad con Close de Equity:
    # Día 1: 10000 -> 10100
    # Día 2: 10100 -> 9800  (drawdown respecto al pico de 10100)
    # Día 3: 9800  -> 10500
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
        # t1 (Día 1)
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
        # t2 (Día 2)
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
        # t3 (Día 3)
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

    # Drawdown máximo
    # Peak = 10100. Trough = 9800. USD DD = 300. Pct DD = 300 / 10100 = 2.9703%
    assert res["max_drawdown_usd"] == pytest.approx(300.0)
    assert res["max_drawdown_pct"] == pytest.approx(300.0 / 10100.0 * 100.0, abs=1e-3)

    # Validar contadores de trades
    assert res["trade_count"] == 2
    assert res["fill_rate"] == 100.0


@pytest.fixture
def temp_db() -> str:
    """Fixture que crea un archivo DuckDB temporal poblado con datos de prueba."""
    fd, path = tempfile.mkstemp(suffix=".duckdb")
    os.close(fd)
    if os.path.exists(path):
        os.remove(path)

    writer = MarketDataWriter(path)

    # 1. Crear mercados
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
    # Generar algunos ticks a lo largo de 10 horas
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
    """Verifica que el BacktestEngine se ejecute correctamente usando la estrategia GLFT."""
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
    Verifica que el BacktestEngine se ejecute correctamente usando la estrategia
    Cartea-Jaimungal.
    """
    engine = BacktestEngine(
        db_path=temp_db,
        market_id="manifold:test_market",
        strategy_name="cartea_jaimungal",
        strategy_params={
            "gamma_I": 0.05,
            "kappa_x": 1.2,
            "phi": 2.0,
            "eta": 0.02,
            "rho": -0.3,
        },
        initial_cash=5000.0,
        order_size=1.0,
    )

    metrics, trace_df = engine.run()

    assert not trace_df.empty
    assert "equity" in trace_df.columns
    assert metrics["trade_count"] >= 0


def test_scenario_sweeps(temp_db: str) -> None:
    """Verifica que el BaseScenario pueda realizar sweeps de parámetros sin errores."""
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
    # Cada corrida debe contener las métricas esperadas
    assert "total_pnl" in results[0]


def test_resolution_spike_scenario(temp_db: str) -> None:
    """Verifica que el ResolutionSpikeScenario analice correctamente el periodo de cierre."""
    scenario = ResolutionSpikeScenario(db_path=temp_db, market_id="manifold:test_market")

    # Simular periodo cercano al cierre (últimas 12 horas alineadas con base_time)
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
