#!/usr/bin/env python3
"""
run_backtest.py
────────────────
Script interactivo CLI para ejecutar simulaciones de Backtesting,
sweeps de parámetros y análisis de near-resolution.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import UTC, datetime, timedelta

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# Asegurar que el directorio raíz está en el PATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backtesting.engine import BacktestEngine
from backtesting.scenarios.base_scenario import BaseScenario
from backtesting.scenarios.resolution_spike import ResolutionSpikeScenario
from normalizer.schema import (
    Market,
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
from storage.reader import MarketDataReader
from storage.writer import MarketDataWriter

console = Console()


def create_synthetic_db() -> str:
    """Crea una base de datos DuckDB temporal con datos sintéticos para demostración."""
    fd, path = tempfile.mkstemp(suffix="_demo.duckdb")
    os.close(fd)
    if os.path.exists(path):
        os.remove(path)

    writer = MarketDataWriter(path)

    # 1. Crear mercados
    mid = MarketId(Venue.KALSHI, "DEMO-MM-BT")
    market = Market(
        market_id=mid,
        question="Will the demonstration backtest complete successfully?",
        category=MarketCategory.SCIENCE,
        resolution=Resolution(resolution_date=datetime.now(UTC) + timedelta(hours=10)),
        status=MarketStatus.OPEN,
    )
    writer.write_market_sync(market)

    # 2. Escribir ticks y features
    base_time = datetime.now(UTC) - timedelta(hours=12)
    ticks = []
    features = []

    # Simular 30 ticks y features
    for i in range(30):
        t = base_time + timedelta(minutes=20 * i)
        # Precio medio oscilando entre 0.45 y 0.55
        mid_p = 0.50 + 0.03 * np.sin(i / 3.0)
        spread = 0.02
        ticks.append(
            Tick(
                market_id=mid,
                timestamp=t,
                tick_type=TickType.QUOTE,
                yes_bid=Price(mid_p - spread / 2),
                yes_ask=Price(mid_p + spread / 2),
                volume=Size(0.0),
                side=None,
            )
        )

        # Añadir trade ocasional
        if i % 6 == 0:
            ticks.append(
                Tick(
                    market_id=mid,
                    timestamp=t + timedelta(seconds=5),
                    tick_type=TickType.TRADE,
                    yes_bid=Price(mid_p - spread / 2),
                    yes_ask=Price(mid_p + spread / 2),
                    volume=Size(10.0),
                    side=Side.YES if i % 12 == 0 else Side.NO,
                )
            )

        features.append(
            {
                "market_id": str(mid),
                "venue": mid.venue.value,
                "timestamp": t,
                "obi": 0.2 * np.sin(i / 2.0),
                "quoted_spread": spread,
                "relative_spread": spread / mid_p,
                "belief_vol": 0.15,
                "ewma_vol": 0.10,
                "tau_years": max(0.0001, (10.0 - 20.0 * i / 60.0) / (365.25 * 24.0)),
                "mu_hat": 0.01 * np.cos(i / 4.0),
            }
        )

    writer.write_ticks_sync(ticks)
    writer.write_features_sync(features)
    writer.close()
    return path


def run_simple_backtest(db_path: str, market_id: str, strategy: str) -> None:
    console.print(Panel.fit(f"[bold cyan]Simple Backtest Run ({strategy.upper()})[/bold cyan]"))

    # Parámetros base
    strategy_params = {"gamma_I": 0.05, "kappa_x": 1.0}
    if strategy == "cartea_jaimungal":
        strategy_params.update({"phi": 1.5, "eta": 0.04, "rho": -0.2})

    engine = BacktestEngine(
        db_path=db_path,
        market_id=market_id,
        strategy_name=strategy,
        strategy_params=strategy_params,
        initial_cash=10000.0,
        order_size=10.0,
    )

    metrics, trace_df = engine.run()
    if trace_df.empty:
        console.print("[red]No traces generated. Is there tick data in the range?[/red]")
        return

    # Mostrar métricas clave en consola
    console.print("\n[bold green]Backtest Completed Successfully![/bold green]")
    console.print(f"  Total Trades: [bold]{metrics['trade_count']}[/bold]")
    console.print(
        f"  Total Return: [bold green]${metrics['total_pnl']:.2f}[/bold green]"
        f" ({metrics['total_return_pct']:.2f}%)"
    )
    console.print(
        f"  Max Drawdown: [bold red]${metrics['max_drawdown_usd']:.2f}[/bold red]"
        f" ({metrics['max_drawdown_pct']:.2f}%)"
    )
    console.print(f"  Sharpe Ratio: [bold]{metrics['sharpe_ratio']:.2f}[/bold]")
    console.print(f"  Average Spread: [bold]{metrics['avg_spread']:.4f}[/bold]")
    console.print(f"  Fill Rate: [bold]{metrics['fill_rate']:.1f}%[/bold]")
    console.print(
        f"  Position Bounds: [bold][{metrics['inv_min']:.0f}, {metrics['inv_max']:.0f}][/bold]"
    )
    console.print(f"  Position Std Dev: [bold]{metrics['inv_std']:.2f}[/bold]\n")


def run_parameter_sweep(db_path: str, market_id: str, strategy: str) -> None:
    console.print(Panel.fit(f"[bold cyan]Parameter Sweep ({strategy.upper()})[/bold cyan]"))
    scenario = BaseScenario(db_path=db_path, market_id=market_id)

    base_params = {"kappa_x": 1.0}
    if strategy == "cartea_jaimungal":
        base_params.update({"phi": 1.5, "eta": 0.04, "rho": -0.2})

    # Sweep de aversión al riesgo gamma_I
    sweep_values = [0.01, 0.05, 0.15, 0.30]

    scenario.run_parameter_sweep(
        strategy_name=strategy,
        sweep_param_name="gamma_I",
        sweep_values=sweep_values,
        base_params=base_params,
        order_size=10.0,
    )


def run_resolution_spike(
    db_path: str, market_id: str, strategy: str, resolution_time: datetime
) -> None:
    console.print(
        Panel.fit(f"[bold cyan]Near-Resolution Spike Analysis ({strategy.upper()})[/bold cyan]")
    )
    scenario = ResolutionSpikeScenario(db_path=db_path, market_id=market_id)

    strategy_params = {"gamma_I": 0.1, "kappa_x": 1.2}
    if strategy == "cartea_jaimungal":
        strategy_params.update({"phi": 2.0, "eta": 0.05, "rho": -0.3})

    scenario.analyze_resolution_period(
        strategy_name=strategy,
        strategy_params=strategy_params,
        resolution_time=resolution_time,
        hours_before=12.0,
        order_size=10.0,
    )


def select_local_market(db_path: str) -> tuple[str, datetime]:
    """Carga los mercados de DuckDB y solicita selección al usuario."""
    with MarketDataReader(db_path) as reader:
        markets_df = reader.markets()
        if markets_df.empty:
            console.print("[red]No markets found in local DuckDB database.[/red]")
            sys.exit(1)

        console.print("\n[bold cyan]Available Markets in DuckDB:[/bold cyan]")
        records = markets_df.to_dict("records")

        # Contar ticks y features por mercado, ordenar por más datos primero
        for r in records:
            mid = r["market_id"]
            r["_ticks"] = reader.count_ticks(mid)
            feat_df = reader.features(mid)
            r["_features"] = len(feat_df) if feat_df is not None else 0
        records.sort(key=lambda r: r["_ticks"], reverse=True)

        for idx, r in enumerate(records, 1):
            console.print(
                f"  [{idx}] {r['market_id']:<35} | "
                f"Ticks: [bold green]{r['_ticks']:>5}[/bold green] | "
                f"Features: [bold yellow]{r['_features']:>5}[/bold yellow] | "
                f"{r['status']}"
            )

        choice = Prompt.ask(
            f"\nSelect a market number (1-{len(records)})",
            default="1",
        )
        try:
            choice_idx = int(choice) - 1
            if choice_idx < 0 or choice_idx >= len(records):
                choice_idx = 0
        except ValueError:
            choice_idx = 0

        selected = records[choice_idx]
        res_date = selected["resolution_date"]
        if res_date.tzinfo is None:
            res_date = res_date.replace(tzinfo=UTC)

        return selected["market_id"], res_date


def main() -> None:
    console.print(
        Panel(
            "[bold green]Prediction Market Backtester CLI[/bold green]\n"
            "Ejecute simulaciones de market-making histórico en mercados de Kalshi y Polymarket.",
            subtitle="Framework de Backtesting",
        )
    )

    # 1. Seleccionar base de datos
    db_options = ["1", "2"]
    console.print("[bold]Database Source Options:[/bold]")
    console.print("  [1] Synthetic Data (Works immediately out-of-the-box)")
    console.print("  [2] Local DuckDB Database (Fits real ingest data)")

    db_choice = Prompt.ask("\nSelect option", choices=db_options, default="1")

    if db_choice == "1":
        db_path = create_synthetic_db()
        market_id = "kalshi:DEMO-MM-BT"
        # La resolución del demo está a +10 horas de base_time
        resolution_time = datetime.now(UTC) - timedelta(hours=12) + timedelta(hours=10)
        is_synthetic = True
    else:
        db_path = "./data/duckdb/markets.duckdb"
        if not os.path.exists(db_path):
            console.print(
                f"[red]Error: Database not found at {db_path}."
                " Please run real data ingest first.[/red]"
            )
            sys.exit(1)
        market_id, resolution_time = select_local_market(db_path)
        is_synthetic = False

    # 2. Seleccionar estrategia
    strategy = Prompt.ask(
        "Select Market Making Strategy",
        choices=["glft", "cartea_jaimungal"],
        default="glft",
    )

    # 3. Seleccionar modo de ejecución
    console.print("\n[bold]Execution Mode Options:[/bold]")
    console.print("  [1] Run Simple Backtest (Single performance report)")
    console.print("  [2] Run Parameter Sweep (Sweeps risk aversion gamma_I and prints comparison)")
    console.print("  [3] Run Near-Resolution Spike (Simulates closing period behaviour)")

    mode_choice = Prompt.ask("\nSelect execution mode", choices=["1", "2", "3"], default="1")

    console.print(f"\nSelected Market: [bold yellow]{market_id}[/bold yellow]")
    console.print(f"Using Strategy:  [bold yellow]{strategy.upper()}[/bold yellow]\n")

    if mode_choice == "1":
        run_simple_backtest(db_path, market_id, strategy)
    elif mode_choice == "2":
        run_parameter_sweep(db_path, market_id, strategy)
    elif mode_choice == "3":
        run_resolution_spike(db_path, market_id, strategy, resolution_time)

    # Cleanup synthetic temp db if generated
    if is_synthetic and os.path.exists(db_path):
        os.remove(db_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Execution cancelled by user.[/yellow]")
