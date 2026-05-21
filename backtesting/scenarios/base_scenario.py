"""
backtesting/scenarios/base_scenario.py
──────────────────────────────────────
Clase base y utilidades para escenarios de backtesting y sweeps de parámetros.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from backtesting.engine import BacktestEngine

log = logging.getLogger(__name__)


class BaseScenario:
    """
    Clase base para construir escenarios y ejecutar análisis cuantitativos ( sweeps).
    """

    def __init__(self, db_path: str, market_id: str) -> None:
        self.db_path = db_path
        self.market_id = market_id
        self.console = Console()

    def run_single(
        self,
        strategy_name: str,
        strategy_params: dict[str, Any],
        start: datetime | None = None,
        end: datetime | None = None,
        initial_cash: float = 10000.0,
        order_size: float = 1.0,
        risk_params: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """
        Ejecuta una corrida de backtest simple.
        """
        engine = BacktestEngine(
            db_path=self.db_path,
            market_id=self.market_id,
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            start=start,
            end=end,
            initial_cash=initial_cash,
            order_size=order_size,
            risk_params=risk_params,
        )
        return engine.run()

    def run_parameter_sweep(
        self,
        strategy_name: str,
        sweep_param_name: str,
        sweep_values: list[Any],
        base_params: dict[str, Any],
        start: datetime | None = None,
        end: datetime | None = None,
        initial_cash: float = 10000.0,
        order_size: float = 1.0,
        risk_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Ejecuta sweeps de parámetros sobre una única variable y muestra una tabla de comparación.
        """
        results = []

        for val in sweep_values:
            params = base_params.copy()
            params[sweep_param_name] = val

            log.info("Running sweep item: %s = %s", sweep_param_name, val)
            metrics, _ = self.run_single(
                strategy_name=strategy_name,
                strategy_params=params,
                start=start,
                end=end,
                initial_cash=initial_cash,
                order_size=order_size,
                risk_params=risk_params,
            )

            result_item = {
                "sweep_value": val,
                **metrics,
            }
            results.append(result_item)

        self._print_sweep_table(strategy_name, sweep_param_name, results)
        return results

    def _print_sweep_table(
        self, strategy_name: str, param_name: str, results: list[dict[str, Any]]
    ) -> None:
        """
        Imprime una tabla comparativa estilizada con los resultados del sweep de parámetros.
        """
        table = Table(
            title=f"Parameter Sweep on Strategy: {strategy_name.upper()} ({self.market_id})",
            header_style="bold magenta",
        )

        table.add_column(f"Param: {param_name}", justify="right")
        table.add_column("Total P&L ($)", justify="right")
        table.add_column("Return (%)", justify="right")
        table.add_column("Sharpe Ratio", justify="right")
        table.add_column("Max DD (%)", justify="right")
        table.add_column("Trades Count", justify="right")
        table.add_column("Fill Rate (%)", justify="right")
        table.add_column("Inv Range [min, max]", justify="center")
        table.add_column("Inv Std Dev", justify="right")

        for r in results:
            val_str = str(r["sweep_value"])
            pnl_style = "green" if r["total_pnl"] >= 0 else "red"

            table.add_row(
                val_str,
                f"[{pnl_style}]${r['total_pnl']:.2f}[/{pnl_style}]",
                f"{r['total_return_pct']:.2f}%",
                f"{r['sharpe_ratio']:.2f}",
                f"{r['max_drawdown_pct']:.2f}%",
                str(r["trade_count"]),
                f"{r['fill_rate']:.1f}%",
                f"[{r['inv_min']:.0f}, {r['inv_max']:.0f}]",
                f"{r['inv_std']:.2f}",
            )

        self.console.print(table)
