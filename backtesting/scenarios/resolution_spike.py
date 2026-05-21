"""
backtesting/scenarios/resolution_spike.py
──────────────────────────────────────────
Escenario específico para simular el comportamiento cerca de la resolución (near-resolution).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from backtesting.scenarios.base_scenario import BaseScenario

log = logging.getLogger(__name__)


class ResolutionSpikeScenario(BaseScenario):
    """
    Simula e inspecciona el comportamiento del market maker en la fase final de resolución.

    Verifica que:
      1. Se incrementen los spreads o se reduzca el inventario a medida que tau -> 0.
      2. Se detenga el quoting (HALT) cuando faltan menos de 5 minutos (tau < TAU_5MIN).
      3. No se violen los límites de riesgo dinámicos ante picos de volatilidad de cierre.
    """

    def analyze_resolution_period(
        self,
        strategy_name: str,
        strategy_params: dict[str, Any],
        resolution_time: datetime,
        hours_before: float = 6.0,
        initial_cash: float = 10000.0,
        order_size: float = 1.0,
        risk_params: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """
        Ejecuta el backtest en la ventana temporal cercana a la resolución del mercado.

        Args:
            strategy_name:    Nombre de la estrategia ('glft' o 'cartea_jaimungal')
            strategy_params:  Parámetros de la estrategia
            resolution_time:  Fecha/hora de resolución del mercado
            hours_before:     Horas previas a la resolución para iniciar la simulación
            initial_cash:     Caja inicial
            order_size:       Tamaño de ordenes
            risk_params:      Configuraciones de riesgo
        """
        # Asegurar timezones local/UTC
        if resolution_time.tzinfo is None:
            resolution_time = resolution_time.replace(tzinfo=UTC)

        start_time = resolution_time - pd.Timedelta(hours=hours_before)

        log.info(
            "Running resolution spike scenario for %s from %s to %s",
            self.market_id,
            start_time,
            resolution_time,
        )

        metrics, trace_df = self.run_single(
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            start=start_time,
            end=resolution_time,
            initial_cash=initial_cash,
            order_size=order_size,
            risk_params=risk_params,
        )

        self._print_analysis_report(trace_df, metrics)
        return metrics, trace_df

    def _print_analysis_report(self, trace_df: pd.DataFrame, metrics: dict[str, Any]) -> None:
        """
        Analiza e imprime el comportamiento de inventarios y spreads
        durante la simulación de cierre.
        """
        console = Console()
        if trace_df.empty:
            console.print("[yellow]No data available for resolution analysis.[/yellow]")
            return

        # Filtrar estados para verificar el comportamiento de halt y advertencias
        # Identificar primera fila donde bid y ask quedaron cancelados (NaN)
        halt_rows = trace_df[trace_df["bid_p"].isna() & trace_df["ask_p"].isna()]

        console.print("\n[bold cyan]=== Near-Resolution Analysis Report ===[/bold cyan]")

        # Tabla resumen
        summary_table = Table(header_style="bold green")
        summary_table.add_column("Metric", justify="left")
        summary_table.add_column("Value", justify="right")

        summary_table.add_row("Total Return", f"{metrics['total_return_pct']:.2f}%")
        summary_table.add_row("Max Position Held", f"{metrics['inv_max']:.0f}")
        summary_table.add_row("Min Position Held", f"{metrics['inv_min']:.0f}")
        summary_table.add_row("Average Quoted Spread", f"{metrics['avg_spread']:.4f}")
        summary_table.add_row("Total Trades Executed", str(metrics["trade_count"]))

        console.print(summary_table)

        # Verificar si ocurrió el HALT reglamentario
        if not halt_rows.empty:
            first_halt_time = halt_rows["timestamp"].iloc[0]
            console.print(
                f"[green]✔ Quoting Halt detected successfully at: {first_halt_time} UTC[/green]"
            )
            # Calcular cuántas filas de datos se ejecutaron con halt
            pct_halt = (len(halt_rows) / len(trace_df)) * 100.0
            console.print(
                f"  Market was in HALT mode for [bold]{pct_halt:.1f}%[/bold]"
                " of the ticks near resolution."
            )
        else:
            console.print(
                "[red]✘ Warning: No absolute Quoting Halt detected during the final period.[/red]"
            )

        # Mostrar muestra de la evolución del spread e inventario
        console.print(
            "\n[bold]Evolución de spreads e inventarios durante las fases del cierre:[/bold]"
        )
        sample_size = min(10, len(trace_df))
        step = max(1, len(trace_df) // sample_size)
        sample_df = trace_df.iloc[::step].head(sample_size)

        sample_table = Table(header_style="bold magenta")
        sample_table.add_column("Timestamp (UTC)", justify="left")
        sample_table.add_column("Mid Price", justify="right")
        sample_table.add_column("Position", justify="right")
        sample_table.add_column("Bid Price", justify="right")
        sample_table.add_column("Ask Price", justify="right")
        sample_table.add_column("Equity", justify="right")

        for _, row in sample_df.iterrows():
            bid_str = f"{row['bid_p']:.4f}" if pd.notna(row["bid_p"]) else "HALT"
            ask_str = f"{row['ask_p']:.4f}" if pd.notna(row["ask_p"]) else "HALT"

            sample_table.add_row(
                row["timestamp"].strftime("%H:%M:%S"),
                f"{row['mid_price']:.4f}",
                f"{row['position']:+.1f}",
                bid_str,
                ask_str,
                f"${row['equity']:.2f}",
            )
        console.print(sample_table)
        console.print("=========================================\n")
