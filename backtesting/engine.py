"""
backtesting/engine.py
─────────────────────
Motor de backtesting histórico síncrono para estrategias de market-making.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from backtesting.metrics import calculate_metrics
from execution.paper.account import PaperAccount
from execution.paper.engine import PaperExecutionEngine
from execution.risk.circuit_breaker import CircuitBreaker
from execution.risk.limits import RiskLimitsChecker
from execution.risk.monitor import RiskMonitor
from execution.router import OrderRouter
from features.resolution import compute_resolution_features
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
)
from storage.reader import MarketDataReader
from strategies.market_making.cartea_jaimungal import CarteaJaimungalQuoter
from strategies.market_making.glft import GLFTQuoter

log = logging.getLogger(__name__)


class BacktestEngine:
    """
    Ejecuta simulaciones de backtesting histórico para estrategias.

    Carga ticks y features de forma síncrona desde DuckDB, realiza el
    emparejamiento temporal (merge_asof), y corre paso a paso
    reutilizando las clases de Paper Trading y Risk Management.
    """

    def __init__(
        self,
        db_path: str,
        market_id: str,
        strategy_name: str,
        strategy_params: dict[str, Any],
        start: datetime | None = None,
        end: datetime | None = None,
        initial_cash: float = 10000.0,
        order_size: float = 1.0,
        risk_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            db_path:         Ruta al archivo DuckDB
            market_id:       ID del mercado a evaluar (ej. 'manifold:xyz')
            strategy_name:   Nombre de la estrategia ('glft' o 'cartea_jaimungal')
            strategy_params: Parámetros del quoter de la estrategia
            start:           Fecha de inicio del backtest
            end:             Fecha de fin del backtest
            initial_cash:    Capital inicial en dólares
            order_size:      Tamaño de los quotes (default 1 contract)
            risk_params:     Configuración de riesgo (limites, circuit breaker)
        """
        self.db_path = db_path
        self.market_id = market_id
        self.strategy_name = strategy_name.lower()
        self.strategy_params = strategy_params
        self.start = start
        self.end = end
        self.initial_cash = initial_cash
        self.order_size = order_size
        self.risk_params = risk_params or {
            "max_daily_loss": 500.0,
            "q_max_base": 100.0,
        }

    def run(self) -> tuple[dict[str, Any], pd.DataFrame]:
        """
        Ejecuta el backtest y calcula las métricas.

        Returns:
            Tuple con:
              - Dict[str, Any] conteniendo las métricas cuantitativas
              - pd.DataFrame con la traza detallada del backtest
        """
        market_id_obj = MarketId.from_str(self.market_id)

        # 1. Conectar a base de datos y leer información
        with MarketDataReader(self.db_path) as reader:
            # Leer metadatos del mercado
            market_df = reader.market(self.market_id)
            if not market_df.empty:
                row = market_df.iloc[0]
                # Asegurar timezone UTC
                res_date = row["resolution_date"]
                if res_date.tzinfo is None:
                    res_date = res_date.replace(tzinfo=UTC)

                market = Market(
                    market_id=market_id_obj,
                    question=row["question"],
                    category=MarketCategory(row["category"]),
                    resolution=Resolution(
                        resolution_date=res_date,
                        resolved_value=row["resolved_value"],
                    ),
                    status=MarketStatus(row["status"]),
                )
            else:
                # Fallback predeterminado para tests o mercados inexistentes
                res_date = datetime.now(tz=UTC) + timedelta(days=30)
                market = Market(
                    market_id=market_id_obj,
                    question="Unknown Market",
                    category=MarketCategory.OTHER,
                    resolution=Resolution(resolution_date=res_date),
                    status=MarketStatus.OPEN,
                )

            # Leer ticks y features
            ticks_df = reader.ticks(self.market_id, start=self.start, end=self.end)
            features_df = reader.features(self.market_id, start=self.start, end=self.end)

        # Si no hay datos, retornar estructura vacía
        if ticks_df.empty:
            log.warning("No ticks found for market %s in range. Backtest aborted.", self.market_id)
            return calculate_metrics(pd.DataFrame(), self.initial_cash), pd.DataFrame()

        # 2. Ordenar y alinear ticks y features temporalmente
        ticks_df["timestamp"] = pd.to_datetime(ticks_df["timestamp"])
        features_df["timestamp"] = pd.to_datetime(features_df["timestamp"])
        ticks_df = ticks_df.sort_values("timestamp")
        features_df = features_df.sort_values("timestamp")

        # merge_asof para alinear los ticks con la feature más reciente en t
        merged_df = pd.merge_asof(
            ticks_df,
            features_df,
            on="timestamp",
            direction="backward",
        )

        # Rellenar valores nulos de features si los hay (e.g. antes de la primera feature)
        merged_df = merged_df.ffill().bfill()

        # 3. Inicializar Estrategia Quoter
        if self.strategy_name == "glft":
            quoter = GLFTQuoter(
                gamma_I=self.strategy_params.get("gamma_I", 0.1),
                kappa_x=self.strategy_params.get("kappa_x", 0.8),
            )
        elif self.strategy_name == "cartea_jaimungal":
            quoter = CarteaJaimungalQuoter(
                gamma_I=self.strategy_params.get("gamma_I", 0.1),
                kappa_x=self.strategy_params.get("kappa_x", 0.8),
                phi=self.strategy_params.get("phi", 1.0),
                eta=self.strategy_params.get("eta", 0.05),
                rho=self.strategy_params.get("rho", 0.0),
            )
        else:
            raise ValueError(f"Unknown strategy name: {self.strategy_name}")

        # 4. Inicializar Componentes de Simulación de Ejecución y Riesgo
        account = PaperAccount(initial_cash=self.initial_cash)
        execution_engine = PaperExecutionEngine(account)

        limits_checker = RiskLimitsChecker()
        circuit_breaker = CircuitBreaker(
            max_daily_loss=self.risk_params.get("max_daily_loss", 500.0)
        )
        risk_monitor = RiskMonitor(
            initial_cash=self.initial_cash,
        )
        router = OrderRouter(
            paper_engine=execution_engine,
            risk_limits=limits_checker,
            circuit_breaker=circuit_breaker,
            risk_monitor=risk_monitor,
            q_max_base=self.risk_params.get("q_max_base", 100.0),
        )

        trace_records = []

        # 5. Loop de Simulación Histórica
        for _, row in merged_df.iterrows():
            ts = row["timestamp"]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)

            # Reconstruir Tick
            volume = float(row.get("volume", 0.0))
            side_str = row.get("side")
            side = Side(side_str) if pd.notna(side_str) and side_str else None

            tick = Tick(
                market_id=market_id_obj,
                timestamp=ts,
                tick_type=TickType(row["tick_type"]),
                yes_bid=Price(row["yes_bid"]),
                yes_ask=Price(row["yes_ask"]),
                volume=Size(volume),
                side=side,
            )

            # Extraer features
            belief_vol = float(row.get("belief_vol", 0.1))
            tau_years = float(row.get("tau_years", 1.0))
            mu_hat = float(row.get("mu_hat", 0.0))

            # Resolver régimen near-resolution
            resolution_feats = compute_resolution_features(
                market.resolution.resolution_date,
                now=ts,
            )
            regime = resolution_feats.regime

            # Generar cotización de la estrategia
            current_position = account.get_position(market_id_obj)

            if self.strategy_name == "glft":
                quote = quoter.quote(
                    market_id=market_id_obj,
                    mid_p=tick.mid,
                    inventory=current_position,
                    tau_years=tau_years,
                    belief_vol=belief_vol,
                    regime=regime,
                    timestamp=ts,
                )
            else:  # cartea_jaimungal
                quote = quoter.quote(
                    market_id=market_id_obj,
                    mid_p=tick.mid,
                    inventory=current_position,
                    tau_years=tau_years,
                    belief_vol=belief_vol,
                    regime=regime,
                    mu_hat=mu_hat,
                    timestamp=ts,
                )

            # Enrutamiento de órdenes óptimas (ajustar las limit orders en el book)
            router.on_quote(
                quote=quote,
                size=self.order_size,
                market=market,
            )

            # Simular emparejamiento contra el tick actual
            filled_orders = execution_engine.process_tick(tick)

            # Registrar snapshots del estado finalizado en t
            equity = account.cash_balance + current_position * tick.mid

            if filled_orders:
                for fo in filled_orders:
                    trace_records.append(
                        {
                            "timestamp": ts,
                            "cash": account.cash_balance,
                            "position": account.get_position(market_id_obj),
                            "mid_price": tick.mid,
                            "bid_p": quote.bid_p if quote.is_valid else np.nan,
                            "ask_p": quote.ask_p if quote.is_valid else np.nan,
                            "equity": equity,
                            "action": fo.action.value,
                            "order_status": fo.status.value,
                            "filled_size": float(fo.filled_size),
                            "order_price": float(fo.price),
                        }
                    )
            else:
                trace_records.append(
                    {
                        "timestamp": ts,
                        "cash": account.cash_balance,
                        "position": current_position,
                        "mid_price": tick.mid,
                        "bid_p": quote.bid_p if quote.is_valid else np.nan,
                        "ask_p": quote.ask_p if quote.is_valid else np.nan,
                        "equity": equity,
                        "action": None,
                        "order_status": None,
                        "filled_size": 0.0,
                        "order_price": np.nan,
                    }
                )

        # 6. Compilar DataFrame final y calcular métricas globales
        trace_df = pd.DataFrame(trace_records)
        metrics = calculate_metrics(trace_df, self.initial_cash)

        return metrics, trace_df
