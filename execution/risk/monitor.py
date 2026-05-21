"""
execution/risk/monitor.py
──────────────────────────
Métricas en tiempo real, cálculo de PnL y seguimiento de pérdida diaria acumulada.
"""

from __future__ import annotations

import logging

from normalizer.schema import MarketId

log = logging.getLogger(__name__)


class RiskMonitor:
    """
    Monitoriza la salud financiera y operativa de la cuenta.

    Cálculo de Equity:
      Equity = Saldo en Caja + Sumatorio(Posición * Mid-Price)

    Cálculo de Pérdida Diaria:
      Daily Loss = max(0.0, Initial Cash - Current Equity)

    Estadísticas adicionales:
      - Volumen total transaccionado (en contratos/shares).
      - Número de ejecuciones exitosas.
    """

    def __init__(self, initial_cash: float) -> None:
        self.initial_cash: float = initial_cash
        self._current_equity: float = initial_cash
        self._pnl: float = 0.0
        self._daily_loss: float = 0.0

        # Métricas operativas
        self.total_trades: int = 0
        self.total_volume: float = 0.0

    @property
    def current_equity(self) -> float:
        """Valor total estimado de la cuenta (Caja + Valor de Posiciones)."""
        return self._current_equity

    @property
    def pnl(self) -> float:
        """PnL acumulado respecto al balance inicial."""
        return self._pnl

    @property
    def daily_loss(self) -> float:
        """Pérdida acumulada respecto al balance inicial (siempre >= 0)."""
        return self._daily_loss

    def update(
        self,
        cash_balance: float,
        positions: dict[MarketId, float],
        mid_prices: dict[MarketId, float],
    ) -> float:
        """
        Actualiza el estado de valoración del portfolio y recalcula el PnL.

        Args:
            cash_balance: Saldo actual en efectivo.
            positions:    Diccionario con la posición firmada de YES por mercado.
            mid_prices:   Diccionario con el último mid-price de cada mercado (probabilidad).

        Returns:
            La pérdida acumulada en el día (daily_loss).
        """
        position_value = 0.0
        for m_id, pos in positions.items():
            # Si no tenemos mid-price reciente, usamos 0.5 por defecto
            mid = mid_prices.get(m_id, 0.5)
            position_value += pos * mid

        self._current_equity = cash_balance + position_value
        self._pnl = self._current_equity - self.initial_cash

        # Pérdida es la diferencia negativa con respecto al capital inicial
        self._daily_loss = max(0.0, -self._pnl)

        log.debug(
            f"portfolio_valuation: cash={cash_balance}, pos_val={position_value}, "
            f"equity={self._current_equity}, pnl={self._pnl}, daily_loss={self._daily_loss}"
        )
        return self._daily_loss

    def record_trade(self, size: float, price: float) -> None:
        """Registra una ejecución en las métricas operativas de volumen."""
        self.total_trades += 1
        self.total_volume += size
        log.info(
            f"trade_recorded_in_monitor: trades={self.total_trades}, "
            f"volume={self.total_volume}, trade_size={size}, trade_price={price}"
        )
