"""
execution/risk/monitor.py
──────────────────────────
Real-time metrics, PnL computation and cumulative daily-loss tracking.
"""

from __future__ import annotations

import logging

from normalizer.schema import MarketId

log = logging.getLogger(__name__)


class RiskMonitor:
    """
    Monitors the account's financial and operational health.

    Equity computation:
      Equity = cash balance + Σ(position * mid price)

    Daily-loss computation:
      Daily Loss = max(0.0, Initial Cash - Current Equity)

    Additional statistics:
      - Volumen total transaccionado (en contratos/shares).
      - Number of successful executions.
    """

    def __init__(self, initial_cash: float) -> None:
        self.initial_cash: float = initial_cash
        self._current_equity: float = initial_cash
        self._pnl: float = 0.0
        self._daily_loss: float = 0.0

        # Operational metrics
        self.total_trades: int = 0
        self.total_volume: float = 0.0

    @property
    def current_equity(self) -> float:
        """Estimated total account value (cash + position value)."""
        return self._current_equity

    @property
    def pnl(self) -> float:
        """Cumulative PnL against the starting balance."""
        return self._pnl

    @property
    def daily_loss(self) -> float:
        """Cumulative loss against the starting balance (always >= 0)."""
        return self._daily_loss

    def update(
        self,
        cash_balance: float,
        positions: dict[MarketId, float],
        mid_prices: dict[MarketId, float],
    ) -> float:
        """
        Update the portfolio valuation and recompute PnL.

        Args:
            cash_balance: the current cash balance.
            positions:    signed YES position per market.
            mid_prices:   latest mid price per market (as a probability).

        Returns:
            The day's cumulative loss (daily_loss).
        """
        position_value = 0.0
        for m_id, pos in positions.items():
            # Without a recent mid price, fall back to 0.5
            mid = mid_prices.get(m_id, 0.5)
            position_value += pos * mid

        self._current_equity = cash_balance + position_value
        self._pnl = self._current_equity - self.initial_cash

        # Loss is the negative difference against the initial capital
        self._daily_loss = max(0.0, -self._pnl)

        log.debug(
            f"portfolio_valuation: cash={cash_balance}, pos_val={position_value}, "
            f"equity={self._current_equity}, pnl={self._pnl}, daily_loss={self._daily_loss}"
        )
        return self._daily_loss

    def record_trade(self, size: float, price: float) -> None:
        """Record one execution in the operational volume metrics."""
        self.total_trades += 1
        self.total_volume += size
        log.info(
            f"trade_recorded_in_monitor: trades={self.total_trades}, "
            f"volume={self.total_volume}, trade_size={size}, trade_price={price}"
        )
