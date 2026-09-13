"""
execution/paper/account.py
──────────────────────────
Simulated account for paper trading.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from execution.order import Order, OrderAction, OrderStatus, OrderType
from normalizer.schema import MarketId, Price, Side, Size

log = logging.getLogger(__name__)


class PaperAccount:
    """
    Manages the cash balance and the simulated paper positions.

    Why a cash balance plus a signed YES position:
      In market-making theory, a negative YES position is equivalent to a
      positive NO position. Modelling it as a signed float on YES (q_t ∈ ℝ)
      simplifies the inventory equations and keeps direct compatibility with
      the GLFT and CJ quoters.
    """

    def __init__(self, initial_cash: float = 10000.0) -> None:
        self._cash_balance: float = initial_cash
        self._positions: dict[MarketId, float] = {}
        self._orders: dict[str, Order] = {}

    @property
    def cash_balance(self) -> float:
        """The current (simulated) cash balance."""
        return self._cash_balance

    @property
    def positions(self) -> dict[MarketId, float]:
        """Signed YES positions, keyed by market."""
        return self._positions.copy()

    @property
    def orders(self) -> dict[str, Order]:
        """All orders, keyed by order_id."""
        return self._orders.copy()

    def get_position(self, market_id: MarketId) -> float:
        """Net signed YES position in a market."""
        return self._positions.get(market_id, 0.0)

    def get_balance(self) -> float:
        """Current cash balance."""
        return self._cash_balance

    def get_active_orders(self, market_id: MarketId | None = None) -> list[Order]:
        """
        Return the resting orders in the system.

        Args:
            market_id: optional; when given, filters to that market only.
        """
        active = [o for o in self._orders.values() if o.is_active]
        if market_id:
            active = [o for o in active if o.market_id == market_id]
        return active

    def create_order(
        self,
        market_id: MarketId,
        action: OrderAction,
        price: Price,
        size: Size,
        outcome: Side = Side.YES,
    ) -> Order:
        """
        Create and register a new order in PENDING state.
        """
        order_id = f"paper_{uuid.uuid4().hex[:8]}"
        now = datetime.now(UTC)

        order = Order(
            order_id=order_id,
            market_id=market_id,
            action=action,
            outcome=outcome,
            price=price,
            size=size,
            order_type=OrderType.LIMIT,
            filled_size=Size(0.0),
            status=OrderStatus.PENDING,
            created_at=now,
            updated_at=now,
        )

        self._orders[order_id] = order
        log.info(
            f"order_created: order_id={order_id}, market_id={market_id},"
            f" action={action.value}, price={float(price)}, size={float(size)}"
        )
        return order

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a resting order.

        Returns:
            True when the order was cancelled, False when it was not cancellable.
        """
        order = self._orders.get(order_id)
        if not order or not order.is_active:
            log.warning(f"cancel_failed — order not active or not found: order_id={order_id}")
            return False

        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now(UTC)
        log.info(f"order_cancelled: order_id={order_id}, market_id={order.market_id}")
        return True

    def fill_order(self, order_id: str, fill_size: Size, fill_price: Price) -> Order | None:
        """
        Apply a partial or complete fill to a resting order.

        Updates the order state, the cash balance and the YES position.

        Returns:
            The modified order, or None when the fill could not be applied.
        """
        order = self._orders.get(order_id)
        if not order or not order.is_active:
            log.warning(f"fill_failed — order not active or not found: order_id={order_id}")
            return None

        if fill_size <= 0:
            log.warning(
                f"fill_failed — fill_size must be positive:"
                f" order_id={order_id}, fill_size={fill_size}"
            )
            return None

        # Never exceed the remaining size
        fill_size = Size(min(fill_size, order.remaining_size))

        order.filled_size = Size(order.filled_size + fill_size)
        if float(order.remaining_size) == 0.0:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.ACTIVE

        order.updated_at = datetime.now(UTC)

        # Update balances according to the action
        # Buying YES: reduces cash, increases the position
        # Selling YES: increases cash, reduces the position
        if order.action == OrderAction.BUY:
            self._cash_balance -= float(fill_size * fill_price)
            self._positions[order.market_id] = self._positions.get(order.market_id, 0.0) + float(
                fill_size
            )
        elif order.action == OrderAction.SELL:
            self._cash_balance += float(fill_size * fill_price)
            self._positions[order.market_id] = self._positions.get(order.market_id, 0.0) - float(
                fill_size
            )

        log.info(
            f"order_filled: order_id={order_id}, market_id={order.market_id}, "
            f"fill_size={float(fill_size)}, fill_price={float(fill_price)}, "
            f"status={order.status.value}, new_position={self._positions[order.market_id]}, "
            f"new_cash={self._cash_balance}"
        )
        return order
