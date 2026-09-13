"""
execution/order.py
───────────────────
Schemas and types for execution-layer orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from normalizer.schema import MarketId, Price, Side, Size


class OrderAction(str, Enum):
    """The order's action: BUY or SELL."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type: LIMIT or MARKET."""

    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(str, Enum):
    """
    The order's lifecycle state.

    PENDING   → Sent to the engine but not yet confirmed by the venue.
    ACTIVE    → Confirmed and resting in the order book.
    FILLED    → Fully executed.
    CANCELLED → Cancelled by us before completion.
    REJECTED  → Rejected by the engine or the venue (e.g. on risk limits).
    """

    PENDING = "pending"
    ACTIVE = "active"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """
    An order placed in the system.

    Why filled_size, status and updated_at are mutable:
      Unlike market-data primitives (ticks, order books), which represent
      immutable events in time, an order is a business entity with mutable
      state — it has a lifecycle.
    """

    order_id: str
    market_id: MarketId
    action: OrderAction
    price: Price
    size: Size
    outcome: Side = Side.YES
    order_type: OrderType = OrderType.LIMIT
    filled_size: Size = field(default_factory=lambda: Size(0.0))
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if not 0.0 <= self.price <= 1.0:
            raise ValueError(f"Price must be in [0, 1], got {self.price}")
        if self.size <= 0:
            raise ValueError(f"Size must be positive, got {self.size}")
        if self.filled_size < 0 or self.filled_size > self.size:
            raise ValueError(f"Invalid filled size {self.filled_size} for order size {self.size}")

    @property
    def remaining_size(self) -> Size:
        """Quantity of the order still to be executed."""
        return Size(self.size - self.filled_size)

    @property
    def is_active(self) -> bool:
        """True while the order can still be filled or cancelled."""
        return self.status in (OrderStatus.PENDING, OrderStatus.ACTIVE)
