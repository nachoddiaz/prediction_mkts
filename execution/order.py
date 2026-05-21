"""
execution/order.py
───────────────────
Esquemas y tipos para órdenes de la capa de ejecución.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from normalizer.schema import MarketId, Price, Side, Size


class OrderAction(str, Enum):
    """Acción de la orden: COMPRAR o VENDER."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Tipo de orden: LIMIT (orden limitada) o MARKET (orden a mercado)."""

    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(str, Enum):
    """
    Estado de ciclo de vida de la orden.

    PENDING   → Enviada al motor pero no confirmada por la venue.
    ACTIVE    → Confirmada y reposando en el libro de órdenes.
    FILLED    → Completada al 100%.
    CANCELLED → Cancelada por el usuario antes de completarse.
    REJECTED  → Rechazada por el motor o por la venue (ej. por límites de riesgo).
    """

    PENDING = "pending"
    ACTIVE = "active"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """
    Representa una orden colocada en el sistema.

    Por qué mutable en filled_size, status y updated_at:
      A diferencia de las primitivas de datos de mercado (ticks, orderbooks)
      que representan eventos inmutables en el tiempo, una orden representa
      una entidad de negocio con estado mutable (ciclo de vida).
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
        """Cantidad restante de la orden por ejecutar."""
        return Size(self.size - self.filled_size)

    @property
    def is_active(self) -> bool:
        """Devuelve True si la orden aún puede ser completada o cancelada."""
        return self.status in (OrderStatus.PENDING, OrderStatus.ACTIVE)
