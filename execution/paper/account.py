"""
execution/paper/account.py
──────────────────────────
Cuenta simulada para paper trading.
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
    Gestiona el saldo de caja y las posiciones simuladas en papel.

    Por qué saldo de caja y posición de YES con signo:
      En teoría de MM, mantener una posición negativa en YES es equivalente
      a mantener una posición positiva en NO. Al modelarlo como un float con
      signo de YES (q_t ∈ ℝ), simplificamos las ecuaciones de inventario y
      mantenemos compatibilidad directa con los quoters de GLFT y CJ.
    """

    def __init__(self, initial_cash: float = 10000.0) -> None:
        self._cash_balance: float = initial_cash
        self._positions: dict[MarketId, float] = {}
        self._orders: dict[str, Order] = {}

    @property
    def cash_balance(self) -> float:
        """Saldo de caja actual (simulado)."""
        return self._cash_balance

    @property
    def positions(self) -> dict[MarketId, float]:
        """Diccionario de posiciones de YES firmadas por mercado."""
        return self._positions.copy()

    @property
    def orders(self) -> dict[str, Order]:
        """Diccionario de todas las órdenes por order_id."""
        return self._orders.copy()

    def get_position(self, market_id: MarketId) -> float:
        """Retorna la posición neta (firmada) de YES en un mercado."""
        return self._positions.get(market_id, 0.0)

    def get_balance(self) -> float:
        """Retorna el saldo de caja."""
        return self._cash_balance

    def get_active_orders(self, market_id: MarketId | None = None) -> list[Order]:
        """
        Retorna las órdenes activas en el sistema.

        Args:
            market_id: Opcional. Si se proporciona, filtra solo las de ese mercado.
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
        Crea y registra una nueva orden en estado PENDING.
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
        Cancela una orden activa.

        Returns:
            True si la orden fue cancelada con éxito, False si no era cancelable.
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
        Ejecuta un fill parcial o total sobre una orden activa.

        Actualiza el estado de la orden, el saldo de caja y la posición de YES.

        Returns:
            La orden modificada, o None si no se pudo aplicar el fill.
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

        # Asegurar no sobrepasar el tamaño restante
        fill_size = Size(min(fill_size, order.remaining_size))

        order.filled_size = Size(order.filled_size + fill_size)
        if order.remaining_size == 0:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.ACTIVE

        order.updated_at = datetime.now(UTC)

        # Actualizar balances según la acción
        # Compra de YES: reduce caja, aumenta posición
        # Venta de YES: aumenta caja, reduce posición
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
