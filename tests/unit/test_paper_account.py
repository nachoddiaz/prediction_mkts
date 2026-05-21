"""
tests/unit/test_paper_account.py
─────────────────────────────────
Tests unitarios para la cuenta simulada (PaperAccount).
"""

from __future__ import annotations

import pytest

from execution.order import OrderAction, OrderStatus
from execution.paper.account import PaperAccount
from normalizer.schema import MarketId, Price, Side, Size, Venue


@pytest.fixture
def market_id() -> MarketId:
    return MarketId(Venue.KALSHI, "KXBTC-2026-T85000")


@pytest.fixture
def account() -> PaperAccount:
    return PaperAccount(initial_cash=10000.0)


def test_initial_state(account):
    """Verifica que el estado inicial de la cuenta sea correcto."""
    assert account.cash_balance == 10000.0
    assert len(account.positions) == 0
    assert len(account.orders) == 0
    assert account.get_balance() == 10000.0


def test_create_order(account, market_id):
    """Verifica la creación correcta de órdenes en estado PENDING."""
    order = account.create_order(
        market_id=market_id,
        action=OrderAction.BUY,
        price=Price(0.45),
        size=Size(10.0),
    )

    assert order.order_id.startswith("paper_")
    assert order.market_id == market_id
    assert order.action == OrderAction.BUY
    assert order.outcome == Side.YES
    assert order.price == 0.45
    assert order.size == 10.0
    assert order.filled_size == 0.0
    assert order.status == OrderStatus.PENDING
    assert order.is_active is True

    # Comprobar que está en el registro de órdenes
    assert order.order_id in account.orders
    assert len(account.get_active_orders(market_id)) == 1


def test_cancel_order(account, market_id):
    """Verifica la cancelación de órdenes activas."""
    order = account.create_order(
        market_id=market_id,
        action=OrderAction.BUY,
        price=Price(0.45),
        size=Size(10.0),
    )

    # Confirmar orden (poner en ACTIVE)
    order.status = OrderStatus.ACTIVE

    # Cancelar la orden
    success = account.cancel_order(order.order_id)
    assert success is True
    assert order.status == OrderStatus.CANCELLED
    assert order.is_active is False
    assert len(account.get_active_orders(market_id)) == 0

    # Re-cancelar debe fallar
    success_retry = account.cancel_order(order.order_id)
    assert success_retry is False


def test_fill_buy_order_total(account, market_id):
    """Verifica un fill completo de una orden de COMPRA."""
    order = account.create_order(
        market_id=market_id,
        action=OrderAction.BUY,
        price=Price(0.40),
        size=Size(10.0),
    )
    order.status = OrderStatus.ACTIVE

    # Fill completo (10 contratos a 0.40 = 4.00 cash gastado)
    filled_o = account.fill_order(order.order_id, Size(10.0), Price(0.40))

    assert filled_o is not None
    assert filled_o.status == OrderStatus.FILLED
    assert filled_o.filled_size == 10.0
    assert filled_o.remaining_size == 0.0

    # Balance de caja: 10000 - 4.00 = 9996.00
    assert account.cash_balance == 9996.0
    # Posición de YES: +10.0
    assert account.get_position(market_id) == 10.0


def test_fill_sell_order_partial(account, market_id):
    """Verifica fills parciales de una orden de VENTA."""
    order = account.create_order(
        market_id=market_id,
        action=OrderAction.SELL,
        price=Price(0.60),
        size=Size(10.0),
    )
    order.status = OrderStatus.ACTIVE

    # Primer fill parcial (3 contratos a 0.60 = +1.80 cash recibido)
    partial_o_1 = account.fill_order(order.order_id, Size(3.0), Price(0.60))

    assert partial_o_1 is not None
    assert partial_o_1.status == OrderStatus.ACTIVE
    assert partial_o_1.filled_size == 3.0
    assert partial_o_1.remaining_size == 7.0
    assert account.cash_balance == 10001.8
    assert account.get_position(market_id) == -3.0

    # Segundo fill parcial (7 contratos excedentes de los restantes)
    partial_o_2 = account.fill_order(order.order_id, Size(10.0), Price(0.60))

    assert partial_o_2 is not None
    assert partial_o_2.status == OrderStatus.FILLED
    assert partial_o_2.filled_size == 10.0
    assert account.cash_balance == 10006.0
    assert account.get_position(market_id) == -10.0
