"""
tests/unit/test_paper_account.py
─────────────────────────────────
Unit tests for the simulated account (PaperAccount).
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
    """The account's initial state is correct."""
    assert account.cash_balance == 10000.0
    assert len(account.positions) == 0
    assert len(account.orders) == 0
    assert account.get_balance() == 10000.0


def test_create_order(account, market_id):
    """Orders are created in PENDING state."""
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

    # Confirm it is in the order registry
    assert order.order_id in account.orders
    assert len(account.get_active_orders(market_id)) == 1


def test_cancel_order(account, market_id):
    """Resting orders can be cancelled."""
    order = account.create_order(
        market_id=market_id,
        action=OrderAction.BUY,
        price=Price(0.45),
        size=Size(10.0),
    )

    # Confirm the order (move it to ACTIVE)
    order.status = OrderStatus.ACTIVE

    # Cancel the order
    success = account.cancel_order(order.order_id)
    assert success is True
    assert order.status == OrderStatus.CANCELLED
    assert order.is_active is False
    assert len(account.get_active_orders(market_id)) == 0

    # Cancelling again must fail
    success_retry = account.cancel_order(order.order_id)
    assert success_retry is False


def test_fill_buy_order_total(account, market_id):
    """A BUY order fills completely."""
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

    # Cash balance: 10000 - 4.00 = 9996.00
    assert account.cash_balance == 9996.0
    # YES position: +10.0
    assert account.get_position(market_id) == 10.0


def test_fill_sell_order_partial(account, market_id):
    """A SELL order fills partially."""
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

    # Second partial fill (7 contracts in excess of those remaining)
    partial_o_2 = account.fill_order(order.order_id, Size(10.0), Price(0.60))

    assert partial_o_2 is not None
    assert partial_o_2.status == OrderStatus.FILLED
    assert partial_o_2.filled_size == 10.0
    assert account.cash_balance == 10006.0
    assert account.get_position(market_id) == -10.0
