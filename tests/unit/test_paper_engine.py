"""
tests/unit/test_paper_engine.py
────────────────────────────────
Unit tests for the simulated execution engine (PaperExecutionEngine).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from execution.order import OrderAction, OrderStatus
from execution.paper.account import PaperAccount
from execution.paper.engine import PaperExecutionEngine
from normalizer.schema import (
    MarketId,
    MarketSnapshot,
    OrderBook,
    OrderBookLevel,
    Price,
    Side,
    Size,
    Tick,
    TickType,
    Venue,
)


@pytest.fixture
def market_id() -> MarketId:
    return MarketId(Venue.KALSHI, "KXBTC-TEST")


@pytest.fixture
def engine() -> PaperExecutionEngine:
    account = PaperAccount(initial_cash=10000.0)
    return PaperExecutionEngine(account)


def test_buy_limit_crossed_by_ask(engine, market_id):
    """A buy order fills when the market's best offer crosses its price."""
    account = engine.account
    order = account.create_order(market_id, OrderAction.BUY, Price(0.45), Size(10.0))
    order.status = OrderStatus.ACTIVE

    # A quote tick where the market ask drops to 0.44, crossing our 0.45 buy
    tick = Tick(
        market_id=market_id,
        timestamp=datetime.now(UTC),
        tick_type=TickType.QUOTE,
        yes_bid=Price(0.42),
        yes_ask=Price(0.44),
    )

    filled = engine.process_tick(tick)
    assert len(filled) == 1
    assert filled[0].order_id == order.order_id
    assert filled[0].status == OrderStatus.FILLED
    assert account.get_position(market_id) == 10.0


def test_buy_limit_passive_trade(engine, market_id):
    """
    Verifies passive filling of a buy order when a trade prints at a price
    inferior o igual.
    """
    account = engine.account
    order = account.create_order(market_id, OrderAction.BUY, Price(0.40), Size(10.0))
    order.status = OrderStatus.ACTIVE

    # Generate a TRADE tick at 0.39 with volume 4.0
    tick = Tick(
        market_id=market_id,
        timestamp=datetime.now(UTC),
        tick_type=TickType.TRADE,
        yes_bid=Price(0.39),
        yes_ask=Price(0.42),
        volume=Size(4.0),
        side=Side.YES,
    )

    filled = engine.process_tick(tick)
    assert len(filled) == 1
    assert filled[0].status == OrderStatus.ACTIVE
    assert filled[0].filled_size == 4.0  # capped by the trade's volume
    assert account.get_position(market_id) == 4.0


def test_sell_limit_crossed_by_bid(engine, market_id):
    """A sell order fills when the market's best bid crosses its price."""
    account = engine.account
    order = account.create_order(market_id, OrderAction.SELL, Price(0.55), Size(5.0))
    order.status = OrderStatus.ACTIVE

    # Ticker with a bid at 0.56 (crossing our sell at 0.55)
    tick = Tick(
        market_id=market_id,
        timestamp=datetime.now(UTC),
        tick_type=TickType.QUOTE,
        yes_bid=Price(0.56),
        yes_ask=Price(0.58),
    )

    filled = engine.process_tick(tick)
    assert len(filled) == 1
    assert filled[0].status == OrderStatus.FILLED
    assert account.get_position(market_id) == -5.0


def test_process_snapshot_crossing(engine, market_id):
    """Verify executions when processing full order book snapshots."""
    account = engine.account

    # Place active buy and sell orders
    buy_order = account.create_order(market_id, OrderAction.BUY, Price(0.52), Size(10.0))
    buy_order.status = OrderStatus.ACTIVE

    sell_order = account.create_order(market_id, OrderAction.SELL, Price(0.43), Size(5.0))
    sell_order.status = OrderStatus.ACTIVE

    # Crear orderbook
    ob = OrderBook(
        market_id=market_id,
        timestamp=datetime.now(UTC),
        bids=(
            OrderBookLevel(Price(0.43), Size(100.0)),
        ),  # mejor bid a 0.43 crosses sell_order at 0.43
        asks=(
            OrderBookLevel(Price(0.52), Size(100.0)),
        ),  # mejor ask a 0.52 crosses buy_order at 0.52
    )

    from normalizer.schema import Market, MarketCategory, MarketStatus, Resolution

    market = Market(
        market_id=market_id,
        question="Test?",
        category=MarketCategory.OTHER,
        resolution=Resolution(datetime.now(UTC)),
        status=MarketStatus.OPEN,
    )
    snapshot = MarketSnapshot(market=market, orderbook=ob)

    filled = engine.process_snapshot(snapshot)
    assert len(filled) == 2
    assert buy_order.status == OrderStatus.FILLED
    assert sell_order.status == OrderStatus.FILLED
    assert account.get_position(market_id) == 10.0 - 5.0  # +5.0 neto
