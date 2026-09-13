"""
tests/unit/test_router.py
─────────────────────────
Unit tests for the order router (OrderRouter).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from execution.order import OrderAction, OrderStatus
from execution.paper.account import PaperAccount
from execution.paper.engine import PaperExecutionEngine
from execution.risk.circuit_breaker import CircuitBreaker
from execution.risk.limits import RiskLimitsChecker
from execution.risk.monitor import RiskMonitor
from execution.router import OrderRouter
from features.resolution import NearResolutionRegime
from normalizer.schema import Market, MarketCategory, MarketId, MarketStatus, Resolution, Venue
from strategies.market_making.glft import Quote


@pytest.fixture
def market_id() -> MarketId:
    return MarketId(Venue.KALSHI, "KXBTC-TEST")


@pytest.fixture
def market(market_id) -> Market:
    return Market(
        market_id=market_id,
        question="Test Question?",
        category=MarketCategory.OTHER,
        resolution=Resolution(resolution_date=datetime.now(UTC)),
        status=MarketStatus.OPEN,
    )


@pytest.fixture
def router() -> OrderRouter:
    account = PaperAccount(initial_cash=10000.0)
    engine = PaperExecutionEngine(account)
    limits = RiskLimitsChecker()
    breaker = CircuitBreaker(max_daily_loss=100.0)
    monitor = RiskMonitor(initial_cash=10000.0)
    return OrderRouter(
        paper_engine=engine,
        risk_limits=limits,
        circuit_breaker=breaker,
        risk_monitor=monitor,
        q_max_base=10.0,
    )


def test_quote_invalid_cancels_all(router, market_id):
    """A quote marked invalid must cancel every resting order."""
    account = router.paper_engine.account
    o1 = account.create_order(market_id, OrderAction.BUY, 0.40, 1.0)
    o1.status = OrderStatus.ACTIVE

    quote = Quote(
        market_id=market_id,
        timestamp=datetime.now(UTC),
        model="glft",
        mid_price_p=0.45,
        mid_price_X=0.0,
        inventory=0.0,
        tau_years=0.1,
        regime=NearResolutionRegime.NORMAL,
        gamma_I=0.1,
        kappa_x=1.0,
        belief_vol=0.1,
        sigma_bar_sq=0.01,
        reservation_X=0.0,
        half_spread_X=0.1,
        signal_skew=0.0,
        bid_X=-0.1,
        ask_X=0.1,
        bid_p=0.42,
        ask_p=0.48,
        is_valid=False,  # <--- INVALID
        invalid_reason="regime halt",
    )

    affected = router.on_quote(quote, size=1.0)
    assert len(affected) == 1
    assert affected[0] == o1.order_id
    assert o1.status == OrderStatus.CANCELLED


def test_circuit_breaker_halts_quoting(router, market_id):
    """
    Verifies that when the circuit breaker trips, every resting order is
    cancelled and quoting stops.
    """
    account = router.paper_engine.account
    o1 = account.create_order(market_id, OrderAction.BUY, 0.40, 1.0)
    o1.status = OrderStatus.ACTIVE

    # Force a large daily loss by adjusting the account cash directly
    account._cash_balance = 9000.0
    router.mid_prices[market_id] = 0.50
    router.risk_monitor.update(
        cash_balance=9000.0, positions={market_id: 0.0}, mid_prices=router.mid_prices
    )

    quote = Quote(
        market_id=market_id,
        timestamp=datetime.now(UTC),
        model="glft",
        mid_price_p=0.50,
        mid_price_X=0.0,
        inventory=0.0,
        tau_years=0.1,
        regime=NearResolutionRegime.NORMAL,
        gamma_I=0.1,
        kappa_x=1.0,
        belief_vol=0.1,
        sigma_bar_sq=0.01,
        reservation_X=0.0,
        half_spread_X=0.1,
        signal_skew=0.0,
        bid_X=-0.1,
        ask_X=0.1,
        bid_p=0.45,
        ask_p=0.55,
        is_valid=True,
        invalid_reason="",
    )

    affected = router.on_quote(quote, size=1.0)
    assert len(affected) == 1
    assert o1.status == OrderStatus.CANCELLED
    assert router.circuit_breaker.is_tripped is True


def test_place_and_replace_orders(router, market_id):
    """The router places new orders and replaces those whose price changed."""
    account = router.paper_engine.account

    quote1 = Quote(
        market_id=market_id,
        timestamp=datetime.now(UTC),
        model="glft",
        mid_price_p=0.50,
        mid_price_X=0.0,
        inventory=0.0,
        tau_years=0.1,
        regime=NearResolutionRegime.NORMAL,
        gamma_I=0.1,
        kappa_x=1.0,
        belief_vol=0.1,
        sigma_bar_sq=0.01,
        reservation_X=0.0,
        half_spread_X=0.1,
        signal_skew=0.0,
        bid_X=-0.1,
        ask_X=0.1,
        bid_p=0.45,
        ask_p=0.55,
        is_valid=True,
        invalid_reason="",
    )

    # 1. Initial placement
    affected = router.on_quote(quote1, size=1.0)
    assert len(affected) == 2  # one buy, one sell
    active = account.get_active_orders(market_id)
    assert len(active) == 2

    buy_o = next(o for o in active if o.action == OrderAction.BUY)
    sell_o = next(o for o in active if o.action == OrderAction.SELL)
    assert buy_o.price == 0.45
    assert sell_o.price == 0.55

    # 2. A redundant call at the same prices must not replace (avoids churning)
    affected2 = router.on_quote(quote1, size=1.0)
    assert len(affected2) == 0

    # 3. Calling with a changed price must cancel and create a new order
    quote2 = Quote(
        market_id=market_id,
        timestamp=datetime.now(UTC),
        model="glft",
        mid_price_p=0.51,
        mid_price_X=0.0,
        inventory=0.0,
        tau_years=0.1,
        regime=NearResolutionRegime.NORMAL,
        gamma_I=0.1,
        kappa_x=1.0,
        belief_vol=0.1,
        sigma_bar_sq=0.01,
        reservation_X=0.0,
        half_spread_X=0.1,
        signal_skew=0.0,
        bid_X=-0.1,
        ask_X=0.1,
        bid_p=0.46,  # <--- CAMBIA COMPRA
        ask_p=0.55,  # igual
        is_valid=True,
        invalid_reason="",
    )

    affected3 = router.on_quote(quote2, size=1.0)
    # It must contain 2 IDs: the cancellation of the old buy, and the new buy
    assert len(affected3) == 2
    assert buy_o.status == OrderStatus.CANCELLED

    active_now = account.get_active_orders(market_id)
    new_buy = next(o for o in active_now if o.action == OrderAction.BUY)
    assert new_buy.price == 0.46
    assert new_buy.order_id != buy_o.order_id


def test_near_resolution_side_halt(router, market_id, market):
    """
    Verifies that under a warning/critical regime, quoting stops on the side
    that would add inventory risk.
    """
    account = router.paper_engine.account

    # Assume we are long (q > 0)
    account._positions[market_id] = 5.0

    # Use a quote timestamp very close to resolution_date to force a
    # CRITICAL/WARNING regime
    now_ts = datetime.now(UTC)

    # Mock the resolution date so it lands 10 minutes out
    from datetime import timedelta

    market = Market(
        market_id=market_id,
        question="?",
        category=MarketCategory.OTHER,
        resolution=Resolution(resolution_date=now_ts + timedelta(minutes=10)),
        status=MarketStatus.OPEN,
    )

    quote = Quote(
        market_id=market_id,
        timestamp=now_ts,
        model="glft",
        mid_price_p=0.50,
        mid_price_X=0.0,
        inventory=5.0,  # largo
        tau_years=10.0 / (365.25 * 24 * 60),
        regime=NearResolutionRegime.CRITICAL,
        gamma_I=0.1,
        kappa_x=1.0,
        belief_vol=0.1,
        sigma_bar_sq=0.01,
        reservation_X=0.0,
        half_spread_X=0.1,
        signal_skew=0.0,
        bid_X=-0.1,
        ask_X=0.1,
        bid_p=0.45,
        ask_p=0.55,
        is_valid=True,
        invalid_reason="",
    )

    router.on_quote(quote, size=1.0, market=market)
    active = account.get_active_orders(market_id)

    # Long inventory under CRITICAL must disable the BUY side
    # So there must be exactly 1 active order, and it must be a SELL
    assert len(active) == 1
    assert active[0].action == OrderAction.SELL
    assert active[0].price == 0.55
