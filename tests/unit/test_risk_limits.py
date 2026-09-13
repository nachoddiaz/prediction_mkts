"""
tests/unit/test_risk_limits.py
──────────────────────────────
Unit tests for the pre-trade risk validator (RiskLimitsChecker).
"""

from __future__ import annotations

import pytest

from execution.order import Order, OrderAction
from execution.risk.limits import RiskLimitsChecker
from normalizer.schema import MarketId, Price, Size, Venue


@pytest.fixture
def market_id() -> MarketId:
    return MarketId(Venue.KALSHI, "KXBTC-TEST")


@pytest.fixture
def checker() -> RiskLimitsChecker:
    return RiskLimitsChecker()


def test_price_bounds(checker, market_id):
    """Prices must lie strictly within [0.0001, 0.9999]."""
    # A normal price passes
    order_ok = Order("t1", market_id, OrderAction.BUY, Price(0.50), Size(1.0))
    ok, reason = checker.check_order(order_ok, current_position=0.0, q_max_effective=10.0)
    assert ok is True
    assert reason == ""

    # A price at the extreme lower edge fails
    order_low = Order("t2", market_id, OrderAction.BUY, Price(0.0), Size(1.0))
    ok, reason = checker.check_order(order_low, current_position=0.0, q_max_effective=10.0)
    assert ok is False
    assert "safe limits" in reason

    # A price at the extreme upper edge fails
    order_high = Order("t3", market_id, OrderAction.BUY, Price(1.0), Size(1.0))
    ok, reason = checker.check_order(order_high, current_position=0.0, q_max_effective=10.0)
    assert ok is False
    assert "safe limits" in reason


def test_inventory_limits_buy(checker, market_id):
    """Verify the signed inventory limit when buying YES."""
    # Maximum inventory limit = 5
    q_max = 5.0

    # A buy taking us to position 3 (accepted)
    o1 = Order("t1", market_id, OrderAction.BUY, Price(0.50), Size(3.0))
    ok, reason = checker.check_order(o1, current_position=0.0, q_max_effective=q_max)
    assert ok is True

    # A buy taking us to position 6 (rejected)
    o2 = Order("t2", market_id, OrderAction.BUY, Price(0.50), Size(6.0))
    ok, reason = checker.check_order(o2, current_position=0.0, q_max_effective=q_max)
    assert ok is False
    assert "exceeds effective limit" in reason

    # A buy reducing a short position (-4 → -2) (accepted)
    o3 = Order("t3", market_id, OrderAction.BUY, Price(0.50), Size(2.0))
    ok, reason = checker.check_order(o3, current_position=-4.0, q_max_effective=q_max)
    assert ok is True


def test_inventory_limits_sell(checker, market_id):
    """Verify the signed inventory limit when selling YES."""
    q_max = 3.0

    # A sell taking us to position -2 (accepted)
    o1 = Order("t1", market_id, OrderAction.SELL, Price(0.50), Size(2.0))
    ok, reason = checker.check_order(o1, current_position=0.0, q_max_effective=q_max)
    assert ok is True

    # A sell taking us to position -4 (rejected)
    o2 = Order("t2", market_id, OrderAction.SELL, Price(0.50), Size(4.0))
    ok, reason = checker.check_order(o2, current_position=0.0, q_max_effective=q_max)
    assert ok is False
    assert "exceeds effective limit" in reason
