"""
tests/unit/test_risk_limits.py
──────────────────────────────
Tests unitarios para el validador de límites de riesgo (RiskLimitsChecker).
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
    """Verifica que los precios estén estrictamente en [0.0001, 0.9999]."""
    # Precio normal pasa
    order_ok = Order("t1", market_id, OrderAction.BUY, Price(0.50), Size(1.0))
    ok, reason = checker.check_order(order_ok, current_position=0.0, q_max_effective=10.0)
    assert ok is True
    assert reason == ""

    # Precio en el borde extremo inferior falla
    order_low = Order("t2", market_id, OrderAction.BUY, Price(0.0), Size(1.0))
    ok, reason = checker.check_order(order_low, current_position=0.0, q_max_effective=10.0)
    assert ok is False
    assert "safe limits" in reason

    # Precio en el borde extremo superior falla
    order_high = Order("t3", market_id, OrderAction.BUY, Price(1.0), Size(1.0))
    ok, reason = checker.check_order(order_high, current_position=0.0, q_max_effective=10.0)
    assert ok is False
    assert "safe limits" in reason


def test_inventory_limits_buy(checker, market_id):
    """Verifica el límite de inventario firmado al comprar YES."""
    # Límite máximo de inventario = 5
    q_max = 5.0

    # Compra que nos lleva a posición 3 (OK)
    o1 = Order("t1", market_id, OrderAction.BUY, Price(0.50), Size(3.0))
    ok, reason = checker.check_order(o1, current_position=0.0, q_max_effective=q_max)
    assert ok is True

    # Compra que nos lleva a posición 6 (Rechazada)
    o2 = Order("t2", market_id, OrderAction.BUY, Price(0.50), Size(6.0))
    ok, reason = checker.check_order(o2, current_position=0.0, q_max_effective=q_max)
    assert ok is False
    assert "exceeds effective limit" in reason

    # Compra que reduce una posición corta (-4 -> -2) (OK)
    o3 = Order("t3", market_id, OrderAction.BUY, Price(0.50), Size(2.0))
    ok, reason = checker.check_order(o3, current_position=-4.0, q_max_effective=q_max)
    assert ok is True


def test_inventory_limits_sell(checker, market_id):
    """Verifica el límite de inventario firmado al vender YES."""
    q_max = 3.0

    # Venta que nos lleva a posición -2 (OK)
    o1 = Order("t1", market_id, OrderAction.SELL, Price(0.50), Size(2.0))
    ok, reason = checker.check_order(o1, current_position=0.0, q_max_effective=q_max)
    assert ok is True

    # Venta que nos lleva a posición -4 (Rechazada)
    o2 = Order("t2", market_id, OrderAction.SELL, Price(0.50), Size(4.0))
    ok, reason = checker.check_order(o2, current_position=0.0, q_max_effective=q_max)
    assert ok is False
    assert "exceeds effective limit" in reason
