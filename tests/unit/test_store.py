"""
tests/unit/test_store.py
──────────────────────────
Tests for the FeatureStore.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from features.store import FeatureStore
from normalizer.schema import (
    Market,
    MarketCategory,
    MarketId,
    MarketSnapshot,
    MarketStatus,
    OrderBook,
    OrderBookLevel,
    Price,
    Resolution,
    Size,
    Tick,
    TickType,
    Venue,
)
from storage.reader import MarketDataReader
from storage.writer import MarketDataWriter

# ---------------------------------------------------------------------------
# DB compartida
# ---------------------------------------------------------------------------


class DB:
    def __init__(self) -> None:
        self.w = MarketDataWriter(db_path=":memory:")
        self.r = MarketDataReader.__new__(MarketDataReader)
        self.r._con = self.w._con
        self.store = FeatureStore(self.r, self.w)

    def close(self) -> None:
        self.w.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MID = MarketId(Venue.KALSHI, "KXBTC-TEST")
RESOLUTION_DATE = datetime(2026, 7, 1, tzinfo=UTC)
NOW = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)


def make_market() -> Market:
    return Market(
        market_id=MID,
        question="Will BTC hit $100k?",
        category=MarketCategory.CRYPTO,
        resolution=Resolution(
            resolution_date=RESOLUTION_DATE,
            resolved_value=None,
        ),
        status=MarketStatus.OPEN,
    )


def make_tick(ts: datetime | None = None) -> Tick:
    return Tick(
        market_id=MID,
        timestamp=ts or NOW,
        tick_type=TickType.QUOTE,
        yes_bid=Price(0.44),
        yes_ask=Price(0.46),
        volume=Size(0.0),
        side=None,
    )


def make_orderbook() -> OrderBook:
    return OrderBook(
        market_id=MID,
        timestamp=NOW,
        bids=(
            OrderBookLevel(Price(0.44), Size(500.0)),
            OrderBookLevel(Price(0.43), Size(300.0)),
        ),
        asks=(
            OrderBookLevel(Price(0.46), Size(200.0)),
            OrderBookLevel(Price(0.47), Size(100.0)),
        ),
    )


def populate(db: DB, n_ticks: int = 20) -> None:
    """Insert a market, an order book and ticks into the database."""
    db.w.write_market_sync(make_market())
    db.w.write_orderbook_sync(make_orderbook())
    base = NOW - timedelta(minutes=n_ticks)
    db.w.write_ticks_sync(
        [
            Tick(
                market_id=MID,
                timestamp=base + timedelta(minutes=i),
                tick_type=TickType.QUOTE,
                yes_bid=Price(round(0.43 + i * 0.0005, 4)),
                yes_ask=Price(round(0.45 + i * 0.0005, 4)),
                volume=Size(0.0),
                side=None,
            )
            for i in range(n_ticks)
        ]
    )


# ---------------------------------------------------------------------------
# Tests compute_and_store
# ---------------------------------------------------------------------------


class TestComputeAndStore:
    def test_returns_true_with_data(self) -> None:
        db = DB()
        populate(db)
        tau = (RESOLUTION_DATE - NOW).total_seconds() / (365.25 * 24 * 3600)
        result = db.store.compute_and_store("kalshi:KXBTC-TEST", tau)
        assert result is True
        db.close()

    def test_returns_false_without_data(self) -> None:
        db = DB()
        result = db.store.compute_and_store("kalshi:NO-EXISTE", 0.5)
        assert result is False
        db.close()

    def test_persists_into_features_table(self) -> None:
        db = DB()
        populate(db)
        tau = (RESOLUTION_DATE - NOW).total_seconds() / (365.25 * 24 * 3600)
        db.store.compute_and_store("kalshi:KXBTC-TEST", tau)
        count = db.w._con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
        assert count == 1
        db.close()

    def test_features_carry_correct_tau(self) -> None:
        db = DB()
        populate(db)
        tau = 0.19
        db.store.compute_and_store("kalshi:KXBTC-TEST", tau)
        row = db.w._con.execute("SELECT tau_years FROM features").fetchone()
        assert row[0] == pytest.approx(tau, rel=1e-4)
        db.close()


# ---------------------------------------------------------------------------
# Tests compute_and_store_batch
# ---------------------------------------------------------------------------


class TestBatch:
    def test_batch_single_market(self) -> None:
        db = DB()
        populate(db)
        n = db.store.compute_and_store_batch([make_market()])
        assert n == 1
        db.close()

    def test_batch_excludes_markets_without_data(self) -> None:
        """A market with no ticks generates no features — it does not count."""
        db = DB()
        # Insert only the market, with no ticks
        db.w.write_market_sync(make_market())
        n = db.store.compute_and_store_batch([make_market()])
        assert n == 0
        db.close()


# ---------------------------------------------------------------------------
# Tests on_tick y on_snapshot
# ---------------------------------------------------------------------------


class TestHooks:
    def test_on_tick_with_data(self) -> None:
        db = DB()
        populate(db)
        result = db.store.on_tick(make_tick(), make_market())
        assert result is True
        db.close()

    def test_on_snapshot_with_data(self) -> None:
        db = DB()
        populate(db)
        snapshot = MarketSnapshot(
            market=make_market(),
            orderbook=make_orderbook(),
            last_tick=make_tick(),
        )
        result = db.store.on_snapshot(snapshot)
        assert result is True
        db.close()

    def test_on_snapshot_without_tick_is_false(self) -> None:
        """A snapshot without last_tick generates no features."""
        db = DB()
        populate(db)
        snapshot = MarketSnapshot(
            market=make_market(),
            orderbook=make_orderbook(),
            last_tick=None,
        )
        result = db.store.on_snapshot(snapshot)
        assert result is False
        db.close()


# ---------------------------------------------------------------------------
# Tests latest
# ---------------------------------------------------------------------------


class TestLatest:
    def test_returns_dict_with_fields(self) -> None:
        db = DB()
        populate(db)
        tau = (RESOLUTION_DATE - NOW).total_seconds() / (365.25 * 24 * 3600)
        db.store.compute_and_store("kalshi:KXBTC-TEST", tau)
        features = db.store.latest("kalshi:KXBTC-TEST")
        assert features is not None
        required = {"obi", "belief_vol", "ewma_vol", "tau_years", "mu_hat"}
        assert required.issubset(features.keys())
        db.close()

    def test_none_when_no_features(self) -> None:
        db = DB()
        assert db.store.latest("kalshi:NO-EXISTE") is None
        db.close()

    def test_values_are_floats(self) -> None:
        db = DB()
        populate(db)
        tau = (RESOLUTION_DATE - NOW).total_seconds() / (365.25 * 24 * 3600)
        db.store.compute_and_store("kalshi:KXBTC-TEST", tau)
        features = db.store.latest("kalshi:KXBTC-TEST")
        assert features is not None
        assert isinstance(features["obi"], float)
        assert isinstance(features["ewma_vol"], float)
        assert isinstance(features["tau_years"], float)
        assert isinstance(features["mu_hat"], float)
        db.close()
