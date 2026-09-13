"""
tests/unit/test_reader.py
Tests for MarketDataReader.
Uses the synchronous writer to populate an in-memory DuckDB, then verifies
the reader reads back exactly what the writer wrote.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from normalizer.schema import (
    Market,
    MarketCategory,
    MarketId,
    MarketStatus,
    OrderBook,
    OrderBookLevel,
    Price,
    Resolution,
    Side,
    Size,
    Tick,
    TickType,
    Venue,
)
from storage.reader import MarketDataReader
from storage.writer import MarketDataWriter

# ---------------------------------------------------------------------------
# Fixture: an in-memory database shared between writer and reader
# ---------------------------------------------------------------------------


class DB:
    """
    Writer and reader pointing at the same :memory: database.
    DuckDB in :memory: mode cannot be shared across separate connections,
    so we use the writer's internal connection directly.
    """

    def __init__(self) -> None:
        self.w = MarketDataWriter(db_path=":memory:")
        # The reader shares the writer's internal connection
        self.r = MarketDataReader.__new__(MarketDataReader)
        self.r._con = self.w._con

    def close(self) -> None:
        self.w.close()


def make_db() -> DB:
    return DB()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mid(venue: Venue = Venue.KALSHI, raw_id: str = "KXBTC-TEST") -> MarketId:
    return MarketId(venue=venue, raw_id=raw_id)


def make_market(mid: MarketId | None = None, status: MarketStatus = MarketStatus.OPEN) -> Market:
    m = mid or make_mid()
    return Market(
        market_id=m,
        question="Will BTC close above $85k?",
        category=MarketCategory.CRYPTO,
        resolution=Resolution(
            resolution_date=datetime(2026, 4, 22, 16, 0, tzinfo=UTC),
            resolved_value=None,
        ),
        status=status,
    )


def make_tick(
    mid: MarketId | None = None,
    bid: float = 0.45,
    ask: float = 0.47,
    ts: datetime | None = None,
    tick_type: TickType = TickType.QUOTE,
    side: Side | None = None,
) -> Tick:
    m = mid or make_mid()
    return Tick(
        market_id=m,
        timestamp=ts or datetime.now(tz=UTC),
        tick_type=tick_type,
        yes_bid=Price(bid),
        yes_ask=Price(ask),
        volume=Size(0.0 if tick_type == TickType.QUOTE else 10.0),
        side=side if tick_type == TickType.TRADE else None,
    )


def make_ob(mid: MarketId | None = None) -> OrderBook:
    m = mid or make_mid()
    return OrderBook(
        market_id=m,
        timestamp=datetime.now(tz=UTC),
        bids=(
            OrderBookLevel(Price(0.45), Size(1000.0)),
            OrderBookLevel(Price(0.44), Size(500.0)),
        ),
        asks=(
            OrderBookLevel(Price(0.47), Size(800.0)),
            OrderBookLevel(Price(0.48), Size(400.0)),
        ),
    )


# ---------------------------------------------------------------------------
# Operational-section tests
# ---------------------------------------------------------------------------


class TestOperacional:
    def test_latest_ticks_returns_n(self) -> None:
        db = make_db()
        ticks = [make_tick(bid=0.44, ask=0.46) for i in range(10)]
        db.w.write_ticks_sync(ticks)
        df = db.r.latest_ticks("kalshi:KXBTC-TEST", n=5)
        assert len(df) == 5
        db.close()

    def test_latest_ticks_ordered_desc(self) -> None:
        """The most recent ticks come first."""
        db = make_db()
        base = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)
        ticks = [make_tick(ts=base + timedelta(seconds=i)) for i in range(5)]
        db.w.write_ticks_sync(ticks)
        df = db.r.latest_ticks("kalshi:KXBTC-TEST", n=5)
        timestamps = df["timestamp"].tolist()
        assert timestamps == sorted(timestamps, reverse=True)
        db.close()

    def test_latest_ticks_market_id_filtrado(self) -> None:
        """Only ticks for the requested market_id are returned."""
        db = make_db()
        k_id = make_mid(Venue.KALSHI, "KXBTC-TEST")
        p_id = make_mid(Venue.POLYMARKET, "0xabc123")
        db.w.write_ticks_sync([make_tick(k_id), make_tick(p_id)])
        df = db.r.latest_ticks("kalshi:KXBTC-TEST")
        assert len(df) == 1
        db.close()

    def test_latest_orderbook(self) -> None:
        db = make_db()
        db.w.write_orderbook_sync(make_ob())
        df = db.r.latest_orderbook("kalshi:KXBTC-TEST")
        assert len(df) == 1
        assert df["best_bid"].iloc[0] == pytest.approx(0.45)
        assert df["best_ask"].iloc[0] == pytest.approx(0.47)
        db.close()

    def test_latest_orderbook_empty_when_absent(self) -> None:
        db = make_db()
        df = db.r.latest_orderbook("kalshi:NO-EXISTE")
        assert len(df) == 0
        db.close()

    def test_latest_features(self) -> None:
        db = make_db()
        rows = [
            {
                "market_id": "kalshi:KXBTC-TEST",
                "venue": "kalshi",
                "timestamp": datetime.now(tz=UTC),
                "obi": 0.3,
                "quoted_spread": 0.02,
                "relative_spread": 4.4,
                "belief_vol": 0.015,
                "ewma_vol": 0.009,
                "tau_years": 0.001,
                "mu_hat": 0.01,
            }
        ]
        db.w.write_features_sync(rows)
        df = db.r.latest_features("kalshi:KXBTC-TEST")
        assert len(df) == 1
        assert df["obi"].iloc[0] == pytest.approx(0.3)
        db.close()

    def test_market_by_id(self) -> None:
        db = make_db()
        db.w.write_market_sync(make_market())
        df = db.r.market("kalshi:KXBTC-TEST")
        assert len(df) == 1
        assert df["category"].iloc[0] == "crypto"
        db.close()

    def test_open_markets(self) -> None:
        db = make_db()
        k_id = make_mid(Venue.KALSHI, "KXBTC-TEST")
        p_id = make_mid(Venue.POLYMARKET, "0xabc123")
        db.w.write_market_sync(make_market(k_id, MarketStatus.OPEN))
        db.w.write_market_sync(make_market(p_id, MarketStatus.OPEN))
        df = db.r.open_markets()
        assert len(df) == 2
        db.close()

    def test_open_markets_filtro_venue(self) -> None:
        db = make_db()
        k_id = make_mid(Venue.KALSHI, "KXBTC-TEST")
        p_id = make_mid(Venue.POLYMARKET, "0xabc123")
        db.w.write_market_sync(make_market(k_id))
        db.w.write_market_sync(make_market(p_id))
        df = db.r.open_markets(venue="kalshi")
        assert len(df) == 1
        assert df["venue"].iloc[0] == "kalshi"
        db.close()

    def test_mid_price_now(self) -> None:
        db = make_db()
        db.w.write_ticks_sync([make_tick(bid=0.44, ask=0.46)])
        mid = db.r.mid_price_now("kalshi:KXBTC-TEST")
        assert mid == pytest.approx(0.45)
        db.close()

    def test_mid_price_now_none_when_empty(self) -> None:
        db = make_db()
        assert db.r.mid_price_now("kalshi:NO-EXISTE") is None
        db.close()


# ---------------------------------------------------------------------------
# Analytical-section tests
# ---------------------------------------------------------------------------


class TestAnalitica:
    def test_ticks_time_filter(self) -> None:
        db = make_db()
        base = datetime(2026, 4, 22, 10, 0, tzinfo=UTC)
        ticks = [make_tick(ts=base + timedelta(hours=i)) for i in range(5)]
        db.w.write_ticks_sync(ticks)

        start = base + timedelta(hours=1)
        end = base + timedelta(hours=3)
        df = db.r.ticks("kalshi:KXBTC-TEST", start=start, end=end)
        assert len(df) == 3
        db.close()

    def test_ticks_type_filter(self) -> None:
        db = make_db()
        db.w.write_ticks_sync(
            [
                make_tick(tick_type=TickType.QUOTE),
                make_tick(tick_type=TickType.TRADE, side=Side.YES),
            ]
        )
        df = db.r.ticks("kalshi:KXBTC-TEST", tick_type="trade")
        assert len(df) == 1
        assert df["tick_type"].iloc[0] == "trade"
        db.close()

    def test_trade_ticks(self) -> None:
        db = make_db()
        db.w.write_ticks_sync(
            [
                make_tick(tick_type=TickType.QUOTE),
                make_tick(tick_type=TickType.TRADE, side=Side.YES),
                make_tick(tick_type=TickType.TRADE, side=Side.NO),
            ]
        )
        df = db.r.trade_ticks("kalshi:KXBTC-TEST")
        assert len(df) == 2
        db.close()

    def test_orderbooks_serie(self) -> None:
        db = make_db()
        for _ in range(3):
            db.w.write_orderbook_sync(make_ob())
        df = db.r.orderbooks("kalshi:KXBTC-TEST")
        assert len(df) == 3
        assert "best_bid" in df.columns
        db.close()

    def test_markets_filtros(self) -> None:
        db = make_db()
        db.w.write_market_sync(make_market())
        df_all = db.r.markets()
        df_kalshi = db.r.markets(venue="kalshi")
        df_open = db.r.markets(status="open")
        df_crypto = db.r.markets(category="crypto")
        df_politics = db.r.markets(category="politics")
        assert len(df_all) == 1
        assert len(df_kalshi) == 1
        assert len(df_open) == 1
        assert len(df_crypto) == 1
        assert len(df_politics) == 0
        db.close()

    def test_daily_volume(self) -> None:
        db = make_db()
        base = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)
        # 3 trades on the same day
        trades = [
            make_tick(ts=base + timedelta(hours=i), tick_type=TickType.TRADE, side=Side.YES)
            for i in range(3)
        ]
        db.w.write_ticks_sync(trades)
        df = db.r.daily_volume("kalshi:KXBTC-TEST")
        assert len(df) == 1
        assert df["n_trades"].iloc[0] == 3
        db.close()

    def test_resolved_markets(self) -> None:
        db = make_db()
        mid = make_mid()
        # Mercado abierto
        db.w.write_market_sync(make_market(mid))
        # Luego resuelto
        resolved = Market(
            market_id=mid,
            question="test",
            category=MarketCategory.CRYPTO,
            resolution=Resolution(
                resolution_date=datetime(2026, 4, 22, 16, 0, tzinfo=UTC),
                resolved_value=1.0,
            ),
            status=MarketStatus.RESOLVED,
        )
        db.w.write_market_sync(resolved)
        df = db.r.resolved_markets_with_outcome()
        assert len(df) == 1
        assert df["resolved_value"].iloc[0] == 1.0
        db.close()


# ---------------------------------------------------------------------------
# Backtesting-section tests
# ---------------------------------------------------------------------------


class TestBacktesting:
    def test_ticks_chunked_yields_every_tick(self) -> None:
        """The generator yields every tick; none is lost."""
        db = make_db()
        ticks = [make_tick(bid=0.40 + i * 0.001) for i in range(25)]
        db.w.write_ticks_sync(ticks)

        total = sum(len(chunk) for chunk in db.r.ticks_chunked("kalshi:KXBTC-TEST", chunk_size=10))
        assert total == 25
        db.close()

    def test_ticks_chunked_correct_size(self) -> None:
        """Every chunk holds exactly chunk_size rows, except the last."""
        db = make_db()
        ticks = [make_tick() for _ in range(25)]
        db.w.write_ticks_sync(ticks)

        chunks = list(db.r.ticks_chunked("kalshi:KXBTC-TEST", chunk_size=10))
        assert len(chunks) == 3  # 10 + 10 + 5
        assert len(chunks[0]) == 10
        assert len(chunks[1]) == 10
        assert len(chunks[2]) == 5
        db.close()

    def test_ticks_chunked_ordered_asc(self) -> None:
        """The generator yields ticks in ascending chronological order."""
        db = make_db()
        base = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)
        ticks = [make_tick(ts=base + timedelta(seconds=i)) for i in range(10)]
        db.w.write_ticks_sync(ticks)

        all_ts = []
        for chunk in db.r.ticks_chunked("kalshi:KXBTC-TEST", chunk_size=5):
            all_ts.extend(chunk["timestamp"].tolist())

        assert all_ts == sorted(all_ts)
        db.close()

    def test_ticks_chunked_time_filter(self) -> None:
        db = make_db()
        base = datetime(2026, 4, 22, 10, 0, tzinfo=UTC)
        ticks = [make_tick(ts=base + timedelta(hours=i)) for i in range(6)]
        db.w.write_ticks_sync(ticks)

        start = base + timedelta(hours=2)
        total = sum(
            len(chunk)
            for chunk in db.r.ticks_chunked("kalshi:KXBTC-TEST", start=start, chunk_size=100)
        )
        assert total == 4
        db.close()

    def test_ticks_chunked_empty(self) -> None:
        """With no ticks the generator produces no chunk at all."""
        db = make_db()
        chunks = list(db.r.ticks_chunked("kalshi:NO-EXISTE"))
        assert chunks == []
        db.close()

    def test_count_ticks(self) -> None:
        db = make_db()
        ticks = [make_tick() for _ in range(7)]
        db.w.write_ticks_sync(ticks)
        assert db.r.count_ticks("kalshi:KXBTC-TEST") == 7
        db.close()

    def test_count_ticks_time_filter(self) -> None:
        db = make_db()
        base = datetime(2026, 4, 22, 10, 0, tzinfo=UTC)
        ticks = [make_tick(ts=base + timedelta(hours=i)) for i in range(5)]
        db.w.write_ticks_sync(ticks)

        start = base + timedelta(hours=2)
        assert db.r.count_ticks("kalshi:KXBTC-TEST", start=start) == 3
        db.close()

    def test_orderbooks_chunked(self) -> None:
        db = make_db()
        for _ in range(12):
            db.w.write_orderbook_sync(make_ob())

        chunks = list(db.r.orderbooks_chunked("kalshi:KXBTC-TEST", chunk_size=5))
        total = sum(len(c) for c in chunks)
        assert total == 12
        assert len(chunks) == 3  # 5 + 5 + 2
        db.close()
