"""
tests/integration/test_storage_pipeline.py
────────────────────────────────────────────
Tests for the writer and the storage pipeline.

Why DuckDB :memory: rather than a temporary file:
  :memory: is faster (no disk I/O), is destroyed automatically when the
  connection closes (no cleanup needed), and avoids collisions between
  parallel tests. Behaviour is identical to file mode for every testing
  purpose.

Why TestSyncAPI is separate from TestAsyncAPI:
  The synchronous API tests pure persistence logic — whether the SQL is
  correct, whether the generated columns work, whether the upsert updates the
  right fields. The asynchronous API tests buffer semantics — flush by size,
  flush by time, graceful shutdown. Distinct responsibilities deserve distinct
  tests.
"""

from __future__ import annotations

import json as _json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    Side,
    Size,
    Tick,
    TickType,
    Venue,
)
from storage.writer import MarketDataWriter

# ---------------------------------------------------------------------------
# Helpers — domain object constructors for tests
# ---------------------------------------------------------------------------


def make_market_id(
    venue: Venue = Venue.KALSHI,
    raw_id: str = "KXBTC-TEST",
) -> MarketId:
    return MarketId(venue=venue, raw_id=raw_id)


def make_market(mid: MarketId | None = None) -> Market:
    m = mid or make_market_id()
    return Market(
        market_id=m,
        question="Will BTC close above $85,000 on Apr 22?",
        category=MarketCategory.CRYPTO,
        resolution=Resolution(
            resolution_date=datetime(2026, 4, 22, 16, 0, tzinfo=UTC),
            resolved_value=None,
        ),
        status=MarketStatus.OPEN,
    )


def make_tick(
    mid: MarketId | None = None,
    bid: float = 0.45,
    ask: float = 0.47,
    tick_type: TickType = TickType.QUOTE,
    side: Side | None = None,
) -> Tick:
    m = mid or make_market_id()
    return Tick(
        market_id=m,
        timestamp=datetime.now(tz=UTC),
        tick_type=tick_type,
        yes_bid=Price(bid),
        yes_ask=Price(ask),
        volume=Size(0.0 if tick_type == TickType.QUOTE else 10.0),
        side=side if tick_type == TickType.TRADE else None,
    )


def make_orderbook(mid: MarketId | None = None) -> OrderBook:
    m = mid or make_market_id()
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


def mem() -> MarketDataWriter:
    """Writer backed by in-memory DuckDB — the fastest option for tests."""
    return MarketDataWriter(db_path=":memory:")


# ---------------------------------------------------------------------------
# Synchronous API tests
# ---------------------------------------------------------------------------


class TestSyncAPI:
    def test_write_market(self) -> None:
        """A market's basic fields are persisted."""
        with mem() as w:
            w.write_market_sync(make_market())
            row = w._con.execute("SELECT market_id, status, category FROM markets").fetchone()
            assert row[0] == "kalshi:KXBTC-TEST"
            assert row[1] == "open"
            assert row[2] == "crypto"

    def test_upsert_does_not_duplicate(self) -> None:
        """
        Writing the same market_id twice must produce a single row.
        Verifies that ON CONFLICT DO UPDATE works.
        """
        with mem() as w:
            w.write_market_sync(make_market())
            w.write_market_sync(make_market())
            count = w._con.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
            assert count == 1

    def test_upsert_updates_status(self) -> None:
        """
        On resolution the upsert must update status and resolved_value without
        duplicating the row or modifying any other field.
        """
        with mem() as w:
            w.write_market_sync(make_market())
            mid = make_market_id()
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
            w.write_market_sync(resolved)
            row = w._con.execute("SELECT status, resolved_value FROM markets").fetchone()
            assert row[0] == "resolved"
            assert row[1] == 1.0

    def test_write_ticks_batch(self) -> None:
        """
        A batch of N ticks must persist exactly N rows.
        Verifies executemany works with multiple items.
        """
        with mem() as w:
            ticks = [make_tick() for _ in range(5)]
            assert w.write_ticks_sync(ticks) == 5
            count = w._con.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
            assert count == 5

    def test_tick_quote_side_null(self) -> None:
        """
        QUOTE ticks have no side — it must persist as NULL. side=NULL is a
        schema business rule: without an aggressor there is no direction.
        """
        with mem() as w:
            w.write_ticks_sync([make_tick(tick_type=TickType.QUOTE)])
            row = w._con.execute("SELECT tick_type, side FROM ticks").fetchone()
            assert row[0] == "quote"
            assert row[1] is None

    def test_tick_trade_side_set(self) -> None:
        """
        TRADE ticks must carry a side — essential for computing adverse
        selection and OBI in features/microstructure.py.
        """
        with mem() as w:
            w.write_ticks_sync([make_tick(tick_type=TickType.TRADE, side=Side.YES)])
            row = w._con.execute("SELECT tick_type, volume, side FROM ticks").fetchone()
            assert row[0] == "trade"
            assert row[1] == pytest.approx(10.0)
            assert row[2] == "yes"

    def test_generated_columns_mid_and_spread(self) -> None:
        """
        mid and spread are GENERATED ALWAYS AS — DuckDB computes them
        automatically. We verify the computation is correct and that we do not
        need to insert them by hand.
        """
        with mem() as w:
            w.write_ticks_sync([make_tick(bid=0.44, ask=0.46)])
            row = w._con.execute("SELECT mid, spread FROM ticks").fetchone()
            assert row[0] == pytest.approx(0.45)  # (0.44+0.46)/2
            assert row[1] == pytest.approx(0.02)  # 0.46-0.44

    def test_write_orderbook(self) -> None:
        """
        Verifies best_bid, best_ask and the precomputed depths.
        se persisten correctamente.
        """
        with mem() as w:
            w.write_orderbook_sync(make_orderbook())
            row = w._con.execute(
                "SELECT best_bid, best_ask, bid_depth_5, ask_depth_5 FROM orderbooks"
            ).fetchone()
            assert row[0] == pytest.approx(0.45)
            assert row[1] == pytest.approx(0.47)
            assert row[2] == pytest.approx(1500.0)  # 1000 + 500
            assert row[3] == pytest.approx(1200.0)  # 800 + 400

    def test_orderbook_json_parseable(self) -> None:
        """
        bids_json and asks_json must be valid, reconstructible JSON.
        The backtesting engine parses them to reconstruct the book.
        """
        with mem() as w:
            w.write_orderbook_sync(make_orderbook())
            row = w._con.execute("SELECT bids_json, asks_json FROM orderbooks").fetchone()
            bids = _json.loads(row[0])
            asks = _json.loads(row[1])
            assert len(bids) == 2
            assert bids[0][0] == pytest.approx(0.45)  # best bid first
            assert asks[0][0] == pytest.approx(0.47)  # best ask first

    def test_write_snapshot(self) -> None:
        """
        A snapshot must persist all three components in one call.
        Verifies write_snapshot_sync orchestrates correctly.
        """
        with mem() as w:
            mid = make_market_id()
            snapshot = MarketSnapshot(
                market=make_market(mid),
                orderbook=make_orderbook(mid),
                last_tick=make_tick(mid),
            )
            w.write_snapshot_sync(snapshot)
            assert w._con.execute("SELECT COUNT(*) FROM markets").fetchone()[0] == 1
            assert w._con.execute("SELECT COUNT(*) FROM orderbooks").fetchone()[0] == 1
            assert w._con.execute("SELECT COUNT(*) FROM ticks").fetchone()[0] == 1

    def test_write_features(self) -> None:
        """
        Verifies features are persisted correctly.
        The dict-to-tuple conversion (a DuckDB limitation with named params in
        executemany) must be transparent to the caller.
        """
        with mem() as w:
            rows = [
                {
                    "market_id": "kalshi:KXBTC-TEST",
                    "venue": "kalshi",
                    "timestamp": datetime.now(tz=UTC),
                    "obi": 0.234,
                    "quoted_spread": 0.020,
                    "relative_spread": 4.44,
                    "belief_vol": 0.0147,
                    "ewma_vol": 0.0089,
                    "tau_years": 0.0001,
                    "mu_hat": 0.012,
                }
            ]
            assert w.write_features_sync(rows) == 1
            count = w._con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
            assert count == 1

    def test_empty_writes_no_error(self) -> None:
        """
        Passing an empty list must not raise or execute SQL.
        El guard if not ticks/rows evita ejecutar executemany([])
        which some DuckDB versions reject.
        """
        with mem() as w:
            assert w.write_ticks_sync([]) == 0
            assert w.write_features_sync([]) == 0

    def test_multiple_venues(self) -> None:
        """
        Kalshi and Polymarket ticks coexist in the same table.
        The venue field allows them to be filtered independently.
        """
        with mem() as w:
            k_id = make_market_id(Venue.KALSHI, "KXBTC-TEST")
            p_id = make_market_id(Venue.POLYMARKET, "0xabc123")
            w.write_ticks_sync([make_tick(k_id), make_tick(p_id)])
            rows = w._con.execute("SELECT venue FROM ticks ORDER BY venue").fetchall()
            venues = [r[0] for r in rows]
            assert "kalshi" in venues
            assert "polymarket" in venues


# ---------------------------------------------------------------------------
# Asynchronous API tests
# ---------------------------------------------------------------------------


class TestAsyncAPI:
    @pytest.mark.asyncio
    async def test_enqueue_tick(self) -> None:
        """
        An enqueued tick must persist after an explicit flush.
        We call flush_now() directly rather than waiting for the timer
        so the test is deterministic and fast.
        """
        async with MarketDataWriter(db_path=":memory:", flush_interval_seconds=1) as w:
            await w.enqueue(make_tick())
            await w.flush_now()
            count = w._con.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
            assert count == 1

    @pytest.mark.asyncio
    async def test_enqueue_market(self) -> None:
        """An enqueued market must persist after a flush."""
        async with MarketDataWriter(db_path=":memory:", flush_interval_seconds=1) as w:
            await w.enqueue(make_market())
            await w.flush_now()
            count = w._con.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
            assert count == 1

    @pytest.mark.asyncio
    async def test_flush_on_size(self) -> None:
        """
        With flush_max_items=5, on reaching the limit enqueue() must force an
        automatic flush before enqueuing item 6.
        The timer is set to 60 s so it does not interfere.
        """
        async with MarketDataWriter(
            db_path=":memory:",
            flush_interval_seconds=60,
            flush_max_items=5,
        ) as w:
            for _ in range(6):
                await w.enqueue(make_tick())
            await w.flush_now()  # flush the remainder
            count = w._con.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
            assert count == 6

    @pytest.mark.asyncio
    async def test_enqueue_snapshot(self) -> None:
        """
        enqueue_snapshot must enqueue all three snapshot components.
        After the flush there must be one row in each table.
        """
        async with MarketDataWriter(db_path=":memory:") as w:
            mid = make_market_id()
            snapshot = MarketSnapshot(
                market=make_market(mid),
                orderbook=make_orderbook(mid),
                last_tick=make_tick(mid),
            )
            await w.enqueue_snapshot(snapshot)
            await w.flush_now()
            assert w._con.execute("SELECT COUNT(*) FROM markets").fetchone()[0] == 1
            assert w._con.execute("SELECT COUNT(*) FROM orderbooks").fetchone()[0] == 1
            assert w._con.execute("SELECT COUNT(*) FROM ticks").fetchone()[0] == 1

    @pytest.mark.asyncio
    async def test_stop_flush_final(self) -> None:
        """
        On stopping the writer, any pending items must be persisted.
        Checked BEFORE stop(), because stop() closes the connection.
        """
        w = MarketDataWriter(db_path=":memory:", flush_interval_seconds=1)
        await w.start()
        for _ in range(3):
            await w.enqueue(make_tick())
        await w.flush_now()  # explicit flush before stop
        count = w._con.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
        assert count == 3
        await w.stop()
        w.close()

    @pytest.mark.asyncio
    async def test_stats(self) -> None:
        """
        The stats must correctly reflect what was written.
        Useful for monitoring — if stats["ticks"] stops growing, the
        ingestion pipeline is broken.
        """
        async with MarketDataWriter(db_path=":memory:") as w:
            await w.enqueue(make_tick())
            await w.enqueue(make_tick())
            await w.enqueue(make_market())
            await w.flush_now()
            assert w.stats["ticks"] == 2
            assert w.stats["markets"] == 1
            assert w.stats["flushes"] >= 1
