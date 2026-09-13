"""
storage/writer.py
──────────────────
Infrastructure layer — persists domain objects into DuckDB through an
asynchronous buffer with periodic flushing.

Why this architecture:
  The connector produces ticks at a variable, unpredictable rate (WebSocket
  bursts). DuckDB is far more efficient with batch inserts than with
  individual ones. The buffer decouples the two rates: the connector enqueues
  in O(1) always, and the writer writes in batches when the buffer is full or
  enough time has passed.

Two flush conditions, whichever fires first:
  - Time: every FLUSH_INTERVAL_SECONDS seconds
  - Size: once the buffer holds FLUSH_MAX_ITEMS items
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from normalizer.schema import (
    Market,
    MarketSnapshot,
    OrderBook,
    Tick,
)

log = logging.getLogger(__name__)


FLUSH_INTERVAL_SECONDS: int = 5
FLUSH_MAX_ITEMS: int = 1_000

# Maximum time stop() waits for the flush loop before cancelling it.
# Deliberately bounded: stop() is invoked from __aexit__ and from the signal
# handler, and a stuck loop must not block shutdown or hang the test suite.
STOP_TIMEOUT_SECONDS: float = 10.0


_QueueItem = Tick | OrderBook | Market


class MarketDataWriter:
    """
    Asynchronous writer with an in-memory buffer for DuckDB.

    Two usage modes:

    PRODUCTION — async context manager:
        async with MarketDataWriter() as writer:
            await writer.enqueue(tick)
            await writer.enqueue(orderbook)
        # On exit it performs a final flush and closes the connection

    TESTS / SCRIPTS — synchronous, direct:
        writer = MarketDataWriter(db_path=":memory:")
        writer.write_ticks_sync([tick1, tick2])
        writer.close()
    """

    def __init__(
        self,
        db_path: str | None = None,
        flush_interval_seconds: int = FLUSH_INTERVAL_SECONDS,
        flush_max_items: int = FLUSH_MAX_ITEMS,
        max_db_size_mb: float = 0.0,
        size_check_every: int = 20,
    ) -> None:
        self._db_path: str = (
            db_path if db_path else os.getenv("DUCKDB_PATH", "./data/duckdb/markets.duckdb")
        )

        # Create the directory if it does not exist. ":memory:" is the test
        # mode and needs none.
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._con = duckdb.connect(self._db_path)
        self._init_schema()

        self._flush_interval = flush_interval_seconds
        self._flush_max_items = flush_max_items

        # Size cap. Checked every db_size_check_every flushes rather than on
        # each one: a stat() per batch is cheap but unnecessary, and 20 flushes
        # is seconds of latency to react.
        self._max_bytes = int(max_db_size_mb * 1024 * 1024) if max_db_size_mb > 0 else 0
        self._size_check_every = max(1, size_check_every)
        self._size_exceeded = False

        self._queue: asyncio.Queue[_QueueItem] = asyncio.Queue()

        # Stop signal. An asyncio.Event and NOT a poison pill in the queue:
        # with the pill, flush_now() consumed it while draining, re-queued it
        # and broke out, leaving items behind; the loop then re-consumed those
        # items and never reached the None, so stop() waited forever — a
        # livelock at 100% CPU, reproducible with any exception raised inside
        # the `async with` block.
        self._stop_event: asyncio.Event = asyncio.Event()

        self._flush_task: asyncio.Task[None] | None = None

        # Cumulative statistics for logging and monitoring.
        self._stats = {"ticks": 0, "orderbooks": 0, "markets": 0, "flushes": 0}

    def _init_schema(self) -> None:
        """
        Apply every pending migration, in order and exactly once.

        Why a runner was needed rather than just executing 001:
          Previously only `001_initial_schema.sql` ran on each startup, and
          "already exists" errors were swallowed. That works as long as the
          schema never changes, but there is no way to apply a new migration
          to an existing database, nor to know which ones have run. Changing
          the schema in production was a manual operation.

        How it works:
          `schema_migrations` records the name of each applied file. On every
          startup only the unregistered ones run, ordered by name. That is the
          minimum needed to make the operation repeatable and auditable.

        Why statement by statement:
          DuckDB does not accept several statements with GENERATED ALWAYS AS
          columns in a single execute(), and the base schema uses them (mid,
          spread, date_).
        """
        migrations_dir = Path(__file__).parent / "migrations"
        if not migrations_dir.is_dir():
            log.warning("Migrations directory not found: %s", migrations_dir)
            return

        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name       VARCHAR     NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL
            )
            """
        )
        applied = {
            row[0] for row in self._con.execute("SELECT name FROM schema_migrations").fetchall()
        }

        for path in sorted(migrations_dir.glob("*.sql")):
            if path.name in applied:
                continue

            # Comments are stripped BEFORE splitting on ";". Splitting the raw
            # text breaks any statement whose comment happens to contain a
            # semicolon, and the resulting fragments only produce a warning —
            # a schema silently left half-created.
            sql = "\n".join(
                line for line in path.read_text().splitlines() if not line.lstrip().startswith("--")
            )
            for stmt in (s.strip() for s in sql.split(";")):
                if not stmt:
                    continue
                try:
                    self._con.execute(stmt)
                except duckdb.Error as e:
                    # "already exists" is expected on a pre-existing database
                    # that never recorded its migrations; anything else is reported.
                    if "already exists" not in str(e).lower():
                        log.warning("Migration %s: %s", path.name, e)

            self._con.execute(
                "INSERT INTO schema_migrations VALUES (?, ?)",
                [path.name, datetime.now(tz=UTC)],
            )
            log.info("Applied migration: %s", path.name)

    # ------------------------------------------------------------------
    # Asynchronous API — production
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Start the flush loop in the background as an asyncio Task.

        Why create_task and not a direct await:
          The flush loop must run concurrently with the connector producing
          ticks. create_task launches it in the background without blocking,
          so the connector can keep enqueuing while the loop writes to disk.
        """
        self._stop_event.clear()
        self._flush_task = asyncio.create_task(
            self._flush_loop(),
            name="writer-flush-loop",
        )
        log.info(
            "MarketDataWriter started (flush every %ds or %d items)",
            self._flush_interval,
            self._flush_max_items,
        )

    async def stop(self) -> None:
        """
        Stop the flush loop cleanly (graceful shutdown):
          1. Set _stop_event — the stop signal for the loop
          2. Wait for the loop to finish, with a bounded timeout
          3. Perform a final synchronous flush of whatever remains buffered
          4. Close the DuckDB connection

        Why an Event rather than a poison pill in the queue:
          The pill travelled down the same channel as the data, so flush_now()
          could consume it while draining. Re-queuing it and cutting the drain
          short left items behind, the loop re-consumed them, and the signal
          never reached its destination: stop() waited indefinitely. An Event
          lives outside the queue and is idempotent.

        Why the timeout:
          stop() is called from __aexit__ and from the signal handler. A stuck
          loop (DuckDB locked, say) must neither block shutdown nor hang the
          test suite. If the timeout expires we cancel it and persist whatever
          is left synchronously.
        """
        self._stop_event.set()

        if self._flush_task and not self._flush_task.done():
            try:
                await asyncio.wait_for(self._flush_task, timeout=STOP_TIMEOUT_SECONDS)
            except TimeoutError:
                log.error(
                    "Flush loop did not stop within %.1fs — cancelling",
                    STOP_TIMEOUT_SECONDS,
                )
                self._flush_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._flush_task
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - the loop already logs the detail
                log.error("Flush loop raised on shutdown: %s", e)

        self._flush_remaining()
        self._con.close()
        log.info("MarketDataWriter stopped — stats: %s", self._stats)

    async def enqueue(self, item: _QueueItem) -> None:
        """
        Enqueue an item for asynchronous writing. Always O(1) — never blocks.

        Why force a flush when the buffer is full:
          Without this control, a burst of activity could grow the queue
          without bound and consume all available memory. On detecting
          flush_max_items waiting, we flush immediately before enqueuing the
          new item. That adds a little latency at that instant but keeps
          memory usage bounded.
        """
        if self._queue.qsize() >= self._flush_max_items:
            await self.flush_now()

        await self._queue.put(item)

    async def enqueue_snapshot(self, snapshot: MarketSnapshot) -> None:
        """
        Enqueue every component of a MarketSnapshot at once.

        Why this convenience method:
          The connector fetches complete snapshots (market + orderbook + tick).
          Without this method the connector would make three enqueue() calls.
          This guarantees the three components are always enqueued in the same
          order, with no risk of forgetting one.
        """
        await self.enqueue(snapshot.market)
        if snapshot.orderbook is not None:
            await self.enqueue(snapshot.orderbook)
        if snapshot.last_tick is not None:
            await self.enqueue(snapshot.last_tick)

    async def _flush_loop(self) -> None:
        """
        Background coroutine that runs until _stop_event is set.

        Timer logic:
          We wait on _stop_event with a flush_interval_seconds timeout. If the
          timeout expires → periodic flush. If the event fires → one final
          flush and exit. Either way exactly one flush happens per iteration,
          so the last window is never lost.

        Why not wait on the queue:
          `asyncio.wait_for(queue.get(), timeout)` can cancel the getter AFTER
          it has received an item, losing it. Waiting on an Event removes that
          race: items are only read in flush_now(), which drains with
          get_nowait() and cannot lose anything.

        Why catching a bare Exception:
          A DuckDB write error (disk full, corruption) must not kill the loop.
          We log it as an error and carry on. If the loop died, every
          subsequent datum would be lost silently.
        """
        while True:
            stopping = False
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._flush_interval,
                )
                stopping = True
            except TimeoutError:
                pass  # timer expired — periodic flush

            try:
                await self.flush_now()
            except Exception as e:
                log.error("Error in flush loop: %s", e, exc_info=True)

            if stopping:
                log.debug("Flush loop received stop signal")
                return

    async def flush_now(self) -> None:
        """
        Drain the queue and write every item into DuckDB as a batch.

        Why drain the whole queue before writing:
          We want a single executemany per type per flush, not one execute per
          item. Items are first classified by type, then inserted one table at
          a time. This is 10-50x faster than individual inserts.

        Why get_nowait instead of get:
          get() blocks when the queue is empty. get_nowait() raises QueueEmpty,
          which we catch to know the queue has been drained. It is the correct
          way to drain an asyncio queue without blocking.
        """
        ticks: list[Tick] = []
        orderbooks: list[OrderBook] = []
        markets: list[Market] = []

        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if isinstance(item, Tick):
                    ticks.append(item)
                elif isinstance(item, OrderBook):
                    orderbooks.append(item)
                elif isinstance(item, Market):
                    markets.append(item)
            except asyncio.QueueEmpty:
                break

        if self._size_limit_reached():
            # The batch is dropped on purpose: writing on would fill the disk.
            # Logged once so it does not flood.
            return

        # Batch insert per table — only when there is data
        if ticks:
            self._write_ticks_batch(ticks)
            self._stats["ticks"] += len(ticks)

        if orderbooks:
            self._write_orderbooks_batch(orderbooks)
            self._stats["orderbooks"] += len(orderbooks)

        if markets:
            self._write_markets_batch(markets)
            self._stats["markets"] += len(markets)

        if ticks or orderbooks or markets:
            self._stats["flushes"] += 1
            log.debug(
                "Flush: %d ticks, %d orderbooks, %d markets",
                len(ticks),
                len(orderbooks),
                len(markets),
            )

    def _size_limit_reached(self) -> bool:
        """
        True once the database has exceeded max_db_size_mb.

        Why it exists:
          Without a cap, unattended ingestion grows until the disk is full and
          you find out when something else breaks. `ticks` grows with every
          WebSocket message; in production that is millions of rows per day.

        Behaviour once exceeded: writes stop being accepted and the condition
        is logged ONCE. Nothing is deleted — deciding what to discard belongs
        to the archiver (storage/archiver.py), not the writer.
        """
        if self._max_bytes <= 0 or self._db_path == ":memory:":
            return self._size_exceeded

        self._stats["flushes"]
        if self._stats["flushes"] % self._size_check_every != 0 and not self._size_exceeded:
            return False

        try:
            size = Path(self._db_path).stat().st_size
        except OSError:
            return False

        if size >= self._max_bytes and not self._size_exceeded:
            self._size_exceeded = True
            log.error(
                "db_size_limit_reached: %.1f MB >= %.1f MB — writes stopped. "
                "Archive or raise MAX_DB_SIZE_MB to resume.",
                size / 1024 / 1024,
                self._max_bytes / 1024 / 1024,
            )
        return self._size_exceeded

    @property
    def size_limit_reached(self) -> bool:
        """So main.py can report it in the periodic statistics."""
        return self._size_exceeded

    def _flush_remaining(self) -> None:
        """
        Final synchronous flush — called from stop() after the async loop has
        been cancelled.

        Why this is needed in addition to the loop:
          Between the loop's last flush and the moment stop() calls this, more
          items may have arrived in the queue — or the loop may have been
          cancelled on timeout. This method persists them before closing the
          connection. It is the guarantee that no data is lost on shutdown.
        """
        ticks: list[Tick] = []
        orderbooks: list[OrderBook] = []
        markets: list[Market] = []

        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if isinstance(item, Tick):
                    ticks.append(item)
                elif isinstance(item, OrderBook):
                    orderbooks.append(item)
                elif isinstance(item, Market):
                    markets.append(item)
            except asyncio.QueueEmpty:
                break

        if ticks:
            self._write_ticks_batch(ticks)
        if orderbooks:
            self._write_orderbooks_batch(orderbooks)
        if markets:
            self._write_markets_batch(markets)

    def _write_ticks_batch(self, ticks: list[Tick]) -> None:
        """
        Batch-insert ticks with executemany.

        Why mid, spread and date_ are not included:
          They are GENERATED ALWAYS AS columns in the SQL schema. DuckDB
          computes them from yes_bid, yes_ask and timestamp automatically, and
          inserting them by hand raises an error. Computing them is the
          database's responsibility rather than the writer's — consistency by
          construction.

        Why ? rather than :named_params:
          DuckDB does not support named parameters (:param) in executemany,
          only positional ones (?). Named params work in a plain execute(),
          not in a batch.
        """
        rows = [
            (
                str(t.market_id),
                t.market_id.venue.value,
                t.timestamp,
                t.tick_type.value,
                float(t.yes_bid),
                float(t.yes_ask),
                float(t.volume),
                t.side.value if t.side else None,
                t.source_id,
            )
            for t in ticks
        ]
        # ON CONFLICT DO NOTHING makes ingestion IDEMPOTENT: a writer retry, or
        # a connector re-poll returning already-seen trades, does not duplicate
        # rows. The conflict is detected by ux_ticks_market_source over
        # (market_id, source_id); quotes carry a NULL source_id, and in SQL two
        # NULLs do not collide, so they always pass through.
        self._con.executemany(
            """
            INSERT INTO ticks
                (market_id, venue, timestamp, tick_type,
                 yes_bid, yes_ask, volume, side, source_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            rows,
        )

    def _write_orderbooks_batch(self, orderbooks: list[OrderBook]) -> None:
        """
        Batch-insert order books.

        Why bids_json and asks_json are stored alongside best_bid/ask:
          best_bid and best_ask exist for fast analytical queries that need no
          JSON parsing, while the JSON columns preserve the full depth —
          needed to reconstruct the book in backtesting, or to compute
          liquidity metrics beyond the top level.

        Why bid_depth_5 and ask_depth_5 are precomputed:
          These are the features the feature store queries most. Computing
          them at insert time avoids parsing JSON on every read: a little disk
          traded for a lot of query time.
        """
        rows = [
            (
                str(ob.market_id),
                ob.market_id.venue.value,
                ob.timestamp,
                float(ob.best_bid) if ob.best_bid is not None else None,
                float(ob.best_ask) if ob.best_ask is not None else None,
                json.dumps([[float(lv.price), float(lv.size)] for lv in ob.bids]),
                json.dumps([[float(lv.price), float(lv.size)] for lv in ob.asks]),
                ob.bid_depth(5),
                ob.ask_depth(5),
            )
            for ob in orderbooks
        ]
        self._con.executemany(
            """
            INSERT INTO orderbooks
                (market_id, venue, timestamp,
                 best_bid, best_ask,
                 bids_json, asks_json,
                 bid_depth_5, ask_depth_5)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _write_markets_batch(self, markets: list[Market]) -> None:
        """
        Upsert markets with ON CONFLICT DO UPDATE.

        Why upsert rather than INSERT:
          The connector rediscovers markets periodically to detect status
          changes (OPEN → RESOLVED). Without an upsert we would accumulate
          duplicate rows. With one, an existing market_id only has the fields
          that can change updated (status, resolved_value, updated_at), while
          the static ones (question, category) are preserved.

        Why updated_at is set here rather than in the schema:
          DuckDB does not support DEFAULT NOW() with TIMESTAMPTZ consistently
          across versions. We assign it explicitly in Python to guarantee it
          is always UTC.
        """
        now = datetime.now(tz=UTC)
        rows = [
            (
                str(m.market_id),
                m.market_id.venue.value,
                m.market_id.raw_id,
                m.question,
                m.category.value,
                m.status.value,
                m.resolution.resolution_date,
                m.resolution.resolved_value,
                now,
                now,
            )
            for m in markets
        ]
        self._con.executemany(
            """
            INSERT INTO markets
                (market_id, venue, raw_id, question, category,
                 status, resolution_date, resolved_value,first_seen_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (market_id) DO UPDATE SET
                status         = excluded.status,
                resolved_value = excluded.resolved_value,
                updated_at     = excluded.updated_at
            """,
            rows,
        )

    def _write_features_batch(self, rows: list[dict[str, Any]]) -> None:
        """
        Batch-insert precomputed features.
        Called from features/store.py, not from the connector.

        Why dicts are converted to tuples before executemany:
          DuckDB does not support :named_params in executemany (only in a
          plain execute()). Each dict is converted to a tuple in the correct
          column order, explicitly.
        """
        tuples = [
            (
                r["market_id"],
                r["venue"],
                r["timestamp"],
                r["obi"],
                r["quoted_spread"],
                r["relative_spread"],
                r["belief_vol"],
                r["ewma_vol"],
                r["tau_years"],
                r["mu_hat"],
            )
            for r in rows
        ]
        self._con.executemany(
            """
            INSERT INTO features
                (market_id, venue, timestamp, obi, quoted_spread,
                 relative_spread, belief_vol, ewma_vol, tau_years, mu_hat)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuples,
        )

    # ------------------------------------------------------------------
    # Synchronous API — tests and scripts
    # Write directly, bypassing the queue.
    # ------------------------------------------------------------------

    def write_ticks_sync(self, ticks: Sequence[Tick]) -> int:
        """
        Write ticks synchronously. For tests and scripts.

        Why enqueue is not used in tests:
          enqueue requires a running event loop, and synchronous tests have
          none. This API allows the persistence logic to be tested without the
          complexity of the async machinery.
        """
        if not ticks:
            return 0
        self._write_ticks_batch(list(ticks))
        return len(ticks)

    def write_orderbook_sync(self, ob: OrderBook) -> None:
        """Write one order book synchronously."""
        self._write_orderbooks_batch([ob])

    def write_market_sync(self, market: Market) -> None:
        """Write one market synchronously (upsert)."""
        self._write_markets_batch([market])

    def write_features_sync(self, rows: list[dict[str, Any]]) -> int:
        """Write features synchronously."""
        if not rows:
            return 0
        self._write_features_batch(rows)
        return len(rows)

    def write_snapshot_sync(self, snapshot: MarketSnapshot) -> None:
        """
        Write a complete MarketSnapshot synchronously.
        Persists market, order book and tick in a single call.
        """
        self.write_market_sync(snapshot.market)
        if snapshot.orderbook is not None:
            self.write_orderbook_sync(snapshot.orderbook)
        if snapshot.last_tick is not None:
            self.write_ticks_sync([snapshot.last_tick])

    # ------------------------------------------------------------------
    # Context managers and housekeeping
    # ------------------------------------------------------------------

    async def __aenter__(self) -> MarketDataWriter:
        """Start the flush loop on entering the context manager."""
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Stop the flush loop and perform a final flush on exit."""
        await self.stop()

    def close(self) -> None:
        """Close the DuckDB connection synchronously. For tests."""
        self._con.close()

    def __enter__(self) -> MarketDataWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def stats(self) -> dict[str, int]:
        """
        Cumulative write statistics since startup.
        Useful for monitoring and periodic logging.
        """
        return dict(self._stats)
