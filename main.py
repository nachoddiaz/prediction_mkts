"""
main.py
────────
Entry point for the data ingestion system.

Orchestrates connectors → writer → feature store in a permanent async loop.

Why asyncio and not threads:
  The connectors are I/O-bound (WebSocket, HTTP). asyncio handles many
  concurrent connections on a single thread, without the overhead or the race
  conditions of Python threads.

Venues enabled by default:
  All three. None requires credentials to read; each can be disabled through
  an environment variable.

Usage:
    # Manifold only (sandbox)
    uv run python main.py

    # Kalshi + Manifold
    KALSHI_API_KEY=xxx KALSHI_PRIVATE_KEY_PATH=./secrets/kalshi.pem uv run python main.py

    # Every venue
    ENABLE_POLYMARKET=true uv run python main.py

    # Change the log level
    LOG_LEVEL=DEBUG uv run python main.py

    # Production (JSON logs)
    LOG_FORMAT=json uv run python main.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging — must be configured before any other project import
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """
    Configure structlog for readable development output or JSON in production.

    LOG_FORMAT=json  → structured JSON (for Datadog, Grafana Loki, and so on)
    LOG_FORMAT=text  → coloured, readable output (default, for development)
    LOG_LEVEL=DEBUG  → verbosity (default INFO)

    Why structlog rather than Python's standard logging:
      structlog allows structured context (venue="kalshi",
      market_id="KXBTC-...") to be attached to each log line without string
      concatenation. In production that context is parseable JSON — you can
      filter by venue or market_id in any log aggregator.
    """
    import logging

    import structlog

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "text").lower()

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level, logging.INFO),
    )

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if log_format == "json":
        # Production: one JSON object per line, parseable by any aggregator
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: coloured output with readable alignment
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level, logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


_configure_logging()

# ---------------------------------------------------------------------------
# Project imports — after logging is configured
# ---------------------------------------------------------------------------

import structlog  # noqa: E402

from config.settings import settings  # noqa: E402
from connectors.manifold import ManifoldConnector  # noqa: E402
from features.store import FeatureStore  # noqa: E402
from normalizer.schema import Market, MarketSnapshot, Tick  # noqa: E402
from storage.reader import MarketDataReader  # noqa: E402
from storage.writer import MarketDataWriter  # noqa: E402

log = structlog.get_logger(__name__)

_START_TIME = datetime.now(tz=UTC).timestamp()


# ---------------------------------------------------------------------------
# Callbacks — wire the connectors to the writer and the feature store
# ---------------------------------------------------------------------------


def make_callbacks(
    writer: MarketDataWriter,
    store: FeatureStore,
    markets_cache: dict[str, Market],
) -> tuple:
    """
    Build the callbacks that receive connector events.

    Why closures rather than methods:
      The connectors only know the (tick) -> None signature. They must know
      nothing about the writer or the store, and closures capture both without
      introducing coupling.
    """

    async def on_tick(tick: Tick) -> None:
        """Persist the tick and recompute features."""
        # No flush_now() here: the writer has its own loop that drains every
        # FLUSH_INTERVAL_SECONDS or on reaching FLUSH_MAX_ITEMS. Forcing a
        # flush per tick defeated that batching entirely — one executemany per
        # event — which is exactly what the writer documents as 10-50x slower.
        await writer.enqueue(tick)

        market_id = str(tick.market_id)
        market = markets_cache.get(market_id)
        if market:
            store.on_tick(tick, market)

        # One line per tick at INFO is unreadable under load: with three venues
        # active that is thousands of lines a minute, burying every warning.
        # The aggregate summary lives in _stats_loop(), which logs every 60 s.
        tau = round(market.resolution.tau, 4) if market else None
        log.debug(
            "tick_received",
            venue=tick.market_id.venue.value,
            market_id=market_id,
            mid=round(tick.mid, 4),
            spread=round(tick.spread, 4),
            tick_type=tick.tick_type.value,
            tau=tau,
        )

    async def on_snapshot(snapshot: MarketSnapshot) -> None:
        """Persist the full snapshot and recompute features."""
        await writer.enqueue_snapshot(snapshot)

        market_id = str(snapshot.market.market_id)
        markets_cache[market_id] = snapshot.market
        store.on_snapshot(snapshot)

        mid = None
        if snapshot.orderbook and snapshot.orderbook.mid is not None:
            mid = round(snapshot.orderbook.mid, 4)

        log.debug(
            "snapshot_received",
            venue=snapshot.market.market_id.venue.value,
            market_id=market_id,
            mid=mid,
            tau=round(snapshot.market.resolution.tau, 4),
        )

        log.info(
            "snapshot_received",
            venue=snapshot.market.market_id.venue.value,
            market_id=market_id,
            mid=mid,
            tau=round(snapshot.market.resolution.tau, 4),
        )

    return on_tick, on_snapshot


# ---------------------------------------------------------------------------
# Setup de venues
# ---------------------------------------------------------------------------


def build_connectors(
    on_tick: object,
    on_snapshot: object,
) -> list[tuple[str, object]]:
    """
    Build the enabled connectors from configuration.

    All three are enabled by default and disabled with ENABLE_<VENUE>=false in
    the .env. None needs credentials to READ: Kalshi serves markets, order
    books and trades unauthenticated from api.elections.kalshi.com, and
    Polymarket's /book endpoint is public too.
    """
    connectors = []

    # --- Manifold ---
    if settings.enable_manifold:
        connectors.append(
            (
                "manifold",
                ManifoldConnector(
                    on_tick=on_tick,
                    on_snapshot=on_snapshot,
                    poll_interval=settings.manifold_poll_interval,
                    backfill_max=settings.backfill_max_ticks,
                    refresh_interval=settings.market_refresh_seconds,
                    max_tau_days=settings.max_tau_days,
                    require_two_sided_book=settings.require_two_sided_book,
                ),
            )
        )
        log.info(
            "connector_enabled",
            venue="manifold",
            poll_interval=settings.manifold_poll_interval,
        )
    else:
        log.info("connector_disabled", venue="manifold", reason="ENABLE_MANIFOLD=false")

    # --- Kalshi ---
    # Enabled unless explicitly disabled. The public API
    # (api.elections.kalshi.com) serves markets, order books and trades
    # UNauthenticated, so requiring KALSHI_API_KEY to ingest excluded the
    # project's primary venue for nothing. The key is only needed for trading
    # endpoints, which are Phase 4.
    if settings.enable_kalshi:
        from connectors.kalshi import KalshiConnector

        connectors.append(
            (
                "kalshi",
                KalshiConnector(
                    on_tick=on_tick,
                    on_snapshot=on_snapshot,
                    api_key=settings.kalshi_api_key,
                    private_key_path=settings.kalshi_private_key_path,
                    env=settings.kalshi_env,
                    refresh_interval=settings.market_refresh_seconds,
                    max_tau_days=settings.max_tau_days,
                    require_two_sided_book=settings.require_two_sided_book,
                ),
            )
        )
        log.info(
            "connector_enabled",
            venue="kalshi",
            env=settings.kalshi_env,
            authenticated=bool(settings.kalshi_api_key),
        )
    else:
        log.info("connector_disabled", venue="kalshi", reason="ENABLE_KALSHI=false")

    # --- Polymarket ---
    if settings.enable_polymarket:
        from connectors.polymarket import PolymarketConnector

        connectors.append(
            (
                "polymarket",
                PolymarketConnector(
                    on_tick=on_tick,
                    on_snapshot=on_snapshot,
                    refresh_interval=settings.market_refresh_seconds,
                    max_tau_days=settings.max_tau_days,
                    require_two_sided_book=settings.require_two_sided_book,
                ),
            )
        )
        log.info("connector_enabled", venue="polymarket")
    else:
        log.info("connector_disabled", venue="polymarket", reason="ENABLE_POLYMARKET=false")

    return connectors


# ---------------------------------------------------------------------------
# Periodic statistics
# ---------------------------------------------------------------------------


def _collect_status(writer: MarketDataWriter, reader: MarketDataReader) -> dict[str, Any]:
    """
    Gather system state for the heartbeat.

    Why the process publishes this rather than an external script querying it:
      DuckDB grants an EXCLUSIVE lock to the writer. While ingestion runs, any
      other process — even in read_only mode — gets "Could not set lock on
      file", so an external monitor cannot look at the database. Here it can:
      the reader lives in the same process and shares the connection, so state
      is computed inside and published as JSON that anyone can read.
    """
    stats = writer.stats
    uptime_min = (datetime.now(tz=UTC).timestamp() - _START_TIME) / 60

    status: dict[str, Any] = {
        "written_at": datetime.now(tz=UTC).isoformat(),
        "pid": os.getpid(),
        "uptime_minutes": round(uptime_min, 1),
        "ticks_total": stats.get("ticks", 0),
        "orderbooks_total": stats.get("orderbooks", 0),
        "markets_total": stats.get("markets", 0),
        "flushes_total": stats.get("flushes", 0),
        "ticks_per_min": round(stats.get("ticks", 0) / max(uptime_min, 0.01), 1),
        "db_size_mb": 0.0,
        "db_size_limit_reached": writer.size_limit_reached,
        "resolved_markets": 0,
        "by_venue": {},
        "last_tick_age_seconds": None,
    }

    with contextlib.suppress(OSError):
        status["db_size_mb"] = round(Path(settings.duckdb_path).stat().st_size / 1024 / 1024, 2)

    # The queries sit inside a broad suppress on purpose: the heartbeat must
    # NEVER bring down ingestion. If one fails, the field keeps its default and
    # the health check reads it as missing data.
    with contextlib.suppress(Exception):
        con = reader._con  # noqa: SLF001 - same connection, same process
        status["resolved_markets"] = con.execute(
            "SELECT count(*) FROM markets WHERE status = 'resolved'"
        ).fetchone()[0]

        status["by_venue"] = {
            venue: {"ticks": ticks, "markets": markets, "features": features}
            for venue, ticks, markets, features in con.execute(
                """
                SELECT m.venue,
                       (SELECT count(*) FROM ticks    t WHERE t.venue = m.venue),
                       count(*),
                       (SELECT count(*) FROM features f WHERE f.venue = m.venue)
                FROM markets m GROUP BY m.venue
                """
            ).fetchall()
        }

        last_ts = con.execute("SELECT max(timestamp) FROM ticks").fetchone()[0]
        if last_ts is not None:
            age = (datetime.now(tz=UTC) - last_ts).total_seconds()
            status["last_tick_age_seconds"] = round(age, 1)

    return status


async def _stats_loop(
    writer: MarketDataWriter,
    reader: MarketDataReader,
    interval: int = 60,
) -> None:
    """
    Log statistics and publish a JSON heartbeat every N seconds.

    The heartbeat is what scripts/health_check.py consumes. It is written
    atomically (temp file + rename) so a reader never sees half a JSON
    medias.
    """
    while True:
        await asyncio.sleep(interval)

        status = _collect_status(writer, reader)
        log.info(
            "system_stats",
            uptime_minutes=status["uptime_minutes"],
            ticks_total=status["ticks_total"],
            ticks_per_min=status["ticks_per_min"],
            resolved_markets=status["resolved_markets"],
            db_size_mb=status["db_size_mb"],
        )

        with contextlib.suppress(OSError):
            path = Path(settings.status_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(status, indent=2))
            tmp.replace(path)


# ---------------------------------------------------------------------------
# Shutdown graceful
# ---------------------------------------------------------------------------


def _setup_signal_handlers(shutdown: asyncio.Event) -> None:
    """
    SIGINT (Ctrl+C) and SIGTERM (kill) set the stop event.

    Why an Event and not loop.stop():
      loop.stop() cut the loop mid run_until_complete, so the
      `async with MarketDataWriter(...)` block NEVER exited: writer.stop()
      never ran, the last window of ticks was lost, and the process stayed
      alive holding the DuckDB lock. Under `timeout 120 python main.py` that
      left zombie processes that made the database impossible to open.

      With an Event, _run() cancels its tasks in order and lets the context
      manager unwind: final flush, connection close, exit.
    """
    loop = asyncio.get_running_loop()

    def _handle(sig: signal.Signals) -> None:
        log.info("shutdown_signal_received", signal=sig.name)
        shutdown.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: _handle(s))


# ---------------------------------------------------------------------------
# Main async
# ---------------------------------------------------------------------------


async def _run() -> None:
    log.info(
        "system_starting",
        duckdb_path=settings.duckdb_path,
        paper_trading=settings.paper_trading,
    )

    Path(settings.duckdb_path).parent.mkdir(parents=True, exist_ok=True)

    shutdown = asyncio.Event()
    _setup_signal_handlers(shutdown)

    async with MarketDataWriter(
        db_path=settings.duckdb_path,
        max_db_size_mb=settings.max_db_size_mb,
        size_check_every=settings.db_size_check_every,
    ) as writer:
        reader = MarketDataReader(db_path=settings.duckdb_path)
        store = FeatureStore(reader, writer)
        markets_cache: dict[str, Market] = {}

        on_tick, on_snapshot = make_callbacks(writer, store, markets_cache)
        connectors = build_connectors(on_tick, on_snapshot)

        if not connectors:
            log.error("no_connectors_active — aborting")
            return

        log.info(
            "system_started",
            active_venues=[name for name, _ in connectors],
        )

        tasks = [asyncio.create_task(connector.run()) for _, connector in connectors]
        tasks.append(asyncio.create_task(_stats_loop(writer, reader)))
        stopper = asyncio.create_task(shutdown.wait(), name="shutdown-signal")

        # Wait for the stop signal OR for ALL connectors to finish.
        #
        # With FIRST_COMPLETED over the individual tasks, the first connector to
        # return — for instance a venue whose markets all fail the quality
        # filter — cancelled the others and took down the whole ingestion. One
        # venue running out of markets is no reason to stop ingesting the rest.
        all_connectors = asyncio.gather(*tasks, return_exceptions=True)

        try:
            done, pending = await asyncio.wait(
                [all_connectors, stopper], return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

            if all_connectors in done:
                for result in all_connectors.result():
                    if isinstance(result, Exception) and not isinstance(
                        result, asyncio.CancelledError
                    ):
                        log.error("connector_failed", error=str(result))
        finally:
            reader.close()
            log.info("system_stopped", final_stats=writer.stats)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    asyncio.run() creates the loop, closes it and cancels any pending tasks.
    Signal handling lives inside _run(), where a loop is already running.
    """
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        # Only if Ctrl+C arrives before the handlers are registered.
        log.info("keyboard_interrupt")
    log.info("event_loop_closed")


if __name__ == "__main__":
    main()
