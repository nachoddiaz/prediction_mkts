"""
main.py
────────
Punto de entrada del sistema de ingesta de datos.

Orquesta connectors → writer → feature store en un loop async permanente.

Por qué asyncio.gather y no threads:
  Los connectors son I/O-bound (WebSocket, HTTP). asyncio permite
  manejar múltiples conexiones concurrentes con un solo thread,
  sin el overhead ni los race conditions de los threads de Python.

Venues activas por defecto:
  Solo Manifold — sandbox sin credenciales, ideal para desarrollo.
  Kalshi y Polymarket se activan via variables de entorno.

Uso:
    # Solo Manifold (sandbox)
    uv run python main.py

    # Kalshi + Manifold
    KALSHI_API_KEY=xxx KALSHI_PRIVATE_KEY_PATH=./secrets/kalshi.pem uv run python main.py

    # Todos los venues
    ENABLE_POLYMARKET=true uv run python main.py

    # Cambiar nivel de log
    LOG_LEVEL=DEBUG uv run python main.py

    # Producción (JSON logs)
    LOG_FORMAT=json uv run python main.py
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging — debe configurarse antes de cualquier otro import del proyecto
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """
    Configura structlog para output legible en desarrollo o JSON en producción.

    LOG_FORMAT=json  → JSON estructurado (para Datadog, Grafana Loki, etc.)
    LOG_FORMAT=text  → output coloreado y legible (default, para desarrollo)
    LOG_LEVEL=DEBUG  → nivel de detalle (default INFO)

    Por qué structlog y no el logging estándar de Python:
      structlog permite añadir contexto estructurado (venue="kalshi",
      market_id="KXBTC-...") a cada línea de log sin concatenar strings.
      En producción ese contexto es JSON parseable — puedes filtrar por
      venue o market_id en cualquier agregador de logs.
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
        # Producción: JSON por línea — parseable por cualquier agregador
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Desarrollo: output con colores y alineación legible
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
# Imports del proyecto — después de configurar el logging
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
# Callbacks — conectan connectors con writer y feature store
# ---------------------------------------------------------------------------


def make_callbacks(
    writer: MarketDataWriter,
    store: FeatureStore,
    markets_cache: dict[str, Market],
) -> tuple:
    """
    Construye los callbacks que reciben eventos de los connectors.

    Por qué closures y no métodos:
      Los connectors solo conocen la firma (tick) -> None.
      No deben saber nada del writer ni del store.
      Las closures capturan writer y store sin acoplamiento.
    """

    async def on_tick(tick: Tick) -> None:
        """Persiste el tick y recalcula features."""
        await writer.enqueue(tick)
        await writer.flush_now()

        market_id = str(tick.market_id)
        market = markets_cache.get(market_id)
        if market:
            store.on_tick(tick, market)

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

        log.info(
            "tick_received",
            venue=tick.market_id.venue.value,
            market_id=market_id,
            mid=round(tick.mid, 4),
            spread=round(tick.spread, 4),
            tick_type=tick.tick_type.value,
            tau=tau,
        )

    async def on_snapshot(snapshot: MarketSnapshot) -> None:
        """Persiste el snapshot completo y recalcula features."""
        await writer.enqueue_snapshot(snapshot)
        await writer.flush_now()

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
    Construye los connectors activos según variables de entorno.

    Manifold:    siempre activo (sandbox, sin credenciales)
    Kalshi:      activo si KALSHI_API_KEY está definida
    Polymarket:  activo si ENABLE_POLYMARKET=true
    """
    connectors = []

    # --- Manifold (siempre activo) ---
    poll_interval = int(os.getenv("MANIFOLD_POLL_INTERVAL", "10"))
    connectors.append(
        (
            "manifold",
            ManifoldConnector(
                on_tick=on_tick,
                on_snapshot=on_snapshot,
                poll_interval=poll_interval,
            ),
        )
    )
    log.info("connector_enabled", venue="manifold", poll_interval=poll_interval)

    # --- Kalshi ---
    if settings.kalshi_api_key:
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
                ),
            )
        )
        log.info("connector_enabled", venue="kalshi", env=settings.kalshi_env)
    else:
        log.info("connector_disabled", venue="kalshi", reason="KALSHI_API_KEY not set")

    # --- Polymarket ---
    if os.getenv("ENABLE_POLYMARKET", "false").lower() == "true":
        from connectors.polymarket import PolymarketConnector

        connectors.append(
            (
                "polymarket",
                PolymarketConnector(on_tick=on_tick, on_snapshot=on_snapshot),
            )
        )
        log.info("connector_enabled", venue="polymarket")
    else:
        log.info("connector_disabled", venue="polymarket", reason="ENABLE_POLYMARKET not set")

    return connectors


# ---------------------------------------------------------------------------
# Stats periódicos
# ---------------------------------------------------------------------------


async def _stats_loop(writer: MarketDataWriter, interval: int = 60) -> None:
    """
    Loguea estadísticas del sistema cada N segundos.
    Útil para monitorizar el health del pipeline sin un dashboard.
    """
    while True:
        await asyncio.sleep(interval)
        stats = writer.stats
        uptime = round((datetime.now(tz=UTC).timestamp() - _START_TIME) / 60, 1)
        log.info(
            "system_stats",
            uptime_minutes=uptime,
            ticks_total=stats.get("ticks", 0),
            orderbooks_total=stats.get("orderbooks", 0),
            markets_total=stats.get("markets", 0),
            flushes_total=stats.get("flushes", 0),
            ticks_per_min=round(stats.get("ticks", 0) / max(uptime, 0.01), 1),
        )


# ---------------------------------------------------------------------------
# Shutdown graceful
# ---------------------------------------------------------------------------


def _setup_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """
    SIGINT (Ctrl+C) y SIGTERM (kill) paran el loop limpiamente.
    Garantiza que el writer hace flush final antes de cerrar.
    """

    def _handle(sig: signal.Signals) -> None:
        log.info("shutdown_signal_received", signal=sig.name)
        loop.stop()

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

    async with MarketDataWriter(db_path=settings.duckdb_path) as writer:
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

        tasks = [connector.run() for _, connector in connectors]
        tasks.append(_stats_loop(writer))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            log.info("tasks_cancelled")
        finally:
            reader.close()
            log.info("system_stopped", final_stats=writer.stats)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------


def main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _setup_signal_handlers(loop)

    try:
        loop.run_until_complete(_run())
    except KeyboardInterrupt:
        log.info("keyboard_interrupt")
    finally:
        pending = asyncio.all_tasks(loop)
        if pending:
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
        log.info("event_loop_closed")


if __name__ == "__main__":
    main()
