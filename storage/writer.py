"""
storage/writer.py
──────────────────
Infrastructure Layer — persiste objetos del dominio en DuckDB
usando un buffer asíncrono con flush periódico.

Por qué esta arquitectura:
  El connector produce ticks a velocidad variable e impredecible
  (ráfagas del WebSocket). DuckDB es más eficiente con batch inserts
  que con inserts individuales. El buffer desacopla ambas velocidades:
  el connector encola en O(1) siempre, y el writer escribe en batch
  cuando el buffer está lleno o ha pasado suficiente tiempo.

Dos condiciones de flush (la que ocurra primero):
  - Tiempo:  cada FLUSH_INTERVAL_SECONDS segundos
  - Tamaño:  cuando el buffer acumula FLUSH_MAX_ITEMS items
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

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


_QueueItem = Tick | OrderBook | Market


class MarketDataWriter:
    """
    Writer asíncrono con buffer en memoria para DuckDB.

    Dos modos de uso:

    PRODUCCIÓN — async context manager:
        async with MarketDataWriter() as writer:
            await writer.enqueue(tick)
            await writer.enqueue(orderbook)
        # Al salir hace flush final y cierra la conexión

    TESTS / SCRIPTS — síncrono directo:
        writer = MarketDataWriter(db_path=":memory:")
        writer.write_ticks_sync([tick1, tick2])
        writer.close()
    """

    def __init__(
        self,
        db_path: str | None = None,
        flush_interval_seconds: int = FLUSH_INTERVAL_SECONDS,
        flush_max_items: int = FLUSH_MAX_ITEMS,
    ) -> None:
        self._db_path = db_path or os.getenv("DUCKDB_PATH", "./data/duckdb/markets.duckdb")

        # Crear directorio si no existe.
        # :memory: es el modo de tests — no necesita directorio.
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._con = duckdb.connect(self._db_path)
        self._init_schema()

        self._flush_interval = flush_interval_seconds
        self._flush_max_items = flush_max_items

        self._queue: asyncio.Queue[_QueueItem | None] = asyncio.Queue()

        self._flush_task: asyncio.Task | None = None

        # Estadísticas acumuladas para logging y monitorización.
        self._stats = {"ticks": 0, "orderbooks": 0, "markets": 0, "flushes": 0}

    def _init_schema(self) -> None:
        """
        Ejecuta la migración SQL inicial si las tablas no existen.

        Por qué statement a statement y no todo de una vez:
            DuckDB no soporta múltiples statements con columnas
            GENERATED ALWAYS AS en un solo execute(). Las columnas
            generadas (mid, spread, date_) usan esta sintaxis,
            así que hay que ejecutar cada CREATE TABLE por separado.

        Por qué ignorar errores:
            Si la tabla ya existe DuckDB lanza un error. Lo ignoramos
            porque usar IF NOT EXISTS no funciona con columnas generadas
            en todas las versiones de DuckDB.
        """
        migration_path = Path(__file__).parent / "migrations" / "001_initial_schema.sql"
        if not migration_path.exists():
            log.warning("Migration file not found: %s", migration_path)
            return

        sql = migration_path.read_text()
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        for stmt in statements:
            try:
                self._con.execute(stmt)
            except duckdb.Error as e:
                msg = str(e).lower()
                # Ignorar solo errores de tabla/índice ya existente
                if "already exists" not in msg:
                    log.warning("Schema migration warning: %s", e)

    # ------------------------------------------------------------------
    # API asíncrona — producción
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Arranca el loop de flush en background como asyncio Task.

        Por qué create_task y no directamente await:
          El flush loop debe correr concurrentemente con el connector
          que produce ticks. create_task lo lanza en el background
          sin bloquear. El connector puede seguir encolando mientras
          el loop escribe en disco.
        """
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
        Para el loop de flush de forma ordenada (graceful shutdown):
          1. Envía None a la queue — señal de parada para el loop
          2. Espera a que el loop termine con await
          3. Hace un flush síncrono final de lo que quede en el buffer
          4. Cierra la conexión DuckDB

        Por qué None como señal de parada:
          Es el patrón estándar para "poison pill" en colas asíncronas.
          El loop comprueba si el item es None antes de procesarlo.
          Así evitamos cancelar la tarea abruptamente, que podría
          dejar datos sin persistir.
        """
        if self._flush_task and not self._flush_task.done():
            await self._queue.put(None)  # poison pill
            await self._flush_task
        self._flush_remaining()
        self._con.close()
        log.info("MarketDataWriter stopped — stats: %s", self._stats)

    async def enqueue(self, item: _QueueItem) -> None:
        """
        Encola un item para escritura asíncrona.
        Siempre O(1) — no bloquea nunca.

        Por qué forzar flush cuando el buffer está lleno:
          Sin este control, en un pico de actividad la queue podría
          crecer sin límite consumiendo toda la memoria disponible.
          Al detectar que hay flush_max_items esperando, hacemos
          un flush inmediato antes de encolar el nuevo item.
          Esto introduce una pequeña latencia en ese momento puntual,
          pero mantiene el uso de memoria acotado.
        """
        if self._queue.qsize() >= self._flush_max_items:
            await self._flush_now()

        await self._queue.put(item)

    async def enqueue_snapshot(self, snapshot: MarketSnapshot) -> None:
        """
        Encola todos los componentes de un MarketSnapshot de una vez.

        Por qué este método de conveniencia:
          El connector fetcha snapshots completos (market + orderbook + tick).
          Sin este método tendría que hacer tres llamadas a enqueue().
          Este método garantiza que los tres componentes se encolan
          en el mismo orden siempre, sin riesgo de olvidar alguno.
        """
        await self.enqueue(snapshot.market)
        if snapshot.orderbook is not None:
            await self.enqueue(snapshot.orderbook)
        if snapshot.last_tick is not None:
            await self.enqueue(snapshot.last_tick)

    async def _flush_loop(self) -> None:
        """
        Coroutine que corre en background indefinidamente.

        Lógica del timer con wait_for:
          Intentamos leer un item de la queue con timeout de
          flush_interval_seconds. Si llega un item antes del timeout,
          lo devolvemos a la queue y hacemos flush de todo.
          Si el timeout expira sin items, también hacemos flush
          (puede ser un flush vacío — es barato).

        Por qué capturar Exception genérica:
          Un error de escritura en DuckDB (disco lleno, corrupción)
          no debe matar el loop. Lo logueamos como error y continuamos.
          Si el loop muriese, perderíamos todos los datos subsiguientes
          silenciosamente.
        """
        while True:
            try:
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=self._flush_interval,
                    )
                    if item is None:
                        # Señal de parada recibida — salir del loop
                        log.debug("Flush loop received stop signal")
                        return
                    # Devolver el item para que _flush_now() lo procese
                    # junto con los demás que puedan haber llegado
                    await self._queue.put(item)
                except TimeoutError:
                    # Timer expirado — hacer flush de lo que haya (puede ser 0)
                    pass

                await self._flush_now()

            except Exception as e:
                log.error("Error in flush loop: %s", e, exc_info=True)

    async def _flush_now(self) -> None:
        """
        Vacía la queue y escribe todos los items en DuckDB en batch.

        Por qué drenar la queue completa antes de escribir:
          Queremos un solo executemany por tipo por flush, no un
          execute por item. Primero clasificamos todos los items
          por tipo, luego hacemos un insert por tabla.
          Esto es 10-50x más eficiente que inserts individuales.

        Por qué get_nowait en lugar de get:
          get() bloquea si la queue está vacía. get_nowait() lanza
          QueueEmpty que capturamos para saber que hemos vaciado todo.
          Es la forma correcta de drenar una queue asíncrona de forma
          no bloqueante.
        """
        ticks: list[Tick] = []
        orderbooks: list[OrderBook] = []
        markets: list[Market] = []

        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item is None:
                    # Señal de parada encontrada vaciando — reponerla
                    await self._queue.put(None)
                    break
                if isinstance(item, Tick):
                    ticks.append(item)
                elif isinstance(item, OrderBook):
                    orderbooks.append(item)
                elif isinstance(item, Market):
                    markets.append(item)
            except asyncio.QueueEmpty:
                break

        # Batch insert por tabla — solo si hay datos
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

    def _flush_remaining(self) -> None:
        """
        Flush síncrono final — llamado desde stop() después de
        cancelar el loop async.

        Por qué necesitamos esto además del loop:
          Entre el momento en que el loop recibe el None y el momento
          en que stop() llama a este método, pueden haber llegado
          más items a la queue. Este método los persiste antes de cerrar.
          Es la garantía de que no perdemos datos en el shutdown.
        """
        ticks: list[Tick] = []
        orderbooks: list[OrderBook] = []
        markets: list[Market] = []

        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item is None:
                    break
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
        Inserta ticks en batch con executemany.

        Por qué no incluimos mid, spread y date_:
          Son columnas GENERATED ALWAYS AS en el schema SQL.
          DuckDB las calcula automáticamente desde yes_bid, yes_ask
          y timestamp. Si intentamos insertarlas manualmente, DuckDB
          lanza un error. La responsabilidad de calcularlas es de la DB,
          no del writer — esto garantiza consistencia siempre.

        Por qué usar ? en lugar de :named_params:
          DuckDB no soporta parámetros nombrados (:param) en executemany.
          Solo soporta posicionales (?). Los named params solo funcionan
          en execute() simple, no en batch.
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
            )
            for t in ticks
        ]
        self._con.executemany(
            """
            INSERT INTO ticks
                (market_id, venue, timestamp, tick_type,
                 yes_bid, yes_ask, volume, side)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _write_orderbooks_batch(self, orderbooks: list[OrderBook]) -> None:
        """
        Inserta orderbooks en batch.

        Por qué guardar bids_json y asks_json además de best_bid/ask:
          best_bid y best_ask son para queries analíticas rápidas
          sin parsear JSON. bids_json y asks_json preservan la
          profundidad completa del libro — necesaria para reconstruir
          el orderbook completo en backtesting o para calcular métricas
          de liquidez más allá del top level.

        Por qué bid_depth_5 y ask_depth_5 pre-computados:
          Son las features más consultadas por el feature store.
          Pre-computarlas en el insert evita parsear el JSON en
          cada query, a costa de un poco más de espacio en disco.
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
        Upsert de markets con ON CONFLICT DO UPDATE.

        Por qué upsert en lugar de INSERT:
          El connector redescubre mercados periódicamente para detectar
          cambios de status (OPEN → RESOLVED). Sin upsert tendríamos
          filas duplicadas. Con upsert, si el market_id ya existe solo
          actualizamos los campos que pueden cambiar (status, resolved_value,
          updated_at), preservando los campos estáticos (question, category).

        Por qué updated_at se asigna aquí y no en el schema:
          DuckDB no soporta DEFAULT NOW() con TIMESTAMPTZ de forma
          consistente entre versiones. Lo asignamos explícitamente
          en Python para garantizar que es UTC siempre.
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

    def _write_features_batch(self, rows: list[dict]) -> None:
        """
        Inserta features pre-computadas en batch.
        Llamado desde features/store.py, no desde el connector.

        Por qué convertir dicts a tuples antes de executemany:
          DuckDB no soporta :named_params en executemany (solo en
          execute simple). Convertimos cada dict a una tuple con
          el orden correcto de columnas explícitamente.
        """
        tuples = [
            (
                r["market_id"],
                r["venue"],
                r["timestamp"],
                r["obi"],
                r["quoted_spread"],
                r["relative_spread"],
                r["bernoulli_vol"],
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
                 relative_spread, bernoulli_vol, ewma_vol, tau_years, mu_hat)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuples,
        )

    # ------------------------------------------------------------------
    # API síncrona — tests y scripts
    # Escribe directamente sin pasar por la queue.
    # ------------------------------------------------------------------

    def write_ticks_sync(self, ticks: Sequence[Tick]) -> int:
        """
        Escribe ticks síncronamente. Para tests y scripts.

        Por qué no usar enqueue en tests:
          enqueue requiere un event loop activo. En tests síncronos
          no hay event loop. Esta API permite testear la lógica de
          persistencia sin la complejidad del sistema async.
        """
        if not ticks:
            return 0
        self._write_ticks_batch(list(ticks))
        return len(ticks)

    def write_orderbook_sync(self, ob: OrderBook) -> None:
        """Escribe un orderbook síncronamente."""
        self._write_orderbooks_batch([ob])

    def write_market_sync(self, market: Market) -> None:
        """Escribe un market síncronamente (upsert)."""
        self._write_markets_batch([market])

    def write_features_sync(self, rows: list[dict]) -> int:
        """Escribe features síncronamente."""
        if not rows:
            return 0
        self._write_features_batch(rows)
        return len(rows)

    def write_snapshot_sync(self, snapshot: MarketSnapshot) -> None:
        """
        Escribe un MarketSnapshot completo síncronamente.
        Persiste market, orderbook y tick en una sola llamada.
        """
        self.write_market_sync(snapshot.market)
        if snapshot.orderbook is not None:
            self.write_orderbook_sync(snapshot.orderbook)
        if snapshot.last_tick is not None:
            self.write_ticks_sync([snapshot.last_tick])

    # ------------------------------------------------------------------
    # Context managers y housekeeping
    # ------------------------------------------------------------------

    async def __aenter__(self) -> MarketDataWriter:
        """Arranca el flush loop al entrar en el context manager."""
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Para el flush loop y hace flush final al salir."""
        await self.stop()

    def close(self) -> None:
        """Cierre síncrono de la conexión DuckDB. Para tests."""
        self._con.close()

    def __enter__(self) -> MarketDataWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def stats(self) -> dict:
        """
        Estadísticas de escritura acumuladas desde el arranque.
        Útil para monitorización y logging periódico.
        """
        return dict(self._stats)
