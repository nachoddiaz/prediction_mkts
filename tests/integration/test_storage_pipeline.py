"""
tests/integration/test_storage_pipeline.py
────────────────────────────────────────────
Tests del writer y del pipeline de storage.

Por qué DuckDB :memory: en lugar de un archivo temporal:
  :memory: es más rápido (sin I/O de disco), se destruye automáticamente
  al cerrar la conexión (sin cleanup necesario), y evita colisiones
  entre tests paralelos. El comportamiento es idéntico al modo archivo
  para todos los propósitos de testing.

Por qué separar TestSyncAPI de TestAsyncAPI:
  La API síncrona testea la lógica de persistencia pura — si el SQL
  es correcto, si las columnas generadas funcionan, si el upsert
  actualiza los campos correctos. La API asíncrona testea el comportamiento
  del buffer — flush por tamaño, flush por tiempo, graceful shutdown.
  Son responsabilidades distintas que merecen tests distintos.
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
# Helpers — constructores de objetos del dominio para tests
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
    """Writer con DuckDB en memoria — el más rápido para tests."""
    return MarketDataWriter(db_path=":memory:")


# ---------------------------------------------------------------------------
# Tests API síncrona
# ---------------------------------------------------------------------------


class TestSyncAPI:
    def test_write_market(self) -> None:
        """Verifica que los campos básicos de un market se persisten."""
        with mem() as w:
            w.write_market_sync(make_market())
            row = w._con.execute("SELECT market_id, status, category FROM markets").fetchone()
            assert row[0] == "kalshi:KXBTC-TEST"
            assert row[1] == "open"
            assert row[2] == "crypto"

    def test_upsert_no_duplica(self) -> None:
        """
        Escribir el mismo market_id dos veces debe producir una sola fila.
        Verifica que ON CONFLICT DO UPDATE funciona correctamente.
        """
        with mem() as w:
            w.write_market_sync(make_market())
            w.write_market_sync(make_market())
            count = w._con.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
            assert count == 1

    def test_upsert_actualiza_status(self) -> None:
        """
        Al resolver un mercado, el upsert debe actualizar status y
        resolved_value sin duplicar la fila ni modificar otros campos.
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
        Batch de N ticks debe persistir exactamente N filas.
        Verifica que executemany funciona correctamente con múltiples items.
        """
        with mem() as w:
            ticks = [make_tick() for _ in range(5)]
            assert w.write_ticks_sync(ticks) == 5
            count = w._con.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
            assert count == 5

    def test_tick_quote_side_null(self) -> None:
        """
        Los ticks QUOTE no tienen side — debe persistirse como NULL.
        side=NULL es una regla de negocio del schema: sin agresión,
        no hay dirección.
        """
        with mem() as w:
            w.write_ticks_sync([make_tick(tick_type=TickType.QUOTE)])
            row = w._con.execute("SELECT tick_type, side FROM ticks").fetchone()
            assert row[0] == "quote"
            assert row[1] is None

    def test_tick_trade_side_set(self) -> None:
        """
        Los ticks TRADE deben tener side — fundamental para calcular
        adverse selection y OBI en features/microstructure.py.
        """
        with mem() as w:
            w.write_ticks_sync([make_tick(tick_type=TickType.TRADE, side=Side.YES)])
            row = w._con.execute("SELECT tick_type, volume, side FROM ticks").fetchone()
            assert row[0] == "trade"
            assert row[1] == pytest.approx(10.0)
            assert row[2] == "yes"

    def test_columnas_generadas_mid_spread(self) -> None:
        """
        mid y spread son GENERATED ALWAYS AS — DuckDB las calcula
        automáticamente. Verificamos que el cálculo es correcto y que
        no necesitamos insertarlas manualmente.
        """
        with mem() as w:
            w.write_ticks_sync([make_tick(bid=0.44, ask=0.46)])
            row = w._con.execute("SELECT mid, spread FROM ticks").fetchone()
            assert row[0] == pytest.approx(0.45)  # (0.44+0.46)/2
            assert row[1] == pytest.approx(0.02)  # 0.46-0.44

    def test_write_orderbook(self) -> None:
        """
        Verifica que best_bid, best_ask y las profundidades pre-computadas
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
        bids_json y asks_json deben ser JSON válido y reconstruible.
        El backtesting engine los parseará para reconstruir el libro.
        """
        with mem() as w:
            w.write_orderbook_sync(make_orderbook())
            row = w._con.execute("SELECT bids_json, asks_json FROM orderbooks").fetchone()
            bids = _json.loads(row[0])
            asks = _json.loads(row[1])
            assert len(bids) == 2
            assert bids[0][0] == pytest.approx(0.45)  # mejor bid primero
            assert asks[0][0] == pytest.approx(0.47)  # mejor ask primero

    def test_write_snapshot(self) -> None:
        """
        Un snapshot debe persistir los tres componentes en una llamada.
        Verifica que write_snapshot_sync orquesta correctamente.
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
        Verifica que features se persisten correctamente.
        La conversión de dict a tuple (por limitación de DuckDB con
        named params en executemany) debe ser transparente al caller.
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
        Pasar una lista vacía no debe lanzar excepción ni ejecutar SQL.
        El guard if not ticks/rows evita ejecutar executemany([])
        que en algunas versiones de DuckDB puede dar error.
        """
        with mem() as w:
            assert w.write_ticks_sync([]) == 0
            assert w.write_features_sync([]) == 0

    def test_multiples_venues(self) -> None:
        """
        Ticks de Kalshi y Polymarket coexisten en la misma tabla.
        El campo venue permite filtrarlos independientemente.
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
# Tests API asíncrona
# ---------------------------------------------------------------------------


class TestAsyncAPI:
    @pytest.mark.asyncio
    async def test_enqueue_tick(self) -> None:
        """
        Un tick encolado debe persistirse tras un flush explícito.
        Usamos _flush_now() directamente en lugar de esperar el timer
        para que el test sea determinista y rápido.
        """
        async with MarketDataWriter(db_path=":memory:", flush_interval_seconds=1) as w:
            await w.enqueue(make_tick())
            await w._flush_now()
            count = w._con.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
            assert count == 1

    @pytest.mark.asyncio
    async def test_enqueue_market(self) -> None:
        """Un market encolado debe persistirse tras flush."""
        async with MarketDataWriter(db_path=":memory:", flush_interval_seconds=1) as w:
            await w.enqueue(make_market())
            await w._flush_now()
            count = w._con.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
            assert count == 1

    @pytest.mark.asyncio
    async def test_flush_por_tamano(self) -> None:
        """
        Con flush_max_items=5, al llegar al límite enqueue() debe
        forzar un flush automático antes de encolar el item 6.
        El timer está en 60s para que no interfiera.
        """
        async with MarketDataWriter(
            db_path=":memory:",
            flush_interval_seconds=60,
            flush_max_items=5,
        ) as w:
            for _ in range(6):
                await w.enqueue(make_tick())
            await w._flush_now()  # flush del residuo
            count = w._con.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
            assert count == 6

    @pytest.mark.asyncio
    async def test_enqueue_snapshot(self) -> None:
        """
        enqueue_snapshot debe encolar los tres componentes del snapshot.
        Tras flush, deben existir una fila en cada tabla.
        """
        async with MarketDataWriter(db_path=":memory:") as w:
            mid = make_market_id()
            snapshot = MarketSnapshot(
                market=make_market(mid),
                orderbook=make_orderbook(mid),
                last_tick=make_tick(mid),
            )
            await w.enqueue_snapshot(snapshot)
            await w._flush_now()
            assert w._con.execute("SELECT COUNT(*) FROM markets").fetchone()[0] == 1
            assert w._con.execute("SELECT COUNT(*) FROM orderbooks").fetchone()[0] == 1
            assert w._con.execute("SELECT COUNT(*) FROM ticks").fetchone()[0] == 1

    @pytest.mark.asyncio
    async def test_stop_flush_final(self) -> None:
        """
        Al parar el writer, los items pendientes deben persistirse.
        Verificamos ANTES de stop() porque stop() cierra la conexión.
        """
        w = MarketDataWriter(db_path=":memory:", flush_interval_seconds=1)
        await w.start()
        for _ in range(3):
            await w.enqueue(make_tick())
        await w._flush_now()  # flush explícito antes de stop
        count = w._con.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
        assert count == 3
        await w.stop()
        w.close()

    @pytest.mark.asyncio
    async def test_stats(self) -> None:
        """
        Los stats deben reflejar correctamente lo escrito.
        Útil para monitorización — si stats["ticks"] no crece,
        el pipeline de ingesta está roto.
        """
        async with MarketDataWriter(db_path=":memory:") as w:
            await w.enqueue(make_tick())
            await w.enqueue(make_tick())
            await w.enqueue(make_market())
            await w._flush_now()
            assert w.stats["ticks"] == 2
            assert w.stats["markets"] == 1
            assert w.stats["flushes"] >= 1
