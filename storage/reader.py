"""
storage/reader.py
──────────────────
Infrastructure Layer — acceso de lectura a DuckDB.

Por qué un solo archivo con tres secciones:
  Los sistemas de trading institucionales usan un único objeto de acceso
  a datos (DataStore, Repository o Reader según la firma) con métodos
  agrupados por propósito. Separar en múltiples clases obligaría a gestionar
  múltiples conexiones a la misma DB — ineficiente y propenso a errores de
  concurrencia. Una sola conexión compartida entre las tres secciones es
  más simple y más rápido.

Tres secciones:
  1. OPERACIONAL  — queries acotadas para el feature store y execution engine
  2. ANALÍTICA    — DataFrames completos para research notebooks
  3. BACKTESTING  — iteración por chunks para históricos largos
"""

from __future__ import annotations

import os
from collections.abc import Generator
from datetime import datetime

import duckdb
import pandas as pd


class MarketDataReader:
    """
    Acceso de lectura a la base de datos DuckDB.

    Por qué read_only=False si es un reader:
      DuckDB en modo read_only=True no permite crear vistas temporales,
      que necesitamos para queries complejas en notebooks. Además,
      en modo :memory: (tests) la conexión debe ser read_only=False
      para compartirla con el writer. El nombre Reader indica intención
      de uso, no restricción técnica.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or os.getenv("DUCKDB_PATH", "./data/duckdb/markets.duckdb")
        self._con = duckdb.connect(self._db_path, read_only=False)

    # ══════════════════════════════════════════════════════════════════
    # SECCIÓN 1 — OPERACIONAL
    # Queries rápidas y siempre acotadas.
    # Nunca devuelven más filas de las solicitadas explícitamente.
    # Usadas en el hot path: feature store y execution engine.
    # ══════════════════════════════════════════════════════════════════

    def latest_ticks(self, market_id: str, n: int = 100) -> pd.DataFrame:
        """
        Últimos N ticks de un mercado, del más reciente al más antiguo.

        Por qué ORDER BY DESC + LIMIT en lugar de una ventana:
          Es la query más eficiente para "dame los últimos N" en DuckDB.
          El índice idx_ticks_market_ts hace este ORDER BY muy rápido.

        Por qué devolver DESC (más reciente primero):
          El feature store necesita el mid-price actual (índice 0)
          y los N anteriores para EWMA. Con DESC, iloc[0] es siempre
          el más reciente sin necesidad de invertir el DataFrame.
          Si necesitas ASC para EWMA, llama .sort_values("timestamp").
        """
        return self._con.execute(
            """
            SELECT timestamp, tick_type, yes_bid, yes_ask,
                   mid, spread, volume, side
            FROM   ticks
            WHERE  market_id = ?
            ORDER  BY timestamp DESC
            LIMIT  ?
            """,
            [market_id, n],
        ).df()

    def latest_orderbook(self, market_id: str) -> pd.DataFrame:
        """
        Snapshot más reciente del orderbook — exactamente una fila.

        Por qué LIMIT 1 y no MAX(timestamp):
          ORDER BY + LIMIT 1 es más eficiente que MAX() porque puede
          usar el índice directamente. MAX() requiere un full scan.

        Devuelve DataFrame vacío si no hay datos — el caller debe
        comprobar if not df.empty antes de acceder a iloc[0].
        """
        return self._con.execute(
            """
            SELECT timestamp, best_bid, best_ask, mid, spread,
                   bid_depth_5, ask_depth_5, bids_json, asks_json
            FROM   orderbooks
            WHERE  market_id = ?
            ORDER  BY timestamp DESC
            LIMIT  1
            """,
            [market_id],
        ).df()

    def latest_features(self, market_id: str) -> pd.DataFrame:
        """
        Features más recientes — una fila.
        Usadas por la estrategia GLFT para leer μ̂, OBI y τ
        en cada ciclo de decisión de quoting.
        """
        return self._con.execute(
            """
            SELECT timestamp, obi, quoted_spread, relative_spread,
                   belief_vol, ewma_vol, tau_years, mu_hat
            FROM   features
            WHERE  market_id = ?
            ORDER  BY timestamp DESC
            LIMIT  1
            """,
            [market_id],
        ).df()

    def market(self, market_id: str) -> pd.DataFrame:
        """
        Metadatos de un mercado específico — una fila o vacío.
        Usado por el connector para verificar si un mercado ya
        está en la DB antes de insertarlo por primera vez.
        """
        return self._con.execute(
            """
            SELECT market_id, venue, question, category,
                   status, resolution_date, resolved_value
            FROM   markets
            WHERE  market_id = ?
            """,
            [market_id],
        ).df()

    def open_markets(self, venue: str | None = None) -> pd.DataFrame:
        """
        Lista de mercados abiertos, con filtro opcional de venue.

        Por qué ORDER BY resolution_date ASC:
          El connector procesa primero los mercados que cierran antes
          — son los más urgentes para el sistema de trading.

        Usado por el connector al arrancar para saber qué mercados
        debe monitorizar.
        """
        if venue:
            return self._con.execute(
                """
                SELECT market_id, venue, question, category,
                       resolution_date, resolved_value
                FROM   markets
                WHERE  status = 'open' AND venue = ?
                ORDER  BY resolution_date ASC
                """,
                [venue],
            ).df()
        return self._con.execute(
            """
            SELECT market_id, venue, question, category,
                   resolution_date, resolved_value
            FROM   markets
            WHERE  status = 'open'
            ORDER  BY resolution_date ASC
            """,
        ).df()

    def mid_price_now(self, market_id: str) -> float | None:
        """
        Mid-price más reciente como float — shortcut para el execution engine.

        Por qué devolver float | None en lugar de DataFrame:
          El execution engine necesita el precio como número para
          los cálculos del GLFT, no como DataFrame. Evitar la
          conversión DataFrame → float en cada ciclo de quoting
          ahorra microsegundos en el hot path.

        Devuelve None si no hay ticks — el caller debe manejar este caso.
        """
        row = self._con.execute(
            """
            SELECT mid FROM ticks
            WHERE  market_id = ?
            ORDER  BY timestamp DESC
            LIMIT  1
            """,
            [market_id],
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else None

    # ══════════════════════════════════════════════════════════════════
    # SECCIÓN 2 — ANALÍTICA
    # Queries exploratorias para research notebooks.
    # Pueden devolver DataFrames grandes — el caller es responsable
    # de no saturar la memoria en períodos muy largos.
    # ══════════════════════════════════════════════════════════════════

    def ticks(
        self,
        market_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
        tick_type: str | None = None,
    ) -> pd.DataFrame:
        """
        Serie temporal completa de ticks para un mercado.

        Por qué construir la query dinámicamente con condiciones:
          SQL estático con todos los filtros opcionales como NULL
          es menos legible y puede ser menos eficiente (el planificador
          no siempre optimiza IS NULL correctamente). Con condiciones
          dinámicas la query usa exactamente los índices necesarios.

        Por qué ORDER BY ASC aquí y DESC en latest_ticks:
          El análisis de series temporales siempre va cronológico.
          pandas, matplotlib y el backtesting engine esperan orden ASC.
        """
        conditions = ["market_id = ?"]
        params: list = [market_id]

        if start:
            conditions.append("timestamp >= ?")
            params.append(start)
        if end:
            conditions.append("timestamp <= ?")
            params.append(end)
        if tick_type:
            conditions.append("tick_type = ?")
            params.append(tick_type)

        where = " AND ".join(conditions)
        return self._con.execute(
            f"""
            SELECT timestamp, tick_type, yes_bid, yes_ask,
                   mid, spread, volume, side
            FROM   ticks
            WHERE  {where}
            ORDER  BY timestamp ASC
            """,
            params,
        ).df()

    def trade_ticks(
        self,
        market_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Solo ticks de tipo TRADE (fills reales).

        Por qué separar trades de quotes:
          La volatilidad realizada se calcula solo sobre trades — usar
          quotes introduciría ruido del bid-ask bounce.
          El adverse selection se calcula sobre trades con su side.
          Esta separación hace explícita la intención del caller.
        """
        return self.ticks(market_id, start=start, end=end, tick_type="trade")

    def orderbooks(
        self,
        market_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Serie temporal de snapshots de orderbook.
        Sin bids_json/asks_json para mantener el DataFrame manejable
        — si necesitas el libro completo usa latest_orderbook().
        """
        conditions = ["market_id = ?"]
        params: list = [market_id]

        if start:
            conditions.append("timestamp >= ?")
            params.append(start)
        if end:
            conditions.append("timestamp <= ?")
            params.append(end)

        where = " AND ".join(conditions)
        return self._con.execute(
            f"""
            SELECT timestamp, best_bid, best_ask, mid, spread,
                   bid_depth_5, ask_depth_5
            FROM   orderbooks
            WHERE  {where}
            ORDER  BY timestamp ASC
            """,
            params,
        ).df()

    def features(
        self,
        market_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Serie temporal de features.
        Usado en notebooks para analizar la evolución de OBI,
        σ_B y μ̂ y para calibrar los modelos del MATH.md.
        """
        conditions = ["market_id = ?"]
        params: list = [market_id]

        if start:
            conditions.append("timestamp >= ?")
            params.append(start)
        if end:
            conditions.append("timestamp <= ?")
            params.append(end)

        where = " AND ".join(conditions)
        return self._con.execute(
            f"""
            SELECT timestamp, obi, quoted_spread, relative_spread,
                   belief_vol, ewma_vol, tau_years, mu_hat
            FROM   features
            WHERE  {where}
            ORDER  BY timestamp ASC
            """,
            params,
        ).df()

    def markets(
        self,
        venue: str | None = None,
        status: str | None = None,
        category: str | None = None,
    ) -> pd.DataFrame:
        """Lista de mercados con filtros opcionales."""
        conditions: list[str] = []
        params: list = []

        if venue:
            conditions.append("venue = ?")
            params.append(venue)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if category:
            conditions.append("category = ?")
            params.append(category)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        return self._con.execute(
            f"""
            SELECT market_id, venue, question, category,
                   status, resolution_date, resolved_value, updated_at
            FROM   markets
            {where}
            ORDER  BY resolution_date ASC
            """,
            params,
        ).df()

    def daily_volume(self, market_id: str) -> pd.DataFrame:
        """
        Volumen diario de trades.

        Por qué GROUP BY date_ en lugar de DATE_TRUNC(timestamp):
          date_ es una columna generada en el schema SQL que ya
          contiene DATE(timestamp). Agrupar por una columna existente
          es más eficiente que aplicar una función en el GROUP BY.
        """
        return self._con.execute(
            """
            SELECT date_,
                   SUM(volume) AS total_volume,
                   COUNT(*)    AS n_trades
            FROM   ticks
            WHERE  market_id = ?
              AND  tick_type  = 'trade'
            GROUP  BY date_
            ORDER  BY date_ ASC
            """,
            [market_id],
        ).df()

    def spread_timeseries(
        self,
        market_id: str,
        freq: str = "5 minutes",
    ) -> pd.DataFrame:
        """
        Spread promedio agregado por ventana temporal.

        Por qué time_bucket en lugar de DATE_TRUNC:
          time_bucket es la función nativa de DuckDB para bucketing
          temporal con intervalos arbitrarios. Más flexible que
          DATE_TRUNC que solo soporta granularidades fijas.

        freq ejemplos: "1 minute", "5 minutes", "1 hour", "1 day"
        """
        return self._con.execute(
            f"""
            SELECT time_bucket(INTERVAL '{freq}', timestamp) AS bucket,
                   AVG(spread)  AS avg_spread,
                   AVG(mid)     AS avg_mid,
                   SUM(volume)  AS volume
            FROM   ticks
            WHERE  market_id = ?
            GROUP  BY bucket
            ORDER  BY bucket ASC
            """,
            [market_id],
        ).df()

    def cross_venue_prices(
        self,
        question_fragment: str,
        start: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Compara mid-prices entre venues para detectar arbitraje.

        Por qué JOIN con markets en lugar de filtrar solo por ticks:
          Los market_ids de Kalshi y Polymarket son completamente
          distintos para el mismo evento. El JOIN sobre la pregunta
          permite encontrar mercados equivalentes entre venues sin
          necesitar un mapeo explícito.

        Usado en research/05_arb_opportunities.ipynb.
        """
        conditions = ["LOWER(m.question) LIKE ?"]
        params: list = [f"%{question_fragment.lower()}%"]

        if start:
            conditions.append("t.timestamp >= ?")
            params.append(start)

        where = " AND ".join(conditions)
        return self._con.execute(
            f"""
            SELECT t.timestamp, t.venue, t.market_id, t.mid, t.spread
            FROM   ticks   t
            JOIN   markets m ON t.market_id = m.market_id
            WHERE  {where}
            ORDER  BY t.timestamp ASC, t.venue ASC
            """,
            params,
        ).df()

    def resolved_markets_with_outcome(self) -> pd.DataFrame:
        """
        Todos los mercados resueltos con su resultado.

        Usado en notebooks para construir el dataset de calibración
        del Brier score — necesitas el par (precio histórico, resultado)
        para evaluar la calibración del mercado.
        """
        return self._con.execute(
            """
            SELECT market_id, venue, category, question,
                   resolution_date, resolved_value
            FROM   markets
            WHERE  status         = 'resolved'
              AND  resolved_value IS NOT NULL
            ORDER  BY resolution_date DESC
            """,
        ).df()

    # ══════════════════════════════════════════════════════════════════
    # SECCIÓN 3 — BACKTESTING
    # Iteración eficiente sobre series temporales largas.
    # Streaming por chunks para uso de memoria constante.
    # ══════════════════════════════════════════════════════════════════

    def ticks_chunked(
        self,
        market_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
        chunk_size: int = 10_000,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Iterador sobre ticks en chunks de chunk_size filas.

        Por qué chunks y no un solo DataFrame:
          6 meses de ticks cada segundo = ~15M filas ≈ 3-5 GB en memoria.
          Con chunks de 10k filas el uso de memoria es constante
          independientemente del tamaño del histórico.

        Por qué fetchmany en lugar de fetchall:
          fetchmany() es el método de streaming de DuckDB — devuelve
          N filas sin cargar el resultado completo en memoria.
          fetchall() cargaría todo el resultado antes de devolver nada.

        Por qué construir DataFrame aquí en lugar de devolver tuples:
          El backtesting engine y los notebooks esperan DataFrames
          con columnas nombradas. Construirlo aquí con los nombres
          correctos evita que cada caller tenga que hacerlo.

        Uso típico:
            total = reader.count_ticks(market_id)
            for i, chunk in enumerate(reader.ticks_chunked(market_id)):
                progress = i * chunk_size / total
                strategy.on_batch(chunk)
        """
        conditions = ["market_id = ?"]
        params: list = [market_id]

        if start:
            conditions.append("timestamp >= ?")
            params.append(start)
        if end:
            conditions.append("timestamp <= ?")
            params.append(end)

        where = " AND ".join(conditions)
        cursor = self._con.execute(
            f"""
            SELECT timestamp, tick_type, yes_bid, yes_ask,
                   mid, spread, volume, side
            FROM   ticks
            WHERE  {where}
            ORDER  BY timestamp ASC
            """,
            params,
        )

        columns = [desc[0] for desc in cursor.description]

        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break
            yield pd.DataFrame(rows, columns=columns)

    def orderbooks_chunked(
        self,
        market_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
        chunk_size: int = 5_000,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Iterador sobre snapshots de orderbook en chunks.

        Por qué chunk_size=5_000 en lugar de 10_000:
          Los orderbooks incluyen bids_json y asks_json que pueden
          ser strings de varios KB. Con 10k filas podríamos tener
          chunks de varios cientos de MB. 5k es más conservador.
        """
        conditions = ["market_id = ?"]
        params: list = [market_id]

        if start:
            conditions.append("timestamp >= ?")
            params.append(start)
        if end:
            conditions.append("timestamp <= ?")
            params.append(end)

        where = " AND ".join(conditions)
        cursor = self._con.execute(
            f"""
            SELECT timestamp, best_bid, best_ask, mid, spread,
                   bid_depth_5, ask_depth_5, bids_json, asks_json
            FROM   orderbooks
            WHERE  {where}
            ORDER  BY timestamp ASC
            """,
            params,
        )

        columns = [desc[0] for desc in cursor.description]

        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break
            yield pd.DataFrame(rows, columns=columns)

    def count_ticks(
        self,
        market_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        """
        Cuenta total de ticks en un rango.

        Por qué COUNT antes de iterar:
          Permite calcular el progreso del backtesting (chunk N de M)
          sin necesidad de cargar todos los datos primero.
          Es una query muy barata — DuckDB usa estadísticas del índice.
        """
        conditions = ["market_id = ?"]
        params: list = [market_id]

        if start:
            conditions.append("timestamp >= ?")
            params.append(start)
        if end:
            conditions.append("timestamp <= ?")
            params.append(end)

        where = " AND ".join(conditions)
        row = self._con.execute(
            f"SELECT COUNT(*) FROM ticks WHERE {where}",
            params,
        ).fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> MarketDataReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
