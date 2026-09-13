"""
storage/reader.py
──────────────────
Infrastructure layer — read access to DuckDB.

Why one file with three sections:
  Institutional trading systems use a single access object (DataStore,
  Repository or Reader depending on the house style) with methods grouped by
  purpose. Splitting into several classes would mean several connections to
  the same database — inefficient and prone to concurrency errors. One shared
  connection across the three sections is simpler and faster.

Three sections:
  1. OPERATIONAL  — bounded queries for the feature store and execution engine
  2. ANALYTICAL   — full DataFrames for research notebooks
  3. BACKTESTING  — chunked iteration over long histories
"""

from __future__ import annotations

import os
from collections.abc import Generator
from datetime import datetime
from typing import Any

import duckdb
import pandas as pd


class MarketDataReader:
    """
    Read access to the DuckDB database.

    Why read_only=False in a reader:
      DuckDB in read_only=True mode does not allow temporary views, which
      complex notebook queries need. In :memory: mode (tests) the connection
      must also be read_only=False. The Reader name signals intended use, not
      a technical restriction.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path: str = (
            db_path if db_path else os.getenv("DUCKDB_PATH", "./data/duckdb/markets.duckdb")
        )
        self._con = duckdb.connect(self._db_path, read_only=False)

    # ══════════════════════════════════════════════════════════════════
    # SECTION 1 — OPERATIONAL
    # Fast, always-bounded queries. They never return more rows than were
    # explicitly asked for. Used on the hot path: feature store and execution
    # engine.
    # ══════════════════════════════════════════════════════════════════

    def latest_ticks(self, market_id: str, n: int = 100) -> pd.DataFrame:
        """
        Last N ticks of a market, most recent first.

        Why ORDER BY DESC + LIMIT rather than a window function:
          It is the most efficient way to ask for "the last N" in DuckDB, and
          the idx_ticks_market_ts index makes this ORDER BY very fast.

        Why return DESC (most recent first):
          The feature store needs the current mid price (index 0) and the N
          before it for the EWMA. With DESC, iloc[0] is always the most recent
          row without reversing the DataFrame. If you need ascending order for
          the EWMA, call .sort_values("timestamp").
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
        Most recent order book snapshot — exactly one row.

        Why LIMIT 1 rather than MAX(timestamp):
          ORDER BY + LIMIT 1 is faster than MAX() because it can use the index
          directly and stop at the first row; MAX() requires a full scan.

        Returns an empty DataFrame when there is no data — the caller must
        check `if not df.empty` before accessing iloc[0].
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
        Most recent features — one row.
        Used by the GLFT strategy to read μ̂, OBI and τ on every
        quoting decision cycle.
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
        Metadata for one market — a single row, or empty.
        Used by the connector to check whether a market is already in the
        database before inserting it for the first time.
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
        List of open markets, optionally filtered by venue.

        Why ORDER BY resolution_date ASC:
          The connector processes the markets closing soonest first — they
          are the most urgent for the trading system.

        Used by the connector at startup to know which markets to monitor.
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
        Most recent mid price as a float — a shortcut for the execution engine.

        Why float | None rather than a DataFrame:
          The execution engine needs the price as a number for the GLFT
          computation, not as a DataFrame. Avoiding the DataFrame → float
          conversion on every quoting cycle saves microseconds on the hot
          path.

        Returns None when there are no ticks — the caller must handle that.
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
    # SECTION 2 — ANALYTICAL
    # Exploratory queries for research notebooks. These can return large
    # DataFrames — the caller is responsible for not exhausting memory over
    # very long periods.
    # ══════════════════════════════════════════════════════════════════

    def ticks(
        self,
        market_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
        tick_type: str | None = None,
    ) -> pd.DataFrame:
        """
        Complete tick time series for one market.

        Why the query is built dynamically from conditions:
          Static SQL with every optional filter as NULL is less readable and
          can be less efficient (the planner does not always optimise IS NULL
          correctly). With dynamic conditions the query uses exactly the
          indexes it needs.

        Why ORDER BY ASC here and DESC in latest_ticks:
          Time-series analysis always runs chronologically: pandas, matplotlib
          and the backtesting engine all expect ascending order.
        """
        conditions = ["market_id = ?"]
        params: list[Any] = [market_id]

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
        TRADE ticks only (real fills).

        Why trades are separated from quotes:
          Realised volatility is computed over trades alone — using
          quotes would inject bid-ask bounce noise. Adverse selection is
          computed over trades together with their side, so keeping the two
          separate makes the caller's intent explicit.
        """
        return self.ticks(market_id, start=start, end=end, tick_type="trade")

    def orderbooks(
        self,
        market_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Order book snapshot time series.
        Without bids_json/asks_json, to keep the DataFrame manageable — use
        latest_orderbook() when the full book is needed.
        """
        conditions = ["market_id = ?"]
        params: list[Any] = [market_id]

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
        Used in notebooks to analyse how OBI, σ_b and μ̂ evolve, and to
        calibrate the models in MATH.md.
        """
        conditions = ["market_id = ?"]
        params: list[Any] = [market_id]

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
        """List of markets with optional filters."""
        conditions: list[str] = []
        params: list[Any] = []

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
        Daily trade volume.

        Why GROUP BY date_ rather than DATE_TRUNC(timestamp):
          date_ is a generated column in the SQL schema that already holds
          DATE(timestamp). Grouping by an existing column is faster than
          applying a function inside the GROUP BY.
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
        Average spread aggregated into time buckets.

        Why time_bucket rather than DATE_TRUNC:
          time_bucket is DuckDB's native function for bucketing over
          arbitrary intervals. More flexible than DATE_TRUNC, which only
          supports fixed granularities.

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
        Compare mid prices across venues to spot arbitrage.

        Why JOIN against markets rather than filtering on ticks alone:
          Kalshi and Polymarket market_ids are completely different
          identifiers for the same event. Joining on the question text finds
          equivalent markets across venues without a hand-maintained mapping.

        Used in research/05_arb_opportunities.ipynb.
        """
        conditions = ["LOWER(m.question) LIKE ?"]
        params: list[Any] = [f"%{question_fragment.lower()}%"]

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
        Every resolved market together with its outcome.

        Used in notebooks to build the Brier calibration dataset — it needs
        the (historical price, outcome) pair to evaluate how well the market
        was calibrated.
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
    # SECTION 3 — BACKTESTING
    # Efficient iteration over long time series.
    # Chunked streaming, for constant memory use.
    # ══════════════════════════════════════════════════════════════════

    def ticks_chunked(
        self,
        market_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
        chunk_size: int = 10_000,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Iterator over ticks in chunks of chunk_size rows.

        Why chunks and not a single DataFrame:
          Six months of one-second ticks is ~15M rows, 3-5 GB in memory. With
          10k-row chunks, memory use is constant regardless of how long the
          history is.

        Why fetchmany rather than fetchall:
          fetchmany() is DuckDB's streaming method — it returns N rows without
          loading the full result into memory. fetchall() would load the whole
          result before returning anything.

        Why the DataFrame is built here rather than returning tuples:
          The backtesting engine and the notebooks expect DataFrames with
          named columns. Building it here with the right names saves every
          caller from doing it.

        Typical use:
            total = reader.count_ticks(market_id)
            for i, chunk in enumerate(reader.ticks_chunked(market_id)):
                progress = i * chunk_size / total
                strategy.on_batch(chunk)
        """
        conditions = ["market_id = ?"]
        params: list[Any] = [market_id]

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
        Iterator over order book snapshots, in chunks.

        Why chunk_size=5_000 rather than 10_000:
          Order books carry bids_json and asks_json, which can be strings of
          several KB. At 10k rows a chunk could reach several hundred MB; 5k
          is more conservative.
        """
        conditions = ["market_id = ?"]
        params: list[Any] = [market_id]

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
        Total number of ticks in a range.

        Why COUNT before iterating:
          It lets the backtester report progress (chunk N of M) without
          loading the data first. The query is very cheap — DuckDB uses index
          statistics.
        """
        conditions = ["market_id = ?"]
        params: list[Any] = [market_id]

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
