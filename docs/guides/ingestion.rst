Data Ingestion
==============

This guide explains how to run the live data ingestion pipeline, configure
which venues to enable, and verify that data is landing in DuckDB correctly.

Overview
--------

The ingestion pipeline is orchestrated by :file:`main.py` and runs as a
single ``asyncio`` event loop.  One coroutine per enabled venue connects
to the WebSocket feed, normalises incoming events, and writes them to
DuckDB via :class:`~storage.writer.MarketDataWriter`.

.. code-block:: text

   main.py
     │
     ├── KalshiConnector.stream()   ─┐
     ├── PolymarketConnector.stream() ├── asyncio.gather()
     └── ManifoldConnector.stream()  ─┘
                  │
                  │  raw dict (venue-specific)
                  ▼
         Normalizer Adapter
                  │
                  │  Market / Tick / OrderBook
                  ▼
         MarketDataWriter
                  │
                  ▼
               DuckDB

Starting the pipeline
---------------------

.. code-block:: bash

   # Manifold only — no credentials required, ideal for development
   uv run python main.py

   # Kalshi demo sandbox
   KALSHI_API_KEY=<key> KALSHI_PRIVATE_KEY_PATH=./secrets/kalshi.pem uv run python main.py

   # All venues
   ENABLE_POLYMARKET=true uv run python main.py

   # Debug logging
   LOG_LEVEL=DEBUG uv run python main.py

   # Production (structured JSON logs for Datadog / Grafana Loki)
   LOG_FORMAT=json uv run python main.py

Enabling venues
---------------

Venue activation is controlled by environment variables.  Manifold is always
enabled (it requires no credentials).  Kalshi and Polymarket are enabled when
their credentials are present:

.. list-table::
   :header-rows: 1

   * - Venue
     - Required variable(s)
   * - Manifold
     - Always active
   * - Kalshi
     - ``KALSHI_API_KEY`` + ``KALSHI_PRIVATE_KEY_PATH``
   * - Polymarket
     - ``ENABLE_POLYMARKET=true`` + ``POLYMARKET_PRIVATE_KEY``

Verifying data
--------------

After the pipeline has been running for a few minutes, open a DuckDB shell
to inspect the data:

.. code-block:: bash

   uv run python -c "
   import duckdb
   con = duckdb.connect('./data/duckdb/markets.duckdb')
   print(con.execute('SELECT venue, COUNT(*) FROM ticks GROUP BY 1').fetchall())
   print(con.execute('SELECT * FROM markets LIMIT 5').fetchdf())
   "

Or use the :class:`~storage.reader.MarketDataReader` from Python:

.. code-block:: python

   from storage.reader import MarketDataReader

   reader = MarketDataReader()

   # Latest 20 ticks for a specific market
   df = reader.latest_ticks("kalshi:KXBTC-26APR2212-T85799.99", n=20)
   print(df)

   # All markets currently tracked
   markets = reader.all_markets()
   print(markets)

Feature computation
-------------------

Features are computed automatically in the ingestion loop, one row per tick.
To inspect them:

.. code-block:: python

   df = reader.latest_features("kalshi:KXBTC-26APR2212-T85799.99", n=50)
   print(df[["timestamp", "obi", "bernoulli_vol", "ewma_vol", "mu_hat"]])

Graceful shutdown
-----------------

Press ``Ctrl+C`` to trigger a graceful shutdown.  The pipeline will:

1. Stop accepting new WebSocket messages.
2. Flush any pending DuckDB writes.
3. Close all WebSocket connections.
4. Exit with code 0.

Archiving to Parquet
--------------------

Run the archiver to export the DuckDB tables to Parquet files for
long-term storage or offline analysis:

.. code-block:: python

   from storage.archiver import DataArchiver

   archiver = DataArchiver()
   archiver.archive_ticks(market_id="kalshi:KXBTC-26APR2212-T85799.99")
