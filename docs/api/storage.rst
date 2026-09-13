storage
=======

Persistence layer backed by an embedded DuckDB database.  One shared
connection is used throughout the process — there is no connection pool
to manage.

The four tables maintained by this layer are:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Table
     - Write frequency
     - Description
   * - ``markets``
     - Low
     - One row per market per venue; updated on status change or resolution.
   * - ``ticks``
     - Very high
     - One row per WebSocket event; the primary time-series table.
   * - ``orderbooks``
     - Medium
     - Full order-book snapshots fetched every N seconds.
   * - ``features``
     - Very high
     - Pre-computed microstructure features; one row per tick.

Writer
------

.. automodule:: storage.writer
   :members:
   :undoc-members:
   :show-inheritance:

Reader
------

Exposes three query tiers:

1. **Operational** — bounded queries for the feature store and execution
   engine hot path.  Never returns more rows than explicitly requested.
2. **Analytical** — full DataFrames for research notebooks and calibration.
3. **Backtesting** — chunked iteration over long historical windows.

.. automodule:: storage.reader
   :members:
   :undoc-members:
   :show-inheritance:

Archiver
--------

Periodically exports the DuckDB tables to Parquet files for long-term
storage and offline analysis.

.. automodule:: storage.archiver
   :members:
   :undoc-members:
   :show-inheritance:
