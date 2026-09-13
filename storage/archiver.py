"""
storage/archiver.py
────────────────────
Archival of DuckDB data to day-partitioned Parquet — not implemented.

What it will do: move the closed partitions of `ticks` and `orderbooks` (the
generated `date_` column) to `data/parquet/{table}/date_=YYYY-MM-DD/` and purge
them from the embedded database, which currently grows without bound.

Why not yet: without real Kalshi/Polymarket volume the whole database weighs
17 MB and archival buys nothing. It gets built once Phase 3 ingestion is
running continuously.
"""

from __future__ import annotations

from datetime import date


def archive_to_parquet(cutoff: date, parquet_base: str | None = None) -> int:
    """Not implemented yet. Will return the number of rows archived."""
    raise NotImplementedError("Parquet archival. Pending continuous Phase 3 ingestion.")
