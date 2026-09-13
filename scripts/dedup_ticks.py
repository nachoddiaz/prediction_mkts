#!/usr/bin/env python3
"""
scripts/dedup_ticks.py
───────────────────────
Removes exactly-duplicated rows from `ticks` and fills in `source_id` where it
can be deduced. Idempotent: running it twice changes nothing the second time.

What counts as a duplicate:
  A row IDENTICAL in (market_id, timestamp, tick_type, yes_bid, yes_ask,
  volume, side). (market_id, timestamp) alone is NOT enough: two distinct
  trades frequently share a timestamp to the millisecond — Manifold matches a
  `yes` and a `no` in the same transaction. In the development database, of
  2,264 groups sharing (market_id, timestamp) only 399 were true duplicates.

Usage:
    python scripts/dedup_ticks.py [--db PATH] [--apply]

Without --apply it only reports. A .bak copy is taken before anything changes.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from storage.writer import MarketDataWriter

DEDUP_KEY = "market_id, timestamp, tick_type, yes_bid, yes_ask, volume, side"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="./data/duckdb/markets.duckdb")
    parser.add_argument("--apply", action="store_true", help="apply the changes")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    # Apply pending migrations first: a database created before migration 002
    # has no source_id column and the SELECT below would fail.
    if args.apply:
        MarketDataWriter(db_path=str(db_path)).close()

    con = duckdb.connect(str(db_path), read_only=not args.apply)
    total = con.execute("SELECT count(*) FROM ticks").fetchone()[0]
    exact = con.execute(
        f"SELECT coalesce(sum(n - 1), 0) FROM "
        f"(SELECT count(*) n FROM ticks GROUP BY {DEDUP_KEY} HAVING count(*) > 1)"
    ).fetchone()[0]
    same_ts = con.execute(
        "SELECT count(*) FROM (SELECT count(*) n FROM ticks "
        "GROUP BY market_id, timestamp HAVING count(*) > 1)"
    ).fetchone()[0]

    print(f"  ticks totales ................ {total:,}")
    print(f"  groups sharing (market_id, ts) . {same_ts:,}  <- includes legitimate trades")
    print(f"  exact duplicate rows .......... {exact:,}  <- what gets removed")

    if not args.apply:
        print("\n  (simulación — usa --apply para ejecutar)")
        return 0
    if exact == 0:
        print("\n  Nothing to do.")
        return 0

    backup = db_path.with_suffix(f".bak-{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}")
    con.close()
    shutil.copy2(db_path, backup)
    print(f"\n  Copia de seguridad: {backup}")

    con = duckdb.connect(str(db_path))

    # The table is REBUILT rather than having rows deleted.
    #
    # DuckDB 1.5 aborts a DELETE against `ticks` with
    #   "Failed to delete all rows from index. Only deleted 0 out of N rows"
    # — a FATAL error that invalidates the connection — because of the table's
    # ART indexes. Dropping the unique index is not enough: the base schema's
    # own indexes (idx_ticks_market_ts, idx_ticks_venue_date) cause the same.
    #
    # Copying the good rows into a new table, swapping and recreating the
    # indexes avoids the DELETE entirely. Generated columns (mid, spread,
    # date_) are not copied: the database recomputes them on reinsert.
    columns = "market_id, venue, timestamp, tick_type, yes_bid, yes_ask, volume, side, source_id"
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            f"""
            CREATE TEMP TABLE ticks_keep AS
            SELECT {columns} FROM ticks
            QUALIFY row_number() OVER (PARTITION BY {DEDUP_KEY} ORDER BY timestamp) = 1
            """
        )
        kept = con.execute("SELECT count(*) FROM ticks_keep").fetchone()[0]

        con.execute("DROP INDEX IF EXISTS ux_ticks_market_source")
        con.execute("DROP INDEX IF EXISTS idx_ticks_market_ts")
        con.execute("DROP INDEX IF EXISTS idx_ticks_venue_date")
        con.execute("DELETE FROM ticks")
        con.execute(f"INSERT INTO ticks ({columns}) SELECT {columns} FROM ticks_keep")
        con.execute("DROP TABLE ticks_keep")

        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_ticks_market_ts ON ticks (market_id, timestamp)"
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_ticks_venue_date ON ticks (venue, date_)")
        con.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_ticks_market_source "
            "ON ticks (market_id, source_id)"
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        con.close()
        raise

    remaining = con.execute("SELECT count(*) FROM ticks").fetchone()[0]
    con.close()
    assert remaining == kept, f"expected {kept} rows, found {remaining}"
    print(f"  Removed {total - remaining:,} rows. {remaining:,} remain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
