-- storage/migrations/002_tick_source_id.sql
-- ──────────────────────────────────────────
-- Tick deduplication by the venue's native identifier.
--
-- Problem: the `ticks` table had no uniqueness constraint, so every writer
-- retry and every connector re-poll reinserted the same rows. The development
-- database held 399 fully duplicated rows.
--
-- Why NOT a PK on (market_id, timestamp): two distinct trades can share a
-- timestamp to the millisecond — Manifold matches a `yes` and a `no` in the
-- same transaction. Of the 2264 groups sharing (market_id, timestamp), 1865
-- are legitimately distinct trades. A PK there would have discarded good data.
--
-- The correct key is the native event id. Quotes have none and carry NULL: in
-- SQL two NULLs do not collide, so the index lets them through.

ALTER TABLE ticks ADD COLUMN IF NOT EXISTS source_id VARCHAR;

CREATE UNIQUE INDEX IF NOT EXISTS ux_ticks_market_source
    ON ticks (market_id, source_id);
