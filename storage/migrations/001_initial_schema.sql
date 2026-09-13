-- storage/migrations/001_initial_schema.sql
-- Initial DuckDB schema for the prediction market system.
--
-- Why GENERATED ALWAYS AS columns:
--   mid, spread and date_ are always derived from other fields.
--   Computing them in Python and inserting them risks inconsistency
--   (a Python bug yields a wrong mid).
--   With GENERATED, the database computes them and guarantees consistency.
--   The writer omits them from the INSERT — including them raises an error.
--
-- Why split on semicolons and execute statement by statement:
--   DuckDB does not support several statements with GENERATED columns in a
--   single execute(). The writer splits them and runs them one at a time.


-- ---------------------------------------------------------------------------
-- markets — metadata for each binary contract
-- One row per market_id. Upserted when the status changes.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS markets (
    -- Canonical cross-venue identity: "kalshi:KXBTC-..." or "polymarket:0xabc..."
    market_id       VARCHAR     NOT NULL,
    venue           VARCHAR     NOT NULL,   -- "kalshi" | "polymarket"
    raw_id          VARCHAR     NOT NULL,   -- id nativo de la venue

    -- Static metadata — unchanged after creation
    question        VARCHAR     NOT NULL,
    category        VARCHAR     NOT NULL,   -- "crypto" | "politics" | ...

    -- Fields that change over the market's lifecycle
    status          VARCHAR     NOT NULL,   -- "open" | "closed" | "resolved"
    resolution_date TIMESTAMPTZ NOT NULL,
    resolved_value  DOUBLE,                 -- NULL until resolved, then 0.0 or 1.0

    -- Audit
    first_seen_at   TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ,

    PRIMARY KEY (market_id)
);


-- ---------------------------------------------------------------------------
-- ticks — atomic price events (quotes and trades)
-- The largest table — it grows with every WebSocket message.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ticks (
    market_id       VARCHAR     NOT NULL,
    venue           VARCHAR     NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,

    -- Tipo de evento
    tick_type       VARCHAR     NOT NULL CHECK (tick_type IN ('quote','trade')),

    -- Prices are always probabilities in [0, 1]
    yes_bid         DOUBLE      NOT NULL,
    yes_ask         DOUBLE      NOT NULL,

    -- Derived columns — DuckDB computes them, the writer does not insert them
    mid             DOUBLE      GENERATED ALWAYS AS ((yes_bid + yes_ask) / 2.0),
    spread          DOUBLE      GENERATED ALWAYS AS (yes_ask - yes_bid),

    -- Campos de trade (NULL en quotes)
    volume          DOUBLE      NOT NULL DEFAULT 0.0,
    side            VARCHAR,               -- "yes" | "no" | NULL for quotes

    -- The venue's native event id. The deduplication key (see 002).
    source_id       VARCHAR,

    -- Partition column for archiving to Parquet by day
    date_           DATE        GENERATED ALWAYS AS (CAST(timestamp AS DATE))
);

-- Index for the reader's most frequent queries:
--   latest_ticks(market_id, n)  →  WHERE market_id = ? ORDER BY timestamp DESC
--   ticks(market_id, start, end) → WHERE market_id = ? AND timestamp BETWEEN
CREATE INDEX IF NOT EXISTS idx_ticks_market_ts
    ON ticks (market_id, timestamp);

-- Index for the archiver, which processes by venue and date
CREATE INDEX IF NOT EXISTS idx_ticks_venue_date
    ON ticks (venue, date_);


-- ---------------------------------------------------------------------------
-- orderbooks — order book snapshots
-- Less frequent than ticks — one row per full-book fetch.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS orderbooks (
    market_id       VARCHAR     NOT NULL,
    venue           VARCHAR     NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,

    -- Top of book, for fast queries with no JSON parsing
    best_bid        DOUBLE,
    best_ask        DOUBLE,

    -- Columns derived from the top of book
    mid             DOUBLE      GENERATED ALWAYS AS (
                        CASE
                            WHEN best_bid IS NOT NULL AND best_ask IS NOT NULL
                            THEN (best_bid + best_ask) / 2.0
                            ELSE NULL
                        END
                    ),
    spread          DOUBLE      GENERATED ALWAYS AS (
                        CASE
                            WHEN best_bid IS NOT NULL AND best_ask IS NOT NULL
                            THEN best_ask - best_bid
                            ELSE NULL
                        END
                    ),

    -- Libro completo serializado como [[price, size], ...]
    -- Needed to reconstruct the book in backtesting
    bids_json       JSON        NOT NULL,
    asks_json       JSON        NOT NULL,

    -- Depths precomputed for the feature store
    -- They avoid parsing JSON on every OBI query
    bid_depth_5     DOUBLE,
    ask_depth_5     DOUBLE,

    -- Daily partition for archival
    date_           DATE        GENERATED ALWAYS AS (CAST(timestamp AS DATE))
);

CREATE INDEX IF NOT EXISTS idx_orderbooks_market_ts
    ON orderbooks (market_id, timestamp);


-- ---------------------------------------------------------------------------
-- features — precomputed microstructure signals
-- One row per tick. Written by features/store.py, read by GLFT.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS features (
    market_id       VARCHAR     NOT NULL,
    venue           VARCHAR     NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,

    -- Order Book Imbalance ∈ [-1, 1]
    -- The w_1 signal in MATH.md's μ̂_t
    obi             DOUBLE,

    -- Spread absoluto y relativo al mid
    quoted_spread   DOUBLE,
    relative_spread DOUBLE,

    -- Logit-space volatility σ_b from the quadratic variation of X = logit(p)
    -- Calibrated empirically — §3 of MATH.md v2.1
    belief_vol      DOUBLE,

    -- Empirical EWMA volatility — complements σ_B
    -- If ewma_vol >> bernoulli_vol → a jump regime
    ewma_vol        DOUBLE,

    -- Time to resolution, in years
    -- τ = (resolution_date - now) / 365.25
    tau_years       DOUBLE,

    -- Estimated signal μ̂_t = w_1·OBI + w_2·News + w_3·OnChain
    -- Until the full signal is calibrated, OBI is used as a proxy
    mu_hat          DOUBLE,

    -- Daily partition
    date_           DATE        GENERATED ALWAYS AS (CAST(timestamp AS DATE))
);

CREATE INDEX IF NOT EXISTS idx_features_market_ts
    ON features (market_id, timestamp);
