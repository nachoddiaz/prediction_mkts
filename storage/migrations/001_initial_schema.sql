-- storage/migrations/001_initial_schema.sql
-- Schema inicial de DuckDB para el sistema de prediction markets.
--
-- Por qué columnas GENERATED ALWAYS AS:
--   mid, spread y date_ se derivan siempre de otros campos.
--   Si las calculamos en Python y las insertamos, corremos el riesgo
--   de inconsistencias (un bug en Python da un mid incorrecto).
--   Con GENERATED la base de datos las calcula y garantiza consistencia.
--   El writer no las incluye en el INSERT — si lo hiciera DuckDB lanzaría error.
--
-- Por qué separar por semicolon y ejecutar statement a statement:
--   DuckDB no soporta múltiples statements con columnas GENERATED
--   en un solo execute(). El writer los separa y ejecuta uno a uno.


-- ---------------------------------------------------------------------------
-- markets — metadatos de cada contrato binario
-- Una fila por market_id. Se actualiza (upsert) cuando cambia el status.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS markets (
    -- Identidad canónica cross-venue: "kalshi:KXBTC-..." o "polymarket:0xabc..."
    market_id       VARCHAR     NOT NULL,
    venue           VARCHAR     NOT NULL,   -- "kalshi" | "polymarket"
    raw_id          VARCHAR     NOT NULL,   -- id nativo de la venue

    -- Metadatos estáticos — no cambian después de la creación
    question        VARCHAR     NOT NULL,
    category        VARCHAR     NOT NULL,   -- "crypto" | "politics" | ...

    -- Campos que cambian con el ciclo de vida del mercado
    status          VARCHAR     NOT NULL,   -- "open" | "closed" | "resolved"
    resolution_date TIMESTAMPTZ NOT NULL,
    resolved_value  DOUBLE,                 -- NULL hasta resolver, luego 0.0 o 1.0

    -- Auditoría
    first_seen_at   TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ,

    PRIMARY KEY (market_id)
);


-- ---------------------------------------------------------------------------
-- ticks — eventos atómicos de precio (quotes y trades)
-- La tabla más grande — crece con cada mensaje del WebSocket.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ticks (
    market_id       VARCHAR     NOT NULL,
    venue           VARCHAR     NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,

    -- Tipo de evento
    tick_type       VARCHAR     NOT NULL,   -- "quote" | "trade"

    -- Precios siempre en probabilidad [0, 1]
    yes_bid         DOUBLE      NOT NULL,
    yes_ask         DOUBLE      NOT NULL,

    -- Columnas derivadas — DuckDB las calcula, el writer no las inserta
    mid             DOUBLE      GENERATED ALWAYS AS ((yes_bid + yes_ask) / 2.0),
    spread          DOUBLE      GENERATED ALWAYS AS (yes_ask - yes_bid),

    -- Campos de trade (NULL en quotes)
    volume          DOUBLE      NOT NULL DEFAULT 0.0,
    side            VARCHAR,               -- "yes" | "no" | NULL para quotes

    -- Columna de partición para archivar a Parquet por día
    date_           DATE        GENERATED ALWAYS AS (CAST(timestamp AS DATE))
);

-- Índice para las queries más frecuentes del reader:
--   latest_ticks(market_id, n)  →  WHERE market_id = ? ORDER BY timestamp DESC
--   ticks(market_id, start, end) → WHERE market_id = ? AND timestamp BETWEEN
CREATE INDEX IF NOT EXISTS idx_ticks_market_ts
    ON ticks (market_id, timestamp);

-- Índice para el archiver que procesa por venue y fecha
CREATE INDEX IF NOT EXISTS idx_ticks_venue_date
    ON ticks (venue, date_);


-- ---------------------------------------------------------------------------
-- orderbooks — snapshots del libro de órdenes
-- Menos frecuente que ticks — una fila por fetch del libro completo.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS orderbooks (
    market_id       VARCHAR     NOT NULL,
    venue           VARCHAR     NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,

    -- Top of book para queries rápidas sin parsear JSON
    best_bid        DOUBLE,
    best_ask        DOUBLE,

    -- Columnas derivadas del top of book
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
    -- Necesario para reconstruir el libro en backtesting
    bids_json       JSON        NOT NULL,
    asks_json       JSON        NOT NULL,

    -- Profundidades pre-computadas para el feature store
    -- Evitan parsear JSON en cada query de OBI
    bid_depth_5     DOUBLE,
    ask_depth_5     DOUBLE,

    -- Partición diaria para archivado
    date_           DATE        GENERATED ALWAYS AS (CAST(timestamp AS DATE))
);

CREATE INDEX IF NOT EXISTS idx_orderbooks_market_ts
    ON orderbooks (market_id, timestamp);


-- ---------------------------------------------------------------------------
-- features — señales de microestructura pre-computadas
-- Una fila por tick. Escritas por features/store.py, leídas por GLFT.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS features (
    market_id       VARCHAR     NOT NULL,
    venue           VARCHAR     NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,

    -- Order Book Imbalance ∈ [-1, 1]
    -- Señal w_1 en μ̂_t del MATH.md
    obi             DOUBLE,

    -- Spread absoluto y relativo al mid
    quoted_spread   DOUBLE,
    relative_spread DOUBLE,

    -- Volatilidad de Bernoulli σ_B(p, τ) = sqrt(p(1-p)/τ)
    -- Endógena — no necesita calibración
    bernoulli_vol   DOUBLE,

    -- Volatilidad EWMA empírica — complementa σ_B
    -- Si ewma_vol >> bernoulli_vol → régimen de jump
    ewma_vol        DOUBLE,

    -- Tiempo hasta resolución en años
    -- τ = (resolution_date - now) / 365.25
    tau_years       DOUBLE,

    -- Señal estimada μ̂_t = w_1·OBI + w_2·News + w_3·OnChain
    -- Hasta calibrar la señal completa, OBI se usa como proxy
    mu_hat          DOUBLE,

    -- Partición diaria
    date_           DATE        GENERATED ALWAYS AS (CAST(timestamp AS DATE))
);

CREATE INDEX IF NOT EXISTS idx_features_market_ts
    ON features (market_id, timestamp);
