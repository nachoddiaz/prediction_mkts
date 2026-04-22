# Prediction Markets Market-Making System — Kalshi & Polymarket

A quantitative market-making system designed to operate simultaneously on Kalshi and Polymarket. The system quotes tight spreads, manages inventory risk, fades directional signals, and captures cross-venue arbitrage when the same event misprices across platforms. Each component is built layer by layer — motivated by the failure of the previous model, from the microstructural reason spreads exist to near-resolution jump risk.

---

## Ingestion Layer

### Overview

The ingestion layer is responsible for fetching raw data from both venues and normalizing it into a unified set of domain objects. It abstracts away the structural and semantic differences between Kalshi and Polymarket so that all downstream components (features, model, execution) operate on a single canonical schema.

### `normalizer/kalshi_adapter.py`

Transforms raw responses from the Kalshi REST API and WebSocket feed into domain objects defined in `schema.py`.

Key conventions handled:
- Prices arrive as whole cents (`45` → `0.45` probability)
- Order book levels are lists of `[price, quantity]` pairs
- Resolution dates come as ISO strings in UTC
- Market status transitions (`open` → `resolved`) are detected and propagated

### `normalizer/polymarket_adapter.py`

Transforms raw responses from two separate Polymarket APIs into the same domain objects:

- **Gamma API** — provides market metadata: title, category, settlement date, status, and estimated prices
- **CLOB API** — provides real-time order book data and trade history

Key conventions handled:
- Timestamps arrive as either Unix milliseconds (`int`) or ISO strings — both are normalized to `datetime` UTC
- Prices are USDC fractional amounts used directly as probabilities
- YES/NO token structure is mapped to a unified bid/ask representation

### Unified Schema (`schema.py`)

Both adapters output the same domain objects:

| Object | Description |
|---|---|
| `Market` | Metadata for a single contract: id, venue, question, category, status, resolution |
| `OrderBook` | Best bid/ask, mid, spread, full depth levels, depth aggregates |
| `Tick` | Atomic price event: timestamp, type (quote/trade), yes bid/ask, volume, side |

All `market_id` values follow the format `venue:raw_id` (e.g., `kalshi:KXBTC-26APR2212-T85799.99`) to ensure global uniqueness across venues.

---

## Data Storage Schema

### `markets` — Contract Metadata

One row per market per venue. Updated when status changes or the contract resolves.

| Field | Example | Notes |
|---|---|---|
| `market_id` | `kalshi:KXBTC-26APR2212-T85799.99` | Global unique id |
| `venue` | `kalshi` | |
| `question` | `Bitcoin price range on Apr 22?` | |
| `category` | `crypto` | |
| `status` | `open` → `resolved` | Updated on change |
| `resolution_date` | `2026-04-22T16:00:00Z` | |
| `resolved_value` | `NULL` → `1.0` | Updated on resolution |

**Write frequency:** low — once on market discovery, then only on status change.

---

### `ticks` — Atomic Price Events

One row per quote or trade arriving from the WebSocket. The largest table — can grow to millions of rows per day in production.

| Field | Example | Notes |
|---|---|---|
| `market_id` | `kalshi:KXBTC-26APR2212-T85799.99` | |
| `timestamp` | `2026-04-22T15:30:00Z` | |
| `tick_type` | `quote \| trade` | |
| `yes_bid` | `0.0100` | |
| `yes_ask` | `0.0200` | |
| `mid` | `0.0150` | Generated column (DuckDB) |
| `spread` | `0.0100` | Generated column (DuckDB) |
| `volume` | `0.0` | `0` for quotes, `> 0` for trades |
| `side` | `NULL \| yes \| no` | |

**Write frequency:** very high — every WebSocket update.

---

### `orderbooks` — Order Book Snapshots

One row per full order book fetch. Heavier than a tick but provides depth information.

| Field | Example | Notes |
|---|---|---|
| `market_id` | `kalshi:KXBTC-26APR2212-T85799.99` | |
| `timestamp` | `2026-04-22T15:30:00Z` | |
| `best_bid` | `0.0200` | |
| `best_ask` | `0.4400` | |
| `bids_json` | `[[0.02, 14039], [0.01, 8500]]` | All levels |
| `asks_json` | `[[0.44, 12000], [0.45, 5000]]` | All levels |
| `bid_depth_5` | `22539.0` | Sum of top 5 bid levels |
| `ask_depth_5` | `17000.0` | Sum of top 5 ask levels |

**Write frequency:** medium — every time the connector fetches the full book (every N seconds, not on every tick).

---

### `features` — Microstructure Signals

One row per tick, with features computed by `features/microstructure.py`. These are consumed directly by the GLFT model.

| Field | Example | Notes |
|---|---|---|
| `market_id` | `kalshi:KXBTC-26APR2212-T85799.99` | |
| `timestamp` | `2026-04-22T15:30:00Z` | |
| `obi` | `0.234` | Order Book Imbalance ∈ [-1, 1] |
| `quoted_spread` | `0.420` | |
| `relative_spread` | `28.0` | |
| `bernoulli_vol` | `0.0147` | σ_B(p, τ) — see MATH.md §5 |
| `ewma_vol` | `0.0089` | |
| `tau_years` | `0.0001` | Time to resolution in years |
| `mu_hat` | `0.012` | Estimated directional signal |

**Write frequency:** same as `ticks` — one feature row per tick.
