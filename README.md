# Prediction Market Making — Kalshi & Polymarket

A quantitative market-making system for binary event contracts, operating on
Kalshi and Polymarket simultaneously. It quotes two-sided markets, controls
inventory through a stochastic-control model, skews on a short-horizon
directional signal, and is designed to capture cross-venue arbitrage when the
same event is mispriced across platforms.

Each model layer is motivated by the failure of the one before it. The
derivations live in **[MATH.md](MATH.md)** — that document, not this one, is
where the substance is.

```
Python 3.13  ·  10,800 LOC (excl. tests)  ·  364 tests  ·  mypy strict on the core
```

---

## 1. The trading thesis

In a prediction market the price **is** a probability: bounded in (0,1), with a
known resolution date at which the contract jumps to 0 or 1. That breaks the
classical market-making models in two places — the price process is not an
unbounded Brownian motion, and volatility is not stationary near expiry.

The design response is to work in **log-odds space**, `X = logit(p)`. There the
state is unbounded, the standard Avellaneda-Stoikov / GLFT / Cartea-Jaimungal
machinery applies with its proofs intact, and mapping back through the sigmoid
compresses the quoted spread near the boundaries automatically — no ad-hoc
clamping required.

### Model hierarchy

| § | Model | Contribution | Why it is not enough |
|---|---|---|---|
| 1 | Glosten-Milgrom | Spread exists because of adverse selection | Descriptive, not prescriptive |
| 2 | Avellaneda-Stoikov | CARA utility + HJB → inventory skew | Order flow only implicit |
| 3 | GLFT | Explicit intensity `λ = A·e^{−κδ}`, κ calibrable | No directional view |
| 4 | Cartea-Jaimungal | OU latent drift + signal μ̂ → skews the quote centre | Gaussian, no jumps |
| 5 | Logit jump-diffusion | Belief-volatility surface σ_b on X | — |
| 6 | Near-resolution | Regimes on τ: γ_eff, q_max, halt | — |
| 7 | Kelly cross-venue | YES/NO arb sizing with frictions | — |
| 8 | Brier + recalibration | 5-term decomposition, isotonic / Venn-Abers | — |

The quoting equations actually implemented (MATH.md §5.4, v2.2):

```
γ_eff        = γ_I · regime_multiplier(τ)              # §6.4
reservation  = X_t − q·γ_eff·σ̄²_b·τ  +  φ₁(τ)·μ̂_t      # §4.5
half_spread  = γ_eff·σ̄²_b·τ/2  +  (1/γ_eff)·ln(1 + γ_eff/κ_x)
bid, ask     = σ(reservation ∓ half_spread), snapped to the venue price ladder
```

---

## 2. Architecture

```
Kalshi REST/WS   Polymarket CLOB   Manifold
      │                │               │
      ▼                ▼               ▼
  connectors/     async clients, exponential backoff, market discovery,
                  periodic metadata refresh (this is what observes resolution)
      │
      ▼
  normalizer/     venue payloads → canonical domain objects
                  Market · OrderBook · Tick · PriceLadder
                  market_id = "venue:raw_id"
      │
      ▼
  storage/        DuckDB, batched async writer, versioned migrations,
                  idempotent inserts keyed on the venue-native event id
      │
      ▼
  features/       OBI · σ_b(logit) · EWMA · τ/regime · μ̂ · Brier decomposition
      │
      ▼
  strategies/     params.py  → calibrates κ, A, φ, η, ρ_μ, w_i
                  glft.py    → r_a, r_b
                  cartea_jaimungal.py → r_a, r_b with signal skew
      │
      ▼
  execution/      router → paper account/engine → risk (limits, breaker, monitor)
      │
      ▼
  backtesting/    historical engine, metrics, near-resolution scenario
```

### Canonical schema

| Object | Contents |
|---|---|
| `Market` | id, venue, question, category, status, resolution, **price ladder** |
| `OrderBook` | best bid/ask, mid, spread, full depth levels, depth aggregates |
| `Tick` | timestamp, type (quote/trade), yes bid/ask, volume, side, **source_id** |
| `PriceLadder` | quotable price grid: per-band tick size, read from venue metadata |

`source_id` is the venue-native event identifier and the deduplication key.
Without it a retry is indistinguishable from two genuine trades in the same
millisecond — and those are the common case, not the exception.

`PriceLadder` exists because a single global tick size is wrong on both venues:
Polymarket declares `minimum_tick_size = 0.001`, and Kalshi publishes a
per-market ladder stepping 0.001 in the main band and 0.0001 in the tails.

---

## 3. Where the parameters come from

A recurring question. **OBI and σ_b are features** — computed continuously from
market data. **κ, A, φ, η, ρ_μ and the signal weights are hyperparameters** —
they require an explicit calibration run.

| | Computed in | Input | Method |
|---|---|---|---|
| **OBI** | `features/microstructure.py` | `orderbooks.bid_depth_5 / ask_depth_5` | `(V_bid − V_ask)/(V_bid + V_ask)`. Instantaneous. |
| **σ_b** | `features/microstructure.py` | series of `ticks.mid` | Realised quadratic variation of `X = logit(p)` on a **fixed 60 s grid**, RiskMetrics EWMA (λ=0.94), winsorised at 99.5%, bounded. |
| **κ, A** | `strategies/market_making/params.py` | `ticks.spread` + fill proxy | Poisson MLE on `λ = A·e^{−κδ}`, then `κ_x = κ_p·p̄(1−p̄)`. |
| **φ, η, ρ_μ, w_i** | `strategies/market_making/params.py` | signal series + forward returns | AR(1) on the OU drift, ridge regression for the weights. |

The 60 s sampling grid is not a detail. Realised variance **diverges as Δt → 0**
under microstructure noise, and estimating tick-by-tick produced σ_b values four
orders of magnitude too large — enough to overflow the sigmoid and abort the
backtester. Fixed-interval subsampling is the standard first-order defence
(Zhang–Mykland–Aït-Sahalia 2005).

### Data required to calibrate

| Quantity | Minimum | Practical | Binding constraint |
|---|---|---|---|
| OBI | 1 order book | immediate | — |
| σ_b | 30–50 grid points | ~30–50 min per market | EWMA half-life is ~11 observations at λ=0.94 |
| κ, A | `MIN_TICKS_GLFT = 100` | thousands of ticks, 1–2 weeks | Needs **spread variation** and observed fills. A constant spread makes κ unidentifiable and pins the optimiser to its bounds. |
| φ, η, ρ_μ, w_i | `MIN_TICKS_CJ = 200` | + out-of-sample validation | Requires resolved markets |
| Brier | 20 observations | **300–500 resolved markets** | Standard errors; this is the long pole (2–3 weeks of ingestion) |

---

## 4. Running it

### Ingestion

```bash
.venv/bin/python main.py                       # foreground, Ctrl+C to stop

nohup .venv/bin/python main.py > ingest.log 2>&1 &   # detached
echo $! > ingest.pid
kill $(cat ingest.pid)                        # graceful: final flush, then exit
```

All three venues are enabled by default and **none requires credentials to
read**. Kalshi serves markets, order books and trades unauthenticated from
`api.elections.kalshi.com`; Polymarket's `/book` is public too. Credentials are
only needed for the Kalshi WebSocket and for order submission (Phase 4) — with
no key the Kalshi connector falls back to REST polling.

### Ingestion quality controls

Configured in `.env`, enforced at the source so rejected data never costs a row:

| Setting | Default | Purpose |
|---|---|---|
| `MAX_TAU_DAYS` | 45 | Drop markets resolving beyond the observation horizon. A contract settling in 74 years can never validate anything. |
| `REQUIRE_TWO_SIDED_BOOK` | true | No bid *and* ask means no mid, no spread, no OBI |
| `BACKFILL_MAX_TICKS` | 200 | Bounds historical backfill per market |
| `MAX_DB_SIZE_MB` | 2048 | Writer stops and logs once exceeded, rather than filling the disk silently |
| `MARKET_REFRESH_SECONDS` | 900 | Metadata refresh — **this is the only path by which a resolution is recorded** |

### Monitoring

```bash
.venv/bin/python scripts/health_check.py       # detailed status
*/10 * * * * cd <repo> && .venv/bin/python scripts/health_check.py --quiet >> alertas.log 2>&1
```

Exit codes: `0` healthy, `1` warnings, `2` critical. Eight checks — process
liveness, heartbeat freshness, stream progress, resolution capture, database
size, disk space, log error patterns, silent venues.

The health check reads a JSON heartbeat published by the ingestion process
rather than querying the database, because **DuckDB grants the writer an
exclusive lock**: no external process can read the file while ingestion runs.

### Backtesting and calibration

```bash
.venv/bin/python run_backtest.py               # interactive
.venv/bin/python run_calibration.py            # κ, A, φ, η, ρ_μ, weights
.venv/bin/python scripts/dedup_ticks.py        # dry run; --apply to execute
```

### Quality gates

```bash
.venv/bin/python -m pytest -m "not live"       # 356 offline tests
.venv/bin/python -m pytest -m live             # 8 tests against real venue APIs
.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy features strategies execution normalizer backtesting storage
```

`mypy --strict` is clean across the modules that decide what gets quoted, at
what risk, and what gets persisted. `connectors/` and `main.py` carry known,
bounded type debt and are checked non-blocking in CI.

---

## 5. Status

**Working end to end:** ingestion from all three venues with real order-book
depth, canonical normalisation, DuckDB persistence with idempotent writes,
microstructure features with non-zero OBI, GLFT and Cartea-Jaimungal quoters
with near-resolution regimes applied, paper execution with pre-trade risk
checks, and a historical backtester.

**Not implemented.** These are stubbed with `NotImplementedError` and a pointer
to the relevant MATH.md section, not silently missing:

- Cross-venue arbitrage (`strategies/arbitrage/`, MATH.md §7) — needs
  simultaneous two-venue ingestion with semantic market matching
- Live execution (`execution/live/`) — Phase 4
- Parquet archival (`storage/archiver.py`)

**Known limitations, stated plainly:**

- The backtester still fills a quote against the tick that generated it and
  forward-fills features with `bfill`. Both are look-ahead. Until that is fixed
  its PnL is not evidence of anything.
- Positions are marked to mid at the end of a backtest; there is no settlement
  at resolution, which in a binary contract is the dominant PnL term.
- `config/settings.py` persists calibrated parameters but no consumer reads
  them back — the quoters currently use documented placeholder defaults.
- Manifold is an AMM with no real order book; its connector synthesises a fixed
  spread. It is useful for pipeline testing, not for microstructure calibration.

---

## 6. Layout

```
connectors/    async venue clients + discovery + metadata refresh
normalizer/    payload → domain objects, price ladders
storage/       DuckDB writer/reader, migrations
features/      microstructure, resolution regimes, signals, calibration
strategies/    parameter estimation + GLFT / Cartea-Jaimungal quoters
execution/     order router, paper trading, risk management
backtesting/   historical engine, metrics, scenarios
scripts/       health check, tick deduplication
tests/         364 tests; network tests marked `live`
MATH.md        model derivations (v2.2)
```
