Architecture
============

The system is organised in six horizontal layers.  Each layer has a single
responsibility and depends only on the layers below it.

.. code-block:: text

   ┌──────────────────────────────────────────────────────────┐
   │  Dashboard  (Streamlit)    ·   run_backtest.py            │
   ├──────────────────────────────────────────────────────────┤
   │  Strategies   market_making/ ·  arbitrage/                │
   ├──────────────────────────────────────────────────────────┤
   │  Execution   router · paper · live · risk                 │
   ├──────────────────────────────────────────────────────────┤
   │  Features    microstructure · resolution · signals        │
   │  Models      vol · calibration · signals                  │
   ├──────────────────────────────────────────────────────────┤
   │  Storage     writer · reader · archiver  (DuckDB)         │
   ├──────────────────────────────────────────────────────────┤
   │  Normalizer  schema · kalshi_adapter · polymarket_adapter │
   ├──────────────────────────────────────────────────────────┤
   │  Connectors  kalshi · polymarket · manifold               │
   └──────────────────────────────────────────────────────────┘

Signal and quoting pipeline
----------------------------

.. code-block:: text

   MarketDataReader
         │
         ├── features/microstructure.py  ──►  OBI, σ_B(p,τ), EWMA vol
         ├── features/resolution.py      ──►  τ, γ_eff, Q_max_eff, regime
         └── features/signals/           ──►  news, on-chain, ensemble μ̂
                       │
                       ▼
         strategies/market_making/params.py
               κ, A  (GLFT)  ·  φ, η, ρ, wᵢ  (Cartea-Jaimungal)
                       │
             ┌─────────┴──────────┐
             ▼                    ▼
          glft.py          cartea_jaimungal.py
          rᵃ, rᵇ           rᵃ, rᵇ + signal μ̂
             └─────────┬──────────┘
                       ▼
             execution/router.py
                  send orders

Layer descriptions
------------------

Connectors
~~~~~~~~~~

``connectors/`` contains one module per venue.  Each connector is an
``asyncio``-based client that:

1. Authenticates with the venue REST API (API key or EIP-712 signature).
2. Opens a WebSocket connection and streams market events.
3. Emits raw Python dicts — no domain objects yet.

All connectors inherit from :class:`connectors.base.BaseConnector` which
enforces the ``connect() / stream() / disconnect()`` interface.

See :doc:`api/connectors`.

Normalizer
~~~~~~~~~~

``normalizer/`` maps raw venue responses to the canonical domain objects
defined in :mod:`normalizer.schema`:

* :class:`~normalizer.schema.Market` — contract metadata (id, question,
  category, status, resolution date).
* :class:`~normalizer.schema.OrderBook` — best bid/ask, full depth, spread.
* :class:`~normalizer.schema.Tick` — atomic price event (quote or trade).

All ``market_id`` values follow the ``venue:raw_id`` format
(e.g. ``kalshi:KXBTC-26APR2212-T85799.99``) to guarantee global uniqueness.

See :doc:`api/normalizer`.

Storage
~~~~~~~

``storage/`` wraps a DuckDB embedded database.  Three tables are maintained:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Table
     - Description
   * - ``markets``
     - One row per market per venue.  Updated on status change / resolution.
   * - ``ticks``
     - One row per WebSocket event.  The largest table — millions of rows
       per day in production.
   * - ``orderbooks``
     - Full order-book snapshots fetched every N seconds.
   * - ``features``
     - Pre-computed microstructure features, one row per tick.

:class:`~storage.writer.MarketDataWriter` handles all inserts.
:class:`~storage.reader.MarketDataReader` exposes three query tiers:
*operational* (bounded, hot-path), *analytical* (full DataFrames for
notebooks), and *backtesting* (chunked iteration over long histories).

See :doc:`api/storage`.

Features and Models
~~~~~~~~~~~~~~~~~~~

``features/`` computes market microstructure signals from raw ticks and order
books:

* :mod:`~features.microstructure` — Order Book Imbalance (OBI), quoted spread,
  Bernoulli volatility :math:`\sigma_B(p, \tau)`, EWMA vol, :math:`\hat{\mu}`.
* :mod:`~features.resolution` — time-to-resolution :math:`\tau` in years,
  near-resolution regime classification (``NORMAL / WARNING / CRITICAL / HALT``).
* :mod:`~features.signals` — news sentiment, on-chain flow, and an ensemble
  signal model that produces the directional estimate :math:`\hat{\mu}_t`.

``models/`` contains pure statistical / mathematical objects that are fitted
offline and then used at runtime:

* :mod:`~models.vol` — EWMA volatility estimator and the Bernoulli volatility
  surface :math:`\sigma_B(p, \tau)`.
* :mod:`~models.calibration` — isotonic regression calibrator, Brier score,
  and a bias detector for model drift.
* :mod:`~models.signals` — base signal interface, news and on-chain signal
  models, and the LightGBM ensemble.

See :doc:`api/features` and :doc:`api/models`.

Strategies
~~~~~~~~~~

``strategies/market_making/`` contains the two quoting models:

* :class:`~strategies.market_making.glft.GLFTQuoter` — Avellaneda-Stoikov
  approximation in logit space (MATH.md §4.1).  Operates entirely on
  :math:`X_t = \text{logit}(p_t)`.  Outputs a :class:`~strategies.market_making.glft.Quote`
  with bid/ask prices, half-spreads, and the reservation log-odds.
* :class:`~strategies.market_making.cartea_jaimungal.CarteaJaimungalQuoter` —
  extends GLFT with a latent directional drift :math:`\hat{\mu}_t` (MATH.md §4.5).

``strategies/arbitrage/`` contains the cross-venue arbitrage engine:

* :class:`~strategies.arbitrage.detector.ArbitrageDetector` — detects when
  the same event is mispriced across Kalshi and Polymarket (after fees, basis,
  and capital cost).
* :class:`~strategies.arbitrage.sizing.ArbitrageSizer` — Kelly-optimal
  position sizing.
* :class:`~strategies.arbitrage.cross_venue.CrossVenueArbitrage` — end-to-end
  orchestration.

See :doc:`api/strategies`.

Execution
~~~~~~~~~

``execution/`` handles order lifecycle management:

* :class:`~execution.router.OrderRouter` — converts a :class:`~strategies.market_making.glft.Quote`
  to orders, checks the circuit breaker, validates risk limits, and routes to
  the appropriate engine.
* :mod:`execution.paper` — a simulated fill engine for backtesting and
  paper trading (probabilistic fills, slippage, fees).
* :mod:`execution.live` — real order submission to Kalshi and Polymarket REST
  APIs.
* :mod:`execution.risk` — three independent safeguards:

  * :class:`~execution.risk.limits.RiskLimitsChecker` — per-market and
    global inventory limits.
  * :class:`~execution.risk.circuit_breaker.CircuitBreaker` — halts quoting
    near resolution or after a daily loss threshold is breached.
  * :class:`~execution.risk.monitor.RiskMonitor` — continuous P&L tracking
    and alerting.

See :doc:`api/execution`.

Backtesting
~~~~~~~~~~~

``backtesting/`` provides a historical simulation engine that replays ticks
from DuckDB through the full signal → quoting → execution stack:

* :class:`~backtesting.engine.BacktestEngine` — loads ticks and features,
  performs ``merge_asof`` alignment, and iterates step-by-step reusing the
  paper trading and risk management classes.
* :mod:`~backtesting.metrics` — Sharpe ratio, Calmar, fill rate, inventory
  statistics, and near-resolution analysis.
* ``backtesting/scenarios/`` — synthetic market scenarios for stress testing
  (e.g. :class:`~backtesting.scenarios.resolution_spike.ResolutionSpikeScenario`).

See :doc:`api/backtesting`.

Dashboard
~~~~~~~~~

``dashboard/`` is a Streamlit multi-page app with three views:

* **Markets** — live order book visualisation and mid-price chart.
* **PnL** — cumulative P&L, inventory over time, daily Sharpe.
* **Calibration** — Brier score trend, isotonic calibration curve, bias
  detector alerts.

See :doc:`api/dashboard`.

Data flow diagram
-----------------

.. code-block:: text

   WebSocket feed
       │
       ▼
   Connector (venue-specific)
       │  raw dict
       ▼
   Normalizer Adapter
       │  Market / Tick / OrderBook
       ▼
   Storage Writer ──────────────────► DuckDB
       │                               │
       │                               │ ticks / orderbooks
       ▼                               ▼
   Feature Store ◄──────── Storage Reader
       │  features row
       ▼
   Strategy Quoter (GLFT / CJ)
       │  Quote
       ▼
   Order Router
       │  Orders
       ├──► Paper Engine (backtest / paper trading)
       └──► Live Executor (Kalshi / Polymarket REST API)
