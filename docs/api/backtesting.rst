backtesting
===========

Historical simulation engine.  Replays DuckDB ticks through the full
signal → quoting → execution stack without any code duplication — the same
:class:`~execution.paper.engine.PaperExecutionEngine` and
:class:`~execution.risk.circuit_breaker.CircuitBreaker` used in live trading
are reused here.

Engine
------

The backtest engine:

1. Loads ticks and pre-computed features from DuckDB for the requested market
   and date range.
2. Aligns ticks and features with ``merge_asof`` (features may arrive slightly
   after the corresponding tick).
3. Iterates step-by-step, feeding each tick into the quoter and routing the
   resulting quote through :class:`~execution.router.OrderRouter`.
4. Returns a :class:`~backtesting.metrics.BacktestResult` with full trade log
   and summary statistics.

.. automodule:: backtesting.engine
   :members:
   :undoc-members:
   :show-inheritance:

Metrics
-------

Computes summary statistics from the trade log and equity curve:

* **Sharpe ratio** (annualised, assuming 365-day year).
* **Calmar ratio** — annualised return / maximum drawdown.
* **Fill rate** — fraction of quoted orders that were filled.
* **Inventory statistics** — mean, max, and time-in-limit-breach.
* **Near-resolution analysis** — P&L breakdown by regime.

.. automodule:: backtesting.metrics
   :members:
   :undoc-members:
   :show-inheritance:

Scenarios
---------

Synthetic market scenarios for stress testing.  Each scenario generates a
synthetic tick sequence with known statistical properties, allowing
controlled experiments independently of historical data availability.

Base scenario
~~~~~~~~~~~~~

.. automodule:: backtesting.scenarios.base_scenario
   :members:
   :undoc-members:
   :show-inheritance:

Resolution spike scenario
~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulates the jump in mid-price that typically occurs in the final minutes
before a contract resolves.  Used to validate the circuit breaker and
near-resolution risk controls.

.. automodule:: backtesting.scenarios.resolution_spike
   :members:
   :undoc-members:
   :show-inheritance:
