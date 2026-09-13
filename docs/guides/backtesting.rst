Backtesting
===========

This guide explains how to run historical simulations using
:class:`~backtesting.engine.BacktestEngine`, interpret results, and run
parameter sweeps.

Interactive CLI
---------------

The quickest way to run a backtest is the interactive CLI:

.. code-block:: bash

   uv run python run_backtest.py

The script prompts for:

* **Strategy** — ``glft`` or ``cj``
* **Market ID** — e.g. ``kalshi:KXBTC-26APR2212-T85799.99``
* **Date range** — ``YYYY-MM-DD`` format
* **Parameters** — ``gamma_I``, ``kappa_x``, etc.

Results are printed as a Rich table.  Pass ``--sweep`` for a grid search.

Programmatic API
----------------

.. code-block:: python

   from datetime import UTC, datetime
   from backtesting.engine import BacktestEngine

   engine = BacktestEngine(
       db_path="./data/duckdb/markets.duckdb",
       market_id="kalshi:KXBTC-26APR2212-T85799.99",
       strategy_name="glft",
       strategy_params={
           "gamma_I": 0.1,
           "kappa_x": 1.5,
       },
       start=datetime(2026, 4, 1, tzinfo=UTC),
       end=datetime(2026, 4, 22, tzinfo=UTC),
       initial_cash=10_000.0,
   )

   result = engine.run()
   print(result.summary())

BacktestResult fields
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``pnl_series``
     - DataFrame with timestamp and cumulative P&L.
   * - ``trade_log``
     - DataFrame with every fill: price, size, side, market.
   * - ``sharpe``
     - Annualised Sharpe ratio of the daily P&L series.
   * - ``calmar``
     - Annualised return divided by maximum drawdown.
   * - ``fill_rate``
     - Fraction of quoted orders that were filled.
   * - ``max_inventory``
     - Peak absolute inventory across all markets.
   * - ``near_res_pnl``
     - P&L breakdown by near-resolution regime.

Parameter sweep
---------------

.. code-block:: python

   import numpy as np
   from backtesting.engine import BacktestEngine
   from backtesting.metrics import calculate_metrics

   results = []
   for gamma in np.linspace(0.05, 0.3, 6):
       for kappa in np.linspace(0.5, 3.0, 6):
           engine = BacktestEngine(
               db_path="./data/duckdb/markets.duckdb",
               market_id="kalshi:KXBTC-26APR2212-T85799.99",
               strategy_name="glft",
               strategy_params={"gamma_I": gamma, "kappa_x": kappa},
               start=datetime(2026, 4, 1, tzinfo=UTC),
               end=datetime(2026, 4, 22, tzinfo=UTC),
           )
           r = engine.run()
           results.append({"gamma_I": gamma, "kappa_x": kappa, "sharpe": r.sharpe})

   import pandas as pd
   df = pd.DataFrame(results).pivot("gamma_I", "kappa_x", "sharpe")
   print(df.to_string())

Synthetic scenarios
-------------------

Use the built-in scenarios to test behaviour without historical data:

.. code-block:: python

   from backtesting.scenarios.resolution_spike import ResolutionSpikeScenario

   scenario = ResolutionSpikeScenario(
       market_id="synthetic:test",
       n_ticks=5000,
       spike_at_tau=0.0005,   # ~4 minutes before resolution
       spike_magnitude=0.4,   # price jumps 40 cents
   )

   ticks = scenario.generate()
   # Pass ticks directly to BacktestEngine via the synthetic_ticks parameter

Near-resolution analysis
------------------------

The :mod:`~backtesting.metrics` module provides a regime-level P&L breakdown
to diagnose whether losses are concentrated near resolution:

.. code-block:: python

   from backtesting.metrics import calculate_metrics

   metrics = calculate_metrics(result.trade_log, result.pnl_series)
   print(metrics["near_resolution_breakdown"])
   # {'NORMAL': 142.3, 'WARNING': -18.5, 'CRITICAL': -55.1, 'HALT': 0.0}
