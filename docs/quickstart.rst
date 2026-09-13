Quick Start
===========

This page walks through the four most common workflows in under five minutes
each.  All examples assume you have completed :doc:`installation` and are
inside the project root with the virtual environment activated.

1. Run the data ingestion loop
------------------------------

The ingestion loop connects to the enabled venues, normalises the WebSocket
feed, and writes ticks, order books, and features to DuckDB continuously.

.. code-block:: bash

   # Manifold only (sandbox — no credentials required)
   uv run python main.py

   # Kalshi demo + Manifold
   KALSHI_API_KEY=xxx KALSHI_PRIVATE_KEY_PATH=./secrets/kalshi.pem uv run python main.py

   # All venues
   ENABLE_POLYMARKET=true uv run python main.py

Press ``Ctrl+C`` to stop gracefully.  The DuckDB file is written to
``./data/duckdb/markets.duckdb`` by default.

2. Run a paper-trading backtest
--------------------------------

``run_backtest.py`` is an interactive CLI that lets you choose the strategy,
market, date range, and parameters without editing code.

.. code-block:: bash

   uv run python run_backtest.py

The script prompts you for:

* **Strategy** — ``glft`` (GLFT optimal quoter) or ``cj`` (Cartea-Jaimungal with directional signal)
* **Market ID** — e.g. ``kalshi:KXBTC-26APR2212-T85799.99``
* **Date range** — start / end datetime in ``YYYY-MM-DD`` format
* **Strategy parameters** — ``gamma_I``, ``kappa_x``, etc. (defaults provided)

Results are printed as a Rich table with PnL, Sharpe, fill rate, and
inventory statistics.  Pass ``--sweep`` to run a grid search over
``gamma_I × kappa_x``.

3. Calibrate model parameters
------------------------------

``run_calibration.py`` fits ``kappa`` (fill-arrival decay) and ``A``
(arrival rate at zero spread) from historical ticks using maximum likelihood,
and optionally re-trains the ensemble signal model.

.. code-block:: bash

   uv run python run_calibration.py --market kalshi:KXBTC-26APR2212-T85799.99

The calibrated parameters are written back to the venue YAML config in
``config/``.

4. Launch the dashboard
------------------------

The Streamlit dashboard provides real-time monitoring of active markets,
PnL, and calibration diagnostics.

.. code-block:: bash

   # Requires: uv sync --extra dashboard
   uv run streamlit run dashboard/app.py

Open ``http://localhost:8501`` in your browser.  The three pages are:

* **Markets** — live order book, mid-price chart, OBI heatmap
* **PnL** — cumulative PnL, inventory, daily Sharpe
* **Calibration** — Brier score over time, isotonic curve, bias detector

Next steps
----------

* Read :doc:`architecture` for a full description of the data flow and
  module responsibilities.
* Read :doc:`math` for the mathematical derivations behind each model.
* Check the :doc:`guides/market_making` how-to for a step-by-step walkthrough
  of GLFT and Cartea-Jaimungal quoting.
