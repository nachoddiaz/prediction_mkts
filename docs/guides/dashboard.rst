Dashboard
=========

The Streamlit dashboard provides real-time monitoring of active markets,
live P&L, and calibration diagnostics.

Installation
------------

The dashboard requires the optional ``dashboard`` dependency group:

.. code-block:: bash

   uv sync --extra dashboard

Starting the dashboard
----------------------

.. code-block:: bash

   uv run streamlit run dashboard/app.py

Open ``http://localhost:8501`` in your browser.

Pages
-----

Markets
~~~~~~~

* **Market selector** — dropdown of all active markets in DuckDB.
* **Order book depth chart** — interactive Plotly chart showing the top 10
  bid and ask levels with volume.
* **Mid-price time series** — last N ticks with a 10-tick rolling average.
* **OBI heatmap** — colour-coded order book imbalance over time.
* **Spread vs GLFT optimal** — current quoted spread compared to the
  :math:`\delta^*` predicted by GLFT.

PnL
~~~

* **Cumulative P&L** — realised + unrealised, updated on every tick.
* **Inventory over time** — signed inventory per market, coloured by
  near-resolution regime.
* **Daily Sharpe** — rolling 30-day annualised Sharpe ratio.
* **Drawdown chart** — underwater equity curve.

Calibration
~~~~~~~~~~~

* **Brier score trend** — rolling 7-day and 30-day Brier scores per venue.
* **Isotonic calibration curve** — predicted probability vs empirical
  resolution frequency.
* **Bias detector alerts** — table of recent model drift alerts with
  timestamps and magnitudes.

Configuration
-------------

The dashboard reads from the same DuckDB database as the ingestion pipeline.
Point it to a different database by setting ``DUCKDB_PATH``:

.. code-block:: bash

   DUCKDB_PATH=./data/duckdb/markets.duckdb uv run streamlit run dashboard/app.py

Auto-refresh interval can be configured in the sidebar (default: 5 seconds).
