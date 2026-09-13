Installation
============

Requirements
------------

* Python **3.12+**
* `uv <https://github.com/astral-sh/uv>`_ (recommended) or ``pip``
* A running PostgreSQL instance is **not** required — the system uses
  DuckDB as an embedded OLAP database.

Clone the repository
--------------------

.. code-block:: bash

   git clone https://github.com/nachoddiaz/prediction-market-system.git
   cd prediction-market-system

Install with uv (recommended)
------------------------------

.. code-block:: bash

   # Core system only
   uv sync

   # Include dashboard (Streamlit + Plotly)
   uv sync --extra dashboard

   # Include research tools (Jupyter, matplotlib, seaborn)
   uv sync --extra research

   # Full development environment (pytest, ruff, mypy, pre-commit)
   uv sync --extra dev

   # Everything
   uv sync --extra all

Install with pip
----------------

.. code-block:: bash

   pip install -e ".[dev,dashboard,research]"

Environment variables
---------------------

Copy ``.env.example`` to ``.env`` and fill in your credentials:

.. code-block:: bash

   cp .env.example .env

The following variables are recognised:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Variable
     - Default
     - Description
   * - ``KALSHI_API_KEY``
     - ``""``
     - Kalshi REST / WebSocket API key.
   * - ``KALSHI_PRIVATE_KEY_PATH``
     - ``./secrets/kalshi_private.pem``
     - Path to the RSA private key used to sign Kalshi requests.
   * - ``KALSHI_ENV``
     - ``demo``
     - ``demo`` for the sandbox, ``prod`` for live trading.
   * - ``POLYMARKET_PRIVATE_KEY``
     - ``""``
     - Ethereum private key (hex) for Polymarket order signing (EIP-712).
   * - ``POLYMARKET_PROXY_ADDRESS``
     - ``""``
     - CTF Exchange proxy address on Polygon.
   * - ``POLYGON_RPC_URL``
     - ``https://polygon-rpc.com``
     - Polygon JSON-RPC endpoint for on-chain reads.
   * - ``DUCKDB_PATH``
     - ``./data/duckdb/markets.duckdb``
     - Path to the DuckDB database file.
   * - ``PARQUET_BASE_PATH``
     - ``./data/parquet``
     - Root directory for Parquet archives.
   * - ``LOG_FORMAT``
     - ``text``
     - ``text`` for coloured dev output, ``json`` for structured production logs.
   * - ``LOG_LEVEL``
     - ``INFO``
     - Standard Python log levels: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``.

Verify the installation
-----------------------

.. code-block:: bash

   # Run the full test suite (unit + integration, no live HTTP calls)
   uv run pytest -m "not live"

   # Quick smoke test — start the ingestion loop against Manifold (no credentials needed)
   uv run python main.py

Build the documentation
-----------------------

.. code-block:: bash

   uv pip install sphinx furo
   cd docs
   make html
   # Open docs/_build/html/index.html in your browser
