Live Trading
============

.. warning::

   Live trading submits real orders that involve real money.  Always validate
   your strategy thoroughly in **paper mode** and with the **Kalshi demo
   environment** before switching to production.

Prerequisites
-------------

1. Complete :doc:`ingestion` — the system must be ingesting data for the
   markets you want to trade.
2. Calibrate strategy parameters using ``run_calibration.py``.
3. Run at least one backtest on the target market (see :doc:`backtesting`).

Switching from paper to live execution
---------------------------------------

The :class:`~execution.router.OrderRouter` uses a
:class:`~execution.paper.engine.PaperExecutionEngine` by default.  To switch
to live execution, replace it with the appropriate live executor:

.. code-block:: python

   from config.settings import Settings
   from execution.live.kalshi_executor import KalshiExecutor
   from execution.router import OrderRouter
   from execution.risk.circuit_breaker import CircuitBreaker
   from execution.risk.limits import RiskLimitsChecker
   from execution.risk.monitor import RiskMonitor
   from execution.paper.account import PaperAccount

   settings = Settings()

   # Use Kalshi live executor
   executor = KalshiExecutor(
       api_key=settings.kalshi_api_key,
       private_key_path=settings.kalshi_private_key_path,
       env=settings.kalshi_env,   # "demo" or "prod"
   )

   account = PaperAccount(initial_cash=10_000.0)  # still tracks local P&L
   risk_limits = RiskLimitsChecker(q_max=5, max_daily_loss=200.0)
   circuit_breaker = CircuitBreaker(max_daily_loss=200.0)
   risk_monitor = RiskMonitor(account=account)

   router = OrderRouter(
       paper_engine=executor,   # drop-in replacement
       risk_limits=risk_limits,
       circuit_breaker=circuit_breaker,
       risk_monitor=risk_monitor,
       q_max_base=5.0,
   )

Kalshi demo sandbox
-------------------

Set ``KALSHI_ENV=demo`` (the default) to submit orders to the Kalshi
sandbox.  Orders are real API calls but use play money:

.. code-block:: bash

   KALSHI_ENV=demo \
   KALSHI_API_KEY=<your-demo-key> \
   KALSHI_PRIVATE_KEY_PATH=./secrets/kalshi_demo.pem \
   uv run python main.py

Polymarket live trading
-----------------------

Polymarket orders are signed with EIP-712 using your Polygon private key.

.. code-block:: bash

   ENABLE_POLYMARKET=true \
   POLYMARKET_PRIVATE_KEY=0x<hex-key> \
   POLYMARKET_PROXY_ADDRESS=0x<proxy> \
   uv run python main.py

.. note::

   Ensure your Polygon wallet has sufficient USDC and MATIC (for gas) before
   enabling live Polymarket execution.

Risk limits
-----------

Always set conservative limits when going live for the first time.  The
following parameters are recommended starting points:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Parameter
     - Suggested value
     - Description
   * - ``q_max``
     - 3–5
     - Maximum absolute inventory per market.
   * - ``max_daily_loss``
     - 1–2% of capital
     - Circuit breaker threshold.
   * - ``gamma_I``
     - 0.1–0.3
     - Higher → wider spreads → lower fill rate but less inventory risk.

Monitoring
----------

While the system is running, monitor it via:

1. **Logs** — structured logs are written to stdout.  In production use
   ``LOG_FORMAT=json`` and pipe to your log aggregator.
2. **Dashboard** — ``uv run streamlit run dashboard/app.py`` opens the
   real-time PnL and risk monitor.
3. **DuckDB** — query the ``features`` and ``ticks`` tables directly to
   inspect market state.

Stopping the system
--------------------

Press ``Ctrl+C`` to trigger a graceful shutdown.  In-flight orders will be
cancelled before the process exits.
