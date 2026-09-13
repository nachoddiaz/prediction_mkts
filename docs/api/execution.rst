execution
=========

Order lifecycle management.  The execution layer receives
:class:`~strategies.market_making.glft.Quote` objects from strategy quoters,
validates them against risk limits, and submits orders to either the paper
engine or a live venue API.

Order
-----

Canonical order object and associated enumerations (side, status, action).

.. automodule:: execution.order
   :members:
   :undoc-members:
   :show-inheritance:

Order router
------------

Orchestrates the full quoting cycle:

1. Receives a :class:`~strategies.market_making.glft.Quote` from the strategy.
2. Evaluates the :class:`~execution.risk.circuit_breaker.CircuitBreaker`.
3. Computes dynamic :math:`Q_{\max}^{\text{eff}}` from the near-resolution regime.
4. Validates proposed orders through :class:`~execution.risk.limits.RiskLimitsChecker`.
5. Sends, replaces, or cancels orders in the paper or live engine.
6. Avoids churning: only re-sends if the quoted price or size changed.

.. automodule:: execution.router
   :members:
   :undoc-members:
   :show-inheritance:

Paper trading
-------------

Simulated execution engine for backtesting and paper trading.  Fills are
probabilistic — the fill probability decays exponentially with the distance
from the best quote, matching the Poisson arrival model assumed by GLFT.

Paper account
~~~~~~~~~~~~~

Tracks cash, inventory, and realised P&L for a single paper account.

.. automodule:: execution.paper.account
   :members:
   :undoc-members:
   :show-inheritance:

Paper engine
~~~~~~~~~~~~

Matches incoming orders against the simulated book and calls back into the
account object on fill events.

.. automodule:: execution.paper.engine
   :members:
   :undoc-members:
   :show-inheritance:

Live execution
--------------

Real order submission to the venue REST APIs.

Kalshi executor
~~~~~~~~~~~~~~~

.. automodule:: execution.live.kalshi_executor
   :members:
   :undoc-members:
   :show-inheritance:

Polymarket executor
~~~~~~~~~~~~~~~~~~~

.. automodule:: execution.live.polymarket_executor
   :members:
   :undoc-members:
   :show-inheritance:

Risk management
---------------

Three independent safeguards operating in the order router.

Risk limits
~~~~~~~~~~~

Per-market and global inventory limits.  Rejects orders that would push
inventory beyond :math:`Q_{\max}`.

.. automodule:: execution.risk.limits
   :members:
   :undoc-members:
   :show-inheritance:

Circuit breaker
~~~~~~~~~~~~~~~

Halts quoting when:

* A near-resolution ``HALT`` regime is detected (:math:`\tau < 5\text{min}`).
* The daily P&L loss exceeds a configurable threshold.

.. automodule:: execution.risk.circuit_breaker
   :members:
   :undoc-members:
   :show-inheritance:

Risk monitor
~~~~~~~~~~~~

Continuous P&L tracking.  Computes unrealised P&L using the current mid-price,
logs structured alerts, and feeds metrics to the dashboard.

.. automodule:: execution.risk.monitor
   :members:
   :undoc-members:
   :show-inheritance:
