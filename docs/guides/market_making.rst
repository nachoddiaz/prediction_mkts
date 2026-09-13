Market Making
=============

This guide walks through the GLFT and Cartea-Jaimungal quoters: what
parameters they require, how to compute a quote, and how quotes flow through
the risk checks into the paper engine.

The mathematics are described in :doc:`../math` §3–4; this guide focuses on
the code.

GLFT quoter
-----------

The :class:`~strategies.market_making.glft.GLFTQuoter` requires three
parameters:

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Parameter
     - Type
     - Meaning
   * - ``gamma_I``
     - ``float > 0``
     - CARA inventory risk aversion.  Higher values → wider spread and lower
       maximum inventory.
   * - ``kappa_x``
     - ``float > 0``
     - Fill-curve decay in logit space.  Calibrated from historical fills.
   * - ``sigma_b``
     - ``float > 0``
     - Instantaneous belief volatility.  Typically set to
       :math:`\sigma_B(p, \tau)` from :mod:`models.vol.bernoulli_surface`.

.. code-block:: python

   from datetime import UTC, datetime
   from strategies.market_making.glft import GLFTQuoter
   from normalizer.schema import MarketId

   quoter = GLFTQuoter(gamma_I=0.1, kappa_x=1.5, sigma_b=0.02)

   quote = quoter.quote(
       market_id=MarketId("kalshi:KXBTC-26APR2212-T85799.99"),
       mid_price=0.55,          # current best mid
       inventory=2,             # contracts held (signed)
       tau=0.005,               # time to resolution in years (~44 hours)
       resolution_date=datetime(2026, 4, 22, 16, tzinfo=UTC),
   )

   print(f"bid={quote.bid_price:.4f}  ask={quote.ask_price:.4f}")
   print(f"half-spread={quote.half_spread:.4f}")
   print(f"reservation_price={quote.reservation_price:.4f}")

Cartea-Jaimungal quoter
-----------------------

:class:`~strategies.market_making.cartea_jaimungal.CarteaJaimungalQuoter`
takes two additional parameters for the signal dynamics:

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Parameter
     - Type
     - Meaning
   * - ``phi``
     - ``float > 0``
     - Mean-reversion speed of the latent drift :math:`\mu_t`.
   * - ``eta``
     - ``float > 0``
     - Volatility of the latent drift.
   * - ``rho``
     - ``float ∈ (-1, 1)``
     - Correlation between price innovations and the signal.

.. code-block:: python

   from strategies.market_making.cartea_jaimungal import CarteaJaimungalQuoter

   cj = CarteaJaimungalQuoter(
       gamma_I=0.1,
       kappa_x=1.5,
       phi=0.05,
       eta=0.003,
       rho=0.3,
   )

   quote = cj.quote(
       market_id=MarketId("kalshi:KXBTC-26APR2212-T85799.99"),
       mid_price=0.55,
       inventory=2,
       tau=0.005,
       sigma_b=0.02,
       mu_hat=0.01,             # directional signal estimate
       resolution_date=datetime(2026, 4, 22, 16, tzinfo=UTC),
   )

Routing quotes to the paper engine
------------------------------------

.. code-block:: python

   from execution.paper.account import PaperAccount
   from execution.paper.engine import PaperExecutionEngine
   from execution.risk.circuit_breaker import CircuitBreaker
   from execution.risk.limits import RiskLimitsChecker
   from execution.risk.monitor import RiskMonitor
   from execution.router import OrderRouter

   account = PaperAccount(initial_cash=10_000.0)
   paper_engine = PaperExecutionEngine(account=account)
   risk_limits = RiskLimitsChecker(q_max=10, max_daily_loss=500.0)
   circuit_breaker = CircuitBreaker(max_daily_loss=500.0)
   risk_monitor = RiskMonitor(account=account)

   router = OrderRouter(
       paper_engine=paper_engine,
       risk_limits=risk_limits,
       circuit_breaker=circuit_breaker,
       risk_monitor=risk_monitor,
       q_max_base=10.0,
   )

   # Feed mid-price to the monitor for unrealised P&L tracking
   router.mid_prices[quote.market_id] = quote.mid_price

   order_ids = router.on_quote(quote, size=1.0)

Near-resolution behaviour
--------------------------

The router automatically adjusts :math:`Q_{\max}` and :math:`\gamma_I` as
:math:`\tau` shrinks.  Pass the :class:`~normalizer.schema.Market` object to
enable this:

.. code-block:: python

   from normalizer.schema import Market, MarketStatus, MarketCategory, Resolution

   market = Market(
       market_id="kalshi:KXBTC-26APR2212-T85799.99",
       venue="kalshi",
       question="Will BTC be above $85,000 on April 22?",
       category=MarketCategory.CRYPTO,
       status=MarketStatus.OPEN,
       resolution=Resolution(date=datetime(2026, 4, 22, 16, tzinfo=UTC)),
   )

   router.on_quote(quote, size=1.0, market=market)

When the circuit breaker fires (``HALT`` regime or daily loss exceeded), the
router skips order submission and returns an empty list.

Calibrating parameters
----------------------

Run ``run_calibration.py`` to fit :math:`\kappa` and :math:`A` from
historical fill data, and to update the venue YAML config:

.. code-block:: bash

   uv run python run_calibration.py --market kalshi:KXBTC-26APR2212-T85799.99 --days 30
