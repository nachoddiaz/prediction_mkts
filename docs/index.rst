Prediction Market System
========================

A quantitative market-making system for `Kalshi <https://kalshi.com>`_ and
`Polymarket <https://polymarket.com>`_.  The system quotes tight spreads,
manages inventory risk, fades directional signals, and captures cross-venue
arbitrage when the same event misprices across platforms.

Each layer is built on the failure of the previous model — from the
microstructural reason spreads must exist (Glosten-Milgrom), through the
stochastic-control optimal quoter (GLFT / Cartea-Jaimungal), to near-resolution
jump risk and Bayesian calibration.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: System Design

   architecture
   math

.. toctree::
   :maxdepth: 1
   :caption: How-To Guides

   guides/ingestion
   guides/market_making
   guides/backtesting
   guides/live_trading
   guides/dashboard

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/config
   api/connectors
   api/normalizer
   api/storage
   api/features
   api/models
   api/strategies
   api/execution
   api/backtesting
   api/dashboard

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
