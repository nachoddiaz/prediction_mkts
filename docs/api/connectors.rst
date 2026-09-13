connectors
==========

Async venue clients.  Each connector authenticates with the venue REST API,
opens a WebSocket stream, and emits raw Python dicts that are then processed
by the :doc:`normalizer`.

All connectors inherit from :class:`~connectors.base.BaseConnector` which
enforces the ``connect() / stream() / disconnect()`` async interface.

Base
----

.. automodule:: connectors.base
   :members:
   :undoc-members:
   :show-inheritance:

Kalshi
------

Connects to the Kalshi REST API (authentication via RSA private key) and
WebSocket feed.  Supports both ``demo`` (sandbox) and ``prod`` environments
controlled by the ``KALSHI_ENV`` environment variable.

.. automodule:: connectors.kalshi
   :members:
   :undoc-members:
   :show-inheritance:

Polymarket
----------

Connects to two independent Polymarket APIs:

* **Gamma API** — market metadata (title, category, settlement date, status).
* **CLOB API** — real-time order book and trade history.

Order submission uses EIP-712 signing via :mod:`eth_account`.

.. automodule:: connectors.polymarket
   :members:
   :undoc-members:
   :show-inheritance:

Manifold
--------

Connects to the Manifold Markets API — a fully public sandbox with no
credentials required.  Used for development and smoke tests.

.. automodule:: connectors.manifold
   :members:
   :undoc-members:
   :show-inheritance:
