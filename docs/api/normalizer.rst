normalizer
==========

Transforms raw venue responses into the canonical domain objects used by all
downstream components.  Both adapters emit the same types; venue-specific
conventions (price units, timestamp formats, market status strings) are
handled entirely inside each adapter.

Schema
------

The unified domain model.  All ``market_id`` values follow the
``venue:raw_id`` format (e.g. ``kalshi:KXBTC-26APR2212-T85799.99``) to
guarantee global uniqueness across venues.

.. automodule:: normalizer.schema
   :members:
   :undoc-members:
   :show-inheritance:

Kalshi adapter
--------------

Key conventions:

* Prices arrive as whole cents — ``45`` → ``0.45`` probability.
* Order book levels are lists of ``[price, quantity]`` pairs.
* Resolution dates come as ISO strings in UTC.
* Market status transitions (``open`` → ``resolved``) are detected and
  propagated.

.. automodule:: normalizer.kalshi_adapter
   :members:
   :undoc-members:
   :show-inheritance:

Polymarket adapter
------------------

Key conventions:

* Timestamps arrive as either Unix milliseconds (``int``) or ISO strings —
  both are normalised to ``datetime`` (UTC).
* Prices are USDC fractional amounts used directly as probabilities.
* YES/NO token structure is mapped to a unified bid/ask representation.

.. automodule:: normalizer.polymarket_adapter
   :members:
   :undoc-members:
   :show-inheritance:
