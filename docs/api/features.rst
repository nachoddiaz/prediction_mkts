features
========

Computes market microstructure signals from raw ticks and order books.
Features are written to the ``features`` table in DuckDB once per tick and
consumed by the strategy quoters at runtime.

Microstructure
--------------

Three computation tiers:

1. **Snapshot** — operate on a single :class:`~normalizer.schema.OrderBook`
   or :class:`~normalizer.schema.Tick`.  :math:`O(1)` — used in the hot path.
2. **Series** — operate on DataFrames of historical ticks.  Used for
   calibration and research notebooks.
3. **Pipeline** — high-level function that reads from DuckDB, computes all
   features, and returns a dict ready to persist.

Key features computed:

* **OBI** — Order Book Imbalance :math:`\in [-1,1]` (see :doc:`../math` §2).
* **Bernoulli vol** :math:`\sigma_B(p,\tau)` — volatility from the binary
  payoff variance.
* **EWMA vol** — empirical volatility from the rolling quadratic variation of
  :math:`X_t = \text{logit}(p_t)`.
* **Quoted spread** — :math:`p^a - p^b`.
* **Relative spread** — :math:`(p^a - p^b) / p_{\text{mid}} \times 100`.
* :math:`\hat{\mu}` — directional signal proxy (OBI until full signal
  calibration).

.. automodule:: features.microstructure
   :members:
   :undoc-members:
   :show-inheritance:

Resolution
----------

Computes :math:`\tau = T - t` (time to resolution in years) and classifies
each market into a near-resolution regime.

The four regimes (see :doc:`../math` §5) drive dynamic risk parameter
adjustments:

.. list-table::
   :header-rows: 1

   * - Regime
     - Threshold
   * - ``NORMAL``
     - :math:`\tau \geq 24\text{h}`
   * - ``WARNING``
     - :math:`1\text{h} \leq \tau < 24\text{h}`
   * - ``CRITICAL``
     - :math:`5\text{min} \leq \tau < 1\text{h}`
   * - ``HALT``
     - :math:`\tau < 5\text{min}`

.. automodule:: features.resolution
   :members:
   :undoc-members:
   :show-inheritance:

Feature store
-------------

Manages the write path from computed feature dicts to the DuckDB
``features`` table.

.. automodule:: features.store
   :members:
   :undoc-members:
   :show-inheritance:

Calibration
-----------

Offline calibration utilities called by ``run_calibration.py`` to fit
:math:`\kappa` and :math:`A` from historical fill data.

.. automodule:: features.calibration
   :members:
   :undoc-members:
   :show-inheritance:

Signals
-------

The signal sub-package produces the directional estimate
:math:`\hat{\mu}_t = \sum_k w_k s_{k,t}`.

News signal
~~~~~~~~~~~

.. automodule:: features.signals.news
   :members:
   :undoc-members:
   :show-inheritance:

On-chain signal
~~~~~~~~~~~~~~~

.. automodule:: features.signals.onchain
   :members:
   :undoc-members:
   :show-inheritance:

Ensemble
~~~~~~~~

.. automodule:: features.signals.ensemble
   :members:
   :undoc-members:
   :show-inheritance:
