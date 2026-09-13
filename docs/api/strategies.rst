strategies
==========

Trading strategy implementations.  All strategies read a
:class:`~strategies.market_making.glft.Quote` or a signal and emit orders
through :class:`~execution.router.OrderRouter`.

Base strategy
-------------

.. automodule:: strategies.base_strategy
   :members:
   :undoc-members:
   :show-inheritance:

Market making
-------------

Optimal quoters based on stochastic-control theory.  Both models operate in
logit space :math:`X_t = \text{logit}(p_t)` so that the boundary conditions
:math:`p \in (0,1)` are automatically respected without clamping hacks.

See :doc:`../math` §3–4 for the derivations.

GLFT quoter
~~~~~~~~~~~

Avellaneda-Stoikov approximation in logit space.  The reservation log-odds
and optimal half-spread reduce to:

.. math::

   \tilde{X} = X_t - q \gamma_I \sigma_b^2 \tau, \qquad
   \frac{\delta^*}{2} = \frac{\gamma_I \sigma_b^2 \tau}{2}
                       + \frac{1}{\kappa_x}\ln\!\left(1+\frac{\gamma_I}{\kappa_x}\right)

.. automodule:: strategies.market_making.glft
   :members:
   :undoc-members:
   :show-inheritance:

Cartea-Jaimungal quoter
~~~~~~~~~~~~~~~~~~~~~~~

Extends GLFT with a latent directional drift :math:`\hat{\mu}_t` (MATH.md §4.5).
The reservation log-odds is skewed by the signal decay factor
:math:`\varphi_1(\tau)`:

.. math::

   \tilde{X}_{\text{CJ}} = \tilde{X}_{\text{GLFT}}
       + \frac{\rho \sigma_b \eta}{\phi}(1 - e^{-\phi\tau})\,\hat{\mu}_t

.. automodule:: strategies.market_making.cartea_jaimungal
   :members:
   :undoc-members:
   :show-inheritance:

Strategy parameters
~~~~~~~~~~~~~~~~~~~

Dataclass holding all calibrated parameters for both quoters.  Loaded from
the venue YAML configuration and updated by ``run_calibration.py``.

.. automodule:: strategies.market_making.params
   :members:
   :undoc-members:
   :show-inheritance:

Arbitrage
---------

Cross-venue arbitrage engine that exploits mispricings between Kalshi and
Polymarket on the same underlying event.

Arbitrage detector
~~~~~~~~~~~~~~~~~~

Detects an actionable opportunity when the net edge after all costs exceeds
zero:

.. math::

   \text{edge} = p^K - p^P - f_K - f_P - b_U - \rho_c C \tau > 0

.. automodule:: strategies.arbitrage.detector
   :members:
   :undoc-members:
   :show-inheritance:

Arbitrage sizing
~~~~~~~~~~~~~~~~

Fractional Kelly criterion for position sizing:

.. math::

   q^* = \varphi_K \cdot \frac{\varepsilon}{\text{Var}[\text{payoff}]}

.. automodule:: strategies.arbitrage.sizing
   :members:
   :undoc-members:
   :show-inheritance:

Cross-venue arbitrage
~~~~~~~~~~~~~~~~~~~~~

End-to-end orchestration of detection, sizing, and execution.

.. automodule:: strategies.arbitrage.cross_venue
   :members:
   :undoc-members:
   :show-inheritance:
