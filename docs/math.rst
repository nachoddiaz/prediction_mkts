Mathematical Foundations
========================

This page summarises the mathematical framework underlying the system.
The full derivations live in :file:`MATH.md` at the project root; this
page provides the essential equations that appear directly in the code.

.. note::

   All prices are probabilities :math:`\in [0,1]`.  Kalshi quotes in
   cents are divided by 100; Polymarket USDC fractional amounts are used
   directly.

Notation
--------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Symbol
     - Meaning
     - Units
   * - :math:`p_t \in (0,1)`
     - Mid-quote / risk-neutral probability
     - dimensionless
   * - :math:`X_t = \text{logit}(p_t)`
     - Log-odds state variable
     - nats
   * - :math:`\tau = T - t`
     - Time remaining to resolution
     - years
   * - :math:`q_t`
     - Inventory in contracts (signed)
     - shares
   * - :math:`\sigma_b(t, X)`
     - Belief volatility (instantaneous, of log-odds)
     - :math:`1/\sqrt{s}`
   * - :math:`\gamma_I`
     - CARA inventory risk aversion
     - :math:`1/\$`
   * - :math:`\kappa_x`
     - Fill-curve decay in logit space
     - dimensionless
   * - :math:`A`
     - Fill arrival rate at zero spread
     - :math:`1/s`
   * - :math:`\hat{\mu}_t`
     - Estimated signal: :math:`\sum_k w_k s_{k,t}`
     - dimensionless
   * - :math:`\phi`
     - Mean-reversion speed of :math:`\mu_t`
     - :math:`1/s`
   * - :math:`\eta`
     - Volatility of :math:`\mu_t`
     - dimensionless
   * - :math:`\rho`
     - Correlation :math:`d\langle W,B\rangle_t = \rho\,dt`
     - dimensionless

Why :math:`X_t = \text{logit}(p_t)` and not :math:`p_t`?
   The logit map transports the boundary to :math:`\pm\infty` where
   standard semimartingale tools (Itô–Lévy) apply, while preserving
   :math:`p \in (0,1)` automatically.  AS, GLFT and CJ all operate in
   their canonical :math:`\mathbb{R}`-valued state space on :math:`X_t`.

1. Why does the spread exist? — Glosten-Milgrom (1985)
------------------------------------------------------

Three types of agents:

* **Market maker (MM)**: risk-neutral, posts bid :math:`r^b` and ask :math:`r^a`.
* **Informed traders**: know the true value :math:`V \in \{0,1\}` with certainty.
* **Noise traders**: trade randomly, independently of :math:`V`.

In log-odds the Bayesian update is **additive**:

.. math::

   X^{\text{buy}}_{t+} = X_t + \ln\text{LR}_+, \qquad
   X^{\text{sell}}_{t+} = X_t - \ln\text{LR}_-

Converting back to probabilities:

.. math::

   p_+ = \sigma(X_t + \ln\text{LR}_+), \qquad
   p_- = \sigma(X_t - \ln\text{LR}_-)

The adverse-selection half-spread:

.. math::

   \delta^{\text{AS}}_X = \frac{\ln\text{LR}_+ + \ln\text{LR}_-}{2}

2. Microstructure features
--------------------------

Order Book Imbalance (OBI)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \text{OBI} = \frac{V_{\text{bid}}^{(L)} - V_{\text{ask}}^{(L)}}
                     {V_{\text{bid}}^{(L)} + V_{\text{ask}}^{(L)}}
                \in [-1, 1]

where :math:`V^{(L)}` is the total volume across the top :math:`L` levels.
OBI is used as the primary component of :math:`\hat{\mu}_t`.

Bernoulli volatility surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a binary contract, the variance of the payoff at resolution is
:math:`p(1-p)`.  The instantaneous volatility in logit space is:

.. math::

   \sigma_B(p, \tau) = \frac{\sqrt{p(1-p)}}{\sqrt{\tau}}

This is implemented in :mod:`models.vol.bernoulli_surface` and
:mod:`features.microstructure`.

3. GLFT optimal quoter
-----------------------

The GLFT model (Guéant-Lehalle-Fernandez-Tapia, 2013) approximation
in logit space gives the **reservation log-odds**:

.. math::

   \tilde{X} = X_t - q \cdot \gamma_I \cdot \sigma_b^2 \cdot \tau

and the **optimal half-spread**:

.. math::

   \frac{\delta^*}{2} = \frac{\gamma_I \sigma_b^2 \tau}{2}
                       + \frac{1}{\kappa_x} \ln\!\left(1 + \frac{\gamma_I}{\kappa_x}\right)

Quotes in probability space:

.. math::

   p^b = \sigma\!\left(\tilde{X} - \frac{\delta^*}{2}\right), \qquad
   p^a = \sigma\!\left(\tilde{X} + \frac{\delta^*}{2}\right)

See :class:`strategies.market_making.glft.GLFTQuoter`.

4. Cartea-Jaimungal with directional signal
-------------------------------------------

The CJ extension adds a latent directional drift :math:`\hat{\mu}_t`.
The **reservation log-odds** becomes:

.. math::

   \tilde{X}_{\text{CJ}} = X_t
     - q \cdot \gamma_I \cdot \sigma_b^2 \cdot \tau
     + \varphi_1(\tau) \cdot \hat{\mu}_t

where the signal decay factor is:

.. math::

   \varphi_1(\tau) = \frac{\rho \sigma_b \eta}{\phi}
                    \left(1 - e^{-\phi \tau}\right)

The optimal half-spread is identical to the GLFT formula above.

See :class:`strategies.market_making.cartea_jaimungal.CarteaJaimungalQuoter`.

5. Near-resolution regimes
--------------------------

As :math:`\tau \to 0`, the Bernoulli volatility :math:`\sigma_B(p,\tau)`
diverges and informed flow dominates.  The system classifies markets into
four regimes:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Regime
     - Condition
     - Effect
   * - ``NORMAL``
     - :math:`\tau \geq 24\text{h}`
     - Normal parameters.
   * - ``WARNING``
     - :math:`1\text{h} \leq \tau < 24\text{h}`
     - :math:`\gamma_{\text{eff}} = 2\gamma_I`, :math:`Q_{\max} = Q/2`.
   * - ``CRITICAL``
     - :math:`5\text{min} \leq \tau < 1\text{h}`
     - :math:`Q_{\max} = 1`, one side of the book halted.
   * - ``HALT``
     - :math:`\tau < 5\text{min}`
     - All quoting halted.

See :mod:`features.resolution` and :class:`execution.risk.circuit_breaker.CircuitBreaker`.

6. Cross-venue arbitrage
-------------------------

An arbitrage opportunity exists when:

.. math::

   p^K - p^P > f_K + f_P + b_U + \rho_c \cdot C \cdot \tau

where :math:`f_K, f_P` are taker fees, :math:`b_U` is the USDC–USD basis,
:math:`\rho_c` is the continuous capital cost, and :math:`C` is the required
collateral.

Position sizing uses a fractional Kelly criterion:

.. math::

   q^* = \varphi_K \cdot \frac{\varepsilon}{\text{Var}[\text{payoff}]}

See :mod:`strategies.arbitrage`.

7. Calibration
--------------

The fill-arrival parameters :math:`\kappa` and :math:`A` are estimated by
maximum likelihood from historical ticks.  The probability model parameters
are calibrated using:

* **Isotonic regression** — for monotone calibration of the signal ensemble.
* **Brier score** — to measure forecast accuracy.
* **Bias detector** — to detect model drift over rolling windows.

See :mod:`models.calibration` and :mod:`features.calibration`.
