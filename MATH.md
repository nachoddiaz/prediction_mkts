# Mathematical Foundations
### Prediction Market System — Kalshi & Polymarket

> This document develops the mathematical framework underlying the system's
> market-making, arbitrage, and calibration components. Each section is
> motivated by the **failures of the previous model**, building a coherent
> narrative from first principles to the original near-resolution extension.

---

> **v2.2 corrections (this revision).** Seven defects found by dimensional and
> sign analysis of v2.1, all now fixed in doc *and* code:
> - **(U)** Notation table gave $\tau$, $\sigma_b$, $\phi$, $A$ in *seconds*; every
>   operative formula and all code use **years**. Units table corrected.
> - **(S1)** §4.4 integrated $\dot\phi_2$ forward instead of backward from
>   $\phi_2(T)=0$, giving $\phi_2=+\gamma\sigma^2\tau/2$ instead of $-\gamma\sigma^2\tau/2$.
> - **(S2)** §4.5 wrote $\tilde p = S - \partial_q g$; §4.3 implies $\tilde p = S + \partial_q g$.
>   (S1) and (S2) cancelled in the inventory term and left the *signal* term inverted.
>   The code was right; the document was wrong.
> - **(D)** §4.4's $\phi_1=(\rho\sigma\eta/\phi)(1-e^{-\phi\tau})$ is dimensionally
>   inconsistent — it makes $\phi_1\hat\mu$ come out in nats²/year² instead of nats.
>   The correct alpha-capture factor is $\phi_1=\rho_\mu(1-e^{-\phi\tau})/\phi$.
> - **(R)** §5.4 eq (2.2) used the A-S approximation $\frac{1}{\kappa}\ln(1+\gamma/\kappa)$
>   that §2.4 explicitly rejects. Unified on the exact $\frac{1}{\gamma}\ln(1+\gamma/\kappa)$.
> - **(N)** §5.3's realised-QV estimator was applied tick-by-tick, so it inherited
>   the microstructure-noise divergence of realised variance as $\Delta t\to0$.
>   Now estimated on a fixed 60 s grid, winsorised, and bounded.
> - **(T)** §5.4 eq (2.3) hard-coded a \$0.01 tick. Both venues quote finer;
>   the tick is now a per-market price ladder read from venue metadata.

---

## Notation Reference

> **v2.1 corrections**: `σ_b` replaces `σ_B`; `γ_I` / `φ_K` replace single `γ`;
> `κ_x` / `κ_p` replace single `κ`; `X_t = logit(p_t)` is the primary state variable.

| Symbol | Meaning | Units |
|--------|---------|-------|
| $p_t \in (0,1)$ | Mid-quote / risk-neutral probability | dimensionless |
| $X_t = \text{logit}(p_t)$ | Log-odds state variable | nats |
| $\sigma(x) = (1+e^{-x})^{-1}$ | Inverse logit | — |
| $\sigma'(x) = p(1-p)$ | First derivative of inverse logit | — |
| $\sigma''(x) = p(1-p)(1-2p)$ | Second derivative | — |
| $T$ | Resolution time (fixed) | — |
| $\tau = T - t$ | Time remaining to resolution | **years** |
| $q_t \in \mathbb{Z}$ | Inventory in contracts (signed) | shares |
| $Q$ | Maximum inventory limit $\vert q \vert \leq Q$ | shares |
| $\sigma_b(t, X)$ | Belief volatility (instantaneous, of log-odds) | $1/\sqrt{\text{year}}$ |
| $\gamma_I$ | CARA inventory risk aversion | $1/\$$ |
| $\varphi_K \in (0,1]$ | Kelly fractional multiplier | dimensionless |
| $\kappa_x,\, \kappa_p$ | Fill-curve decay in logit / price space | dimensionless, $1/\$$ |
| $A$ | Fill arrival rate at zero spread | $1/\text{year}$ |
| $\varepsilon = p^{\mathbb{P}} - p^{\text{mkt}}$ | Edge (model minus market) | dimensionless |
| $\delta^b, \delta^a$ | Bid and ask half-spreads | — |
| $\lambda^b(\delta), \lambda^a(\delta)$ | Arrival intensities of market orders | $1/\text{year}$ |
| $W_t$ | $\mathbb{Q}$-Brownian motion (log-odds noise) | — |
| $B_t$ | $\mathbb{P}$-Brownian motion (signal noise) | — |
| $\mu_t$ | Latent alpha-drift of $X_t$ under $\mathbb{P}$ (OU) | nats/year |
| $\hat{\mu}_t$ | Estimated drift: $\sum_k w_k s_{k,t}$ (the $w_k$ carry the units) | nats/year |
| $\phi$ | Mean-reversion speed of $\mu_t$ | $1/\text{year}$ |
| $\eta$ | Volatility of $\mu_t$ | nats/year$^{3/2}$ |
| $\rho_\mu \in (0,1]$ | Measure-change discount on alpha | dimensionless |
| $\rho$ | Correlation $d\langle W,B\rangle_t = \rho\,dt$ | dimensionless |
| $r$ | Oracle reversal probability | dimensionless |
| $b_U$ | USDC–USD basis (Polymarket leg) | dimensionless |
| $f_t$ | Proportional taker fee | dimensionless |
| $\rho_c$ | Continuous capital cost rate | $1/\text{year}$ |

All prices are expressed as probabilities $\in [0,1]$. Kalshi quotes in cents
are divided by 100; Polymarket USDC fractional amounts are used directly.

**Why $X_t$ and not $p_t$?** The logit map transports the boundary to $\pm\infty$
where standard semimartingale tools (Itô–Lévy) apply, while preserving $p \in (0,1)$
automatically. AS, GLFT and CJ all operate in their canonical $\mathbb{R}$-valued
state space on $X_t$.

---

## 1. Why Does the Spread Exist? — Glosten-Milgrom (1985)

### 1.1 Motivation

Before asking *what spread to quote*, we need to understand *why* a spread
must exist at all. The answer is not inventory risk — it is **adverse
selection**.

### 1.2 The Model

Three types of agents:

- **Market maker (MM)**: risk-neutral, posts bid $r^b$ and ask $r^a$
- **Informed traders**: know the true value $V \in \{0,1\}$ with certainty
- **Noise traders**: trade randomly, independently of $V$

Let $\mu \in (0,1)$ be the prior probability that the contract resolves YES
($V=1$). Let $\alpha \in [0,1]$ be the fraction of order flow that is informed.

### 1.3 Asymmetric Likelihood Ratios

> **v2.1 correction (defect K):** A single $\alpha$ implicitly assumed
> sensitivity = specificity. Empirically, informed flow is asymmetric on the
> long-tail side. We now use separate LR+ and LR–.

For a $\$1/\$0$ binary contract with true value $V \in \{0,1\}$:

$$\text{LR}_+ := \frac{\Pr(\text{buy}\mid V=1)}{\Pr(\text{buy}\mid V=0)}, \qquad
\text{LR}_- := \frac{\Pr(\text{sell}\mid V=1)}{\Pr(\text{sell}\mid V=0)}$$

### 1.4 Bayesian Posteriors in Log-Odds

In log-odds the Bayesian update is **additive**:

$$X^{\text{buy}}_{t+} = X_t + \ln\text{LR}_+, \qquad X^{\text{sell}}_{t+} = X_t - \ln\text{LR}_-$$

where $X_t = \text{logit}(p_t)$. Converting back to probabilities:

$$\boxed{p_+ = \sigma(X_t + \ln\text{LR}_+), \qquad p_- = \sigma(X_t - \ln\text{LR}_-)}$$

### 1.5 Adverse-Selection Half-Spread

The half-spread induced by adverse selection in log-odds:

$$\boxed{\delta^x_{\text{AS}} = \tfrac{1}{2}(\ln\text{LR}_+ + \ln\text{LR}_-) \geq 0}$$

with equality iff the trade carries no information. In price domain:
$\delta^p_{\text{AS}} \approx p(1-p)\cdot\delta^x_{\text{AS}}$, which automatically
vanishes at the boundary — no separate boundary treatment needed.

The v1 formula $\Lambda = (1+\alpha)/(1-\alpha)$ (single symmetric LR) is
recovered as the special case $\text{LR}_+ = \text{LR}_-^{-1} = \alpha/(1-\alpha)$.

### 1.6 Zero-Profit Quotes

$$r^a = p_+ = \sigma(X_t + \ln\text{LR}_+), \qquad r^b = p_- = \sigma(X_t - \ln\text{LR}_-)$$

**Key properties:**
- $\text{LR}_+ = \text{LR}_- = 1$: no information, spread $= 0$.
- $p_t \in \{0,1\}$: spread $= 0$. Resolved market has no information value.
- Near resolution $\text{LR}_+$ rises sharply — microstructural justification for
  halting quotes when $\tau < 5\text{ min}$ (§6.4).

### 1.7 Failure: Descriptive, Not Prescriptive

Glosten-Milgrom explains **why** the spread exists but not how to dynamically
adjust quotes as inventory accumulates. A MM following this model ignores the
risk of holding a large directional position into resolution.

---

## 2. Inventory-Optimal Market Making — Avellaneda-Stoikov (2008)

### 2.1 Setup

$$dS_t = \sigma\,dW_t$$

The MM posts $r^b_t = S_t - \delta^b_t$ and $r^a_t = S_t + \delta^a_t$.
Order arrivals are Poisson with intensity decaying in the half-spread.
Assuming exponential valuations with parameter $\kappa$:

$$\lambda^a(\delta^a) = Ae^{-\kappa\delta^a}, \qquad \lambda^b(\delta^b) = Ae^{-\kappa\delta^b}$$

The exponential is chosen for tractability (memoryless property) and empirical
fit. There is no deep economic reason valuations must be exponential — it is a
reduced-form approximation that breaks near resolution when flow is
predominantly informed.

**Cash and inventory dynamics:**

$$dX_t = r^a_t\,dN^a_t - r^b_t\,dN^b_t, \qquad dq_t = dN^b_t - dN^a_t$$

Terminal wealth: $W_T = X_T + q_T S_T$.

### 2.2 CARA Utility and the Control Problem

$$\max_{\delta^b,\delta^a}\;\mathbb{E}\!\left[-e^{-\gamma(X_T+q_T S_T)}\right]$$

CARA utility $U(W)=-e^{-\gamma W}$ has constant absolute risk aversion
$A(W)=-U''(W)/U'(W)=\gamma$, independent of wealth level.

### 2.3 HJB via Separability Ansatz

Bellman's principle + Itô's lemma gives the HJB:

$$\frac{\partial V}{\partial t}+\frac{1}{2}\sigma^2\frac{\partial^2 V}{\partial S^2}
+\max_{\delta^a}\!\left[\lambda^a(V(t,S,q{-}1,X{+}r^a)-V)\right]
+\max_{\delta^b}\!\left[\lambda^b(V(t,S,q{+}1,X{-}r^b)-V)\right]=0$$

**Ansatz:** $V(t,S,q,X)=-e^{-\gamma(X+qS+h(t,q))}$

Consistent with terminal condition $V(T,S,q,X)=-e^{-\gamma(X+qS)}$ when
$h(T,q)=0$. Substituting and dividing by $V\neq 0$:

$$-\gamma\dot{h}-\frac{1}{2}\gamma^2\sigma^2 q^2
+\max_{\delta^a}\!\left[Ae^{-\kappa\delta^a}(e^{-\gamma(\delta^a+\Delta^-h)}-1)\right]
+\max_{\delta^b}\!\left[Ae^{-\kappa\delta^b}(e^{-\gamma(\delta^b-\Delta^+h)}-1)\right]=0$$

where $\Delta^-h=h(t,q{-}1)-h(t,q)$, $\Delta^+h=h(t,q{+}1)-h(t,q)$.

### 2.4 Optimal Half-Spreads

First-order condition for $\delta^{a*}$:

$$-\kappa(e^{-\gamma(\delta^a+\Delta^-h)}-1)-\gamma e^{-\gamma(\delta^a+\Delta^-h)}=0
\implies e^{-\gamma(\delta^a+\Delta^-h)}=\frac{\kappa}{\kappa+\gamma}$$

$$\boxed{\delta^{a*}=\frac{1}{\gamma}\ln\!\left(1+\frac{\gamma}{\kappa}\right)-\Delta^-h, \qquad
\delta^{b*}=\frac{1}{\gamma}\ln\!\left(1+\frac{\gamma}{\kappa}\right)+\Delta^+h}$$

**Note on A-S notation:** A-S apply the further approximation
$\frac{1}{\gamma}\ln(1+\gamma/\kappa)\approx\frac{1}{\kappa}\ln(1+\gamma/\kappa)$
valid only when $\gamma\ll\kappa$. In prediction markets $\gamma$ and $\kappa$
are of similar magnitude, so we retain the exact form.

### 2.5 Closed Form via Quadratic Ansatz

Linearising $e^x\approx 1+x+x^2/2$ and proposing $h(t,q)=\alpha(t)+\eta(t)q^2$:

Terms in $q^2$: $\dot\eta=-\gamma\sigma^2/2 \implies \eta(t)=\frac{\gamma\sigma^2}{2}\tau$

**Reservation price:**

$$\boxed{\tilde{p}_t = S_t - q_t\gamma\sigma^2\tau}$$

**Optimal half-spread:**

$$\boxed{\frac{\delta^*}{2} = \frac{\gamma\sigma^2\tau}{2}+\frac{1}{\gamma}\ln\!\left(1+\frac{\gamma}{\kappa}\right)}$$

### 2.6 Failures

1. **Approximate HJB**: Taylor expansion breaks for large $\gamma$ or $q$
2. **Gaussian mid-price**: wrong for binary contracts bounded in $[0,1]$
3. **Static parameters**: $\sigma,\kappa$ constant
4. **Symmetric spread**: forces $\delta^b=\delta^a$, ignoring directional signals
5. **No explicit order book**: exponential intensity is reduced-form

---

## 3. Exact Optimal MM with Explicit Order Flow — GLFT (2013)

*Guéant, Lehalle, Fernandez-Tapia*

### 3.1 What GLFT Fixes

GLFT solves the HJB **exactly** without Taylor expansion. The substitution
$h(t,q)=-\frac{1}{\kappa}\ln(u(t,q)/u(t,0))$ transforms the nonlinear ODE
into a **linear system** for $u(t,q)$, solvable in closed form.

The exact optimal half-spreads retain the same first-order conditions as A-S:

$$\delta^{a*}=\frac{1}{\gamma}\ln\!\left(1+\frac{\gamma}{\kappa}\right)-\Delta^-h, \qquad
\delta^{b*}=\frac{1}{\gamma}\ln\!\left(1+\frac{\gamma}{\kappa}\right)+\Delta^+h$$

but now $\Delta^\pm h$ are computed from the exact solution rather than the
quadratic approximation. This naturally produces **asymmetric half-spreads**
$\delta^{a*}\neq\delta^{b*}$ when $q\neq 0$, which A-S cannot.

### 3.2 Parameter Calibration

Calibrate $\kappa$ and $A$ per market by MLE over observed fill rates:

$$\hat\kappa,\hat A = \arg\max\sum_i\Bigl[f_i\ln\lambda(\delta_i;A,\kappa)
-\lambda(\delta_i;A,\kappa)\,\Delta t_i\Bigr]
= \arg\max\sum_i\Bigl[f_i(\ln A-\kappa\delta_i)-Ae^{-\kappa\delta_i}\Delta t_i\Bigr]$$

where $f_i\in\{0,1\}$ indicates a fill in window $i$. **The $-\lambda\Delta t$ term is
not optional:** without it the objective is monotone increasing in $A$ and the
MLE is unbounded. (v2.1 omitted it; the implementation always had it.)

**Identifiability.** The estimator needs variation in $\delta_i$ *and* $\sum_i f_i>0$.
A venue that reports a constant synthetic spread — Manifold — satisfies neither, and
the optimiser lands on the box boundary. `GLFTCalibrator` now refuses to return a
result in that case rather than reporting the bound as a fit.

Run per category (crypto vs political markets have different $\kappa$) and
recalibrate with each new batch of ticks.

### 3.3 Failure: No Directional Signal

GLFT assumes the MM has no view on price direction. In prediction markets,
external signals carry genuine information. A neutral MM leaves alpha on the
table.

---

## 4. Market Making with a Directional Signal — Cartea-Jaimungal (2015)

### 4.1 Augmented Price Process

$$dS_t = \mu_t\,dt + \sigma\,dW_t$$

The **latent drift** $\mu_t$ follows an Ornstein-Uhlenbeck process:

$$d\mu_t = -\phi\mu_t\,dt + \eta\,dB_t, \qquad d\langle W,B\rangle_t=\rho\,dt$$

Quadratic variation terms: $d\langle S,S\rangle_t=\sigma^2\,dt$,
$d\langle\mu,\mu\rangle_t=\eta^2\,dt$, $d\langle S,\mu\rangle_t=\rho\sigma\eta\,dt$.

**Distinction**: $\mu_t$ is the unobservable latent drift. The observable
signal $\hat\mu_t$ (§4.6) is its estimator.

### 4.2 Modified Objective and HJB

Cartea-Jaimungal use linear utility with quadratic inventory penalty:

$$\max_{\delta^b,\delta^a}\;\mathbb{E}\!\left[X_T+q_T S_T-\frac{\gamma}{2}\int_t^T q_s^2\sigma^2\,ds\right]$$

Ansatz $V(t,S,q,X)=X+qS+g(t,q,\mu)$. Since $g$ is linear in $S$,
$\partial_{SS}g=0$. The HJB becomes:

$$\partial_t g+\mu q+\frac{1}{2}\eta^2\partial_{\mu\mu}g
+\rho\sigma\eta\,q\,\partial_\mu g-\phi\mu\partial_\mu g
-\frac{\gamma}{2}\sigma^2 q^2+\mathcal{H}^a[g]+\mathcal{H}^b[g]=0$$

### 4.3 Optimal Half-Spreads

With $\lambda=Ae^{-\kappa\delta}$:

$$\mathcal{H}^a[g]=\max_{\delta^a}\!\left[Ae^{-\kappa\delta^a}(\delta^a-\partial_q g)\right]$$

First-order condition: $-\kappa(\delta^a-\partial_q g)+1=0$, giving:

$$\boxed{\delta^{a*}=\frac{1}{\kappa}+\partial_q g, \qquad \delta^{b*}=\frac{1}{\kappa}-\partial_q g}$$

### 4.4 Quadratic Ansatz and ODEs

Propose $g(t,q,\mu)=\phi_0(t)+\phi_1(t)\mu q+\phi_2(t)q^2$:

- $\phi_1(t)\mu q$: signal-inventory interaction — all directionality lives here
- $\phi_2(t)q^2$: quadratic inventory penalty

$\partial_q g=\phi_1(t)\mu+2\phi_2(t)q$, $\partial_\mu g=\phi_1(t)q$,
$\partial_{\mu\mu}g=0$ (g is linear in $\mu$).

Substituting and separating by powers of $q$ — with terminal condition
$g(T,q,\mu)=0$, hence $\phi_1(T)=\phi_2(T)=0$:

**Terms in $q^2$:**

The cross-variation term $\rho\sigma\eta\,q\,\partial_\mu g=\rho\sigma\eta\,\phi_1 q^2$
also lands here, so the exact ODE is
$\dot\phi_2=\frac{\gamma\sigma^2}{2}-\rho\sigma\eta\,\phi_1(t)$, giving

$$\phi_2(t)=-\frac{\gamma\sigma^2}{2}\tau+\rho\sigma\eta\!\int_t^T\!\phi_1(s)\,ds$$

The second term is second-order in the signal ($\rho\eta$ small relative to
$\gamma\sigma$ for any realistic calibration) and we drop it, keeping

$$\dot\phi_2=\frac{\gamma\sigma^2}{2}
\;\Longrightarrow\;
\phi_2(t)=\phi_2(T)-\int_t^T\dot\phi_2\,ds
=\boxed{-\frac{\gamma\sigma^2}{2}\tau}$$

> **v2.2 (S1).** v2.1 printed $\phi_2=+\gamma\sigma^2\tau/2$. The ODE is integrated
> *backward* from $\phi_2(T)=0$, so $\phi_2$ is **negative** — as it must be, since
> $\phi_2 q^2$ is an inventory *penalty* inside a value function being maximised.

**Terms in $\mu q$:**

$$\dot\phi_1-\phi\phi_1+1=0, \quad \phi_1(T)=0$$

**Solving the ODE for $\phi_1$:** general solution
$\phi_1(t)=Ce^{\phi t}+\frac{1}{\phi}$. Applying $\phi_1(T)=0$:
$C=-\frac{1}{\phi}e^{-\phi T}$. Therefore, adding the measure-change discount
$\rho_\mu\in(0,1]$ (the signal is estimated under $\mathbb{P}$, the quotes live under $\mathbb{Q}$):

$$\boxed{\phi_1(\tau)=\rho_\mu\cdot\frac{1-e^{-\phi\tau}}{\phi}}$$

> **v2.2 (D).** v2.1 printed $\phi_1=(\rho\sigma\eta/\phi)(1-e^{-\phi\tau})$, which is
> **dimensionally impossible**: with $[\mu]=$ nats/year, $\phi_1\hat\mu$ must come out
> in nats, so $[\phi_1]=$ years. But $[\rho\sigma\eta/\phi]=(1/\sqrt{\text{yr}})\cdot
> (\text{nats}/\text{yr}^{3/2})\cdot\text{yr}=\text{nats}/\text{yr}$, giving
> $\phi_1\hat\mu$ in nats²/year². The error came from placing $\rho\sigma\eta$ (which
> belongs to the $q^2$ cross-variation term) into the $\mu q$ equation, where the
> coefficient of $\mu q$ in the HJB is simply $1$.
>
> $\eta$ therefore does **not** enter $\phi_1$. It remains part of the OU model —
> calibrated by `CJCalibrator` and reported — but it prices the *option* to trade on
> future alpha, which lives in $\phi_0$ and does not affect the quotes.

**Interpretation.** $\phi_1$ is exactly the alpha a unit of inventory can still capture:

$$\frac{1}{\mu_t}\int_t^T\mathbb{E}[\mu_s\mid\mu_t]\,ds
=\int_0^\tau e^{-\phi u}\,du=\frac{1-e^{-\phi\tau}}{\phi}$$

**Verification:** $\phi_1(T)=0$ ✓. As $\tau\to\infty$: $\phi_1\to\rho_\mu/\phi$ (bounded —
an infinitely-lived signal is still worth only one mean-reversion time). As $\tau\to 0$:
$\phi_1\to 0$ — no time left to exploit the signal. Units: years ✓.

### 4.5 Reservation Price with Signal

The reservation price is the **midpoint of the two quotes**. From §4.3,
$\text{ask}=S+\delta^{a*}$ and $\text{bid}=S-\delta^{b*}$, so

$$\tilde{p}_t=\frac{(S_t+\delta^{a*})+(S_t-\delta^{b*})}{2}
=S_t+\frac{\delta^{a*}-\delta^{b*}}{2}
=S_t+\partial_q g=S_t+2\phi_2 q_t+\phi_1\hat\mu_t$$

Substituting $\phi_2=-\gamma\sigma^2\tau/2$ and $\phi_1=\rho_\mu(1-e^{-\phi\tau})/\phi$:

$$\boxed{\tilde{p}_t=\underbrace{S_t-q_t\gamma\sigma^2\tau}_{\text{inventory skew (A-S)}}
\;+\;\underbrace{\rho_\mu\frac{1-e^{-\phi\tau}}{\phi}\cdot\hat\mu_t}_{\text{signal skew}}}$$

> **v2.2 (S2).** v2.1 wrote $\tilde p=S-\partial_q g$, contradicting its own §4.3.
> Combined with the $\phi_2$ sign error (S1) the two mistakes cancelled in the
> inventory term — which is why $S-q\gamma\sigma^2\tau$ looked right — and left the
> **signal** term with the wrong sign. Both the derivation above and the economics
> agree on $+$: a positive expected drift means the MM wants to *accumulate* long
> inventory, so it lifts both of its quotes. `cartea_jaimungal.py` had this right.

The half-spread is unchanged from GLFT — the signal only shifts the centre:

$$\delta^*=\gamma\sigma^2\tau+\frac{2}{\gamma}\ln\!\left(1+\frac{\gamma}{\kappa}\right)$$

### 4.6 Observable Signal and Calibration Pipeline

The latent $\mu_t$ is not directly observed. We construct:

$$\hat\mu_t = w_1\cdot\text{OBI}_t + w_2\cdot\text{NewsSignal}_t + w_3\cdot\text{OnChain}_t$$

**Units matter here.** Each raw signal $s_k$ is a normalised score in $[-1,1]$;
$\hat\mu_t$ must be a drift in nats/year. The conversion lives entirely in the
weights $w_k$, which is exactly what the ridge regression of §4.6 estimates
(signals regressed on subsequent log-odds returns per unit time). An
*uncalibrated* ensemble — e.g. the placeholder $w=(1,0,0)$ — returns a score,
not a drift, and must not be fed to the quoter.

**Step 1 — Calibrate weights $w_i$** by Ridge regression on next-tick returns:

$$\hat w=\arg\min_w\sum_t(\Delta p_{t+1}-w^\top f_t)^2+\lambda\|w\|^2$$

**Step 2 — Construct series $\{\hat\mu_t\}$** from features.

**Step 3 — Fit AR(1) to $\{\hat\mu_t\}$** by OLS:

$$\hat\alpha=\frac{\sum_t\hat\mu_t\hat\mu_{t+1}}{\sum_t\hat\mu_t^2}, \qquad
\hat\phi=-\frac{\ln\hat\alpha}{\Delta t}, \qquad
\hat\eta=\hat\nu\sqrt{\frac{2\hat\phi}{1-\hat\alpha^2}}$$

where $\hat\nu^2=\frac{1}{n}\sum_t(\hat\mu_{t+1}-\hat\alpha\hat\mu_t)^2$.

**Step 4 — Estimate $\sigma$** by EWMA on price increments.

**Step 5 — Estimate $\rho$** by sample correlation:

$$\hat\rho=\frac{\text{Cov}(\Delta p_t,\Delta\hat\mu_t)}{\hat\sigma\cdot\hat\eta\cdot\Delta t}$$

### 4.7 Failure: Gaussian Price Process

All models so far assume continuous Brownian motion. For binary contracts:
prices are bounded in $[0,1]$, variance at resolution is Bernoulli, and
near resolution the process exhibits jump-like behaviour.

---

## 5. Logit Jump-Diffusion Kernel and Belief-Volatility Surface

> **v2.1 replaces v1 §5 "Bernoulli surface".**
> Defects corrected: A ($\sigma$ vs $\sigma_B$ conflation), B (variance double-count),
> T (measure ambiguity).

### 5.1 Why the Bernoulli Surface Was Wrong

$\sigma_B(p,\tau) = \sqrt{p(1-p)/\tau}$ is the **unconditional terminal standard
deviation** flattened over $\tau$. It is a parameter of the terminal distribution,
not of the increment distribution. Inserting it into an HJB whose generator is a
Laplacian in $p$ is a category error (defect A). Moreover, having both a
$\sigma_B$ diffusion term and a jump term on $p$ double-counts variance (defect B).

### 5.2 The Correct Kernel: Logit Jump-Diffusion

Under $\mathbb{Q}$ we model $X_t = \text{logit}(p_t)$:

$$dX_t = \mu_X(t,X_t)\,dt + \sigma_b(t,X_t)\,dW_t + \int_{\mathbb{R}} z\,\tilde{N}(dt,dz) \tag{K}$$

where $W$ is a $\mathbb{Q}$-Brownian motion and $\tilde{N}$ is the compensated
jump measure with $\mathbb{Q}$-compensator $\nu_t(dz)\,dt$.

Because $p_t = \sigma(X_t)$ must be a $\mathbb{Q}$-martingale, Itô on $\sigma$
pins down the drift:

$$\mu_X(t,x) = -\frac{1}{\sigma'(x)}\!\left[\tfrac{1}{2}\sigma''(x)\,\sigma_b^2
+ \int_{\mathbb{R}}\!\bigl(\sigma(x{+}z)-\sigma(x)-\sigma'(x)\chi(z)\bigr)\nu_t(dz)\right]
\tag{K-drift}$$

with $\chi(z)=z\cdot\mathbf{1}_{|z|<1}$. This constraint is what v1's
$\sigma\to\sigma_B$ substitution silently broke.

**Key consequence:** $\text{Var}_{\mathbb{Q}}(p_T\mid\mathcal{F}_t) \to p_t(1-p_t)$
as $T \to$ resolution automatically, for any $\sigma_b$, $\nu_t$ trajectory,
because $p_T \in \{0,1\}$. The Bernoulli bound is a model output, not an input.

### 5.3 Belief-Volatility Surface

$\sigma_b(t,X)$ is estimated from realised quadratic variation of $X$ over
short windows (EWMA, tick-rule denoised):

$$\hat{\sigma}_b^2(t) = \lambda\,\hat\sigma_b^2(t^-) + (1-\lambda)\,
\frac{(\Delta X_i)^2}{\Delta t_i},\qquad \lambda=0.94$$

(RiskMetrics convention: 94% weight on history, 6% on the new observation.)

**Sampling is not a detail.** $\sum(\Delta X_i)^2/\Delta t_i$ is a realised-variance
estimator, and realised variance **diverges as $\Delta t\to 0$** in the presence of
microstructure noise: the bid-ask bounce contributes a fixed $(\Delta X)^2$ per tick
while $\Delta t_i\to 0$ in the denominator. On this repository's own tick data
(median $\Delta t \approx 1.1$ s) the tick-by-tick estimator overstates
$\hat\sigma_b$ by roughly an order of magnitude; the noise ratio
$\hat\sigma_b^{\text{tick}}/\hat\sigma_b^{\text{grid}}$ measured 3–16 across markets.

We therefore estimate $\sigma_b$ on a **fixed sampling grid** of
$\Delta t_{\text{sample}}$ (default 60 s) rather than tick-by-tick, which is the
standard first-order defence against noise-induced RV explosion
(Zhang–Mykland–Aït-Sahalia 2005). Increments are additionally winsorised and the
result is clipped to a documented admissible band; a clip is logged, never silent.

Two empirical regularities:

1. $\sigma_b$ is **U-shaped in $p$** (peaked near $p\in\{0.4,0.6\}$, flat near
   boundaries), the opposite of $\sqrt{p(1-p)}$.
2. $\sigma_b$ rises ~25–60% in the 30 min before a scheduled resolution event.

The implementation is `belief_vol_from_ticks()` in `features/microstructure.py`,
which returns $\sigma_b$ in units of $1/\sqrt{\text{year}}$ for direct use in
`sigma_bar_sq = σ_b^2 · τ_\text{years}`.

### 5.4 Reservation Price and Spread in Logit Space

$$\boxed{\tilde{X}(t,q) = X_t - q\cdot\gamma_I\cdot\bar{\sigma}_b^2(t)\cdot\tau} \tag{2.1}$$

$$\boxed{\frac{\delta^*_X}{2} = \frac{\gamma_I\bar{\sigma}_b^2(t)\,\tau}{2}
+ \frac{1}{\gamma_I}\ln\!\left(1+\frac{\gamma_I}{\kappa_x}\right)} \tag{2.2}$$

> **v2.2 (R).** v2.1 wrote the rent term as $\frac{1}{\kappa_x}\ln(1+\gamma_I/\kappa_x)$,
> the A-S approximation that §2.4 explicitly rejects — and which is in fact neither
> the exact form $\frac1{\gamma}\ln(1+\gamma/\kappa)$ nor its $\gamma\ll\kappa$ limit
> $\frac1\kappa$; it sits below both. At $\gamma_I=0.1,\ \kappa_x=0.8$ it understates
> the rent by a factor of 8. Doc and code now both use the exact form.
>
> Note the consequence for calibration: with the exact rent term, $\kappa_x$ must be
> large ($\mathcal{O}(10)$) for the half-spread to be sane. $\kappa_x\!\approx\!0.8$
> implies a 53-cent spread at $p=0.5$.

$$\text{bid}_p = \sigma(\tilde{X} - \delta^*_X/2), \qquad
\text{ask}_p = \sigma(\tilde{X} + \delta^*_X/2)$$

Tick floor (automatic boundary protection):

$$\delta^{\text{quote}}_p = \max\!\bigl(p(1-p)\cdot\delta^*_X,\; \text{tick}(p)\bigr) \tag{2.3}$$

> **v2.2 (T).** v2.1 fixed the tick at \$0.01. That is wrong on both target
> venues: Polymarket declares `minimum_tick_size = 0.001`, and Kalshi publishes a
> per-market ladder (`price_ranges`) whose main band steps by 0.001, with 0.0001
> below \$0.01 and above \$0.99. With a \$0.01 tick, a market quoting
> 0.0030/0.0040 received a bid of 0.0100 — buying at a cent what the book offers
> at four tenths of a cent. $\text{tick}(p)$ is now the step of the band
> containing $p$, read from venue metadata (`normalizer/price_grid.PriceLadder`),
> and both quotes are anchored to that grid — bid down, ask up, so rounding can
> only widen the spread, never tighten it.

```python
# glft.py — espacio logit (implementation)
gamma_eff     = gamma_I * regime.gamma_multiplier      # v2.2: §6.4 now applied
reservation_X = X_t - inventory * gamma_eff * sigma_bar_sq
half_spread_X = gamma_eff * sigma_bar_sq / 2 + (1/gamma_eff) * log(1 + gamma_eff/kappa_x)
bid_p = sigma(reservation_X - half_spread_X)
ask_p = sigma(reservation_X + half_spread_X)
```

---

## 6. Near-Resolution Extension — Jump-Diffusion

### 6.1 Motivation

As $\tau\to 0$, two observations break all diffusion models:

1. **Liquidity dries up**: the last active participants are disproportionately informed
2. **Jump resolution**: the price jumps to 0 or 1 when the outcome becomes known

### 6.2 Jump-Diffusion Process on $X_t$ (v2.1)

> **v2.1 corrections (defects B, C, L):**
> - Process now on $X_t = \text{logit}(p_t)$, not $p_t$ — keeps $p \in (0,1)$ automatically.
> - Jump intensity corrected to model information acceleration (defect L).
> - MM loss formula corrected to second moment (defect C).

The jump-diffusion kernel (K) from §5 already contains the jump term. For the
near-resolution information-acceleration regime, we parameterise the jump intensity as:

$$\boxed{\lambda_J(\tau) = \lambda_0\cdot\left(\frac{\tau^*}{\tau+\varepsilon}\right)^\eta},
\qquad \eta\in(0,1),\; \varepsilon\sim\text{tick spacing in seconds}$$

This is bounded and integrable (unlike the v1 $e^{\beta\tau}$ which is bounded by
$\lambda_0$ at $\tau=0$ and cannot model acceleration), and converges to
$\lambda_0(\tau^*/\varepsilon)^\eta$ as $\tau\to 0$.

Under the martingale constraint, jumps have **zero first moment** ($\mathbb{E}[p_+ - p \mid \text{jump}] = 0$).
Therefore the correct CARA penalty from a jump is **second-moment and quadratic in $q$**:

$$\boxed{\Delta\text{Loss}(q,\tau) = \tfrac{1}{2}\gamma_I q^2\cdot\sigma_J^2\cdot\lambda_J(\tau)\cdot\Delta t}$$

where $\sigma_J^2 = \mathbb{E}[(p_+ - p)^2]$ over the jump distribution. The v1
expression $|q|\cdot p(1-p)\cdot\lambda_J\cdot\Delta t$ was a first-moment term —
incorrect once the martingale constraint is imposed.

### 6.3 Resolution Risk as a Separate Option

Holding a winning Polymarket token involves a short option on the UMA dispute outcome:

$$\text{Settlement value} = 1 - L\cdot\mathbf{1}\{\text{UMA reverses}\}$$

with $L=\$1$ (full token loss). Approximating $\Pr(\text{reverse})\approx r$:
- Non-contested (sports, Chainlink feeds): $r\approx 0.001$
- Contested (ambiguous resolution criteria): $r\approx 0.005{-}0.02$

The option-adjusted inventory cap (guarantees strict positivity for all $r,\psi$):

$$\boxed{q_{\max}(t) = q^0_{\max}\cdot\exp(-r\cdot\psi(\tau)),
\qquad \psi(\tau) = \exp(-\tau/\tau^*)} \tag{8.2}$$

### 6.4 Practical Near-Resolution Rules

Implemented in `features/resolution.py` (`effective_gamma`, `effective_q_max`) and
applied by both quoters — `glft.py` and `cartea_jaimungal.py` — via
`gamma_eff = gamma_I * rf.gamma_multiplier` before the spread and skew are formed.
(Through v2.1 the multiplier was computed but never applied; fixed in v2.2.)

```
τ ≥ 24h:  NORMAL   — q_max = Q·exp(-r·ψ(τ)),  γ_eff = γ
τ < 24h:  WARNING  — q_max ≤ Q/2,             γ_eff = 2γ
τ < 1h:   CRITICAL — q_max ≤ Q/10,            γ_eff = 4γ, halt one side
τ < 5min: HALT     — halt all quoting
```

---

## 7. Cross-Venue Arbitrage Sizing — Kelly Criterion

> **v2.1 corrections (defects H, O, P):**
> - Separate $\gamma_I$ (CARA) from $\varphi_K$ (Kelly fraction).
> - Oracle reversal probability $r$ added; effective probability $p_\text{eff}$.
> - USDC–USD basis $b_U$ included.
> - Denominator $c(1-c)$ explained as down×up, not Bernoulli variance.

### 7.1 Single-Venue Binary Kelly with Frictions

For a binary contract bought at price $c$ with model probability $p$ and
oracle reversal probability $r$ (independent of $Y$):

$$p_\text{eff} = p(1-r) + (1-p)r = p + r(1-2p)$$

After proportional taker fee $f_t$, capital cost $\rho_c\tau$, and USDC basis $b_U$:

$$\boxed{f^* = \varphi_K\cdot\frac{[p + r(1-2p)] - c - f_t - \rho_c\tau - b_U}{c(1-c)}} \tag{8.1}$$

with $\varphi_K \in [0.25, 0.5]$. The denominator $c(1-c)$ is the product
(downside per dollar staked) $\times$ (upside per dollar staked) from
$\arg\max_f\{p\ln(1+f\cdot(1-c)/c)+(1-p)\ln(1-f)\}$ — not the Bernoulli variance
(they coincide only when $c=p$, i.e., zero edge).

Note: when $p > 0.5$, the oracle term $r(1-2p) < 0$ automatically reduces the edge.

### 7.2 Cross-Venue (Kalshi YES + Polymarket NO) Arb Sizing

Buy YES on Kalshi at $c_K$ and NO on Polymarket at $c_P$. Expected joint P&L:

$$\mathbb{E}[\text{P\&L}] = (1 - c_K - c_P) + r_P(2p-1)$$

Static arb exists when $\mathbb{E}[\text{P\&L}] > \Pi_X$ where:

$$\Pi_X = f_{t,K} + f_{t,P} + b_U\cdot\tau + \rho_c\cdot\tau + \text{gas} + \text{slippage}$$

The Polymarket dispute risk $r_P$ enters as outcome-asymmetric P&L, not as a
deterministic premium in $\Pi_X$. When $p\approx 0.5$ the asymmetry vanishes.

### 7.3 Minimum Viable Edge

$$|\varepsilon_t| > \Pi_{\min} := f_t + \text{slippage} + r\cdot c + \rho_c\tau + b_U$$

On a 50-cent contract: $\Pi_{\min}\approx 1.8\%$ (Kalshi) and $\approx 0.5\%$
(Polymarket, politics fee-free) excluding capital cost.

```python
# Fractional Kelly — §8.1 MATH.md v2.1
def kelly_fraction(c, p, r_oracle, f_t, rho_c, tau, b_U=0.0, phi_K=0.25):
    p_eff  = p + r_oracle * (1 - 2 * p)
    numer  = p_eff - c - f_t - rho_c * tau - b_U
    denom  = c * (1 - c)                    # down × up, not Bernoulli variance
    return max(0.0, phi_K * numer / denom)
```

---

## 8. Calibration Framework — Brier Score & Recalibration

### 8.1 Brier Score Decomposition (v2.1 — with within-bin terms)

> **v2.1 correction (defect Q):** The v1 identity $\text{Br}=\text{REL}-\text{RES}+\text{UNC}$
> holds only in expectation. With finite binning into $K$ bins it acquires two
> within-bin terms (Stephenson–Casati–Wilks 2008).

$$\boxed{\text{Br} = \text{REL} - \text{RES} + \text{UNC} + \text{WBV} - 2\,\text{WBC}}$$

where:
- $\text{WBV}_k = \text{Var}_k(f)$ — within-bin forecast variance
- $\text{WBC}_k = \text{Cov}_k(f, o)$ — within-bin forecast–outcome covariance
- The factor 2 comes from the cross-term $-2(f_i-\bar{f}_k)(o_i-\bar{o}_k)$ in $(f_i-o_i)^2$

Defining **generalised resolution** $\text{GRES} := \text{RES} + 2\,\text{WBC} - \text{WBV}$,
the compact form $\text{Br} = \text{REL} - \text{GRES} + \text{UNC}$ is bin-width invariant.

We use $K=20$ quantile bins and report all five components with bootstrap CIs.

### 8.2 Recalibration

> **v2.1 correction (defect R):** Isotonic regression is biased on autocorrelated
> series. Use sequential / out-of-time refit and Venn–Abers predictors instead.

Recommended calibration pipeline:
1. **Out-of-time split**: fit on weeks $[w-4, w-1]$, score on week $w$, roll forward.
2. **Venn–Abers predictors** (Vovk–Petej 2014): valid coverage even under
   exchangeability violations. Default for our pipeline.
3. **Beta calibration** fallback when $|\text{labelled markets}| < 2000$.

### 8.3 Bayesian Signal Update in Log-Odds

With conditionally-independent signals $s_k$ given outcome $Y$:

$$X^{\text{posterior}} = X^{\text{prior}} + \sum_k \ln\Lambda_k(s_k), \qquad
\Lambda_k(s) = \frac{\Pr(s_k=s\mid Y=1)}{\Pr(s_k=s\mid Y=0)}$$

**Valid only after orthogonalisation.** OBI is mechanically caused by the same news
that drives NewsSignal on the same horizon, so $\text{OBI}\perp\text{News}\mid Y$
is empirically violated (defect J). We Cholesky-factor the empirical covariance of
$(s_1,\ldots,s_K)$ and update only on the residuals.

---

## 9. Summary: Model Evolution (v2.1)

| Model | Key contribution | Remaining open |
|-------|-----------------|----------------|
| Glosten-Milgrom | Adverse selection; asymmetric LR+/LR– in log-odds | Trader heterogeneity (MoE) |
| Avellaneda-Stoikov | Inventory-optimal quotes, CARA + HJB in logit | Taylor approx for large $q$ |
| GLFT exact | Exact HJB (ODE system), asymmetric spreads | Numerically costly for large $Q$ |
| Cartea-Jaimungal | OU $\mu_t$ under $\mathbb{P}$; alpha-capture $\phi_1=\rho_\mu(1-e^{-\phi\tau})/\phi$ | Multi-signal orthogonalisation |
| Logit kernel (§5) | Single $\mathbb{Q}$-martingale; $\sigma_b(t,X)$ from noise-robust realised QV | Two-scale / Hawkes-driven $\sigma_b$ |
| Jump-diffusion (§6) | $\lambda_J\propto\tau^{-\eta}$; second-moment loss; oracle option | MOOV2 proposer concentration |
| Kelly v2.1 (§7) | $p_\text{eff}=p+r(1-2p)$; USDC basis; $c(1-c)$ denominator | Latency arbitrage (73% bots) |
| Calibration (§8) | Brier + WBV−2WBC; Venn–Abers; out-of-time refit | Regime change / concept drift |

---

## References

- Glosten, L.R. & Milgrom, P.R. (1985). *Bid, ask and transaction prices in a specialist market with heterogeneously informed traders.* Journal of Financial Economics, 14(1), 71–100.
- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book.* Quantitative Finance, 8(3), 217–224.
- Guéant, O., Lehalle, C.A. & Fernandez-Tapia, J. (2013). *Dealing with the inventory risk: a solution to the market making problem.* Mathematics and Financial Economics, 7(4), 477–507.
- Cartea, Á. & Jaimungal, S. (2015). *Enhancing trading strategies with order book signals.* Applied Mathematical Finance, 23(6), 1–35.
- Kelly, J.L. (1956). *A new interpretation of information rate.* Bell System Technical Journal, 35(4), 917–926.
- Murphy, A.H. (1973). *A new vector partition of the probability score.* Journal of Applied Meteorology, 12(4), 595–600.
- Zhang, L., Mykland, P.A. & Aït-Sahalia, Y. (2005). *A tale of two time scales: determining integrated volatility with noisy high-frequency data.* Journal of the American Statistical Association, 100(472), 1394–1411.
- Stephenson, D.B., Coelho, C.A.S. & Jolliffe, I.T. (2008). *Two extra components in the Brier score decomposition.* Weather and Forecasting, 23(4), 752–757.
