# Mathematical Foundations
### Prediction Market System — Kalshi & Polymarket

> This document develops the mathematical framework underlying the system's
> market-making, arbitrage, and calibration components. Each section is
> motivated by the **failures of the previous model**, building a coherent
> narrative from first principles to the original near-resolution extension.

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $S_t$ | Mid-price of the contract at time $t$ |
| $p_t$ | Implied probability $\in (0,1)$ — for binary contracts $S_t \equiv p_t$ |
| $T$ | Resolution time (fixed) |
| $\tau = T - t$ | Time remaining to resolution |
| $q_t$ | Inventory: net position held by the market maker (signed) |
| $Q$ | Maximum inventory limit $\vert q \vert \leq Q$ |
| $\delta^b, \delta^a$ | Bid and ask half-spreads around mid-price |
| $r^b = p_t - \delta^b$ | Bid quote |
| $r^a = p_t + \delta^a$ | Ask quote |
| $\lambda^b(\delta), \lambda^a(\delta)$ | Arrival intensities of buy/sell market orders |
| $\kappa$ | Order book depth parameter (decay of arrival intensity) |
| $A$ | Baseline order arrival rate |
| $\gamma$ | Risk aversion coefficient of the market maker |
| $\sigma$ | Volatility of the mid-price process |
| $\sigma_B(p,\tau)$ | Bernoulli volatility surface (defined in §5) |
| $W_t$ | Standard Brownian motion (price noise) |
| $B_t$ | Standard Brownian motion (signal noise) |
| $N_t^b, N_t^a$ | Counting processes for buyer/seller-initiated trades |
| $X_t$ | Cash account of the market maker |
| $V(t, p, q, x)$ | Value function of the MM stochastic control problem |
| $\mu_t$ | Latent drift of the price process (OU) |
| $\hat{\mu}_t$ | Estimated signal: $w_1\cdot\text{OBI}_t + w_2\cdot\text{News}_t + w_3\cdot\text{OnChain}_t$ |
| $\phi$ | Mean-reversion speed of $\mu_t$ |
| $\eta$ | Volatility of $\mu_t$ |
| $\rho$ | Correlation: $d\langle W,B\rangle_t = \rho\,dt$ |
| $\Pi$ | Arbitrage profit per unit: $p^P - p^K$ |
| $C$ | Round-trip transaction cost |

All prices are expressed as probabilities $\in [0,1]$. Kalshi quotes in cents
are divided by 100; Polymarket USDC fractional amounts are used directly.

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

### 1.3 Order Arrival Probabilities

An informed trader **buys** only if $V=1$ and **sells** only if $V=0$. A
noise trader buys or sells with equal probability $\frac{1}{2}$:

$$\Pr(\text{buy} \mid V=1) = \alpha\cdot 1 + (1-\alpha)\cdot\tfrac{1}{2} = \tfrac{1+\alpha}{2}$$

$$\Pr(\text{buy} \mid V=0) = \alpha\cdot 0 + (1-\alpha)\cdot\tfrac{1}{2} = \tfrac{1-\alpha}{2}$$

By the law of total probability:

$$\Pr(\text{buy}) = \tfrac{1+\alpha}{2}\cdot\mu + \tfrac{1-\alpha}{2}\cdot(1-\mu)
= \frac{1+\alpha(2\mu-1)}{2}$$

### 1.4 Bayesian Posteriors

Applying Bayes exactly, and defining the likelihood ratio $\Lambda = \frac{1+\alpha}{1-\alpha} > 1$:

$$\boxed{\Pr(V=1 \mid \text{buy}) = \frac{\Lambda\mu}{\Lambda\mu + (1-\mu)}}$$

Symmetrically, an informed trader sells only if $V=0$:

$$\Pr(\text{sell}\mid V=1) = \tfrac{1-\alpha}{2}, \qquad \Pr(\text{sell}\mid V=0) = \tfrac{1+\alpha}{2}$$

$$\boxed{\Pr(V=1 \mid \text{sell}) = \frac{\mu}{\mu + \Lambda(1-\mu)}}$$

A buy is evidence for $V=1$; a sell is evidence against it.

### 1.5 Zero-Profit Quotes

The MM is competitive and risk-neutral. Zero expected profit per trade:

$$r^a = \Pr(V=1\mid\text{buy}), \qquad r^b = \Pr(V=1\mid\text{sell})$$

### 1.6 Equilibrium Spread

Defining $D^a = \frac{1+\alpha}{2}\mu + \frac{1-\alpha}{2}(1-\mu)$ and
$D^b = \frac{1-\alpha}{2}\mu + \frac{1+\alpha}{2}(1-\mu)$:

$$(1+\alpha)D^b - (1-\alpha)D^a = (1-\mu)\left[(1+\alpha)^2-(1-\alpha)^2\right] = 4\alpha(1-\mu)$$

$$\boxed{r^a - r^b = \frac{4\alpha\mu(1-\mu)}{(2D^a)(2D^b)}}$$

**Key properties:**
- $\alpha=0$: spread $=0$. No asymmetric information, no spread needed.
- $\mu\in\{0,1\}$: spread $=0$. Resolved market has no information value.
- Spread maximised at $\mu=0.5$ — maximum uncertainty, maximum value of private information.
- For $\alpha\to 0.5$, $\mu=0.5$: spread $=0.5$. The MM quotes 0.25/0.75 — noise traders exit, liquidity collapses.

There exists a threshold $\alpha^*$ above which the market breaks down entirely.
Near resolution $\alpha$ rises sharply — the microstructural justification for
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

$$\hat\kappa,\hat A = \arg\max\sum_i\ln\lambda(\delta_i;A,\kappa)
= \arg\max\sum_i[\ln A-\kappa\delta_i]$$

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

$\partial_q g=\phi_1(t)\mu+2\phi_2(t)q$, $\partial_\mu g=\phi_1(t)q$

Substituting and separating by powers of $q$:

**Terms in $q^2$:**

$$\dot\phi_2=\frac{\gamma\sigma^2}{2} \implies \phi_2(t)=\frac{\gamma\sigma^2}{2}\tau$$

**Terms in $\mu q$:**

$$\dot\phi_1-\phi\phi_1+\rho\sigma\eta=0, \quad \phi_1(T)=0$$

**Solving the ODE for $\phi_1$:** general solution
$\phi_1(t)=Ce^{\phi t}+\frac{\rho\sigma\eta}{\phi}$. Applying $\phi_1(T)=0$:
$C=-\frac{\rho\sigma\eta}{\phi}e^{-\phi T}$. Therefore:

$$\boxed{\phi_1(t)=\frac{\rho\sigma\eta}{\phi}(1-e^{-\phi\tau})}$$

**Verification:** $\phi_1(T)=0$ ✓. As $\tau\to\infty$: $\phi_1\to\rho\sigma\eta/\phi$ (bounded). As $\tau\to 0$: $\phi_1\to 0$ — no time to exploit the signal.

### 4.5 Reservation Price with Signal

$$\tilde{p}_t=S_t-\partial_q g=S_t-2\phi_2 q_t-\phi_1\hat\mu_t$$

$$\boxed{\tilde{p}_t=\underbrace{S_t-q_t\gamma\sigma^2\tau}_{\text{inventory skew (A-S)}}
\;-\;\underbrace{\frac{\rho\sigma\eta}{\phi}(1-e^{-\phi\tau})\cdot\hat\mu_t}_{\text{signal skew}}}$$

The spread $\delta^*=2/\kappa+\gamma\sigma^2\tau$ is unchanged from GLFT —
the signal only shifts the centre of the quotes.

### 4.6 Observable Signal and Calibration Pipeline

The latent $\mu_t$ is not directly observed. We construct:

$$\hat\mu_t = w_1\cdot\text{OBI}_t + w_2\cdot\text{NewsSignal}_t + w_3\cdot\text{OnChain}_t$$

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

## 5. Volatility Structure of Binary Contracts — Bernoulli Surface

### 5.1 The Core Insight

A binary contract pays $\mathbf{1}_{V=1}$ at $T$. The variance of the
terminal payoff conditional on current information:

$$\text{Var}(V\mid\mathcal{F}_t) = p_t(1-p_t)$$

Distributing over remaining time $\tau=T-t$:

$$\boxed{\sigma_B(p_t,\tau)=\sqrt{\frac{p_t(1-p_t)}{\tau}}}$$

Unlike Black-Scholes where $\sigma$ is exogenous, $\sigma_B$ is **endogenous**
— fully determined by the current probability and time to resolution.

### 5.2 Properties

- Symmetric around $p=0.5$; zero at $p\in\{0,1\}$
- $\sigma_B\to\infty$ as $\tau\to 0$ — small information causes large probability moves
- Maximum uncertainty ($p=0.5$) gives maximum volatility at any given $\tau$

### 5.3 Substitution into Reservation Price and Spread

Replacing $\sigma$ with $\sigma_B(p_t,\tau)$ — the $\tau$ cancels in the
inventory term:

$$\tilde{p}_t = p_t - q_t\gamma\cdot\frac{p_t(1-p_t)}{\tau}\cdot\tau
+ \frac{\rho\sigma_B\eta}{\phi}(1-e^{-\phi\tau})\cdot\hat\mu_t$$

$$\boxed{\tilde{p}_t = p_t - q_t\gamma p_t(1-p_t)
+ \frac{\rho\sigma_B\eta}{\phi}(1-e^{-\phi\tau})\cdot\hat\mu_t}$$

The inventory skew is **purely a function of current probability**, not time.
Risk of holding inventory depends on how uncertain the outcome is, not on
how much time remains.

**Optimal half-spread:**

$$\boxed{\frac{\delta^*}{2}=\frac{\gamma p_t(1-p_t)}{2}+\frac{1}{\gamma}\ln\!\left(1+\frac{\gamma}{\kappa}\right)}$$

```python
def bernoulli_vol(p: float, tau: float) -> float:
    return np.sqrt(p * (1 - p) / tau) if tau > 1e-6 else np.inf

def reservation_price(p, q, gamma, mu_hat, rho, sigma_B, eta, phi, tau):
    inventory_skew = q * gamma * p * (1 - p)
    signal_skew    = (rho * sigma_B * eta / phi) * (1 - np.exp(-phi * tau)) * mu_hat
    return p - inventory_skew + signal_skew

def optimal_half_spread(p, gamma, kappa):
    return gamma * p * (1 - p) / 2 + np.log(1 + gamma / kappa) / gamma
```

---

## 6. Near-Resolution Extension — Jump-Diffusion

### 6.1 Motivation

As $\tau\to 0$, two observations break all diffusion models:

1. **Liquidity dries up**: the last active participants are disproportionately informed
2. **Jump resolution**: the price jumps to 0 or 1 when the outcome becomes known

$\sigma_B\to\infty$ as $\tau\to 0$, causing optimal spreads to diverge.

### 6.2 Proposed Jump-Diffusion Process

For $\tau<\tau^*$:

$$dp_t = \hat\mu_t\,dt + \sigma_B(p_t,\tau)\,dW_t + (J_t-p_t)\,dN_t$$

where:
- $dN_t\sim\text{Poisson}(\lambda_J(\tau)\,dt)$: jump intensity increasing as $\tau\to 0$
- $J_t\in\{0,1\}$: $\Pr(J_t=1)=p_t$
- $(J_t-p_t)$: jump size — at $p=0.7$: YES jump is $+0.3$, NO jump is $-0.7$

Jump intensity:

$$\lambda_J(\tau) = \lambda_0\cdot e^{\beta\tau}$$

Rare far from resolution, increasingly frequent as $\tau\to 0$.

### 6.3 Expected MM Loss from a Resolution Jump

$$\mathbb{E}[\text{Loss}\mid q,p,\tau] = |q|\cdot p(1-p)\cdot\lambda_J(\tau)\cdot\Delta t$$

### 6.4 Practical Near-Resolution Rules

Implemented in `execution/risk/limits.py`:

```
τ < 24h:  Q_max = Q/2,   γ_effective = 2γ
τ < 1h:   Q_max = 1,     halt quoting on inventory-heavy side
τ < 5min: halt all quoting
```

---

## 7. Cross-Venue Arbitrage Sizing — Kelly Criterion

### 7.1 The Arbitrage Setup

When $p^K < p^P$ for the same event on Kalshi and Polymarket:

- Buy YES on Kalshi at $p^K$
- Buy NO on Polymarket at $1-p^P$

Locked profit per unit: $\Pi = p^P - p^K > 0$

Risks: execution risk, resolution divergence, platform risk, liquidity risk.

### 7.2 Kelly Criterion

For win probability $p_{\text{exec}}$ and odds $b=\frac{\Pi}{1-\Pi+C}$:

$$\boxed{f^* = p_{\text{exec}} - \frac{(1-p_{\text{exec}})(1-\Pi+C)}{\Pi}}$$

Use **half-Kelly** in practice: $f_{\text{actual}} = \frac{1}{2}f^*$

### 7.3 Minimum Viable Edge

$f^* > 0$ requires:

$$\boxed{p_{\text{exec}} > \frac{1-\Pi+C}{1+C}}$$

With $C\approx 0.002$ and $p_{\text{exec}}\geq 0.90$: $\Pi_{\min}\approx 1.1\%$

```python
def kelly_fraction(spread, p_exec, cost=0.002, fraction=0.5):
    b = spread / (1 - spread + cost)
    f_full = p_exec - (1 - p_exec) / b
    return max(0.0, fraction * f_full)
```

---

## 8. Calibration Framework — Bayesian Updating & Brier Score

### 8.1 Brier Score Decomposition

$$\text{BS}=\frac{1}{N}\sum_i(p_i-o_i)^2
=\underbrace{\text{REL}}_{\text{calibration}}
-\underbrace{\text{RES}}_{\text{resolution}}
+\underbrace{\text{UNC}}_{\text{uncertainty}}$$

REL $=0$ is perfect calibration. Higher RES means more informative predictions.

### 8.2 Isotonic Recalibration

If REL $> 0$, apply isotonic regression (Pool Adjacent Violators, $O(N)$):

$$\hat p_i=\arg\min_{f\,\text{non-decreasing}}\sum_i(f(p_i)-o_i)^2$$

### 8.3 Bayesian Signal Update

Given prior $p_0$ and likelihood ratio $\Lambda=\alpha/(1-\alpha)$:

$$p_{\text{post}}=\frac{\Lambda p_0}{\Lambda p_0+(1-p_0)}$$

Multiple independent signals combine in log-odds space:

$$\text{logit}(p_{\text{final}})=\text{logit}(p_0)+\sum_k\ln\Lambda_k$$

---

## 9. Summary: Model Evolution

| Model | Key contribution | Remaining gap |
|-------|-----------------|---------------|
| Glosten-Milgrom | Spread from adverse selection, exact Bayesian quotes | Not prescriptive |
| Avellaneda-Stoikov | Dynamic inventory-optimal quotes, CARA + HJB | Approx HJB, no signal, Gaussian $p$ |
| GLFT | Exact HJB, explicit order flow, asymmetric spreads | No directional signal |
| Cartea-Jaimungal | Signal $\hat\mu_t$ shifts reservation price | Gaussian $p$, constant $\sigma$ |
| + Bernoulli surface | Correct $\sigma_B(p,\tau)$, $\tau$ cancels in inventory skew | Near-resolution breaks |
| + Jump-diffusion | Resolution risk, expected loss, halt rules | No closed form |
| + Kelly sizing | Principled arb sizing, minimum viable edge $\Pi_{\min}$ | — |

---

## References

- Glosten, L.R. & Milgrom, P.R. (1985). *Bid, ask and transaction prices in a specialist market with heterogeneously informed traders.* Journal of Financial Economics, 14(1), 71–100.
- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book.* Quantitative Finance, 8(3), 217–224.
- Guéant, O., Lehalle, C.A. & Fernandez-Tapia, J. (2013). *Dealing with the inventory risk: a solution to the market making problem.* Mathematics and Financial Economics, 7(4), 477–507.
- Cartea, Á. & Jaimungal, S. (2015). *Enhancing trading strategies with order book signals.* Applied Mathematical Finance, 23(6), 1–35.
- Kelly, J.L. (1956). *A new interpretation of information rate.* Bell System Technical Journal, 35(4), 917–926.
- Murphy, A.H. (1973). *A new vector partition of the probability score.* Journal of Applied Meteorology, 12(4), 595–600.
