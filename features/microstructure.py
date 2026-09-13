"""
features/microstructure.py
───────────────────────────
Microstructure features from order books and tick series.
Only market-observable signals are computed here.

Tres secciones:

  1. SNAPSHOT  — operate on a single OrderBook or Tick.
                 O(1); used on the feature store's hot path.

  2. SERIES    — operate on DataFrames of historical ticks.
                 Used for calibration and notebook analysis.

  3. PIPELINE  — high-level entry point: reads from DuckDB, computes
                 everything and returns a dict ready to persist.
                 persist into the features table.

Mapping to MATH.md:
  OBI            → w_1 en μ̂_t = w_1·OBI + w_2·News + w_3·OnChain (§4.6)
  quoted_spread  → compared against the GLFT optimal δ* (§3)
  belief_vol     → σ_b(X) from the quadratic variation of X = logit(p) (§5.3)
  ewma_vol       → tick-level counterpart of σ_b; noise diagnostic (§5.3)
  mu_hat         → proxy for μ̂_t until the full signal is calibrated (§4.6)
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from normalizer.schema import OrderBook, Tick
from storage.reader import MarketDataReader

if TYPE_CHECKING:
    from features.signals.ensemble import SignalEnsemble

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constante EWMA
#
# λ = 0.94 is the RiskMetrics standard for daily data. For intraday tick data,
# higher values (0.97-0.99) react faster to regime changes — worth tuning per
# market during calibration.
# ---------------------------------------------------------------------------
EWMA_LAMBDA: float = 0.94

# Seconds per year — single annualisation factor for the whole module.
SECONDS_PER_YEAR: float = 365.25 * 24 * 3600

# ---------------------------------------------------------------------------
# Sampling of belief volatility
#
# Realised quadratic variation Σ(ΔX)²/Δt DIVERGES as Δt → 0 under microstructure
# noise: the bid-ask bounce contributes a roughly fixed (ΔX)² per tick while Δt
# shrinks in the denominator. This is not a theoretical nicety — on this repo's
# real data (true median Δt of 1.1 s) the tick-by-tick estimator produced
# σ_b as large as 405,538, six orders of magnitude too big, and the backtester
# aborted with OverflowError when mapping the half-spread through the sigmoid.
#
# The standard first-order defence is to estimate on a FIXED GRID rather than
# tick by tick (Zhang-Mykland-Aït-Sahalia 2005). 60 s is the usual compromise:
# long enough for the noise to average out, short enough to track an intraday
# regime.
# ---------------------------------------------------------------------------
VOL_SAMPLE_SECONDS: float = 60.0

# Floor on Δt. Guards against duplicate or out-of-order timestamps — 72 of 1019
# in the busiest market of the current database.
MIN_DT_SECONDS: float = 1e-3

# Admissible band for σ_b (annualised, in logit space).
#
# This is not a calibration, it is a NONSENSE detector, and the bar belongs
# where it separates the impossible from the merely extreme.
#
# 50 was far too low. On real Kalshi data it clipped 76 markets in a single
# minute, with raw values from 55 to 270 — and those were not errors: they were
# sports markets resolving in HOURS. Belief volatility genuinely explodes near
# resolution; that is precisely the regime MATH.md §6 describes, not a
# measurement failure. Clipping them biased down exactly the case the
# near-resolution model exists to handle.
#
# 300 still catches what is truly broken — the 405,538 produced by the old
# estimator sits three orders of magnitude above it — without mutilating the
# resolution regime. The quoter's real protection remains MAX_HALF_SPREAD_X.
SIGMA_B_MIN: float = 0.0
SIGMA_B_MAX: float = 300.0

# Minimum grid points before the EWMA can be trusted.
#
# At λ=0.94 the half-life is ~11 observations: below that the estimator is
# essentially the first increment, not a volatility. A freshly discovered
# market has 2 or 3 points, and that noise ended up clipped at the cap and
# persisted as if it were signal. Below this minimum we return 0.0, which makes
# the quoter fall back to the rent term alone: a wide, prudent spread, which is
# the correct answer to "I do not know the volatility yet".
MIN_GRID_POINTS: int = 10

# Increment winsorising: trims the upper percentile of |ΔX| before
# accumulating. A single resolution jump must not set the level of σ_b for the
# rest of the session.
WINSOR_QUANTILE: float = 0.995


# ══════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — SNAPSHOT
# Operate on a single domain object.
# O(1) — used on the feature store's hot path.
# ══════════════════════════════════════════════════════════════════════


def order_book_imbalance(ob: OrderBook, levels: int = 5) -> float:
    """
    Order Book Imbalance (OBI) ∈ [-1, 1].

    OBI = (V_bid - V_ask) / (V_bid + V_ask)

    Interpretation:
      +1 → all liquidity on the bid (extreme buying pressure)
      -1 → all liquidity on the ask (extreme selling pressure)
       0 → balanced book

    Mapping to MATH.md:
      This is the w_1·OBI_t component of the signal μ̂_t in §4.6.
      OBI positivo → drift alcista esperado → reservation price sube
      via the signal skew φ_1(t)·μ̂_t.

    Why the top N levels and not the full book:
      Distant levels have little bearing on the immediate price.
      Top 5 is the standard choice in the microstructure literature.
      Illiquid markets may also have very few levels to begin with.

    Args:
        ob:     order book snapshot (canonical domain object)
        levels: number of levels to include

    Returns:
        float in [-1, 1]; 0.0 when the book is empty
    """
    bid_vol = ob.bid_depth(levels)
    ask_vol = ob.ask_depth(levels)
    denom = bid_vol + ask_vol

    if denom < 1e-9:
        return 0.0

    return float((bid_vol - ask_vol) / denom)


def quoted_spread(ob: OrderBook) -> float | None:
    """
    Absolute spread between best bid and best ask.

    δ = r^a - r^b ∈ [0, 1]

    This is the immediate cost of crossing the book as a taker.
    In GLFT, δ* is the optimal spread the maker should quote — comparing it
    against the observed quoted spread tells us whether the incumbent maker
    is tighter or wider than the theoretical optimum.

    Returns:
        float in [0, 1]; None when the book is incomplete
    """
    return ob.spread


def relative_spread(ob: OrderBook) -> float | None:
    """
    Spread relativo al mid-price.

    δ_rel = δ / mid

    Normalises the spread by the price level, making liquidity comparable
    across markets trading at different probabilities.

    Example:
      A market at 50%, spread 0.02 → δ_rel = 4%
      A market at 5%,  spread 0.02 → δ_rel = 40% (far less liquid)

    Returns:
        float ≥ 0; None when the book is incomplete
    """
    mid = ob.mid
    spr = ob.spread

    if mid is None or spr is None or mid < 1e-9:
        return None

    return float(spr / mid)


# volatilidad en espacio logit
def epoch_seconds(ts: pd.Series) -> np.ndarray:
    """
    Convert a timestamp column to seconds since epoch, whatever the dtype
    resolution.

    Why not `.astype("int64") / 1e9`:
      That pattern assumes datetime64[ns]. DuckDB returns **datetime64[us]**,
      so dividing by 1e9 produced Δt values a THOUSAND times too small. In this
      repo the "measured" median Δt was 1.1 ms when the real one is 1.097 s;
      since σ_b² = Σ(ΔX)²/Δt, that alone inflated σ_b by √1000 ≈ 31.6×, on top
      of the microstructure-noise divergence. Two compounding defects.

      Subtracting the epoch with `.dt.total_seconds()` is exact for any unit
      (s, ms, us, ns) and for columns with or without a timezone.
    """
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    seconds: np.ndarray = (
        (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy(dtype=float)
    )
    return seconds


def _logit_increments_on_grid(
    ticks_df: pd.DataFrame,
    sample_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project the mid series onto a fixed time grid and return (ΔX, Δt_years)
    between consecutive non-empty cells.

    Why a grid and not the raw ticks:
      Σ(ΔX)²/Δt is a realised-variance estimator, and RV diverges as Δt → 0
      under microstructure noise: the bid-ask bounce contributes a roughly
      constant (ΔX)² per tick while Δt shrinks. Sampling at a
      fixed interval is the first-order defence (Zhang-Mykland-Aït-Sahalia).

    Why the last mid in each cell rather than the mean:
      Averaging introduces a moving-average component that biases the
      autocovariance of the increments downward. Taking the last observed
      as differences between prices that were actually quoted.

    Returns empty arrays when there are too few cells to form an increment.
    """
    if "timestamp" not in ticks_df.columns or len(ticks_df) < 2:
        return np.empty(0), np.empty(0)

    df = ticks_df[["timestamp", "mid"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "mid"]).sort_values("timestamp")
    if len(df) < 2:
        return np.empty(0), np.empty(0)

    epoch = epoch_seconds(df["timestamp"])  # seconds, robust to dtype resolution
    bucket = np.floor(epoch / sample_seconds).astype("int64")

    # Last tick within each grid cell
    keep = np.append(bucket[1:] != bucket[:-1], True)
    epoch_g = epoch[keep]
    mids_g = np.clip(df["mid"].to_numpy(dtype=float)[keep], 1e-6, 1 - 1e-6)
    if len(mids_g) < 2:
        return np.empty(0), np.empty(0)

    x_g = np.log(mids_g / (1.0 - mids_g))
    d_x = np.diff(x_g)
    dt_s = np.maximum(np.diff(epoch_g), MIN_DT_SECONDS)

    finite = np.isfinite(d_x) & np.isfinite(dt_s)
    return d_x[finite], dt_s[finite] / SECONDS_PER_YEAR


def _ewma_of_squared_increments(
    d_x: np.ndarray,
    dt_years: np.ndarray,
    ewma_lambda: float,
) -> float:
    """
    RiskMetrics EWMA over (ΔX)²/Δt, with winsorised increments.

        σ²_t = λ·σ²_{t-1} + (1-λ)·(ΔX_t)²/Δt_t

    λ weights the HISTORY (0.94 = 94% on the prior estimate, 6% on the new point).
    The v2.1 docstring had λ and (1-λ) swapped relative to its own code and to
    and to RiskMetrics; here doc, code and MATH.md §5.3 agree.

    Winsorising prevents a single resolution jump from setting the level of σ_b
    for the rest of the session.
    """
    if len(d_x) == 0:
        return 0.0

    if len(d_x) >= 20:
        cap = float(np.quantile(np.abs(d_x), WINSOR_QUANTILE))
        if cap > 0.0:
            d_x = np.clip(d_x, -cap, cap)

    per_step = d_x**2 / dt_years
    var = float(per_step[0])
    for value in per_step[1:]:
        var = ewma_lambda * var + (1.0 - ewma_lambda) * float(value)
    return max(var, 0.0)


def belief_vol_from_ticks(
    ticks_df: pd.DataFrame,
    ewma_lambda: float = EWMA_LAMBDA,
    sample_seconds: float = VOL_SAMPLE_SECONDS,
    market_id: str | None = None,
) -> float:
    """
    Estimate σ_b from the realised quadratic variation of X = logit(p),
    sampled on a fixed grid. MATH.md v2.2 §5.3.

    σ_b comes out in 1/√year, which is what
        sigma_bar_sq = σ_b² · τ_years   (adimensional)
    in glft.py requires: each increment is normalised by Δt_years before the EWMA.

    Four defences against microstructure noise, in this order:
      1. Grid sampling at `sample_seconds` (60 s by default).
      2. A floor on Δt and rejection of non-finite increments.
      3. Winsorising of |ΔX| at the 99.5th percentile.
      4. Clipping to [SIGMA_B_MIN, SIGMA_B_MAX], with a warning log.

    Without (1), on this repo's real data (true median Δt of 1.1 s) the
    estimator returned σ_b up to 405,538 and the backtester aborted.

    Args:
        ticks_df:       DataFrame with 'timestamp' and 'mid' columns.
        ewma_lambda:    weight on history (RiskMetrics: 0.94).
        sample_seconds: step of the sampling grid.
        market_id:      only used in the log message when clipping occurs.

    Returns:
        Annualised σ_b. 0.0 when the grid holds too little data.
    """
    d_x, dt_years = _logit_increments_on_grid(ticks_df, sample_seconds)
    if len(d_x) < MIN_GRID_POINTS:
        # Series too short to estimate anything. 0.0 does not mean "no
        # volatility" but "no estimate", which the quoter turns into a wide spread.
        log.debug(
            "belief_vol_insufficient_history: market=%s grid_points=%d min=%d",
            market_id or "?",
            len(d_x),
            MIN_GRID_POINTS,
        )
        return 0.0

    sigma_b = math.sqrt(_ewma_of_squared_increments(d_x, dt_years, ewma_lambda))

    if not math.isfinite(sigma_b) or sigma_b > SIGMA_B_MAX:
        log.warning(
            "belief_vol_clipped: market=%s raw=%.6g max=%.1f n_grid=%d sample_s=%.0f",
            market_id or "?",
            sigma_b,
            SIGMA_B_MAX,
            len(d_x),
            sample_seconds,
        )
        return SIGMA_B_MAX

    return max(sigma_b, SIGMA_B_MIN)


def noise_ratio_from_ticks(
    ticks_df: pd.DataFrame,
    ewma_lambda: float = EWMA_LAMBDA,
    sample_seconds: float = VOL_SAMPLE_SECONDS,
) -> float:
    """
    Ratio of the tick-by-tick volatility estimate to the grid estimate.

    This is the classical microstructure-noise diagnostic: absent noise both
    estimate the same quantity and the ratio is ≈ 1. The more bid-ask bounce
    there is, the more the tick-level estimate diverges and the higher the ratio.

    It replaces the v2.1 comparison of ewma_vol against bernoulli_vol, which was
    invalid: it compared a quantity in price space against one in logit space —
    logit space — different units.

    Returns:
        σ_b(tick) / σ_b(grid). 0.0 when not computable.
    """
    grid = belief_vol_from_ticks(ticks_df, ewma_lambda, sample_seconds)
    tick = belief_vol_from_ticks(ticks_df, ewma_lambda, sample_seconds=MIN_DT_SECONDS)
    if grid <= 0.0:
        return 0.0
    return tick / grid


# REMOVED: bernoulli_vol_from_tick — obsolete under MATH.md v2.1.
# Volatility is now computed from the quadratic variation of logit(p) via
# belief_vol_from_ticks(), which operates on tick series rather than single ticks.


# ══════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — SERIES
# Operate on DataFrames of historical ticks.
# Used in calibration and in notebook analysis.
# ══════════════════════════════════════════════════════════════════════


def ewma_vol(
    ticks_df: pd.DataFrame,
    lam: float = EWMA_LAMBDA,
) -> float:
    """
    Volatilidad EWMA TICK A TICK de X = logit(p), anualizada.

        σ²_t = λ·σ²_{t-1} + (1-λ)·(ΔX_t)²/Δt_t

    λ weights the HISTORY. At λ = 0.94: 94% on the previous estimate, 6% on the
    new observation. (The v2.1 docstring wrote the formula with λ and (1-λ)
    swapped relative to its own code and to RiskMetrics.)

    Difference from belief_vol_from_ticks:
      This one samples at the native tick frequency; belief_vol samples on a
      fixed 60 s grid. Both estimate the SAME quantity absent microstructure
      noise, so their ratio — noise_ratio_from_ticks() — measures how much
      noise there is: ≈1 with none, rising with the bid-ask bounce.

    Why logit space now:
      Through v2.1 this function operated on Δp (price space) while belief_vol
      operated on ΔX (logit space). Comparing them to detect a "jump regime",
      which is what its own docstring claimed, was mathematically invalid: they
      are quantities in different units. MATH.md v2.1
      already defined everything in logit; the code had not been updated.

    Args:
        ticks_df: DataFrame with 'mid' and 'timestamp' columns.
        lam:      decay parameter ∈ (0, 1).

    Returns:
        Annualised tick-level σ in logit space. 0.0 with fewer than 2 ticks.

    Note: by not subsampling, this estimator IS noise-sensitive by construction
    — which is exactly its purpose here. Do not feed it to the quoter; use
    belief_vol_from_ticks() for that.
    """
    d_x, dt_years = _logit_increments_on_grid(ticks_df, sample_seconds=MIN_DT_SECONDS)
    if len(d_x) == 0:
        return 0.0
    return float(math.sqrt(_ewma_of_squared_increments(d_x, dt_years, lam)))


def ewma_vol_series(
    ticks_df: pd.DataFrame,
    lam: float = EWMA_LAMBDA,
) -> pd.Series:
    """
    Time series of EWMA volatility in logit space — one estimate per tick.

    Useful for visualising how volatility evolves in notebooks and for spotting
    jump regimes by eye. Like ewma_vol(), it is tick-by-tick and therefore
    sensitive to microstructure noise: a diagnostic tool, not a quoter input.


    Returns:
        pd.Series sharing the index of ticks_df.
        NaN in the first element (no prior estimate).
    """
    if len(ticks_df) < 2:
        return pd.Series([float("nan")] * len(ticks_df), index=ticks_df.index)

    df = ticks_df.sort_values("timestamp").reset_index(drop=True)
    # Logit space, as in ewma_vol() and belief_vol_from_ticks() — v2.1 computed
    # this series on Δp while the scalar version used ΔX.
    mids = np.clip(df["mid"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(mids / (1.0 - mids))
    dp = np.diff(x)
    dt_s = np.diff(epoch_seconds(df["timestamp"]))
    dt_years = np.maximum(dt_s, MIN_DT_SECONDS) / SECONDS_PER_YEAR

    vols = [float("nan")]
    var = float("nan")

    for r, t in zip(dp, dt_years, strict=False):
        if t <= 0 or math.isnan(var):
            var = r**2 / t if t > 0 else float("nan")
        else:
            var = lam * var + (1 - lam) * (r**2 / t)
        vols.append(math.sqrt(max(var, 0.0)) if not math.isnan(var) else float("nan"))

    return pd.Series(vols, index=df.index).reindex(ticks_df.index)


def obi_series(
    ticks_df: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    Approximate OBI over a series of TRADE ticks.

    Without the full order book at every tick, OBI is approximated from
    trade flow:
      YES trade (aggressive buy)  → +1 (buying pressure)
      NO  trade (aggressive sell) → -1 (selling pressure)
      QUOTE tick                  →  0 (carries no directional information)

    Normalised rolling sum over a window of N ticks.

    Args:
        ticks_df: DataFrame with a 'side' column ('yes'|'no'|None)
        window:   rolling window, in number of ticks

    Returns:
        pd.Series of approximate OBI ∈ [-1, 1]
    """
    if "side" not in ticks_df.columns:
        return pd.Series([0.0] * len(ticks_df), index=ticks_df.index)

    signal = ticks_df["side"].map({"yes": 1.0, "no": -1.0}).fillna(0.0)
    rolling_sum = signal.rolling(window, min_periods=1).sum()
    rolling_count = signal.abs().rolling(window, min_periods=1).sum()

    obi = rolling_sum / rolling_count.replace(0, float("nan"))
    return obi.fillna(0.0)


def spread_timeseries_from_df(ticks_df: pd.DataFrame) -> pd.Series:
    """
    Time series of spreads from a DataFrame of ticks.
    Requires 'yes_bid' and 'yes_ask' columns.
    """
    if "yes_bid" not in ticks_df.columns or "yes_ask" not in ticks_df.columns:
        raise ValueError("ticks_df must have columns yes_bid and yes_ask")
    return (ticks_df["yes_ask"] - ticks_df["yes_bid"]).rename("spread")


# ══════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — PIPELINE
# Reads from DuckDB, computes every feature, returns a dict ready to persist
# in the features table.
# ══════════════════════════════════════════════════════════════════════


def compute_features_from_db(
    market_id: str,
    reader: MarketDataReader,
    tau_years: float,
    ewma_window: int = 50,
    obi_levels: int = 5,
    ensemble: SignalEnsemble | None = None,
    news: float = 0.0,
    onchain: float = 0.0,
    orderbook: OrderBook | None = None,
    tick: Tick | None = None,
) -> dict[str, Any] | None:
    """
    Main pipeline — reads from DuckDB and computes every feature.

    Called by features/store.py on each new tick or snapshot. Returns a dict
    ready for writer.write_features_sync().

    Flujo:
      1. Read the latest order book → OBI, quoted_spread, relative_spread
      2. Read the last N ticks      → EWMA vol, current mid
      3. Compute σ_b            → from the mid series
      4. μ̂                     → ensemble.compute_mu_hat(obi, news, onchain)
                                  or the OBI proxy when no ensemble is fitted

    Args:
        market_id:   canonical "venue:raw_id" string
        reader:      a MarketDataReader instance (with an open connection)
        tau_years:   time to resolution in years (from Resolution.tau)
        ewma_window: number of historical ticks for the EWMA
        obi_levels:  book levels used for OBI
        ensemble:    fitted SignalEnsemble; None = use OBI as the proxy
        news:        normalised news signal ∈ [-1, 1]
        onchain:     normalised on-chain signal ∈ [-1, 1]

    Returns:
        Dict of all features, or None when there is not enough data.
    """
    # --- 1. Order book for OBI and spreads ---
    #
    # `orderbook` lets the caller pass the book IN HAND instead of re-reading it.
    # This is not an optimisation: the connector enqueues the snapshot into the
    # writer, which flushes in batches, so re-reading here returned the PREVIOUS
    # book — or none at all on a market's first snapshot, in which case no
    # feature was computed. That is why Kalshi and Polymarket produced no
    # features unless a flush was forced before calling in here.
    obi_val = 0.0
    q_spread = None
    r_spread = None
    mid_from_ob = None

    if orderbook is not None:
        best_bid = orderbook.best_bid
        best_ask = orderbook.best_ask
        bid_depth = orderbook.bid_depth(obi_levels)
        ask_depth = orderbook.ask_depth(obi_levels)
        has_book = True
    else:
        ob_df = reader.latest_orderbook(market_id)
        has_book = not ob_df.empty
        if has_book:
            best_bid = ob_df["best_bid"].iloc[0]
            best_ask = ob_df["best_ask"].iloc[0]
            bid_depth = ob_df["bid_depth_5"].iloc[0]
            ask_depth = ob_df["ask_depth_5"].iloc[0]

    if has_book:
        # OBI from the book's pre-computed depths
        denom = float(bid_depth + ask_depth)
        if denom > 1e-9:
            obi_val = float((bid_depth - ask_depth) / denom)

        # Spread from best bid/ask
        if best_bid is not None and best_ask is not None:
            bid_f = float(best_bid)
            ask_f = float(best_ask)
            q_spread = ask_f - bid_f
            mid_from_ob = (bid_f + ask_f) / 2.0
            if mid_from_ob > 1e-9:
                r_spread = q_spread / mid_from_ob

    # --- 2. Last N ticks for the EWMA and the current mid ---
    #
    # Missing history does NOT abort the computation. The volatilities need a
    # series — without one they come out 0.0 and are marked as such — but OBI,
    # spread and τ come from the book in hand and are perfectly valid features.
    #
    # Previously this returned None as soon as `latest_ticks` came back empty,
    # which is the normal case on a market's FIRST snapshot: the writer has not
    # flushed yet. The result was that a new market produced no features until
    # the second cycle — and with connectors that snapshot each market once per
    # round, never.
    ticks_df = reader.latest_ticks(market_id, n=ewma_window)

    if ticks_df.empty and tick is None and not has_book:
        return None

    if not ticks_df.empty:
        timestamp = ticks_df["timestamp"].iloc[0]
        ticks_asc = ticks_df.sort_values("timestamp").reset_index(drop=True)
        ewma = ewma_vol(ticks_asc)
        bvol = belief_vol_from_ticks(ticks_asc, market_id=market_id)
    else:
        timestamp = tick.timestamp if tick is not None else datetime.now(tz=UTC)
        ewma = 0.0
        bvol = 0.0

    bvol_stored = float(bvol) if not math.isinf(bvol) else None

    # --- 4. μ̂ via ensemble o proxy OBI ---
    if ensemble is not None:
        mu_hat = ensemble.compute_mu_hat(obi=obi_val, news=news, onchain=onchain)
    else:
        mu_hat = obi_val

    # --- Venue from market_id ---
    venue = market_id.split(":")[0] if ":" in market_id else "unknown"

    # Convertir timestamp a datetime aware
    if hasattr(timestamp, "to_pydatetime"):
        ts = timestamp.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
    else:
        ts = datetime.now(tz=UTC)

    return {
        "market_id": market_id,
        "venue": venue,
        "timestamp": ts,
        "obi": round(obi_val, 6),
        "quoted_spread": round(float(q_spread), 6) if q_spread is not None else None,
        "relative_spread": round(float(r_spread), 6) if r_spread is not None else None,
        "belief_vol": round(bvol_stored, 6) if bvol_stored is not None else None,
        "ewma_vol": round(ewma, 6),
        "tau_years": round(tau_years, 8),
        "mu_hat": round(mu_hat, 6),
    }


def compute_features_batch(
    market_ids: list[str],
    reader: MarketDataReader,
    tau_map: dict[str, float],
) -> list[dict[str, Any]]:
    """
    Compute features for several markets at once.
    Markets without enough data are excluded (None results dropped).

    Args:
        market_ids: lista de market_id strings
        reader:     instancia de MarketDataReader
        tau_map:    dict {market_id: tau_years}
    """
    results = []
    for mid in market_ids:
        tau = tau_map.get(mid, 0.0)
        row = compute_features_from_db(mid, reader, tau)
        if row is not None:
            results.append(row)
    return results
