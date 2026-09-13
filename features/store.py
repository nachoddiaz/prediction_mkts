"""
features/store.py
──────────────────
Feature store orchestrator — wires microstructure.py, resolution.py, the
reader and the writer into a single entry point.

Responsibilities:
  1. Receive a market_id plus a new snapshot/tick
  2. Read what is needed from DuckDB (order book, recent ticks)
  3. Compute every feature (microstructure + resolution)
  4. Persist them into the features table

Why this file rather than calling microstructure directly:
  The feature store centralises the "when and how to compute" logic. The
  connector only calls store.on_tick() — it needs to know neither which
  features exist nor how they are computed. Adding a new feature means
  touching this file only.

Relation to MATH.md:
  This file implements the §4.6 pipeline:
    ticks → μ̂_t = w_1·OBI + w_2·News + w_3·OnChain
  For now μ̂_t = OBI (a proxy) until the w_i are calibrated in notebooks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from features.microstructure import compute_features_from_db
from normalizer.schema import Market, MarketSnapshot, OrderBook, Tick
from storage.reader import MarketDataReader
from storage.writer import MarketDataWriter

if TYPE_CHECKING:
    from features.signals.ensemble import SignalEnsemble
    from features.signals.news import NewsSignal
    from features.signals.onchain import OnChainSignal

log = logging.getLogger(__name__)


class FeatureStore:
    """
    Compute and persist features for a set of active markets.

    Production use (called by the connector, async):

        store = FeatureStore(reader, writer)

        # On every tick or snapshot from the stream
        await store.on_tick(tick, market)
        await store.on_snapshot(snapshot)

    Use from scripts and tests (synchronous):

        store = FeatureStore(reader, writer)
        store.compute_and_store(market_id, tau_years)
    """

    def __init__(
        self,
        reader: MarketDataReader,
        writer: MarketDataWriter,
        ensemble: SignalEnsemble | None = None,
        news_signal: NewsSignal | None = None,
        onchain_signal: OnChainSignal | None = None,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._ensemble = ensemble
        self._news_signal = news_signal
        self._onchain_signal = onchain_signal

    # ------------------------------------------------------------------
    # Main API — called by the connector
    # ------------------------------------------------------------------

    def compute_and_store(
        self,
        market_id: str,
        tau_years: float,
        ewma_window: int = 50,
        orderbook: OrderBook | None = None,
        tick: Tick | None = None,
    ) -> bool:
        """
        Compute every feature for one market and persist them.

        Why tau_years is passed in from outside:
          tau is computed by the connector from market.resolution.tau, which
          already holds the resolution date. Passing it as a parameter saves
          the store an extra query against the markets table.

        Args:
            market_id:   canonical "venue:raw_id" string
            tau_years:   time to resolution, in years
            ewma_window: number of ticks for the EWMA

        Returns:
            True if features were computed and persisted, False if there was
            not enough data.
        """
        news_val = self._news_signal.get(market_id) if self._news_signal else 0.0
        onchain_val = (
            self._onchain_signal.get(condition_id=market_id) if self._onchain_signal else 0.0
        )

        row = compute_features_from_db(
            market_id=market_id,
            reader=self._reader,
            tau_years=tau_years,
            ewma_window=ewma_window,
            ensemble=self._ensemble,
            news=news_val,
            onchain=onchain_val,
            orderbook=orderbook,
            tick=tick,
        )

        if row is None:
            log.debug("No data for features: %s", market_id)
            return False

        self._writer.write_features_sync([row])
        log.debug(
            "Features stored: %s | obi=%.3f belief_vol=%.4f tau=%.4f",
            market_id,
            row.get("obi", 0),
            row.get("belief_vol") or 0,
            tau_years,
        )
        return True

    def compute_and_store_batch(
        self,
        markets: list[Market],
    ) -> int:
        """
        Compute and persist features for several markets.

        Used by the connector at startup to process every active market
        before streaming begins.

        Args:
            markets: list of domain Market objects

        Returns:
            Number of markets whose features were persisted successfully.
        """
        rows = []
        for market in markets:
            tau = market.resolution.tau
            mid = str(market.market_id)
            news_val = self._news_signal.get(mid) if self._news_signal else 0.0
            onchain_val = (
                self._onchain_signal.get(condition_id=mid) if self._onchain_signal else 0.0
            )
            row = compute_features_from_db(
                market_id=mid,
                reader=self._reader,
                tau_years=tau,
                ensemble=self._ensemble,
                news=news_val,
                onchain=onchain_val,
            )
            if row is not None:
                rows.append(row)

        if rows:
            self._writer.write_features_sync(rows)
            log.info("Batch features stored: %d/%d markets", len(rows), len(markets))

        return len(rows)

    # ------------------------------------------------------------------
    # Connector hooks — invoked on each stream event
    # ------------------------------------------------------------------

    def on_tick(self, tick: Tick, market: Market) -> bool:
        """
        Hook called by the connector on every incoming tick.

        Why recompute features on every tick:
          OBI and the EWMA change with each tick, and GLFT needs current
          features to produce correct quotes. The computation is O(N) over
          ewma_window ticks — microseconds.

        Args:
            tick:   the tick that just arrived from the stream
            market: market metadata (to obtain tau)

        Returns:
            True if features were computed and persisted, False if there was
            no data.
        """
        return self.compute_and_store(
            market_id=str(market.market_id),
            tau_years=market.resolution.tau,
        )

    def on_snapshot(self, snapshot: MarketSnapshot) -> bool:
        """
        Hook called by the connector on a complete snapshot
        (market + orderbook + tick).

        Why a snapshot and not just a tick:
          The snapshot includes the order book, which is what allows OBI to be
          computed from real depth rather than from a flow proxy. Where a
          snapshot is available, the features are more accurate.

        Args:
            snapshot: the connector's complete MarketSnapshot

        Returns:
            True si features calculadas y persistidas.
        """
        if snapshot.last_tick is None:
            return False

        # The book is passed IN HAND: re-reading it from DuckDB returned the
        # previous snapshot — or none the first time — because the writer
        return self.compute_and_store(
            market_id=str(snapshot.market.market_id),
            tau_years=snapshot.market.resolution.tau,
            orderbook=snapshot.orderbook,
            tick=snapshot.last_tick,
        )

    # ------------------------------------------------------------------
    # Recent feature lookup — for the execution engine
    # ------------------------------------------------------------------

    def latest(self, market_id: str) -> dict[str, Any] | None:
        """
        Return a market's most recent features as a dict.

        Used by the execution engine before computing quotes: it needs OBI,
        belief_vol and tau_years in native form, and a plain dict avoids going
        through pandas on the hot path.

        Returns:
            Dict of the most recent features, or None when there is no data.
        """
        df = self._reader.latest_features(market_id)
        if df.empty:
            return None

        row = df.iloc[0]
        return {
            "market_id": market_id,
            "timestamp": row["timestamp"],
            "obi": float(row["obi"]) if row["obi"] is not None else 0.0,
            "quoted_spread": float(row["quoted_spread"])
            if row["quoted_spread"] is not None
            else None,
            "relative_spread": float(row["relative_spread"])
            if row["relative_spread"] is not None
            else None,
            "belief_vol": float(row["belief_vol"]) if row["belief_vol"] is not None else None,
            "ewma_vol": float(row["ewma_vol"]) if row["ewma_vol"] is not None else 0.0,
            "tau_years": float(row["tau_years"]) if row["tau_years"] is not None else 0.0,
            "mu_hat": float(row["mu_hat"]) if row["mu_hat"] is not None else 0.0,
        }
