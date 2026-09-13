"""
connectors/manifold.py
───────────────────────
Manifold Markets connector — a play-money (Mana) sandbox.

Why Manifold as the sandbox:
  - No auth, no credentials, no financial risk
  - A public API with 500 req/min
  - Same binary market structure as Kalshi and Polymarket
  - Lets the whole pipeline be exercised end to end

Key difference: no native WebSocket.
  Streaming is simulated with periodic polling every POLL_INTERVAL seconds.
  Enough to validate that the writer, feature store and strategies behave
  correctly before connecting to the real APIs.

Limitations:
  - Very low liquidity (play money)
  - No real order book — last_price only
  - No real-time trades — bet polling only
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from connectors.base import (
    BaseConnector,
    SnapshotCallback,
    TickCallback,
)
from normalizer.schema import (
    Market,
    MarketCategory,
    MarketId,
    MarketSnapshot,
    MarketStatus,
    OrderBook,
    OrderBookLevel,
    Price,
    Resolution,
    Side,
    Size,
    Tick,
    TickType,
    Venue,
)

log = logging.getLogger(__name__)

MANIFOLD_BASE = "https://api.manifold.markets/v0"
POLL_INTERVAL = 10  # seconds between polls
MARKET_LIMIT = 20  # markets to fetch


class ManifoldConnector(BaseConnector):
    """
    Manifold Markets connector (sandbox).

    Why polling rather than a WebSocket:
      Manifold has no WebSocket. Polling every 10 s is enough to exercise the
      pipeline — production uses Kalshi and Polymarket, which do have real
      streaming feeds.

    Why it is useful despite the limitations:
      It allows the full system to run (connector → writer → features →
      strategies) locally, with no credentials and no financial risk.
      If it works against Manifold, it works against the real APIs.
    """

    def __init__(
        self,
        on_tick: TickCallback,
        on_snapshot: SnapshotCallback,
        poll_interval: int = POLL_INTERVAL,
        backfill_max: int = 200,
        **kwargs: object,
    ) -> None:
        super().__init__(on_tick, on_snapshot, **kwargs)  # type: ignore[arg-type]
        self._poll_interval = poll_interval
        self._backfill_max = backfill_max
        # Cache of the last price per market_id, to detect changes.
        # If the price has not moved since the last poll, no tick is emitted.
        self._last_prices: dict[str, float] = {}

    def _build_headers(self) -> dict[str, str]:
        """Manifold no requiere auth."""
        return {"Accept": "application/json"}

    async def get_markets(self) -> list[Market]:
        """
        Fetch active binary markets from Manifold.

        Manifold returns many market types (multiple choice, numeric, and so
        on). Only the binary YES/NO ones are kept, since those are the ones
        equivalent to Kalshi's and Polymarket's binary contracts.

        Sorted by trading activity, descending, to prioritise the markets with
        the most data — useful for accumulating ticks quickly for calibration
        and backtesting.
        """
        data = await self._get(
            f"{MANIFOLD_BASE}/search-markets",
            params={
                "term": "",
                "limit": MARKET_LIMIT,
                "sort": "most-popular",
                "filter": "open",
                "contractType": "BINARY",
            },
        )

        if not data or not isinstance(data, list):
            return []

        # Sort by total trade count (totalBets) descending so the busiest
        # markets come first.
        data.sort(key=lambda m: m.get("volume", 0), reverse=True)

        markets = []
        for raw in data:
            try:
                market = self._raw_to_market(raw)
                if market:
                    markets.append(market)
            except Exception as e:
                log.warning("Manifold: failed to parse market: %s", e)

        log.info("Manifold: fetched %d binary markets (sorted by volume desc)", len(markets))
        return markets

    async def get_snapshot(self, market_id: str) -> MarketSnapshot | None:
        """
        Fetch the current state of one Manifold market.

        Manifold has no real order book — only a probability (the mid price),
        so a one-level synthetic book with a fixed spread is constructed.

        Why a fixed 0.02 spread:
          With no real book we do not know the spread. 0.02 (2%) is a
          conservative approximation for a play-money venue. In backtesting
          this is flagged clearly as synthetic.
        """
        raw_id = market_id.split(":", 1)[-1]
        data = await self._get(f"{MANIFOLD_BASE}/market/{raw_id}")

        if not data:
            return None

        try:
            market = self._raw_to_market(data)
            if not market:
                return None

            prob = float(data.get("probability", 0.5))
            ob = self._synthetic_orderbook(market.market_id, prob)
            tick = Tick(
                market_id=market.market_id,
                timestamp=datetime.now(tz=UTC),
                tick_type=TickType.QUOTE,
                yes_bid=Price(ob.best_bid or prob),
                yes_ask=Price(ob.best_ask or prob),
                volume=Size(0.0),
                side=None,
            )
            return MarketSnapshot(market=market, orderbook=ob, last_tick=tick)

        except Exception as e:
            log.warning("Manifold: failed to build snapshot for %s: %s", market_id, e)
            return None

    async def backfill_market(self, market_id: str) -> None:
        """
        Download a market's historical bets and emit them as ticks.
        This seeds DuckDB with real history immediately on startup.
        """
        raw_id = market_id.split(":", 1)[-1]
        log.info("Manifold: backfilling bets for %s...", market_id)

        # How many historical bets are pulled per market.
        #
        # This was hard-coded at 1000 and was the main source of junk: 31,000 of
        # the database's 31,100 rows were this backfill, on markets resolving
        # 117 days out or more. It is now set by backfill_max_ticks (200 by
        # default), enough to seed the σ_b series without flooding the table.
        # 0 disables it.
        if self._backfill_max <= 0:
            return

        # Fetch the most recent bets
        data = await self._get(
            f"{MANIFOLD_BASE}/bets",
            params={
                "contractId": raw_id,
                "limit": self._backfill_max,
            },
        )

        if not data or not isinstance(data, list):
            log.info("Manifold: no historical bets found for %s", market_id)
            return

        # Bets arrive most recent first, so they are reversed to be processed
        # chronologically.
        data.reverse()

        ticks_emitted = 0
        spread = 0.02
        for b in data:
            try:
                prob = b.get("probAfter")
                if prob is None or prob <= 0 or prob >= 1:
                    continue

                created_time = b.get("createdTime")
                if not created_time:
                    continue

                timestamp = datetime.fromtimestamp(created_time / 1000, tz=UTC)

                bid = max(0.001, round(prob - spread / 2, 4))
                ask = min(0.999, round(prob + spread / 2, 4))
                if bid >= ask:
                    bid = round(prob - 0.001, 4)
                    ask = round(prob + 0.001, 4)

                tick = Tick(
                    market_id=MarketId(Venue.MANIFOLD, raw_id),
                    timestamp=timestamp,
                    tick_type=TickType.TRADE,
                    yes_bid=Price(bid),
                    yes_ask=Price(ask),
                    volume=Size(abs(b.get("amount", 0.0))),
                    side=Side.YES if b.get("outcome") == "YES" else Side.NO,
                    # Manifold's bet id. Without it every re-poll reinserted the
                    # same trades: the writer had no way to tell a retry from
                    # two genuine matches in the same millisecond, which here is
                    # the norm (a `yes` and a `no`).
                    source_id=b.get("id"),
                )

                await self._on_tick(tick)
                ticks_emitted += 1
            except Exception as e:
                log.warning("Manifold: error parsing bet for %s: %s", market_id, e)

        log.info("Manifold: backfilled %d ticks for %s", ticks_emitted, market_id)

    async def subscribe(self, market_ids: list[str]) -> None:
        """
        Simulate streaming with periodic polling.

        For each market: fetch the current state and compare it with the last
        known price. If it moved, emit a Tick.

        Why compare against the last price:
          Without a WebSocket we do not know exactly when the price changed.
          Emitting a tick only on movement avoids flooding the writer with
          identical ticks, which would inflate storage without adding any new
          information.
        """
        log.info(
            "Manifold: starting polling loop (%ds interval) for %d markets",
            self._poll_interval,
            len(market_ids),
        )

        while True:
            for market_id in market_ids:
                try:
                    await self._poll_market(market_id)
                except Exception as e:
                    log.warning("Manifold: poll error for %s: %s", market_id, e)

            await asyncio.sleep(self._poll_interval)

    async def _poll_market(self, market_id: str) -> None:
        """
        Fetch a market's state and emit a tick when the price has moved.
        """
        raw_id = market_id.split(":", 1)[-1]
        data = await self._get(f"{MANIFOLD_BASE}/market/{raw_id}")

        if not data:
            return

        prob = float(data.get("probability", 0.0))

        # A resolved market is ALWAYS emitted, even at probability 0 or 1.
        #
        # The previous guard (`if prob <= 0 or prob >= 1: return`) discarded
        # exactly the moment of interest: on resolution Manifold's probability
        # goes to precisely 0 or 1. The single event calibration needs was the
        # single one that never got recorded.
        market = self._raw_to_market(data)
        if market is not None and market.status == MarketStatus.RESOLVED:
            await self._on_snapshot(MarketSnapshot(market=market))
            log.info("Manifold: market resolved %s (prob=%.3f)", market_id, prob)
            return

        if prob <= 0 or prob >= 1:
            return

        last = self._last_prices.get(market_id)

        # Emit a tick when the price moved more than 0.1% since the last poll
        if last is None or abs(prob - last) > 0.001:
            self._last_prices[market_id] = prob

            market = self._raw_to_market(data)
            if not market:
                return

            ob = self._synthetic_orderbook(market.market_id, prob)
            tick = Tick(
                market_id=market.market_id,
                timestamp=datetime.now(tz=UTC),
                tick_type=TickType.QUOTE,
                yes_bid=Price(ob.best_bid or prob),
                yes_ask=Price(ob.best_ask or prob),
                volume=Size(0.0),
                side=None,
            )
            snapshot = MarketSnapshot(
                market=market,
                orderbook=ob,
                last_tick=tick,
            )
            await self._on_snapshot(snapshot)
            log.debug(
                "Manifold: price update %s %.4f → %.4f",
                market_id,
                last or 0,
                prob,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _raw_to_market(self, raw: dict) -> Market | None:
        """
        Convert a raw Manifold market into the canonical domain.

        Manifold uses slugs as IDs ("will-btc-hit-150k-2026") rather than
        tickers or condition_ids.
        """
        slug = raw.get("id") or raw.get("slug")
        if not slug:
            return None

        # Close date — Manifold uses closeTime in Unix ms
        close_ms = raw.get("closeTime")
        if not close_ms:
            return None

        from datetime import datetime

        resolution_date = datetime.fromtimestamp(close_ms / 1000, tz=UTC)

        # Outcome, when resolved
        resolved_value: float | None = None
        if raw.get("isResolved"):
            resolution = raw.get("resolution", "")
            if resolution == "YES":
                resolved_value = 1.0
            elif resolution == "NO":
                resolved_value = 0.0

        status = MarketStatus.RESOLVED if raw.get("isResolved") else MarketStatus.OPEN

        return Market(
            market_id=MarketId(Venue.MANIFOLD, slug),
            question=raw.get("question", slug),
            category=self._infer_category(raw.get("question", "")),
            resolution=Resolution(
                resolution_date=resolution_date,
                resolved_value=resolved_value,
            ),
            status=status,
        )

    @staticmethod
    def _synthetic_orderbook(
        market_id: MarketId,
        prob: float,
        spread: float = 0.02,
    ) -> OrderBook:
        """
        Build a one-level synthetic order book from the probability.

        Why synthetic:
          Manifold has no CLOB; the probability is the mid price. We build
          bid = prob - spread/2 and ask = prob + spread/2 to preserve the
          schema invariant (bid < ask).

        Args:
            market_id: the market's MarketId
            prob:      the current probability (mid price)
            spread:    the fixed synthetic spread (default 2%)
        """
        bid = max(0.001, round(prob - spread / 2, 4))
        ask = min(0.999, round(prob + spread / 2, 4))

        # Guarantee bid < ask even when prob sits at an extreme
        if bid >= ask:
            bid = round(prob - 0.001, 4)
            ask = round(prob + 0.001, 4)

        return OrderBook(
            market_id=market_id,
            timestamp=datetime.now(tz=UTC),
            bids=(OrderBookLevel(Price(bid), Size(0.0)),),
            asks=(OrderBookLevel(Price(ask), Size(0.0)),),
        )

    @staticmethod
    def _infer_category(question: str) -> MarketCategory:
        q = question.lower()
        if any(w in q for w in ["bitcoin", "btc", "eth", "crypto"]):
            return MarketCategory.CRYPTO
        if any(w in q for w in ["election", "president", "senate", "vote"]):
            return MarketCategory.POLITICS
        if any(w in q for w in ["fed", "rate", "inflation", "gdp"]):
            return MarketCategory.ECONOMICS
        return MarketCategory.OTHER
