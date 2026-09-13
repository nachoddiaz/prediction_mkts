"""
connectors/polymarket.py
─────────────────────────
Polymarket connector — an on-chain venue on Polygon.

Dos APIs:
  Gamma API  → market metadata (no auth)
  CLOB API   → order book and trades in real time (no auth for reads)

WebSocket:
  CLOB WS → price_change events y trade events
  No auth required to subscribe to price events.

Why the full order book is available without auth:
  The book is read from /book, which is public — only order submission
  requires an EIP-712 signature. Trading endpoints arrive in Phase 4, with
  credentials.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from datetime import UTC, datetime

import websockets

from connectors.base import (
    BaseConnector,
    SnapshotCallback,
    TickCallback,
)
from normalizer.polymarket_adapter import (
    _parse_clob_token_ids,
    polymarket_market_to_domain,
    polymarket_merged_book_to_domain,
    polymarket_price_update_to_tick,
    polymarket_quote_to_tick,
    polymarket_trade_to_tick,
)
from normalizer.price_grid import parse_polymarket_tick_size
from normalizer.schema import (
    Market,
    MarketSnapshot,
)

log = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

MARKET_FETCH_LIMIT = 50


class PolymarketConnector(BaseConnector):
    """
    Polymarket connector.

    Diferencia clave respecto a Kalshi:
      Reads require no auth — no special headers needed.
      The token IDs (clobTokenIds) are needed for the WebSocket — they are
      the on-chain identifiers of each market's YES token.
    """

    def __init__(
        self,
        on_tick: TickCallback,
        on_snapshot: SnapshotCallback,
        **kwargs: object,
    ) -> None:
        super().__init__(on_tick, on_snapshot, **kwargs)  # type: ignore[arg-type]
        # Cache: canonical market_id → yes_token_id
        # Needed to build snapshots from WS events, which carry only the
        # token_id rather than the canonical market_id
        self._token_to_market: dict[str, str] = {}
        self._market_tokens: dict[str, tuple[str, str]] = {}
        self._markets_cache: dict[str, Market] = {}

    def _build_headers(self) -> dict[str, str]:
        """Polymarket requires no auth for reads."""
        return {"Accept": "application/json"}

    async def get_markets(self) -> list[Market]:
        """
        Fetch active markets from the Gamma API.

        Why order by volume24hr:
          The highest-volume markets are the most liquid and the most
          interesting for a trading system. Fetching the top N by volume
          guarantees we work with markets that actually have flow.
          These are the markets where an optimal spread is meaningful.
        """
        url = f"{GAMMA_BASE}/markets"
        data = await self._get(
            url,
            params={
                "limit": MARKET_FETCH_LIMIT,
                "active": "true",
                "order": "volume24hr",
                "ascending": "false",
            },
        )

        if not data:
            return []

        raw_markets = data if isinstance(data, list) else data.get("markets", [])
        markets = []

        for raw in raw_markets:
            try:
                # Infer tags where they arrive empty (a known Gamma API bug)
                tags = raw.get("tags", [])
                if not tags:
                    tags = self._infer_tags(raw.get("question", ""))
                raw_enriched = {**raw, "tags": tags}

                market = polymarket_market_to_domain(raw_enriched)
                markets.append(market)

                # Guardar en cache: token_id → market_id
                token_ids = _parse_clob_token_ids(raw)
                if token_ids:
                    yes_id, no_id = token_ids
                    self._token_to_market[yes_id] = str(market.market_id)
                    # The NO token is needed too: its book holds the other
                    # half of the YES liquidity (see merged_book_to_domain).
                    self._market_tokens[str(market.market_id)] = (yes_id, no_id)
                    self._markets_cache[str(market.market_id)] = market

            except Exception as e:
                log.warning("Polymarket: failed to parse market: %s", e)

        log.info("Polymarket: fetched %d active markets", len(markets))
        return markets

    async def get_snapshot(self, market_id: str) -> MarketSnapshot | None:
        """
        Fetch a Polymarket market's full book from /book.

        Why /book rather than /midpoint + /price:
          The previous comment claimed /book "requires EIP-712 auth". That is
          false: it returns HTTP 200 without credentials (verified). What does
          require a signature is SUBMITTING orders, not reading the book.

          The consequence of not using it was serious. /midpoint and /price
          return a single price per side and no size, so the snapshot was
          built with `Size(0.0)`: bid_depth_5 and ask_depth_5 came out 0, and
          with them OBI, the principal component of μ̂. In other words,
          Cartea-Jaimungal degenerated to GLFT on Polymarket because of a
          limitation that did not exist. /book returns the full depth — 153
          levels in the first market checked — with a price and size per level.

        A note on ordering: /book returns bids ASCENDING and asks DESCENDING,
        i.e. both sides with the best price LAST. Nothing is assumed here:
        polymarket_orderbook_to_domain() re-sorts by price.
        """
        market = self._markets_cache.get(market_id)
        if not market:
            return None

        tokens = self._market_tokens.get(market_id)
        if not tokens:
            return None
        yes_token, no_token = tokens

        # Both books in parallel: the NO book supplies the YES liquidity that
        # does not appear in the YES book (see polymarket_merged_book_to_domain).
        yes_book, no_book = await asyncio.gather(
            self._get(f"{CLOB_BASE}/book", params={"token_id": yes_token}),
            self._get(f"{CLOB_BASE}/book", params={"token_id": no_token}),
            return_exceptions=True,
        )

        if not isinstance(yes_book, dict):
            return None
        book_data = yes_book

        try:
            ob = polymarket_merged_book_to_domain(
                market.market_id,
                yes_book,
                no_book if isinstance(no_book, dict) else None,
                datetime.now(tz=UTC),
            )
            if ob.best_bid is None or ob.best_ask is None:
                # One-sided book: there is no mid to record.
                return None

            tick = polymarket_quote_to_tick(market.market_id, ob)

            # /book itself publishes the market's tick size, and it is more
            # reliable than Gamma's metadata because it comes from the engine.
            ladder = parse_polymarket_tick_size(book_data.get("tick_size")) or market.price_ladder
            market_with_grid = (
                market if ladder is market.price_ladder else replace(market, price_ladder=ladder)
            )

            return MarketSnapshot(
                market=market_with_grid,
                orderbook=ob,
                last_tick=tick,
            )

        except Exception as e:
            log.warning("Polymarket: failed to build snapshot for %s: %s", market_id, e)
            return None

    async def subscribe(self, market_ids: list[str]) -> None:
        """
        Subscribe to Polymarket's CLOB WebSocket.

        The CLOB WS uses on-chain token_ids, not canonical market_ids.
        That is why the _token_to_market cache built in get_markets() is
        needed.

        Message types:
          price_change → QUOTE Tick
          trade        → TRADE Tick
          book         → full OrderBook
        """
        # Collect the yes_token_ids for the markets we want
        token_ids = [token for token, mid in self._token_to_market.items() if mid in market_ids]

        if not token_ids:
            log.warning("Polymarket WS: no token IDs found for market_ids")
            return

        async with websockets.connect(CLOB_WS) as ws:
            # Subscribe to every token_id
            await ws.send(
                json.dumps(
                    {
                        "assets_ids": token_ids,
                        "type": "Market",
                    }
                )
            )

            log.info("Polymarket WS: subscribed to %d tokens", len(token_ids))

            async for raw_msg in ws:
                try:
                    msg = json.loads(raw_msg)
                    if isinstance(msg, list):
                        for event in msg:
                            await self._handle_ws_event(event)
                    else:
                        await self._handle_ws_event(msg)
                except Exception as e:
                    log.warning("Polymarket WS: error handling message: %s", e)

    async def _handle_ws_event(self, event: dict) -> None:
        """
        Handle one CLOB WebSocket event.

        event_type:
          price_change → QUOTE Tick via polymarket_price_update_to_tick
          trade        → TRADE Tick via polymarket_trade_to_tick
        """
        event_type = event.get("event_type") or event.get("type", "")
        asset_id = event.get("asset_id", "")

        market_id_str = self._token_to_market.get(asset_id)
        if not market_id_str:
            return

        market = self._markets_cache.get(market_id_str)
        if not market:
            return

        if event_type in ("price_change", "book"):
            tick = polymarket_price_update_to_tick(market.market_id, event)
            if tick:
                await self._on_tick(tick)

        elif event_type == "trade":
            tick = polymarket_trade_to_tick(market.market_id, event)
            await self._on_tick(tick)

    @staticmethod
    def _infer_tags(question: str) -> list[str]:
        """
        Infer tags from the question when the API returns none.
        See polymarket_adapter.py for the same logic — duplicated here to
        avoid importing from the adapter inside the connector.
        """
        q = question.lower()
        if any(w in q for w in ["bitcoin", "btc", "eth", "crypto", "sol"]):
            return ["crypto"]
        if any(w in q for w in ["election", "president", "senate"]):
            return ["politics"]
        if any(w in q for w in ["fed", "rate", "inflation", "gdp"]):
            return ["economics"]
        return []
