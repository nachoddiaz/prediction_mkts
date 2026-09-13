"""
connectors/kalshi.py
─────────────────────
Kalshi connector — a CFTC-regulated venue.

Auth: RSA signing
  Every REST request carries:
    KALSHI-ACCESS-KEY:       the API key
    KALSHI-ACCESS-TIMESTAMP: Unix timestamp in milliseconds
    KALSHI-ACCESS-SIGNATURE: base64(RSA-SHA256(timestamp + method + path))

  The signature is computed over: timestamp (ms) + method (GET/POST) + path
  (/trade-api/v2/markets). The private key is a PEM file on disk — never in
  the code.

WebSocket:
  A proprietary protocol with login + subscribe commands.
  Channels: "orderbook_delta", "trade", "ticker"
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from connectors.base import (
    BaseConnector,
    SnapshotCallback,
    TickCallback,
)
from normalizer.kalshi_adapter import (
    build_series_cache,
    kalshi_market_to_domain,
    kalshi_orderbook_to_domain,
    kalshi_quote_to_tick,
    kalshi_to_snapshot,
    kalshi_trade_to_tick,
)
from normalizer.schema import Market, MarketSnapshot, MarketStatus, Venue

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-environment URLs
# ---------------------------------------------------------------------------

# ORIGIN only — without the /trade-api/v2 path, which the callers append.
#
# These constants used to include the /trade-api/v2 suffix while get_markets()
# and get_snapshot() appended it again, so EVERY Kalshi REST request went to
# .../trade-api/v2/trade-api/v2/... and returned 404. Since _get() treats 4xx
# as non-retryable and returns None, the connector merely logged "no active
# markets found": Kalshi ingestion had never worked, and failed silently.
KALSHI_REST_DEMO = "https://demo-api.kalshi.co"

# The public read host. `trading-api.kalshi.com` is the historical host and
# today returns 401 even to list markets; `api.elections.kalshi.com` serves
# markets, order books and trades unauthenticated (verified). Since ingestion
# is read-only, that is the correct host. Authenticated trading endpoints live
# on the historical host and will be configured in Phase 4.
KALSHI_REST_PROD = "https://api.elections.kalshi.com"
KALSHI_WS_DEMO = "wss://demo-api.kalshi.co/trade-api/ws/v2"
KALSHI_WS_PROD = "wss://trading-api.kalshi.com/trade-api/ws/v2"

# How many active markets to fetch at startup
# Markets actually ingested, after filtering and ranking by liquidity.
MARKET_FETCH_LIMIT = 100

# Discovery from the trade tape: 4 pages of 1000 trades yield ~800 distinct
# tickers, enough to pick 100 with a real book without scanning the whole
# catalogue on every startup.
TRADE_PAGE_SIZE = 1000
TRADE_SCAN_PAGES = 4

# Markets per request when fetching metadata with `tickers=`.
TICKER_BATCH_SIZE = 100

# REST polling interval when there are no WebSocket credentials.
POLL_INTERVAL_SECONDS = 30


def _to_float(value: object) -> float:
    """Kalshi's numeric fields arrive as strings ('0.0000', '14.00')."""
    try:
        return float(str(value).replace("$", "").replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _liquidity_score(raw: dict[str, Any]) -> float:
    """
    Liquidity proxy used to rank candidates.

    It combines the three fields the API publishes — declared liquidity, 24h
    volume and total volume — because none is reliable on its own: a freshly
    opened market has zero volume but may have a book, and a closing one has
    high volume and an empty book. What actually matters is having TWO sides to
    quote, so a non-null bid and ask are required as well.
    """
    has_book = _to_float(raw.get("yes_bid_dollars")) > 0 and (
        _to_float(raw.get("yes_ask_dollars")) > 0
    )
    score = (
        _to_float(raw.get("liquidity_dollars"))
        + _to_float(raw.get("volume_24h_fp"))
        + 0.1 * _to_float(raw.get("volume_fp"))
    )
    return score * (2.0 if has_book else 1.0)


class KalshiConnector(BaseConnector):
    """
    Kalshi connector.

    Usage:
        connector = KalshiConnector(
            on_tick=writer.enqueue,
            on_snapshot=store.on_snapshot,
        )
        await connector.run()
    """

    def __init__(
        self,
        on_tick: TickCallback,
        on_snapshot: SnapshotCallback,
        api_key: str | None = None,
        private_key_path: str | None = None,
        env: str = "demo",
        **kwargs: object,
    ) -> None:
        """
        Args:
            on_tick:          callback for each Tick
            on_snapshot:      callback for each MarketSnapshot
            api_key:          Kalshi API key (or KALSHI_API_KEY from the env)
            private_key_path: path to the PEM file (or KALSHI_PRIVATE_KEY_PATH)
            env:              "demo" | "prod"
        """
        super().__init__(on_tick, on_snapshot, **kwargs)  # type: ignore[arg-type]

        self._api_key = api_key or os.getenv("KALSHI_API_KEY", "")
        key_path = private_key_path or os.getenv(
            "KALSHI_PRIVATE_KEY_PATH", "./secrets/kalshi_private.pem"
        )

        # Load the RSA private key from disk.
        # Why load it in __init__ rather than per request:
        #   Reading and parsing a PEM file is not free. Loading it once at
        #   construction avoids paying that cost on every request, of which
        #   there can be dozens per second.
        pem_path = Path(key_path)
        if pem_path.exists():
            with open(pem_path, "rb") as f:
                self._private_key = serialization.load_pem_private_key(f.read(), password=None)
        else:
            self._private_key = None
            log.warning("Kalshi private key not found at %s", key_path)

        self._base_url = KALSHI_REST_PROD if env == "prod" else KALSHI_REST_DEMO
        self._ws_url = KALSHI_WS_PROD if env == "prod" else KALSHI_WS_DEMO

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _build_headers(self) -> dict[str, str]:
        """
        Build Kalshi's RSA authentication headers.

        The signature is base64(RSA-SHA256(timestamp_ms + method + path)).
        The timestamp is included so Kalshi can verify the request is not a
        replay of an older one.

        Why RSA and not HMAC:
          Kalshi uses RSA so keys can be rotated without sharing the secret
          with anyone — only the public key goes to Kalshi. HMAC would require
          sharing the secret itself.
        """
        return {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self._api_key,
        }

    def _sign(self, method: str, path: str) -> dict[str, str]:
        """
        Generate the signature headers for one specific request.

        Why a millisecond timestamp:
          Kalshi uses milliseconds for finer resolution in the anti-replay
          window. Requests with a timestamp more than 30s in the past are
          rejected.

        Args:
            method: "GET" | "POST"
            path:   endpoint path, e.g. "/trade-api/v2/markets"

        Returns:
            A dict with the three authentication headers.
        """
        timestamp_ms = str(int(time.time() * 1000))
        message = (timestamp_ms + method + path).encode("utf-8")

        if self._private_key is not None:
            signature = self._private_key.sign(
                message,
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            sig_b64 = base64.b64encode(signature).decode("utf-8")
        else:
            sig_b64 = ""

        return {
            "KALSHI-ACCESS-KEY": self._api_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": sig_b64,
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Implementation of the abstract methods
    # ------------------------------------------------------------------

    async def get_markets(self) -> list[Market]:
        """
        Fetch active Kalshi markets.

        Flow:
          1. GET /series → populate the category cache
          2. Discover markets from the trade tape

        Why /series first:
          A market's category lives on its series, not on the market itself.
          Without the series cache every market would come out as
          category=OTHER. See kalshi_adapter.py for detail.
        """
        # Step 1: populate the category cache
        path = "/trade-api/v2/series"
        series_data = await self._get(
            self._base_url + path,
        )
        if series_data and "series" in series_data:
            build_series_cache(series_data["series"])
            log.debug("Kalshi series cache built: %d series", len(series_data["series"]))

        # Step 2: discover ACTIVE markets from the trade tape.
        #
        # Why not list /markets: the catalogue returns, with or without
        # `status=open`, casi exclusivamente shards KXMVECROSSCATEGORY
        # auto-generated shards — 198 out of every 200 — with zero liquidity
        # and an empty book. Paginating 1000 markets did not surface a single
        # two-sided one, so the system ingested 100 markets of which none had a
        # book: hence depth and OBI coming out identically zero on Kalshi.
        #
        # The trade tape does reflect real activity: 6 pages give ~1000 distinct
        # tickers, and requesting them via `tickers=` returned a bid AND an ask
        # on every one checked. That is the right order of discovery — start
        # where the flow is, not at the catalogue.
        raw_markets = await self._fetch_traded_markets(pages=TRADE_SCAN_PAGES)
        if not raw_markets:
            return []

        raw_markets.sort(key=_liquidity_score, reverse=True)

        markets = []
        for raw in raw_markets:
            if len(markets) >= MARKET_FETCH_LIMIT:
                break
            if _liquidity_score(raw) <= 0.0:
                # A market with neither liquidity nor volume produces no
                # features: there is no book to measure. Ingesting it only
                continue
            try:
                markets.append(kalshi_market_to_domain(raw))
            except Exception as e:
                log.warning("Failed to parse Kalshi market: %s", e)

        log.info(
            "Kalshi: fetched %d liquid markets out of %d scanned",
            len(markets),
            len(raw_markets),
        )
        return markets

    async def _fetch_traded_markets(self, pages: int) -> list[dict[str, Any]]:
        """
        Discover markets with recent activity and return their payloads.

        Two phases:
          1. Paginate /markets/trades to collect distinct tickers. One page of
             1000 trades gives ~200 unique tickers; four pages reach ~800,
             comfortably more than the 100 needed.
          2. Fetch metadata in batches via `tickers=`, which is one request per
             batch rather than one per market.
        """
        tickers: list[str] = []
        seen: set[str] = set()
        cursor: str | None = None

        for _ in range(pages):
            params: dict[str, Any] = {"limit": TRADE_PAGE_SIZE}
            if cursor:
                params["cursor"] = cursor

            data = await self._get(self._base_url + "/trade-api/v2/markets/trades", params=params)
            if not data or not data.get("trades"):
                break

            for trade in data["trades"]:
                ticker = trade.get("ticker")
                if ticker and ticker not in seen:
                    seen.add(ticker)
                    tickers.append(ticker)

            cursor = data.get("cursor")
            if not cursor:
                break

        out: list[dict[str, Any]] = []
        for start in range(0, len(tickers), TICKER_BATCH_SIZE):
            batch = tickers[start : start + TICKER_BATCH_SIZE]
            data = await self._get(
                self._base_url + "/trade-api/v2/markets",
                params={"tickers": ",".join(batch)},
            )
            if data and "markets" in data:
                out.extend(data["markets"])

        return out

    async def get_snapshot(self, market_id: str) -> MarketSnapshot | None:
        """
        Fetch the current state of one Kalshi market.

        market_id has the form "kalshi:KXBTC-26APR22-T85000".
        The raw_id (KXBTC-26APR22-T85000) is extracted to build the URL.
        """
        raw_id = market_id.split(":", 1)[-1]

        # Fetch market and order book in parallel to cut latency
        market_url = f"{self._base_url}/trade-api/v2/markets/{raw_id}"
        orderbook_url = f"{self._base_url}/trade-api/v2/markets/{raw_id}/orderbook"

        market_data, orderbook_data = await asyncio.gather(
            self._get(market_url),
            self._get(orderbook_url),
            return_exceptions=True,
        )

        if not market_data or isinstance(market_data, Exception):
            log.warning("Kalshi: failed to fetch market %s", raw_id)
            return None

        # The response arrives wrapped: {"market": {...}}. get_markets() does
        # unwraps ({"markets": [...]}) but here the whole envelope was passed
        # through, so the adapter looked for 'ticker' at the wrong level and
        # get_snapshot() returned None for EVERY market: Kalshi discovered 100
        # markets and persisted none.
        raw_market = market_data.get("market", market_data)

        try:
            return kalshi_to_snapshot(
                raw_market=raw_market,
                raw_orderbook=orderbook_data if isinstance(orderbook_data, dict) else None,
                timestamp=datetime.now(tz=UTC),
            )
        except Exception as e:
            log.warning("Kalshi: failed to build snapshot for %s: %s", raw_id, e)
            return None

    async def subscribe(self, market_ids: list[str]) -> None:
        """
        Feed the data stream: WebSocket where credentials exist, REST polling
        otherwise.

        Why the fallback:
          Kalshi's WebSocket REJECTS the connection with HTTP 401 when
          unauthenticated (verified: 10 consecutive retries in a real run,
          backing off to 64 s). Without a key the connector stopped at the
          initial snapshots and never emitted another tick: 100 markets
          tracked and 22 ticks for the whole session.

          The read-only REST endpoints are public, so without a key we poll.
          It is slower than the WebSocket but produces a real time series, which
          is what calibration needs.
        """
        if self._private_key is not None and self._api_key:
            await self._subscribe_websocket(market_ids)
        else:
            log.info(
                "Kalshi: no credentials — using REST polling (%ds) instead of WebSocket",
                POLL_INTERVAL_SECONDS,
            )
            await self._poll_rest(market_ids)

    async def _poll_rest(self, market_ids: list[str]) -> None:
        """
        REST polling of the order books of the tracked markets.

        Emits one snapshot per market per round. The interval applies BETWEEN
        rounds, not between markets: with 100 markets at ~0.3 s per request a
        round already takes half a minute.
        """
        while True:
            polled = 0
            for market_id in market_ids:
                # Skip those the refresh loop has already marked resolved.
                # A resolved market has no book: querying it burns one request
                # every 30 s to receive `mid=None` every time. With 93 already
                # resolved sports markets that was ~186 wasted requests per
                # minuto tiradas a la basura.
                if self._market_status.get(market_id, MarketStatus.OPEN) != MarketStatus.OPEN:
                    continue

                try:
                    snapshot = await self.get_snapshot(market_id)
                    if snapshot is not None:
                        await self._on_snapshot(snapshot)
                        polled += 1
                except Exception as e:
                    log.debug("Kalshi: poll error for %s: %s", market_id, e)

            if polled == 0:
                log.info("Kalshi: no open markets left to poll — waiting for rediscovery")

            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def _subscribe_websocket(self, market_ids: list[str]) -> None:
        """
        Subscribe to Kalshi's WebSocket for the given markets.

        Kalshi WebSocket protocol:
          1. Connect
          2. Send a login message with the API key
          3. Send subscribe messages per channel and market
          4. Receive messages indefinitely

        Available channels:
          orderbook_delta → book updates
          trade           → executed trades
          ticker          → price / best bid / best ask changes

        Why raw_ids are extracted before connecting:
          Kalshi's WebSocket uses the native tickers (KXBTC-...), not the
          canonical market_ids (kalshi:KXBTC-...).
        """
        raw_ids = [mid.split(":", 1)[-1] for mid in market_ids]

        async with websockets.connect(self._ws_url) as ws:
            # Paso 1: login
            await ws.send(
                json.dumps(
                    {
                        "id": 1,
                        "cmd": "login",
                        "params": {"api_key": self._api_key},
                    }
                )
            )
            await ws.recv()  # respuesta de login — ignorar

            # Paso 2: suscribirse a channels
            msg_id = 2
            for raw_id in raw_ids:
                for channel in ("orderbook_delta", "trade", "ticker"):
                    await ws.send(
                        json.dumps(
                            {
                                "id": msg_id,
                                "cmd": "subscribe",
                                "params": {
                                    "channels": [channel],
                                    "market_tickers": [raw_id],
                                },
                            }
                        )
                    )
                    msg_id += 1

            log.info("Kalshi WS: subscribed to %d markets", len(raw_ids))

            # Paso 3: loop de mensajes
            async for raw_msg in ws:
                try:
                    await self._handle_ws_message(json.loads(raw_msg))
                except Exception as e:
                    log.warning("Kalshi WS: error handling message: %s", e)

    async def _handle_ws_message(self, msg: dict) -> None:
        """
        Handle one message from Kalshi's WebSocket.

        Tipos de mensaje:
          trade          → construir Tick TRADE → on_tick
          orderbook_delta → construir OrderBook → on_snapshot
          ticker         → construir Tick QUOTE → on_tick
        """
        msg_type = msg.get("type")

        if msg_type == "trade":
            ticker = msg.get("msg", {}).get("market_ticker", "")
            from normalizer.schema import MarketId, Venue

            market_id = MarketId(Venue.KALSHI, ticker)
            tick = kalshi_trade_to_tick(market_id, msg)
            await self._on_tick(tick)

        elif msg_type == "orderbook_delta":
            # Orderbook delta — reconstruir snapshot completo
            msg_data = msg.get("msg", {})
            ticker = msg_data.get("market_ticker", "")
            from normalizer.schema import MarketId, Venue

            market_id = MarketId(Venue.KALSHI, ticker)
            ob = kalshi_orderbook_to_domain(
                market_id,
                {"orderbook": msg_data},
                datetime.now(tz=UTC),
            )
            tick = kalshi_quote_to_tick(market_id, ob)
            if tick:
                snapshot = MarketSnapshot(
                    market=await self._get_market_cached(str(market_id)),
                    orderbook=ob,
                    last_tick=tick,
                )
                await self._on_snapshot(snapshot)

        elif msg_type == "ticker":
            msg_data = msg.get("msg", {})
            ticker = msg_data.get("market_ticker", "")
            from normalizer.schema import MarketId, Price, Size, Tick, TickType, Venue

            market_id = MarketId(Venue.KALSHI, ticker)
            yes_bid = float(msg_data.get("yes_bid", 0)) / 100
            yes_ask = float(msg_data.get("yes_ask", 100)) / 100
            if 0 < yes_bid < yes_ask < 1:
                tick = Tick(
                    market_id=market_id,
                    timestamp=datetime.now(tz=UTC),
                    tick_type=TickType.QUOTE,
                    yes_bid=Price(yes_bid),
                    yes_ask=Price(yes_ask),
                    volume=Size(0.0),
                    side=None,
                )
                await self._on_tick(tick)

    async def _get_market_cached(self, market_id: str) -> Market:
        """
        Return a minimal Market for building snapshots in the WS handler.
        In production this would come from an in-memory market cache.
        """
        from normalizer.schema import (
            Market,
            MarketCategory,
            MarketId,
            MarketStatus,
            Resolution,
        )

        raw_id = market_id.split(":", 1)[-1]
        return Market(
            market_id=MarketId(Venue.KALSHI, raw_id),
            question="",
            category=MarketCategory.OTHER,
            resolution=Resolution(
                resolution_date=datetime.now(tz=UTC),
                resolved_value=None,
            ),
            status=MarketStatus.OPEN,
        )
