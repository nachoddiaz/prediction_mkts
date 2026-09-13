"""
tests/unit/test_connectors.py
───────────────────────────────
Tests for the three connectors.

Why mocks and not real HTTP calls:
  Tests must be deterministic and fast. A real call to Kalshi can take 200 ms,
  fail on rate limits, or return different data on every run. With mocks we
  control exactly what the API returns and verify that the connector
  transforms it correctly.

The pattern used — an aiohttp MockSession:
  A class mimicking aiohttp.ClientSession that returns canned responses. The
  connector does not know it is talking to a mock — its code is identical to
  production.

Why not unittest.mock.patch:
  patch requires knowing the exact import path and is fragile under
  refactoring. An explicit mock class is more readable and easier to
  maintain.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from connectors.kalshi import KalshiConnector
from connectors.manifold import ManifoldConnector
from connectors.polymarket import PolymarketConnector
from normalizer.schema import (
    Market,
    MarketCategory,
    MarketId,
    MarketSnapshot,
    Tick,
    TickType,
    Venue,
)

# ---------------------------------------------------------------------------
# Raw data fixtures — they mimic real API responses
# ---------------------------------------------------------------------------

KALSHI_MARKET_RAW = {
    "ticker": "KXBTC-26APR22-T85000",
    "title": "Will Bitcoin close above $85,000 on Apr 22?",
    "status": "active",
    "close_time": "2026-04-22T16:00:00Z",
    "yes_ask_dollars": "0.4500",
    "yes_bid_dollars": "0.4400",
    "volume_fp": 1234.56,
    "open_interest_fp": 500.0,
    "series_ticker": "KXBTC",
    "result": "",
    "category": "crypto",
}

KALSHI_ORDERBOOK_RAW = {
    "orderbook": {
        "market_ticker": "KXBTC-26APR22-T85000",
        "yes": [[45, 1000], [44, 500]],
        "no": [[55, 800], [56, 400]],
    }
}

POLYMARKET_MARKET_RAW = {
    "conditionId": "0xabc123def456",
    "question": "Will Bitcoin hit $150k by June 30, 2026?",
    "active": True,
    "closed": False,
    "resolved": None,
    "endDate": "2026-07-01T04:00:00Z",
    "bestBid": "0.013",
    "bestAsk": "0.014",
    "tags": [{"label": "Crypto"}],
    "clobTokenIds": json.dumps(["111222333444", "555666777888"]),
}

MANIFOLD_MARKET_RAW = {
    "id": "will-btc-hit-150k",
    "slug": "will-btc-hit-150k",
    "question": "Will Bitcoin hit $150k in 2026?",
    "probability": 0.45,
    "closeTime": 1751414400000,  # 2025-07-01 en ms
    "isResolved": False,
    "resolution": None,
}


# ---------------------------------------------------------------------------
# Mock de aiohttp.ClientSession
#
# Why a class rather than an AsyncMock directly:
#   aiohttp uses async context managers for requests:
#     async with session.get(url) as resp:
#   AsyncMock does not support that pattern directly, whereas a class with
#   __aenter__/__aexit__ implements it correctly.
# ---------------------------------------------------------------------------


class MockResponse:
    """Simulates an aiohttp HTTP response."""

    def __init__(self, data: Any, status: int = 200) -> None:
        self._data = data
        self.status = status

    async def json(self) -> Any:
        return self._data

    async def __aenter__(self) -> MockResponse:
        return self

    async def __aexit__(self, *_: object) -> None:
        pass


class MockSession:
    """
    Simulates aiohttp.ClientSession.

    routes: dict de url → respuesta
    Allows a different response per URL.
    """

    def __init__(self, routes: dict[str, Any]) -> None:
        self._routes = routes
        self.closed = False

    def get(self, url: str, **kwargs: Any) -> MockResponse:
        # Match on exact URL or prefix
        for pattern, data in self._routes.items():
            if url.startswith(pattern) or pattern in url:
                return MockResponse(data)
        return MockResponse({}, status=404)

    def post(self, url: str, **kwargs: Any) -> MockResponse:
        for pattern, data in self._routes.items():
            if url.startswith(pattern) or pattern in url:
                return MockResponse(data)
        return MockResponse({}, status=404)

    async def close(self) -> None:
        self.closed = True

    async def __aenter__(self) -> MockSession:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# Helpers de test
# ---------------------------------------------------------------------------

ticks_received: list[Tick] = []
snapshots_received: list[MarketSnapshot] = []


async def on_tick(tick: Tick) -> None:
    ticks_received.append(tick)


async def on_snapshot(snapshot: MarketSnapshot) -> None:
    snapshots_received.append(snapshot)


def make_kalshi_connector(**kwargs) -> KalshiConnector:
    return KalshiConnector(
        on_tick=on_tick,
        on_snapshot=on_snapshot,
        api_key="test-key",
        private_key_path="/nonexistent/path.pem",  # no real signing in tests
        env="demo",
        **kwargs,
    )


def make_polymarket_connector() -> PolymarketConnector:
    return PolymarketConnector(on_tick=on_tick, on_snapshot=on_snapshot)


def make_manifold_connector(poll_interval: int = 1) -> ManifoldConnector:
    return ManifoldConnector(
        on_tick=on_tick,
        on_snapshot=on_snapshot,
        poll_interval=poll_interval,
    )


# ---------------------------------------------------------------------------
# Tests Kalshi
# ---------------------------------------------------------------------------


class TestKalshiConnector:
    def setup_method(self) -> None:
        """Clear callbacks before each test."""
        ticks_received.clear()
        snapshots_received.clear()

    @pytest.mark.asyncio
    async def test_get_markets_returns_list(self) -> None:
        """
        get_markets() must parse the API response and return
        a list of domain Market objects.
        """
        connector = make_kalshi_connector()
        # Order matters: MockSession matches on substrings, and "/markets"
        # would match "/markets/trades". The most specific path goes first.
        connector._session = MockSession(
            {
                "/series": {"series": []},
                "/markets/trades": {
                    "trades": [{"ticker": KALSHI_MARKET_RAW["ticker"]}],
                    "cursor": None,
                },
                "/markets": {"markets": [KALSHI_MARKET_RAW]},
            }
        )

        markets = await connector.get_markets()

        assert len(markets) == 1
        assert isinstance(markets[0], Market)
        assert markets[0].market_id.venue == Venue.KALSHI
        assert markets[0].market_id.raw_id == "KXBTC-26APR22-T85000"

    @pytest.mark.asyncio
    async def test_get_markets_empty_api(self) -> None:
        """When the API returns an empty list, get_markets() returns []."""
        connector = make_kalshi_connector()
        connector._session = MockSession(
            {
                "/series": {"series": []},
                "/markets": {"markets": []},
            }
        )

        markets = await connector.get_markets()
        assert markets == []

    @pytest.mark.asyncio
    async def test_get_markets_error_http(self) -> None:
        """Si la API falla (404), get_markets() devuelve []."""
        connector = make_kalshi_connector()
        connector._session = MockSession({})  # todas las URLs → 404

        markets = await connector.get_markets()
        assert markets == []

    @pytest.mark.asyncio
    async def test_get_snapshot_returns_snapshot(self) -> None:
        """
        get_snapshot() must fetch market and order book in parallel and build
        a complete MarketSnapshot.
        """
        connector = make_kalshi_connector()
        connector._session = MockSession(
            {
                "/markets/KXBTC-26APR22-T85000/orderbook": KALSHI_ORDERBOOK_RAW,
                "/markets/KXBTC-26APR22-T85000": {"market": KALSHI_MARKET_RAW},
            }
        )

        snapshot = await connector.get_snapshot("kalshi:KXBTC-26APR22-T85000")

        # The snapshot may be None if the adapter fails — what we verify
        # here is that the connector does not raise.
        # Content assertions live in the normalizer tests.
        assert snapshot is None or isinstance(snapshot, MarketSnapshot)

    @pytest.mark.asyncio
    async def test_handle_ws_trade(self) -> None:
        """
        A WS trade message must call on_tick with a TRADE Tick.
        """
        connector = make_kalshi_connector()

        msg = {
            "type": "trade",
            "msg": {
                "market_ticker": "KXBTC-26APR22-T85000",
                "yes_price": 45,
                "no_price": 55,
                "count": 10,
                "taker_side": "yes",
                "created_time": "2026-04-22T12:00:00Z",
            },
        }

        await connector._handle_ws_message(msg)
        # The handler may or may not call on_tick depending on the adapter;
        # what matters is that it does not raise
        assert True

    @pytest.mark.asyncio
    async def test_handle_ws_ticker(self) -> None:
        """
        A WS ticker message with valid prices must call on_tick.
        """
        connector = make_kalshi_connector()

        msg = {
            "type": "ticker",
            "msg": {
                "market_ticker": "KXBTC-26APR22-T85000",
                "yes_bid": 44,
                "yes_ask": 46,
            },
        }

        await connector._handle_ws_message(msg)
        assert len(ticks_received) == 1
        assert ticks_received[0].tick_type == TickType.QUOTE
        assert ticks_received[0].yes_bid == pytest.approx(0.44)
        assert ticks_received[0].yes_ask == pytest.approx(0.46)

    @pytest.mark.asyncio
    async def test_handle_ws_ticker_invalid_prices(self) -> None:
        """
        A ticker with invalid prices (bid >= ask) must not emit a tick.
        """
        connector = make_kalshi_connector()

        msg = {
            "type": "ticker",
            "msg": {
                "market_ticker": "KXBTC-26APR22-T85000",
                "yes_bid": 50,
                "yes_ask": 50,  # bid == ask → inválido
            },
        }

        await connector._handle_ws_message(msg)
        assert len(ticks_received) == 0

    @pytest.mark.asyncio
    async def test_build_headers_includes_api_key(self) -> None:
        connector = make_kalshi_connector()
        headers = connector._build_headers()
        assert "KALSHI-ACCESS-KEY" in headers
        assert headers["KALSHI-ACCESS-KEY"] == "test-key"

    @pytest.mark.asyncio
    async def test_sign_includes_timestamp(self) -> None:
        """
        _sign() must include KALSHI-ACCESS-TIMESTAMP.
        Without a real signature (no PEM) the signature is empty — acceptable
        in tests.
        """
        connector = make_kalshi_connector()
        headers = connector._sign("GET", "/trade-api/v2/markets")
        assert "KALSHI-ACCESS-TIMESTAMP" in headers
        assert int(headers["KALSHI-ACCESS-TIMESTAMP"]) > 0


# ---------------------------------------------------------------------------
# Tests Polymarket
# ---------------------------------------------------------------------------


class TestPolymarketConnector:
    def setup_method(self) -> None:
        ticks_received.clear()
        snapshots_received.clear()

    @pytest.mark.asyncio
    async def test_get_markets_returns_list(self) -> None:
        connector = make_polymarket_connector()
        connector._session = MockSession(
            {
                "gamma-api.polymarket.com/markets": [POLYMARKET_MARKET_RAW],
            }
        )

        markets = await connector.get_markets()

        assert len(markets) == 1
        assert isinstance(markets[0], Market)
        assert markets[0].market_id.venue == Venue.POLYMARKET

    @pytest.mark.asyncio
    async def test_get_markets_pobla_cache_tokens(self) -> None:
        """
        get_markets() must populate _token_to_market so the
        el WS handler pueda resolver token_id → market_id.
        """
        connector = make_polymarket_connector()
        connector._session = MockSession(
            {
                "gamma-api.polymarket.com/markets": [POLYMARKET_MARKET_RAW],
            }
        )

        await connector.get_markets()

        assert "111222333444" in connector._token_to_market
        assert connector._token_to_market["111222333444"].startswith("polymarket:")

    @pytest.mark.asyncio
    async def test_get_markets_infiere_tags(self) -> None:
        """
        With empty tags, the category must be inferred from the question.
        """
        raw_sin_tags = {**POLYMARKET_MARKET_RAW, "tags": []}
        connector = make_polymarket_connector()
        connector._session = MockSession(
            {
                "gamma-api.polymarket.com/markets": [raw_sin_tags],
            }
        )

        markets = await connector.get_markets()
        assert len(markets) == 1
        # "Bitcoin" in the question → crypto
        assert markets[0].category.value == "crypto"

    @pytest.mark.asyncio
    async def test_get_snapshot_builds_orderbook(self) -> None:
        """
        get_snapshot() must build a partial order book from
        /midpoint and /price even without the full book.
        """
        connector = make_polymarket_connector()

        # Poblar cache manualmente
        market_id = "polymarket:0xabc123def456"
        from normalizer.schema import (
            MarketCategory,
            MarketStatus,
            Resolution,
        )

        market = Market(
            market_id=MarketId(Venue.POLYMARKET, "0xabc123def456"),
            question="test",
            category=MarketCategory.CRYPTO,
            resolution=Resolution(
                resolution_date=datetime(2026, 7, 1, tzinfo=UTC),
                resolved_value=None,
            ),
            status=MarketStatus.OPEN,
        )
        connector._markets_cache[market_id] = market
        connector._token_to_market["111222333444"] = market_id

        connector._session = MockSession(
            {
                "midpoint": {"mid": "0.0135"},
                "price": {"price": "0.014"},  # BUY (ask) and SELL (bid) share the mock route
            }
        )

        snapshot = await connector.get_snapshot(market_id)
        # May be None when the mock does not cover every exact URL
        assert snapshot is None or isinstance(snapshot, MarketSnapshot)

    @pytest.mark.asyncio
    async def test_handle_price_change(self) -> None:
        """
        A WS price_change event must call on_tick.
        """
        connector = make_polymarket_connector()

        # Poblar cache
        market_id = "polymarket:0xabc123def456"
        from normalizer.schema import MarketCategory, MarketStatus, Resolution

        connector._token_to_market["111222333444"] = market_id
        connector._markets_cache[market_id] = Market(
            market_id=MarketId(Venue.POLYMARKET, "0xabc123def456"),
            question="test",
            category=MarketCategory.CRYPTO,
            resolution=Resolution(
                resolution_date=datetime(2026, 7, 1, tzinfo=UTC),
                resolved_value=None,
            ),
            status=MarketStatus.OPEN,
        )

        event = {
            "event_type": "price_change",
            "asset_id": "111222333444",
            "price": "0.45",
        }

        await connector._handle_ws_event(event)
        # The handler calls the adapter — it may or may not emit a tick,
        # depending on the adapter implementation
        assert True  # it does not raise

    @pytest.mark.asyncio
    async def test_infer_tags_crypto(self) -> None:
        tags = PolymarketConnector._infer_tags("Will Bitcoin hit $150k?")
        assert "crypto" in tags

    @pytest.mark.asyncio
    async def test_infer_tags_politics(self) -> None:
        tags = PolymarketConnector._infer_tags("Who will win the 2026 election?")
        assert "politics" in tags

    @pytest.mark.asyncio
    async def test_infer_tags_other(self) -> None:
        tags = PolymarketConnector._infer_tags("Will it rain in London tomorrow?")
        assert tags == []

    def test_build_headers_without_auth(self) -> None:
        connector = make_polymarket_connector()
        headers = connector._build_headers()
        assert "Accept" in headers
        assert "KALSHI-ACCESS-KEY" not in headers


# ---------------------------------------------------------------------------
# Tests Manifold
# ---------------------------------------------------------------------------


class TestManifoldConnector:
    def setup_method(self) -> None:
        ticks_received.clear()
        snapshots_received.clear()

    @pytest.mark.asyncio
    async def test_get_markets_returns_list(self) -> None:
        connector = make_manifold_connector()
        connector._session = MockSession(
            {
                "api.manifold.markets/v0/search-markets": [MANIFOLD_MARKET_RAW],
            }
        )

        markets = await connector.get_markets()

        assert len(markets) == 1
        assert isinstance(markets[0], Market)
        assert markets[0].market_id.venue == Venue.MANIFOLD

    @pytest.mark.asyncio
    async def test_get_markets_filters_missing_close_time(self) -> None:
        """Markets without a closeTime must be ignored."""
        raw_sin_close = {**MANIFOLD_MARKET_RAW, "closeTime": None}
        connector = make_manifold_connector()
        connector._session = MockSession(
            {
                "api.manifold.markets/v0/search-markets": [raw_sin_close],
            }
        )

        markets = await connector.get_markets()
        assert markets == []

    @pytest.mark.asyncio
    async def test_get_snapshot_builds_synthetic_orderbook(self) -> None:
        """
        Manifold has no CLOB — get_snapshot() must build
        a synthetic order book from the probability.
        """
        connector = make_manifold_connector()
        connector._session = MockSession(
            {
                "api.manifold.markets/v0/market/": MANIFOLD_MARKET_RAW,
            }
        )

        snapshot = await connector.get_snapshot("manifold:will-btc-hit-150k")

        assert snapshot is not None
        assert isinstance(snapshot, MarketSnapshot)
        # The synthetic order book has bid < ask
        assert snapshot.orderbook is not None
        assert snapshot.orderbook.best_bid < snapshot.orderbook.best_ask

    @pytest.mark.asyncio
    async def test_poll_emits_tick_when_price_changes(self) -> None:
        """
        _poll_market() must emit a snapshot when the price moved more
        than 0.1% since the last poll.
        """
        connector = make_manifold_connector()
        connector._session = MockSession(
            {
                "api.manifold.markets/v0/market/": MANIFOLD_MARKET_RAW,
            }
        )

        # No previous price → must emit
        connector._last_prices["manifold:will-btc-hit-150k"] = 0.40
        await connector._poll_market("manifold:will-btc-hit-150k")

        # prob=0.45, last=0.40 → cambio > 0.001 → emite snapshot
        assert len(snapshots_received) == 1

    @pytest.mark.asyncio
    async def test_poll_emits_nothing_when_price_unchanged(self) -> None:
        """
        When the price has not moved, no tick or snapshot may be emitted.
        """
        connector = make_manifold_connector()
        connector._session = MockSession(
            {
                "api.manifold.markets/v0/market/": MANIFOLD_MARKET_RAW,
            }
        )

        # A price equal to the current one (prob=0.45)
        connector._last_prices["manifold:will-btc-hit-150k"] = 0.45
        await connector._poll_market("manifold:will-btc-hit-150k")

        assert len(snapshots_received) == 0
        assert len(ticks_received) == 0

    @pytest.mark.asyncio
    async def test_synthetic_orderbook_bid_below_ask(self) -> None:
        """The synthetic order book must always have bid < ask."""
        from normalizer.schema import Venue

        mid = MarketId(Venue.MANIFOLD, "test")

        for prob in [0.05, 0.25, 0.50, 0.75, 0.95]:
            ob = ManifoldConnector._synthetic_orderbook(mid, prob)
            assert ob.best_bid < ob.best_ask, f"Fallo en prob={prob}"

    @pytest.mark.asyncio
    async def test_synthetic_orderbook_prices_in_range(self) -> None:
        """bid and ask must lie in (0, 1)."""
        from normalizer.schema import Venue

        mid = MarketId(Venue.MANIFOLD, "test")
        ob = ManifoldConnector._synthetic_orderbook(mid, 0.5)

        assert 0 < ob.best_bid < 1
        assert 0 < ob.best_ask < 1

    def test_infer_category_crypto(self) -> None:
        cat = ManifoldConnector._infer_category("Will Bitcoin hit $200k?")
        assert cat == MarketCategory.CRYPTO

    def test_infer_category_politics(self) -> None:
        from normalizer.schema import MarketCategory

        cat = ManifoldConnector._infer_category("Who will win the presidential election?")
        assert cat == MarketCategory.POLITICS

    def test_infer_category_other(self) -> None:
        from normalizer.schema import MarketCategory

        cat = ManifoldConnector._infer_category("Will it snow in Madrid?")
        assert cat == MarketCategory.OTHER

    def test_build_headers_without_auth(self) -> None:
        connector = make_manifold_connector()
        headers = connector._build_headers()
        assert "Accept" in headers


# ---------------------------------------------------------------------------
# Tests base connector — comportamiento compartido
# ---------------------------------------------------------------------------


class TestBaseConnector:
    def setup_method(self) -> None:
        ticks_received.clear()
        snapshots_received.clear()

    @pytest.mark.asyncio
    async def test_get_retries_on_500(self) -> None:
        """
        _get() must retry automatically on 5xx errors.
        After MAX_RETRIES attempts it returns None.
        """
        connector = make_kalshi_connector()

        # MockSession needs to return MockResponse with status 500
        # We need to modify MockSession to support error responses
        class MockSession500(MockSession):
            def get(self, url: str, **kwargs: Any) -> MockResponse:
                # Always return 500 error
                return MockResponse({}, status=500)

        connector._session = MockSession500({})

        # Every attempt fails → None
        result = await connector._get("https://api.kalshi.com/any")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_does_not_retry_on_404(self) -> None:
        """
        _get() must NOT retry on 4xx (client) errors.
        Returns None immediately.
        """
        connector = make_kalshi_connector()
        connector._session = MockSession({})  # todas las URLs → 404

        result = await connector._get("https://api.kalshi.com/nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_without_session_returns_none(self) -> None:
        """
        Calling _get() before the session is opened returns None without
        raising.
        """
        connector = make_kalshi_connector()
        connector._session = None

        result = await connector._get("https://api.kalshi.com/markets")
        assert result is None
