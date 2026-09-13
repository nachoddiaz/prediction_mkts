"""
tests/integration/test_venues_live.py
───────────────────────────────────────
Integration tests against the real Kalshi, Polymarket and Manifold APIs. They
make real HTTP requests and require an internet connection.

They do NOT run in CI. Only manually, to verify the connectors work against
the real APIs.

To run:
    uv run pytest tests/integration/test_venues_live.py -v -s

Why -s:
    Shows print() output live — useful for seeing the data each API returns
    while the test runs.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import aiohttp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from normalizer.kalshi_adapter import kalshi_market_to_domain
from normalizer.polymarket_adapter import (
    _parse_clob_token_ids,
    polymarket_market_to_domain,
)
from normalizer.schema import Tick, Venue

# ---------------------------------------------------------------------------
# Visual separator for the terminal
# ---------------------------------------------------------------------------


def header(title: str) -> None:
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def raw_line(key: str, value: object) -> None:
    v = str(value)
    if len(v) > 72:
        v = v[:69] + "..."
    print(f"  RAW  {key:<26} {v}")


def norm_line(key: str, value: object) -> None:
    print(f"  NORM {key:<26} {value}")


def divider() -> None:
    print(f"  {'·' * 56}")


# ---------------------------------------------------------------------------
# Kalshi
# ---------------------------------------------------------------------------

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


@pytest.mark.live
@pytest.mark.asyncio
async def test_kalshi_markets_live() -> None:
    """
    Fetch real Kalshi markets and show raw versus normalised.
    Verifies the adapter processes the real response correctly.
    """
    header("KALSHI — MARKETS (live)")

    async with aiohttp.ClientSession() as session:
        url = f"{KALSHI_BASE}/markets"
        params = {"status": "open", "limit": 3}

        async with session.get(url, params=params) as resp:
            assert resp.status == 200, f"HTTP {resp.status}"
            data = await resp.json()

    markets_raw = data.get("markets", [])
    assert len(markets_raw) > 0, "the API returned 0 markets"

    print(f"\n  Mercados recibidos: {len(markets_raw)}")

    for i, raw in enumerate(markets_raw, 1):
        print(f"\n  ▶ Market {i}/{len(markets_raw)}")
        print(f"\n  {'── RAW ──':─<50}")
        raw_line("ticker", raw.get("ticker"))
        raw_line("title", str(raw.get("title", ""))[:50])
        raw_line("status", raw.get("status"))
        raw_line("close_time", raw.get("close_time"))
        raw_line("yes_bid_dollars", raw.get("yes_bid_dollars"))
        raw_line("yes_ask_dollars", raw.get("yes_ask_dollars"))
        raw_line("volume_fp", raw.get("volume_fp"))
        raw_line("series_ticker", raw.get("series_ticker") or "(empty)")

        print(f"\n  {'── NORMALIZADO ──':─<50}")
        try:
            market = kalshi_market_to_domain(raw)
            norm_line("market_id", str(market.market_id))
            norm_line("question", market.question[:50])
            norm_line("status", market.status.value)
            norm_line("category", market.category.value)
            norm_line(
                "resolution_date", market.resolution.resolution_date.strftime("%Y-%m-%d %H:%M UTC")
            )
            norm_line("tau (years)", f"{market.resolution.tau:.4f}")
            norm_line("tau (days)", f"{market.resolution.tau * 365.25:.1f}")
            norm_line("is_resolved", market.resolution.is_resolved())
            norm_line("is_tradeable", market.is_tradeable())
            print("\n  ✓ Normalisation correct")
        except Exception as e:
            print(f"\n  ✗ Normalisation error: {e}")
            raise

        divider()


@pytest.mark.live
@pytest.mark.asyncio
async def test_kalshi_orderbook_live() -> None:
    """
    Fetch a real Kalshi market's order book.
    Uses the first market with liquidity it finds.
    """
    header("KALSHI — ORDERBOOK (live)")

    async with aiohttp.ClientSession() as session:
        # First find a ticker with volume
        url = f"{KALSHI_BASE}/markets"
        params = {"status": "open", "limit": 10}
        async with session.get(url, params=params) as resp:
            data = await resp.json()

        markets = data.get("markets", [])
        ticker = None
        for m in markets:
            if float(m.get("volume_fp") or 0) > 0:
                ticker = m["ticker"]
                break

        if not ticker:
            pytest.skip("No markets with volume found")

        print(f"\n  Fetching orderbook for: {ticker}")

        ob_url = f"{KALSHI_BASE}/markets/{ticker}/orderbook"
        async with session.get(ob_url) as resp:
            assert resp.status == 200
            ob_data = await resp.json()

    book = ob_data.get("orderbook", ob_data)
    raw_yes = book.get("yes", [])
    raw_no = book.get("no", [])

    print(f"\n  {'── RAW ──':─<50}")
    raw_line("yes (top 3 bids)", raw_yes[:3] if raw_yes else "(empty)")
    raw_line("no  (top 3 asks)", raw_no[:3] if raw_no else "(empty)")
    raw_line("total yes levels", len(raw_yes))
    raw_line("total no  levels", len(raw_no))

    print(f"\n  {'── NORMALIZADO ──':─<50}")
    from normalizer.kalshi_adapter import kalshi_orderbook_to_domain, kalshi_quote_to_tick
    from normalizer.schema import MarketId

    market_id = MarketId(Venue.KALSHI, ticker)
    ob = kalshi_orderbook_to_domain(market_id, ob_data, datetime.now(tz=UTC))
    tick = kalshi_quote_to_tick(market_id, ob)

    norm_line("best_bid", f"{ob.best_bid:.4f}" if ob.best_bid is not None else "None")
    norm_line("best_ask", f"{ob.best_ask:.4f}" if ob.best_ask is not None else "None")
    norm_line("mid", f"{ob.mid:.4f}" if ob.mid is not None else "None")
    norm_line("spread", f"{ob.spread:.4f}" if ob.spread is not None else "None")
    norm_line("bid_depth(5)", f"{ob.bid_depth(5):.0f} contratos")
    norm_line("ask_depth(5)", f"{ob.ask_depth(5):.0f} contratos")
    norm_line("bid levels", len(ob.bids))
    norm_line("ask levels", len(ob.asks))

    if tick:
        print(f"\n  {'── QUOTE TICK ──':─<50}")
        norm_line("tick_type", tick.tick_type.value)
        norm_line("yes_bid", f"{tick.yes_bid:.4f}")
        norm_line("yes_ask", f"{tick.yes_ask:.4f}")
        norm_line("mid", f"{tick.mid:.4f}")
        norm_line("spread", f"{tick.spread:.4f}")
        print("\n  ✓ Tick generado correctamente")
    else:
        print("\n  ⚠ Empty book — no tick generated")


# ---------------------------------------------------------------------------
# Polymarket
# ---------------------------------------------------------------------------

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


@pytest.mark.live
@pytest.mark.asyncio
async def test_polymarket_markets_live() -> None:
    """
    Fetch real Polymarket markets and show raw versus normalised.
    """
    header("POLYMARKET — MARKETS (live)")

    async with aiohttp.ClientSession() as session:
        url = f"{GAMMA_BASE}/markets"
        params = {"limit": 3, "active": "true", "order": "volume24hr", "ascending": "false"}
        async with session.get(url, params=params) as resp:
            assert resp.status == 200
            data = await resp.json()

    markets_raw = data if isinstance(data, list) else data.get("markets", [])
    assert len(markets_raw) > 0

    print(f"\n  Mercados recibidos: {len(markets_raw)}")

    for i, raw in enumerate(markets_raw, 1):
        print(f"\n  ▶ Market {i}/{len(markets_raw)}")

        # Infer tags where they arrive empty
        tags = raw.get("tags", [])
        if not tags:
            q = raw.get("question", "").lower()
            if any(w in q for w in ["bitcoin", "btc", "eth", "crypto"]):
                tags = [{"label": "Crypto"}]
        raw_enriched = {**raw, "tags": tags}

        print(f"\n  {'── RAW ──':─<50}")
        raw_line("conditionId", raw.get("conditionId"))
        raw_line("question", str(raw.get("question", ""))[:50])
        raw_line("active", raw.get("active"))
        raw_line("endDate", raw.get("endDate"))
        raw_line("bestBid", raw.get("bestBid"))
        raw_line("bestAsk", raw.get("bestAsk"))
        raw_line("tags (raw)", raw.get("tags", []))
        raw_line("tags (usado)", tags)
        clob = str(raw.get("clobTokenIds", ""))
        raw_line("clobTokenIds", clob[:60] + "...")

        print(f"\n  {'── NORMALIZADO ──':─<50}")
        try:
            market = polymarket_market_to_domain(raw_enriched)
            norm_line("market_id", str(market.market_id)[:50])
            norm_line("question", market.question[:50])
            norm_line("status", market.status.value)
            norm_line("category", market.category.value)
            norm_line(
                "resolution_date", market.resolution.resolution_date.strftime("%Y-%m-%d %H:%M UTC")
            )
            norm_line("tau (years)", f"{market.resolution.tau:.4f}")
            norm_line("tau (days)", f"{market.resolution.tau * 365.25:.1f}")
            norm_line("is_tradeable", market.is_tradeable())

            token_ids = _parse_clob_token_ids(raw)
            if token_ids:
                yes_id, no_id = token_ids
                norm_line("yes_token_id", yes_id[:40] + "...")
                norm_line("no_token_id", no_id[:40] + "...")

            print("\n  ✓ Normalisation correct")
        except Exception as e:
            print(f"\n  ✗ Normalisation error: {e}")
            raise

        divider()


@pytest.mark.live
@pytest.mark.asyncio
async def test_polymarket_orderbook_live() -> None:
    """
    Fetch a real Polymarket order book using the public endpoints.
    """
    header("POLYMARKET — ORDERBOOK (live)")

    async with aiohttp.ClientSession() as session:
        # Take the first market carrying a token ID
        url = f"{GAMMA_BASE}/markets"
        params = {"limit": 5, "active": "true", "order": "volume24hr", "ascending": "false"}
        async with session.get(url, params=params) as resp:
            data = await resp.json()

        markets_raw = data if isinstance(data, list) else []
        yes_token = None
        market_used = None

        for raw in markets_raw:
            token_ids = _parse_clob_token_ids(raw)
            if token_ids:
                yes_token = token_ids[0]
                market_used = raw
                break

        if not yes_token:
            pytest.skip("No token IDs found")

        print(f"\n  Token YES: {yes_token[:24]}...")
        print(f"  Mercado:   {market_used.get('question', '')[:50]}")

        # Fetch mid and prices in parallel
        async with session.get(f"{CLOB_BASE}/midpoint?token_id={yes_token}") as r:
            mid_data = await r.json() if r.status == 200 else None
        async with session.get(f"{CLOB_BASE}/price?token_id={yes_token}&side=BUY") as r:
            buy_data = await r.json() if r.status == 200 else None
        async with session.get(f"{CLOB_BASE}/price?token_id={yes_token}&side=SELL") as r:
            sell_data = await r.json() if r.status == 200 else None

    print(f"\n  {'── RAW ──':─<50}")
    raw_line("GET /midpoint → mid", mid_data.get("mid") if mid_data else "error")
    raw_line("GET /price?side=BUY → ask", buy_data.get("price") if buy_data else "error")
    raw_line("GET /price?side=SELL → bid", sell_data.get("price") if sell_data else "error")

    if mid_data and buy_data and sell_data:
        best_ask = float(buy_data["price"])
        best_bid = float(sell_data["price"])
        mid = float(mid_data["mid"])

        # Fix them if inverted or outside the [0, 1] bounds
        if best_bid >= best_ask:
            best_bid = max(0.0001, mid - 0.001)
            best_ask = min(0.9999, mid + 0.001)
            print("\n  ⚠ bid >= ask — ajustando spread mínimo")

        from normalizer.schema import MarketId, OrderBook, OrderBookLevel, Price, Size

        market_id = MarketId(Venue.POLYMARKET, yes_token)
        ob = OrderBook(
            market_id=market_id,
            timestamp=datetime.now(tz=UTC),
            bids=(OrderBookLevel(Price(round(best_bid, 6)), Size(0.0)),),
            asks=(OrderBookLevel(Price(round(best_ask, 6)), Size(0.0)),),
        )

        print(f"\n  {'── NORMALIZADO ──':─<50}")
        norm_line("best_bid (SELL)", f"{ob.best_bid:.4f}")
        norm_line("best_ask (BUY)", f"{ob.best_ask:.4f}")
        norm_line("mid (API)", f"{mid:.4f}")
        norm_line("spread", f"{ob.spread:.4f}")
        print("\n  ✓ OrderBook construido correctamente")


# ---------------------------------------------------------------------------
# Manifold
# ---------------------------------------------------------------------------

MANIFOLD_BASE = "https://manifold.markets/api/v0"


@pytest.mark.live
@pytest.mark.asyncio
async def test_manifold_markets_live() -> None:
    """
    Fetch real Manifold markets and show raw versus normalised.
    Manifold is the simplest — no auth, no CLOB, just a probability.
    """
    header("MANIFOLD — MARKETS (live)")

    async with aiohttp.ClientSession() as session:
        url = f"{MANIFOLD_BASE}/markets"
        # Manifold API changed - 'filter' and 'contractType' params no longer supported
        # Fetch more markets and filter manually
        params = {"limit": 10}
        async with session.get(url, params=params) as resp:
            assert resp.status == 200
            data = await resp.json()

    # Filter for open binary markets
    data = [m for m in data if not m.get("isResolved") and m.get("outcomeType") == "BINARY"][:3]
    assert isinstance(data, list) and len(data) > 0

    print(f"\n  Mercados recibidos: {len(data)}")

    for i, raw in enumerate(data, 1):
        print(f"\n  ▶ Market {i}/{len(data)}")

        print(f"\n  {'── RAW ──':─<50}")
        raw_line("id", raw.get("id"))
        raw_line("slug", raw.get("slug"))
        raw_line("question", str(raw.get("question", ""))[:50])
        raw_line("probability", raw.get("probability"))
        raw_line("closeTime", raw.get("closeTime"))
        raw_line("isResolved", raw.get("isResolved"))
        raw_line("totalLiquidity", raw.get("totalLiquidity"))

        print(f"\n  {'── NORMALIZADO ──':─<50}")
        try:
            from connectors.manifold import ManifoldConnector

            connector = ManifoldConnector.__new__(ManifoldConnector)
            market = connector._raw_to_market(raw)

            if market is None:
                print("  ⚠ Market skipped (no closeTime or slug)")
                divider()
                continue

            norm_line("market_id", str(market.market_id))
            norm_line("question", market.question[:50])
            norm_line("status", market.status.value)
            norm_line("category", market.category.value)
            norm_line(
                "resolution_date", market.resolution.resolution_date.strftime("%Y-%m-%d %H:%M UTC")
            )
            norm_line("tau (years)", f"{market.resolution.tau:.4f}")
            norm_line("tau (days)", f"{market.resolution.tau * 365.25:.1f}")
            norm_line("is_tradeable", market.is_tradeable())

            # Synthetic order book
            prob = float(raw.get("probability", 0.5))
            ob = ManifoldConnector._synthetic_orderbook(market.market_id, prob)
            norm_line("synthetic_bid", f"{ob.best_bid:.4f}")
            norm_line("synthetic_ask", f"{ob.best_ask:.4f}")
            norm_line("synthetic_mid", f"{ob.mid:.4f}")
            norm_line("synthetic_spread", f"{ob.spread:.4f}")

            print("\n  ✓ Normalisation correct")
        except Exception as e:
            print(f"\n  ✗ Error: {e}")
            raise

        divider()


@pytest.mark.live
@pytest.mark.asyncio
async def test_manifold_bets_live() -> None:
    """
    Fetch Manifold's recent trades.
    Equivalent to Kalshi's and Polymarket's TRADE ticks.
    """
    header("MANIFOLD — BETS/TRADES (live)")

    async with aiohttp.ClientSession() as session:
        # First obtain a market with activity
        url = f"{MANIFOLD_BASE}/markets"
        # Manifold API changed - filter manually for open BINARY markets
        params = {"limit": 10}
        async with session.get(url, params=params) as resp:
            all_markets = await resp.json()

        # Filter for open binary markets
        markets = [
            m for m in all_markets if not m.get("isResolved") and m.get("outcomeType") == "BINARY"
        ]

        if not markets:
            pytest.skip("No markets found")

        market_id = markets[0]["id"]
        market_slug = markets[0].get("slug", market_id)

        print(f"\n  Mercado: {markets[0].get('question', '')[:50]}")
        print(f"  Slug: {market_slug}")

        bets_url = f"{MANIFOLD_BASE}/bets"
        async with session.get(bets_url, params={"contractId": market_id, "limit": 5}) as resp:
            assert resp.status == 200
            bets = await resp.json()

    print(f"\n  Bets recibidos: {len(bets)}")

    for i, bet in enumerate(bets[:3], 1):
        print(f"\n  ▶ Bet {i}")
        raw_line("id", bet.get("id"))
        raw_line("amount", bet.get("amount"))
        raw_line("outcome", bet.get("outcome"))  # YES | NO
        raw_line("probAfter", bet.get("probAfter"))  # price after the trade
        raw_line("probBefore", bet.get("probBefore"))  # price before the trade
        raw_line("createdTime", bet.get("createdTime"))

        # Construir Tick equivalente
        prob_after = float(bet.get("probAfter", 0.5))
        prob_before = float(bet.get("probBefore", 0.5))
        outcome = bet.get("outcome", "YES")

        from normalizer.schema import (
            MarketId,
            Price,
            Side,
            Size,
            TickType,
            Venue,
        )

        mid_price = (prob_before + prob_after) / 2
        spread = 0.02

        tick = Tick(
            market_id=MarketId(Venue.MANIFOLD, market_slug),
            timestamp=datetime.now(tz=UTC),
            tick_type=TickType.TRADE,
            yes_bid=Price(round(mid_price - spread / 2, 4)),
            yes_ask=Price(round(mid_price + spread / 2, 4)),
            volume=Size(float(bet.get("amount", 0))),
            side=Side.YES if outcome == "YES" else Side.NO,
        )

        norm_line("tick_type", tick.tick_type.value)
        norm_line("side", tick.side.value if tick.side else "None")
        norm_line("yes_bid", f"{tick.yes_bid:.4f}")
        norm_line("yes_ask", f"{tick.yes_ask:.4f}")
        norm_line("mid", f"{tick.mid:.4f}")
        norm_line("volume", f"{tick.volume:.2f} Mana")
        print("  ✓ Tick construido correctamente")
        divider()
