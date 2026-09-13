"""
tests/integration/inspect_adapters.py
────────────────────────────────────────
Manual integration script — it makes real HTTP requests to Kalshi,
Polymarket and Manifold, and prints the result after normalisation.

NOT a pytest test — it has no assertions. It exists to visually verify that
the adapters transform real API data correctly.

Usage (from the repository root):
    uv run python tests/integration/inspect_adapters.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from normalizer.kalshi_adapter import (
    kalshi_market_to_domain,
    kalshi_orderbook_to_domain,
    kalshi_quote_to_tick,
)
from normalizer.polymarket_adapter import (
    _parse_clob_token_ids,
    polymarket_market_to_domain,
    polymarket_quote_to_tick,
)

# ---------------------------------------------------------------------------
# Helpers de display
# ---------------------------------------------------------------------------

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"


def header(title: str) -> None:
    width = 60
    print(f"\n{BOLD}{CYAN}{'─' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * width}{RESET}")


def section(title: str) -> None:
    print(f"\n{YELLOW}  ▶ {title}{RESET}")


def raw_field(key: str, value: Any) -> None:
    v = str(value)
    if len(v) > 80:
        v = v[:77] + "..."
    print(f"    {DIM}{key:<28}{RESET} {v}")


def norm_field(key: str, value: Any, highlight: bool = False) -> None:
    color = GREEN if highlight else RESET
    print(f"    {color}{key:<28}{RESET} {color}{value}{RESET}")


def divider() -> None:
    print(f"  {DIM}{'·' * 56}{RESET}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}  ⚠ {msg}{RESET}")


def error(msg: str) -> None:
    print(f"  {RED}  ✗ {msg}{RESET}")


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def fetch(url: str, timeout: int = 10) -> dict | list | None:
    try:
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "pms-inspect/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"  {RED}HTTP {e.code} — {url}{RESET}")
        return None
    except urllib.error.URLError as e:
        print(f"  {RED}URL error — {e.reason}{RESET}")
        return None
    except Exception as e:
        print(f"  {RED}Error — {e}{RESET}")
        return None


# ---------------------------------------------------------------------------
# KALSHI
# We look for markets further from resolution, since they carry liquidity.
# The general endpoint is used without a series filter, to get the most liquid.
# ---------------------------------------------------------------------------

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


def inspect_kalshi_markets(n: int = 3) -> list[str]:
    """
    Fetch Kalshi markets that have real liquidity.
    Avoids intraday BTC markets that close in minutes and carry no orders.
    Returns the tickers for the order book inspection.
    """
    header("KALSHI — MARKETS")

    # Look for markets with volume — no series filter, to get the most liquid
    url = f"{KALSHI_BASE}/events?limit={n}&status=open&with_nested_markets=true"
    print(f"\n  {DIM}GET {url}{RESET}\n")

    data = fetch(url)
    if not data:
        # Fallback: the direct markets endpoint, with no series filter
        url = f"{KALSHI_BASE}/markets?limit={n}&status=open"
        print(f"  Fallback → {DIM}GET {url}{RESET}\n")
        data = fetch(url)
        if not data:
            return []
        markets_raw = data.get("markets", [])
    else:
        # Extract the markets out of the events
        markets_raw = []
        for event in data.get("events", []):
            markets_raw.extend(event.get("markets", []))
        markets_raw = markets_raw[:n]

    tickers: list[str] = []

    for i, raw in enumerate(markets_raw, 1):
        section(f"Market {i}/{len(markets_raw)}")

        # --- RAW ---
        print(f"\n  {DIM}── RAW (relevant fields) ──{RESET}")
        raw_field("ticker", raw.get("ticker"))
        raw_field("title", raw.get("title"))
        raw_field("status", raw.get("status"))
        raw_field("close_time", raw.get("close_time"))
        raw_field("yes_ask_dollars", raw.get("yes_ask_dollars"))
        raw_field("yes_bid_dollars", raw.get("yes_bid_dollars"))
        raw_field("volume_fp", raw.get("volume_fp"))
        raw_field("open_interest_fp", raw.get("open_interest_fp"))
        raw_field("series_ticker", raw.get("series_ticker") or "(no field)")
        raw_field("result", raw.get("result") or "(empty)")

        # --- NORMALIZADO ---
        try:
            market = kalshi_market_to_domain(raw)
            print(f"\n  {DIM}── NORMALIZADO ──{RESET}")
            norm_field("market_id", str(market.market_id), highlight=True)
            norm_field("question", market.question[:55])
            norm_field("status", market.status.value, highlight=True)
            norm_field("category", market.category.value, highlight=True)
            norm_field(
                "resolution_date", market.resolution.resolution_date.strftime("%Y-%m-%d %H:%M UTC")
            )
            norm_field("tau (years)", f"{market.resolution.tau:.4f}", highlight=True)
            norm_field("tau (days)", f"{market.resolution.tau * 365.25:.1f}")
            norm_field("is_resolved", market.resolution.is_resolved())
            norm_field("resolved_value", market.resolution.resolved_value)
            norm_field("is_tradeable", market.is_tradeable())
            tickers.append(raw["ticker"])
        except Exception as e:
            error(f"Normalisation error: {e}")

        divider()

    return tickers


def inspect_kalshi_orderbooks(tickers: list[str]) -> None:
    header("KALSHI — ORDERBOOKS")

    for ticker in tickers[:2]:
        url = f"{KALSHI_BASE}/markets/{ticker}/orderbook"
        print(f"\n  {DIM}GET {url}{RESET}")

        data = fetch(url)
        if not data:
            continue

        section(f"Orderbook: {ticker}")

        book = data.get("orderbook", data)
        raw_yes = book.get("yes", [])
        raw_no = book.get("no", [])

        # --- RAW ---
        print(f"\n  {DIM}── RAW ──{RESET}")
        raw_field("yes (top 3 bids)", raw_yes[:3] if raw_yes else "(empty)")
        raw_field("no  (top 3 asks)", raw_no[:3] if raw_no else "(empty)")
        raw_field("total yes levels", len(raw_yes))
        raw_field("total no  levels", len(raw_no))

        if not raw_yes and not raw_no:
            warn(
                "Empty book — market with no liquidity "
                "(typical of intraday markets near expiry, or brand-new ones)"
            )
            divider()
            continue

        # --- NORMALIZADO ---
        from normalizer.schema import MarketId, Venue

        market_id = MarketId(Venue.KALSHI, ticker)
        try:
            ob = kalshi_orderbook_to_domain(market_id, data)
            tick = kalshi_quote_to_tick(market_id, ob)

            print(f"\n  {DIM}── NORMALIZADO ──{RESET}")
            norm_field(
                "best_bid",
                f"{ob.best_bid:.4f}" if ob.best_bid is not None else "None",
                highlight=ob.best_bid is not None,
            )
            norm_field(
                "best_ask",
                f"{ob.best_ask:.4f}" if ob.best_ask is not None else "None",
                highlight=ob.best_ask is not None,
            )
            norm_field(
                "mid",
                f"{ob.mid:.4f}" if ob.mid is not None else "None",
                highlight=ob.mid is not None,
            )
            norm_field("spread", f"{ob.spread:.4f}" if ob.spread is not None else "None")
            norm_field("bid_depth(5)", f"{ob.bid_depth(5):.0f} contratos")
            norm_field("ask_depth(5)", f"{ob.ask_depth(5):.0f} contratos")
            norm_field("bid levels", len(ob.bids))
            norm_field("ask levels", len(ob.asks))

            if tick:
                print(f"\n  {DIM}── QUOTE TICK derivado ──{RESET}")
                norm_field("tick_type", tick.tick_type.value)
                norm_field("yes_bid", f"{tick.yes_bid:.4f}")
                norm_field("yes_ask", f"{tick.yes_ask:.4f}")
                norm_field("mid", f"{tick.mid:.4f}", highlight=True)
                norm_field("spread", f"{tick.spread:.4f}")
            else:
                warn("Libro incompleto — no se generó QUOTE tick")

        except Exception as e:
            error(f"Normalisation error: {e}")

        divider()


# ---------------------------------------------------------------------------
# POLYMARKET
# ---------------------------------------------------------------------------

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


def _get_tags(raw: dict) -> list[str]:
    """
    Extract tags from the market. The API sometimes returns tags: []
    while the real tags live in events[0].tags or in the event's
    category field.
    """
    # Attempt 1: the tags field directly
    tags = raw.get("tags", [])
    if tags:
        return tags

    # Intento 2: dentro de events anidados
    events = raw.get("events", [])
    if events and isinstance(events, list):
        event_tags = events[0].get("tags", [])
        if event_tags:
            return event_tags

    # Attempt 3: infer it from the question by hand
    question = raw.get("question", "").lower()
    if any(w in question for w in ["bitcoin", "btc", "eth", "crypto", "sol"]):
        return ["crypto"]
    if any(w in question for w in ["election", "president", "senate", "congress"]):
        return ["politics"]
    if any(w in question for w in ["fed", "rate", "inflation", "gdp"]):
        return ["economics"]

    return []


def inspect_polymarket_markets(n: int = 3) -> list[str]:
    header("POLYMARKET — MARKETS  (Gamma API)")

    url = f"{GAMMA_BASE}/markets?limit={n}&active=true&order=volume24hr&ascending=false"
    print(f"\n  {DIM}GET {url}{RESET}\n")

    data = fetch(url)
    if not data:
        return []

    markets_raw: list[dict] = data if isinstance(data, list) else data.get("markets", [])
    yes_token_ids: list[str] = []

    for i, raw in enumerate(markets_raw, 1):
        section(f"Market {i}/{len(markets_raw)}")

        # Enrich with inferred tags where they arrive empty
        inferred_tags = _get_tags(raw)

        # --- RAW ---
        print(f"\n  {DIM}── RAW (relevant fields) ──{RESET}")
        raw_field("conditionId", raw.get("conditionId"))
        raw_field("question", raw.get("question", "")[:55])
        raw_field("active", raw.get("active"))
        raw_field("closed", raw.get("closed"))
        raw_field("resolved", raw.get("resolved"))
        raw_field("endDate", raw.get("endDate"))
        raw_field("bestBid", raw.get("bestBid"))
        raw_field("bestAsk", raw.get("bestAsk"))
        raw_field("tags (raw)", raw.get("tags", []))
        raw_field("tags (inferred)", inferred_tags)
        clob_raw = str(raw.get("clobTokenIds", ""))
        raw_field("clobTokenIds", clob_raw[:60] + "...")

        # Inject the inferred tags so the adapter uses them
        raw_enriched = {**raw, "tags": inferred_tags}

        # --- NORMALIZADO ---
        try:
            market = polymarket_market_to_domain(raw_enriched)
            print(f"\n  {DIM}── NORMALIZADO ──{RESET}")
            norm_field("market_id", str(market.market_id)[:55], highlight=True)
            norm_field("question", market.question[:55])
            norm_field("status", market.status.value, highlight=True)
            norm_field("category", market.category.value, highlight=True)
            norm_field(
                "resolution_date", market.resolution.resolution_date.strftime("%Y-%m-%d %H:%M UTC")
            )
            norm_field("tau (years)", f"{market.resolution.tau:.4f}", highlight=True)
            norm_field("tau (days)", f"{market.resolution.tau * 365.25:.1f}")
            norm_field("is_resolved", market.resolution.is_resolved())
            norm_field("resolved_value", market.resolution.resolved_value)
            norm_field("is_tradeable", market.is_tradeable())

            token_ids = _parse_clob_token_ids(raw)
            if token_ids:
                yes_id, no_id = token_ids
                norm_field("yes_token_id", yes_id[:45] + "...")
                norm_field("no_token_id", no_id[:45] + "...")
                yes_token_ids.append(yes_id)
            else:
                warn("No se pudieron extraer token IDs")

        except Exception as e:
            error(f"Normalisation error: {e}")

        divider()

    return yes_token_ids


def inspect_polymarket_orderbooks(yes_token_ids: list[str]) -> None:
    """
    The CLOB /book endpoint requires auth.
    We use the three public endpoints: /midpoint, /price?side=BUY, /price?side=SELL.

    Polymarket convention:
      /price?side=BUY  → lowest price anyone sells YES at = BEST ASK
      /price?side=SELL → highest price anyone buys YES at  = BEST BID
    """
    header("POLYMARKET — ORDERBOOKS  (CLOB API — endpoints públicos)")

    for token_id in yes_token_ids[:2]:
        short_id = token_id[:24] + "..."
        section(f"Token YES: {short_id}")

        url_mid = f"{CLOB_BASE}/midpoint?token_id={token_id}"
        url_buy = f"{CLOB_BASE}/price?token_id={token_id}&side=BUY"
        url_sell = f"{CLOB_BASE}/price?token_id={token_id}&side=SELL"

        print(f"\n  {DIM}GET {CLOB_BASE}/midpoint?token_id=...{RESET}")
        print(f"  {DIM}GET {CLOB_BASE}/price?token_id=...&side=BUY{RESET}")
        print(f"  {DIM}GET {CLOB_BASE}/price?token_id=...&side=SELL{RESET}")

        mid_data = fetch(url_mid)
        buy_data = fetch(url_buy)
        sell_data = fetch(url_sell)

        if not (mid_data and buy_data and sell_data):
            warn("Could not fetch CLOB data")
            divider()
            continue

        # Polymarket CLOB convention:
        #   BUY  price = the price you can buy YES at now  = BEST ASK
        #   SELL price = the price you can sell YES at now = BEST BID
        best_ask = float(buy_data["price"])  # what you pay to buy YES
        best_bid = float(sell_data["price"])  # what you receive for selling YES
        mid = float(mid_data["mid"])

        # --- RAW ---
        print(f"\n  {DIM}── RAW ──{RESET}")
        raw_field("GET /midpoint → mid", mid_data.get("mid"))
        raw_field("GET /price?side=BUY  → ask", buy_data.get("price"))
        raw_field("GET /price?side=SELL → bid", sell_data.get("price"))

        # --- NORMALIZADO ---
        from normalizer.schema import MarketId, OrderBook, OrderBookLevel, Price, Size, Venue

        market_id = MarketId(Venue.POLYMARKET, token_id)

        try:
            if best_bid >= best_ask:
                # Can happen in very liquid markets where bid≈ask≈1, or when
                # the API momentarily returns inconsistent prices
                warn(f"bid ({best_bid}) >= ask ({best_ask}) — mercado en equilibrio extremo")
                warn("Widening to a minimal spread so the OrderBook can be built")
                # Here bid and ask are practically equal, so we use the mid as
                # the reference with a minimal spread
                best_bid = mid - 0.001
                best_ask = mid + 0.001

            ob = OrderBook(
                market_id=market_id,
                timestamp=datetime.now(tz=UTC),
                bids=(OrderBookLevel(Price(round(best_bid, 6)), Size(0.0)),),
                asks=(OrderBookLevel(Price(round(best_ask, 6)), Size(0.0)),),
            )
            tick = polymarket_quote_to_tick(market_id, ob)

            print(f"\n  {DIM}── NORMALIZADO ──{RESET}")
            norm_field("best_bid (SELL price)", f"{ob.best_bid:.4f}", highlight=True)
            norm_field("best_ask (BUY  price)", f"{ob.best_ask:.4f}", highlight=True)
            norm_field("mid (API midpoint)", f"{mid:.4f}", highlight=True)
            norm_field("spread", f"{ob.spread:.4f}")
            norm_field("bid levels", len(ob.bids))
            norm_field("ask levels", len(ob.asks))
            print(f"  {DIM}  (depth and sizes unavailable without auth on /book){RESET}")

            if tick:
                print(f"\n  {DIM}── QUOTE TICK derivado ──{RESET}")
                norm_field("tick_type", tick.tick_type.value)
                norm_field("yes_bid", f"{tick.yes_bid:.4f}")
                norm_field("yes_ask", f"{tick.yes_ask:.4f}")
                norm_field("mid", f"{tick.mid:.4f}", highlight=True)
                norm_field("spread", f"{tick.spread:.4f}")

        except Exception as e:
            error(f"Error construyendo OrderBook: {e}")

        divider()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  Prediction Market System — Adapter Inspection{RESET}")
    print(f"{BOLD}  {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}")

    kalshi_tickers = inspect_kalshi_markets(n=3)
    inspect_kalshi_orderbooks(kalshi_tickers)

    yes_token_ids = inspect_polymarket_markets(n=3)
    inspect_polymarket_orderbooks(yes_token_ids)

    print(f"\n{BOLD}{GREEN}{'═' * 60}{RESET}")
    print(f"{BOLD}{GREEN}  Inspection complete{RESET}")
    print(f"{BOLD}{GREEN}{'═' * 60}{RESET}\n")


if __name__ == "__main__":
    main()
