"""
tests/integration/inspect_adapters.py
────────────────────────────────────────
Script de integración manual — hace peticiones HTTP reales a Kalshi
y Polymarket, imprime los campos más relevantes del JSON crudo y
el resultado tras la normalización.

NO es un test pytest — no tiene asserts. Es para verificar visualmente
que los adapters transforman correctamente los datos reales de la API.

Uso (desde la raíz del proyecto):
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
# Buscamos mercados con más tiempo hasta resolución para tener liquidez.
# Usamos el endpoint general sin filtro de serie para coger los más líquidos.
# ---------------------------------------------------------------------------

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


def inspect_kalshi_markets(n: int = 3) -> list[str]:
    """
    Fetcha mercados de Kalshi con liquidez real.
    Evita los mercados de BTC intraday que cierran en minutos y no tienen órdenes.
    Devuelve los tickers para el inspect de orderbooks.
    """
    header("KALSHI — MARKETS")

    # Buscamos mercados con volumen — sin filtro de serie para coger los más líquidos
    url = f"{KALSHI_BASE}/events?limit={n}&status=open&with_nested_markets=true"
    print(f"\n  {DIM}GET {url}{RESET}\n")

    data = fetch(url)
    if not data:
        # Fallback: endpoint directo de markets sin filtro de serie
        url = f"{KALSHI_BASE}/markets?limit={n}&status=open"
        print(f"  Fallback → {DIM}GET {url}{RESET}\n")
        data = fetch(url)
        if not data:
            return []
        markets_raw = data.get("markets", [])
    else:
        # Extraer markets de los events
        markets_raw = []
        for event in data.get("events", []):
            markets_raw.extend(event.get("markets", []))
        markets_raw = markets_raw[:n]

    tickers: list[str] = []

    for i, raw in enumerate(markets_raw, 1):
        section(f"Market {i}/{len(markets_raw)}")

        # --- RAW ---
        print(f"\n  {DIM}── RAW (campos relevantes) ──{RESET}")
        raw_field("ticker", raw.get("ticker"))
        raw_field("title", raw.get("title"))
        raw_field("status", raw.get("status"))
        raw_field("close_time", raw.get("close_time"))
        raw_field("yes_ask_dollars", raw.get("yes_ask_dollars"))
        raw_field("yes_bid_dollars", raw.get("yes_bid_dollars"))
        raw_field("volume_fp", raw.get("volume_fp"))
        raw_field("open_interest_fp", raw.get("open_interest_fp"))
        raw_field("series_ticker", raw.get("series_ticker") or "(no field)")
        raw_field("result", raw.get("result") or "(vacío)")

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
            norm_field("tau (años)", f"{market.resolution.tau:.4f}", highlight=True)
            norm_field("tau (días)", f"{market.resolution.tau * 365.25:.1f}")
            norm_field("is_resolved", market.resolution.is_resolved())
            norm_field("resolved_value", market.resolution.resolved_value)
            norm_field("is_tradeable", market.is_tradeable())
            tickers.append(raw["ticker"])
        except Exception as e:
            error(f"Error en normalización: {e}")

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
        raw_field("yes (top 3 bids)", raw_yes[:3] if raw_yes else "(vacío)")
        raw_field("no  (top 3 asks)", raw_no[:3] if raw_no else "(vacío)")
        raw_field("total yes levels", len(raw_yes))
        raw_field("total no  levels", len(raw_no))

        if not raw_yes and not raw_no:
            warn(
                "Libro vacío — mercado sin liquidez "
                "(típico en mercados intraday cercanos al cierre o nuevos)"
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
            error(f"Error en normalización: {e}")

        divider()


# ---------------------------------------------------------------------------
# POLYMARKET
# ---------------------------------------------------------------------------

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


def _get_tags(raw: dict) -> list[str]:
    """
    Extrae tags del market. La API a veces devuelve tags: []
    pero los tags reales están en events[0].tags o en el campo
    de categoría del evento.
    """
    # Intento 1: campo tags directo
    tags = raw.get("tags", [])
    if tags:
        return tags

    # Intento 2: dentro de events anidados
    events = raw.get("events", [])
    if events and isinstance(events, list):
        event_tags = events[0].get("tags", [])
        if event_tags:
            return event_tags

    # Intento 3: inferir de la question manualmente
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

        # Enriquecer con tags inferidos si vienen vacíos
        inferred_tags = _get_tags(raw)

        # --- RAW ---
        print(f"\n  {DIM}── RAW (campos relevantes) ──{RESET}")
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

        # Inyectar tags inferidos para que el adapter los use
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
            norm_field("tau (años)", f"{market.resolution.tau:.4f}", highlight=True)
            norm_field("tau (días)", f"{market.resolution.tau * 365.25:.1f}")
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
            error(f"Error en normalización: {e}")

        divider()

    return yes_token_ids


def inspect_polymarket_orderbooks(yes_token_ids: list[str]) -> None:
    """
    El endpoint /book del CLOB requiere auth.
    Usamos los tres endpoints públicos: /midpoint, /price?side=BUY, /price?side=SELL.

    Convención de Polymarket:
      /price?side=BUY  → precio mínimo al que alguien vende YES = BEST ASK
      /price?side=SELL → precio máximo al que alguien compra YES = BEST BID
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
            warn("No se pudo obtener datos del CLOB")
            divider()
            continue

        # Convención CLOB de Polymarket:
        #   BUY  price = precio al que puedes comprar YES ahora = BEST ASK
        #   SELL price = precio al que puedes vender YES ahora  = BEST BID
        best_ask = float(buy_data["price"])  # lo que pagas para comprar YES
        best_bid = float(sell_data["price"])  # lo que recibes al vender YES
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
                # Puede ocurrir en mercados muy líquidos donde bid≈ask≈1
                # o cuando la API devuelve temporalmente precios inconsistentes
                warn(f"bid ({best_bid}) >= ask ({best_ask}) — mercado en equilibrio extremo")
                warn("Ajustando spread mínimo para construir el OrderBook")
                # En este caso bid y ask son prácticamente iguales
                # Usamos el mid como referencia con spread mínimo
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
            print(f"  {DIM}  (profundidad y tamaños no disponibles sin auth en /book){RESET}")

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
    print(f"{BOLD}{GREEN}  Inspección completada{RESET}")
    print(f"{BOLD}{GREEN}{'═' * 60}{RESET}\n")


if __name__ == "__main__":
    main()
