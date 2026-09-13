# This adapter  transforms raw responses from the Kalshi API into
# domain objects defined in schema. py
# Kalshi API conventions:
# - Prices in whole cents: 45 means $0.45 = probability 0.45
# - Timestamps in ISO-8601 with UTC suffix
# - Tickers in “SERIES-DATE-THRESHOLD” format, e.g., “BTCZ-24DEC31-T50000”
# - Hierarchy: Series → Events → Markets
# - The category resides in Series, not directly in Market

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from normalizer.price_grid import parse_kalshi_price_ranges
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

_series_category_cache: dict[str, str] = {}


def build_series_cache(series_list: list[dict[str, Any]]) -> None:
    for s in series_list:
        ticker = s.get("ticker", "")
        category = s.get("category", "")
        if ticker:
            _series_category_cache[ticker] = category.lower()


# ---------------------------------------------------------------------------
# Map from Kalshi categories to the canonical domain categories
# ---------------------------------------------------------------------------
_KALSHI_CATEGORY_MAP: dict[str, MarketCategory] = {
    "crypto": MarketCategory.CRYPTO,
    "politics": MarketCategory.POLITICS,
    "economics": MarketCategory.ECONOMICS,
    "financials": MarketCategory.ECONOMICS,
    "sports": MarketCategory.SPORTS,
    "science": MarketCategory.SCIENCE,
    "weather": MarketCategory.OTHER,
    "pop culture": MarketCategory.OTHER,
    "health": MarketCategory.OTHER,
}


def _infer_category(raw: dict[str, Any]) -> MarketCategory:
    """
    Infer the canonical category of a raw Kalshi market.

    Two-step strategy:
      1. Look up series_ticker in the cache (the authoritative source)
      2. If absent from the cache, fall back to MarketCategory.OTHER

    Why not raise when the series is missing:
      The cache can be incomplete at startup, or Kalshi may add a new series
      between refreshes. Categorising as OTHER is preferable to failing the
      whole ingestion.

    Args:
        raw: a raw market object from the Kalshi API
    """
    series_ticker = raw.get("series_ticker", "")
    raw_category = _series_category_cache.get(series_ticker, "")
    return _KALSHI_CATEGORY_MAP.get(raw_category, MarketCategory.OTHER)


# ---------------------------------------------------------------------------
# Parsing de timestamps
# ---------------------------------------------------------------------------


def _parse_ts(ts: str) -> datetime:
    """
    Parse a Kalshi ISO-8601 timestamp into a UTC-aware datetime.

    Why two formats:
      Kalshi returns timestamps with and without microseconds depending on the
      endpoint:
        "2024-12-31T23:59:59Z"        → without microseconds
        "2024-12-31T23:59:59.123456Z" → with microseconds
      We try the most specific first to avoid losing precision.

    Args:
        ts: a timestamp string from the Kalshi API

    Raises:
        ValueError: if the format is not recognised
    """
    ts = ts.rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(ts, fmt)
            return dt.replace(tzinfo=UTC)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse Kalshi timestamp: {ts!r}")


# ---------------------------------------------------------------------------
# Price conversion
# ---------------------------------------------------------------------------


def _cents_to_prob(cents: int | float) -> Price:
    """
    Convert a Kalshi cent price (0-100) into a probability (0.0-1.0).
    """
    return Price(round(float(cents) / 100.0, 6))


def _dollars_to_prob(value: int | float | str) -> Price:
    """
    Convert a Kalshi price into a probability in [0, 1].

    The REST API returns dollar strings: "0.0100" = probability 0.01.
    The order book endpoint uses cents: 45 = probability 0.45.
    The distinguishing rule: a value > 1.0 is cents, ≤ 1.0 is already a
    probability.
    """
    v = float(value)
    if v > 1.0:
        return Price(round(v / 100.0, 6))
    return Price(round(v, 6))


# ---------------------------------------------------------------------------
# Market adapter
# ---------------------------------------------------------------------------


def kalshi_market_to_domain(raw: dict[str, Any]) -> Market:
    """
    Convert a market object from GET /markets/{ticker} into the canonical domain.

    Status logic:
      Kalshi uses "open", "closed", "settled" and "finalized".
      "settled" and "finalized" both map to RESOLVED because both mean the
      outcome is known. "closed" means trading has stopped but the outcome is
      still pending (an election in progress, say).

    resolved_value logic:
      Only populated when status is RESOLVED.
      "yes" → 1.0, "no" → 0.0, None → None (unresolved).
      A float rather than a bool, for consistency with the storage schema and
      with the Bernoulli vol computation: σ_B = sqrt(p(1-p)/τ).

    Args:
        raw: dict holding the GET /markets/{ticker} response

    Raises:
        ValueError: when close_time is missing (required to compute τ)
    """
    ticker: str = raw["ticker"]
    market_id = MarketId(venue=Venue.KALSHI, raw_id=ticker)

    # --- Mapeo de status ---
    # A dict lookup rather than if/elif, for extensibility: adding a new
    # Kalshi status is adding one line to the dict.
    raw_status = raw.get("status", "open").lower()
    status_map = {
        "open": MarketStatus.OPEN,
        "active": MarketStatus.OPEN,  # API real devuelve "active"
        "closed": MarketStatus.CLOSED,
        "settled": MarketStatus.RESOLVED,
        "finalized": MarketStatus.RESOLVED,
    }
    status = status_map.get(raw_status, MarketStatus.OPEN)

    close_time_str: str | None = raw.get("close_time") or raw.get("expiration_time")
    if close_time_str is None:
        raise ValueError(f"Market {ticker} has no close_time")
    resolution_date = _parse_ts(close_time_str)

    # --- Resolved value ---
    # "result" is only consulted when the market is resolved.
    # While open it may be None or absent — that is normal.
    result_str: str | None = raw.get("result")
    resolved_value: float | None = None
    if result_str == "yes":
        resolved_value = 1.0
    elif result_str == "no":
        resolved_value = 0.0

    resolution = Resolution(
        resolution_date=resolution_date,
        resolved_value=resolved_value,
    )

    return Market(
        market_id=market_id,
        question=raw.get("title") or raw.get("question") or ticker,
        category=_infer_category(raw),
        resolution=resolution,
        status=status,
        # Kalshi publishes the grid PER MARKET in `price_ranges`. Across the 200
        # markets sampled the main band steps by 0.001 (not 0.01), with 0.0001
        # in the tails — but some markets do use a single 0.01 band, so a fixed
        # grid cannot be assumed.
        price_ladder=parse_kalshi_price_ranges(raw.get("price_ranges")),
    )


# ---------------------------------------------------------------------------
# OrderBook adapter
# ---------------------------------------------------------------------------


def _extract_book_sides(
    raw: dict[str, Any],
) -> tuple[list[list[str | float | int]], list[list[str | float | int]], bool]:
    """
    Locate both sides of the book and report which units they use.

    Returns:
        (yes_levels, no_levels, in_dollars). `in_dollars` distinguishes the
        "fp" schema (decimal dollar prices, as strings) from the classic one
        (integer cents). If neither is recognised, empty lists are returned.
    """
    # The "fp" schema — the current one. It can arrive wrapped or bare.
    for container in (raw.get("orderbook_fp"), raw):
        if isinstance(container, dict) and (
            "yes_dollars" in container or "no_dollars" in container
        ):
            return (
                container.get("yes_dollars") or [],
                container.get("no_dollars") or [],
                True,
            )

    # The classic schema, in cents.
    book = raw.get("orderbook") if isinstance(raw.get("orderbook"), dict) else raw
    if isinstance(book, dict) and ("yes" in book or "no" in book):
        return book.get("yes") or [], book.get("no") or [], False

    return [], [], False


def kalshi_orderbook_to_domain(
    market_id: MarketId,
    raw: dict[str, Any],
    timestamp: datetime | None = None,
) -> OrderBook:
    """
    Convert the GET /markets/{ticker}/orderbook response into the domain.

    Kalshi serves TWO different schemas and both must be accepted:

      Classic schema (integer cents):
        {"orderbook": {"yes": [[45, 1200], ...], "no": [[53, 800], ...]}}

      "fp" schema — what the API returns today (dollars as STRINGS):
        {"orderbook_fp": {"yes_dollars": [["0.4500", "1200.00"], ...],
                          "no_dollars":  [["0.5300", "800.00"], ...]}}

    Why this matters: the adapter only looked at `orderbook`/`yes`/`no`. Against
    the current API, `raw.get("orderbook", raw)` fell through to the fallback
    and `book.get("yes")` returned [] — so EVERY live Kalshi order book came
    through EMPTY, without raising anything. Verified against a market with a
    real book (KXSERIEAGAME-26SEP05FIOTOR-TOR): 0 bids, 0 asks, in silence.

    Downstream consequence: no levels means no depth, so OBI came out
    identically 0 and with it μ̂ = 0 — that is, Cartea-Jaimungal degenerated to
    GLFT on Kalshi data too, not only on Manifold's.

    The NO side must be transformed:

      Kalshi quotes YES and NO separately. Buying NO at 0.53 is equivalent to
      selling YES at 0.47, so each NO bid becomes a YES ask at 1 - price.

    Bids are sorted descending and asks ascending.

    Why crossed levels are filtered rather than raising:
      In illiquid markets, or during book updates, a crossed level can appear
      momentarily. That is an API artefact, not a bug on our side. We filter
      the offending levels and continue — a partial book beats no book.


    Args:
        market_id: already-constructed MarketId (so it is not recomputed)
        raw: dict holding the order book endpoint response
        timestamp: si None, usa datetime.now(UTC)
    """
    ts = timestamp or datetime.now(tz=UTC)

    raw_yes, raw_no, in_dollars = _extract_book_sides(raw)

    def _price(value: str | float | int) -> Price:
        """Level price → probability, according to the detected schema."""
        return Price(float(value)) if in_dollars else _cents_to_prob(int(value))

    def _complement(value: str | float | int) -> Price:
        """NO side → equivalent YES price: p_yes = 1 - p_no."""
        return Price(1.0 - float(value)) if in_dollars else _cents_to_prob(100 - int(value))

    yes_bids: list[OrderBookLevel] = [
        OrderBookLevel(price=_price(p), size=Size(float(s))) for p, s in raw_yes if float(s) > 0
    ]

    yes_asks: list[OrderBookLevel] = [
        OrderBookLevel(price=_complement(p), size=Size(float(s))) for p, s in raw_no if float(s) > 0
    ]

    # Sort: bids descending, asks ascending
    yes_bids.sort(key=lambda lv: lv.price, reverse=True)
    yes_asks.sort(key=lambda lv: lv.price)

    # Filter out crossed levels, if any
    if yes_bids and yes_asks:
        best_bid = yes_bids[0].price
        best_ask = yes_asks[0].price
        if best_bid >= best_ask:
            yes_bids = [lv for lv in yes_bids if lv.price < best_ask]
            yes_asks = [lv for lv in yes_asks if lv.price > best_bid]

    return OrderBook(
        market_id=market_id,
        timestamp=ts,
        bids=tuple(yes_bids),
        asks=tuple(yes_asks),
    )


# ---------------------------------------------------------------------------
# Tick adapters
# ---------------------------------------------------------------------------


def kalshi_trade_to_tick(
    market_id: MarketId,
    raw: dict[str, Any],
) -> Tick:
    """
    Convert a Kalshi WebSocket trade message into a canonical Tick.

    Kalshi WS message format:
      {
        "type": "trade",
        "msg": {
          "market_ticker": "BTCZ-...",
          "yes_price": 45,     ← execution price, in cents
          "no_price":  55,     ← always 100 - yes_price
          "count": 10,         ← number of contracts
          "taker_side": "yes", ← which side was the aggressor
          "created_time": "2024-12-31T12:00:00Z"
        }
      }

    bid == ask == execution price.

    The taker is whoever sent the market order that crossed the spread.
    That information is essential for computing adverse selection.
    en features/microstructure.py.

    """
    msg = raw.get("msg", raw)

    yes_price_cents: int = msg["yes_price"]
    no_price_cents: int = msg["no_price"]

    # yes_bid = the price at which the YES executed
    # yes_ask = the implied NO price converted to YES
    # In practice always equal, but the structure is kept
    # for consistency with the schema
    yes_bid = _cents_to_prob(yes_price_cents)
    yes_ask = _cents_to_prob(100 - no_price_cents)

    taker_side_raw: str = msg.get("taker_side", "yes").lower()
    side = Side.YES if taker_side_raw == "yes" else Side.NO

    ts_str: str = msg.get("created_time") or msg.get("ts", "")
    ts = _parse_ts(ts_str) if ts_str else datetime.now(tz=UTC)

    return Tick(
        market_id=market_id,
        timestamp=ts,
        tick_type=TickType.TRADE,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        volume=Size(float(msg.get("count", 0))),
        side=side,
    )


def kalshi_quote_to_tick(
    market_id: MarketId,
    orderbook: OrderBook,
) -> Tick | None:
    """
    Derive a QUOTE tick from an OrderBook snapshot.

    This Tick represents "the mid price at this instant", with no trade having
    occurred.

    An empty side is valid in very illiquid markets, in which case None is
    returned. The caller decides what to do with it (typically: skip that
    snapshot).

    Args:
        market_id: the market's MarketId
        orderbook: an already-built and validated book snapshot

    Returns:
        A QUOTE Tick, or None when the book is incomplete
    """
    if orderbook.best_bid is None or orderbook.best_ask is None:
        return None

    return Tick(
        market_id=market_id,
        timestamp=orderbook.timestamp,
        tick_type=TickType.QUOTE,
        yes_bid=orderbook.best_bid,
        yes_ask=orderbook.best_ask,
        # volume=0 because there was no trade — this is a price update only
        volume=Size(0.0),
        # side=None because a QUOTE carries no aggressor
        side=None,
    )


# ---------------------------------------------------------------------------
# Convenience: full snapshot from REST responses
# ---------------------------------------------------------------------------


def kalshi_to_snapshot(
    raw_market: dict[str, Any],
    raw_orderbook: dict[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> MarketSnapshot:
    """
    Build a complete MarketSnapshot from REST API responses.

    Args:
        raw_market: respuesta de GET /markets/{ticker}
        raw_orderbook: the GET /markets/{ticker}/orderbook response (optional)
        timestamp: si None, usa datetime.now(UTC)
    """
    market = kalshi_market_to_domain(raw_market)
    ts = timestamp or datetime.now(tz=UTC)

    orderbook: OrderBook | None = None
    last_tick: Tick | None = None

    if raw_orderbook is not None:
        orderbook = kalshi_orderbook_to_domain(market.market_id, raw_orderbook, ts)
        last_tick = kalshi_quote_to_tick(market.market_id, orderbook)

    return MarketSnapshot(
        market=market,
        orderbook=orderbook,
        last_tick=last_tick,
        fetched_at=ts,
    )
