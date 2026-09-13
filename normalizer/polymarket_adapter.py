# This adapter  transforms raw responses from the Polymarket API
# into domain objects defined in schema.py.
# In Polymarket we have 2 different APIs:
#   Gamma API:  Metadata (title, category, settlement date, status), estimated prices
#   CLOB API: Real-time Order Book, Real trades, WS for streaming, requires EIP-721

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from normalizer.price_grid import parse_polymarket_tick_size
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

_POLY_TAG_MAP: dict[str, MarketCategory] = {
    "crypto": MarketCategory.CRYPTO,
    "bitcoin": MarketCategory.CRYPTO,
    "ethereum": MarketCategory.CRYPTO,
    "solana": MarketCategory.CRYPTO,
    "defi": MarketCategory.CRYPTO,
    "politics": MarketCategory.POLITICS,
    "elections": MarketCategory.POLITICS,
    "us-politics": MarketCategory.POLITICS,
    "president": MarketCategory.POLITICS,
    "sports": MarketCategory.SPORTS,
    "nba": MarketCategory.SPORTS,
    "nfl": MarketCategory.SPORTS,
    "soccer": MarketCategory.SPORTS,
    "economics": MarketCategory.ECONOMICS,
    "fed": MarketCategory.ECONOMICS,
    "inflation": MarketCategory.ECONOMICS,
    "science": MarketCategory.SCIENCE,
    "ai": MarketCategory.SCIENCE,
}


def _parse_clob_token_ids(raw: dict[str, Any]) -> tuple[str, str] | None:
    """
    Extract the YES and NO token IDs from the clobTokenIds field.

    La Gamma API devuelve clobTokenIds como string JSON, no como array:
      "clobTokenIds": "[\"13915...\", \"13290...\"]"

    json.loads() is required to parse it.
    Returns (yes_token_id, no_token_id), or None when absent.
    """
    raw_ids = raw.get("clobTokenIds")
    if not raw_ids:
        return None
    try:
        ids = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
        if len(ids) >= 2:
            return str(ids[0]), str(ids[1])
    except (json.JSONDecodeError, IndexError):
        pass
    return None


def _infer_category_poly(tags: list[str] | list[dict[str, Any]]) -> MarketCategory:
    for tag in tags:
        # Handle both string tags and dict tags with 'label' key
        tag_str = tag.lower() if isinstance(tag, str) else tag.get("label", "").lower()
        if tag_str in _POLY_TAG_MAP:
            return _POLY_TAG_MAP[tag_str]
    return MarketCategory.OTHER


# ---------------------------------------------------------------------------
# Timestamp parsing
#
# Why two separate functions (_parse_ts_gamma and _parse_ts_clob):
#   The Gamma API and the CLOB API use different timestamp formats.
# ---------------------------------------------------------------------------


def _parse_ts_gamma(ts: int | str) -> datetime:
    if isinstance(ts, int | float):
        # Unix milliseconds → seconds → UTC datetime
        return datetime.fromtimestamp(ts / 1000.0, tz=UTC)

    # Fallback: ISO string (some Gamma endpoints use this)
    ts_str = str(ts).rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(ts_str, fmt)
            return dt.replace(tzinfo=UTC)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse Gamma timestamp: {ts!r}")


def _parse_ts_clob(ts: str) -> datetime:
    ts_str = ts.rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(ts_str, fmt)
            return dt.replace(tzinfo=UTC)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse CLOB timestamp: {ts!r}")


# ---------------------------------------------------------------------------
# Market adapter — from the Gamma API
# ---------------------------------------------------------------------------


def polymarket_market_to_domain(raw: dict[str, Any]) -> Market:
    """
    Convert a Gamma API market object into the canonical domain.

    Status logic — why this priority order:
      "resolved" is checked first because a market can carry resolved=True and
      closed=True simultaneously. Checking closed first would lose the
      resolution information.

    resolved_value logic:
      After resolution the winning token has price ≈ 1.0 and the loser
      price ≈ 0.0. We use >= 0.99 rather than == 1.0 because Polymarket may
      return 0.9999... for floating-point reasons inside the ERC-1155
      contract.

    Args:
        raw: dict holding the Gamma API GET /markets response

    Raises:
        ValueError: when end_date is missing (required to compute τ)
    """
    condition_id: str = raw.get("conditionId") or raw.get("condition_id") or raw["id"]
    market_id = MarketId(venue=Venue.POLYMARKET, raw_id=condition_id)

    # --- Status ---
    # The check order matters: resolved > closed > active
    if raw.get("resolved", False):
        status = MarketStatus.RESOLVED
    elif raw.get("closed", False):
        status = MarketStatus.CLOSED
    elif raw.get("active", True):
        status = MarketStatus.OPEN
    else:
        status = MarketStatus.CLOSED

    end_date_raw = raw.get("endDate") or raw.get("end_date_iso") or raw.get("end_date")
    if end_date_raw is None:
        raise ValueError(f"Market {condition_id} has no end_date")
    resolution_date = _parse_ts_gamma(end_date_raw)

    # A YES token priced >= 0.99 indicates YES won.
    resolved_value: float | None = None
    if status == MarketStatus.RESOLVED:
        tokens: list[dict[str, Any]] = raw.get("tokens", [])
        for token in tokens:
            if token.get("outcome", "").lower() == "yes":
                price = float(token.get("price", 0.0))
                # >= 0.99 rather than == 1.0, for floating-point precision
                resolved_value = 1.0 if price >= 0.99 else 0.0
                break

    return Market(
        market_id=market_id,
        question=raw.get("question") or condition_id,
        category=_infer_category_poly(raw.get("tags", [])),
        resolution=Resolution(
            resolution_date=resolution_date,
            resolved_value=resolved_value,
        ),
        status=status,
        # Polymarket declares `orderPriceMinTickSize` in Gamma and
        # `minimum_tick_size` in the CLOB. Both are 0.001 today; we read
        # whichever the payload carries rather than hard-coding it.
        price_ladder=parse_polymarket_tick_size(
            raw.get("orderPriceMinTickSize") or raw.get("minimum_tick_size")
        ),
    )


# ---------------------------------------------------------------------------
# OrderBook adapter — from the CLOB API
# ---------------------------------------------------------------------------


def polymarket_merged_book_to_domain(
    market_id: MarketId,
    yes_book: dict[str, Any],
    no_book: dict[str, Any] | None,
    timestamp: datetime | None = None,
) -> OrderBook:
    """
    Merge the YES-token and NO-token books into a single YES book.

    Why merging is necessary:
      On Polymarket, YES and NO are two distinct ERC-1155 tokens with SEPARATE
      BOOKS. Looking only at the YES book, markets appear with 92 bids and 0
      asks — not because nobody wants to sell YES, but because that liquidity
      is expressed as NO bids. Without merging, those markets were discarded
      entirely: 4 out of every 6 in the volume-ranked sample.

    The equivalence is the same one the Kalshi adapter already applies:
        buying NO at p_no   ≡  selling YES at (1 - p_no)
        selling NO at p_no  ≡  buying YES at (1 - p_no)

    Args:
        yes_book: /book payload for the YES token
        no_book:  /book payload for the NO token. None = use the YES book only.
    """
    ts = timestamp or datetime.now(tz=UTC)

    def levels(raw: list[dict[str, Any]], complement: bool) -> list[dict[str, Any]]:
        out = []
        for lv in raw:
            try:
                price = float(lv["price"])
                size = float(lv["size"])
            except (KeyError, TypeError, ValueError):
                continue
            if size <= 0:
                continue
            out.append({"price": 1.0 - price if complement else price, "size": size})
        return out

    merged = {
        "bids": levels(yes_book.get("bids", []), complement=False)
        + levels((no_book or {}).get("asks", []), complement=True),
        "asks": levels(yes_book.get("asks", []), complement=False)
        + levels((no_book or {}).get("bids", []), complement=True),
    }
    return polymarket_orderbook_to_domain(market_id, merged, ts)


def polymarket_orderbook_to_domain(
    market_id: MarketId,
    raw: dict[str, Any],
    timestamp: datetime | None = None,
) -> OrderBook:
    """
    Convert the CLOB GET /book response into the canonical domain.

    CLOB format (prices already in [0,1], not cents):
      {
        "bids": [{"price": "0.45", "size": "100"}, ...],
        "asks": [{"price": "0.47", "size": "150"}, ...]
      }

    The CLOB API returns prices as strings ("0.45"), not as numbers, so each
    floats.

    Args:
        market_id: MarketId ya construido
        raw: CLOB GET /book response
        timestamp: si None, usa datetime.now(UTC)
    """
    ts = timestamp or datetime.now(tz=UTC)

    raw_bids: list[dict[str, Any]] = raw.get("bids", [])
    raw_asks: list[dict[str, Any]] = raw.get("asks", [])

    # Build the bids — size == 0 levels (empty ones) are filtered out
    bids: list[OrderBookLevel] = [
        OrderBookLevel(
            price=Price(round(float(lv["price"]), 6)),
            size=Size(float(lv["size"])),
        )
        for lv in raw_bids
        if float(lv.get("size", 0)) > 0
    ]

    # Build asks — same treatment as bids
    asks: list[OrderBookLevel] = [
        OrderBookLevel(
            price=Price(round(float(lv["price"]), 6)),
            size=Size(float(lv["size"])),
        )
        for lv in raw_asks
        if float(lv.get("size", 0)) > 0
    ]

    # Ordenar: bids descendente, asks ascendente
    bids.sort(key=lambda lv: lv.price, reverse=True)
    asks.sort(key=lambda lv: lv.price)

    # Filter crossed levels — same logic as Kalshi
    if bids and asks:
        best_bid = bids[0].price
        best_ask = asks[0].price
        if best_bid >= best_ask:
            bids = [lv for lv in bids if lv.price < best_ask]
            asks = [lv for lv in asks if lv.price > best_bid]

    return OrderBook(
        market_id=market_id,
        timestamp=ts,
        bids=tuple(bids),
        asks=tuple(asks),
    )


# ---------------------------------------------------------------------------
# Tick adapters
# ---------------------------------------------------------------------------


def polymarket_trade_to_tick(
    market_id: MarketId,
    raw: dict[str, Any],
) -> Tick:
    """
    Convert a CLOB trade event into a canonical Tick.

    Trade format in both the CLOB REST and WebSocket feeds:
      {
        "price":     "0.46",          ← execution price in [0,1]
        "size":      "50",            ← contracts executed
        "side":      "BUY",           ← BUY = bought YES, SELL = sold YES
        "timestamp": "2024-12-31T12:00:00Z"
      }

    Hence "BUY" → Side.YES and "SELL" → Side.NO.

    Args:
        market_id: an already-built MarketId
        raw: CLOB trade event
    """
    price = Price(round(float(raw["price"]), 6))

    raw_side = raw.get("side", "BUY").upper()
    side = Side.YES if raw_side == "BUY" else Side.NO

    ts_raw = raw.get("timestamp") or raw.get("time", "")
    ts = _parse_ts_clob(ts_raw) if ts_raw else datetime.now(tz=UTC)

    return Tick(
        market_id=market_id,
        timestamp=ts,
        tick_type=TickType.TRADE,
        # bid == ask == execution price on trades
        yes_bid=price,
        yes_ask=price,
        volume=Size(float(raw.get("size", 0))),
        side=side,
    )


def polymarket_quote_to_tick(
    market_id: MarketId,
    orderbook: OrderBook,
) -> Tick | None:
    """
    Derive a QUOTE tick from a Polymarket OrderBook snapshot.

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
        volume=Size(0.0),  # no trade, no volume
        side=None,  # no aggressor, no side
    )


def polymarket_price_update_to_tick(
    market_id: MarketId,
    raw: dict[str, Any],
) -> Tick | None:
    """
    Convert a CLOB WebSocket price_change message into a Tick.

    Polymarket emits these when the mid price moves without a completed
    trade.

    WS message format:
      {
        "asset_id":  "...",       ← token_id of the YES contract
        "price":     "0.46",      ← the new mid price
        "side":      "BUY",
        "size":      "0",         ← 0 because there is no fill
        "timestamp": 1704067200000  ← Unix ms
      }

    Args:
        market_id: MarketId ya construido
        raw: raw CLOB WebSocket message

    Returns:
        A QUOTE Tick, or None when the message carries no price
    """
    price_str = raw.get("price")
    if price_str is None:
        # Message without a price — heartbeat or status message
        return None

    price = Price(round(float(price_str), 6))

    # The timestamp can arrive as Unix ms (int) or as an ISO string
    ts_raw = raw.get("timestamp")
    if isinstance(ts_raw, int | float):
        ts = datetime.fromtimestamp(ts_raw / 1000.0, tz=UTC)
    else:
        ts = datetime.now(tz=UTC)

    # For price updates, bid ≈ ask ≈ price (an approximation).
    # This message carries no book depth, so the spread cannot be recovered —
    # only the mid price. The feature store treats it as a QUOTE.
    return Tick(
        market_id=market_id,
        timestamp=ts,
        tick_type=TickType.QUOTE,
        yes_bid=price,
        yes_ask=price,
        volume=Size(0.0),
        side=None,
    )


# ---------------------------------------------------------------------------
# Convenience: full snapshot from REST responses
# ---------------------------------------------------------------------------


def polymarket_to_snapshot(
    raw_market: dict[str, Any],
    raw_orderbook: dict[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> MarketSnapshot:
    """
    Build a complete MarketSnapshot from the REST API responses.

    raw_market comes from Gamma and raw_orderbook from the CLOB.

    The flow is identical to kalshi_to_snapshot:
      raw_market    → Market (metadata from Gamma)
      raw_orderbook → OrderBook (book from the CLOB)
      OrderBook     → QUOTE Tick (latest price)

    Args:
        raw_market: respuesta de Gamma API GET /markets
        raw_orderbook: the CLOB API GET /book response (optional)
        timestamp: si None, usa datetime.now(UTC)
    """
    market = polymarket_market_to_domain(raw_market)
    ts = timestamp or datetime.now(tz=UTC)

    orderbook: OrderBook | None = None
    last_tick: Tick | None = None

    if raw_orderbook is not None:
        orderbook = polymarket_orderbook_to_domain(market.market_id, raw_orderbook, ts)
        last_tick = polymarket_quote_to_tick(market.market_id, orderbook)

    return MarketSnapshot(
        market=market,
        orderbook=orderbook,
        last_tick=last_tick,
        fetched_at=ts,
    )
