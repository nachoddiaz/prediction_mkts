# This adapter  transforms raw responses from the Polymarket API
# into domain objects defined in schema.py.
# In Polymarket we have 2 different APIs:
#   Gamma API:  Metadata (title, category, settlement date, status), estimated prices
#   CLOB API: Real-time Order Book, Real trades, WS for streaming, requires EIP-721

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

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
    Extrae los token IDs YES y NO del campo clobTokenIds.

    La Gamma API devuelve clobTokenIds como string JSON, no como array:
      "clobTokenIds": "[\"13915...\", \"13290...\"]"

    Necesitamos json.loads() para parsearlo.
    Devuelve (yes_token_id, no_token_id) o None si no existe.
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


def _infer_category_poly(tags: list[str] | list[dict]) -> MarketCategory:
    for tag in tags:
        # Handle both string tags and dict tags with 'label' key
        tag_str = tag.lower() if isinstance(tag, str) else tag.get("label", "").lower()
        if tag_str in _POLY_TAG_MAP:
            return _POLY_TAG_MAP[tag_str]
    return MarketCategory.OTHER


# ---------------------------------------------------------------------------
# Parsing de timestamps
#
# Por qué dos funciones separadas (_parse_ts_gamma y _parse_ts_clob):
#   Gamma API y CLOB API usan formatos de timestamp distintos.
# ---------------------------------------------------------------------------


def _parse_ts_gamma(ts: int | str) -> datetime:
    if isinstance(ts, int | float):
        # Unix milliseconds → segundos → datetime UTC
        return datetime.fromtimestamp(ts / 1000.0, tz=UTC)

    # Fallback: ISO string (algunos endpoints de Gamma usan esto)
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
# Market adapter — desde Gamma API
# ---------------------------------------------------------------------------


def polymarket_market_to_domain(raw: dict[str, Any]) -> Market:
    """
    Convierte un objeto market de la Gamma API al dominio canónico.

    Lógica de status — por qué este orden de prioridad:
      Comprobamos "resolved" primero porque un mercado puede tener
      resolved=True y closed=True simultáneamente. Si comprobáramos
      closed primero, perderíamos la información de resolución.

    Lógica de resolved_value:
      Después de resolución, el token ganador tiene price ≈ 1.0
      y el perdedor price ≈ 0.0. Usamos >= 0.99 en lugar de == 1.0
      porque Polymarket puede devolver 0.9999... por precisión
      de punto flotante en el contrato ERC-1155.

    Args:
        raw: dict con la respuesta de Gamma API GET /markets

    Raises:
        ValueError: si falta end_date (requerido para calcular τ)
    """
    condition_id: str = raw.get("conditionId") or raw.get("condition_id") or raw["id"]
    market_id = MarketId(venue=Venue.POLYMARKET, raw_id=condition_id)

    # --- Status ---
    # Orden de comprobación importante: resolved > closed > active
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

    # El token YES con price >= 0.99 indica que YES ganó.
    resolved_value: float | None = None
    if status == MarketStatus.RESOLVED:
        tokens: list[dict] = raw.get("tokens", [])
        for token in tokens:
            if token.get("outcome", "").lower() == "yes":
                price = float(token.get("price", 0.0))
                # >= 0.99 en lugar de == 1.0 por precisión floating point
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
    )


# ---------------------------------------------------------------------------
# OrderBook adapter — desde CLOB API
# ---------------------------------------------------------------------------


def polymarket_orderbook_to_domain(
    market_id: MarketId,
    raw: dict[str, Any],
    timestamp: datetime | None = None,
) -> OrderBook:
    """
    Convierte la respuesta de GET /book del CLOB al dominio canónico.

    Formato del CLOB (precios ya en [0,1], no en centavos):
      {
        "bids": [{"price": "0.45", "size": "100"}, ...],
        "asks": [{"price": "0.47", "size": "150"}, ...]
      }

    La CLOB API devuelve precios como strings ("0.45"), no como
    floats.

    Args:
        market_id: MarketId ya construido
        raw: respuesta de GET /book del CLOB
        timestamp: si None, usa datetime.now(UTC)
    """
    ts = timestamp or datetime.now(tz=UTC)

    raw_bids: list[dict] = raw.get("bids", [])
    raw_asks: list[dict] = raw.get("asks", [])

    # Construir bids — filtramos size == 0 (niveles vacíos)
    bids: list[OrderBookLevel] = [
        OrderBookLevel(
            price=Price(round(float(lv["price"]), 6)),
            size=Size(float(lv["size"])),
        )
        for lv in raw_bids
        if float(lv.get("size", 0)) > 0
    ]

    # Construir asks — mismo tratamiento que bids
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

    # Filtrar niveles cruzados — misma lógica que Kalshi
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
    Convierte un evento de trade del CLOB a un Tick canónico.

    Formato del trade en el CLOB REST y WebSocket:
      {
        "price":     "0.46",          ← precio de ejecución en [0,1]
        "size":      "50",            ← contratos ejecutados
        "side":      "BUY",           ← BUY=compró YES, SELL=vendió YES
        "timestamp": "2024-12-31T12:00:00Z"
      }

    Por qué "BUY" → Side.YES y "SELL" → Side.NO

    Args:
        market_id: MarketId ya construido
        raw: evento de trade del CLOB
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
        # bid == ask == precio de ejecución en trades
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
    Deriva un QUOTE tick desde un snapshot de OrderBook de Polymarket.

    Returns:
        Tick de tipo QUOTE, o None si el libro está incompleto
    """
    if orderbook.best_bid is None or orderbook.best_ask is None:
        return None

    return Tick(
        market_id=market_id,
        timestamp=orderbook.timestamp,
        tick_type=TickType.QUOTE,
        yes_bid=orderbook.best_bid,
        yes_ask=orderbook.best_ask,
        volume=Size(0.0),  # sin trade, sin volumen
        side=None,  # sin agresión, sin side
    )


def polymarket_price_update_to_tick(
    market_id: MarketId,
    raw: dict[str, Any],
) -> Tick | None:
    """
    Convierte un mensaje de price_change del WebSocket del CLOB a un Tick.

    Polymarket emite estos mensajes cuando el mid-price cambia
    sin que haya un trade completo.

    Formato del mensaje WS:
      {
        "asset_id":  "...",       ← token_id del contrato YES
        "price":     "0.46",      ← nuevo mid-price
        "side":      "BUY",
        "size":      "0",         ← 0 porque no hay fill
        "timestamp": 1704067200000  ← Unix ms
      }

    Args:
        market_id: MarketId ya construido
        raw: mensaje raw del WebSocket del CLOB

    Returns:
        Tick de tipo QUOTE, o None si el mensaje no tiene precio
    """
    price_str = raw.get("price")
    if price_str is None:
        # Mensaje sin precio — heartbeat o mensaje de status
        return None

    price = Price(round(float(price_str), 6))

    # El timestamp puede venir como Unix ms (int) o ISO string
    ts_raw = raw.get("timestamp")
    if isinstance(ts_raw, int | float):
        ts = datetime.fromtimestamp(ts_raw / 1000.0, tz=UTC)
    else:
        ts = datetime.now(tz=UTC)

    # Para price updates, bid ≈ ask ≈ precio (aproximación)
    # No tenemos profundidad del libro en este mensaje,
    # solo el mid-price. El feature store lo trata como QUOTE.
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
# Convenience: snapshot completo desde respuestas REST
# ---------------------------------------------------------------------------


def polymarket_to_snapshot(
    raw_market: dict[str, Any],
    raw_orderbook: dict[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> MarketSnapshot:
    """
    Construye un MarketSnapshot completo desde respuestas de las APIs REST.

    raw_market viene de Gamma y raw_orderbook del CLOB

    Flujo idéntico al de kalshi_to_snapshot:
      raw_market    → Market (metadatos desde Gamma)
      raw_orderbook → OrderBook (libro desde CLOB)
      OrderBook     → Tick de tipo QUOTE (último precio)

    Args:
        raw_market: respuesta de Gamma API GET /markets
        raw_orderbook: respuesta de CLOB API GET /book (opcional)
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
