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
# Mapa de categorías Kalshi → categorías canónicas del dominio
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
    Infiere la categoría canónica de un market raw de Kalshi.

    Estrategia en dos pasos:
      1. Busca series_ticker en el cache (fuente autoritativa)
      2. Si no está en el cache, fallback a MarketCategory.OTHER

    Por qué no lanzar excepción si no está en el cache:
      El cache puede estar incompleto en el arranque o si Kalshi
      añade una nueva serie entre actualizaciones. Es preferible
      categorizar como OTHER que fallar la ingesta entera.

    Args:
        raw: objeto market raw de la API de Kalshi
    """
    series_ticker = raw.get("series_ticker", "")
    raw_category = _series_category_cache.get(series_ticker, "")
    return _KALSHI_CATEGORY_MAP.get(raw_category, MarketCategory.OTHER)


# ---------------------------------------------------------------------------
# Parsing de timestamps
# ---------------------------------------------------------------------------


def _parse_ts(ts: str) -> datetime:
    """
    Parsea un timestamp ISO-8601 de Kalshi a datetime UTC-aware.

    Por qué dos formatos:
      Kalshi devuelve timestamps con y sin microsegundos dependiendo
      del endpoint:
        "2024-12-31T23:59:59Z"        → sin microsegundos
        "2024-12-31T23:59:59.123456Z" → con microsegundos
      Probamos el más específico primero para evitar pérdida de precisión.

    Args:
        ts: string de timestamp de la API de Kalshi

    Raises:
        ValueError: si el formato no es reconocible
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
# Conversión de precios
# ---------------------------------------------------------------------------


def _cents_to_prob(cents: int | float) -> Price:
    """
    Convierte el precio en centavos de Kalshi (0–100) a probabilidad (0.0–1.0).
    """
    return Price(round(float(cents) / 100.0, 6))


def _dollars_to_prob(value: int | float | str) -> Price:
    """
    Convierte precio de Kalshi a probabilidad en [0, 1].

    La API REST devuelve strings dólar: "0.0100" = prob 0.01.
    El endpoint de orderbook usa centavos: 45 = prob 0.45.
    Diferencia clave: si el valor > 1.0 son centavos, si <= 1.0 ya es probabilidad.
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
    Convierte un objeto market de GET /markets/{ticker} al dominio canónico.

    Lógica de status:
      Kalshi usa "open", "closed", "settled", "finalized".
      Mapeamos "settled" y "finalized" a RESOLVED porque ambos significan
      que el resultado es conocido. "closed" significa trading parado
      pero resultado pendiente (e.g., elecciones en curso).

    Lógica de resolved_value:
      Solo se rellena si status es RESOLVED.
      "yes" → 1.0, "no" → 0.0, None → None (no resuelto).
      Usamos float en lugar de bool para consistencia con el schema
      y con el cálculo de Bernoulli vol: σ_B = sqrt(p(1-p)/τ).

    Args:
        raw: dict con la respuesta de GET /markets/{ticker}

    Raises:
        ValueError: si falta el campo close_time (requerido para τ)
    """
    ticker: str = raw["ticker"]
    market_id = MarketId(venue=Venue.KALSHI, raw_id=ticker)

    # --- Mapeo de status ---
    # Usamos dict lookup en lugar de if/elif para extensibilidad:
    # añadir un nuevo status de Kalshi es añadir una línea al dict.
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
    # Solo miramos "result" si el mercado está resuelto.
    # Si está abierto, result puede ser None o no existir — es normal.
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
    )


# ---------------------------------------------------------------------------
# OrderBook adapter
# ---------------------------------------------------------------------------


def kalshi_orderbook_to_domain(
    market_id: MarketId,
    raw: dict[str, Any],
    timestamp: datetime | None = None,
) -> OrderBook:
    """
    Convierte la respuesta de GET /markets/{ticker}/orderbook al dominio.

    Estructura del orderbook de Kalshi:
      {
        "orderbook": {
          "yes": [[price_cents, size], [price_cents, size], ...],
          "no":  [[price_cents, size], ...]
        }
      }

    Necesitamos transformar el lado NO:

        Comprador de NO a precio p_no
        = Vendedor de YES a precio (1 - p_no/100)

      Ejemplo: alguien dispuesto a pagar 53 centavos por NO
      equivale a alguien dispuesto a vender YES a 0.47.

    Ordenamos bids desc y asks asc.

    Por qué filtramos niveles cruzados en lugar de lanzar excepción:
      En mercados ilíquidos o durante actualizaciones de libro,
      puede aparecer momentáneamente un nivel cruzado. Es un artefacto
      de la API, no un error de nuestro código. Filtramos los niveles
      problemáticos y continuamos — un libro parcial es mejor que
      no tener libro.

    Args:
        market_id: MarketId ya construido (para no recalcularlo)
        raw: dict con la respuesta del endpoint de orderbook
        timestamp: si None, usa datetime.now(UTC)
    """
    ts = timestamp or datetime.now(tz=UTC)

    book = raw.get("orderbook", raw)

    raw_yes: list[list[int]] = book.get("yes", [])
    raw_no: list[list[int]] = book.get("no", [])

    yes_bids: list[OrderBookLevel] = [
        OrderBookLevel(price=_cents_to_prob(p), size=Size(float(s))) for p, s in raw_yes if s > 0
    ]

    yes_asks: list[OrderBookLevel] = [
        OrderBookLevel(price=_cents_to_prob(100 - p), size=Size(float(s)))
        for p, s in raw_no
        if s > 0
    ]

    # Ordenar: bids descendente, asks ascendente
    yes_bids.sort(key=lambda lv: lv.price, reverse=True)
    yes_asks.sort(key=lambda lv: lv.price)

    # Filtrar niveles cruzados si los hay
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
    Convierte un mensaje de trade del WebSocket de Kalshi a un Tick canónico.

    Formato del mensaje WS de Kalshi:
      {
        "type": "trade",
        "msg": {
          "market_ticker": "BTCZ-...",
          "yes_price": 45,     ← precio al que se ejecutó en centavos
          "no_price":  55,     ← siempre 100 - yes_price
          "count": 10,         ← número de contratos
          "taker_side": "yes", ← quién fue el agresivo
          "created_time": "2024-12-31T12:00:00Z"
        }
      }

    Bid == ask == precio de ejecución.

    El taker es quien envió la market order que cruzó el spread.
    Esta información es fundamental para calcular adverse selection
    en features/microstructure.py.

    """
    msg = raw.get("msg", raw)

    yes_price_cents: int = msg["yes_price"]
    no_price_cents: int = msg["no_price"]

    # yes_bid = precio al que se ejecutó el YES
    # yes_ask = precio implícito del NO convertido a YES
    # En práctica siempre iguales, pero mantenemos la estructura
    # por consistencia con el schema
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
    Deriva un QUOTE tick desde un snapshot de OrderBook.

    Este Tick representa "el mid-price en este instante"
    sin que haya habido un trade real.

    Un orderbook vacío en un lado es válido en mercados muy
    ilíquidos. El caller decide qué hacer con None
    (típicamente: ignorar ese snapshot).

    Args:
        market_id: MarketId del mercado
        orderbook: snapshot del libro ya construido y validado

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
        # volume=0 porque no hubo trade — es solo una actualización de precio
        volume=Size(0.0),
        # side=None porque no hay agresión en un QUOTE
        side=None,
    )


# ---------------------------------------------------------------------------
# Convenience: snapshot completo desde respuestas REST
# ---------------------------------------------------------------------------


def kalshi_to_snapshot(
    raw_market: dict[str, Any],
    raw_orderbook: dict[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> MarketSnapshot:
    """
    Construye un MarketSnapshot completo desde respuestas de la API REST.

    Args:
        raw_market: respuesta de GET /markets/{ticker}
        raw_orderbook: respuesta de GET /markets/{ticker}/orderbook (opcional)
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
