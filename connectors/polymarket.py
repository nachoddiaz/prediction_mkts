"""
connectors/polymarket.py
─────────────────────────
Connector para Polymarket — venue on-chain sobre Polygon.

Dos APIs:
  Gamma API  → metadatos de mercados (sin auth)
  CLOB API   → orderbook y trades en tiempo real (sin auth para lectura)

WebSocket:
  CLOB WS → price_change events y trade events
  Sin auth para suscribirse a eventos de precio.

Por qué no tenemos orderbook completo sin auth:
  El endpoint /book del CLOB requiere autenticación EIP-712.
  Para lectura usamos /midpoint y /price como fallback.
  El orderbook completo estará disponible en Fase 4 con credenciales.
"""

from __future__ import annotations

import asyncio
import json
import logging
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
    polymarket_price_update_to_tick,
    polymarket_quote_to_tick,
    polymarket_trade_to_tick,
)
from normalizer.schema import (
    Market,
    MarketSnapshot,
    OrderBook,
    OrderBookLevel,
    Price,
    Size,
)

log = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

MARKET_FETCH_LIMIT = 50


class PolymarketConnector(BaseConnector):
    """
    Connector para Polymarket.

    Diferencia clave respecto a Kalshi:
      No hay auth para lectura — ningún header especial necesario.
      Los token IDs (clobTokenIds) son necesarios para el WebSocket —
      son los identificadores on-chain del token YES de cada mercado.
    """

    def __init__(
        self,
        on_tick: TickCallback,
        on_snapshot: SnapshotCallback,
    ) -> None:
        super().__init__(on_tick, on_snapshot)
        # Cache: market_id canónico → yes_token_id
        # Necesario para construir snapshots desde eventos del WS
        # que solo contienen el token_id, no el market_id canónico
        self._token_to_market: dict[str, str] = {}
        self._markets_cache: dict[str, Market] = {}

    def _build_headers(self) -> dict[str, str]:
        """Polymarket no requiere auth para lectura."""
        return {"Accept": "application/json"}

    async def get_markets(self) -> list[Market]:
        """
        Fetcha mercados activos desde la Gamma API.

        Por qué ordenar por volume24hr:
          Los mercados con más volumen son los más líquidos y los
          más interesantes para el sistema de trading. Fetchar los
          top N por volumen garantiza que trabajamos con mercados
          donde el spread óptimo tiene sentido.
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
                # Inferir tags si vienen vacíos (bug conocido de la Gamma API)
                tags = raw.get("tags", [])
                if not tags:
                    tags = self._infer_tags(raw.get("question", ""))
                raw_enriched = {**raw, "tags": tags}

                market = polymarket_market_to_domain(raw_enriched)
                markets.append(market)

                # Guardar en cache: token_id → market_id
                token_ids = _parse_clob_token_ids(raw)
                if token_ids:
                    yes_id, _ = token_ids
                    self._token_to_market[yes_id] = str(market.market_id)
                    self._markets_cache[str(market.market_id)] = market

            except Exception as e:
                log.warning("Polymarket: failed to parse market: %s", e)

        log.info("Polymarket: fetched %d active markets", len(markets))
        return markets

    async def get_snapshot(self, market_id: str) -> MarketSnapshot | None:
        """
        Fetcha el estado actual de un mercado de Polymarket.

        Por qué usar /midpoint y /price en lugar de /book:
          /book requiere auth EIP-712.
          /midpoint y /price son endpoints públicos que nos dan
          el mid, bid y ask sin credenciales.
          Suficiente para el snapshot inicial — el orderbook completo
          llegará por el WebSocket.
        """
        market = self._markets_cache.get(market_id)
        if not market:
            return None

        # Buscar el yes_token_id para este market_id
        yes_token = None
        for token, mid in self._token_to_market.items():
            if mid == market_id:
                yes_token = token
                break

        if not yes_token:
            return None

        # Fetchear mid, bid y ask en paralelo
        mid_url = f"{CLOB_BASE}/midpoint?token_id={yes_token}"
        buy_url = f"{CLOB_BASE}/price?token_id={yes_token}&side=BUY"
        sell_url = f"{CLOB_BASE}/price?token_id={yes_token}&side=SELL"

        mid_data, buy_data, sell_data = await asyncio.gather(
            self._get(mid_url),
            self._get(buy_url),
            self._get(sell_url),
            return_exceptions=True,
        )

        if not (mid_data and buy_data and sell_data):
            return None
        if any(isinstance(d, Exception) for d in (mid_data, buy_data, sell_data)):
            return None

        try:
            # BUY price = best ask, SELL price = best bid
            best_ask = float(buy_data["price"])
            best_bid = float(sell_data["price"])

            # Corregir si están invertidos (puede ocurrir en mercados extremos)
            if best_bid >= best_ask:
                mid = float(mid_data["mid"])
                best_bid = mid - 0.001
                best_ask = mid + 0.001

            ob = OrderBook(
                market_id=market.market_id,
                timestamp=datetime.now(tz=UTC),
                bids=(OrderBookLevel(Price(round(best_bid, 6)), Size(0.0)),),
                asks=(OrderBookLevel(Price(round(best_ask, 6)), Size(0.0)),),
            )
            tick = polymarket_quote_to_tick(market.market_id, ob)

            return MarketSnapshot(
                market=market,
                orderbook=ob,
                last_tick=tick,
            )

        except Exception as e:
            log.warning("Polymarket: failed to build snapshot for %s: %s", market_id, e)
            return None

    async def subscribe(self, market_ids: list[str]) -> None:
        """
        Se suscribe al WebSocket del CLOB de Polymarket.

        El WS del CLOB usa token_ids (on-chain), no market_ids canónicos.
        Por eso necesitamos el cache _token_to_market construido en
        get_markets().

        Tipos de mensaje:
          price_change → Tick QUOTE
          trade        → Tick TRADE
          book         → OrderBook completo (si hay auth)
        """
        # Obtener los yes_token_ids para los markets que queremos
        token_ids = [token for token, mid in self._token_to_market.items() if mid in market_ids]

        if not token_ids:
            log.warning("Polymarket WS: no token IDs found for market_ids")
            return

        async with websockets.connect(CLOB_WS) as ws:
            # Suscribirse a todos los token_ids
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
        Procesa un evento del WebSocket del CLOB.

        event_type:
          price_change → Tick QUOTE desde polymarket_price_update_to_tick
          trade        → Tick TRADE desde polymarket_trade_to_tick
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
        Infiere tags desde la pregunta cuando la API devuelve tags vacíos.
        Ver polymarket_adapter.py para la misma lógica — duplicada aquí
        para no importar desde el adapter en el connector.
        """
        q = question.lower()
        if any(w in q for w in ["bitcoin", "btc", "eth", "crypto", "sol"]):
            return ["crypto"]
        if any(w in q for w in ["election", "president", "senate"]):
            return ["politics"]
        if any(w in q for w in ["fed", "rate", "inflation", "gdp"]):
            return ["economics"]
        return []
