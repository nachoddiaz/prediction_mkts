"""
connectors/kalshi.py
─────────────────────
Connector para Kalshi — venue regulada por la CFTC.

Auth: RSA signing
  Cada request REST lleva:
    KALSHI-ACCESS-KEY:       el API key
    KALSHI-ACCESS-TIMESTAMP: Unix timestamp en milisegundos
    KALSHI-ACCESS-SIGNATURE: base64(RSA-SHA256(timestamp + method + path))

  La firma se calcula sobre: timestamp (ms) + method (GET/POST) + path (/trade-api/v2/markets)
  La private key es un archivo PEM en disco — nunca en el código.

WebSocket:
  Protocolo propio con login + subscribe commands.
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
from normalizer.schema import Market, MarketSnapshot, Venue

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URLs por entorno
# ---------------------------------------------------------------------------

KALSHI_REST_DEMO = "https://demo-api.kalshi.co/trade-api/v2"
KALSHI_REST_PROD = "https://trading-api.kalshi.com/trade-api/v2"
KALSHI_WS_DEMO = "wss://demo-api.kalshi.co/trade-api/ws/v2"
KALSHI_WS_PROD = "wss://trading-api.kalshi.com/trade-api/ws/v2"

# Cuántos mercados activos fetchar al arrancar
MARKET_FETCH_LIMIT = 100


class KalshiConnector(BaseConnector):
    """
    Connector para Kalshi.

    Uso:
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
    ) -> None:
        """
        Args:
            on_tick:          callback para cada Tick
            on_snapshot:      callback para cada MarketSnapshot
            api_key:          Kalshi API key (o KALSHI_API_KEY del env)
            private_key_path: ruta al archivo PEM (o KALSHI_PRIVATE_KEY_PATH)
            env:              "demo" | "prod"
        """
        super().__init__(on_tick, on_snapshot)

        self._api_key = api_key or os.getenv("KALSHI_API_KEY", "")
        key_path = private_key_path or os.getenv(
            "KALSHI_PRIVATE_KEY_PATH", "./secrets/kalshi_private.pem"
        )

        # Cargar la private key RSA desde disco
        # Por qué cargarla en __init__ y no en cada request:
        #   Leer y parsear un archivo PEM tiene un coste no trivial.
        #   Cargándola una vez al inicializar evitamos ese coste
        #   en cada request — que puede ser decenas por segundo.
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
        Construye headers de autenticación RSA para Kalshi.

        La firma es: base64(RSA-SHA256(timestamp_ms + method + path))
        El timestamp se incluye en el header para que Kalshi pueda
        verificar que el request no es un replay de un request antiguo.

        Por qué RSA y no HMAC:
          Kalshi usa RSA para que puedas rotar keys sin compartir
          el secreto con nadie — solo la public key va a Kalshi.
          Con HMAC tendrías que compartir el secret con Kalshi.
        """
        return {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self._api_key,
        }

    def _sign(self, method: str, path: str) -> dict[str, str]:
        """
        Genera los headers de firma para un request específico.

        Por qué timestamp en milisegundos:
          Kalshi usa ms para mayor precisión en la ventana anti-replay.
          Requests con timestamp > 30s en el pasado son rechazados.

        Args:
            method: "GET" | "POST"
            path:   path del endpoint e.g. "/trade-api/v2/markets"

        Returns:
            Dict con los tres headers de autenticación.
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
    # Implementación de los métodos abstractos
    # ------------------------------------------------------------------

    async def get_markets(self) -> list[Market]:
        """
        Fetcha mercados activos de Kalshi.

        Flujo:
          1. GET /series → poblar cache de categorías
          2. GET /markets?status=open → lista de mercados

        Por qué /series primero:
          La categoría de un market vive en su series, no en el market.
          Sin el cache de series, todos los mercados tendrían
          category=OTHER. Ver kalshi_adapter.py para más detalle.
        """
        # Paso 1: poblar cache de categorías
        path = "/trade-api/v2/series"
        series_data = await self._get(
            self._base_url + path,
        )
        if series_data and "series" in series_data:
            build_series_cache(series_data["series"])
            log.debug("Kalshi series cache built: %d series", len(series_data["series"]))

        # Paso 2: mercados activos
        path = "/trade-api/v2/markets"
        data = await self._get(
            self._base_url + path,
            params={"status": "open", "limit": MARKET_FETCH_LIMIT},
        )

        if not data or "markets" not in data:
            return []

        markets = []
        for raw in data["markets"]:
            try:
                markets.append(kalshi_market_to_domain(raw))
            except Exception as e:
                log.warning("Failed to parse Kalshi market: %s", e)

        log.info("Kalshi: fetched %d active markets", len(markets))
        return markets

    async def get_snapshot(self, market_id: str) -> MarketSnapshot | None:
        """
        Fetcha el estado actual de un mercado de Kalshi.

        market_id tiene formato "kalshi:KXBTC-26APR22-T85000".
        Extraemos el raw_id (KXBTC-26APR22-T85000) para construir la URL.
        """
        raw_id = market_id.split(":", 1)[-1]

        # Fetchear market y orderbook en paralelo para reducir latencia
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

        try:
            return kalshi_to_snapshot(
                raw_market=market_data,
                raw_orderbook=orderbook_data if isinstance(orderbook_data, dict) else None,
                timestamp=datetime.now(tz=UTC),
            )
        except Exception as e:
            log.warning("Kalshi: failed to build snapshot for %s: %s", raw_id, e)
            return None

    async def subscribe(self, market_ids: list[str]) -> None:
        """
        Se suscribe al WebSocket de Kalshi para los mercados dados.

        Protocolo del WS de Kalshi:
          1. Conectar
          2. Enviar mensaje login con el API key
          3. Enviar mensajes subscribe por cada channel y market
          4. Recibir mensajes indefinidamente

        Channels disponibles:
          orderbook_delta → actualizaciones del libro
          trade           → trades ejecutados
          ticker          → cambios de precio/best bid/ask

        Por qué extraer raw_ids antes de conectar:
          El WS de Kalshi usa los tickers nativos (KXBTC-...), no
          los market_ids canónicos (kalshi:KXBTC-...).
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
        Procesa un mensaje del WebSocket de Kalshi.

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
        Devuelve un Market mínimo para construir snapshots en el WS handler.
        En producción esto vendría de una cache de markets en memoria.
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
