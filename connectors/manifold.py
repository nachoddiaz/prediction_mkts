"""
connectors/manifold.py
───────────────────────
Connector para Manifold Markets — sandbox con play-money (Mana).

Por qué Manifold como sandbox:
  - Sin auth, sin credenciales, sin riesgo financiero
  - API pública con 500 req/min
  - Misma estructura de mercados binarios que Kalshi/Polymarket
  - Permite testear el pipeline completo end-to-end

Diferencia clave: sin WebSocket nativo.
  Simulamos streaming con polling periódico cada POLL_INTERVAL segundos.
  Suficiente para validar que writer, feature store y strategies
  funcionan correctamente antes de conectar las APIs reales.

Limitaciones:
  - Liquidez muy baja (play-money)
  - Sin orderbook real — solo last_price
  - Sin trades en tiempo real — solo polling de bets
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from connectors.base import (
    BaseConnector,
    SnapshotCallback,
    TickCallback,
)
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
    Size,
    Tick,
    TickType,
    Venue,
)

log = logging.getLogger(__name__)

MANIFOLD_BASE = "https://manifold.markets/api/v0"
POLL_INTERVAL = 10  # segundos entre polls
MARKET_LIMIT = 20  # mercados a fetchar


class ManifoldConnector(BaseConnector):
    """
    Connector para Manifold Markets (sandbox).

    Por qué polling en lugar de WebSocket:
      Manifold no tiene WebSocket. El polling cada 10s es suficiente
      para testear el pipeline — en producción usaríamos Kalshi/Polymarket
      que sí tienen WebSocket real.

    Por qué útil a pesar de las limitaciones:
      Permite correr el sistema completo (connector → writer → features
      → strategies) en local sin credenciales ni riesgo financiero.
      Si el sistema funciona con Manifold, funciona con las APIs reales.
    """

    def __init__(
        self,
        on_tick: TickCallback,
        on_snapshot: SnapshotCallback,
        poll_interval: int = POLL_INTERVAL,
    ) -> None:
        super().__init__(on_tick, on_snapshot)
        self._poll_interval = poll_interval
        # Cache de último precio por market_id para detectar cambios
        # Si el precio no cambió desde el último poll, no emitimos tick
        self._last_prices: dict[str, float] = {}

    def _build_headers(self) -> dict[str, str]:
        """Manifold no requiere auth."""
        return {"Accept": "application/json"}

    async def get_markets(self) -> list[Market]:
        """
        Fetcha mercados binarios activos de Manifold.

        Manifold devuelve muchos tipos de mercados (múltiple choice,
        numeric, etc.). Filtramos solo los binarios (YES/NO) que son
        equivalentes a los contratos binarios de Kalshi/Polymarket.
        """
        data = await self._get(
            f"{MANIFOLD_BASE}/markets",
            params={
                "limit": MARKET_LIMIT,
                "sort": "liquidity",
                "filter": "open",
                "contractType": "BINARY",
            },
        )

        if not data or not isinstance(data, list):
            return []

        markets = []
        for raw in data:
            try:
                market = self._raw_to_market(raw)
                if market:
                    markets.append(market)
            except Exception as e:
                log.warning("Manifold: failed to parse market: %s", e)

        log.info("Manifold: fetched %d binary markets", len(markets))
        return markets

    async def get_snapshot(self, market_id: str) -> MarketSnapshot | None:
        """
        Fetcha el estado actual de un mercado de Manifold.

        Manifold no tiene orderbook real — solo probability (mid-price).
        Construimos un orderbook sintético de un nivel con spread fijo.

        Por qué spread fijo de 0.02:
          Sin orderbook real no sabemos el spread. 0.02 (2%) es
          una aproximación conservadora para mercados de play-money.
          En backtesting esto se marca claramente como sintético.
        """
        raw_id = market_id.split(":", 1)[-1]
        data = await self._get(f"{MANIFOLD_BASE}/market/{raw_id}")

        if not data:
            return None

        try:
            market = self._raw_to_market(data)
            if not market:
                return None

            prob = float(data.get("probability", 0.5))
            ob = self._synthetic_orderbook(market.market_id, prob)
            tick = Tick(
                market_id=market.market_id,
                timestamp=datetime.now(tz=UTC),
                tick_type=TickType.QUOTE,
                yes_bid=Price(ob.best_bid or prob),
                yes_ask=Price(ob.best_ask or prob),
                volume=Size(0.0),
                side=None,
            )
            return MarketSnapshot(market=market, orderbook=ob, last_tick=tick)

        except Exception as e:
            log.warning("Manifold: failed to build snapshot for %s: %s", market_id, e)
            return None

    async def subscribe(self, market_ids: list[str]) -> None:
        """
        Simula streaming con polling periódico.

        Por cada mercado: fetcha el estado actual y lo compara con
        el último precio conocido. Si cambió, emite un Tick.

        Por qué comparar con el último precio:
          Sin WebSocket no sabemos exactamente cuándo cambió el precio.
          Emitir un tick solo cuando el precio cambia evita inundar
          el writer con ticks idénticos — que subirían el storage
          sin añadir información nueva.
        """
        log.info(
            "Manifold: starting polling loop (%ds interval) for %d markets",
            self._poll_interval,
            len(market_ids),
        )

        while True:
            for market_id in market_ids:
                try:
                    await self._poll_market(market_id)
                except Exception as e:
                    log.warning("Manifold: poll error for %s: %s", market_id, e)

            await asyncio.sleep(self._poll_interval)

    async def _poll_market(self, market_id: str) -> None:
        """
        Fetcha el estado de un mercado y emite tick si el precio cambió.
        """
        raw_id = market_id.split(":", 1)[-1]
        data = await self._get(f"{MANIFOLD_BASE}/market/{raw_id}")

        if not data:
            return

        prob = float(data.get("probability", 0.0))
        if prob <= 0 or prob >= 1:
            return

        last = self._last_prices.get(market_id)

        # Emitir tick si el precio cambió más de 0.1% desde el último poll
        if last is None or abs(prob - last) > 0.001:
            self._last_prices[market_id] = prob

            market = self._raw_to_market(data)
            if not market:
                return

            ob = self._synthetic_orderbook(market.market_id, prob)
            tick = Tick(
                market_id=market.market_id,
                timestamp=datetime.now(tz=UTC),
                tick_type=TickType.QUOTE,
                yes_bid=Price(ob.best_bid or prob),
                yes_ask=Price(ob.best_ask or prob),
                volume=Size(0.0),
                side=None,
            )
            snapshot = MarketSnapshot(
                market=market,
                orderbook=ob,
                last_tick=tick,
            )
            await self._on_snapshot(snapshot)
            log.debug(
                "Manifold: price update %s %.4f → %.4f",
                market_id,
                last or 0,
                prob,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _raw_to_market(self, raw: dict) -> Market | None:
        """
        Convierte un market raw de Manifold al dominio canónico.

        Manifold usa slugs como IDs (e.g. "will-btc-hit-150k-2026")
        en lugar de tickers o condition_ids.
        """
        slug = raw.get("slug") or raw.get("id")
        if not slug:
            return None

        # Fecha de cierre — Manifold usa closeTime en Unix ms
        close_ms = raw.get("closeTime")
        if not close_ms:
            return None

        from datetime import datetime

        resolution_date = datetime.fromtimestamp(close_ms / 1000, tz=UTC)

        # Resultado si está resuelto
        resolved_value: float | None = None
        if raw.get("isResolved"):
            resolution = raw.get("resolution", "")
            if resolution == "YES":
                resolved_value = 1.0
            elif resolution == "NO":
                resolved_value = 0.0

        status = MarketStatus.RESOLVED if raw.get("isResolved") else MarketStatus.OPEN

        return Market(
            market_id=MarketId(Venue.MANIFOLD, slug),
            question=raw.get("question", slug),
            category=self._infer_category(raw.get("question", "")),
            resolution=Resolution(
                resolution_date=resolution_date,
                resolved_value=resolved_value,
            ),
            status=status,
        )

    @staticmethod
    def _synthetic_orderbook(
        market_id: MarketId,
        prob: float,
        spread: float = 0.02,
    ) -> OrderBook:
        """
        Construye un orderbook sintético de un nivel desde la probabilidad.

        Por qué sintético:
          Manifold no tiene CLOB. La probability es el mid-price.
          Construimos bid = prob - spread/2 y ask = prob + spread/2
          para mantener la invariante del schema (bid < ask).

        Args:
            market_id: MarketId del mercado
            prob:      probabilidad actual (mid-price)
            spread:    spread sintético fijo (default 2%)
        """
        bid = max(0.001, round(prob - spread / 2, 4))
        ask = min(0.999, round(prob + spread / 2, 4))

        # Garantizar que bid < ask aunque prob esté en los extremos
        if bid >= ask:
            bid = round(prob - 0.001, 4)
            ask = round(prob + 0.001, 4)

        return OrderBook(
            market_id=market_id,
            timestamp=datetime.now(tz=UTC),
            bids=(OrderBookLevel(Price(bid), Size(0.0)),),
            asks=(OrderBookLevel(Price(ask), Size(0.0)),),
        )

    @staticmethod
    def _infer_category(question: str) -> MarketCategory:
        q = question.lower()
        if any(w in q for w in ["bitcoin", "btc", "eth", "crypto"]):
            return MarketCategory.CRYPTO
        if any(w in q for w in ["election", "president", "senate", "vote"]):
            return MarketCategory.POLITICS
        if any(w in q for w in ["fed", "rate", "inflation", "gdp"]):
            return MarketCategory.ECONOMICS
        return MarketCategory.OTHER
