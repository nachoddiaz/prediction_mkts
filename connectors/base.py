"""
connectors/base.py
───────────────────
Interfaz abstracta común para todos los connectors.

Por qué una clase abstracta y no un protocolo (typing.Protocol):
  Protocol es más flexible pero no fuerza la implementación.
  ABC lanza NotImplementedError en runtime si un subclase olvida
  implementar un método — fallo rápido y claro durante el desarrollo.

Por qué los métodos HTTP (_get, _post) están aquí y no en cada connector:
  Retry, timeout y logging son idénticos en todos los connectors.
  Implementarlos una vez en la base evita duplicación y garantiza
  comportamiento consistente ante errores de red.

Por qué los callbacks en subscribe son Callable y no queues:
  Los callbacks permiten que el caller decida qué hacer con cada
  evento — escribir en DuckDB, calcular features, loguear, o todo
  a la vez. Con una queue el connector tendría que saber quién
  la consume. Con callbacks el connector es completamente agnóstico
  del destino de los datos.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

import aiohttp

from normalizer.schema import Market, MarketSnapshot, Tick

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tipos de los callbacks
#
# Por qué Awaitable[None] y no solo None:
#   Los callbacks se llaman desde un loop async. Si el callback hace
#   I/O (escribir en DuckDB, calcular features), necesita ser awaitable.
#   Awaitable[None] permite tanto funciones async como coroutines.
# ---------------------------------------------------------------------------

TickCallback = Callable[[Tick], Awaitable[None]]
SnapshotCallback = Callable[[MarketSnapshot], Awaitable[None]]

# ---------------------------------------------------------------------------
# Constantes de retry
#
# Por qué exponential backoff:
#   Si la API de Kalshi cae momentáneamente, hacer retry inmediato
#   satura el servidor y puede resultar en ban de IP.
#   Con backoff exponencial el primer retry espera 1s, el segundo 2s,
#   el tercero 4s — da tiempo a que el servidor se recupere.
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_SECONDS: int = 10
MAX_RETRIES: int = 3
RETRY_BACKOFF_BASE: float = 1.0  # segundos, se dobla con cada retry


class BaseConnector(ABC):
    """
    Clase base para todos los connectors del sistema.

    Subclases deben implementar:
      - get_markets()    → descubrir mercados activos
      - get_snapshot()   → estado actual de un mercado
      - subscribe()      → stream en tiempo real
      - _build_headers() → autenticación específica de cada venue

    La clase base provee:
      - _get()  → HTTP GET con retry y logging
      - _post() → HTTP POST con retry y logging
      - run()   → loop principal: descubrir → inicializar → suscribir
    """

    def __init__(
        self,
        on_tick: TickCallback,
        on_snapshot: SnapshotCallback,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """
        Args:
            on_tick:     callback llamado por cada Tick del WebSocket
            on_snapshot: callback llamado por cada MarketSnapshot
            timeout:     timeout en segundos para requests HTTP
        """
        self._on_tick = on_tick
        self._on_snapshot = on_snapshot
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------------
    # Métodos abstractos — cada connector los implementa
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_markets(self) -> list[Market]:
        """
        Fetcha la lista de mercados activos desde la API REST.

        Returns:
            Lista de objetos Market del dominio canónico.
            Lista vacía si no hay mercados o hay error.
        """
        ...

    @abstractmethod
    async def get_snapshot(self, market_id: str) -> MarketSnapshot | None:
        """
        Fetcha el estado actual completo de un mercado.

        Args:
            market_id: string canónico "venue:raw_id"

        Returns:
            MarketSnapshot con market + orderbook + last_tick,
            o None si el mercado no existe o hay error.
        """
        ...

    @abstractmethod
    async def subscribe(self, market_ids: list[str]) -> None:
        """
        Se suscribe al stream en tiempo real para los mercados dados.

        Por cada evento del WebSocket llama a:
          self._on_tick(tick)           → trades y quote updates
          self._on_snapshot(snapshot)   → orderbook updates completos

        Este método no retorna hasta que la conexión se cierra.
        El loop de reconexión está en run().

        Args:
            market_ids: lista de strings canónicos "venue:raw_id"
        """
        ...

    @abstractmethod
    def _build_headers(self) -> dict[str, str]:
        """
        Construye los headers de autenticación para esta venue.

        Kalshi:     KALSHI-ACCESS-KEY + KALSHI-ACCESS-SIGNATURE (RSA)
        Polymarket: sin auth para lectura
        Manifold:   sin auth

        Returns:
            Dict de headers HTTP listos para incluir en cada request.
        """
        ...

    # ------------------------------------------------------------------
    # Loop principal — igual para todos los connectors
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Loop principal del connector:
          1. Abrir sesión HTTP
          2. Descubrir mercados activos
          3. Obtener snapshot inicial de cada mercado
          4. Suscribirse al stream WebSocket
          5. Si cae la conexión → esperar y reconectar desde el paso 4

        Por qué reconectar solo desde el paso 4 y no desde el 2:
          Los mercados activos no cambian con cada reconexión.
          Re-fetchar la lista entera en cada reconexión generaría
          carga innecesaria. Solo re-suscribimos el WebSocket.
          La lista de mercados se refresca cada MARKET_REFRESH_INTERVAL.
        """
        async with aiohttp.ClientSession(
            timeout=self._timeout,
            headers=self._build_headers(),
        ) as session:
            self._session = session

            log.info("%s connector starting", self.__class__.__name__)

            # Paso 1: descubrir mercados activos
            markets = await self.get_markets()
            if not markets:
                log.warning("%s: no active markets found", self.__class__.__name__)
                return

            market_ids = [str(m.market_id) for m in markets]
            log.info(
                "%s: found %d active markets",
                self.__class__.__name__,
                len(markets),
            )

            # Paso 2: snapshot inicial de cada mercado y backfill si aplica
            for market_id in market_ids:
                snapshot = await self.get_snapshot(market_id)
                if snapshot is not None:
                    await self._on_snapshot(snapshot)
                    log.debug("Initial snapshot: %s", market_id)
                    # Backfill si la venue lo soporta
                    if hasattr(self, "backfill_market"):
                        try:
                            await self.backfill_market(market_id)
                        except Exception as e:
                            log.warning("Failed to backfill market %s: %s", market_id, e)

            # Paso 3: suscribirse con reconexión automática
            retry = 0
            while True:
                try:
                    log.info(
                        "%s: subscribing to %d markets (attempt %d)",
                        self.__class__.__name__,
                        len(market_ids),
                        retry + 1,
                    )
                    await self.subscribe(market_ids)
                    # subscribe() retornó normalmente — fin del stream
                    log.info("%s: stream ended normally", self.__class__.__name__)
                    break

                except Exception as e:
                    retry += 1
                    wait = RETRY_BACKOFF_BASE * (2 ** min(retry, 6))
                    log.warning(
                        "%s: stream error (attempt %d): %s — retrying in %.1fs",
                        self.__class__.__name__,
                        retry,
                        e,
                        wait,
                    )
                    await asyncio.sleep(wait)

    # ------------------------------------------------------------------
    # HTTP helpers — retry + logging compartidos
    # ------------------------------------------------------------------

    async def _get(
        self,
        url: str,
        params: dict | None = None,
    ) -> dict | list | None:
        """
        HTTP GET con retry exponencial.

        Por qué devolver None en lugar de lanzar excepción:
          Un error de red en un connector no debe derribar el sistema
          entero. El caller decide si None es aceptable o si debe
          reintentar. El logging aquí da visibilidad sin propagar el error.

        Args:
            url:    URL completa del endpoint
            params: query params opcionales

        Returns:
            JSON parseado como dict o list, None si todos los retries fallan.
        """
        if self._session is None:
            log.error("_get called before session was opened")
            return None

        for attempt in range(MAX_RETRIES):
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 200:
                        return await resp.json()

                    # 429 = rate limit — esperar más
                    if resp.status == 429:
                        wait = RETRY_BACKOFF_BASE * (2**attempt) * 2
                        log.warning("Rate limited on GET %s — waiting %.1fs", url, wait)
                        await asyncio.sleep(wait)
                        continue

                    # 4xx client errors — no tiene sentido reintentar
                    if 400 <= resp.status < 500:
                        log.error("Client error %d on GET %s", resp.status, url)
                        return None

                    # 5xx server errors — reintentar
                    log.warning(
                        "Server error %d on GET %s (attempt %d/%d)",
                        resp.status,
                        url,
                        attempt + 1,
                        MAX_RETRIES,
                    )

            except aiohttp.ClientError as e:
                log.warning(
                    "Network error on GET %s (attempt %d/%d): %s",
                    url,
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                )

            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF_BASE * (2**attempt)
                await asyncio.sleep(wait)

        log.error("GET %s failed after %d attempts", url, MAX_RETRIES)
        return None

    async def _post(
        self,
        url: str,
        body: dict,
    ) -> dict | None:
        """
        HTTP POST con retry exponencial.
        Misma lógica que _get — ver comentarios allí.
        """
        if self._session is None:
            log.error("_post called before session was opened")
            return None

        for attempt in range(MAX_RETRIES):
            try:
                async with self._session.post(url, json=body) as resp:
                    if resp.status in (200, 201):
                        return await resp.json()

                    if resp.status == 429:
                        wait = RETRY_BACKOFF_BASE * (2**attempt) * 2
                        await asyncio.sleep(wait)
                        continue

                    if 400 <= resp.status < 500:
                        log.error("Client error %d on POST %s", resp.status, url)
                        return None

                    log.warning(
                        "Server error %d on POST %s (attempt %d/%d)",
                        resp.status,
                        url,
                        attempt + 1,
                        MAX_RETRIES,
                    )

            except aiohttp.ClientError as e:
                log.warning(
                    "Network error on POST %s (attempt %d/%d): %s",
                    url,
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                )

            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_BACKOFF_BASE * (2**attempt))

        log.error("POST %s failed after %d attempts", url, MAX_RETRIES)
        return None

    # ------------------------------------------------------------------
    # Context manager — para uso con async with
    # ------------------------------------------------------------------

    async def __aenter__(self) -> BaseConnector:
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
