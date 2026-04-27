"""
features/store.py
──────────────────
Orquestador del feature store — conecta microstructure.py,
resolution.py, el reader y el writer en un único punto de entrada.

Responsabilidades:
  1. Recibir un market_id + snapshot/tick nuevo
  2. Leer los datos necesarios de DuckDB (orderbook, ticks recientes)
  3. Calcular todas las features (microestructura + resolución)
  4. Persistir en la tabla features

Por qué este archivo y no llamar a microstructure directamente:
  El feature store centraliza la lógica de "cuándo y cómo calcular".
  El connector solo llama a store.on_tick() — no necesita saber
  qué features existen ni cómo se calculan.
  Si añades una nueva feature, solo tocas este archivo.

Relación con el MATH.md:
  Este archivo implementa el pipeline de §4.6:
    ticks → μ̂_t = w_1·OBI + w_2·News + w_3·OnChain
  Por ahora μ̂_t = OBI (proxy) hasta calibrar w_i en notebooks.
"""

from __future__ import annotations

import logging

from features.microstructure import compute_features_from_db
from normalizer.schema import Market, MarketSnapshot, Tick
from storage.reader import MarketDataReader
from storage.writer import MarketDataWriter

log = logging.getLogger(__name__)


class FeatureStore:
    """
    Calcula y persiste features para un conjunto de mercados activos.

    Uso en producción (llamado por el connector, async):

        store = FeatureStore(reader, writer)

        # Cada vez que llega un tick o snapshot del WebSocket
        await store.on_tick(tick, market)
        await store.on_snapshot(snapshot)

    Uso en scripts/tests (síncrono):

        store = FeatureStore(reader, writer)
        store.compute_and_store(market_id, tau_years)
    """

    def __init__(
        self,
        reader: MarketDataReader,
        writer: MarketDataWriter,
    ) -> None:
        self._reader = reader
        self._writer = writer

    # ------------------------------------------------------------------
    # API principal — llamada por el connector
    # ------------------------------------------------------------------

    def compute_and_store(
        self,
        market_id: str,
        tau_years: float,
        ewma_window: int = 50,
    ) -> bool:
        """
        Calcula todas las features para un mercado y las persiste.

        Por qué tau_years como parámetro externo:
          tau lo calcula el connector desde market.resolution.tau,
          que ya tiene la fecha de resolución. Pasarlo como parámetro
          evita que el store tenga que hacer una query extra a markets.

        Args:
            market_id:   string canónico "venue:raw_id"
            tau_years:   tiempo hasta resolución en años
            ewma_window: número de ticks para EWMA

        Returns:
            True si se calcularon y persistieron features,
            False si no había datos suficientes.
        """
        row = compute_features_from_db(
            market_id=market_id,
            reader=self._reader,
            tau_years=tau_years,
            ewma_window=ewma_window,
        )

        if row is None:
            log.debug("No data for features: %s", market_id)
            return False

        self._writer.write_features_sync([row])
        log.debug(
            "Features stored: %s | obi=%.3f bernoulli_vol=%.4f tau=%.4f",
            market_id,
            row.get("obi", 0),
            row.get("bernoulli_vol") or 0,
            tau_years,
        )
        return True

    def compute_and_store_batch(
        self,
        markets: list[Market],
    ) -> int:
        """
        Calcula y persiste features para múltiples mercados.

        Usado por el connector al arrancar para procesar todos los
        mercados activos de una vez antes de empezar el streaming.

        Args:
            markets: lista de objetos Market del dominio

        Returns:
            Número de mercados con features persistidas exitosamente.
        """
        rows = []
        for market in markets:
            tau = market.resolution.tau
            row = compute_features_from_db(
                market_id=str(market.market_id),
                reader=self._reader,
                tau_years=tau,
            )
            if row is not None:
                rows.append(row)

        if rows:
            self._writer.write_features_sync(rows)
            log.info("Batch features stored: %d/%d markets", len(rows), len(markets))

        return len(rows)

    # ------------------------------------------------------------------
    # Hooks para el connector — llamados en cada evento del WebSocket
    # ------------------------------------------------------------------

    def on_tick(self, tick: Tick, market: Market) -> bool:
        """
        Hook llamado por el connector cada vez que llega un tick.

        Por qué recalcular features en cada tick:
          OBI y EWMA cambian con cada tick. El GLFT necesita features
          actualizadas para calcular quotes correctos. La latencia
          de cálculo es O(N) sobre ewma_window ticks — microsegundos.

        Args:
            tick:   tick nuevo que acaba de llegar del WebSocket
            market: metadatos del mercado (para obtener tau)

        Returns:
            True si features calculadas y persistidas, False si no hay datos.
        """
        return self.compute_and_store(
            market_id=str(market.market_id),
            tau_years=market.resolution.tau,
        )

    def on_snapshot(self, snapshot: MarketSnapshot) -> bool:
        """
        Hook llamado por el connector cuando llega un snapshot completo
        (market + orderbook + tick).

        Por qué snapshot y no solo tick:
          El snapshot incluye el orderbook — que es necesario para
          calcular OBI con profundidad real en lugar del proxy de flujo.
          Cuando hay snapshot disponible, las features son más precisas.

        Args:
            snapshot: MarketSnapshot completo del connector

        Returns:
            True si features calculadas y persistidas.
        """
        if snapshot.last_tick is None:
            return False

        return self.compute_and_store(
            market_id=str(snapshot.market.market_id),
            tau_years=snapshot.market.resolution.tau,
        )

    # ------------------------------------------------------------------
    # Consulta de features recientes — para el execution engine
    # ------------------------------------------------------------------

    def latest(self, market_id: str) -> dict | None:
        """
        Devuelve las features más recientes de un mercado como dict.

        Usado por el execution engine antes de calcular quotes —
        necesita OBI, bernoulli_vol y tau_years en formato nativo
        sin pasar por pandas.

        Returns:
            Dict con las features más recientes, None si no hay datos.
        """
        df = self._reader.latest_features(market_id)
        if df.empty:
            return None

        row = df.iloc[0]
        return {
            "market_id": market_id,
            "timestamp": row["timestamp"],
            "obi": float(row["obi"]) if row["obi"] is not None else 0.0,
            "quoted_spread": float(row["quoted_spread"])
            if row["quoted_spread"] is not None
            else None,
            "relative_spread": float(row["relative_spread"])
            if row["relative_spread"] is not None
            else None,
            "bernoulli_vol": float(row["bernoulli_vol"])
            if row["bernoulli_vol"] is not None
            else None,
            "ewma_vol": float(row["ewma_vol"]) if row["ewma_vol"] is not None else 0.0,
            "tau_years": float(row["tau_years"]) if row["tau_years"] is not None else 0.0,
            "mu_hat": float(row["mu_hat"]) if row["mu_hat"] is not None else 0.0,
        }
