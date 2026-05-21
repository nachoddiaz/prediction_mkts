"""
execution/paper/engine.py
─────────────────────────
Motor de emparejamiento y ejecución simulada (paper trading) basado en ticks y snapshots.
"""

from __future__ import annotations

import logging

from execution.order import Order, OrderAction
from execution.paper.account import PaperAccount
from normalizer.schema import MarketSnapshot, Side, Size, Tick, TickType

log = logging.getLogger(__name__)


class PaperExecutionEngine:
    """
    Simula la ejecución de órdenes limitadas de compra/venta contra eventos reales.

    Heurísticas de ejecución:
      1. Cruce del libro (Garantizado):
         - Si nuestra orden de COMPRA está a un precio >= yes_ask del mercado,
           se ejecuta (alguien vende a nuestro precio o menos).
         - Si nuestra orden de VENTA está a un precio <= yes_bid del mercado,
           se ejecuta (alguien compra a nuestro precio o más).

      2. Ejecución pasiva por Trades (Probabilístico/Volumen):
         - Si recibimos un TRADE en el mercado a un precio <= que nuestra orden de COMPRA,
           simulamos un fill.
         - Si recibimos un TRADE en el mercado a un precio >= que nuestra orden de VENTA,
           simulamos un fill.
         - El tamaño del fill se limita al volumen del trade si está disponible y es positivo.
    """

    def __init__(self, account: PaperAccount) -> None:
        self._account = account

    @property
    def account(self) -> PaperAccount:
        return self._account

    def process_tick(self, tick: Tick) -> list[Order]:
        """
        Procesa un Tick individual de mercado y simula ejecuciones contra él.

        Returns:
            Lista de órdenes que sufrieron algún fill en esta llamada.
        """
        active_orders = self._account.get_active_orders(tick.market_id)
        filled_orders: list[Order] = []

        for order in active_orders:
            # Solo ejecutamos órdenes del outcome YES por simplicidad y alineación con quoters
            if order.outcome != Side.YES:
                continue

            fill_size = Size(0.0)
            fill_price = order.price

            if order.action == OrderAction.BUY:
                # 1. Cruce directo con el ask del mercado
                if tick.yes_ask <= order.price:
                    fill_size = order.remaining_size
                # 2. Match pasivo con trades del mercado
                elif tick.tick_type == TickType.TRADE and tick.yes_bid <= order.price:
                    # Si el tick de trade viene con volumen válido, tomamos como máximo ese volumen
                    if tick.volume > 0:
                        fill_size = Size(min(order.remaining_size, tick.volume))
                    else:
                        fill_size = order.remaining_size

            elif order.action == OrderAction.SELL:
                # 1. Cruce directo con el bid del mercado
                if tick.yes_bid >= order.price:
                    fill_size = order.remaining_size
                # 2. Match pasivo con trades del mercado
                elif tick.tick_type == TickType.TRADE and tick.yes_ask >= order.price:
                    if tick.volume > 0:
                        fill_size = Size(min(order.remaining_size, tick.volume))
                    else:
                        fill_size = order.remaining_size

            # Si hay ejecución, actualizar cuenta
            if fill_size > 0:
                updated_order = self._account.fill_order(order.order_id, fill_size, fill_price)
                if updated_order:
                    filled_orders.append(updated_order)

        return filled_orders

    def process_snapshot(self, snapshot: MarketSnapshot) -> list[Order]:
        """
        Procesa un snapshot completo de mercado (libro de órdenes).

        Actualiza el estado de las órdenes usando la mejor oferta/demanda del libro.
        """
        filled_orders: list[Order] = []

        # Procesar primero el tick asociado si está presente
        if snapshot.last_tick:
            filled_orders.extend(self.process_tick(snapshot.last_tick))

        if not snapshot.orderbook:
            return filled_orders

        # Cruce con la parte superior del orderbook
        best_bid = snapshot.orderbook.best_bid
        best_ask = snapshot.orderbook.best_ask
        market_id = snapshot.market.market_id

        active_orders = self._account.get_active_orders(market_id)

        for order in active_orders:
            if order.outcome != Side.YES:
                continue

            fill_size = Size(0.0)

            if order.action == OrderAction.BUY and best_ask is not None:
                if best_ask <= order.price:
                    fill_size = order.remaining_size

            elif order.action == OrderAction.SELL and best_bid is not None:
                if best_bid >= order.price:
                    fill_size = order.remaining_size

            if fill_size > 0:
                updated_order = self._account.fill_order(order.order_id, fill_size, order.price)
                if updated_order:
                    filled_orders.append(updated_order)

        return filled_orders
