"""
execution/paper/engine.py
─────────────────────────
Matching and simulated execution engine (paper trading), driven by ticks and
snapshots.
"""

from __future__ import annotations

import logging

from execution.order import Order, OrderAction
from execution.paper.account import PaperAccount
from normalizer.schema import MarketSnapshot, Side, Size, Tick, TickType

log = logging.getLogger(__name__)


class PaperExecutionEngine:
    """
    Simulates execution of resting limit orders against real market events.

    Execution heuristics:
      1. Book crossing (guaranteed):
         - If our BUY order sits at a price >= the market's yes_ask, it fills
           (someone is selling at our price or better).
         - If our SELL order sits at a price <= the market's yes_bid, it fills
           (someone is buying at our price or better).

      2. Passive execution against trades (probabilistic / volume-based):
         - A market TRADE at a price <= our BUY order simulates a fill.
           simulamos un fill.
         - A market TRADE at a price >= our SELL order simulates a fill.
           simulamos un fill.
         - Fill size is capped by the trade volume when it is available and positive.
    """

    def __init__(self, account: PaperAccount) -> None:
        self._account = account

    @property
    def account(self) -> PaperAccount:
        return self._account

    def process_tick(self, tick: Tick) -> list[Order]:
        """
        Process one market Tick and simulate executions against it.

        Returns:
            List of orders that received a fill on this call.
        """
        active_orders = self._account.get_active_orders(tick.market_id)
        filled_orders: list[Order] = []

        for order in active_orders:
            # Only YES orders are executed, for simplicity and to match the quoters
            if order.outcome != Side.YES:
                continue

            fill_size = Size(0.0)
            fill_price = order.price

            if order.action == OrderAction.BUY:
                # 1. Direct crossing against the market ask
                if tick.yes_ask <= order.price:
                    fill_size = order.remaining_size
                # 2. Passive match against market trades
                elif tick.tick_type == TickType.TRADE and tick.yes_bid <= order.price:
                    # Where the trade tick carries a valid volume, cap the fill at it
                    if tick.volume > 0:
                        fill_size = Size(min(order.remaining_size, tick.volume))
                    else:
                        fill_size = order.remaining_size

            elif order.action == OrderAction.SELL:
                # 1. Direct crossing against the market bid
                if tick.yes_bid >= order.price:
                    fill_size = order.remaining_size
                # 2. Passive match against market trades
                elif tick.tick_type == TickType.TRADE and tick.yes_ask >= order.price:
                    if tick.volume > 0:
                        fill_size = Size(min(order.remaining_size, tick.volume))
                    else:
                        fill_size = order.remaining_size

            # On a fill, update the account
            if fill_size > 0:
                updated_order = self._account.fill_order(order.order_id, fill_size, fill_price)
                if updated_order:
                    filled_orders.append(updated_order)

        return filled_orders

    def process_snapshot(self, snapshot: MarketSnapshot) -> list[Order]:
        """
        Process a complete market snapshot (order book).

        Updates order state using the book's best bid and offer.
        """
        filled_orders: list[Order] = []

        # Process the attached tick first, when present
        if snapshot.last_tick:
            filled_orders.extend(self.process_tick(snapshot.last_tick))

        if not snapshot.orderbook:
            return filled_orders

        # Crossing against the top of the book
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
