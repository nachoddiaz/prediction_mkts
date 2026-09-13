"""
execution/router.py
───────────────────
Order router and order-state manager (OrderRouter).
"""

from __future__ import annotations

import logging

from execution.order import Order, OrderAction, OrderStatus
from execution.paper.engine import PaperExecutionEngine
from execution.risk.circuit_breaker import CircuitBreaker
from execution.risk.limits import RiskLimitsChecker
from execution.risk.monitor import RiskMonitor
from features.resolution import compute_resolution_features
from normalizer.schema import Market, MarketId, Price, Side, Size
from strategies.market_making.glft import Quote

log = logging.getLogger(__name__)


class OrderRouter:
    """
    Orchestrates order placement from the Quotes produced by the quoters.


    Responsabilidades:
      1. Receive a `Quote` from the strategy (GLFT or CJ).
      2. Evaluate the CircuitBreaker (near-resolution regime or daily loss).
      3. Compute dynamic inventory limits (`Q_max_effective`) when a `Market`
         is supplied.
      4. Validate the proposed buy/sell orders through `RiskLimitsChecker`.
      5. Send, replace or cancel orders on `PaperExecutionEngine` (or a live engine).
      6. Avoid churning: only cancel and resend when the quoted price or size
         has actually changed.
    """

    def __init__(
        self,
        paper_engine: PaperExecutionEngine,
        risk_limits: RiskLimitsChecker,
        circuit_breaker: CircuitBreaker,
        risk_monitor: RiskMonitor,
        q_max_base: float,
    ) -> None:
        self.paper_engine = paper_engine
        self.risk_limits = risk_limits
        self.circuit_breaker = circuit_breaker
        self.risk_monitor = risk_monitor
        self.q_max_base = q_max_base

        # Keep historical tick mid prices for the monitor's valuation
        self.mid_prices: dict[MarketId, float] = {}

    def on_quote(
        self,
        quote: Quote,
        size: float = 1.0,
        market: Market | None = None,
    ) -> list[str]:
        """
        Process one optimal quote and update the resting orders in the market.

        Args:
            quote:  quote produced by the model.
            size:   default size for the limit orders to place.
            market: optional market metadata, used to compute Q_max_effective
                    from the time to resolution.

        Returns:
            List of affected order IDs (created, cancelled or modified).
        """
        market_id = quote.market_id
        self.mid_prices[market_id] = quote.mid_price_p

        # 1. Read the simulated account's state
        account = self.paper_engine.account
        current_position = account.get_position(market_id)
        cash = account.cash_balance
        positions = account.positions

        # 2. Update the monitor and read the daily loss
        daily_loss = self.risk_monitor.update(cash, positions, self.mid_prices)

        # 3. Evaluate the global circuit breaker / regime
        breaker_tripped = self.circuit_breaker.check(quote.regime, daily_loss)

        # 4. Resolve dynamic near-resolution limits where metadata is available
        q_max_effective = self.q_max_base
        should_halt_side = False

        if market:
            rf = compute_resolution_features(market.resolution.resolution_date, now=quote.timestamp)
            q_max_effective = self.q_max_base * rf.q_max_fraction
            should_halt_side = rf.should_halt_side

        # Decide whether quoting this market is globally permitted
        is_quoting_allowed = quote.is_valid and not breaker_tripped

        affected_order_ids: list[str] = []

        # If quoting is not permitted, or the breaker tripped, cancel everything
        if not is_quoting_allowed:
            active_orders = account.get_active_orders(market_id)
            for o in active_orders:
                if account.cancel_order(o.order_id):
                    affected_order_ids.append(o.order_id)
            return affected_order_ids

        # Retrieve the market's resting orders
        active_orders = account.get_active_orders(market_id)
        active_buy: Order | None = next(
            (o for o in active_orders if o.action == OrderAction.BUY), None
        )
        active_sell: Order | None = next(
            (o for o in active_orders if o.action == OrderAction.SELL), None
        )

        # --- BUY SIDE MANAGEMENT ---
        # When should_halt_side is set and the position is long (> 0),
        # stop buying to reduce risk
        halt_buy = should_halt_side and current_position > 0
        target_bid: float | None = quote.bid_p if not halt_buy else None

        if target_bid is not None:
            bid_price = Price(target_bid)
            order_size = Size(size)

            # Temporary order used for the pre-trade risk check
            temp_order = Order(
                order_id="temp_buy",
                market_id=market_id,
                action=OrderAction.BUY,
                outcome=Side.YES,
                price=bid_price,
                size=order_size,
            )
            passed_risk, _ = self.risk_limits.check_order(
                temp_order, current_position, q_max_effective
            )

            if passed_risk:
                # Check whether an active buy order already exists and differs
                if active_buy:
                    if float(active_buy.price) != target_bid or float(active_buy.size) != size:
                        # Cancel the old one and create a new one
                        account.cancel_order(active_buy.order_id)
                        affected_order_ids.append(active_buy.order_id)
                        new_o = account.create_order(
                            market_id, OrderAction.BUY, bid_price, order_size
                        )
                        new_o.status = OrderStatus.ACTIVE  # Confirmada
                        affected_order_ids.append(new_o.order_id)
                else:
                    new_o = account.create_order(market_id, OrderAction.BUY, bid_price, order_size)
                    new_o.status = OrderStatus.ACTIVE  # Confirmada
                    affected_order_ids.append(new_o.order_id)
            else:
                # If it fails risk, cancel any resting order
                if active_buy:
                    account.cancel_order(active_buy.order_id)
                    affected_order_ids.append(active_buy.order_id)
        else:
            # With no quoted bid, cancel any active buy
            if active_buy:
                account.cancel_order(active_buy.order_id)
                affected_order_ids.append(active_buy.order_id)

        # --- SELL SIDE MANAGEMENT ---
        # When should_halt_side is set and the position is short (< 0), stop selling
        halt_sell = should_halt_side and current_position < 0
        target_ask: float | None = quote.ask_p if not halt_sell else None

        if target_ask is not None:
            ask_price = Price(target_ask)
            order_size = Size(size)

            temp_order = Order(
                order_id="temp_sell",
                market_id=market_id,
                action=OrderAction.SELL,
                outcome=Side.YES,
                price=ask_price,
                size=order_size,
            )
            passed_risk, _ = self.risk_limits.check_order(
                temp_order, current_position, q_max_effective
            )

            if passed_risk:
                if active_sell:
                    if float(active_sell.price) != target_ask or float(active_sell.size) != size:
                        account.cancel_order(active_sell.order_id)
                        affected_order_ids.append(active_sell.order_id)
                        new_o = account.create_order(
                            market_id, OrderAction.SELL, ask_price, order_size
                        )
                        new_o.status = OrderStatus.ACTIVE  # Confirmada
                        affected_order_ids.append(new_o.order_id)
                else:
                    new_o = account.create_order(market_id, OrderAction.SELL, ask_price, order_size)
                    new_o.status = OrderStatus.ACTIVE  # Confirmada
                    affected_order_ids.append(new_o.order_id)
            else:
                if active_sell:
                    account.cancel_order(active_sell.order_id)
                    affected_order_ids.append(active_sell.order_id)
        else:
            if active_sell:
                account.cancel_order(active_sell.order_id)
                affected_order_ids.append(active_sell.order_id)

        return affected_order_ids
