"""
execution/risk/limits.py
─────────────────────────
Pre-trade risk limit checks, applied before an order is sent.
"""

from __future__ import annotations

import logging

from execution.order import Order, OrderAction

log = logging.getLogger(__name__)


class RiskLimitsChecker:
    """
    Per-order limit checks performed before insertion.

    Limits validated:
      1. Effective maximum inventory limit:
         Prevents orders that would push the net position beyond
         Q_max_effective, which is dynamic under near-resolution regimes.

      2. Sane price bounds:
         Prices must lie strictly within [0.0001, 0.9999], to avoid logit
         computation errors and prices the venues would reject.
    """

    def __init__(self, max_position_loss: float = 999999.0) -> None:
        self.max_position_loss = max_position_loss

    def check_order(
        self,
        order: Order,
        current_position: float,
        q_max_effective: float,
    ) -> tuple[bool, str]:
        """
        Validate that an order satisfies the risk limits.

        Args:
            order:            the order to validate.
            current_position: current signed YES position in this market.
            q_max_effective:  the configured or computed effective inventory
                              cap at this instant.

        Returns:
            Tuple[bool, str] -> (True, "") if it passes, (False, reason) if rejected.
        """
        # 1. Price within a safe range, avoiding division by zero in logit
        if not (0.0001 <= float(order.price) <= 0.9999):
            reason = f"Price {order.price} is outside safe limits [0.0001, 0.9999]"
            log.warning(f"pre_trade_risk_rejected: order_id={order.order_id}, reason={reason}")
            return False, reason

        # 2. Validate the signed inventory limit
        position_change = (
            float(order.size) if order.action == OrderAction.BUY else -float(order.size)
        )
        projected_position = current_position + position_change

        if abs(projected_position) > q_max_effective:
            # Allow it when the order reduces risk
            # (it reduces the absolute value of the position)
            if abs(projected_position) >= abs(current_position):
                reason = (
                    f"Projected position {projected_position:+.1f} "
                    f"exceeds effective limit Q_max_eff={q_max_effective:.1f} "
                    f"(current_pos={current_position:+.1f}, order_size={float(order.size)})"
                )
                log.warning(f"pre_trade_risk_rejected: order_id={order.order_id}, reason={reason}")
                return False, reason

        return True, ""
