"""
execution/risk/limits.py
─────────────────────────
Verificaciones de límites de riesgo previas al envío de órdenes (pre-trade risk).
"""

from __future__ import annotations

import logging

from execution.order import Order, OrderAction

log = logging.getLogger(__name__)


class RiskLimitsChecker:
    """
    Realiza chequeos de límites individuales de órdenes antes de su inserción.

    Límites validados:
      1. Límite de Inventario Máximo Efectivo:
         Evita enviar órdenes que lleven la posición neta más allá de Q_max_effective.
         Q_max_effective puede ser dinámico si estamos en régimen near-resolution.

      2. Límites de precio razonable:
         Los precios deben residir estrictamente en [0.0001, 0.9999] para evitar
         errores de cálculo de logit o precios inválidos de las venues.
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
        Valida que una orden cumpla con los límites de riesgo.

        Args:
            order:            La orden a validar.
            current_position: Posición firmada de YES actual para este mercado.
            q_max_effective:  Límite de inventario efectivo configurado/calculado
                              para este instante.

        Returns:
            Tuple[bool, str] -> (True, "") si pasa, (False, razón) si se rechaza.
        """
        # 1. Validar precio en rango seguro para evitar divisiones por cero en logit
        if not (0.0001 <= float(order.price) <= 0.9999):
            reason = f"Price {order.price} is outside safe limits [0.0001, 0.9999]"
            log.warning(f"pre_trade_risk_rejected: order_id={order.order_id}, reason={reason}")
            return False, reason

        # 2. Validar límite de inventario firmado
        position_change = (
            float(order.size) if order.action == OrderAction.BUY else -float(order.size)
        )
        projected_position = current_position + position_change

        if abs(projected_position) > q_max_effective:
            # Permitir si es una orden que reduce el riesgo
            # (reduce el valor absoluto de la posición)
            if abs(projected_position) >= abs(current_position):
                reason = (
                    f"Projected position {projected_position:+.1f} "
                    f"exceeds effective limit Q_max_eff={q_max_effective:.1f} "
                    f"(current_pos={current_position:+.1f}, order_size={float(order.size)})"
                )
                log.warning(f"pre_trade_risk_rejected: order_id={order.order_id}, reason={reason}")
                return False, reason

        return True, ""
