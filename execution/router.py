"""
execution/router.py
───────────────────
Enrutador y gestor de órdenes (OrderRouter).
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
    Orquesta la colocación de órdenes a partir de las cotizaciones (Quotes)
    sugeridas por los quoters.

    Responsabilidades:
      1. Recibir un `Quote` de la estrategia (GLFT o CJ).
      2. Evaluar el CircuitBreaker (para régimen de near-resolution o pérdida diaria).
      3. Calcular límites de inventario dinámicos (`Q_max_effective`) si se proporciona el `Market`.
      4. Validar las órdenes propuestas (compra/venta) a través del `RiskLimitsChecker`.
      5. Enviar, reemplazar o cancelar órdenes en el `PaperExecutionEngine` (o motores reales).
      6. Evitar sobre-operar (churning): solo cancela y re-envía órdenes si el
         precio/tamaño cotizado cambia.
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

        # Guardar mid-prices históricos de ticks para valoración del monitor
        self.mid_prices: dict[MarketId, float] = {}

    def on_quote(
        self,
        quote: Quote,
        size: float = 1.0,
        market: Market | None = None,
    ) -> list[str]:
        """
        Procesa una cotización óptima y actualiza el estado de las órdenes en el mercado.

        Args:
            quote:  Cotización generada por el modelo.
            size:   Tamaño por defecto de las órdenes limitadas a colocar.
            market: Opcional. Metadatos del mercado para calcular Q_max_effective
                    según tiempo de resolución.

        Returns:
            Lista de IDs de órdenes afectadas (creadas, canceladas o modificadas).
        """
        market_id = quote.market_id
        self.mid_prices[market_id] = quote.mid_price_p

        # 1. Recuperar estado de la cuenta simulada
        account = self.paper_engine.account
        current_position = account.get_position(market_id)
        cash = account.cash_balance
        positions = account.positions

        # 2. Actualizar el monitor y obtener pérdida diaria
        daily_loss = self.risk_monitor.update(cash, positions, self.mid_prices)

        # 3. Evaluar el circuit breaker global/régimen
        breaker_tripped = self.circuit_breaker.check(quote.regime, daily_loss)

        # 4. Resolver límites dinámicos de near-resolution si disponemos de metadatos
        q_max_effective = self.q_max_base
        should_halt_side = False

        if market:
            rf = compute_resolution_features(market.resolution.resolution_date, now=quote.timestamp)
            q_max_effective = self.q_max_base * rf.q_max_fraction
            should_halt_side = rf.should_halt_side

        # Determinar si el quoting de este mercado está permitido globalmente
        is_quoting_allowed = quote.is_valid and not breaker_tripped

        affected_order_ids: list[str] = []

        # Si el quoting no está permitido o el breaker saltó, cancelamos todo
        if not is_quoting_allowed:
            active_orders = account.get_active_orders(market_id)
            for o in active_orders:
                if account.cancel_order(o.order_id):
                    affected_order_ids.append(o.order_id)
            return affected_order_ids

        # Recuperar órdenes activas del mercado
        active_orders = account.get_active_orders(market_id)
        active_buy: Order | None = next(
            (o for o in active_orders if o.action == OrderAction.BUY), None
        )
        active_sell: Order | None = next(
            (o for o in active_orders if o.action == OrderAction.SELL), None
        )

        # --- GESTIÓN LADO COMPRA (BUY) ---
        # Si should_halt_side es activo y la posición es larga (> 0),
        # detenemos compra para mitigar riesgo
        halt_buy = should_halt_side and current_position > 0
        target_bid: float | None = quote.bid_p if not halt_buy else None

        if target_bid is not None:
            bid_price = Price(target_bid)
            order_size = Size(size)

            # Crear orden temporal para chequeo de límites de riesgo pre-trade
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
                # Comprobar si ya existe orden de compra activa y si es diferente
                if active_buy:
                    if float(active_buy.price) != target_bid or float(active_buy.size) != size:
                        # Cancelar antigua y crear nueva
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
                # Si no pasa el riesgo, cancelamos si existía una activa
                if active_buy:
                    account.cancel_order(active_buy.order_id)
                    affected_order_ids.append(active_buy.order_id)
        else:
            # Si no hay bid cotizado, cancelamos cualquier compra activa
            if active_buy:
                account.cancel_order(active_buy.order_id)
                affected_order_ids.append(active_buy.order_id)

        # --- GESTIÓN LADO VENTA (SELL) ---
        # Si should_halt_side es activo y la posición es corta (< 0), detenemos venta
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
