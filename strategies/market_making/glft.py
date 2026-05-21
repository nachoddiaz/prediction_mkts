"""
strategies/market_making/glft.py
─────────────────────────────────
Modelo GLFT en espacio logit — aproximación AS (§4.1 MATH.md v2.1).

Opera enteramente en X = logit(p) ∈ ℝ donde los modelos clásicos
(AS, GLFT, CJ) aplican con sus pruebas intactas.

Los quotes se calculan en X y se mapean a precio via σ(X) = 1/(1+e^{-X})
al final — así el spread se comprime automáticamente cerca de los bordes
sin ningún hack adicional.

Por qué aproximación AS y no GLFT exacto (ODE):
  El GLFT exacto requiere integrar numéricamente 2Q+1 ODEs acopladas
  (una por nivel de inventario) hacia atrás desde T. Para Q=10 son 21
  ecuaciones que tardan ~ms por quote. La aproximación AS en logit es
  suficiente para Fase 1 — cuando tengamos datos reales para validar
  la diferencia con el ODE exacto, lo añadiremos como glft_exact().

Relación con MATH.md v2.1:
  §4.1 → reservation log-odds y spread óptimo
  §4.2 → estructura del GLFT exacto (no implementado aquí)
  §4.3 → calibración κ_p vs κ_x y conversión
  (2.3) → tick floor
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

from features.resolution import TAU_1H, NearResolutionRegime
from normalizer.schema import MarketId

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Tick mínimo cotizable en Kalshi y Polymarket
TICK = 0.01

# Logit de los extremos cotizables
# σ(LOGIT_MIN) ≈ 0.01,  σ(LOGIT_MAX) ≈ 0.99
LOGIT_MIN = math.log(TICK / (1 - TICK))  # ≈ -4.595
LOGIT_MAX = math.log((1 - TICK) / TICK)  # ≈  4.595


# ---------------------------------------------------------------------------
# Primitivas logit
# ---------------------------------------------------------------------------


def logit(p: float) -> float:
    """X = logit(p) = ln(p/(1-p)). Clampea p a (ε, 1-ε)."""
    p = max(1e-9, min(1 - 1e-9, p))
    return math.log(p / (1.0 - p))


def sigma(x: float) -> float:
    """p = σ(X) = 1/(1+e^{-X}). Logit inversa."""
    return 1.0 / (1.0 + math.exp(-x))


def sigma_prime(x: float) -> float:
    """σ'(X) = σ(X)·(1-σ(X)) = p(1-p). Jacobiano de la transformación."""
    p = sigma(x)
    return p * (1.0 - p)


# ---------------------------------------------------------------------------
# Quote dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Quote:
    """
    Resultado completo de un ciclo de quoting.

    Contiene inputs, cálculos intermedios y outputs — todo visible
    para debug, backtesting y logging. El execution engine solo usa
    bid_p, ask_p e is_valid.

    Por qué frozen:
      Un Quote es una decisión tomada en un momento t. No debe
      mutarse después de creado — si las condiciones cambian se
      crea un nuevo Quote.
    """

    # --- Identidad ---
    market_id: MarketId
    timestamp: datetime
    model: str  # "glft" | "cartea_jaimungal"

    # --- Inputs del modelo ---
    mid_price_p: float  # p_t en espacio precio ∈ (0,1)
    mid_price_X: float  # X_t = logit(p_t) ∈ ℝ
    inventory: float  # q_t — posición actual del MM
    tau_years: float  # τ = T - t en años
    regime: NearResolutionRegime

    # --- Parámetros usados ---
    gamma_I: float  # γ_I — CARA inventory risk aversion
    kappa_x: float  # κ_x — fill curve decay en logit
    belief_vol: float  # σ_b — volatilidad instantánea de logit(p)

    # --- Cálculos intermedios ---
    sigma_bar_sq: float  # σ̄²_b·τ — varianza integrada
    reservation_X: float  # r̃_X = X - q·γ_I·σ̄²_b·τ
    half_spread_X: float  # δ*/2 en logit
    signal_skew: float  # componente de señal (0.0 en GLFT puro)

    # --- Quotes en logit ---
    bid_X: float  # r̃_X - δ*/2
    ask_X: float  # r̃_X + δ*/2

    # --- Quotes en precio — lo que se envía al mercado ---
    bid_p: float  # σ(bid_X), ajustado al tick si es necesario
    ask_p: float  # σ(ask_X), ajustado al tick si es necesario

    # --- Validez ---
    is_valid: bool  # False → el execution engine no cotiza
    invalid_reason: str  # "" si is_valid=True

    @property
    def spread_p(self) -> float:
        """Spread en espacio precio."""
        return self.ask_p - self.bid_p

    @property
    def spread_X(self) -> float:
        """Spread en espacio logit."""
        return self.ask_X - self.bid_X

    @property
    def mid_quoted_p(self) -> float:
        """Mid de los quotes cotizados en espacio precio."""
        return (self.bid_p + self.ask_p) / 2.0

    def log_line(self) -> str:
        """Una línea de log con los campos más relevantes."""
        return (
            f"model={self.model} "
            f"p={self.mid_price_p:.4f} q={self.inventory:+.0f} "
            f"τ={self.tau_years*365:.1f}d "
            f"σ_b={self.belief_vol:.4f} "
            f"r̃_p={sigma(self.reservation_X):.4f} "
            f"bid={self.bid_p:.4f} ask={self.ask_p:.4f} "
            f"spread={self.spread_p:.4f} "
            f"valid={self.is_valid}"
        )


# ---------------------------------------------------------------------------
# GLFTQuoter
# ---------------------------------------------------------------------------


class GLFTQuoter:
    """
    Market maker GLFT en espacio logit — aproximación AS (§4.1).

    Fórmulas (MATH.md v2.1 §4.1):

      Varianza integrada:
        σ̄²_b = σ²_b (constante en esta aproximación)

      Reservation log-odds:
        r̃_X(t,q) = X_t - q·γ_I·σ̄²_b·τ            (2.1)

      Optimal half-spread en logit:
        δ*/2 = γ_I·σ̄²_b·τ/2 + (1/κ_x)·ln(1+γ_I/κ_x)  (2.2)

      Quotes en precio:
        bid_p = σ(r̃_X - δ*/2)                       (2.3)
        ask_p = σ(r̃_X + δ*/2)

      Tick floor (§4.1 eq 2.3):
        si bid_p < TICK o ask_p > 1-TICK:
          τ < TAU_1H → is_valid=False
          τ ≥ TAU_1H → ajustar spread a TICK alrededor de σ(r̃_X)

    Uso:
        quoter = GLFTQuoter(gamma_I=0.1, kappa_x=0.8)
        quote  = quoter.quote(
            market_id=mid,
            mid_p=0.45,
            inventory=3,
            tau_years=0.19,
            belief_vol=0.12,
            regime=NearResolutionRegime.NORMAL,
        )
        if quote.is_valid:
            execution_engine.submit(quote.bid_p, quote.ask_p)
    """

    def __init__(
        self,
        gamma_I: float,
        kappa_x: float,
    ) -> None:
        """
        Args:
            gamma_I: CARA inventory risk aversion (1/$).
                     Leer desde config.kalshi_config.risk.gamma.
                     Mayor γ_I → spreads más anchos, menos inventario.

            kappa_x: fill curve decay en espacio logit (adimensional).
                     Calibrado desde GLFTCalibrator — campo kappa_x del resultado.
                     Mayor κ_x → fills más sensibles al spread.
        """
        if gamma_I <= 0:
            raise ValueError(f"gamma_I must be positive, got {gamma_I}")
        if kappa_x <= 0:
            raise ValueError(f"kappa_x must be positive, got {kappa_x}")

        self.gamma_I = gamma_I
        self.kappa_x = kappa_x

    def quote(
        self,
        market_id: MarketId,
        mid_p: float,
        inventory: float,
        tau_years: float,
        belief_vol: float,
        regime: NearResolutionRegime,
        timestamp: datetime | None = None,
    ) -> Quote:
        """
        Calcula los quotes óptimos para el estado actual del mercado.

        Args:
            market_id:  identificador canónico del mercado
            mid_p:      mid-price actual ∈ (0, 1)
            inventory:  posición actual del MM en contratos (signed)
            tau_years:  tiempo hasta resolución en años
            belief_vol: σ_b — volatilidad instantánea de logit(p),
                        calculada por belief_vol_from_ticks() en microstructure.py
            regime:     NearResolutionRegime desde resolution.py

        Returns:
            Quote con todos los campos. El execution engine solo necesita
            is_valid, bid_p y ask_p.
        """
        ts = timestamp or datetime.now(tz=UTC)

        # --- Halt inmediato si el régimen lo requiere ---
        if regime in (NearResolutionRegime.HALT, NearResolutionRegime.RESOLVED):
            return self._invalid_quote(
                market_id=market_id,
                timestamp=ts,
                mid_p=mid_p,
                inventory=inventory,
                tau_years=tau_years,
                belief_vol=belief_vol,
                regime=regime,
                reason=f"regime={regime.value}",
            )

        # --- Estado en espacio logit ---
        X_t = logit(mid_p)

        # --- Varianza integrada σ̄²_b·τ ---
        # En esta aproximación σ_b es constante → σ̄²_b = σ²_b
        # En una versión futura: integrar σ²_b(u) sobre [t,T]
        sigma_bar_sq = belief_vol**2 * tau_years

        # --- Reservation log-odds (2.1) ---
        # r̃_X = X_t - q·γ_I·σ̄²_b·τ
        # El skew de inventario desplaza el centro de los quotes:
        #   q > 0 (long) → r̃_X < X_t → cotizar más bajo para vender
        #   q < 0 (short) → r̃_X > X_t → cotizar más alto para comprar
        reservation_X = X_t - inventory * self.gamma_I * sigma_bar_sq

        # --- Optimal half-spread en logit (2.2) ---
        # δ*/2 = γ_I·σ̄²_b·τ/2 + (1/κ_x)·ln(1 + γ_I/κ_x)
        #
        # Primer término: compensación por riesgo de inventario
        # Segundo término: "microstructure rent" — cota inferior del spread
        #   independiente del inventario y del tiempo (asintóticamente)
        inventory_term = self.gamma_I * sigma_bar_sq / 2.0
        rent_term = (1.0 / self.kappa_x) * math.log(1.0 + self.gamma_I / self.kappa_x)
        half_spread_X = inventory_term + rent_term

        # --- Quotes en logit ---
        bid_X = reservation_X - half_spread_X
        ask_X = reservation_X + half_spread_X

        # --- Mapear a espacio precio via σ(X) ---
        bid_p_raw = sigma(bid_X)
        ask_p_raw = sigma(ask_X)

        # --- Tick floor (§4.1 eq 2.3) ---
        bid_p, ask_p, is_valid, reason = self._apply_tick_floor(
            bid_p=bid_p_raw,
            ask_p=ask_p_raw,
            reservation_X=reservation_X,
            tau_years=tau_years,
            regime=regime,
        )

        return Quote(
            market_id=market_id,
            timestamp=ts,
            model="glft",
            mid_price_p=mid_p,
            mid_price_X=X_t,
            inventory=inventory,
            tau_years=tau_years,
            regime=regime,
            gamma_I=self.gamma_I,
            kappa_x=self.kappa_x,
            belief_vol=belief_vol,
            sigma_bar_sq=sigma_bar_sq,
            reservation_X=reservation_X,
            half_spread_X=half_spread_X,
            signal_skew=0.0,  # GLFT puro — sin señal direccional
            bid_X=bid_X,
            ask_X=ask_X,
            bid_p=bid_p,
            ask_p=ask_p,
            is_valid=is_valid,
            invalid_reason=reason,
        )

    def _apply_tick_floor(
        self,
        bid_p: float,
        ask_p: float,
        reservation_X: float,
        tau_years: float,
        regime: NearResolutionRegime,
    ) -> tuple[float, float, bool, str]:
        """
        Aplica el tick floor de §4.1 (ecuación 2.3).

        Lógica:
          Si bid_p ≥ TICK y ask_p ≤ 1-TICK → quotes válidos, sin ajuste
          Si alguno viola el tick:
            τ < TAU_1H → near-resolution, is_valid=False
            τ ≥ TAU_1H → ajustar spread al tick mínimo alrededor de σ(r̃_X)

        Returns:
            (bid_p, ask_p, is_valid, reason)
        """
        # Verificar si algún quote viola el tick
        bid_below = bid_p < TICK
        ask_above = ask_p > 1.0 - TICK

        if not bid_below and not ask_above:
            # Caso normal — quotes válidos sin ajuste
            return bid_p, ask_p, True, ""

        # Violación detectada — decidir según τ
        if tau_years < TAU_1H:
            # Near-resolution — no cotizar
            return (
                bid_p,
                ask_p,
                False,
                (
                    f"tick_floor_near_resolution: "
                    f"bid={bid_p:.4f} ask={ask_p:.4f} tau_hours={tau_years*8760:.1f}"
                ),
            )

        # Lejos de vencimiento — ajustar spread al tick mínimo
        # Mantener el reservation_p como centro y poner spread = TICK
        reservation_p = sigma(reservation_X)
        bid_p_adj = max(TICK, reservation_p - TICK / 2.0)
        ask_p_adj = min(1.0 - TICK, reservation_p + TICK / 2.0)

        # Verificar que el ajuste es coherente
        if bid_p_adj >= ask_p_adj:
            bid_p_adj = TICK
            ask_p_adj = 2 * TICK

        return bid_p_adj, ask_p_adj, True, "tick_floor_adjusted"

    def _invalid_quote(
        self,
        market_id: MarketId,
        timestamp: datetime,
        mid_p: float,
        inventory: float,
        tau_years: float,
        belief_vol: float,
        regime: NearResolutionRegime,
        reason: str,
    ) -> Quote:
        """Construye un Quote inválido — para casos de halt o near-resolution."""
        X_t = logit(mid_p)
        return Quote(
            market_id=market_id,
            timestamp=timestamp,
            model="glft",
            mid_price_p=mid_p,
            mid_price_X=X_t,
            inventory=inventory,
            tau_years=tau_years,
            regime=regime,
            gamma_I=self.gamma_I,
            kappa_x=self.kappa_x,
            belief_vol=belief_vol,
            sigma_bar_sq=0.0,
            reservation_X=X_t,
            half_spread_X=0.0,
            signal_skew=0.0,
            bid_X=X_t,
            ask_X=X_t,
            bid_p=mid_p,
            ask_p=mid_p,
            is_valid=False,
            invalid_reason=reason,
        )
