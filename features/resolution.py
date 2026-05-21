"""
features/resolution.py
───────────────────────
Cálculo de tau y flags de near-resolution.

Por qué este archivo separado de microstructure.py:
  tau es un input de casi todas las fórmulas del MATH.md — GLFT,
  Cartea-Jaimungal, Bernoulli vol, Kelly. Es tan fundamental que
  merece su propio módulo, separado de las señales de microestructura
  que dependen del orderbook.

  Además, los near-resolution flags son reglas de negocio que
  afectan al execution engine (circuit breakers), no son features
  de trading. Tenerlos aquí hace explícita esa distinción.

Relación con el MATH.md:
  - tau              → τ = T - t en años, aparece en todas las fórmulas
  - σ_B(p, τ)        → usa tau directamente
  - reservation price → p̃ = p - q·γ·p(1-p) [τ se cancela con σ_B]
  - signal skew      → φ₁(t) = (ρση/φ)(1 - e^{-φτ})
  - near-resolution  → §6.4: reglas de halt quoting
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

# ---------------------------------------------------------------------------
# Umbrales de near-resolution — del §6.4 del MATH.md
#
# Por qué estos valores concretos:
#   24h: el riesgo de inventario empieza a ser significativo.
#        γ_effective = 2γ y Q_max = Q/2.
#   1h:  la vol de Bernoulli empieza a divergir rápidamente.
#        Solo se admite 1 contrato de inventario máximo.
#   5min: near-resolution extremo. El libro está dominado por
#         informed traders (α → 1 en Glosten-Milgrom).
#         Halt total de quoting.
# ---------------------------------------------------------------------------
TAU_24H = 24.0 / (365.25 * 24)  # 24 horas en años
TAU_1H = 1.0 / (365.25 * 24)  # 1 hora en años
TAU_5MIN = 5.0 / (365.25 * 24 * 60)  # 5 minutos en años


class NearResolutionRegime(str, Enum):
    """
    Régimen de near-resolution según §6.4 del MATH.md.

    NORMAL      → τ ≥ 24h. Sistema opera con parámetros normales.
    WARNING     → 1h ≤ τ < 24h. Reducir inventario máximo, doblar γ.
    CRITICAL    → 5min ≤ τ < 1h. Inventario máximo = 1, halt un lado.
    HALT        → τ < 5min. Halt total de quoting.
    RESOLVED    → τ ≤ 0. El mercado ya ha resuelto.
    """

    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    HALT = "halt"
    RESOLVED = "resolved"


@dataclass(frozen=True)
class ResolutionFeatures:
    """
    Todas las features relacionadas con el tiempo hasta resolución.

    Por qué un dataclass frozen:
      Igual que los objetos del dominio — inmutable una vez calculado.
      Si necesitas nuevas features, crea un nuevo objeto.

    Campos:
      tau_years          → τ en años — input directo a todas las fórmulas
      tau_days           → τ en días — para logging y dashboard
      tau_hours          → τ en horas — para near-resolution decisions
      tau_minutes        → τ en minutos — para halt decisions
      regime             → NearResolutionRegime según §6.4
      gamma_multiplier   → factor por el que multiplicar γ en el modelo
      q_max_fraction     → fracción de Q_max permitida (1.0 = normal)
      should_halt        → True si hay que detener todo quoting
      should_halt_side   → True si hay que detener quoting en un lado
    """

    tau_years: float
    tau_days: float
    tau_hours: float
    tau_minutes: float
    regime: NearResolutionRegime
    gamma_multiplier: float
    q_max_fraction: float
    should_halt: bool
    should_halt_side: bool


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------


def compute_resolution_features(
    resolution_date: datetime,
    now: datetime | None = None,
) -> ResolutionFeatures:
    """
    Calcula todas las features de resolución dado la fecha de cierre.

    Por qué recibir now como parámetro opcional:
      En tests necesitamos controlar el tiempo para verificar que
      los flags se activan en los umbrales correctos. Con now=None
      usa datetime.now(UTC) en producción. Con now=<datetime> en tests
      podemos simular cualquier momento sin mocks.

    Args:
        resolution_date: fecha y hora de resolución del contrato (UTC)
        now:             momento actual. None = datetime.now(UTC)

    Returns:
        ResolutionFeatures con tau y todos los flags calculados.
    """
    if now is None:
        now = datetime.now(tz=UTC)

    # Asegurar que ambos datetimes son UTC-aware
    if resolution_date.tzinfo is None:
        resolution_date = resolution_date.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    # --- Calcular tau en distintas unidades ---
    delta_seconds = (resolution_date - now).total_seconds()

    # Si el mercado ya resolvió, tau es 0 en todas las unidades
    if delta_seconds <= 0:
        return ResolutionFeatures(
            tau_years=0.0,
            tau_days=0.0,
            tau_hours=0.0,
            tau_minutes=0.0,
            regime=NearResolutionRegime.RESOLVED,
            gamma_multiplier=1.0,  # sin operaciones activas
            q_max_fraction=0.0,  # no abrir nuevas posiciones
            should_halt=True,
            should_halt_side=True,
        )

    tau_years = delta_seconds / (365.25 * 24 * 3600)
    tau_days = delta_seconds / (24 * 3600)
    tau_hours = delta_seconds / 3600
    tau_minutes = delta_seconds / 60

    # --- Determinar régimen según §6.4 del MATH.md ---
    # Parámetros para q_max exponencial (§8.3 MATH.md v2.1)
    # q_max = Q_0 * exp(-r * psi(tau)) donde psi(tau) = exp(-tau/tau_star)
    # r: probabilidad de reversal del oracle (típico 0.001-0.01)
    # tau_star: escala temporal característica (típico 24h)
    r_oracle = 0.001  # TODO: mover a config
    tau_star = TAU_24H  # 24 horas
    psi_tau = math.exp(-tau_years / tau_star)
    q_max_fraction_exp = math.exp(-r_oracle * psi_tau)

    if tau_years < TAU_5MIN:
        # τ < 5min: halt total
        # Informed traders dominan el libro (α → 1 en Glosten-Milgrom)
        # σ_b diverge — ningún spread óptimo es calculable
        regime = NearResolutionRegime.HALT
        gamma_multiplier = 4.0  # no se usa pero refleja el riesgo extremo
        q_max_fraction = 0.0  # no abrir posiciones nuevas (override exponencial)
        should_halt = True
        should_halt_side = True

    elif tau_years < TAU_1H:
        # 5min ≤ τ < 1h: crítico
        # Inventario máximo reducido exponencialmente
        regime = NearResolutionRegime.CRITICAL
        gamma_multiplier = 4.0  # γ_effective = 4γ
        q_max_fraction = min(0.1, q_max_fraction_exp)  # exponencial limitado al 10%
        should_halt = False
        should_halt_side = True  # halt en el lado con inventario

    elif tau_years < TAU_24H:
        # 1h ≤ τ < 24h: warning
        # Decaimiento exponencial de Q_max según §8.3 v2.1
        regime = NearResolutionRegime.WARNING
        gamma_multiplier = 2.0  # γ_effective = 2γ del §6.4
        q_max_fraction = min(0.5, q_max_fraction_exp)  # exponencial limitado al 50%
        should_halt = False
        should_halt_side = False

    else:
        # τ ≥ 24h: operación normal con decaimiento exponencial suave
        regime = NearResolutionRegime.NORMAL
        gamma_multiplier = 1.0
        q_max_fraction = q_max_fraction_exp  # fórmula exponencial pura
        should_halt = False
        should_halt_side = False

    return ResolutionFeatures(
        tau_years=tau_years,
        tau_days=tau_days,
        tau_hours=tau_hours,
        tau_minutes=tau_minutes,
        regime=regime,
        gamma_multiplier=gamma_multiplier,
        q_max_fraction=q_max_fraction,
        should_halt=should_halt,
        should_halt_side=should_halt_side,
    )


# ---------------------------------------------------------------------------
# Helpers para uso directo desde el execution engine
# ---------------------------------------------------------------------------


def effective_gamma(gamma: float, rf: ResolutionFeatures) -> float:
    """
    Devuelve γ_effective = γ · gamma_multiplier.

    Usado por GLFT y Cartea-Jaimungal para escalar la aversión
    al riesgo según el régimen de near-resolution.

    Args:
        gamma: coeficiente de aversión al riesgo base
        rf:    ResolutionFeatures ya calculadas

    Returns:
        γ ajustado por el régimen actual
    """
    return gamma * rf.gamma_multiplier


def effective_q_max(q_max: float, rf: ResolutionFeatures) -> float:
    """
    Devuelve Q_max_effective = Q_max · q_max_fraction.

    Usado por el risk manager para limitar el inventario
    según el régimen de near-resolution.

    Args:
        q_max: límite de inventario máximo en condiciones normales
        rf:    ResolutionFeatures ya calculadas

    Returns:
        Q_max ajustado por el régimen actual
    """
    return q_max * rf.q_max_fraction


# ELIMINADO: bernoulli_vol_safe — función obsoleta según MATH.md v2.1
# La volatilidad ahora es σ_b calibrada desde variación cuadrática de logit(p),
# no calculada analíticamente desde p y τ.
# El execution engine debe usar belief_vol_from_ticks() de microstructure.py
# y respetar los flags should_halt y should_halt_side en lugar de depender
# de un valor centinela arbitrario (100.0).
