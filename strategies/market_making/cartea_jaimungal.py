"""
strategies/market_making/cartea_jaimungal.py
─────────────────────────────────────────────
Modelo Cartea-Jaimungal (CJ) en espacio logit — aproximación AS (§4.5 MATH.md v2.1).

Incorpora una señal direccional de corto plazo (drift latente mu_t) que sesga
el precio de reserva.

Relación con MATH.md v2.1:
  §4.5 → precio de reserva con señal direccional
  §5.4 → reservation price y spread en espacio logit
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from features.resolution import TAU_1H, NearResolutionRegime
from normalizer.schema import MarketId
from strategies.market_making.glft import TICK, Quote, logit, sigma


class CarteaJaimungalQuoter:
    """
    Quoter basado en el modelo de Cartea-Jaimungal en espacio logit.

    Fórmulas (MATH.md v2.1 §4.5 & §5.4):
      Varianza integrada:
        σ̄²_b = σ²_b (constante)

      Fórmula de decaimiento temporal de la señal (phi_1):
        φ_1(τ) = (ρ·σ_b·η / φ) · (1 - e^{-φ·τ})

      Reservation log-odds:
        r̃_X = X_t - q·γ_I·σ̄²_b·τ + φ_1(τ)·μ̂_t

      Optimal half-spread en logit (idéntico a GLFT):
        δ*/2 = γ_I·σ̄²_b·τ/2 + (1/κ_x)·ln(1+γ_I/κ_x)

      Quotes:
        bid_p = σ(r̃_X - δ*/2)
        ask_p = σ(r̃_X + δ*/2)
    """

    def __init__(
        self,
        gamma_I: float,
        kappa_x: float,
        phi: float,
        eta: float,
        rho: float,
    ) -> None:
        """
        Args:
            gamma_I: CARA inventory risk aversion.
            kappa_x: fill curve decay en logit.
            phi: mean-reversion speed del drift latente (phi > 0).
            eta: volatilidad del drift latente (eta > 0).
            rho: correlación entre innovaciones de precio y señal (rho ∈ [-0.99, 0.99]).
        """
        if gamma_I <= 0:
            raise ValueError(f"gamma_I must be positive, got {gamma_I}")
        if kappa_x <= 0:
            raise ValueError(f"kappa_x must be positive, got {kappa_x}")
        if phi <= 0:
            raise ValueError(f"phi must be positive, got {phi}")
        if eta <= 0:
            raise ValueError(f"eta must be positive, got {eta}")
        if not (-0.999 <= rho <= 0.999):
            raise ValueError(f"rho must be in [-0.99, 0.99], got {rho}")

        self.gamma_I = gamma_I
        self.kappa_x = kappa_x
        self.phi = phi
        self.eta = eta
        self.rho = rho

    def quote(
        self,
        market_id: MarketId,
        mid_p: float,
        inventory: float,
        tau_years: float,
        belief_vol: float,
        regime: NearResolutionRegime,
        mu_hat: float = 0.0,
        timestamp: datetime | None = None,
    ) -> Quote:
        """
        Calcula quotes de Cartea-Jaimungal incorporando la señal mu_hat.
        """
        ts = timestamp or datetime.now(tz=UTC)

        # Halt inmediato si el régimen lo requiere
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

        # Estado en espacio logit
        X_t = logit(mid_p)

        # Varianza integrada
        sigma_bar_sq = belief_vol**2 * tau_years

        # Signal skew: φ_1(τ) · μ̂_t
        # φ_1(τ) = (ρ · σ_b · η / φ) · (1 - e^{-φ·τ})
        # Si tau_years es muy pequeño o cero, phi_1(tau) -> 0
        if tau_years > 0:
            phi_1 = (self.rho * belief_vol * self.eta / self.phi) * (
                1.0 - math.exp(-self.phi * tau_years)
            )
            signal_skew = phi_1 * mu_hat
        else:
            signal_skew = 0.0

        # Reservation log-odds
        reservation_X = X_t - inventory * self.gamma_I * sigma_bar_sq + signal_skew

        # Optimal half-spread en logit (igual que GLFT)
        inventory_term = self.gamma_I * sigma_bar_sq / 2.0
        rent_term = (1.0 / self.kappa_x) * math.log(1.0 + self.gamma_I / self.kappa_x)
        half_spread_X = inventory_term + rent_term

        # Quotes en logit
        bid_X = reservation_X - half_spread_X
        ask_X = reservation_X + half_spread_X

        # Mapear a espacio precio
        bid_p_raw = sigma(bid_X)
        ask_p_raw = sigma(ask_X)

        # Aplicar tick floor
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
            model="cartea_jaimungal",
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
            signal_skew=signal_skew,
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
        Aplica el tick floor. Lógica heredada de GLFT.
        """
        bid_below = bid_p < TICK
        ask_above = ask_p > 1.0 - TICK

        if not bid_below and not ask_above:
            return bid_p, ask_p, True, ""

        if tau_years < TAU_1H:
            return (
                bid_p,
                ask_p,
                False,
                (
                    f"tick_floor_near_resolution: "
                    f"bid={bid_p:.4f} ask={ask_p:.4f} tau_hours={tau_years*8760:.1f}"
                ),
            )

        reservation_p = sigma(reservation_X)
        bid_p_adj = max(TICK, reservation_p - TICK / 2.0)
        ask_p_adj = min(1.0 - TICK, reservation_p + TICK / 2.0)

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
        X_t = logit(mid_p)
        return Quote(
            market_id=market_id,
            timestamp=timestamp,
            model="cartea_jaimungal",
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
