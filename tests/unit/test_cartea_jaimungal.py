"""
tests/unit/test_cartea_jaimungal.py
────────────────────────────────────
Tests de strategies/market_making/cartea_jaimungal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from features.resolution import NearResolutionRegime
from normalizer.schema import MarketId, Venue
from strategies.market_making.cartea_jaimungal import CarteaJaimungalQuoter
from strategies.market_making.glft import TICK, logit


class TestCarteaJaimungalQuoterInit:
    def test_valida_gamma_I(self) -> None:
        with pytest.raises(ValueError, match="gamma_I must be positive"):
            CarteaJaimungalQuoter(gamma_I=0.0, kappa_x=0.8, phi=1.0, eta=0.05, rho=0.5)
        with pytest.raises(ValueError, match="gamma_I must be positive"):
            CarteaJaimungalQuoter(gamma_I=-0.1, kappa_x=0.8, phi=1.0, eta=0.05, rho=0.5)

    def test_valida_kappa_x(self) -> None:
        with pytest.raises(ValueError, match="kappa_x must be positive"):
            CarteaJaimungalQuoter(gamma_I=0.1, kappa_x=0.0, phi=1.0, eta=0.05, rho=0.5)
        with pytest.raises(ValueError, match="kappa_x must be positive"):
            CarteaJaimungalQuoter(gamma_I=0.1, kappa_x=-0.8, phi=1.0, eta=0.05, rho=0.5)

    def test_valida_phi(self) -> None:
        with pytest.raises(ValueError, match="phi must be positive"):
            CarteaJaimungalQuoter(gamma_I=0.1, kappa_x=0.8, phi=0.0, eta=0.05, rho=0.5)
        with pytest.raises(ValueError, match="phi must be positive"):
            CarteaJaimungalQuoter(gamma_I=0.1, kappa_x=0.8, phi=-1.0, eta=0.05, rho=0.5)

    def test_valida_eta(self) -> None:
        with pytest.raises(ValueError, match="eta must be positive"):
            CarteaJaimungalQuoter(gamma_I=0.1, kappa_x=0.8, phi=1.0, eta=0.0, rho=0.5)
        with pytest.raises(ValueError, match="eta must be positive"):
            CarteaJaimungalQuoter(gamma_I=0.1, kappa_x=0.8, phi=1.0, eta=-0.05, rho=0.5)

    def test_valida_rho(self) -> None:
        with pytest.raises(ValueError, match="rho must be in"):
            CarteaJaimungalQuoter(gamma_I=0.1, kappa_x=0.8, phi=1.0, eta=0.05, rho=1.1)
        with pytest.raises(ValueError, match="rho must be in"):
            CarteaJaimungalQuoter(gamma_I=0.1, kappa_x=0.8, phi=1.0, eta=0.05, rho=-1.1)


class TestCarteaJaimungalQuoting:
    @pytest.fixture
    def market_id(self) -> MarketId:
        return MarketId(Venue.KALSHI, "KXBTC-TEST")

    @pytest.fixture
    def quoter(self) -> CarteaJaimungalQuoter:
        return CarteaJaimungalQuoter(
            gamma_I=0.1,
            kappa_x=0.8,
            phi=1.5,
            eta=0.04,
            rho=0.5,
        )

    def test_halt_en_regimenes_inactivos(
        self, quoter: CarteaJaimungalQuoter, market_id: MarketId
    ) -> None:
        for regime in (NearResolutionRegime.HALT, NearResolutionRegime.RESOLVED):
            quote = quoter.quote(
                market_id=market_id,
                mid_p=0.5,
                inventory=0.0,
                tau_years=0.1,
                belief_vol=0.1,
                regime=regime,
            )
            assert not quote.is_valid
            assert "regime=" in quote.invalid_reason
            assert quote.model == "cartea_jaimungal"

    def test_quote_simetrico_sin_inventario_y_sin_senal(
        self, quoter: CarteaJaimungalQuoter, market_id: MarketId
    ) -> None:
        mid_p = 0.45
        quote = quoter.quote(
            market_id=market_id,
            mid_p=mid_p,
            inventory=0.0,
            tau_years=0.5,
            belief_vol=0.1,
            regime=NearResolutionRegime.NORMAL,
            mu_hat=0.0,
        )

        assert quote.is_valid
        assert quote.mid_price_p == pytest.approx(mid_p)
        assert quote.reservation_X == pytest.approx(logit(mid_p))
        assert quote.signal_skew == 0.0

        # En logit, la distancia bid y ask a la reserva debe ser igual
        assert quote.reservation_X - quote.bid_X == pytest.approx(quote.half_spread_X)
        assert quote.ask_X - quote.reservation_X == pytest.approx(quote.half_spread_X)

    def test_skew_por_inventario(self, quoter: CarteaJaimungalQuoter, market_id: MarketId) -> None:
        mid_p = 0.5
        tau = 0.2
        vol = 0.15

        # Cotizar con inventario positivo (largo)
        quote_long = quoter.quote(
            market_id=market_id,
            mid_p=mid_p,
            inventory=5.0,
            tau_years=tau,
            belief_vol=vol,
            regime=NearResolutionRegime.NORMAL,
            mu_hat=0.0,
        )

        # Cotizar con inventario negativo (corto)
        quote_short = quoter.quote(
            market_id=market_id,
            mid_p=mid_p,
            inventory=-5.0,
            tau_years=tau,
            belief_vol=vol,
            regime=NearResolutionRegime.NORMAL,
            mu_hat=0.0,
        )

        # Con inventario largo, los precios de cotización deben ser inferiores
        # a los del corto para incentivar la venta y desincentivar la compra
        assert quote_long.reservation_X < logit(mid_p)
        assert quote_short.reservation_X > logit(mid_p)

        assert quote_long.bid_p < quote_short.bid_p
        assert quote_long.ask_p < quote_short.ask_p

    def test_skew_por_senal_direccion(
        self, quoter: CarteaJaimungalQuoter, market_id: MarketId
    ) -> None:
        mid_p = 0.5
        tau = 0.2
        vol = 0.15

        # Quoter sin señal (mu_hat = 0)
        quote_base = quoter.quote(
            market_id=market_id,
            mid_p=mid_p,
            inventory=0.0,
            tau_years=tau,
            belief_vol=vol,
            regime=NearResolutionRegime.NORMAL,
            mu_hat=0.0,
        )

        # Señal alcista (mu_hat > 0)
        quote_bull = quoter.quote(
            market_id=market_id,
            mid_p=mid_p,
            inventory=0.0,
            tau_years=tau,
            belief_vol=vol,
            regime=NearResolutionRegime.NORMAL,
            mu_hat=2.0,
        )

        # Señal bajista (mu_hat < 0)
        quote_bear = quoter.quote(
            market_id=market_id,
            mid_p=mid_p,
            inventory=0.0,
            tau_years=tau,
            belief_vol=vol,
            regime=NearResolutionRegime.NORMAL,
            mu_hat=-2.0,
        )

        # Dado que rho > 0:
        # - mu_hat > 0 debe desplazar los quotes hacia arriba
        # - mu_hat < 0 debe desplazar los quotes hacia abajo
        assert quote_bull.signal_skew > 0.0
        assert quote_bear.signal_skew < 0.0

        assert quote_bull.reservation_X > quote_base.reservation_X
        assert quote_bear.reservation_X < quote_base.reservation_X

        assert quote_bull.bid_p > quote_base.bid_p
        assert quote_bull.ask_p > quote_base.ask_p
        assert quote_bear.bid_p < quote_base.bid_p
        assert quote_bear.ask_p < quote_base.ask_p

    def test_rho_negativo_revierte_skew(self, market_id: MarketId) -> None:
        quoter_neg_rho = CarteaJaimungalQuoter(
            gamma_I=0.1,
            kappa_x=0.8,
            phi=1.5,
            eta=0.04,
            rho=-0.5,  # rho negativo
        )

        mid_p = 0.5
        tau = 0.2
        vol = 0.15

        # Con rho < 0 y mu_hat > 0 (bullish), la correlación negativa
        # hace que esperemos bajadas de precio. Por tanto, el skew debe ser negativo.
        quote = quoter_neg_rho.quote(
            market_id=market_id,
            mid_p=mid_p,
            inventory=0.0,
            tau_years=tau,
            belief_vol=vol,
            regime=NearResolutionRegime.NORMAL,
            mu_hat=2.0,
        )

        assert quote.signal_skew < 0.0
        assert quote.reservation_X < logit(mid_p)

    def test_tau_limite_cero_desvanece_senal(
        self, quoter: CarteaJaimungalQuoter, market_id: MarketId
    ) -> None:
        # Cuando tau -> 0, 1 - e^{-phi*tau} -> 0, por lo que el skew de señal debe ir a cero.
        mid_p = 0.5
        vol = 0.15
        mu = 5.0

        quote_tiny_tau = quoter.quote(
            market_id=market_id,
            mid_p=mid_p,
            inventory=0.0,
            tau_years=1e-8,
            belief_vol=vol,
            regime=NearResolutionRegime.NORMAL,
            mu_hat=mu,
        )

        # Debería ser muy cercano a 0
        assert abs(quote_tiny_tau.signal_skew) < 1e-6

        quote_zero_tau = quoter.quote(
            market_id=market_id,
            mid_p=mid_p,
            inventory=0.0,
            tau_years=0.0,
            belief_vol=vol,
            regime=NearResolutionRegime.NORMAL,
            mu_hat=mu,
        )
        assert quote_zero_tau.signal_skew == 0.0

    def test_tick_floor_ajuste(self, market_id: MarketId) -> None:
        # Usamos un quoter con gamma_I alto para ensanchar el spread y forzar
        # que el bid caiga por debajo de TICK (0.01)
        quoter = CarteaJaimungalQuoter(
            gamma_I=2.0,
            kappa_x=0.8,
            phi=1.5,
            eta=0.04,
            rho=0.5,
        )
        quote = quoter.quote(
            market_id=market_id,
            mid_p=0.02,
            inventory=5.0,
            tau_years=0.5,
            belief_vol=0.2,
            regime=NearResolutionRegime.NORMAL,
            mu_hat=-5.0,
        )

        # El bid_p debería ser TICK
        assert quote.is_valid
        assert quote.bid_p >= TICK
        assert quote.ask_p >= TICK + TICK
        assert quote.invalid_reason == "tick_floor_adjusted"
