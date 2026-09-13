"""
tests/unit/test_cartea_jaimungal.py
────────────────────────────────────
Tests de strategies/market_making/cartea_jaimungal.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from features.resolution import NearResolutionRegime
from normalizer.price_grid import PriceLadder, ladder_for_venue
from normalizer.schema import MarketId, Venue
from strategies.market_making.cartea_jaimungal import CarteaJaimungalQuoter
from strategies.market_making.glft import logit


class TestCarteaJaimungalQuoterInit:
    def test_validates_gamma_I(self) -> None:
        with pytest.raises(ValueError, match="gamma_I must be positive"):
            CarteaJaimungalQuoter(gamma_I=0.0, kappa_x=0.8, phi=1.0, eta=0.05, rho=0.5)
        with pytest.raises(ValueError, match="gamma_I must be positive"):
            CarteaJaimungalQuoter(gamma_I=-0.1, kappa_x=0.8, phi=1.0, eta=0.05, rho=0.5)

    def test_validates_kappa_x(self) -> None:
        with pytest.raises(ValueError, match="kappa_x must be positive"):
            CarteaJaimungalQuoter(gamma_I=0.1, kappa_x=0.0, phi=1.0, eta=0.05, rho=0.5)
        with pytest.raises(ValueError, match="kappa_x must be positive"):
            CarteaJaimungalQuoter(gamma_I=0.1, kappa_x=-0.8, phi=1.0, eta=0.05, rho=0.5)

    def test_validates_phi(self) -> None:
        with pytest.raises(ValueError, match="phi must be positive"):
            CarteaJaimungalQuoter(gamma_I=0.1, kappa_x=0.8, phi=0.0, eta=0.05, rho=0.5)
        with pytest.raises(ValueError, match="phi must be positive"):
            CarteaJaimungalQuoter(gamma_I=0.1, kappa_x=0.8, phi=-1.0, eta=0.05, rho=0.5)

    def test_validates_eta(self) -> None:
        with pytest.raises(ValueError, match="eta must be positive"):
            CarteaJaimungalQuoter(gamma_I=0.1, kappa_x=0.8, phi=1.0, eta=0.0, rho=0.5)
        with pytest.raises(ValueError, match="eta must be positive"):
            CarteaJaimungalQuoter(gamma_I=0.1, kappa_x=0.8, phi=1.0, eta=-0.05, rho=0.5)

    def test_validates_rho_mu(self) -> None:
        """
        rho is ρ_μ, the measure-change discount ∈ (0, 1] — not a correlation.

        Through v2.1 the range [-0.99, 0.99] was accepted because it was used
        as the price-signal correlation inside φ_1. A negative ρ inverted the
        sense of the signal, which is precisely what ρ_μ cannot do: it shrinks
        confidence in μ̂, it never flips its sign.
        """
        for bad in (1.1, 0.0, -0.5, -1.1):
            with pytest.raises(ValueError, match="rho_mu"):
                CarteaJaimungalQuoter(gamma_I=0.1, kappa_x=12.0, phi=1.0, eta=0.05, rho=bad)

        # The valid extremes are accepted
        CarteaJaimungalQuoter(gamma_I=0.1, kappa_x=12.0, phi=1.0, eta=0.05, rho=1.0)
        CarteaJaimungalQuoter(gamma_I=0.1, kappa_x=12.0, phi=1.0, eta=0.05, rho=1e-6)


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

    def test_halt_in_inactive_regimes(
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

    def test_quotes_symmetric_without_inventory_or_signal(
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

        # In logit space, bid and ask must sit equidistant from the reservation
        assert quote.reservation_X - quote.bid_X == pytest.approx(quote.half_spread_X)
        assert quote.ask_X - quote.reservation_X == pytest.approx(quote.half_spread_X)

    def test_skew_from_inventory(self, quoter: CarteaJaimungalQuoter, market_id: MarketId) -> None:
        mid_p = 0.5
        tau = 0.2
        vol = 0.15

        # Quote with positive inventory (long)
        quote_long = quoter.quote(
            market_id=market_id,
            mid_p=mid_p,
            inventory=5.0,
            tau_years=tau,
            belief_vol=vol,
            regime=NearResolutionRegime.NORMAL,
            mu_hat=0.0,
        )

        # Quote with negative inventory (short)
        quote_short = quoter.quote(
            market_id=market_id,
            mid_p=mid_p,
            inventory=-5.0,
            tau_years=tau,
            belief_vol=vol,
            regime=NearResolutionRegime.NORMAL,
            mu_hat=0.0,
        )

        # With long inventory the quoted prices must sit below those quoted
        # when short, to encourage selling and discourage buying
        assert quote_long.reservation_X < logit(mid_p)
        assert quote_short.reservation_X > logit(mid_p)

        assert quote_long.bid_p < quote_short.bid_p
        assert quote_long.ask_p < quote_short.ask_p

    def test_skew_direction_from_signal(
        self, quoter: CarteaJaimungalQuoter, market_id: MarketId
    ) -> None:
        mid_p = 0.5
        tau = 0.2
        vol = 0.15

        # Quoter with no signal (mu_hat = 0)
        quote_base = quoter.quote(
            market_id=market_id,
            mid_p=mid_p,
            inventory=0.0,
            tau_years=tau,
            belief_vol=vol,
            regime=NearResolutionRegime.NORMAL,
            mu_hat=0.0,
        )

        # Bullish signal (mu_hat > 0)
        quote_bull = quoter.quote(
            market_id=market_id,
            mid_p=mid_p,
            inventory=0.0,
            tau_years=tau,
            belief_vol=vol,
            regime=NearResolutionRegime.NORMAL,
            mu_hat=2.0,
        )

        # Bearish signal (mu_hat < 0)
        quote_bear = quoter.quote(
            market_id=market_id,
            mid_p=mid_p,
            inventory=0.0,
            tau_years=tau,
            belief_vol=vol,
            regime=NearResolutionRegime.NORMAL,
            mu_hat=-2.0,
        )

        # Given rho > 0:
        # - mu_hat > 0 must shift the quotes upwards
        # - mu_hat < 0 must shift the quotes downwards
        assert quote_bull.signal_skew > 0.0
        assert quote_bear.signal_skew < 0.0

        assert quote_bull.reservation_X > quote_base.reservation_X
        assert quote_bear.reservation_X < quote_base.reservation_X

        assert quote_bull.bid_p > quote_base.bid_p
        assert quote_bull.ask_p > quote_base.ask_p
        assert quote_bear.bid_p < quote_base.bid_p
        assert quote_bear.ask_p < quote_base.ask_p

    def test_rho_mu_scales_skew_without_changing_its_sign(self, market_id: MarketId) -> None:
        """
        ρ_μ shrinks the signal proportionally; it never inverts it.

        φ_1 = ρ_μ·(1-e^{-φτ})/φ is linear in ρ_μ, so halving ρ_μ must halve
        the skew exactly, preserving the sign of μ̂.
        """
        kwargs = {"gamma_I": 0.1, "kappa_x": 12.0, "phi": 1.5, "eta": 0.04}
        q_full = CarteaJaimungalQuoter(rho=1.0, **kwargs)
        q_half = CarteaJaimungalQuoter(rho=0.5, **kwargs)

        args = {
            "market_id": market_id,
            "mid_p": 0.5,
            "inventory": 0.0,
            "tau_years": 0.2,
            "belief_vol": 0.15,
            "regime": NearResolutionRegime.NORMAL,
            "mu_hat": 2.0,
        }
        skew_full = q_full.quote(**args).signal_skew
        skew_half = q_half.quote(**args).signal_skew

        assert skew_full > 0.0
        assert skew_half > 0.0
        assert skew_half == pytest.approx(skew_full / 2.0, rel=1e-12)

    def test_tau_limit_zero_vanishes_the_signal(
        self, quoter: CarteaJaimungalQuoter, market_id: MarketId
    ) -> None:
        # As tau → 0, 1 - e^{-phi*tau} → 0, so the signal skew must vanish.
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

        # Should be very close to 0
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

    def test_coarse_grid_never_quotes_below_its_minimum(self, market_id: MarketId) -> None:
        """
        With loaded inventory and a bearish signal the reservation price falls
        to 0.0069. On a one-cent grid that price is simply not quotable: the
        minimum is 0.01. The quoter detects that and does NOT quote.

        This is correct behaviour, not a limitation: inventing a quote at
        0.0100/0.0200 around a reservation of 0.0069 would be quoting a price
        the model never asked for.
        """
        quoter = CarteaJaimungalQuoter(gamma_I=2.0, kappa_x=0.8, phi=1.5, eta=0.04, rho=0.5)
        quote = quoter.quote(
            market_id=market_id,
            mid_p=0.02,
            inventory=5.0,
            tau_years=0.5,
            belief_vol=0.2,
            regime=NearResolutionRegime.NORMAL,
            mu_hat=-5.0,
            ladder=PriceLadder.uniform(0.01),
        )
        assert not quote.is_valid
        assert "straddle_reservation" in quote.invalid_reason

    def test_fine_kalshi_grid_avoids_the_tick_floor(self, market_id: MarketId) -> None:
        """
        The same quote that needed adjustment on a one-cent grid is quotable
        as-is on Kalshi's real grid (0.0001 below $0.01).

        That is the point of the change: to stop discarding the low band.
        """
        quoter = CarteaJaimungalQuoter(gamma_I=2.0, kappa_x=0.8, phi=1.5, eta=0.04, rho=0.5)
        quote = quoter.quote(
            market_id=market_id,
            mid_p=0.02,
            inventory=5.0,
            tau_years=0.5,
            belief_vol=0.2,
            regime=NearResolutionRegime.NORMAL,
            mu_hat=-5.0,
        )
        grid = ladder_for_venue(market_id.venue)
        assert quote.is_valid
        assert quote.invalid_reason == ""
        assert quote.bid_p < quote.ask_p
        for side in (quote.bid_p, quote.ask_p):
            tick = grid.tick_at(side)
            assert side / tick == pytest.approx(round(side / tick), abs=1e-6)


# ---------------------------------------------------------------------------
# φ_1 — the alpha-capture factor (MATH.md v2.2 §4.4)
# ---------------------------------------------------------------------------


class TestPhi1AlphaCapture:
    """
    φ_1(τ) = ρ_μ·(1 - e^{-φτ})/φ is the accumulated drift one unit of inventory
    can capture over the remaining horizon:

        (1/μ_t)·∫_t^T E[μ_s|μ_t] ds = ∫_0^τ e^{-φu} du = (1-e^{-φτ})/φ

    v2.1 used φ_1 = (ρ·σ_b·η/φ)(1-e^{-φτ}), which is dimensionally impossible:
    with [μ̂] = nats/year, φ_1·μ̂ must come out in nats, hence [φ_1] = years.
    But [ρσ_bη/φ] = nats/year and the product came out in nats²/year². The
    ρσ_bη belongs to the q² term (the cross variation d<S,μ>), not to the μq
    one.
    """

    MID = MarketId(Venue.KALSHI, "KXPHI")

    @staticmethod
    def _skew(rho: float, phi: float, tau: float, mu: float, vol: float = 0.2) -> float:
        q = CarteaJaimungalQuoter(gamma_I=0.1, kappa_x=12.0, phi=phi, eta=0.05, rho=rho)
        return q.quote(
            market_id=TestPhi1AlphaCapture.MID,
            mid_p=0.5,
            inventory=0.0,
            tau_years=tau,
            belief_vol=vol,
            regime=NearResolutionRegime.NORMAL,
            mu_hat=mu,
        ).signal_skew

    def test_matches_the_closed_form(self) -> None:
        rho, phi, tau, mu = 0.7, 1.5, 0.4, 2.0
        esperado = rho * (1.0 - math.exp(-phi * tau)) / phi * mu
        assert self._skew(rho, phi, tau, mu) == pytest.approx(esperado, rel=1e-12)

    def test_does_not_depend_on_eta_or_sigma_b(self) -> None:
        """
        A direct test of the dimensional defect: under the v2.1 formula the
        skew scaled with σ_b·η. Under the correct one it can depend on neither.
        """
        base = CarteaJaimungalQuoter(gamma_I=0.1, kappa_x=12.0, phi=1.5, eta=0.05, rho=0.7)
        other_eta = CarteaJaimungalQuoter(gamma_I=0.1, kappa_x=12.0, phi=1.5, eta=5.0, rho=0.7)
        args = {
            "market_id": self.MID,
            "mid_p": 0.5,
            "inventory": 0.0,
            "tau_years": 0.4,
            "regime": NearResolutionRegime.NORMAL,
            "mu_hat": 2.0,
        }
        assert base.quote(belief_vol=0.2, **args).signal_skew == pytest.approx(
            other_eta.quote(belief_vol=0.2, **args).signal_skew, rel=1e-12
        )
        assert base.quote(belief_vol=0.2, **args).signal_skew == pytest.approx(
            base.quote(belief_vol=3.0, **args).signal_skew, rel=1e-12
        )

    def test_zero_tau_cancels_the_signal(self) -> None:
        assert self._skew(0.7, 1.5, 0.0, 2.0) == 0.0

    def test_saturates_at_rho_over_phi(self) -> None:
        """τ→∞ ⟹ φ_1 → ρ_μ/φ: an eternal signal is worth one reversion time."""
        rho, phi, mu = 0.7, 1.5, 2.0
        assert self._skew(rho, phi, 1e6, mu) == pytest.approx(rho / phi * mu, rel=1e-9)

    def test_is_increasing_in_tau(self) -> None:
        skews = [self._skew(0.7, 1.5, t, 2.0) for t in (0.01, 0.1, 0.5, 2.0, 10.0)]
        assert skews == sorted(skews)

    def test_large_phi_exhausts_the_signal_sooner(self) -> None:
        """Faster mean reversion = less alpha capturable over the same τ."""
        assert self._skew(0.7, 5.0, 0.4, 2.0) < self._skew(0.7, 0.5, 0.4, 2.0)


class TestSignalSkewSign:
    """
    The sign of the signal term in the reservation price.

    MATH.md v2.1 §4.5 wrote p̃ = S − ∂_q g, contradicting its own §4.3 (from
    which p̃ = S + ∂_q g follows, the midpoint between ask = S + δ^a and
    bid = S − δ^b with δ^{a,b} = 1/κ ± ∂_q g). Together with the wrong sign of
    φ_2 in §4.4, the two errors cancelled in the inventory term and left the
    signal term inverted. The code always had the right sign; this test pins it
    so it is not "corrected" by reading the old document.
    """

    MID = MarketId(Venue.KALSHI, "KXSIGN")

    def _quote(self, mu: float, inventory: float = 0.0):  # noqa: ANN202
        q = CarteaJaimungalQuoter(gamma_I=0.1, kappa_x=12.0, phi=1.5, eta=0.05, rho=1.0)
        return q.quote(
            market_id=self.MID,
            mid_p=0.5,
            inventory=inventory,
            tau_years=0.4,
            belief_vol=0.2,
            regime=NearResolutionRegime.NORMAL,
            mu_hat=mu,
        )

    def test_positive_drift_raises_both_quotes(self) -> None:
        """Expecting a rise means wanting to go long: lift BOTH bid and ask."""
        neutro, alcista = self._quote(0.0), self._quote(2.0)
        assert alcista.signal_skew > 0.0
        assert alcista.reservation_X > neutro.reservation_X
        assert alcista.bid_p > neutro.bid_p
        assert alcista.ask_p > neutro.ask_p

    def test_negative_drift_lowers_both_quotes(self) -> None:
        neutro, bajista = self._quote(0.0), self._quote(-2.0)
        assert bajista.signal_skew < 0.0
        assert bajista.bid_p < neutro.bid_p
        assert bajista.ask_p < neutro.ask_p

    def test_signal_does_not_change_spread_width(self) -> None:
        """In CJ the signal shifts the centre; the half-spread is GLFT's."""
        anchos = {self._quote(m).half_spread_X for m in (-3.0, 0.0, 3.0)}
        assert len(anchos) == 1

    def test_inventory_and_signal_oppose_correctly(self) -> None:
        """q > 0 empuja abajo, μ̂ > 0 empuja arriba: signos opuestos."""
        solo_inv = self._quote(0.0, inventory=5.0)
        solo_sig = self._quote(2.0, inventory=0.0)
        base = self._quote(0.0, inventory=0.0)
        assert solo_inv.reservation_X < base.reservation_X
        assert solo_sig.reservation_X > base.reservation_X


class TestCJEffectiveGamma:
    """§6.4's γ_eff must apply in CJ too, not only in GLFT."""

    MID = MarketId(Venue.KALSHI, "KXCJG")

    def _reservation(self, regime: NearResolutionRegime) -> float:
        q = CarteaJaimungalQuoter(gamma_I=0.1, kappa_x=12.0, phi=1.5, eta=0.05, rho=1.0)
        return q.quote(
            market_id=self.MID,
            mid_p=0.5,
            inventory=4.0,
            tau_years=0.5,
            belief_vol=0.3,
            regime=regime,
            mu_hat=0.0,
        ).reservation_X

    @pytest.mark.parametrize(
        ("regime", "mult"),
        [(NearResolutionRegime.WARNING, 2.0), (NearResolutionRegime.CRITICAL, 4.0)],
    )
    def test_inventory_skew_scales(self, regime: NearResolutionRegime, mult: float) -> None:
        base = logit(0.5) - self._reservation(NearResolutionRegime.NORMAL)
        assert base > 0.0
        assert logit(0.5) - self._reservation(regime) == pytest.approx(base * mult, rel=1e-12)
