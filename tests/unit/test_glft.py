"""
tests/unit/test_glft.py
Tests for the GLFT quoter in logit space — MATH.md v2.2 §5.4 and §6.4.

This file did not exist: glft.py, the module that decides what price gets
quoted, had not a single test of its own. The invariants pinned here are the
ones that failed silently up to v2.2.
"""

from __future__ import annotations

import math

import pytest

from features.resolution import (
    GAMMA_MULTIPLIER_BY_REGIME,
    NearResolutionRegime,
)
from normalizer.price_grid import ladder_for_venue
from normalizer.schema import MarketId, Venue
from strategies.market_making.glft import (
    DEFAULT_GAMMA_I,
    DEFAULT_KAPPA_X,
    LOGIT_MAX,
    MAX_HALF_SPREAD_X,
    TICK,
    GLFTQuoter,
    apply_tick_floor,
    logit,
    sigma,
)

MID = MarketId(Venue.KALSHI, "KXTEST")


@pytest.fixture
def quoter() -> GLFTQuoter:
    return GLFTQuoter(gamma_I=DEFAULT_GAMMA_I, kappa_x=DEFAULT_KAPPA_X)


# ---------------------------------------------------------------------------
# Numerical stability of the sigmoid
# ---------------------------------------------------------------------------


class TestSigmaNumericalStability:
    """
    σ(x) = 1/(1+e^{-x}) overflowed with OverflowError for x < -709.

    Not hypothetical: on the repository's own data (σ_b up to 4·10^5, from
    annualising Δt a thousand times smaller than the real one and, on top of
    that, without filtering microstructure noise) the backtester aborted with
    "math range error" before emitting a single trace row.
    """

    @pytest.mark.parametrize("x", [-1e300, -1e9, -800.0, -710.0, -709.0, -50.0, 0.0, 50.0, 1e300])
    def test_does_not_overflow_across_the_range(self, x: float) -> None:
        p = sigma(x)
        assert math.isfinite(p)
        assert 0.0 <= p <= 1.0

    def test_symmetry(self) -> None:
        for x in (0.5, 1.0, 5.0, 20.0, 100.0):
            assert sigma(-x) == pytest.approx(1.0 - sigma(x), abs=1e-15)

    def test_monotonia(self) -> None:
        xs = [-800.0, -100.0, -1.0, 0.0, 1.0, 100.0, 800.0]
        vals = [sigma(x) for x in xs]
        assert vals == sorted(vals)

    def test_is_inverse_of_logit(self) -> None:
        for p in (0.001, 0.01, 0.3, 0.5, 0.7, 0.99, 0.999):
            assert sigma(logit(p)) == pytest.approx(p, rel=1e-12)


# ---------------------------------------------------------------------------
# Rent term — the exact form (§2.4), not the approximation
# ---------------------------------------------------------------------------


class TestRentTerm:
    """
    The rent term must be (1/γ)·ln(1+γ/κ), the exact A-S first-order condition.

    v2.1 implemented (1/κ)·ln(1+γ/κ), which is neither the exact form nor its
    γ≪κ limit (which would be 1/κ): it sits below both. At γ=0.1, κ=0.8 it
    underestimated by a factor of 8, and the maker quoted far tighter than
    prescribed.
    """

    @staticmethod
    def _rent_from_quote(gamma: float, kappa: float) -> float:
        """Isolate the rent term: with σ_b = 0 the inventory term vanishes."""
        q = GLFTQuoter(gamma_I=gamma, kappa_x=kappa)
        quote = q.quote(
            market_id=MID,
            mid_p=0.5,
            inventory=0.0,
            tau_years=1.0,
            belief_vol=0.0,
            regime=NearResolutionRegime.NORMAL,
        )
        return quote.half_spread_X

    @pytest.mark.parametrize(
        ("gamma", "kappa"), [(0.1, 12.0), (0.5, 1.0), (1.0, 1.0), (0.05, 20.0)]
    )
    def test_uses_the_exact_form(self, gamma: float, kappa: float) -> None:
        exacta = (1.0 / gamma) * math.log(1.0 + gamma / kappa)
        assert self._rent_from_quote(gamma, kappa) == pytest.approx(exacta, rel=1e-12)

    @pytest.mark.parametrize(("gamma", "kappa"), [(0.1, 12.0), (0.5, 1.0), (0.05, 20.0)])
    def test_does_not_use_the_v21_approximation(self, gamma: float, kappa: float) -> None:
        aproximada = (1.0 / kappa) * math.log(1.0 + gamma / kappa)
        assert self._rent_from_quote(gamma, kappa) != pytest.approx(aproximada, rel=1e-6)

    def test_tends_to_one_over_kappa_for_small_gamma(self) -> None:
        """The γ→0 limit of (1/γ)ln(1+γ/κ) is 1/κ — A-S's actual approximation."""
        kappa = 12.0
        assert self._rent_from_quote(1e-6, kappa) == pytest.approx(1.0 / kappa, rel=1e-5)


# ---------------------------------------------------------------------------
# Effective γ per regime — §6.4
# ---------------------------------------------------------------------------


class TestEffectiveGammaIsApplied:
    """
    §6.4 prescribes γ_eff = 2γ in WARNING and 4γ in CRITICAL.

    Up to v2.1 `effective_gamma()` existed, had tests of its own... and no
    quoter called it. The regime only served to stop quoting below five
    minutes, so the entire near-resolution extension — the section that gives
    this project its edge — was inert.

    Inventory skew is the cleanest probe: r̃_X − X = −q·γ_eff·σ̄²_b·τ is exactly
    linear in γ_eff, so the ratio between regimes must reproduce the multiplier
    to machine precision.
    """

    ARGS = {
        "market_id": MID,
        "mid_p": 0.5,
        "inventory": 4.0,
        "tau_years": 0.5,
        "belief_vol": 0.3,
    }

    def _skew(self, quoter: GLFTQuoter, regime: NearResolutionRegime) -> float:
        q = quoter.quote(regime=regime, **self.ARGS)
        return logit(self.ARGS["mid_p"]) - q.reservation_X

    @pytest.mark.parametrize(
        ("regime", "expected"),
        [
            (NearResolutionRegime.WARNING, 2.0),
            (NearResolutionRegime.CRITICAL, 4.0),
        ],
    )
    def test_skew_scales_with_the_multiplier(
        self, quoter: GLFTQuoter, regime: NearResolutionRegime, expected: float
    ) -> None:
        base = self._skew(quoter, NearResolutionRegime.NORMAL)
        assert base > 0.0
        assert self._skew(quoter, regime) == pytest.approx(base * expected, rel=1e-12)
        assert GAMMA_MULTIPLIER_BY_REGIME[regime] == expected

    def test_spread_widens_approaching_resolution(self, quoter: GLFTQuoter) -> None:
        widths = [
            quoter.quote(regime=r, **self.ARGS).half_spread_X
            for r in (
                NearResolutionRegime.NORMAL,
                NearResolutionRegime.WARNING,
                NearResolutionRegime.CRITICAL,
            )
        ]
        assert widths[0] < widths[1] < widths[2]

    def test_halt_and_resolved_do_not_quote(self, quoter: GLFTQuoter) -> None:
        for regime in (NearResolutionRegime.HALT, NearResolutionRegime.RESOLVED):
            assert not quoter.quote(regime=regime, **self.ARGS).is_valid


# ---------------------------------------------------------------------------
# Half-spread sanity bound
# ---------------------------------------------------------------------------


class TestHalfSpreadClamp:
    def test_runaway_sigma_b_flags_invalid_without_crashing(self, quoter: GLFTQuoter) -> None:
        """
        405538.28 is the real maximum σ_b stored in data/duckdb/markets.duckdb.
        It arose from two compounding defects: Δt computed 1000× too small
        (dividing by 1e9 assuming datetime64[ns] over datetime64[us] columns)
        and quadratic variation without subsampling. With it, the backtester
        aborted with OverflowError.
        """
        q = quoter.quote(
            market_id=MID,
            mid_p=0.5,
            inventory=0.0,
            tau_years=2.7,
            belief_vol=405538.28,
            regime=NearResolutionRegime.NORMAL,
        )
        assert math.isfinite(q.bid_p) and math.isfinite(q.ask_p)
        assert q.half_spread_X == pytest.approx(MAX_HALF_SPREAD_X)
        assert not q.is_valid
        assert "half_spread_clamped" in q.invalid_reason

    def test_normal_quote_is_not_clipped(self, quoter: GLFTQuoter) -> None:
        q = quoter.quote(
            market_id=MID,
            mid_p=0.5,
            inventory=0.0,
            tau_years=0.02,
            belief_vol=1.5,
            regime=NearResolutionRegime.NORMAL,
        )
        assert q.is_valid
        assert q.half_spread_X < MAX_HALF_SPREAD_X
        assert 0.0 < q.ask_p - q.bid_p < 0.2


# ---------------------------------------------------------------------------
# Tick floor — invariantes de cotizabilidad
# ---------------------------------------------------------------------------


class TestTickFloor:
    """
    The previous clamp produced SUB-TICK spreads near the edges
    (bid=0.0100, ask=0.0169 → 0.0069), which no venue accepts: Kalshi quotes in
    whole cents. On top of that, neither side landed on the grid.
    """

    @pytest.mark.parametrize(
        "reservation_p", [0.0001, 0.0005, 0.0119, 0.05, 0.5, 0.83, 0.95, 0.9994, 0.99999]
    )
    def test_exactly_one_tick_spread_on_grid(self, reservation_p: float) -> None:
        bid, ask, valid, _ = apply_tick_floor(
            bid_p=0.0, ask_p=1.0, reservation_X=logit(reservation_p), tau_years=1.0
        )
        assert valid
        assert ask - bid == pytest.approx(TICK, abs=1e-9)
        assert TICK <= bid < ask <= 1.0 - TICK
        for side in (bid, ask):
            assert side / TICK == pytest.approx(round(side / TICK), abs=1e-9)

    def test_valid_quotes_pass_through_untouched(self) -> None:
        bid, ask, valid, reason = apply_tick_floor(
            bid_p=0.48, ask_p=0.52, reservation_X=0.0, tau_years=1.0
        )
        assert (bid, ask, valid, reason) == (0.48, 0.52, True, "")

    def test_near_resolution_invalidates_rather_than_clamps(self) -> None:
        """Below τ = 1h no spread is invented: quoting simply stops."""
        _, _, valid, reason = apply_tick_floor(
            bid_p=0.0001, ask_p=0.9999, reservation_X=LOGIT_MAX, tau_years=1e-5
        )
        assert not valid
        assert "tick_floor_near_resolution" in reason


class TestQuoteStraddlesReservation:
    """
    The invariant is bid ≤ σ(r̃_X) ≤ ask — the quote straddles the RESERVATION
    RESERVA, no al mid.

    A maker carrying inventory shifts both quotes to the same side of the mid
    in order to flatten; requiring them to straddle the mid rejected exactly
    the inventory skew the model exists to produce.
    """

    @pytest.mark.parametrize("venue", [Venue.KALSHI, Venue.POLYMARKET])
    def test_millesimal_market_is_now_quotable(self, quoter: GLFTQuoter, venue: Venue) -> None:
        """
        With a global 0.01 tick, a market at 0.0035 received a bid of 0.0100 —
        buying at a cent what the book offers at four tenths of one. With each
        venue's real grid (0.0001 / 0.001) it quotes correctly.
        """
        grid = ladder_for_venue(venue)
        q = quoter.quote(
            market_id=MarketId(venue, "T"),
            mid_p=0.0035,
            inventory=0.0,
            tau_years=0.03,
            belief_vol=2.0,
            regime=NearResolutionRegime.NORMAL,
        )
        assert q.is_valid
        assert q.bid_p < 0.0035 < q.ask_p
        for side in (q.bid_p, q.ask_p):
            tick = grid.tick_at(side)
            assert side / tick == pytest.approx(round(side / tick), abs=1e-6)

    def test_coarse_grid_does_not_quote_below_its_minimum(self, quoter: GLFTQuoter) -> None:
        """Manifold uses a one-cent grid: 0.0035 is not quotable there."""
        q = quoter.quote(
            market_id=MarketId(Venue.MANIFOLD, "T"),
            mid_p=0.0035,
            inventory=0.0,
            tau_years=0.03,
            belief_vol=2.0,
            regime=NearResolutionRegime.NORMAL,
        )
        assert not q.is_valid
        assert "straddle_reservation" in q.invalid_reason

    @pytest.mark.parametrize("mid_p", [0.05, 0.25, 0.5, 0.75, 0.95])
    def test_quotes_straddle_the_reservation(self, quoter: GLFTQuoter, mid_p: float) -> None:
        q = quoter.quote(
            market_id=MID,
            mid_p=mid_p,
            inventory=0.0,
            tau_years=0.03,
            belief_vol=2.0,
            regime=NearResolutionRegime.NORMAL,
        )
        assert q.is_valid
        assert q.bid_p <= sigma(q.reservation_X) <= q.ask_p

    def test_inventory_skew_may_move_past_the_mid(self, quoter: GLFTQuoter) -> None:
        """
        With large long inventory BOTH quotes fall below the mid.
        That is correct — the maker wants to sell — and must remain valid.
        """
        q = quoter.quote(
            market_id=MID,
            mid_p=0.5,
            inventory=10.0,
            tau_years=0.5,
            belief_vol=1.0,
            regime=NearResolutionRegime.NORMAL,
        )
        assert q.is_valid
        assert q.ask_p < 0.5
        assert q.bid_p <= sigma(q.reservation_X) <= q.ask_p
