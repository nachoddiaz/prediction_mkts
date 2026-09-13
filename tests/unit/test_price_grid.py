"""
tests/unit/test_price_grid.py
Quotable price grid, per venue and per market.

The reference values are not invented: they come from sampling the APIs.
  - Kalshi: `price_ranges` steps by 0.0001 below $0.01, 0.001 between $0.01 and
    $0.99, and 0.0001 above — identical across the 200 open markets sampled.
    Some markets use a single 0.01 band.
  - Polymarket: minimum_tick_size = 0.001 (CLOB) / orderPriceMinTickSize = 0.001
    (Gamma), uniforme.
"""

from __future__ import annotations

import pytest

from normalizer.price_grid import (
    KALSHI_DEFAULT_LADDER,
    POLYMARKET_DEFAULT_LADDER,
    PriceBand,
    PriceLadder,
    ladder_for_venue,
    parse_kalshi_price_ranges,
    parse_polymarket_tick_size,
)
from normalizer.schema import Venue

KALSHI_RANGES_RAW = [
    {"end": "0.0100", "start": "0.0000", "step": "0.0001"},
    {"end": "0.9900", "start": "0.0100", "step": "0.0010"},
    {"end": "1.0000", "start": "0.9900", "step": "0.0001"},
]


class TestPriceBand:
    def test_rejects_non_positive_step(self) -> None:
        with pytest.raises(ValueError, match="step must be positive"):
            PriceBand(start=0.0, end=1.0, step=0.0)

    def test_rejects_inverted_range(self) -> None:
        with pytest.raises(ValueError, match="end must exceed start"):
            PriceBand(start=0.5, end=0.5, step=0.01)


class TestTickAt:
    @pytest.mark.parametrize(
        ("price", "expected"),
        [
            (0.0005, 0.0001),
            (0.0099, 0.0001),
            (0.01, 0.001),
            (0.5, 0.001),
            (0.9899, 0.001),
            (0.99, 0.0001),
            (0.9999, 0.0001),
        ],
    )
    def test_kalshi_changes_step_in_the_tails(self, price: float, expected: float) -> None:
        assert KALSHI_DEFAULT_LADDER.tick_at(price) == pytest.approx(expected)

    def test_polymarket_is_uniform(self) -> None:
        for p in (0.001, 0.05, 0.5, 0.95, 0.999):
            assert POLYMARKET_DEFAULT_LADDER.tick_at(p) == pytest.approx(0.001)

    def test_bands_sort_themselves(self) -> None:
        unsorted_ladder = PriceLadder(
            bands=(
                PriceBand(start=0.01, end=1.0, step=0.001),
                PriceBand(start=0.0, end=0.01, step=0.0001),
            )
        )
        assert unsorted_ladder.bands[0].start == 0.0

    def test_empty_ladder_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="at least one band"):
            PriceLadder(bands=())


class TestSnapping:
    """
    Snapping invariants. Rounding the ratio is not cosmetic: 0.0035/0.0001
    evaluates to 34.99999999999999 in floating point, so without it `ceil` of
    a price already on the grid pushed it up by one tick.
    """

    LADDERS = [KALSHI_DEFAULT_LADDER, POLYMARKET_DEFAULT_LADDER, PriceLadder.uniform(0.01)]

    @pytest.mark.parametrize("ladder", LADDERS)
    def test_floor_is_below_and_ceil_above(self, ladder: PriceLadder) -> None:
        for i in range(1, 1000):
            p = i / 1000
            assert ladder.floor_to_grid(p) <= p + 1e-9
            assert ladder.ceil_to_grid(p) >= p - 1e-9

    @pytest.mark.parametrize("ladder", LADDERS)
    def test_gap_never_exceeds_one_tick(self, ladder: PriceLadder) -> None:
        for i in range(1, 1000):
            p = i / 1000
            gap = ladder.ceil_to_grid(p) - ladder.floor_to_grid(p)
            assert gap <= ladder.tick_at(p) + 1e-9

    @pytest.mark.parametrize("ladder", LADDERS)
    def test_idempotent_on_prices_already_on_grid(self, ladder: PriceLadder) -> None:
        for i in range(1, 200):
            on_grid = ladder.floor_to_grid(i / 200)
            assert ladder.floor_to_grid(on_grid) == pytest.approx(on_grid, abs=1e-9)
            assert ladder.ceil_to_grid(on_grid) == pytest.approx(on_grid, abs=1e-9)

    def test_produces_no_binary_noise(self) -> None:
        """0.30000000000000004 y familia."""
        grid = PriceLadder.uniform(0.01)
        assert grid.floor_to_grid(0.3) == 0.3
        assert grid.ceil_to_grid(0.7) == 0.7

    def test_kalshi_anchors_in_the_right_band(self) -> None:
        # Banda fina: paso 0.0001
        assert KALSHI_DEFAULT_LADDER.floor_to_grid(0.00317) == pytest.approx(0.0031)
        assert KALSHI_DEFAULT_LADDER.ceil_to_grid(0.00317) == pytest.approx(0.0032)
        # Banda principal: paso 0.001
        assert KALSHI_DEFAULT_LADDER.floor_to_grid(0.4567) == pytest.approx(0.456)
        assert KALSHI_DEFAULT_LADDER.ceil_to_grid(0.4567) == pytest.approx(0.457)


class TestRangoCotizable:
    def test_kalshi_reaches_one_ten_thousandth(self) -> None:
        assert KALSHI_DEFAULT_LADDER.min_quotable == pytest.approx(0.0001)
        assert KALSHI_DEFAULT_LADDER.max_quotable == pytest.approx(0.9999)

    def test_polymarket_reaches_one_thousandth(self) -> None:
        assert POLYMARKET_DEFAULT_LADDER.min_quotable == pytest.approx(0.001)
        assert POLYMARKET_DEFAULT_LADDER.max_quotable == pytest.approx(0.999)

    def test_the_fixed_cent_grid_was_ten_times_coarser(self) -> None:
        """
        Reason for the change: on a one-cent grid the minimum quotable price
        is 0.01 — ten times Polymarket's and a hundred times Kalshi's low band.
        A market at 0.0035 fell outside the quotable range entirely.
        """
        cent_grid = PriceLadder.uniform(0.01)
        assert cent_grid.min_quotable == pytest.approx(0.01)
        assert cent_grid.min_quotable > POLYMARKET_DEFAULT_LADDER.min_quotable
        assert cent_grid.min_quotable > KALSHI_DEFAULT_LADDER.min_quotable


class TestParseoDeMetadatos:
    def test_kalshi_price_ranges(self) -> None:
        ladder = parse_kalshi_price_ranges(KALSHI_RANGES_RAW)
        assert ladder is not None
        assert len(ladder.bands) == 3
        assert ladder.tick_at(0.5) == pytest.approx(0.001)
        assert ladder.tick_at(0.005) == pytest.approx(0.0001)

    def test_kalshi_single_one_cent_band(self) -> None:
        """Some Kalshi markets do use a single 0.01 band."""
        ladder = parse_kalshi_price_ranges([{"end": "1.0000", "start": "0.0000", "step": "0.0100"}])
        assert ladder is not None
        assert ladder.tick_at(0.5) == pytest.approx(0.01)

    @pytest.mark.parametrize("raw", [None, [], "not a list", [{"malformed": 1}], 42])
    def test_kalshi_invalid_input_returns_none(self, raw: object) -> None:
        assert parse_kalshi_price_ranges(raw) is None

    def test_polymarket_tick_size(self) -> None:
        ladder = parse_polymarket_tick_size("0.001")
        assert ladder is not None
        assert ladder.tick_at(0.5) == pytest.approx(0.001)

    @pytest.mark.parametrize("raw", [None, "", "abc", 0, -1])
    def test_polymarket_invalid_input_returns_none(self, raw: object) -> None:
        assert parse_polymarket_tick_size(raw) is None


class TestLadderPorVenue:
    @pytest.mark.parametrize(
        ("venue", "tick_medio"),
        [(Venue.KALSHI, 0.001), (Venue.POLYMARKET, 0.001), (Venue.MANIFOLD, 0.01)],
    )
    def test_defaults(self, venue: Venue, tick_medio: float) -> None:
        assert ladder_for_venue(venue).tick_at(0.5) == pytest.approx(tick_medio)
