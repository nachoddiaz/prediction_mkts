"""
normalizer/price_grid.py
─────────────────────────
Quotable price grid, per venue and per market.

Why this module exists:
  The system used to assume a global `TICK = 0.01`. That is wrong on both
  target venues, and not by a little:

    - Polymarket declara `minimum_tick_size = 0.001` (CLOB) /
      `orderPriceMinTickSize = 0.001` (Gamma).
    - Kalshi publishes `price_ranges` per market. Across the 200 open markets
      sampled, the main band steps by **0.0010**, with 0.0001 in the tails
      below $0.01 and above $0.99. Some markets do use a single 0.0100 band.


  Consequence of the old value: in a market quoting 0.0030/0.0040 the tick
  floor pushed our bid to 0.0100 — buying at a cent what the book offers at
  four tenths of one. The "quote must straddle the mid" guard stopped that
  from becoming an order, but at the cost of refusing to quote across the
  entire low band, which is where much of Polymarket lives.

Modelo:
  A `PriceLadder` is a list of [start, end) bands with a constant step inside
  each. `tick_at(p)` returns the step in force at p; `floor/ceil` snap a price
  onto the grid. A market without metadata falls back to its venue's default
  grid.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from normalizer.schema import Venue

# Quantisation, so binary arithmetic does not produce 0.30000000000000004.
# Six decimals cover the finest observed step (0.0001) with two orders of
# margin, and match the precision at which both venues return prices.
_DECIMALS = 6
_EPS = 10.0 ** -(_DECIMALS + 1)


@dataclass(frozen=True)
class PriceBand:
    """A [start, end) band with a constant step."""

    start: float
    end: float
    step: float

    def __post_init__(self) -> None:
        if self.step <= 0:
            raise ValueError(f"step must be positive, got {self.step}")
        if self.end <= self.start:
            raise ValueError(f"end must exceed start, got [{self.start}, {self.end}]")

    def contains(self, price: float) -> bool:
        return self.start - _EPS <= price < self.end - _EPS


@dataclass(frozen=True)
class PriceLadder:
    """
    A market's quotable price grid.

    Bands are stored sorted by `start` and should cover (0, 1). A price outside
    every band uses the step of the nearest one.
    """

    bands: tuple[PriceBand, ...]

    def __post_init__(self) -> None:
        if not self.bands:
            raise ValueError("PriceLadder needs at least one band")
        object.__setattr__(self, "bands", tuple(sorted(self.bands, key=lambda b: b.start)))

    @classmethod
    def uniform(cls, step: float) -> PriceLadder:
        """Grid with a constant step across all of (0, 1)."""
        return cls(bands=(PriceBand(start=0.0, end=1.0, step=step),))

    # ------------------------------------------------------------------
    # Consulta
    # ------------------------------------------------------------------

    def tick_at(self, price: float) -> float:
        """Step in force at `price`. Out of range, the nearest band's step."""
        for band in self.bands:
            if band.contains(price):
                return band.step
        return self.bands[0].step if price < self.bands[0].start else self.bands[-1].step

    @property
    def min_tick(self) -> float:
        """Finest step in the grid."""
        return min(b.step for b in self.bands)

    @property
    def min_quotable(self) -> float:
        """Lowest quotable price: one tick above zero."""
        return _quantize(self.tick_at(0.0))

    @property
    def max_quotable(self) -> float:
        """Highest quotable price: one tick below one."""
        top = self.bands[-1]
        return _quantize(1.0 - top.step)

    # ------------------------------------------------------------------
    # Snapping to the grid
    # ------------------------------------------------------------------

    def floor_to_grid(self, price: float) -> float:
        """Largest grid price less than or equal to `price`."""
        return self._snap(price, math.floor)

    def ceil_to_grid(self, price: float) -> float:
        """Smallest grid price greater than or equal to `price`."""
        return self._snap(price, math.ceil)

    def _snap(self, price: float, rounder: Callable[[float], int]) -> float:
        """
        Snap `price` onto the grid with `rounder` (math.floor or math.ceil).

        Rounding the ratio to 9 decimals is not cosmetic: 0.0035/0.0001 evaluates
        to 34.99999999999999 in floating point, so without it `ceil` of a price
        ALREADY on the grid pushed it up by one tick.
        """
        band = self._band_for(price)
        ratio = round((price - band.start) / band.step, 9)
        snapped = band.start + rounder(ratio) * band.step
        return _quantize(min(max(snapped, 0.0), 1.0))

    def _band_for(self, price: float) -> PriceBand:
        for band in self.bands:
            if band.contains(price):
                return band
        return self.bands[0] if price < self.bands[0].start else self.bands[-1]


def _quantize(value: float) -> float:
    return round(value, _DECIMALS)


# ---------------------------------------------------------------------------
# Default ladders per venue
#
# Used when a market carries no grid metadata. They are the fallback, not the
# source of truth: when the venue publishes its grid (Kalshi via `price_ranges`,
# Polymarket via `minimum_tick_size`), the published grid wins.
# ---------------------------------------------------------------------------

# Kalshi: grid observed across the 200 open markets sampled.
KALSHI_DEFAULT_LADDER = PriceLadder(
    bands=(
        PriceBand(start=0.0, end=0.01, step=0.0001),
        PriceBand(start=0.01, end=0.99, step=0.001),
        PriceBand(start=0.99, end=1.0, step=0.0001),
    )
)

# Polymarket: minimum_tick_size = 0.001, uniforme.
POLYMARKET_DEFAULT_LADDER = PriceLadder.uniform(0.001)

# Manifold is an AMM with no book; its connector synthesises a fixed 0.02
# spread. The one-cent grid is the convention that connector already used.
MANIFOLD_DEFAULT_LADDER = PriceLadder.uniform(0.01)

DEFAULT_LADDER_BY_VENUE: dict[Venue, PriceLadder] = {
    Venue.KALSHI: KALSHI_DEFAULT_LADDER,
    Venue.POLYMARKET: POLYMARKET_DEFAULT_LADDER,
    Venue.MANIFOLD: MANIFOLD_DEFAULT_LADDER,
}


def ladder_for_venue(venue: Venue) -> PriceLadder:
    """Venue default ladder. Falls back to one cent for an unknown venue."""
    return DEFAULT_LADDER_BY_VENUE.get(venue, PriceLadder.uniform(0.01))


# ---------------------------------------------------------------------------
# Parsing venue metadata
# ---------------------------------------------------------------------------


def parse_kalshi_price_ranges(raw: Any) -> PriceLadder | None:  # noqa: ANN401
    """
    Build a PriceLadder from Kalshi's `price_ranges` field.

    Formato:
        [{"start": "0.0000", "end": "0.0100", "step": "0.0001"}, ...]

    All three values arrive as decimal strings. Returns None when the field is
    absent or unparseable — the caller then uses the default ladder.
    """
    if not isinstance(raw, list) or not raw:
        return None

    bands: list[PriceBand] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            bands.append(
                PriceBand(
                    start=float(entry["start"]),
                    end=float(entry["end"]),
                    step=float(entry["step"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    return PriceLadder(bands=tuple(bands)) if bands else None


def parse_polymarket_tick_size(raw: Any) -> PriceLadder | None:  # noqa: ANN401
    """
    Build a PriceLadder from `minimum_tick_size` (CLOB) or
    `orderPriceMinTickSize` (Gamma). Both describe a uniform step.
    """
    if raw is None:
        return None
    try:
        step = float(raw)
    except (TypeError, ValueError):
        return None
    return PriceLadder.uniform(step) if step > 0 else None
