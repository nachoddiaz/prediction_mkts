"""
strategies/arbitrage/cross_venue.py
────────────────────────────────────
Execution of the two-leg YES(Kalshi) + NO(Polymarket) trade — PHASE 3, not
implemented.

What it will do (MATH.md §7.2): coordinate the two orders so that the risk of
being left with a single leg is bounded, and unwind the position if the second
leg does not complete within a time window.
"""

from __future__ import annotations


def execute_cross_venue_pair(*_args: object, **_kwargs: object) -> None:
    """Aún no implementado — Fase 3. Ver MATH.md §7.2."""
    raise NotImplementedError(
        "Cross-venue execution (Phase 3). Depends on execution/live/. See MATH.md §7.2."
    )
