"""
strategies/arbitrage/sizing.py
───────────────────────────────
Fractional Kelly sizing with frictions — PHASE 3, not implemented.

The formula is derived in closed form in MATH.md §7.1 eq (8.1):

    f* = φ_K · ( [p + r(1-2p)] - c - f_t - ρ_c·τ - b_U ) / ( c(1-c) )

with φ_K ∈ [0.25, 0.5]. What is missing is not the mathematics but the inputs:
a calibrated `p` (which needs resolved markets, Phase 3) and `r` per category.
"""

from __future__ import annotations


def kelly_fraction(*_args: object, **_kwargs: object) -> float:
    """Aún no implementado — Fase 3. Ver MATH.md §7.1 ec (8.1)."""
    raise NotImplementedError(
        "Kelly sizing (Phase 3). The formula is in MATH.md §7.1; it still needs a "
        "calibrated p and a per-category r, both of which need resolved markets."
    )
