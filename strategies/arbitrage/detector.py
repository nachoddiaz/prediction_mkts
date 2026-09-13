"""
strategies/arbitrage/detector.py
─────────────────────────────────
Detección de arbitraje cross-venue — FASE 3, no implementado.

What it will do (MATH.md §7.2):
  Match the same underlying event on Kalshi and Polymarket and detect when
  `c_K + c_P < 1 - Π_X`, where Π_X aggregates taker fees, USDC-USD basis,
  cost of capital, gas and slippage.

Why it is not built yet: it requires simultaneous ingestion from both venues
with semantically matched markets, which is a Phase 3 prerequisite.
"""

from __future__ import annotations

from normalizer.schema import MarketSnapshot


def detect_cross_venue_arbitrage(
    kalshi: MarketSnapshot,
    polymarket: MarketSnapshot,
    frictions: float = 0.018,
) -> None:
    """Aún no implementado — Fase 3. Ver MATH.md §7.2."""
    raise NotImplementedError(
        "Cross-venue arbitrage (Phase 3). Requires simultaneous ingestion from "
        "Kalshi and Polymarket with matched markets. See MATH.md §7.2."
    )
