"""
execution/live/kalshi_executor.py
──────────────────────────────────
Live executor against Kalshi — PHASE 4, not implemented.

Blocking prerequisite: `connectors/kalshi.py::_sign()` builds the
KALSHI-ACCESS-{KEY,TIMESTAMP,SIGNATURE} headers but never attaches them to the
request (the session only carries static headers). Without that, any
authenticated endpoint returns 401.

While `PAPER_TRADING=true` in .env, the router must keep using
`execution/paper/engine.py`.
"""

from __future__ import annotations

from execution.order import Order


class KalshiExecutor:
    """Aún no implementado — Fase 4."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise NotImplementedError(
            "Live execution on Kalshi (Phase 4). Requires per-request RSA signing "
            "in connectors/kalshi.py and an order reconciliation flow."
        )

    async def submit(self, order: Order) -> None:
        raise NotImplementedError
