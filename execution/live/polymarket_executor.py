"""
execution/live/polymarket_executor.py
──────────────────────────────────────
Live executor against the Polymarket CLOB — PHASE 4, not implemented.

Requires pyproject's `live` extra (py-clob-client, eth-account, web3):
orders are signed with EIP-712 and settle on-chain on Polygon.
"""

from __future__ import annotations

from execution.order import Order


class PolymarketExecutor:
    """Aún no implementado — Fase 4."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise NotImplementedError(
            "Live execution on Polymarket (Phase 4). Install the `live` extra "
            "y configurar POLYMARKET_PRIVATE_KEY / POLYMARKET_PROXY_ADDRESS."
        )

    async def submit(self, order: Order) -> None:
        raise NotImplementedError
