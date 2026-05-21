"""
features/signals/onchain.py
────────────────────────────
OnChainSignal — on-chain activity signal for Polymarket prediction markets.

§8.3 MATH.md v2.1: w_3·OnChain is the on-chain component of μ̂_t.

What this monitors (when implemented):
  - USDC net flow into/out of the Polymarket CTF exchange contract
  - Large-order events from whale wallets (>10k USDC)
  - Open interest delta in the YES/NO outcome token pair

For Kalshi: centralized exchange — on-chain data not applicable, returns 0.0.

Current implementation: stub returning 0.0.
Production path: provide a Polygon RPC URL (web3_provider_url) — the web3.py
dependency is already in pyproject.toml.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Polymarket CTF exchange contract on Polygon
_CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"


class OnChainSignal:
    """
    On-chain activity signal for Polymarket markets.

    Usage:
        signal = OnChainSignal()                                    # stub
        signal = OnChainSignal(web3_provider_url="https://...")     # production
        mu_chain = signal.get(condition_id="0xabc...")
    """

    def __init__(self, web3_provider_url: str | None = None) -> None:
        self._provider_url = web3_provider_url
        if web3_provider_url:
            log.info("OnChainSignal: provider configured — production mode (not yet implemented)")
        else:
            log.debug("OnChainSignal: no provider — returning 0.0 stub")

    def get(
        self,
        token_address: str | None = None,
        condition_id: str | None = None,
        window_blocks: int = 100,
    ) -> float:
        """
        Fetch on-chain activity signal for a Polymarket market.

        Args:
            token_address: ERC-20 YES outcome token address on Polygon
            condition_id:  Polymarket condition ID (hex string)
            window_blocks: lookback window (~2 seconds per block on Polygon)

        Returns:
            Normalized signal in [-1, 1]; 0.0 when unavailable.
        """
        return 0.0

    @property
    def is_available(self) -> bool:
        """True when real on-chain data flows."""
        return False
