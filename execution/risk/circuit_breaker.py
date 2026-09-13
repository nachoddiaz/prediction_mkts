"""
execution/risk/circuit_breaker.py
──────────────────────────────────
Extreme-condition monitoring and temporary quoting shutdown (circuit breaker).
"""

from __future__ import annotations

import logging

from features.resolution import NearResolutionRegime

log = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Controls the system's emergency quoting halt.

    Trip conditions:
      1. A critical or resolved near-resolution regime (HALT or RESOLVED).
      2. Cumulative daily loss exceeding max_daily_loss.
    """

    def __init__(self, max_daily_loss: float) -> None:
        self.max_daily_loss = max_daily_loss
        self._is_tripped: bool = False
        self._trip_reason: str = ""

    @property
    def is_tripped(self) -> bool:
        """True when the breaker has tripped (system halted)."""
        return self._is_tripped

    @property
    def trip_reason(self) -> str:
        """Reason for the most recent trip."""
        return self._trip_reason

    def check(self, regime: NearResolutionRegime, daily_loss: float) -> bool:
        """
        Evaluate market state and losses, tripping the breaker when needed.

        Args:
            regime:     the current NearResolutionRegime.
            daily_loss: cumulative loss for the day (positive means a loss).

        Returns:
            True when tripped (halted), False otherwise.
        """
        # 1. Regime check
        if regime in (NearResolutionRegime.HALT, NearResolutionRegime.RESOLVED):
            self._is_tripped = True
            self._trip_reason = f"Regime near-resolution limit reached: {regime.value}"
            log.error(f"circuit_breaker_tripped: reason={self._trip_reason}")
            return True

        # 2. Daily-loss check
        if daily_loss >= self.max_daily_loss:
            self._is_tripped = True
            self._trip_reason = (
                f"Daily loss limit exceeded: {daily_loss:.2f} >= {self.max_daily_loss:.2f}"
            )
            log.error(f"circuit_breaker_tripped: reason={self._trip_reason}")
            return True

        return self._is_tripped

    def reset(self) -> None:
        """Restore the circuit breaker to its initial operating state."""
        if self._is_tripped:
            log.info("circuit_breaker_reset")
            self._is_tripped = False
            self._trip_reason = ""
