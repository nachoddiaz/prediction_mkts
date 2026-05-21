"""
execution/risk/circuit_breaker.py
──────────────────────────────────
Monitoreo de condiciones extremas y desactivación temporal del quoting (circuit breaker).
"""

from __future__ import annotations

import logging

from features.resolution import NearResolutionRegime

log = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Controla el apagado de emergencia (halt) del quoting del sistema.

    Condiciones de disparo:
      1. Régimen de near-resolution crítico o resuelto (HALT o RESOLVED).
      2. Superación del límite de pérdidas diarias acumuladas (max_daily_loss).
    """

    def __init__(self, max_daily_loss: float) -> None:
        self.max_daily_loss = max_daily_loss
        self._is_tripped: bool = False
        self._trip_reason: str = ""

    @property
    def is_tripped(self) -> bool:
        """Devuelve True si el circuit breaker está activado (sistema detenido)."""
        return self._is_tripped

    @property
    def trip_reason(self) -> str:
        """Motivo del último disparo."""
        return self._trip_reason

    def check(self, regime: NearResolutionRegime, daily_loss: float) -> bool:
        """
        Evalúa el estado del mercado y las pérdidas y dispara el breaker si es necesario.

        Args:
            regime:     El NearResolutionRegime actual.
            daily_loss: Pérdida acumulada en el día (número positivo para pérdidas).

        Returns:
            True si está disparado/activo (halt), False en caso contrario.
        """
        # 1. Chequeo de régimen
        if regime in (NearResolutionRegime.HALT, NearResolutionRegime.RESOLVED):
            self._is_tripped = True
            self._trip_reason = f"Regime near-resolution limit reached: {regime.value}"
            log.error(f"circuit_breaker_tripped: reason={self._trip_reason}")
            return True

        # 2. Chequeo de pérdida diaria
        if daily_loss >= self.max_daily_loss:
            self._is_tripped = True
            self._trip_reason = (
                f"Daily loss limit exceeded: {daily_loss:.2f} >= {self.max_daily_loss:.2f}"
            )
            log.error(f"circuit_breaker_tripped: reason={self._trip_reason}")
            return True

        return self._is_tripped

    def reset(self) -> None:
        """Restablece el circuit breaker a su estado inicial operacional."""
        if self._is_tripped:
            log.info("circuit_breaker_reset")
            self._is_tripped = False
            self._trip_reason = ""
