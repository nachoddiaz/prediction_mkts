"""
features/signals/news.py
─────────────────────────
NewsSignal — normalized news sentiment for a prediction market.

§8.3 MATH.md v2.1: w_2·News is the news component of μ̂_t.

Interface contract:
  get(market_title) → float in [-1, 1]
    +1  = strongly bullish news (YES outcome more likely)
    -1  = strongly bearish news
     0  = neutral or data unavailable

Current implementation: stub returning 0.0.
Production path: subclass NewsSignal and override get() + is_available,
or configure with a real API key (GDELT, NewsAPI, or custom scraper).
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class NewsSignal:
    """
    News sentiment signal for a prediction market.

    Usage:
        signal = NewsSignal()                       # stub mode
        signal = NewsSignal(api_key="...")          # production mode (when implemented)
        mu_news = signal.get("Will BTC exceed 100k before Jan 2026?")
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        if api_key:
            log.info("NewsSignal: API key configured — production mode (not yet implemented)")
        else:
            log.debug("NewsSignal: no API key — returning 0.0 stub")

    def get(self, market_title: str, window_hours: float = 24.0) -> float:
        """
        Fetch news sentiment for a market.

        Args:
            market_title:  market title used as search query
            window_hours:  lookback window for news aggregation in hours

        Returns:
            Normalized sentiment in [-1, 1]; 0.0 when unavailable.
        """
        return 0.0

    @property
    def is_available(self) -> bool:
        """True when real news data flows (not stub 0.0)."""
        return False
