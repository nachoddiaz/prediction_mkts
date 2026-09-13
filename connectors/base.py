"""
connectors/base.py
───────────────────
Common abstract interface for every connector.

Why an abstract base class and not a typing.Protocol:
  Protocol is more flexible but does not enforce implementation. An ABC raises
  NotImplementedError at runtime if a subclass forgets a method — a fast,
  clear failure during development.

Why the HTTP helpers (_get, _post) live here rather than in each connector:
  Retry, timeout and logging are identical across every connector.
  Implementing them once in the base avoids duplication and guarantees
  consistent behaviour under network errors.

Why the subscribe callbacks are Callables and not queues:
  Callbacks let the caller decide what to do with each event — write it to
  DuckDB, compute features, log it, feed a strategy, or all of them. With a
  queue the connector would need to know who consumes the data; callbacks
  keep it entirely agnostic about the destination.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

import aiohttp

from normalizer.schema import Market, MarketSnapshot, MarketStatus, Tick

log = logging.getLogger(__name__)

# An explicit and TRUTHFUL User-Agent on every request.
#
# Why not a browser User-Agent:
#   "Mozilla/5.0 ... Chrome/124" was tried, and Cloudflare ended up returning
#   403 (an HTML challenge) from the second request against clob.polymarket.com
#   onwards. The block turned out to be transient — repeating the experiment
#   later, all three UA variants passed — but the operational conclusion
#   stands: a browser UA with no cookies and no JS execution is a bot
#   signature, and it only attracts anti-bot heuristics without buying
#   anything in return.
#
#   Our own identifier passes just as well on both venues (verified, 4/4
#   requests returning 200 on Kalshi and Polymarket) and additionally lets the
#   venue identify us if rate limits ever need discussing.
USER_AGENT = "prediction-market-system/0.1.0 (+https://github.com/nachoddiaz)"

# ---------------------------------------------------------------------------
# Callback types
#
# Why Awaitable[None] and not plain None:
#   Callbacks are invoked from an async loop. If a callback performs blocking
#   I/O (writing to DuckDB, computing features) it needs to be awaitable.
#   Awaitable[None] accepts both async functions and coroutines.
# ---------------------------------------------------------------------------

TickCallback = Callable[[Tick], Awaitable[None]]
SnapshotCallback = Callable[[MarketSnapshot], Awaitable[None]]

# ---------------------------------------------------------------------------
# Retry constants
#
# Why exponential backoff:
#   If the Kalshi API goes down momentarily, retrying immediately saturates
#   the server and can get the IP banned. With exponential backoff the first
#   retry waits 1s, the second 2s, the third 4s — giving the server time to
#   recover.
# ---------------------------------------------------------------------------

# 30 s and not 10: Kalshi's /series endpoint returns ~16 MB.
DEFAULT_TIMEOUT_SECONDS: int = 30

# How often the status of tracked markets is re-queried.
#
# This loop did NOT exist: run()'s docstring promised a MARKET_REFRESH_INTERVAL
# that was never defined anywhere. get_markets() ran once at startup and the
# list froze, so nobody ever re-checked a market's `status` and
# `resolved_value` was NEVER written. Consequence: however many days ingestion
# ran, the resolved-market counter stayed at 0 — and with it the Brier
# calibration that depends on those markets.
MARKET_REFRESH_INTERVAL: int = 900

# Minimum fraction of tracked markets that must still be open. Below it the
# basket counts as exhausted and rediscovery is triggered. 0.5 avoids churning
# on a single resolution while still reacting before there is nothing to measure.
MIN_ALIVE_FRACTION: float = 0.5
MAX_RETRIES: int = 3
RETRY_BACKOFF_BASE: float = 1.0  # seconds; doubles on each retry


class BaseConnector(ABC):
    """
    Base class for every connector in the system.

    Subclasses must implement:
      - get_markets()    → discover active markets
      - get_snapshot()   → current state of one market
      - subscribe()      → real-time stream
      - _build_headers() → venue-specific authentication

    The base class provides:
      - _get()  → HTTP GET with retry and logging
      - _post() → HTTP POST with retry and logging
      - run()   → main loop: discover → initialise → subscribe
    """

    def __init__(
        self,
        on_tick: TickCallback,
        on_snapshot: SnapshotCallback,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        refresh_interval: int = MARKET_REFRESH_INTERVAL,
        max_tau_days: float = 0.0,
        require_two_sided_book: bool = False,
    ) -> None:
        """
        Args:
            on_tick:      callback invoked for each Tick from the stream
            on_snapshot:  callback invoked for each MarketSnapshot
            timeout:      HTTP request timeout, in seconds
            refresh_interval: how often, in seconds, the status of tracked
                          markets is re-queried (this is what detects resolution)
            max_tau_days: drop markets resolving beyond this horizon.
                          0 disables the filter.
            require_two_sided_book: drop markets without both a bid and an ask.
        """
        self._on_tick = on_tick
        self._on_snapshot = on_snapshot
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None
        self._refresh_interval = refresh_interval
        self._max_tau_days = max_tau_days
        self._require_two_sided_book = require_two_sided_book
        self._market_status: dict[str, MarketStatus] = {}

        # Set when the basket of tracked markets has gone stale (too many
        # resolved) and rediscovery is needed.
        self._restart_stream: asyncio.Event = asyncio.Event()

    # ------------------------------------------------------------------
    # Quality filter — which markets are worth the storage
    # ------------------------------------------------------------------

    def is_worth_ingesting(self, market: Market) -> tuple[bool, str]:
        """
        Decide whether a market is worth admitting into the database.

        Why filter at the SOURCE rather than afterwards:
          A row once written already costs disk, index and query time. In the
          development database, 31,000 of 31,100 ticks were historical backfill
          from Manifold markets resolving between 117 days and 74 YEARS out: we
          will never see their outcome, so they serve neither Brier nor signal
          validation, and yet they dominated every aggregate.

        Returns:
            (accept, reason_for_rejection)
        """
        if market.status != MarketStatus.OPEN:
            return False, "not_open"

        if self._max_tau_days > 0:
            tau_days = market.resolution.tau * 365.25
            if tau_days > self._max_tau_days:
                return False, f"tau_too_long ({tau_days:.0f}d > {self._max_tau_days:.0f}d)"
            if tau_days <= 0:
                return False, "already_expired"

        return True, ""

    # ------------------------------------------------------------------
    # Abstract methods — implemented by each connector
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_markets(self) -> list[Market]:
        """
        Fetch the list of active markets from the REST API.

        Returns:
            List of canonical domain Market objects.
            Empty list when there are no markets or on error.
        """
        ...

    @abstractmethod
    async def get_snapshot(self, market_id: str) -> MarketSnapshot | None:
        """
        Fetch the complete current state of one market.

        Args:
            market_id: canonical "venue:raw_id" string

        Returns:
            A MarketSnapshot with market + orderbook + last_tick, or
            None when the market does not exist or on error.
        """
        ...

    @abstractmethod
    async def subscribe(self, market_ids: list[str]) -> None:
        """
        Subscribe to the real-time stream for the given markets.

        For each WebSocket event it calls:
          self._on_tick(tick)           → trades y quote updates
          self._on_snapshot(snapshot)   → orderbook updates completos

        This method does not return until the connection closes.
        The reconnection loop lives in run().

        Args:
            market_ids: list of canonical "venue:raw_id" strings
        """
        ...

    @abstractmethod
    def _build_headers(self) -> dict[str, str]:
        """
        Build the authentication headers for this venue.

        Kalshi:     KALSHI-ACCESS-KEY + KALSHI-ACCESS-SIGNATURE (RSA)
        Polymarket: no auth required for reads
        Manifold:   no auth required

        Returns:
            Dict of HTTP headers ready to attach to every request.
        """
        ...

    async def fetch_market(self, market_id: str) -> Market | None:
        """
        Re-query the metadata of ONE already-known market.

        Different from get_snapshot(): a resolved market has no book left, so
        get_snapshot() returns None and the resolution would never be seen.
        This asks only for metadata, which is what changes on resolution.

        Default implementation: derive it from the snapshot. Each venue should
        override this with its own metadata endpoint.
        """
        snapshot = await self.get_snapshot(market_id)
        return snapshot.market if snapshot else None

    async def _refresh_loop(self, market_ids: list[str], interval: int) -> None:
        """
        Periodically re-query the status of tracked markets and emit those that
        changed, so the writer persists status and resolved_value.

        This is the only path by which a resolution reaches the database.
        """
        while True:
            await asyncio.sleep(interval)

            changed = 0
            for market_id in market_ids:
                try:
                    market = await self.fetch_market(market_id)
                except Exception as e:
                    log.debug("refresh failed for %s: %s", market_id, e)
                    continue

                if market is None:
                    continue

                previous = self._market_status.get(market_id)
                if previous == market.status and not market.resolution.is_resolved():
                    continue

                self._market_status[market_id] = market.status
                changed += 1
                await self._on_snapshot(MarketSnapshot(market=market))

                if market.resolution.is_resolved():
                    log.info(
                        "market_resolved: %s outcome=%s",
                        market_id,
                        market.resolution.resolved_value,
                    )

            if changed:
                log.info(
                    "%s: refreshed %d/%d markets with status changes",
                    self.__class__.__name__,
                    changed,
                    len(market_ids),
                )

            # If most tracked markets are no longer tradeable, the basket is
            # exhausted and rediscovery is needed.
            #
            # Without this the list stayed frozen from startup. It happened for
            # real: Kalshi discovered 93 sports markets, all resolved within
            # hours, and polling then spent 17 HOURS querying dead markets —
            # producing no ticks and burning one request every 30 s — because
            # _poll_rest() is a `while True` that never returns and rediscovery
            # only happened when the stream ended.
            alive = sum(
                1
                for mid in market_ids
                if self._market_status.get(mid, MarketStatus.OPEN) == MarketStatus.OPEN
            )
            if alive < max(1, len(market_ids) * MIN_ALIVE_FRACTION):
                log.info(
                    "%s: only %d/%d markets still open — rediscovering",
                    self.__class__.__name__,
                    alive,
                    len(market_ids),
                )
                self._restart_stream.set()
                return

    # ------------------------------------------------------------------
    # Main loop — identical for every connector
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Connector main loop:
          1. Open the HTTP session
          2. Discover active markets
          3. Take an initial snapshot of each market
          4. Subscribe to the WebSocket stream
          5. On disconnect → wait and reconnect from step 4

        Why reconnect from step 4 and not from step 2:
          The set of active markets does not change on every reconnect.
          Re-fetching the whole list each time would generate needless load.
          carga innecesaria. Solo re-suscribimos el WebSocket.
          The market list is refreshed every MARKET_REFRESH_INTERVAL.
        """
        # The User-Agent goes first so a venue can override it if needed, while
        # no venue ends up without one by omission.
        session_headers = {"User-Agent": USER_AGENT, **self._build_headers()}

        async with aiohttp.ClientSession(
            timeout=self._timeout,
            headers=session_headers,
        ) as session:
            self._session = session

            log.info("%s connector starting", self.__class__.__name__)

            # Outer loop: when the stream ends or there are no markets left to
            # follow, rediscover rather than declaring the connector dead.
            while True:
                await self._discover_and_stream()

    async def _discover_and_stream(self) -> None:
        """One full round: discover, filter, initial snapshot and stream."""
        # Step 1: discover active markets, retrying while there are none.
        #
        # This used to `return`, killing the connector for good. With the
        # quality filter that goes from a rare case to a routine one: if no
        # market on a venue passed the filter, the venue stayed disconnected
        # for the rest of the run instead of looking again later.
        self._restart_stream.clear()

        markets: list[Market] = []
        while not markets:
            markets = await self.get_markets()
            if not markets:
                log.warning(
                    "%s: no active markets found — retrying in %ds",
                    self.__class__.__name__,
                    self._refresh_interval,
                )
                await asyncio.sleep(self._refresh_interval)

        # Quality filter BEFORE touching the database: what fails here never
        # costs a row.
        kept: list[Market] = []
        rejected: dict[str, int] = {}
        for market in markets:
            ok, reason = self.is_worth_ingesting(market)
            if ok:
                kept.append(market)
            else:
                key = reason.split(" ")[0]
                rejected[key] = rejected.get(key, 0) + 1

        market_ids = [str(m.market_id) for m in kept]
        for market in kept:
            self._market_status[str(market.market_id)] = market.status

        log.info(
            "%s: tracking %d markets (%d discarded: %s)",
            self.__class__.__name__,
            len(market_ids),
            len(markets) - len(market_ids),
            rejected or "none",
        )

        if not market_ids:
            log.warning(
                "%s: no markets passed the quality filter — retrying in %ds",
                self.__class__.__name__,
                self._refresh_interval,
            )
            await asyncio.sleep(self._refresh_interval)
            return

        # Step 2: initial snapshot per market, plus backfill where supported
        for market_id in market_ids:
            snapshot = await self.get_snapshot(market_id)
            if snapshot is not None:
                await self._on_snapshot(snapshot)
                log.debug("Initial snapshot: %s", market_id)
                # Backfill where the venue supports it
                if hasattr(self, "backfill_market"):
                    try:
                        await self.backfill_market(market_id)
                    except Exception as e:
                        log.warning("Failed to backfill market %s: %s", market_id, e)

        # Step 3: background metadata refresh loop.
        # It runs alongside the stream: this is what detects resolutions.
        refresh_task = asyncio.create_task(
            self._refresh_loop(market_ids, self._refresh_interval),
            name=f"{self.__class__.__name__}-refresh",
        )

        # Step 4: subscribe, with automatic reconnection
        retry = 0
        while True:
            try:
                log.info(
                    "%s: subscribing to %d markets (attempt %d)",
                    self.__class__.__name__,
                    len(market_ids),
                    retry + 1,
                )
                # The stream races the rediscovery signal:
                # whichever finishes first wins. Without this race, a stream
                # that never returns (REST polling, a stable WebSocket) blocked
                # market-list refresh forever.
                stream_task = asyncio.create_task(self.subscribe(market_ids))
                restart_task = asyncio.create_task(self._restart_stream.wait())

                done, pending = await asyncio.wait(
                    [stream_task, restart_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

                if stream_task in done:
                    error = stream_task.exception()
                    if error is not None:
                        raise error

                log.info("%s: stream ended — rediscovering", self.__class__.__name__)
                refresh_task.cancel()
                break

            except Exception as e:
                retry += 1
                wait = RETRY_BACKOFF_BASE * (2 ** min(retry, 6))
                log.warning(
                    "%s: stream error (attempt %d): %s — retrying in %.1fs",
                    self.__class__.__name__,
                    retry,
                    e,
                    wait,
                )
                await asyncio.sleep(wait)

    # ------------------------------------------------------------------
    # HTTP helpers — retry + logging compartidos
    # ------------------------------------------------------------------

    async def _get(
        self,
        url: str,
        params: dict | None = None,
    ) -> dict | list | None:
        """
        HTTP GET with exponential retry.

        Why None is returned rather than raising:
          A network error in one connector must not bring the whole system
          down. The caller decides whether None is acceptable or whether to
          retry. Logging here gives visibility without propagating the error.

        Args:
            url:    full endpoint URL
            params: optional query params

        Returns:
            Parsed JSON as a dict or list, None if every retry fails.
        """
        if self._session is None:
            log.error("_get called before session was opened")
            return None

        for attempt in range(MAX_RETRIES):
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 200:
                        return await resp.json()

                    # 429 = rate limit — wait longer
                    if resp.status == 429:
                        wait = RETRY_BACKOFF_BASE * (2**attempt) * 2
                        log.warning("Rate limited on GET %s — waiting %.1fs", url, wait)
                        await asyncio.sleep(wait)
                        continue

                    # 4xx client errors — retrying makes no sense
                    if 400 <= resp.status < 500:
                        log.error("Client error %d on GET %s", resp.status, url)
                        return None

                    # 5xx server errors — reintentar
                    log.warning(
                        "Server error %d on GET %s (attempt %d/%d)",
                        resp.status,
                        url,
                        attempt + 1,
                        MAX_RETRIES,
                    )

            # TimeoutError va aparte de ClientError: asyncio.TimeoutError NO hereda
            # of aiohttp.ClientError, so without catching it a slow endpoint was
            # not retried — it escaped _get(), propagated through
            # asyncio.gather() and took down the WHOLE ingestion. That is what
            # happened with Kalshi's /series endpoint: 16.6 MB, over a 10 s timeout.
            except (aiohttp.ClientError, TimeoutError) as e:
                log.warning(
                    "Network error on GET %s (attempt %d/%d): %s",
                    url,
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                )

            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF_BASE * (2**attempt)
                await asyncio.sleep(wait)

        log.error("GET %s failed after %d attempts", url, MAX_RETRIES)
        return None

    async def _post(
        self,
        url: str,
        body: dict,
    ) -> dict | None:
        """
        HTTP POST with exponential retry.
        Same logic as _get — see the comments there.
        """
        if self._session is None:
            log.error("_post called before session was opened")
            return None

        for attempt in range(MAX_RETRIES):
            try:
                async with self._session.post(url, json=body) as resp:
                    if resp.status in (200, 201):
                        return await resp.json()

                    if resp.status == 429:
                        wait = RETRY_BACKOFF_BASE * (2**attempt) * 2
                        await asyncio.sleep(wait)
                        continue

                    if 400 <= resp.status < 500:
                        log.error("Client error %d on POST %s", resp.status, url)
                        return None

                    log.warning(
                        "Server error %d on POST %s (attempt %d/%d)",
                        resp.status,
                        url,
                        attempt + 1,
                        MAX_RETRIES,
                    )

            except (aiohttp.ClientError, TimeoutError) as e:
                log.warning(
                    "Network error on POST %s (attempt %d/%d): %s",
                    url,
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                )

            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_BACKOFF_BASE * (2**attempt))

        log.error("POST %s failed after %d attempts", url, MAX_RETRIES)
        return None

    # ------------------------------------------------------------------
    # Context manager — for use with `async with`
    # ------------------------------------------------------------------

    async def __aenter__(self) -> BaseConnector:
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
