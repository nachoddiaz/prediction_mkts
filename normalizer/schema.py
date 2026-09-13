# We need a schema file that translates those different formats into a common one
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from normalizer.price_grid import PriceLadder

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import NewType

# creation of variables that prevent inverted arguments
Price = NewType("Price", float)
Size = NewType("Size", float)


# Setting a common time zone
def utcnow() -> datetime:
    return datetime.now(tz=UTC)


# Creating an enum for the different venues
class Venue(str, Enum):
    KALSHI = "kalshi"
    POLYMARKET = "polymarket"
    MANIFOLD = "manifold"

    def __str__(self) -> str:
        return self.value


class MarketStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"  # trading stopped, not yet resolved
    RESOLVED = "resolved"  # resultado conocido


class TickType(str, Enum):
    QUOTE = "quote"  # a mid-price update from the order book
    TRADE = "trade"  # fill real


class MarketCategory(str, Enum):
    CRYPTO = "crypto"
    POLITICS = "politics"
    SPORTS = "sports"
    ECONOMICS = "economics"
    SCIENCE = "science"
    OTHER = "other"


class Side(str, Enum):
    YES = "yes"
    NO = "no"


# With this Id we can identify the market:raw_id -> this is what we are going to store
@dataclass(frozen=True)
class MarketId:
    venue: Venue
    raw_id: str

    def __str__(self) -> str:
        return f"{self.venue}:{self.raw_id}"

    # Given a MarketId we can get the venue and the market
    @classmethod
    def from_str(cls, s: str) -> MarketId:
        venue_str, raw_id = s.split(":", 1)
        return cls(venue=Venue(venue_str), raw_id=raw_id)


# Here we calculate TAU so the whole system can consume it (GLFT & near-resol risk mngmt)
@dataclass(frozen=True)
class Resolution:
    resolution_date: datetime
    resolved_value: float | None = None

    def is_resolved(self) -> bool:
        return self.resolved_value is not None

    def resolved_yes(self) -> bool | None:
        if self.resolved_value is None:
            return None
        return self.resolved_value == 1.0

    @property
    def tau(self) -> float:
        now = utcnow()
        if now >= self.resolution_date:
            return 0.0
        delta = (self.resolution_date - now).total_seconds()
        return delta / (365.25 * 24 * 3600)


@dataclass(frozen=True)
class OrderBookLevel:
    price: Price
    size: Size

    def __post_init__(self) -> None:
        if not 0.0 <= self.price <= 1.0:
            raise ValueError(f"Price must be in [0, 1], got {self.price}")
        if self.size < 0:
            raise ValueError(f"Size must be non-negative, got {self.size}")


@dataclass(frozen=True)
class OrderBook:
    market_id: MarketId
    timestamp: datetime
    bids: tuple[OrderBookLevel, ...]
    asks: tuple[OrderBookLevel, ...]

    def __post_init__(self) -> None:  # ← dentro
        if self.bids and self.asks:
            if self.bids[0].price >= self.asks[0].price:
                raise ValueError("Locked book")

    @property
    def best_bid(self) -> Price | None:  # ← dentro
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Price | None:  # ← dentro
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Price | None:  # ← dentro
        if self.best_bid is None or self.best_ask is None:
            return None
        return Price((self.best_bid + self.best_ask) / 2)

    def bid_depth(self, levels: int = 5) -> float:  # ← dentro
        return sum(lv.size for lv in self.bids[:levels])

    def ask_depth(self, levels: int = 5) -> float:
        return sum(lv.size for lv in self.asks[:levels])

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return float(self.best_ask - self.best_bid)


@dataclass(frozen=True)
class Tick:
    market_id: MarketId
    timestamp: datetime
    tick_type: TickType
    yes_bid: Price
    yes_ask: Price
    volume: Size = Size(0.0)
    side: Side | None = None

    # Venue-native event identifier (Manifold bet id, Kalshi/Polymarket trade
    # id). This is the deduplication key: without it, a writer retry is
    # indistinguishable from two genuine trades in the same millisecond — and
    # those are the common case. In the current database, of the 2,264 groups
    # sharing (market_id, timestamp) only 399 were true duplicates; the rest were a
    # `yes` and a `no` matched at the same instant.
    #
    # None for quotes, which have no identity of their own. The unique index
    # ignores them because in SQL two NULLs are distinct.
    source_id: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.yes_bid <= 1.0:
            raise ValueError(f"yes_bid must be in [0,1], got {self.yes_bid}")
        if not 0.0 <= self.yes_ask <= 1.0:
            raise ValueError(f"yes_ask must be in [0,1], got {self.yes_ask}")
        if self.yes_bid > self.yes_ask:
            raise ValueError(f"yes_bid ({self.yes_bid}) > yes_ask ({self.yes_ask})")
        if self.tick_type == TickType.TRADE and self.side is None:
            raise ValueError("TRADE ticks must have a side")

    @property
    def mid(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2

    @property
    def spread(self) -> float:
        return float(self.yes_ask - self.yes_bid)


@dataclass(frozen=True)
class Market:
    market_id: MarketId
    question: str
    category: MarketCategory
    resolution: Resolution
    status: MarketStatus
    created_at: datetime = field(default_factory=utcnow)

    # The market's quotable price grid (Kalshi publishes one per market via
    # `price_ranges`). None = fall back to the venue default ladder.
    price_ladder: PriceLadder | None = None

    def is_tradeable(self) -> bool:
        return self.status == MarketStatus.OPEN


@dataclass(frozen=True)
class MarketSnapshot:
    market: Market
    orderbook: OrderBook | None = None
    last_tick: Tick | None = None
    fetched_at: datetime = field(default_factory=utcnow)
