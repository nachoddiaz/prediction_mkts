"""
tests/unit/test_microstructure.py
──────────────────────────────────
Tests de features/microstructure.py.

Sección 1 — snapshot functions (OrderBook, Tick)
Sección 2 — series functions (DataFrames)
Sección 3 — pipeline (reader + DuckDB :memory:)
"""

from __future__ import annotations

import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from features.microstructure import (
    compute_features_batch,
    compute_features_from_db,
    ewma_vol,
    ewma_vol_series,
    obi_series,
    order_book_imbalance,
    quoted_spread,
    relative_spread,
    spread_timeseries_from_df,
)
from normalizer.schema import (
    Market,
    MarketCategory,
    MarketId,
    MarketStatus,
    OrderBook,
    OrderBookLevel,
    Price,
    Resolution,
    Side,
    Size,
    Tick,
    TickType,
    Venue,
)
from storage.reader import MarketDataReader
from storage.writer import MarketDataWriter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ob(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
) -> OrderBook:
    mid = MarketId(Venue.KALSHI, "KXBTC-TEST")
    return OrderBook(
        market_id=mid,
        timestamp=datetime.now(tz=UTC),
        bids=tuple(OrderBookLevel(Price(p), Size(s)) for p, s in bids),
        asks=tuple(OrderBookLevel(Price(p), Size(s)) for p, s in asks),
    )


def make_tick(
    bid: float = 0.44,
    ask: float = 0.46,
    tick_type: TickType = TickType.QUOTE,
    side: Side | None = None,
    ts: datetime | None = None,
) -> Tick:
    return Tick(
        market_id=MarketId(Venue.KALSHI, "KXBTC-TEST"),
        timestamp=ts or datetime.now(tz=UTC),
        tick_type=tick_type,
        yes_bid=Price(bid),
        yes_ask=Price(ask),
        volume=Size(0.0 if tick_type == TickType.QUOTE else 10.0),
        side=side if tick_type == TickType.TRADE else None,
    )


def make_ticks_df(
    mids: list[float],
    interval_secs: int = 60,
    base_ts: datetime | None = None,
) -> pd.DataFrame:
    base = base_ts or datetime(2026, 4, 22, 12, 0, tzinfo=UTC)
    return pd.DataFrame(
        [
            {
                "timestamp": base + timedelta(seconds=i * interval_secs),
                "mid": m,
                "yes_bid": m - 0.01,
                "yes_ask": m + 0.01,
                "side": None,
                "tick_type": "quote",
            }
            for i, m in enumerate(mids)
        ]
    )


# ---------------------------------------------------------------------------
# DB compartida writer + reader en :memory:
# ---------------------------------------------------------------------------


class DB:
    def __init__(self) -> None:
        self.w = MarketDataWriter(db_path=":memory:")
        self.r = MarketDataReader.__new__(MarketDataReader)
        self.r._con = self.w._con

    def close(self) -> None:
        self.w.close()


# ---------------------------------------------------------------------------
# Sección 1 — Snapshot
# ---------------------------------------------------------------------------


class TestOrderBookImbalance:
    def test_equilibrado(self) -> None:
        ob = make_ob(bids=[(0.45, 100), (0.44, 100)], asks=[(0.47, 100), (0.48, 100)])
        assert order_book_imbalance(ob) == pytest.approx(0.0)

    def test_presion_compradora(self) -> None:
        ob = make_ob(bids=[(0.45, 300)], asks=[(0.47, 100)])
        assert order_book_imbalance(ob) == pytest.approx(0.5)

    def test_presion_vendedora(self) -> None:
        ob = make_ob(bids=[(0.45, 100)], asks=[(0.47, 300)])
        assert order_book_imbalance(ob) == pytest.approx(-0.5)

    def test_rango_menos1_1(self) -> None:
        ob = make_ob(bids=[(0.45, 1000)], asks=[(0.47, 1)])
        assert -1.0 <= order_book_imbalance(ob) <= 1.0

    def test_libro_vacio_cero(self) -> None:
        ob = OrderBook(
            market_id=MarketId(Venue.KALSHI, "KXBTC-TEST"),
            timestamp=datetime.now(tz=UTC),
            bids=(),
            asks=(),
        )
        assert order_book_imbalance(ob) == 0.0


class TestSpreads:
    def test_quoted_spread(self) -> None:
        ob = make_ob(bids=[(0.44, 100)], asks=[(0.46, 100)])
        assert quoted_spread(ob) == pytest.approx(0.02)

    def test_quoted_spread_none_si_vacio(self) -> None:
        ob = OrderBook(
            market_id=MarketId(Venue.KALSHI, "KXBTC-TEST"),
            timestamp=datetime.now(tz=UTC),
            bids=(),
            asks=(),
        )
        assert quoted_spread(ob) is None

    def test_relative_spread(self) -> None:
        ob = make_ob(bids=[(0.44, 100)], asks=[(0.46, 100)])
        assert relative_spread(ob) == pytest.approx(0.02 / 0.45, rel=1e-4)

    def test_relative_spread_mercado_iliquido(self) -> None:
        """Mercado al 5% con spread 2% tiene relative spread ~40%."""
        ob = make_ob(bids=[(0.04, 100)], asks=[(0.06, 100)])
        assert relative_spread(ob) == pytest.approx(0.02 / 0.05, rel=1e-4)

    def test_relative_spread_none_si_vacio(self) -> None:
        ob = OrderBook(
            market_id=MarketId(Venue.KALSHI, "KXBTC-TEST"),
            timestamp=datetime.now(tz=UTC),
            bids=(),
            asks=(),
        )
        assert relative_spread(ob) is None


# ELIMINADO: TestBernoulliVol — función obsoleta según MATH.md v2.1
# La volatilidad ahora se calcula desde variación cuadrática de logit(p)
# usando belief_vol_from_ticks(), no analíticamente desde p y τ.
# Ver tests de belief_vol_from_ticks más abajo (si existen).


# ---------------------------------------------------------------------------
# Sección 2 — Series
# ---------------------------------------------------------------------------


class TestEWMAVol:
    def test_menos_de_2_ticks(self) -> None:
        df = make_ticks_df([0.45])
        assert ewma_vol(df) == 0.0

    def test_precio_constante_vol_cero(self) -> None:
        df = make_ticks_df([0.45] * 20)
        assert ewma_vol(df) == pytest.approx(0.0, abs=1e-10)

    def test_mayor_variacion_mayor_vol(self) -> None:
        low = make_ticks_df([0.450, 0.451, 0.450, 0.451] * 5)
        high = make_ticks_df([0.430, 0.470, 0.430, 0.470] * 5)
        assert ewma_vol(high) > ewma_vol(low)

    def test_devuelve_float_no_negativo(self) -> None:
        df = make_ticks_df([0.40, 0.42, 0.41, 0.43, 0.44])
        vol = ewma_vol(df)
        assert isinstance(vol, float)
        assert vol >= 0.0

    def test_series_longitud_correcta(self) -> None:
        df = make_ticks_df([0.40, 0.42, 0.41, 0.43, 0.44])
        s = ewma_vol_series(df)
        assert len(s) == len(df)

    def test_series_primer_nan(self) -> None:
        df = make_ticks_df([0.40, 0.42, 0.41])
        s = ewma_vol_series(df)
        assert math.isnan(s.iloc[0])

    def test_series_resto_no_nan(self) -> None:
        df = make_ticks_df([0.40, 0.42, 0.41, 0.43])
        s = ewma_vol_series(df)
        assert not s.iloc[1:].isna().any()


class TestOBISeries:
    def test_todos_yes_positivo(self) -> None:
        df = pd.DataFrame({"side": ["yes"] * 10})
        assert (obi_series(df) > 0).all()

    def test_todos_no_negativo(self) -> None:
        df = pd.DataFrame({"side": ["no"] * 10})
        assert (obi_series(df) < 0).all()

    def test_equilibrado_cero(self) -> None:
        df = pd.DataFrame({"side": ["yes", "no"] * 5})
        s = obi_series(df, window=10)
        assert s.iloc[-1] == pytest.approx(0.0)

    def test_sin_columna_side(self) -> None:
        df = make_ticks_df([0.45] * 5)
        assert (obi_series(df) == 0.0).all()


class TestSpreadTimeseries:
    def test_spread_correcto(self) -> None:
        df = pd.DataFrame({"yes_bid": [0.44, 0.43], "yes_ask": [0.46, 0.45]})
        s = spread_timeseries_from_df(df)
        assert list(s) == pytest.approx([0.02, 0.02])

    def test_sin_columnas_raises(self) -> None:
        with pytest.raises(ValueError):
            spread_timeseries_from_df(pd.DataFrame({"mid": [0.45]}))


# ---------------------------------------------------------------------------
# Sección 3 — Pipeline
# ---------------------------------------------------------------------------


class TestComputeFeaturesFromDB:
    def _populate(self, db: DB, n_ticks: int = 20) -> None:
        mid = MarketId(Venue.KALSHI, "KXBTC-TEST")
        db.w.write_market_sync(
            Market(
                market_id=mid,
                question="test",
                category=MarketCategory.CRYPTO,
                resolution=Resolution(
                    resolution_date=datetime(2026, 7, 1, tzinfo=UTC),
                    resolved_value=None,
                ),
                status=MarketStatus.OPEN,
            )
        )
        db.w.write_orderbook_sync(
            OrderBook(
                market_id=mid,
                timestamp=datetime.now(tz=UTC),
                bids=(
                    OrderBookLevel(Price(0.44), Size(300.0)),
                    OrderBookLevel(Price(0.43), Size(200.0)),
                ),
                asks=(
                    OrderBookLevel(Price(0.46), Size(100.0)),
                    OrderBookLevel(Price(0.47), Size(100.0)),
                ),
            )
        )
        base = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)
        db.w.write_ticks_sync(
            [
                Tick(
                    market_id=mid,
                    timestamp=base + timedelta(minutes=i),
                    tick_type=TickType.QUOTE,
                    yes_bid=Price(round(0.43 + i * 0.0005, 4)),
                    yes_ask=Price(round(0.45 + i * 0.0005, 4)),
                    volume=Size(0.0),
                    side=None,
                )
                for i in range(n_ticks)
            ]
        )

    def test_devuelve_dict(self) -> None:
        db = DB()
        self._populate(db)
        result = compute_features_from_db("kalshi:KXBTC-TEST", db.r, 0.19)
        assert result is not None
        assert isinstance(result, dict)
        db.close()

    def test_campos_obligatorios(self) -> None:
        db = DB()
        self._populate(db)
        result = compute_features_from_db("kalshi:KXBTC-TEST", db.r, 0.19)
        assert result is not None
        required = {
            "market_id",
            "venue",
            "timestamp",
            "obi",
            "quoted_spread",
            "relative_spread",
            "belief_vol",
            "ewma_vol",
            "tau_years",
            "mu_hat",
        }
        assert required.issubset(result.keys())
        db.close()

    def test_obi_positivo_con_libro_sesgado(self) -> None:
        """bid_depth=500 > ask_depth=200 → OBI > 0."""
        db = DB()
        self._populate(db)
        result = compute_features_from_db("kalshi:KXBTC-TEST", db.r, 0.19)
        assert result is not None
        assert result["obi"] > 0
        db.close()

    def test_belief_vol_positivo(self) -> None:
        """belief_vol debe ser > 0 si hay suficientes ticks."""
        db = DB()
        self._populate(db)
        r1 = compute_features_from_db("kalshi:KXBTC-TEST", db.r, tau_years=1.0)
        assert r1 is not None
        assert r1["belief_vol"] > 0.0
        db.close()

    def test_none_si_sin_datos(self) -> None:
        db = DB()
        result = compute_features_from_db("kalshi:NO-EXISTE", db.r, 0.5)
        assert result is None
        db.close()

    def test_tau_se_guarda(self) -> None:
        db = DB()
        self._populate(db)
        tau = 0.19123
        result = compute_features_from_db("kalshi:KXBTC-TEST", db.r, tau)
        assert result is not None
        assert result["tau_years"] == pytest.approx(tau, rel=1e-4)
        db.close()

    def test_venue_correcto(self) -> None:
        db = DB()
        self._populate(db)
        result = compute_features_from_db("kalshi:KXBTC-TEST", db.r, 0.19)
        assert result is not None
        assert result["venue"] == "kalshi"
        db.close()

    def test_batch_excluye_sin_datos(self) -> None:
        db = DB()
        self._populate(db)
        tau_map = {"kalshi:KXBTC-TEST": 0.19, "kalshi:NO-EXISTE": 0.5}
        results = compute_features_batch(list(tau_map.keys()), db.r, tau_map)
        assert len(results) == 1
        assert results[0]["market_id"] == "kalshi:KXBTC-TEST"
        db.close()
