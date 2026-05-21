"""
tests/unit/test_signals.py
Tests for features/signals — ensemble, news stub, onchain stub.
"""

from __future__ import annotations

import numpy as np
import pytest

from features.signals.ensemble import SignalEnsemble
from features.signals.news import NewsSignal
from features.signals.onchain import OnChainSignal

# ---------------------------------------------------------------------------
# NewsSignal
# ---------------------------------------------------------------------------


class TestNewsSignal:
    def test_stub_returns_zero(self) -> None:
        s = NewsSignal()
        assert s.get("Will BTC exceed 100k?") == 0.0

    def test_is_available_false(self) -> None:
        assert NewsSignal().is_available is False

    def test_with_api_key_still_stub(self) -> None:
        # Even with an API key configured, current impl is still a stub
        s = NewsSignal(api_key="test-key")
        assert s.get("any market") == 0.0


# ---------------------------------------------------------------------------
# OnChainSignal
# ---------------------------------------------------------------------------


class TestOnChainSignal:
    def test_stub_returns_zero(self) -> None:
        s = OnChainSignal()
        assert s.get() == 0.0
        assert s.get(token_address="0xabc", condition_id="0xdef") == 0.0

    def test_is_available_false(self) -> None:
        assert OnChainSignal().is_available is False


# ---------------------------------------------------------------------------
# SignalEnsemble
# ---------------------------------------------------------------------------


class TestSignalEnsemble:
    def test_obi_only_default(self) -> None:
        ens = SignalEnsemble()
        assert ens.compute_mu_hat(obi=0.5) == pytest.approx(0.5, abs=1e-9)
        assert ens.compute_mu_hat(obi=0.5, news=0.5, onchain=0.5) == pytest.approx(0.5, abs=1e-9)

    def test_weighted_sum(self) -> None:
        ens = SignalEnsemble(weights={"obi": 0.6, "news": 0.3, "onchain": 0.1})
        result = ens.compute_mu_hat(obi=1.0, news=1.0, onchain=1.0)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_zero_signal_returns_zero(self) -> None:
        ens = SignalEnsemble(weights={"obi": 1.0, "news": 0.5, "onchain": 0.5})
        assert ens.compute_mu_hat(obi=0.0, news=0.0, onchain=0.0) == pytest.approx(0.0, abs=1e-9)

    def test_obi_only_classmethod(self) -> None:
        ens = SignalEnsemble.obi_only()
        assert ens.compute_mu_hat(obi=0.42) == pytest.approx(0.42, abs=1e-9)
        # news and onchain have zero weight
        assert ens.compute_mu_hat(obi=0.42, news=99.0, onchain=-99.0) == pytest.approx(
            0.42, abs=1e-9
        )

    def test_from_cj_result(self) -> None:
        class FakeCJ:
            w_obi = 0.7
            w_news = 0.2
            w_onchain = 0.1

        ens = SignalEnsemble.from_cj_result(FakeCJ())
        result = ens.compute_mu_hat(obi=1.0, news=0.0, onchain=0.0)
        assert result == pytest.approx(0.7, abs=1e-9)

    def test_update_weights(self) -> None:
        ens = SignalEnsemble()
        ens.update_weights({"obi": 0.5, "news": 0.5})
        assert ens.compute_mu_hat(obi=1.0, news=0.0) == pytest.approx(0.5, abs=1e-9)

    def test_wrong_matrix_shape_raises(self) -> None:
        ens = SignalEnsemble()
        with pytest.raises(ValueError, match="shape"):
            ens.fit_orthogonalization(np.ones((100, 2)))  # needs 3 columns

    def test_fit_orthogonalization_changes_output(self) -> None:
        rng = np.random.default_rng(0)
        obi = rng.standard_normal(200)
        news = 0.8 * obi + 0.2 * rng.standard_normal(200)  # correlated
        onchain = rng.standard_normal(200) * 0.1
        signal_matrix = np.column_stack([obi, news, onchain])

        ens_raw = SignalEnsemble(weights={"obi": 0.5, "news": 0.5, "onchain": 0.0})
        ens_ortho = SignalEnsemble(weights={"obi": 0.5, "news": 0.5, "onchain": 0.0})
        ens_ortho.fit_orthogonalization(signal_matrix)

        test_obi, test_news, test_onchain = 0.5, 0.4, 0.0
        raw = ens_raw.compute_mu_hat(test_obi, test_news, test_onchain)
        ortho = ens_ortho.compute_mu_hat(test_obi, test_news, test_onchain)

        # After orthogonalization, result should differ (correlation removed)
        assert raw != pytest.approx(ortho, abs=1e-6)

    def test_all_zero_signals_with_ortho(self) -> None:
        """Orthogonalization of constant signals should not crash."""
        ens = SignalEnsemble()
        signal_matrix = np.column_stack(
            [
                np.zeros(100),
                np.zeros(100),
                np.zeros(100),
            ]
        )
        # Should not raise — regularization handles singular covariance
        ens.fit_orthogonalization(signal_matrix)
        result = ens.compute_mu_hat(0.0, 0.0, 0.0)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_mu_hat_sign_preserved(self) -> None:
        ens = SignalEnsemble(weights={"obi": 1.0, "news": 0.0, "onchain": 0.0})
        assert ens.compute_mu_hat(obi=0.3) > 0
        assert ens.compute_mu_hat(obi=-0.3) < 0
