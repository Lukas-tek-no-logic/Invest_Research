"""Tests for deterministic market regime classification."""

import pytest

from src.regime import compute_regime


def _flat(value: float, n: int) -> list[float]:
    return [value] * n


def test_calm_bull_market():
    """VIX 16, SPY above SMA50 → BULL_TREND (the 2026-07-09 misclassification case)."""
    vix = _flat(16.0, 60)
    # Rising SPY: above its own SMA50
    spy = [700 + i * 0.5 for i in range(250)]
    result = compute_regime(vix, spy)
    assert result.regime == "BULL_TREND"
    assert "16.0" in result.reasoning


def test_vix_level_triggers_high_volatility():
    vix = _flat(16.0, 55) + _flat(28.0, 5)
    spy = [700 + i * 0.5 for i in range(250)]
    result = compute_regime(vix, spy)
    assert result.regime == "HIGH_VOLATILITY"


def test_vix_spike_triggers_high_volatility_before_25():
    """Feb-2020-style ramp: 15 → 22 in 5 sessions (+47%) fires before VIX crosses 25."""
    vix = _flat(15.0, 55) + [16.0, 17.5, 19.0, 21.0, 22.0]
    spy = [700 + i * 0.5 for i in range(250)]
    result = compute_regime(vix, spy)
    assert result.regime == "HIGH_VOLATILITY"


def test_spy_drawdown_triggers_high_volatility():
    vix = _flat(18.0, 60)
    spy = [700.0] * 245 + [700, 690, 680, 670, 665]  # -5% in 5 sessions
    result = compute_regime(vix, spy)
    assert result.regime == "HIGH_VOLATILITY"


def test_hysteresis_holds_until_three_calm_sessions():
    """After a spike, two calm closes are not enough to exit HIGH_VOLATILITY."""
    vix = _flat(16.0, 50) + _flat(28.0, 8) + [19.0, 19.0]
    spy = [700 + i * 0.5 for i in range(250)]
    result = compute_regime(vix, spy)
    assert result.regime == "HIGH_VOLATILITY"


def test_hysteresis_exits_after_three_calm_sessions():
    vix = _flat(16.0, 47) + _flat(28.0, 8) + [19.0, 19.0, 19.0, 18.5, 18.0]
    spy = [700 + i * 0.5 for i in range(250)]
    result = compute_regime(vix, spy)
    assert result.regime == "BULL_TREND"


def test_bear_trend_below_both_smas():
    vix = _flat(18.0, 60)
    # Slow decline: below SMA50 and SMA200, but never >4% in 5 sessions
    spy = [900 - i * 0.8 for i in range(250)]
    result = compute_regime(vix, spy)
    assert result.regime == "BEAR_TREND"


def test_sideways_when_below_sma50_but_above_sma200():
    vix = _flat(17.0, 60)
    # Long rise then a slow drift down to 890: below SMA50 (~905+) but above
    # SMA200; never more than ~0.1%/session, so no drawdown trigger fires
    spy = [700 + i * 1.0 for i in range(220)] + [919 - i for i in range(30)]
    result = compute_regime(vix, spy)
    assert result.regime == "SIDEWAYS"


def test_insufficient_data_raises():
    with pytest.raises(ValueError):
        compute_regime([16.0] * 5, [700.0] * 250)
    with pytest.raises(ValueError):
        compute_regime([16.0] * 60, [700.0] * 10)


def test_prompt_text_is_authoritative():
    vix = _flat(16.0, 60)
    spy = [700 + i * 0.5 for i in range(250)]
    text = compute_regime(vix, spy).to_prompt_text()
    assert "AUTHORITATIVE" in text
    assert "BULL_TREND" in text
    assert "Do NOT re-classify" in text
