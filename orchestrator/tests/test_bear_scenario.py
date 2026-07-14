"""Bear-market scenario tests: regime flips, hysteresis, guardian, stale fallback."""

from datetime import datetime, timedelta

import pytest

from src.decision_parser import DecisionResult
from src.portfolio_state import PortfolioState, Position
from src.regime import compute_regime, load_regime_state, save_regime_state
from src.risk_manager import RiskManager


def _pos(symbol: str, value: float, pl_pct: float, avg_cost: float = 100.0) -> Position:
    qty = value / (avg_cost * (1 + pl_pct / 100))
    return Position(
        symbol=symbol, name=symbol, quantity=qty, avg_cost=avg_cost,
        current_price=avg_cost * (1 + pl_pct / 100),
        market_value=value, unrealized_pl=value * pl_pct / 100,
        unrealized_pl_pct=pl_pct, sector="Tech",
        first_buy_date="2026-01-01",
    )


def _portfolio(positions: list[Position], cash: float) -> PortfolioState:
    total = cash + sum(p.market_value for p in positions)
    return PortfolioState(
        account_id="a", account_name="Test", total_value=total,
        cash=cash, invested=sum(p.market_value for p in positions),
        positions=positions,
    )


# ── Scenario 1: Feb-2020-style crash ────────────────────────────────────────

def test_crash_2020_flips_regime_within_two_sessions():
    """SPY -4% in 5 sessions + VIX ramp flips HIGH_VOLATILITY before VIX hits 25."""
    vix = [15.0] * 55 + [15.5, 16.5, 18.0, 20.0, 22.0]      # +42% in 5 sessions
    spy = [770.0] * 245 + [770, 765, 755, 745, 738]          # -4.2% from 5d high
    result = compute_regime(vix, spy)
    assert result.regime == "HIGH_VOLATILITY"
    assert result.vix < 25  # fired before the absolute level trigger


# ── Scenario 2: 2022-style grinding bear (no VIX spike) ─────────────────────

def test_grinding_bear_classified_via_smas():
    """Slow decline below SMA50 and SMA200 with VIX in the low 20s → BEAR_TREND."""
    vix = [21.0] * 60  # elevated but never >25, never +40%/5d
    spy = [900 - i * 0.8 for i in range(250)]  # -0.1%/session, no 5d drawdown trigger
    result = compute_regime(vix, spy)
    assert result.regime == "BEAR_TREND"


# ── Scenario 3: recovery hysteresis ─────────────────────────────────────────

def test_recovery_requires_three_calm_sessions_then_floor_returns():
    base_vix = [16.0] * 44
    spike = [30.0] * 8
    spy = [700 + i * 0.5 for i in range(250)]

    # Two calm sessions: still HIGH_VOLATILITY
    assert compute_regime(base_vix + spike + [19.0, 19.0], spy).regime == "HIGH_VOLATILITY"
    # Third calm session: exits to BULL (SPY above SMA50)
    calmed = compute_regime(base_vix[:-1] + spike + [19.0, 19.0, 19.0], spy)
    assert calmed.regime == "BULL_TREND"


# ── Guardian semantics (via RiskManager primitives) ─────────────────────────

def test_stop_loss_check_triggers_forced_sell():
    """Position at -14% with a -12% threshold → forced SELL (guardian's only job)."""
    portfolio = _portfolio([_pos("NVDA", 2000, -14.0), _pos("SPY", 3000, +2.0)], cash=1000)
    forced = RiskManager({"stop_loss_pct": -12})._check_stop_losses(portfolio)
    assert [a.symbol for a in forced] == ["NVDA"]
    assert forced[0].type == "SELL"
    assert "STOP-LOSS" in forced[0].thesis


def test_stop_loss_check_does_not_include_drawdown_reduction():
    """_check_stop_losses alone must not force-reduce a -25% portfolio: the
    drawdown reduction is cycle-only (daily in the guardian it would liquidate
    a crashed portfolio 50% per day into the bottom)."""
    portfolio = _portfolio(
        [_pos("SPY", 4000, -25.0)], cash=1000
    )
    # -25% portfolio, but the position is above the -30% stop threshold
    forced = RiskManager({"stop_loss_pct": -30})._check_stop_losses(portfolio)
    assert forced == []


# ── Floor is regime-gated; no regime (stale/unavailable) → no floor ─────────

def test_no_floor_when_regime_is_none():
    """Stale or unavailable regime must not force-buy (main.py passes None)."""
    portfolio = _portfolio([], cash=10_000)
    result = RiskManager({"min_cash_pct": 10}).validate(
        decision=DecisionResult(reasoning="hold", actions=[]),
        portfolio=portfolio, quotes={},
        regime=None, core_symbols=["SPY", "QQQ"],
    )
    assert result.forced_actions == []


def test_floor_topup_in_bull_regime():
    portfolio = _portfolio([], cash=10_000)
    result = RiskManager({"min_cash_pct": 10, "max_position_pct": 25}).validate(
        decision=DecisionResult(reasoning="hold", actions=[]),
        portfolio=portfolio, quotes={},
        regime="BULL_TREND", core_symbols=["SPY", "QQQ", "IWM"],
    )
    topup = sum(a.amount_usd for a in result.forced_actions if a.type == "BUY")
    assert topup >= 6000 * 0.99  # 60% floor


def test_no_floor_in_high_volatility():
    portfolio = _portfolio([], cash=10_000)
    result = RiskManager({"min_cash_pct": 10}).validate(
        decision=DecisionResult(reasoning="hold", actions=[]),
        portfolio=portfolio, quotes={},
        regime="HIGH_VOLATILITY", core_symbols=["SPY"],
    )
    assert result.forced_actions == []


# ── Regime state persistence ────────────────────────────────────────────────

def test_regime_state_roundtrip_marks_stale(tmp_path):
    path = tmp_path / "regime_state.json"
    vix = [16.0] * 60
    spy = [700 + i * 0.5 for i in range(250)]
    fresh = compute_regime(vix, spy)
    assert fresh.stale is False

    save_regime_state(fresh, path=path)
    loaded = load_regime_state(path=path)
    assert loaded is not None
    assert loaded.regime == fresh.regime
    assert loaded.stale is True
    assert "stale" in loaded.reasoning


def test_regime_state_too_old_returns_none(tmp_path):
    path = tmp_path / "regime_state.json"
    vix = [16.0] * 60
    spy = [700 + i * 0.5 for i in range(250)]
    save_regime_state(compute_regime(vix, spy), path=path)

    # Rewrite the timestamp to 30h ago
    import json
    data = json.loads(path.read_text())
    data["computed_at"] = (datetime.now() - timedelta(hours=30)).isoformat()
    path.write_text(json.dumps(data))

    assert load_regime_state(path=path) is None
