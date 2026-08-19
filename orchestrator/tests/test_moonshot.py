"""Tests for moonshot_mode: a multi-year sleeve with no price-based exits.

The mode must disable every mechanical exit (stop-loss, drawdown reduction,
zombie cleanup) and the regime exposure floor, while keeping position caps,
trade limits and thesis-driven SELLs fully functional.
"""

import pytest

from src.decision_parser import DecisionResult, TradeAction
from src.risk_manager import RiskManager
from tests.test_data_gate import make_portfolio, make_position, make_quote

MOONSHOT_PROFILE = {
    "max_position_pct": 15,
    "min_cash_pct": 0,
    "max_trades_per_cycle": 2,
    "min_order_usd": 250,
    "min_holding_days": 30,
    "moonshot_mode": True,
    "max_sector_exposure_pct": 100,
}


class TestMoonshotNoMechanicalExits:
    def test_deep_loser_not_stop_lossed(self):
        rm = RiskManager({**MOONSHOT_PROFILE, "stop_loss_pct": -15})
        crashed = make_position("PLTR", quantity=100, price=3.0, avg_cost=10.0)  # -70%
        result = rm.validate(
            decision=DecisionResult(actions=[]),
            portfolio=make_portfolio([crashed]),
            quotes={},
        )
        assert result.forced_actions == []

    def test_portfolio_drawdown_not_force_reduced(self):
        rm = RiskManager(MOONSHOT_PROFILE)
        # Whole book down -40%: past the -20% brake threshold
        book = [make_position(s, quantity=10, price=60, avg_cost=100)
                for s in ("AAA", "BBB", "CCC")]
        result = rm.validate(
            decision=DecisionResult(actions=[]),
            portfolio=make_portfolio(book, cash=100),
            quotes={},
        )
        assert result.forced_actions == []
        assert not any("FORCED REDUCTION" in (a.thesis or "") for a in result.forced_actions)

    def test_zombie_not_cleaned(self):
        rm = RiskManager(MOONSHOT_PROFILE)
        dust = make_position("RDW", quantity=1.0, price=3.0, avg_cost=50.0)  # $3 left
        result = rm.validate(
            decision=DecisionResult(actions=[]),
            portfolio=make_portfolio([dust]),
            quotes={},
        )
        assert result.forced_actions == []

    def test_exposure_floor_ignored_even_in_bull(self):
        rm = RiskManager(MOONSHOT_PROFILE)
        result = rm.validate(
            decision=DecisionResult(actions=[]),
            portfolio=make_portfolio([], cash=10000.0),  # 100% cash
            quotes={},
            regime="BULL_TREND",
        )
        assert not any("EXPOSURE FLOOR" in (a.thesis or "") for a in result.forced_actions)

    def test_stop_loss_none_disables_check_without_mode(self):
        """stop_loss_pct: null in yaml must not crash nor force sells."""
        rm = RiskManager({"max_position_pct": 50, "min_cash_pct": 0,
                          "max_trades_per_cycle": 5, "stop_loss_pct": None})
        crashed = make_position("XYZ", quantity=10, price=50, avg_cost=100)
        result = rm.validate(
            decision=DecisionResult(actions=[]),
            portfolio=make_portfolio([crashed]),
            quotes={},
        )
        assert result.forced_actions == []


class TestMoonshotStillGuarded:
    def test_position_cap_still_enforced(self):
        rm = RiskManager(MOONSHOT_PROFILE)
        result = rm.validate(
            decision=DecisionResult(actions=[
                TradeAction(type="BUY", symbol="NVDA", amount_usd=5000)]),  # 50% of 10k
            portfolio=make_portfolio([], cash=10000.0),
            quotes={"NVDA": make_quote("NVDA")},
        )
        # capped to max_position_pct (15%) or rejected — never approved at full size
        approved_full = [a for a in result.approved_actions if a.amount_usd >= 5000]
        assert approved_full == []

    def test_thesis_break_sell_still_allowed(self):
        rm = RiskManager(MOONSHOT_PROFILE)
        pos = make_position("INTC", quantity=20, price=25, avg_cost=40,)
        result = rm.validate(
            decision=DecisionResult(actions=[
                TradeAction(type="SELL", symbol="INTC", amount_usd=500,
                            thesis="THESIS BREAK: foundry roadmap abandoned")]),
            portfolio=make_portfolio([pos]),
            quotes={},
        )
        assert len(result.approved_actions) == 1

    def test_normal_account_unaffected(self):
        """Without moonshot_mode all mechanical exits keep working."""
        rm = RiskManager({"max_position_pct": 25, "min_cash_pct": 10,
                          "max_trades_per_cycle": 3, "stop_loss_pct": -15})
        crashed = make_position("XYZ", quantity=10, price=50, avg_cost=100)
        result = rm.validate(
            decision=DecisionResult(actions=[]),
            portfolio=make_portfolio([crashed]),
            quotes={},
        )
        assert any(a.symbol == "XYZ" for a in result.forced_actions)
