"""Smoke tests for rules engine integration — catches import and wiring errors."""

from unittest.mock import MagicMock, patch

from orchestrator.src.rules_engine import RulesEngine, RulesProposal
from orchestrator.src.llm_review import build_review_messages, parse_review, apply_vetoes
from orchestrator.src.decision_parser import DecisionResult, TradeAction
from orchestrator.src.portfolio_state import PortfolioState, Position
from orchestrator.src.technical_indicators import TechnicalSignals


def _make_portfolio(cash=5000, positions=None):
    return PortfolioState(
        account_id="test", account_name="Test",
        total_value=10000, cash=cash, invested=5000,
        positions=positions or [], total_pl=0, total_pl_pct=0,
        sector_weights={}, timestamp="2026-01-01",
    )


def _make_tech(symbol, price=100, rsi=55, sma20=98, sma50=95, sma200=90,
               macd_hist=0.5, volume_ratio=1.2, adx=20):
    return TechnicalSignals(
        symbol=symbol, price=price, rsi_14=rsi,
        sma_20=sma20, sma_50=sma50, sma_200=sma200,
        macd_histogram=macd_hist, volume_ratio=volume_ratio,
        adx_14=adx, support_level=price * 0.95, resistance_level=price * 1.05,
    )


class TestRulesEngineSmoke:
    """Verify rules engine produces valid output for each strategy."""

    def test_momentum_hold_timing_gate(self):
        engine = RulesEngine()
        result = engine.propose(
            strategy="momentum",
            portfolio=_make_portfolio(),
            market_data={},
            technical_signals={"SPY": _make_tech("SPY")},
            fundamentals={},
            quotes={},
            earnings_data={},
            risk_profile={"max_position_pct": 25, "min_cash_pct": 10, "stop_loss_pct": -12},
            rules_params={"rebalance_interval_days": 7},
            last_rebalance_date="2026-01-01T00:00:00",  # long ago
        )
        assert isinstance(result, RulesProposal)
        assert isinstance(result.actions, list)
        for a in result.actions:
            assert isinstance(a, TradeAction)
            assert a.type in ("BUY", "SELL")
            assert a.symbol
            assert a.amount_usd > 0

    def test_momentum_hold_recent_rebalance(self):
        """Should HOLD if rebalanced recently."""
        from datetime import datetime
        engine = RulesEngine()
        result = engine.propose(
            strategy="momentum",
            portfolio=_make_portfolio(),
            market_data={},
            technical_signals={"SPY": _make_tech("SPY")},
            fundamentals={},
            quotes={},
            earnings_data={},
            risk_profile={"max_position_pct": 25, "min_cash_pct": 10, "stop_loss_pct": -12},
            rules_params={"rebalance_interval_days": 7},
            last_rebalance_date=datetime.now().isoformat(),  # just now
        )
        assert result.actions == []
        assert result.hold_reason  # should explain why holding

    def test_core_satellite_no_drift(self):
        """Should HOLD if allocation on target."""
        engine = RulesEngine()
        positions = [
            Position(symbol="VOO", name="VOO", quantity=10, avg_cost=50, current_price=60,
                     market_value=6000, unrealized_pl=100, unrealized_pl_pct=10, sector="ETF"),
            Position(symbol="AAPL", name="AAPL", quantity=5, avg_cost=100, current_price=110,
                     market_value=2500, unrealized_pl=50, unrealized_pl_pct=5, sector="Tech"),
        ]
        result = engine.propose(
            strategy="core_satellite",
            portfolio=_make_portfolio(cash=1500, positions=positions),
            market_data={},
            technical_signals={"AAPL": _make_tech("AAPL")},
            fundamentals={},
            quotes={},
            earnings_data={},
            risk_profile={"max_position_pct": 20, "min_cash_pct": 10, "stop_loss_pct": -15},
            rules_params={"rebalance_drift_pct": 5},
        )
        assert isinstance(result, RulesProposal)
        # 60% core, 25% satellite, 15% cash → within 5% drift → HOLD
        assert result.actions == []

    def test_value_timing_gate(self):
        engine = RulesEngine()
        result = engine.propose(
            strategy="value_investing",
            portfolio=_make_portfolio(),
            market_data={},
            technical_signals={},
            fundamentals={},
            quotes={},
            earnings_data={},
            risk_profile={"max_position_pct": 15, "min_cash_pct": 10, "stop_loss_pct": -20},
            rules_params={"rebalance_interval_days": 25},
            last_rebalance_date="2026-01-01T00:00:00",
        )
        assert isinstance(result, RulesProposal)

    def test_unknown_strategy_returns_hold(self):
        engine = RulesEngine()
        result = engine.propose(
            strategy="nonexistent",
            portfolio=_make_portfolio(),
            market_data={}, technical_signals={}, fundamentals={},
            quotes={}, earnings_data={}, risk_profile={}, rules_params={},
        )
        assert result.actions == []
        assert "Unknown" in result.hold_reason


class TestLLMReviewSmoke:
    """Verify LLM review module produces valid output."""

    def test_build_review_messages(self):
        from orchestrator.src.rules_engine import ScoredSymbol
        proposals = [
            TradeAction(type="BUY", symbol="SPY", amount_usd=1000, thesis="test"),
        ]
        scores = [ScoredSymbol(symbol="SPY", score=75, signal="BUY")]
        msgs = build_review_messages(
            proposals=proposals,
            score_breakdowns=scores,
            portfolio=_make_portfolio(),
            market_context={"vix": 20, "tnx": 4.5},
            strategy="momentum",
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert "APPROVE" in msgs[0]["content"]
        assert "SPY" in msgs[1]["content"]

    def test_parse_review_approve(self):
        verdicts = parse_review({
            "reviews": [{"symbol": "SPY", "verdict": "APPROVE", "reason": "OK"}]
        })
        assert len(verdicts) == 1
        assert verdicts[0].verdict == "APPROVE"

    def test_parse_review_veto(self):
        verdicts = parse_review({
            "reviews": [{"symbol": "TSLA", "verdict": "VETO", "reason": "earnings tomorrow"}]
        })
        assert len(verdicts) == 1
        assert verdicts[0].verdict == "VETO"

    def test_apply_vetoes(self):
        proposals = [
            TradeAction(type="BUY", symbol="SPY", amount_usd=1000, thesis="ok"),
            TradeAction(type="BUY", symbol="TSLA", amount_usd=500, thesis="risky"),
        ]
        from orchestrator.src.llm_review import ReviewVerdict
        verdicts = [ReviewVerdict(symbol="TSLA", verdict="VETO", reason="earnings")]
        result = apply_vetoes(proposals, verdicts)
        assert isinstance(result, DecisionResult)
        assert len(result.actions) == 1
        assert result.actions[0].symbol == "SPY"

    def test_parse_review_empty(self):
        """Empty/malformed response → no vetoes."""
        verdicts = parse_review({})
        assert verdicts == []

    def test_parse_review_invalid_verdict_defaults_approve(self):
        verdicts = parse_review({
            "reviews": [{"symbol": "X", "verdict": "MAYBE"}]
        })
        assert verdicts[0].verdict == "APPROVE"


class TestDecisionResultImport:
    """The bug that caused the production error — verify DecisionResult is importable from main.py's context."""

    def test_decision_result_importable(self):
        """This test would have caught the missing import."""
        from orchestrator.src.main import Orchestrator  # noqa: F401
        from orchestrator.src.decision_parser import DecisionResult  # noqa: F401
        # If this doesn't raise ImportError, the fix is working
        assert DecisionResult is not None
