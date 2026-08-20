"""Tests for options/spreads_risk_manager.py."""

from datetime import date, timedelta
from types import SimpleNamespace

from orchestrator.src.options.spreads_decision_parser import SpreadAction, SpreadDecision
from orchestrator.src.options.spreads_risk_manager import SpreadsRiskManager
from orchestrator.src.options.positions import OptionsPosition
from orchestrator.src.portfolio_state import PortfolioState


def _make_portfolio(cash=5000, total_value=10000):
    return PortfolioState(
        account_id="test-id",
        account_name="test",
        cash=cash,
        total_value=total_value,
        invested=total_value - cash,
    )


def _make_position(
    id=1, symbol="SPY", spread_type="IRON_CONDOR",
    dte=30, current_pl=None, max_profit=100, max_loss=200,
    entry_debit=-1.0, profit_captured=None, entry_date="2026-02-01",
):
    pos = OptionsPosition(
        id=id, account_key="test", symbol=symbol,
        spread_type=spread_type, status="open",
        contracts=1, expiration_date="2026-04-01",
        buy_strike=550.0, buy_option_type="put",
        buy_premium=1.0, sell_strike=555.0,
        sell_option_type="put", sell_premium=2.0,
        max_profit=max_profit, max_loss=max_loss,
        entry_debit=entry_debit, entry_date=entry_date,
        dte=dte, current_pl=current_pl,
    )
    return pos


RISK_PROFILE = {
    "max_open_spreads": 3,
    "min_cash_pct": 20,
    "max_spread_width": 10,
    "take_profit_pct": 50,
    "stop_loss_pct": 100,
    "auto_close_dte": 3,
    "target_dte_min": 21,
    "target_dte_max": 45,
}


class TestSpreadsRiskManagerOpens:
    """Test OPEN_SPREAD validation."""

    def test_approve_single_open(self):
        mgr = SpreadsRiskManager(RISK_PROFILE)
        decision = SpreadDecision(actions=[
            SpreadAction(type="OPEN_SPREAD", symbol="SPY", spread_type="iron_condor",
                         contracts=1, reason="Good setup"),
        ])
        result = mgr.validate(decision, [], _make_portfolio())
        assert len(result.approved_opens) == 1
        assert len(result.rejected_opens) == 0

    def test_reject_over_max_spreads(self):
        mgr = SpreadsRiskManager(RISK_PROFILE)
        existing = [
            _make_position(id=i+1, symbol=f"SYM{i}") for i in range(3)
        ]
        decision = SpreadDecision(actions=[
            SpreadAction(type="OPEN_SPREAD", symbol="NEW", spread_type="bull_call",
                         contracts=1, reason="Should be rejected"),
        ])
        result = mgr.validate(decision, existing, _make_portfolio())
        assert len(result.approved_opens) == 0
        assert len(result.rejected_opens) == 1
        assert "Max open spreads" in result.rejected_opens[0]["reason"]

    def test_approve_after_close_frees_slot(self):
        """Closing a position should free a slot for a new open."""
        mgr = SpreadsRiskManager(RISK_PROFILE)
        existing = [
            _make_position(id=i+1, symbol=f"SYM{i}") for i in range(3)
        ]
        decision = SpreadDecision(actions=[
            SpreadAction(type="CLOSE", symbol="SYM0", position_id=1, reason="Take profit"),
            SpreadAction(type="OPEN_SPREAD", symbol="NEW", spread_type="bull_call",
                         contracts=1, reason="Replace closed position"),
        ])
        result = mgr.validate(decision, existing, _make_portfolio())
        assert len(result.approved_closes) == 1
        assert len(result.approved_opens) == 1

    def test_reject_insufficient_cash(self):
        """Should reject when estimated max loss exceeds available cash."""
        mgr = SpreadsRiskManager(RISK_PROFILE)
        # Cash=2500, total_value=10000 → 25% cash
        # max_spread_width=10 → estimated_max_loss = 10*100 = $1000
        # After: (2500-1000)/10000 = 15% < 20% min → reject
        portfolio = _make_portfolio(cash=2500, total_value=10000)
        decision = SpreadDecision(actions=[
            SpreadAction(type="OPEN_SPREAD", symbol="SPY", spread_type="iron_condor",
                         contracts=1, reason="test"),
        ])
        result = mgr.validate(decision, [], portfolio)
        assert len(result.rejected_opens) == 1
        assert "Insufficient cash" in result.rejected_opens[0]["reason"]

    def test_approve_with_enough_cash(self):
        """Cash=5000, total=10000 → 50%. After -$1000 → 40% > 20%."""
        mgr = SpreadsRiskManager(RISK_PROFILE)
        portfolio = _make_portfolio(cash=5000, total_value=10000)
        decision = SpreadDecision(actions=[
            SpreadAction(type="OPEN_SPREAD", symbol="SPY", spread_type="iron_condor",
                         contracts=1, reason="test"),
        ])
        result = mgr.validate(decision, [], portfolio)
        assert len(result.approved_opens) == 1

    def test_reject_earnings_flag(self):
        mgr = SpreadsRiskManager(RISK_PROFILE)
        decision = SpreadDecision(actions=[
            SpreadAction(type="OPEN_SPREAD", symbol="TSLA", spread_type="bull_call",
                         contracts=1, reason="Earnings in 3 days, risky"),
        ])
        result = mgr.validate(decision, [], _make_portfolio())
        assert len(result.rejected_opens) == 1
        assert "near-earnings" in result.rejected_opens[0]["reason"]

    def test_safe_earnings_phrase_passes(self):
        mgr = SpreadsRiskManager(RISK_PROFILE)
        decision = SpreadDecision(actions=[
            SpreadAction(type="OPEN_SPREAD", symbol="AAPL", spread_type="iron_condor",
                         contracts=1, reason="No earnings for 6 weeks, safe to sell premium"),
        ])
        result = mgr.validate(decision, [], _make_portfolio())
        assert len(result.approved_opens) == 1


class TestSpreadsRiskManagerCloses:
    """Test CLOSE validation."""

    def test_approve_valid_close(self):
        mgr = SpreadsRiskManager(RISK_PROFILE)
        existing = [_make_position(id=5)]
        decision = SpreadDecision(actions=[
            SpreadAction(type="CLOSE", symbol="SPY", position_id=5, reason="take profit"),
        ])
        result = mgr.validate(decision, existing, _make_portfolio())
        assert len(result.approved_closes) == 1

    def test_reject_close_unknown_id(self):
        mgr = SpreadsRiskManager(RISK_PROFILE)
        decision = SpreadDecision(actions=[
            SpreadAction(type="CLOSE", symbol="SPY", position_id=999, reason="unknown"),
        ])
        result = mgr.validate(decision, [], _make_portfolio())
        assert len(result.approved_closes) == 0
        assert any("unknown position ID 999" in w for w in result.warnings)


class TestSpreadsAutoClose:
    """Test auto-close rules (DTE, take-profit, stop-loss)."""

    def test_auto_close_low_dte(self):
        mgr = SpreadsRiskManager(RISK_PROFILE)
        pos = _make_position(id=1, dte=2)  # below auto_close_dte=3
        decision = SpreadDecision(actions=[])
        result = mgr.validate(decision, [pos], _make_portfolio())
        assert len(result.forced_closes) == 1
        assert "DTE=2" in result.forced_closes[0].reason

    def test_no_auto_close_above_dte(self):
        mgr = SpreadsRiskManager(RISK_PROFILE)
        pos = _make_position(id=1, dte=15)
        decision = SpreadDecision(actions=[])
        result = mgr.validate(decision, [pos], _make_portfolio())
        assert len(result.forced_closes) == 0

    def test_auto_close_take_profit(self):
        mgr = SpreadsRiskManager(RISK_PROFILE)
        # max_profit=100, current_pl=60 → 60% captured ≥ 50%
        pos = _make_position(id=1, dte=20, max_profit=100, current_pl=60)
        decision = SpreadDecision(actions=[])
        result = mgr.validate(decision, [pos], _make_portfolio())
        assert len(result.forced_closes) == 1
        assert "Take-profit" in result.forced_closes[0].reason

    def test_no_take_profit_below_threshold(self):
        mgr = SpreadsRiskManager(RISK_PROFILE)
        # max_profit=100, current_pl=30 → 30% < 50%
        pos = _make_position(id=1, dte=20, max_profit=100, current_pl=30)
        decision = SpreadDecision(actions=[])
        result = mgr.validate(decision, [pos], _make_portfolio())
        assert len(result.forced_closes) == 0

    def test_auto_close_stop_loss(self):
        mgr = SpreadsRiskManager(RISK_PROFILE)
        # max_loss=200, current_pl=-200 → loss_pct = 200/200 = 100% ≥ 100%
        pos = _make_position(id=1, dte=20, max_loss=200, current_pl=-200)
        decision = SpreadDecision(actions=[])
        result = mgr.validate(decision, [pos], _make_portfolio())
        assert len(result.forced_closes) == 1
        assert "Stop-loss" in result.forced_closes[0].reason

    def test_no_stop_loss_below_threshold(self):
        mgr = SpreadsRiskManager(RISK_PROFILE)
        # max_loss=200, current_pl=-100 → 50% < 100%
        pos = _make_position(id=1, dte=20, max_loss=200, current_pl=-100)
        decision = SpreadDecision(actions=[])
        result = mgr.validate(decision, [pos], _make_portfolio())
        assert len(result.forced_closes) == 0

    def test_llm_close_not_duplicated_by_auto(self):
        """If LLM already requested CLOSE, auto-close should not duplicate."""
        mgr = SpreadsRiskManager(RISK_PROFILE)
        pos = _make_position(id=1, dte=2)  # would trigger auto-close
        decision = SpreadDecision(actions=[
            SpreadAction(type="CLOSE", symbol="SPY", position_id=1, reason="manual close"),
        ])
        result = mgr.validate(decision, [pos], _make_portfolio())
        assert len(result.approved_closes) == 1
        assert len(result.forced_closes) == 0  # not duplicated


class TestSpreadsRiskManagerCashAccounting:
    """Test cash accounting across multiple opens."""

    def test_sequential_opens_deplete_cash(self):
        """Each approved open reduces available cash for subsequent ones."""
        mgr = SpreadsRiskManager({
            **RISK_PROFILE,
            "max_open_spreads": 10,  # high limit
            "min_cash_pct": 20,
            "max_spread_width": 10,  # $1000 per spread
        })
        # Cash=4000, total=10000 → 40%
        # First open: 4000-1000=3000 → 30% OK
        # Second: 3000-1000=2000 → 20% OK (borderline)
        # Third: 2000-1000=1000 → 10% < 20% → REJECT
        portfolio = _make_portfolio(cash=4000, total_value=10000)
        decision = SpreadDecision(actions=[
            SpreadAction(type="OPEN_SPREAD", symbol="A", spread_type="bull_call", contracts=1, reason=""),
            SpreadAction(type="OPEN_SPREAD", symbol="B", spread_type="bear_put", contracts=1, reason=""),
            SpreadAction(type="OPEN_SPREAD", symbol="C", spread_type="iron_condor", contracts=1, reason=""),
        ])
        result = mgr.validate(decision, [], portfolio)
        assert len(result.approved_opens) == 2
        assert len(result.rejected_opens) == 1


class TestSpreadsExitDiscipline:
    """min_holding_days gating and short-strike breach exits."""

    def _mgr(self, **overrides):
        return SpreadsRiskManager({**RISK_PROFILE, "min_holding_days": 5, **overrides})

    def test_no_mark_based_exit_before_min_holding(self):
        """Fresh position: noisy marks must not trigger TP or SL."""
        mgr = self._mgr()
        pos = _make_position(
            current_pl=90, entry_date=date.today().isoformat(),  # "90% captured"
        )
        assert mgr.check_position_exits(pos) is None

        pos_loss = _make_position(
            current_pl=-250, entry_date=date.today().isoformat(),  # > max_loss stop
        )
        assert mgr.check_position_exits(pos_loss) is None

    def test_take_profit_after_min_holding(self):
        mgr = self._mgr()
        pos = _make_position(
            current_pl=90,
            entry_date=(date.today() - timedelta(days=6)).isoformat(),
        )
        forced = mgr.check_position_exits(pos)
        assert forced is not None and "Take-profit" in forced.reason

    def test_stop_loss_after_min_holding(self):
        mgr = self._mgr(stop_loss_pct=50)
        pos = _make_position(
            current_pl=-150, max_loss=200,
            entry_date=(date.today() - timedelta(days=6)).isoformat(),
        )
        forced = mgr.check_position_exits(pos)
        assert forced is not None and "Stop-loss" in forced.reason

    def test_short_strike_breach_overrides_min_holding(self):
        """Real directional threat closes immediately, even on day one."""
        mgr = self._mgr()
        # Credit position, short put 555 — spot below the strike = breach
        pos = _make_position(entry_date=date.today().isoformat())
        forced = mgr.check_position_exits(pos, spot=550.0)
        assert forced is not None and "breach" in forced.reason.lower()

    def test_no_breach_when_spot_safe(self):
        mgr = self._mgr()
        pos = _make_position(entry_date=date.today().isoformat())
        assert mgr.check_position_exits(pos, spot=560.0) is None

    def test_dte_close_ignores_min_holding(self):
        mgr = self._mgr()
        pos = _make_position(dte=2, entry_date=date.today().isoformat())
        forced = mgr.check_position_exits(pos)
        assert forced is not None and "DTE" in forced.reason

    def test_breach_ignored_for_debit_spreads(self):
        """Breach logic applies to credit structures only."""
        mgr = self._mgr()
        pos = _make_position(entry_debit=2.5, entry_date=date.today().isoformat())
        assert mgr.check_position_exits(pos, spot=550.0) is None


class TestTrendConsistencyFilter:
    """Directional spreads must not fight a clear trend."""

    def _decision(self, spread_type, symbol="NVDA"):
        return SpreadDecision(actions=[
            SpreadAction(type="OPEN_SPREAD", symbol=symbol,
                         spread_type=spread_type, contracts=1, reason="setup"),
        ])

    def test_reject_bear_spread_in_uptrend(self):
        mgr = SpreadsRiskManager(RISK_PROFILE)
        signals = {"NVDA": SimpleNamespace(price=200.0, sma_50=180.0, rsi_14=65.0)}
        result = mgr.validate(self._decision("bear_call"), [], _make_portfolio(),
                              tech_signals=signals)
        assert len(result.approved_opens) == 0
        assert "against uptrend" in result.rejected_opens[0]["reason"]

    def test_reject_bull_spread_in_downtrend(self):
        mgr = SpreadsRiskManager(RISK_PROFILE)
        signals = {"NVDA": SimpleNamespace(price=150.0, sma_50=180.0, rsi_14=38.0)}
        result = mgr.validate(self._decision("bull_put"), [], _make_portfolio(),
                              tech_signals=signals)
        assert len(result.approved_opens) == 0
        assert "against downtrend" in result.rejected_opens[0]["reason"]

    def test_allow_bear_spread_when_trend_down(self):
        mgr = SpreadsRiskManager(RISK_PROFILE)
        signals = {"NVDA": SimpleNamespace(price=150.0, sma_50=180.0, rsi_14=38.0)}
        result = mgr.validate(self._decision("bear_call"), [], _make_portfolio(),
                              tech_signals=signals)
        assert len(result.approved_opens) == 1

    def test_neutral_structure_ignores_trend(self):
        mgr = SpreadsRiskManager(RISK_PROFILE)
        signals = {"NVDA": SimpleNamespace(price=200.0, sma_50=180.0, rsi_14=65.0)}
        result = mgr.validate(self._decision("iron_condor"), [], _make_portfolio(),
                              tech_signals=signals)
        assert len(result.approved_opens) == 1

    def test_missing_signals_pass(self):
        mgr = SpreadsRiskManager(RISK_PROFILE)
        result = mgr.validate(self._decision("bear_call"), [], _make_portfolio(),
                              tech_signals=None)
        assert len(result.approved_opens) == 1


class TestVrpPilotKnobs:
    """allowed_spread_types + allow_same_symbol_spreads (Phase 5 VRP pilot)."""

    VRP_PROFILE = {
        **RISK_PROFILE,
        "allowed_spread_types": ["bull_put"],
        "allow_same_symbol_spreads": True,
    }

    def test_disallowed_structure_rejected(self):
        mgr = SpreadsRiskManager(self.VRP_PROFILE)
        decision = SpreadDecision(actions=[
            SpreadAction(type="OPEN_SPREAD", symbol="SPY", spread_type="iron_condor",
                         contracts=1, reason="neutral market"),
        ])
        result = mgr.validate(decision, [], _make_portfolio())
        assert result.approved_opens == []
        assert any("not in allowed set" in r["reason"] for r in result.rejected_opens)

    def test_allowed_structure_passes(self):
        mgr = SpreadsRiskManager(self.VRP_PROFILE)
        decision = SpreadDecision(actions=[
            SpreadAction(type="OPEN_SPREAD", symbol="SPY", spread_type="bull_put",
                         contracts=1, reason="IV rank elevated"),
        ])
        result = mgr.validate(decision, [], _make_portfolio())
        assert len(result.approved_opens) == 1

    def test_case_insensitive_whitelist(self):
        mgr = SpreadsRiskManager(self.VRP_PROFILE)
        decision = SpreadDecision(actions=[
            SpreadAction(type="OPEN_SPREAD", symbol="SPY", spread_type="BULL_PUT",
                         contracts=1, reason="ok"),
        ])
        result = mgr.validate(decision, [], _make_portfolio())
        assert len(result.approved_opens) == 1

    def test_same_symbol_ladder_allowed(self):
        """With the flag, a second SPY spread may open next to an existing one."""
        mgr = SpreadsRiskManager(self.VRP_PROFILE)
        existing = _make_position(id=1, symbol="SPY", spread_type="BULL_PUT")
        decision = SpreadDecision(actions=[
            SpreadAction(type="OPEN_SPREAD", symbol="SPY", spread_type="bull_put",
                         contracts=1, reason="ladder next expiry"),
        ])
        result = mgr.validate(decision, [existing], _make_portfolio())
        assert len(result.approved_opens) == 1

    def test_same_symbol_still_blocked_without_flag(self):
        mgr = SpreadsRiskManager({**RISK_PROFILE, "allowed_spread_types": ["bull_put"]})
        existing = _make_position(id=1, symbol="SPY", spread_type="BULL_PUT")
        decision = SpreadDecision(actions=[
            SpreadAction(type="OPEN_SPREAD", symbol="SPY", spread_type="bull_put",
                         contracts=1, reason="ladder"),
        ])
        result = mgr.validate(decision, [existing], _make_portfolio())
        assert result.approved_opens == []

    def test_no_whitelist_means_everything_allowed(self):
        mgr = SpreadsRiskManager(RISK_PROFILE)
        decision = SpreadDecision(actions=[
            SpreadAction(type="OPEN_SPREAD", symbol="NVDA", spread_type="iron_condor",
                         contracts=1, reason="neutral"),
        ])
        result = mgr.validate(decision, [], _make_portfolio())
        assert len(result.approved_opens) == 1


class TestVixTermGate:
    """vix_term_max_ratio: sell premium only in contango; fail closed."""

    GATED = {**RISK_PROFILE, "vix_term_max_ratio": 0.95}

    def _open(self):
        return SpreadDecision(actions=[
            SpreadAction(type="OPEN_SPREAD", symbol="SPY", spread_type="bull_put",
                         contracts=1, reason="IV elevated"),
        ])

    def test_contango_passes(self):
        mgr = SpreadsRiskManager(self.GATED)
        result = mgr.validate(self._open(), [], _make_portfolio(), vix_term_ratio=0.88)
        assert len(result.approved_opens) == 1

    def test_backwardation_rejected(self):
        mgr = SpreadsRiskManager(self.GATED)
        result = mgr.validate(self._open(), [], _make_portfolio(), vix_term_ratio=1.04)
        assert result.approved_opens == []
        assert any("backwardation" in r["reason"] for r in result.rejected_opens)

    def test_missing_ratio_fails_closed(self):
        mgr = SpreadsRiskManager(self.GATED)
        result = mgr.validate(self._open(), [], _make_portfolio(), vix_term_ratio=None)
        assert result.approved_opens == []
        assert any("unavailable" in r["reason"] for r in result.rejected_opens)

    def test_closes_never_blocked_by_gate(self):
        mgr = SpreadsRiskManager(self.GATED)
        pos = _make_position(id=9, symbol="SPY", spread_type="BULL_PUT")
        decision = SpreadDecision(actions=[
            SpreadAction(type="CLOSE", symbol="SPY", spread_type="bull_put",
                         contracts=1, reason="take profit", position_id=9),
        ])
        result = mgr.validate(decision, [pos], _make_portfolio(), vix_term_ratio=1.10)
        assert len(result.approved_closes) == 1

    def test_gate_absent_means_no_filtering(self):
        mgr = SpreadsRiskManager(RISK_PROFILE)
        result = mgr.validate(self._open(), [], _make_portfolio(), vix_term_ratio=None)
        assert len(result.approved_opens) == 1
