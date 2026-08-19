"""Tests for the Phase-1 data gate and measurement rig.

Covers: symbol normalization/shape filtering, no-quote BUY rejection,
stale-price stop-loss skip, options commissions (calculate_option_cost and
its netting in close/expire), the mid-price execution haircut, fee-aware
trade journal, audit-log migrations (model_used / valuation_carried /
benchmark_return_pct), and guardian labeling in decision history.
"""

import sqlite3

import pandas as pd
import pytest

from src.audit_logger import AuditLogger
from src.decision_parser import DecisionResult, TradeAction
from src.market_data import StockQuote
from src.options.positions import OptionsPositionTracker
from src.options.selector import _mid_price as wheel_mid
from src.options.spreads_selector import _mid_price as spreads_mid
from src.portfolio_state import PortfolioState, Position
from src.prompt_builder import format_decision_history
from src.risk_manager import RiskManager
from src.scanner import build_scan_messages
from src.transaction_costs import calculate_option_cost


# ── helpers ──────────────────────────────────────────────────────────────────

def make_quote(symbol: str, price: float = 100.0, avg_volume: int = 1_000_000) -> StockQuote:
    return StockQuote(
        symbol=symbol, price=price, change_pct=0, volume=100000,
        avg_volume_10d=avg_volume, market_cap=1e9, pe_ratio=20, forward_pe=18,
        pb_ratio=3, dividend_yield=0.02, week52_high=120, week52_low=80,
        sector="Technology", industry="Software", name=f"{symbol} Inc",
    )


def make_position(symbol: str, quantity: float = 10.0, price: float = 100.0,
                  avg_cost: float | None = None, price_stale: bool = False) -> Position:
    avg_cost = avg_cost if avg_cost is not None else price
    mv = quantity * price
    inv = quantity * avg_cost
    return Position(
        symbol=symbol, name=f"{symbol} Inc", quantity=quantity, avg_cost=avg_cost,
        current_price=price, market_value=mv, unrealized_pl=mv - inv,
        unrealized_pl_pct=((mv - inv) / inv * 100) if inv else 0.0,
        sector="Technology", price_stale=price_stale,
    )


def make_portfolio(positions=None, cash: float = 5000.0) -> PortfolioState:
    positions = positions or []
    total_market = sum(p.market_value for p in positions)
    return PortfolioState(
        account_id="acct-1", account_name="Test",
        total_value=total_market + cash, cash=cash,
        invested=sum(p.avg_cost * p.quantity for p in positions),
        positions=positions,
    )


RISK_PROFILE = {
    "max_position_pct": 50,
    "min_cash_pct": 0,
    "max_trades_per_cycle": 10,
}


# ── symbol normalization / shape filter ──────────────────────────────────────

class TestSymbolNormalization:
    def test_lowercase_symbol_uppercased(self):
        a = TradeAction(type="BUY", symbol="nvda", amount_usd=500)
        assert a.symbol == "NVDA"

    def test_mixed_case_and_whitespace(self):
        a = TradeAction(type="SELL", symbol="  Brk-b ", amount_usd=500)
        assert a.symbol == "BRK-B"

    def test_invalid_symbol_dropped_in_decision(self):
        d = DecisionResult.model_validate({
            "actions": [
                {"type": "BUY", "symbol": "NVDA", "amount_usd": 500},
                {"type": "BUY", "symbol": "N/A$", "amount_usd": 500},
                {"type": "BUY", "symbol": "TOOLONGSYM", "amount_usd": 500},
            ],
        })
        assert [a.symbol for a in d.actions] == ["NVDA"]

    def test_lowercase_normalized_in_decision(self):
        d = DecisionResult.model_validate({
            "actions": [{"type": "BUY", "ticker": "msft", "amount": 400}],
        })
        assert d.actions[0].symbol == "MSFT"


# ── no-quote BUY rejection ───────────────────────────────────────────────────

class TestNoQuoteGate:
    def test_buy_without_quote_rejected(self):
        rm = RiskManager(RISK_PROFILE)
        result = rm.validate(
            decision=DecisionResult(actions=[
                TradeAction(type="BUY", symbol="GHOST", amount_usd=500)]),
            portfolio=make_portfolio(),
            quotes={},
        )
        assert not result.approved_actions
        assert any("No market quote" in (r.rejection_reason or "")
                   for r in result.rejected_actions)

    def test_buy_with_zero_price_quote_rejected(self):
        rm = RiskManager(RISK_PROFILE)
        result = rm.validate(
            decision=DecisionResult(actions=[
                TradeAction(type="BUY", symbol="ZERO", amount_usd=500)]),
            portfolio=make_portfolio(),
            quotes={"ZERO": make_quote("ZERO", price=0.0)},
        )
        assert not result.approved_actions

    def test_sell_without_quote_not_blocked_by_gate(self):
        """Exits must never be stranded by a data outage."""
        rm = RiskManager(RISK_PROFILE)
        result = rm.validate(
            decision=DecisionResult(actions=[
                TradeAction(type="SELL", symbol="AAPL", amount_usd=500,
                            thesis="take profit")]),
            portfolio=make_portfolio([make_position("AAPL", quantity=10, price=100)]),
            quotes={},
        )
        rejected_reasons = [r.rejection_reason or "" for r in result.rejected_actions]
        assert not any("No market quote" in r for r in rejected_reasons)

    def test_buy_with_quote_still_passes(self):
        rm = RiskManager(RISK_PROFILE)
        result = rm.validate(
            decision=DecisionResult(actions=[
                TradeAction(type="BUY", symbol="AAPL", amount_usd=500)]),
            portfolio=make_portfolio(),
            quotes={"AAPL": make_quote("AAPL")},
        )
        assert len(result.approved_actions) == 1


# ── stale-price stop-loss skip ───────────────────────────────────────────────

class TestStaleStopLoss:
    def test_stale_position_not_stop_lossed(self):
        rm = RiskManager({**RISK_PROFILE, "stop_loss_pct": -15})
        # Stale position: valued at avg_cost → P/L reads 0.0%
        stale = make_position("DEAD", quantity=10, price=100, price_stale=True)
        result = rm.validate(
            decision=DecisionResult(actions=[]),
            portfolio=make_portfolio([stale]),
            quotes={},
        )
        assert not any(a.symbol == "DEAD" for a in result.forced_actions)

    def test_fresh_losing_position_still_stop_lossed(self):
        rm = RiskManager({**RISK_PROFILE, "stop_loss_pct": -15})
        loser = make_position("LOSS", quantity=10, price=80, avg_cost=100)
        result = rm.validate(
            decision=DecisionResult(actions=[]),
            portfolio=make_portfolio([loser]),
            quotes={},
        )
        assert any(a.symbol == "LOSS" for a in result.forced_actions)


# ── options commissions ──────────────────────────────────────────────────────

class TestOptionCosts:
    def test_ibkr_single_contract_minimum(self):
        assert calculate_option_cost("ibkr", 1, 1) == 1.00

    def test_ibkr_multi_contract_multi_leg(self):
        # 4 contracts × 2 legs × $0.65 = $5.20
        assert calculate_option_cost("ibkr", 4, 2) == pytest.approx(5.20)

    def test_unknown_broker_free(self):
        assert calculate_option_cost("", 4, 2) == 0.0
        assert calculate_option_cost("robinhood", 1, 1) == 0.0

    def test_close_position_nets_costs(self, tmp_path):
        tracker = OptionsPositionTracker(db_path=tmp_path / "audit.db")
        pid = tracker.open_position(
            account_key="t", symbol="SPY", spread_type="BULL_CALL", contracts=1,
            expiration_date="2026-12-18", buy_strike=500.0, buy_option_type="call",
            buy_premium=10.0, sell_strike=510.0, sell_option_type="call",
            sell_premium=5.0, max_profit=500.0, max_loss=500.0, entry_debit=5.00,
        )
        # gross P/L = (7.00 - 5.00) × 1 × 100 = $200; costs $2.60 → $197.40
        pl = tracker.close_position(pid, 7.00, "TP", costs=2.60)
        assert pl == pytest.approx(197.40)

    def test_expire_position_nets_opening_cost(self, tmp_path):
        tracker = OptionsPositionTracker(db_path=tmp_path / "audit.db")
        pid = tracker.open_position(
            account_key="t", symbol="SPY", spread_type="CASH_SECURED_PUT", contracts=2,
            expiration_date="2026-12-18", buy_strike=0.0, buy_option_type="put",
            buy_premium=0.0, sell_strike=480.0, sell_option_type="put",
            sell_premium=1.50, max_profit=300.0, max_loss=0.0, entry_debit=-1.50,
        )
        tracker.expire_position(pid, costs=1.30)
        with sqlite3.connect(tmp_path / "audit.db") as conn:
            realized = conn.execute(
                "SELECT realized_pl FROM options_positions WHERE id=?", (pid,)
            ).fetchone()[0]
        # full premium 1.50 × 2 × 100 = $300, minus $1.30 opening commission
        assert realized == pytest.approx(298.70)

    def test_get_legs_roundtrip(self, tmp_path):
        """sqlite3.Row has no .get(): the old get_legs raised on every call,
        was swallowed, and silently degraded condors to 2-leg pricing."""
        tracker = OptionsPositionTracker(db_path=tmp_path / "audit.db")
        pid = tracker.open_position(
            account_key="t", symbol="SPY", spread_type="IRON_CONDOR", contracts=1,
            expiration_date="2026-12-18", buy_strike=480.0, buy_option_type="put",
            buy_premium=1.0, sell_strike=490.0, sell_option_type="put",
            sell_premium=2.0, max_profit=200.0, max_loss=800.0, entry_debit=-2.00,
        )
        tracker.save_legs(pid, [
            {"option_type": "put", "side": "buy", "strike": 480.0,
             "premium": 1.0, "contract_symbol": "SPY261218P480"},
            {"option_type": "put", "side": "sell", "strike": 490.0,
             "premium": 2.0, "contract_symbol": "SPY261218P490"},
            {"option_type": "call", "side": "sell", "strike": 520.0,
             "premium": 2.0, "contract_symbol": "SPY261218C520"},
            {"option_type": "call", "side": "buy", "strike": 530.0,
             "premium": 1.0, "contract_symbol": "SPY261218C530"},
        ])
        legs = tracker.get_legs(pid)
        assert len(legs) == 4
        assert [l.side for l in legs] == ["buy", "sell", "sell", "buy"]
        assert legs[1].premium == pytest.approx(2.0)
        assert legs[2].contract_symbol == "SPY261218C520"

    def test_costs_default_zero_backcompat(self, tmp_path):
        tracker = OptionsPositionTracker(db_path=tmp_path / "audit.db")
        pid = tracker.open_position(
            account_key="t", symbol="SPY", spread_type="BULL_CALL", contracts=1,
            expiration_date="2026-12-18", buy_strike=500.0, buy_option_type="call",
            buy_premium=10.0, sell_strike=510.0, sell_option_type="call",
            sell_premium=5.0, max_profit=500.0, max_loss=500.0, entry_debit=5.00,
        )
        assert tracker.close_position(pid, 7.00, "TP") == pytest.approx(200.0)


# ── mid-price execution haircut ──────────────────────────────────────────────

class TestMidPriceHaircut:
    ROW = pd.Series({"bid": 1.00, "ask": 1.40, "lastPrice": 1.15})

    @pytest.mark.parametrize("mid_fn", [wheel_mid, spreads_mid])
    def test_raw_mid_unchanged(self, mid_fn):
        assert mid_fn(self.ROW) == pytest.approx(1.20)

    @pytest.mark.parametrize("mid_fn", [wheel_mid, spreads_mid])
    def test_sell_fills_below_mid(self, mid_fn):
        # half-spread 0.20, haircut 25% → 1.20 - 0.05 = 1.15
        assert mid_fn(self.ROW, "sell") == pytest.approx(1.15)

    @pytest.mark.parametrize("mid_fn", [wheel_mid, spreads_mid])
    def test_buy_fills_above_mid(self, mid_fn):
        assert mid_fn(self.ROW, "buy") == pytest.approx(1.25)

    @pytest.mark.parametrize("mid_fn", [wheel_mid, spreads_mid])
    def test_fallback_lastprice_haircut(self, mid_fn):
        row = pd.Series({"bid": 0.0, "ask": 0.0, "lastPrice": 2.00})
        assert mid_fn(row, "sell") == pytest.approx(1.98)
        assert mid_fn(row, "buy") == pytest.approx(2.02)
        assert mid_fn(row) == pytest.approx(2.00)


# ── fee-aware trade journal ──────────────────────────────────────────────────

class TestJournalFees:
    def test_fees_subtracted_from_realized_pl(self, tmp_path):
        audit = AuditLogger(logs_dir=tmp_path / "logs", db_path=tmp_path / "audit.db")
        # +0.5% gross on a $215 position with $2 round-trip fees = net loss
        audit.log_closed_trade(
            account_key="t", symbol="AAPL", quantity=2.0,
            entry_price=107.50, exit_price=108.04, entry_date="2026-08-01",
            fees=2.00,
        )
        trades = audit.get_trade_journal("t", since="2026-01-01")
        assert len(trades) == 1
        assert trades[0]["realized_pl"] == pytest.approx(-0.92, abs=0.01)
        assert trades[0]["realized_pl_pct"] < 0

    def test_no_fees_backcompat(self, tmp_path):
        audit = AuditLogger(logs_dir=tmp_path / "logs", db_path=tmp_path / "audit.db")
        audit.log_closed_trade(
            account_key="t", symbol="AAPL", quantity=2.0,
            entry_price=100.0, exit_price=110.0, entry_date="2026-08-01",
        )
        trades = audit.get_trade_journal("t", since="2026-01-01")
        assert trades[0]["realized_pl"] == pytest.approx(20.0)


# ── audit-log rig: migrations, carried flag, model_used, benchmark ──────────

class TestAuditRig:
    def test_migration_adds_columns_to_existing_db(self, tmp_path):
        db = tmp_path / "audit.db"
        # Simulate a pre-migration database
        with sqlite3.connect(db) as conn:
            conn.execute("""CREATE TABLE decision_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                account_key TEXT NOT NULL, account_name TEXT NOT NULL,
                model TEXT NOT NULL, market_regime TEXT, portfolio_outlook TEXT,
                confidence REAL, actions_count INTEGER DEFAULT 0,
                forced_actions_count INTEGER DEFAULT 0, rejected_count INTEGER DEFAULT 0,
                portfolio_value REAL, portfolio_pl_pct REAL, cash REAL,
                log_file TEXT, success INTEGER DEFAULT 1, error TEXT)""")
        AuditLogger(logs_dir=tmp_path / "logs", db_path=db)
        with sqlite3.connect(db) as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(decision_log)")}
        assert {"model_used", "valuation_carried", "benchmark_return_pct"} <= cols

    def test_model_used_and_benchmark_persisted(self, tmp_path):
        audit = AuditLogger(logs_dir=tmp_path / "logs", db_path=tmp_path / "audit.db")
        audit.log_cycle(
            account_key="t", account_name="Test", model="Qwen3-Next",
            portfolio_after={"total_value": 10000, "total_pl_pct": 0, "cash": 10000},
            model_used="Nemotron", benchmark_return_pct=2.5,
        )
        with sqlite3.connect(tmp_path / "audit.db") as conn:
            row = conn.execute(
                "SELECT model, model_used, benchmark_return_pct, valuation_carried "
                "FROM decision_log"
            ).fetchone()
        assert row == ("Qwen3-Next", "Nemotron", 2.5, 0)

    def test_carried_valuation_flagged_and_not_reused(self, tmp_path):
        audit = AuditLogger(logs_dir=tmp_path / "logs", db_path=tmp_path / "audit.db")
        # Fresh valuation first
        audit.log_cycle(
            account_key="t", account_name="Test", model="m",
            portfolio_after={"total_value": 9500, "total_pl_pct": -5, "cash": 500},
        )
        # Valuation outage → carried
        audit.log_cycle(
            account_key="t", account_name="Test", model="m",
            portfolio_after={"total_value": 0, "total_pl_pct": None, "cash": None},
        )
        with sqlite3.connect(tmp_path / "audit.db") as conn:
            rows = conn.execute(
                "SELECT portfolio_value, valuation_carried FROM decision_log ORDER BY id"
            ).fetchall()
        assert rows[0] == (9500, 0)
        assert rows[1] == (9500, 1)
        # And the carry source is still the FRESH row, not the carried one
        assert audit._last_known_valuation("t")[0] == 9500


# ── guardian labeling in decision history ────────────────────────────────────

class TestGuardianHistory:
    def test_guardian_cycle_labeled_and_forced_rendered(self):
        history = [{
            "date": "2026-08-18",
            "source": "guardian",
            "outlook": "N/A",
            "confidence": "N/A",
            "actions": [],
            "forced_actions": [{"type": "SELL", "symbol": "PLTR",
                                "amount_usd": 850, "thesis": "STOP-LOSS: -16%"}],
            "hold_reason": "",
        }]
        text = format_decision_history(history)
        assert "RISK GUARDIAN CYCLE" in text
        assert "FORCED BY RISK MGMT: SELL PLTR" in text
        assert "HOLD (no trades)" not in text

    def test_llm_hold_still_renders_hold(self):
        history = [{
            "date": "2026-08-18", "source": "llm", "outlook": "NEUTRAL",
            "confidence": 0.6, "actions": [], "forced_actions": [],
            "hold_reason": "waiting for CPI",
        }]
        text = format_decision_history(history)
        assert "HOLD (no trades)" in text


# ── scanner data gaps ────────────────────────────────────────────────────────

class TestScannerDataGaps:
    def _portfolio(self):
        return make_portfolio(cash=10000)

    def test_missing_prev_price_renders_na(self):
        msgs = build_scan_messages(
            portfolio=self._portfolio(),
            market_data={"AAPL": {"price": 200.0, "change_pct": 1.0}},
            last_cycle_prices={},
            strategy_config={"strategy": "momentum"},
        )
        user = msgs[1]["content"]
        assert "30m Δ=n/a" in user
        assert "Δ=+0.00%" not in user

    def test_missing_price_listed_as_gap(self):
        msgs = build_scan_messages(
            portfolio=self._portfolio(),
            market_data={"AAPL": {"price": 0.0, "change_pct": 0.0},
                         "MSFT": {"price": 400.0, "change_pct": 0.5}},
            last_cycle_prices={"MSFT": 398.0},
            strategy_config={"strategy": "momentum"},
        )
        user = msgs[1]["content"]
        assert "DATA GAPS" in user and "AAPL" in user
        assert "MSFT: $400.00" in user
