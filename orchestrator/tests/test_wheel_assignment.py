"""Tests for the wheel assignment path: CSP assignment → covered call → call-away.

This code path never ran in production (the wheel accounts could not open CSPs
until 2026-07-09), so it is covered here before the first real expiration.
"""

from datetime import date
from unittest.mock import MagicMock

from orchestrator.src.options.executor import OptionsExecutor
from orchestrator.src.options.positions import OptionsPosition, OptionsPositionTracker


RISK_PROFILE = {"take_profit_pct": 50, "auto_close_dte": 3}


# ── Tracker state-machine tests (real SQLite in tmp dir) ─────────────────────

def _make_tracker(tmp_path) -> OptionsPositionTracker:
    return OptionsPositionTracker(db_path=tmp_path / "audit.db")


def _open_csp(tracker, symbol="F", strike=13.0, premium=0.35, contracts=1) -> int:
    return tracker.open_position(
        account_key="wheel_test",
        symbol=symbol,
        spread_type="CASH_SECURED_PUT",
        contracts=contracts,
        expiration_date=date.today().isoformat(),
        buy_strike=0.0, buy_option_type="put", buy_premium=0.0,
        sell_strike=strike, sell_option_type="put", sell_premium=premium,
        max_profit=premium * 100 * contracts,
        max_loss=strike * 100 * contracts,
        entry_debit=-premium,   # credit received
        ghostfolio_order_id="gf-open-1",
    )


class TestTrackerWheelStateMachine:
    def test_assign_position_sets_cost_basis_and_state(self, tmp_path):
        tracker = _make_tracker(tmp_path)
        pos_id = _open_csp(tracker, strike=13.0, premium=0.35)

        cost_basis = tracker.assign_position(pos_id, stock_price=12.40)

        # cost basis = strike - premium received
        assert cost_basis == 12.65
        assigned = tracker.get_assigned_positions("wheel_test")
        assert len(assigned) == 1
        assert assigned[0].wheel_state == "ASSIGNED"
        assert assigned[0].wheel_cost_basis == 12.65
        assert assigned[0].wheel_shares == 100

    def test_expire_position_keeps_full_premium(self, tmp_path):
        tracker = _make_tracker(tmp_path)
        pos_id = _open_csp(tracker, premium=0.35, contracts=2)

        tracker.expire_position(pos_id)

        active = tracker.get_active_positions("wheel_test")
        assert active == []
        # realized = -entry_debit * contracts * 100 = 0.35 * 2 * 100
        import sqlite3
        with sqlite3.connect(tmp_path / "audit.db") as conn:
            row = conn.execute(
                "SELECT status, realized_pl, wheel_state FROM options_positions WHERE id=?",
                (pos_id,),
            ).fetchone()
        assert row[0] == "expired"
        assert row[1] == 70.0
        assert row[2] == "COMPLETE"

    def test_call_away_full_wheel_pl(self, tmp_path):
        tracker = _make_tracker(tmp_path)
        # CC position with wheel context: cost basis 12.65, 100 shares
        cc_id = tracker.open_position(
            account_key="wheel_test", symbol="F",
            spread_type="COVERED_CALL", contracts=1,
            expiration_date=date.today().isoformat(),
            buy_strike=0.0, buy_option_type="call", buy_premium=0.0,
            sell_strike=14.0, sell_option_type="call", sell_premium=0.25,
            max_profit=25.0, max_loss=0.0, entry_debit=-0.25,
        )
        import sqlite3
        with sqlite3.connect(tmp_path / "audit.db") as conn:
            conn.execute(
                "UPDATE options_positions SET wheel_cost_basis=12.65, wheel_shares=100 WHERE id=?",
                (cc_id,),
            )

        realized = tracker.call_away_position(cc_id, cc_strike=14.0, cc_premium=0.25)

        # (14.00 - 12.65) * 100 + 0.25 * 100 = 135 + 25 = 160
        assert realized == 160.0
        with sqlite3.connect(tmp_path / "audit.db") as conn:
            row = conn.execute(
                "SELECT status, wheel_state FROM options_positions WHERE id=?", (cc_id,),
            ).fetchone()
        assert row == ("closed", "COMPLETE")


# ── Executor expiry-day handling (mocked tracker/ghostfolio) ─────────────────

def _make_executor(spot_price: float, dry_run=False):
    ghostfolio = MagicMock()
    ghostfolio.create_order.return_value = {"id": "gf-123"}
    market_data = MagicMock()
    market_data.get_current_price.return_value = spot_price
    tracker = MagicMock()
    tracker.assign_position.return_value = 12.65
    tracker.call_away_position.return_value = 160.0
    executor = OptionsExecutor(
        ghostfolio=ghostfolio,
        market_data=market_data,
        tracker=tracker,
        account_id="acct-1",
        risk_profile=RISK_PROFILE,
        dry_run=dry_run,
        account_key="wheel_test",
    )
    return executor, ghostfolio, tracker


def _csp_position(strike=13.0) -> OptionsPosition:
    return OptionsPosition(
        id=7, account_key="wheel_test", symbol="F",
        spread_type="CASH_SECURED_PUT", status="open",
        contracts=1, expiration_date=date.today().isoformat(),
        buy_strike=0.0, buy_option_type="put", buy_premium=0.0,
        sell_strike=strike, sell_option_type="put", sell_premium=0.35,
        max_profit=35.0, max_loss=strike * 100, entry_debit=-0.35,
        entry_date="2026-06-15", dte=0, wheel_state="CSP_OPEN",
    )


def _cc_position(strike=14.0) -> OptionsPosition:
    return OptionsPosition(
        id=8, account_key="wheel_test", symbol="F",
        spread_type="COVERED_CALL", status="open",
        contracts=1, expiration_date=date.today().isoformat(),
        buy_strike=0.0, buy_option_type="call", buy_premium=0.0,
        sell_strike=strike, sell_option_type="call", sell_premium=0.25,
        max_profit=25.0, max_loss=0.0, entry_debit=-0.25,
        entry_date="2026-06-15", dte=0,
        wheel_state="CC_OPEN", wheel_cost_basis=12.65, wheel_shares=100,
    )


class TestExecutorExpiryDay:
    def test_itm_csp_assigns_and_settles_synthetic(self):
        """Spot below strike at expiry → assignment: stock BUY + synthetic closed at $0."""
        executor, ghostfolio, tracker = _make_executor(spot_price=12.40)

        results = executor.update_active_positions([_csp_position(strike=13.0)])

        assert results[0].action == "ASSIGNMENT"
        assert results[0].success is True
        tracker.assign_position.assert_called_once()
        orders = [c.kwargs for c in ghostfolio.create_order.call_args_list]
        # Stock BUY at strike
        stock = [o for o in orders if o["symbol"] == "F"]
        assert len(stock) == 1
        assert stock[0]["order_type"] == "BUY"
        assert stock[0]["quantity"] == 100.0
        assert stock[0]["unit_price"] == 13.0
        # Synthetic GF_WHEEL settled (no phantom holding left open)
        synthetic = [o for o in orders if o["symbol"].startswith("GF_WHEEL")]
        assert len(synthetic) == 1
        assert synthetic[0]["order_type"] == "BUY"
        assert synthetic[0]["unit_price"] == 0.0

    def test_otm_csp_expires_and_settles_synthetic(self):
        """Spot above strike at expiry → worthless: expire + synthetic closed at $0."""
        executor, ghostfolio, tracker = _make_executor(spot_price=13.80)

        results = executor.update_active_positions([_csp_position(strike=13.0)])

        assert results[0].success is True
        tracker.expire_position.assert_called_once()
        tracker.assign_position.assert_not_called()
        orders = [c.kwargs for c in ghostfolio.create_order.call_args_list]
        assert len(orders) == 1
        assert orders[0]["symbol"].startswith("GF_WHEEL")
        assert orders[0]["unit_price"] == 0.0

    def test_itm_cc_called_away_and_settles_synthetic(self):
        """Spot above CC strike at expiry → shares sold + synthetic closed at $0."""
        executor, ghostfolio, tracker = _make_executor(spot_price=14.90)

        results = executor.update_active_positions([_cc_position(strike=14.0)])

        assert results[0].action == "CALLED_AWAY"
        assert results[0].realized_pl == 160.0
        tracker.call_away_position.assert_called_once()
        orders = [c.kwargs for c in ghostfolio.create_order.call_args_list]
        stock = [o for o in orders if o["symbol"] == "F"]
        assert stock[0]["order_type"] == "SELL"
        assert stock[0]["unit_price"] == 14.0
        synthetic = [o for o in orders if o["symbol"].startswith("GF_WHEEL")]
        assert len(synthetic) == 1

    def test_dry_run_never_touches_ghostfolio(self):
        executor, ghostfolio, tracker = _make_executor(spot_price=12.40, dry_run=True)

        results = executor.update_active_positions([_csp_position(strike=13.0)])

        assert results[0].action == "ASSIGNMENT"
        ghostfolio.create_order.assert_not_called()
