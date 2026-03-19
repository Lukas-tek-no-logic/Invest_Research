"""Tests for options/spreads_executor.py."""

from unittest.mock import MagicMock, patch

from orchestrator.src.options.spreads_decision_parser import SpreadAction
from orchestrator.src.options.spreads_executor import SpreadsExecutor, SpreadsTradeResult
from orchestrator.src.options.spreads_selector import SelectedSpread, SelectedLeg
from orchestrator.src.options.positions import OptionsPosition


def _make_executor(dry_run=False):
    mock_ghostfolio = MagicMock()
    mock_ghostfolio.create_order.return_value = {"id": "gf-order-123"}
    mock_market_data = MagicMock()
    mock_tracker = MagicMock()
    mock_tracker.open_position.return_value = 42  # new position ID
    mock_tracker.close_position.return_value = 15.50  # realized P&L

    executor = SpreadsExecutor(
        ghostfolio=mock_ghostfolio,
        market_data=mock_market_data,
        tracker=mock_tracker,
        account_id="test-account-id",
        risk_profile={
            "target_dte_min": 21,
            "target_dte_max": 45,
            "max_spread_width": 10,
        },
        dry_run=dry_run,
        account_key="test_key",
    )
    return executor, mock_ghostfolio, mock_tracker


def _make_selected_spread():
    return SelectedSpread(
        symbol="SPY",
        spread_type="iron_condor",
        expiration="2026-04-01",
        dte=35,
        underlying_price=550.0,
        legs=[
            SelectedLeg(option_type="put", strike=530.0, premium=1.50, iv=0.25,
                        delta=-0.15, contract_symbol="SPY260401P530", side="buy"),
            SelectedLeg(option_type="put", strike=535.0, premium=2.50, iv=0.25,
                        delta=-0.20, contract_symbol="SPY260401P535", side="sell"),
            SelectedLeg(option_type="call", strike=565.0, premium=2.50, iv=0.25,
                        delta=0.20, contract_symbol="SPY260401C565", side="sell"),
            SelectedLeg(option_type="call", strike=570.0, premium=1.50, iv=0.25,
                        delta=0.15, contract_symbol="SPY260401C570", side="buy"),
        ],
        net_debit=-2.0,  # credit received
        max_profit=200.0,
        max_loss=300.0,
        contracts=1,
    )


def _make_position(id=1, symbol="SPY", dte=20):
    return OptionsPosition(
        id=id, account_key="test_key", symbol=symbol,
        spread_type="IRON_CONDOR", status="open",
        contracts=1, expiration_date="2026-04-01",
        buy_strike=530.0, buy_option_type="put",
        buy_premium=1.50, sell_strike=535.0,
        sell_option_type="put", sell_premium=2.50,
        max_profit=200.0, max_loss=300.0,
        entry_debit=-2.0, entry_date="2026-02-01",
        dte=dte, current_value=1.0, current_pl=50.0,
    )


class TestSpreadsExecutorOpens:
    """Test execute_opens()."""

    @patch("orchestrator.src.options.spreads_executor.select_spread")
    def test_open_success(self, mock_select):
        mock_select.return_value = _make_selected_spread()
        executor, mock_gf, mock_tracker = _make_executor()

        action = SpreadAction(
            type="OPEN_SPREAD", symbol="SPY",
            spread_type="iron_condor", contracts=1,
            reason="Good setup",
        )
        results = executor.execute_opens([action])

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].action == "OPEN_SPREAD"
        assert results[0].symbol == "SPY"
        assert results[0].position_id == 42
        # Verify tracker was called
        mock_tracker.open_position.assert_called_once()
        mock_tracker.update_position.assert_called_once()
        # Verify Ghostfolio was called
        mock_gf.create_order.assert_called_once()

    @patch("orchestrator.src.options.spreads_executor.select_spread")
    def test_open_no_chain(self, mock_select):
        mock_select.return_value = None
        executor, mock_gf, mock_tracker = _make_executor()

        action = SpreadAction(
            type="OPEN_SPREAD", symbol="SPY",
            spread_type="bull_call", contracts=1,
        )
        results = executor.execute_opens([action])

        assert len(results) == 1
        assert results[0].success is False
        assert "selection failed" in results[0].error
        mock_tracker.open_position.assert_not_called()
        mock_gf.create_order.assert_not_called()

    @patch("orchestrator.src.options.spreads_executor.select_spread")
    def test_open_dry_run(self, mock_select):
        mock_select.return_value = _make_selected_spread()
        executor, mock_gf, mock_tracker = _make_executor(dry_run=True)

        action = SpreadAction(
            type="OPEN_SPREAD", symbol="SPY",
            spread_type="iron_condor", contracts=1,
        )
        results = executor.execute_opens([action])

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].ghostfolio_order_id == "DRY_RUN"
        # Tracker should still be called (records position)
        mock_tracker.open_position.assert_called_once()
        # Ghostfolio should NOT be called in dry-run
        mock_gf.create_order.assert_not_called()


class TestSpreadsExecutorCloses:
    """Test execute_closes()."""

    @patch("orchestrator.src.options.spreads_executor.get_current_option_price")
    def test_close_success(self, mock_price):
        mock_price.return_value = 0.50
        executor, mock_gf, mock_tracker = _make_executor()

        pos = _make_position(id=5)
        action = SpreadAction(
            type="CLOSE", symbol="SPY", position_id=5,
            reason="Take profit",
        )
        results = executor.execute_closes([action], [pos])

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].action == "CLOSE"
        assert results[0].realized_pl == 15.50  # from mock_tracker
        mock_tracker.close_position.assert_called_once()

    def test_close_unknown_position(self):
        executor, mock_gf, mock_tracker = _make_executor()

        action = SpreadAction(
            type="CLOSE", symbol="SPY", position_id=999,
            reason="unknown",
        )
        results = executor.execute_closes([action], [])

        assert len(results) == 1
        assert results[0].success is False
        assert "not found" in results[0].error

    @patch("orchestrator.src.options.spreads_executor.get_current_option_price")
    def test_close_dry_run(self, mock_price):
        mock_price.return_value = 0.50
        executor, mock_gf, mock_tracker = _make_executor(dry_run=True)

        pos = _make_position(id=5)
        action = SpreadAction(
            type="CLOSE", symbol="SPY", position_id=5,
            reason="Take profit",
        )
        results = executor.execute_closes([action], [pos])

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].ghostfolio_order_id == "DRY_RUN"
        mock_gf.create_order.assert_not_called()


class TestSpreadsExecutorUpdates:
    """Test update_active_positions()."""

    @patch("orchestrator.src.options.spreads_executor.get_current_option_price")
    def test_update_credit_spread(self, mock_price):
        """Credit spread: P&L = (|entry_credit| - current_value) * contracts * 100."""
        # Two-leg spread: buy_price=2.0, sell_price=1.0 → current_value = 2.0 - 1.0 = 1.0
        mock_price.side_effect = [2.0, 1.0]
        executor, _, mock_tracker = _make_executor()

        # entry_debit=-2.0 means credit of $2.00 received
        pos = _make_position(id=1, dte=20)
        results = executor.update_active_positions([pos])

        assert len(results) == 1
        assert results[0].success is True
        # Verify tracker was updated with correct P&L
        mock_tracker.update_position.assert_called_once()
        call_args = mock_tracker.update_position.call_args
        assert call_args[0][0] == 1  # position_id
        assert call_args[1]["current_value"] == 1.0
        # P&L for credit: (2.0 - 1.0) * 1 * 100 = 100.0
        assert call_args[1]["current_pl"] == 100.0

    @patch("orchestrator.src.options.spreads_executor.get_current_option_price")
    def test_update_no_price(self, mock_price):
        mock_price.return_value = None
        executor, _, mock_tracker = _make_executor()

        pos = _make_position(id=1, dte=20)
        results = executor.update_active_positions([pos])

        assert len(results) == 1
        assert results[0].success is False
        assert "Could not fetch" in results[0].error


class TestSpreadsExecutorRolls:
    """Test rolls (no-op for spreads)."""

    def test_rolls_noop(self):
        executor, _, _ = _make_executor()
        results = executor.execute_rolls(["something"], [])
        assert results == []


class TestSpreadsExecutorMultiple:
    """Test processing multiple actions."""

    @patch("orchestrator.src.options.spreads_executor.select_spread")
    def test_multiple_opens(self, mock_select):
        mock_select.return_value = _make_selected_spread()
        executor, _, mock_tracker = _make_executor()

        actions = [
            SpreadAction(type="OPEN_SPREAD", symbol="SPY", spread_type="iron_condor", contracts=1),
            SpreadAction(type="OPEN_SPREAD", symbol="AAPL", spread_type="bull_call", contracts=1),
        ]
        results = executor.execute_opens(actions)

        assert len(results) == 2
        assert all(r.success for r in results)
        assert mock_tracker.open_position.call_count == 2

    @patch("orchestrator.src.options.spreads_executor.get_current_option_price")
    def test_mixed_close_results(self, mock_price):
        mock_price.return_value = 0.50
        executor, _, mock_tracker = _make_executor()

        pos = _make_position(id=5)
        actions = [
            SpreadAction(type="CLOSE", symbol="SPY", position_id=5, reason="ok"),
            SpreadAction(type="CLOSE", symbol="AAPL", position_id=999, reason="unknown"),
        ]
        results = executor.execute_closes(actions, [pos])

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False
