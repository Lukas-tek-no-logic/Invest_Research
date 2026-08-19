"""Tests for ledger integrity in portfolio_state.

The order replay in `get_portfolio_state` had three defects that no test covered:
it never sorted orders by date, it took `first_buy_date` from whichever order
appeared first regardless of type, and it clamped oversells to zero so the
resulting deficit was invisible to every downstream rule.
"""

from unittest.mock import MagicMock

import pytest

from src.portfolio_state import get_portfolio_state

ACCOUNT_ID = "acct-001"
ACCOUNT_NAME = "Test Account"


def _make_ghostfolio(balance, value_in_base, orders=None, holdings=None) -> MagicMock:
    gf = MagicMock()
    gf.list_accounts.return_value = {
        "accounts": [{
            "id": ACCOUNT_ID, "name": ACCOUNT_NAME, "balance": balance,
            "valueInBaseCurrency": value_in_base, "currency": "USD",
        }]
    }
    gf.list_orders.return_value = {"activities": orders or []}
    gf.get_portfolio_holdings.return_value = holdings or []
    return gf


def _order(otype, symbol, qty, price, date):
    return {
        "accountId": ACCOUNT_ID, "type": otype, "quantity": qty,
        "unitPrice": price, "fee": 0, "date": date,
        "SymbolProfile": {"symbol": symbol},
    }


def _holding(symbol, price):
    return {
        "SymbolProfile": {"symbol": symbol}, "symbol": symbol,
        "marketPrice": price, "sectors": [{"name": "Technology"}], "name": symbol,
    }


class TestOrderOrdering:
    def test_sell_listed_before_its_buy_is_still_applied(self):
        """Ghostfolio does not promise chronological order.

        Unsorted, the SELL hits an empty aggregate, is dropped, and the position
        reads as if nothing was ever sold.
        """
        orders = [
            _order("SELL", "AAPL", 4, 110, "2026-02-10"),
            _order("BUY", "AAPL", 10, 100, "2026-01-10"),
        ]
        gf = _make_ghostfolio(1000, 660, orders, [_holding("AAPL", 110)])
        state = get_portfolio_state(gf, ACCOUNT_ID, ACCOUNT_NAME)
        pos = state.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == pytest.approx(6.0), "SELL was dropped — orders not sorted"

    def test_first_buy_date_is_the_earliest_buy(self):
        """Holding period must not restart when a position is topped up."""
        orders = [
            _order("BUY", "AAPL", 5, 100, "2026-01-10"),
            _order("BUY", "AAPL", 5, 120, "2026-06-01"),
        ]
        gf = _make_ghostfolio(1000, 1100, orders, [_holding("AAPL", 110)])
        state = get_portfolio_state(gf, ACCOUNT_ID, ACCOUNT_NAME)
        assert state.get_position("AAPL").first_buy_date == "2026-01-10"

    def test_first_buy_date_ignores_sells(self):
        orders = [
            _order("BUY", "AAPL", 10, 100, "2026-03-01"),
            _order("SELL", "AAPL", 2, 110, "2026-04-01"),
        ]
        gf = _make_ghostfolio(1000, 880, orders, [_holding("AAPL", 110)])
        state = get_portfolio_state(gf, ACCOUNT_ID, ACCOUNT_NAME)
        assert state.get_position("AAPL").first_buy_date == "2026-03-01"


class TestOversellVisibility:
    def test_oversell_is_recorded_not_swallowed(self):
        orders = [
            _order("BUY", "RDW", 10, 50, "2026-01-10"),
            _order("SELL", "RDW", 14.74, 40, "2026-02-10"),
        ]
        gf = _make_ghostfolio(1000, 0, orders, [_holding("RDW", 40)])
        state = get_portfolio_state(gf, ACCOUNT_ID, ACCOUNT_NAME)
        assert "RDW" in state.oversold
        assert state.oversold["RDW"] == pytest.approx(4.74)
        # Position itself is gone — which is exactly why the deficit had to be
        # surfaced separately rather than left on the position.
        assert state.get_position("RDW") is None

    def test_deficit_accumulates_across_round_trips(self):
        """The clamp let small oversells compound invisibly over many cycles."""
        orders = []
        for i in range(4):
            orders.append(_order("BUY", "OPEN", 10, 10, f"2026-0{i + 1}-05"))
            orders.append(_order("SELL", "OPEN", 10.5, 9.5, f"2026-0{i + 1}-20"))
        gf = _make_ghostfolio(1000, 0, orders, [_holding("OPEN", 9.5)])
        state = get_portfolio_state(gf, ACCOUNT_ID, ACCOUNT_NAME)
        assert state.oversold["OPEN"] == pytest.approx(2.0), state.oversold

    def test_clean_history_reports_no_deficit(self):
        orders = [
            _order("BUY", "AAPL", 10, 100, "2026-01-10"),
            _order("SELL", "AAPL", 10, 110, "2026-02-10"),
        ]
        gf = _make_ghostfolio(1100, 0, orders, [_holding("AAPL", 110)])
        state = get_portfolio_state(gf, ACCOUNT_ID, ACCOUNT_NAME)
        assert state.oversold == {}


class TestStalePrice:
    def test_missing_market_price_is_flagged(self):
        """Falling back to avg_cost yields P/L 0.0% and disables the stop-loss."""
        orders = [_order("BUY", "DARK", 10, 100, "2026-01-10")]
        gf = _make_ghostfolio(1000, 1000, orders, holdings=[])  # no price available
        state = get_portfolio_state(gf, ACCOUNT_ID, ACCOUNT_NAME)
        pos = state.get_position("DARK")
        assert pos.price_stale is True
        assert pos.current_price == pytest.approx(100.0)
        assert pos.unrealized_pl_pct == pytest.approx(0.0)

    def test_priced_position_is_not_flagged(self):
        orders = [_order("BUY", "AAPL", 10, 100, "2026-01-10")]
        gf = _make_ghostfolio(1000, 1100, orders, [_holding("AAPL", 110)])
        state = get_portfolio_state(gf, ACCOUNT_ID, ACCOUNT_NAME)
        assert state.get_position("AAPL").price_stale is False


class TestTotalReturn:
    def test_return_since_inception_survives_full_exits(self):
        """`total_pl_pct` only sees held lots, so a rotating account reads ~0%."""
        orders = [
            _order("BUY", "AAPL", 10, 100, "2026-01-10"),
            _order("SELL", "AAPL", 10, 80, "2026-02-10"),  # realised -$200
        ]
        gf = _make_ghostfolio(9800, 0, orders, [_holding("AAPL", 80)])
        state = get_portfolio_state(gf, ACCOUNT_ID, ACCOUNT_NAME, initial_budget=10_000)
        assert state.total_pl_pct == pytest.approx(0.0), "no open lots left"
        assert state.total_return_pct == pytest.approx(-2.0)

    def test_return_is_none_without_a_budget(self):
        orders = [_order("BUY", "AAPL", 10, 100, "2026-01-10")]
        gf = _make_ghostfolio(1000, 1100, orders, [_holding("AAPL", 110)])
        state = get_portfolio_state(gf, ACCOUNT_ID, ACCOUNT_NAME)
        assert state.total_return_pct is None

    def test_total_return_appears_in_prompt(self):
        orders = [_order("BUY", "AAPL", 10, 100, "2026-01-10")]
        gf = _make_ghostfolio(500, 1100, orders, [_holding("AAPL", 110)])
        state = get_portfolio_state(gf, ACCOUNT_ID, ACCOUNT_NAME, initial_budget=2000)
        text = state.to_prompt_text()
        assert "TOTAL RETURN since inception" in text


class TestHoldingsMap:
    def test_holdings_map_includes_parked_tbills(self):
        orders = [
            _order("BUY", "AAPL", 10, 100, "2026-01-10"),
            _order("BUY", "BIL", 20, 91.5, "2026-01-11"),
        ]
        gf = _make_ghostfolio(500, 2930, orders,
                              [_holding("AAPL", 110), _holding("BIL", 91.5)])
        state = get_portfolio_state(gf, ACCOUNT_ID, ACCOUNT_NAME)
        # BIL is deliberately hidden from `positions` / `get_position`...
        assert state.get_position("BIL") is None
        # ...but must be visible for clamping and exits.
        assert state.get_holding("BIL") is not None
        assert state.holdings_map()["BIL"] == pytest.approx(20.0)
        assert state.holdings_map()["AAPL"] == pytest.approx(10.0)
