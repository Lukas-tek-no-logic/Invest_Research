"""Tests for portfolio_state: value calculation, cash computation, position building."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.portfolio_state import (
    PortfolioState,
    ValuationUnavailable,
    get_portfolio_state,
    compute_cash_from_orders,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

ACCOUNT_ID = "acct-001"
ACCOUNT_NAME = "Test Account"


def _make_ghostfolio(
    balance: float,
    value_in_base: float,
    orders: list[dict] | None = None,
    holdings: list[dict] | None = None,
) -> MagicMock:
    """Build a minimal mock GhostfolioClient."""
    gf = MagicMock()
    gf.list_accounts.return_value = {
        "accounts": [
            {
                "id": ACCOUNT_ID,
                "name": ACCOUNT_NAME,
                "balance": balance,
                "valueInBaseCurrency": value_in_base,
                "currency": "USD",
            }
        ]
    }
    gf.list_orders.return_value = {"activities": orders or []}
    gf.get_portfolio_holdings.return_value = holdings or []
    return gf


def _buy(symbol: str, qty: float, price: float, date: str = "2026-01-10") -> dict:
    return {
        "accountId": ACCOUNT_ID,
        "type": "BUY",
        "quantity": qty,
        "unitPrice": price,
        "fee": 0,
        "date": date,
        "SymbolProfile": {"symbol": symbol},
    }


def _sell(symbol: str, qty: float, price: float, date: str = "2026-01-20") -> dict:
    return {
        "accountId": ACCOUNT_ID,
        "type": "SELL",
        "quantity": qty,
        "unitPrice": price,
        "fee": 0,
        "date": date,
        "SymbolProfile": {"symbol": symbol},
    }


def _holding(symbol: str, price: float) -> dict:
    return {
        "SymbolProfile": {"symbol": symbol},
        "symbol": symbol,
        "marketPrice": price,
        "sectors": [{"name": "Technology"}],
        "name": symbol,
    }


# ── compute_cash_from_orders ──────────────────────────────────────────────────

class TestComputeCashFromOrders:
    def test_empty_account_returns_budget(self):
        gf = MagicMock()
        gf.list_orders.return_value = {"activities": []}
        result = compute_cash_from_orders(gf, ACCOUNT_ID, 10_000)
        assert result == 10_000.0

    def test_single_buy_reduces_cash(self):
        gf = MagicMock()
        gf.list_orders.return_value = {"activities": [_buy("AAPL", 10, 200.0)]}
        result = compute_cash_from_orders(gf, ACCOUNT_ID, 10_000)
        assert result == pytest.approx(10_000 - 10 * 200, abs=0.01)

    def test_buy_then_sell_restores_cash(self):
        gf = MagicMock()
        gf.list_orders.return_value = {"activities": [
            _buy("AAPL", 10, 200.0),
            _sell("AAPL", 10, 220.0),
        ]}
        # bought 10 × $200 = $2000 spent; sold 10 × $220 = $2200 received
        result = compute_cash_from_orders(gf, ACCOUNT_ID, 10_000)
        assert result == pytest.approx(10_000 - 2000 + 2200, abs=0.01)  # $10,200

    def test_orders_from_other_accounts_ignored(self):
        other_order = _buy("SPY", 5, 500.0)
        other_order["accountId"] = "other-acct"
        gf = MagicMock()
        gf.list_orders.return_value = {"activities": [other_order]}
        result = compute_cash_from_orders(gf, ACCOUNT_ID, 10_000)
        assert result == 10_000.0  # unaffected

    def test_cash_never_negative(self):
        # If orders somehow exceed budget, floor at 0
        gf = MagicMock()
        gf.list_orders.return_value = {"activities": [_buy("AAPL", 100, 200.0)]}
        result = compute_cash_from_orders(gf, ACCOUNT_ID, 1_000)
        assert result == 0.0

    def test_with_fees(self):
        order = _buy("AAPL", 10, 200.0)
        order["fee"] = 5.0
        gf = MagicMock()
        gf.list_orders.return_value = {"activities": [order]}
        result = compute_cash_from_orders(gf, ACCOUNT_ID, 10_000)
        assert result == pytest.approx(10_000 - 10 * 200 - 5, abs=0.01)


# ── get_portfolio_state ───────────────────────────────────────────────────────

class TestGetPortfolioState:
    def test_empty_account_cash_only(self):
        """Cash-only account: valueInBaseCurrency ≈ balance → no double-counting."""
        gf = _make_ghostfolio(balance=10_000, value_in_base=10_000)
        state = get_portfolio_state(gf, ACCOUNT_ID, ACCOUNT_NAME)
        assert state.cash == pytest.approx(10_000, abs=1)
        assert state.total_value == pytest.approx(10_000, abs=1)
        assert state.positions == []

    def test_double_count_detection_cash_only(self):
        """Ghostfolio echoes cash as valueInBaseCurrency — must not add twice."""
        gf = _make_ghostfolio(balance=9_999, value_in_base=10_000)
        state = get_portfolio_state(gf, ACCOUNT_ID, ACCOUNT_NAME)
        # Within 0.5% tolerance → detected as double-count
        assert state.total_value == pytest.approx(9_999, abs=1)

    def test_account_with_positions(self):
        """Buy 10 AAPL @ $200; current price $220 → market value $2200."""
        gf = _make_ghostfolio(
            balance=8_000,          # $10K - $2K spent
            value_in_base=2_200,    # current market value of AAPL position
            orders=[_buy("AAPL", 10, 200.0)],
            holdings=[_holding("AAPL", 220.0)],
        )
        state = get_portfolio_state(gf, ACCOUNT_ID, ACCOUNT_NAME)
        assert state.total_value == pytest.approx(8_000 + 2_200, abs=1)
        assert state.cash == pytest.approx(8_000, abs=1)
        assert len(state.positions) == 1
        pos = state.positions[0]
        assert pos.symbol == "AAPL"
        assert pos.quantity == pytest.approx(10.0, abs=0.001)
        assert pos.current_price == pytest.approx(220.0, abs=0.01)
        assert pos.market_value == pytest.approx(2_200, abs=1)
        assert pos.unrealized_pl == pytest.approx(200.0, abs=0.01)  # $2200 - $2000

    def test_buy_then_partial_sell(self):
        """Buy 20 shares, sell 10 → 10 shares remain."""
        gf = _make_ghostfolio(
            balance=7_000,
            value_in_base=1_500,  # 10 shares @ $150
            orders=[
                _buy("SPY", 20, 140.0),
                _sell("SPY", 10, 155.0),
            ],
            holdings=[_holding("SPY", 150.0)],
        )
        state = get_portfolio_state(gf, ACCOUNT_ID, ACCOUNT_NAME)
        assert len(state.positions) == 1
        pos = state.positions[0]
        assert pos.symbol == "SPY"
        assert pos.quantity == pytest.approx(10.0, abs=0.001)
        assert state.total_value == pytest.approx(7_000 + 1_500, abs=1)

    def test_position_fully_sold_not_shown(self):
        """Buy then sell all shares → no open positions."""
        gf = _make_ghostfolio(
            balance=10_200,
            value_in_base=10_200,  # cash-only after full sell → double-count detection
            orders=[
                _buy("VTI", 10, 200.0),
                _sell("VTI", 10, 220.0),
            ],
            holdings=[],
        )
        state = get_portfolio_state(gf, ACCOUNT_ID, ACCOUNT_NAME)
        assert state.positions == []
        # After sell, cash = 10K - 2000 + 2200 = 10200; value_in_base ≈ cash → no double-count
        assert state.total_value == pytest.approx(10_200, abs=1)

    def test_multiple_positions(self):
        """Buy two different stocks, verify both show up."""
        gf = _make_ghostfolio(
            balance=5_000,
            value_in_base=5_100,  # $2100 AAPL + $3000 SPY
            orders=[
                _buy("AAPL", 10, 200.0),
                _buy("SPY", 5, 400.0),
            ],
            holdings=[
                _holding("AAPL", 210.0),
                _holding("SPY", 600.0),
            ],
        )
        state = get_portfolio_state(gf, ACCOUNT_ID, ACCOUNT_NAME)
        syms = {p.symbol for p in state.positions}
        assert "AAPL" in syms
        assert "SPY" in syms
        assert state.cash == pytest.approx(5_000, abs=1)
        assert state.total_value == pytest.approx(5_000 + 5_100, abs=1)

    def test_pl_calculated_correctly(self):
        """P/L = current market value - cost basis."""
        gf = _make_ghostfolio(
            balance=8_000,
            value_in_base=2_500,  # 10 shares @ $250 current
            orders=[_buy("AAPL", 10, 200.0)],
            holdings=[_holding("AAPL", 250.0)],
        )
        state = get_portfolio_state(gf, ACCOUNT_ID, ACCOUNT_NAME)
        assert state.total_pl == pytest.approx(500.0, abs=0.01)  # $2500 - $2000
        assert state.total_pl_pct == pytest.approx(25.0, abs=0.1)  # 25%

    def test_ghostfolio_unavailable_raises(self):
        """If Ghostfolio raises, propagate ValuationUnavailable — never a $0 state."""
        gf = MagicMock()
        gf.list_accounts.side_effect = Exception("connection refused")
        with pytest.raises(ValuationUnavailable):
            get_portfolio_state(gf, ACCOUNT_ID, ACCOUNT_NAME)

    def test_account_missing_raises(self):
        """Account id absent from Ghostfolio account list → ValuationUnavailable."""
        gf = _make_ghostfolio(balance=10_000, value_in_base=10_000)
        with pytest.raises(ValuationUnavailable):
            get_portfolio_state(gf, "no-such-account", ACCOUNT_NAME)
