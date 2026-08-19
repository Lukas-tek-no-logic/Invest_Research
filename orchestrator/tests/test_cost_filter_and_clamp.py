"""Tests for the cost-breakeven filter, SELL clamping and ledger integrity.

These three areas had no coverage at all, which is why a 50x error in the cost
filter threshold, a missing SELL clamp and a silently-hidden oversell deficit all
survived in production.
"""

import pytest

from src.decision_parser import TradeAction
from src.portfolio_state import PortfolioState, Position
from src.risk_manager import RiskManager, filter_by_cost_breakeven
from src.trade_executor import TradeExecutor


# ── helpers ──────────────────────────────────────────────────────────────────

def make_position(
    symbol: str,
    quantity: float = 10.0,
    price: float = 100.0,
    avg_cost: float | None = None,
    price_stale: bool = False,
) -> Position:
    avg_cost = avg_cost if avg_cost is not None else price
    market_value = quantity * price
    investment = quantity * avg_cost
    return Position(
        symbol=symbol,
        name=f"{symbol} Inc",
        quantity=quantity,
        avg_cost=avg_cost,
        current_price=price,
        market_value=market_value,
        unrealized_pl=market_value - investment,
        unrealized_pl_pct=((market_value - investment) / investment * 100) if investment else 0.0,
        sector="Technology",
        price_stale=price_stale,
    )


def make_portfolio(
    positions: list[Position] | None = None,
    cash: float = 5000.0,
    cash_equivalents: list[Position] | None = None,
    oversold: dict[str, float] | None = None,
) -> PortfolioState:
    positions = positions or []
    cash_equivalents = cash_equivalents or []
    total_market = sum(p.market_value for p in (*positions, *cash_equivalents))
    return PortfolioState(
        account_id="acct-1",
        account_name="Test",
        total_value=total_market + cash,
        cash=cash,
        invested=sum(p.avg_cost * p.quantity for p in positions),
        positions=positions,
        cash_equivalents=cash_equivalents,
        oversold=oversold or {},
    )


def buy(symbol: str, amount: float, thesis: str = "discretionary buy") -> TradeAction:
    return TradeAction(type="BUY", symbol=symbol, amount_usd=amount,
                       urgency="MEDIUM", thesis=thesis, exit_condition="n/a")


def sell(symbol: str, amount: float, thesis: str = "discretionary trim") -> TradeAction:
    return TradeAction(type="SELL", symbol=symbol, amount_usd=amount,
                       urgency="MEDIUM", thesis=thesis, exit_condition="n/a")


class FakeMarketData:
    """Returns a fixed execution price, independent of the valuation price."""

    def __init__(self, price: float):
        self.price = price

    def get_current_price(self, symbol: str) -> float:
        return self.price


class FakeGhostfolio:
    def __init__(self):
        self.orders: list[dict] = []

    def create_order(self, **kwargs) -> dict:
        self.orders.append(kwargs)
        return {"id": f"order-{len(self.orders)}"}


# ── cost filter: the threshold itself ────────────────────────────────────────

class TestCostFilterThreshold:
    """multiplier=2.0 must mean 0.5% of trade value, i.e. a $200 floor on ibkr."""

    @pytest.mark.parametrize("amount,expected_pass", [
        (50.0, False),    # $1 fee = 2.00%
        (150.0, False),   # 0.67%
        (199.0, False),   # 0.503%
        (201.0, True),    # 0.498%
        (500.0, True),    # 0.20%
        (1000.0, True),   # 0.10%
    ])
    def test_ibkr_floor_is_200(self, amount, expected_pass):
        portfolio = make_portfolio([make_position("AAPL", quantity=50, price=100)])
        approved, filtered = filter_by_cost_breakeven(
            [buy("AAPL", amount)], portfolio, "ibkr", 2.0,
        )
        assert (len(approved) == 1) is expected_pass
        assert (len(filtered) == 1) is (not expected_pass)

    def test_tiny_trade_is_rejected(self):
        """The regression that mattered: a $10 order paying a 10% fee."""
        portfolio = make_portfolio([make_position("AAPL", quantity=50, price=100)])
        approved, filtered = filter_by_cost_breakeven(
            [buy("AAPL", 10.0)], portfolio, "ibkr", 2.0,
        )
        assert approved == []
        assert len(filtered) == 1
        assert "10.00%" in filtered[0]["reason"]

    def test_higher_multiplier_is_stricter(self):
        portfolio = make_portfolio([make_position("AAPL", quantity=50, price=100)])
        # 0.25% threshold → needs $400
        approved, filtered = filter_by_cost_breakeven(
            [buy("AAPL", 300.0)], portfolio, "ibkr", 4.0,
        )
        assert approved == []
        approved, filtered = filter_by_cost_breakeven(
            [buy("AAPL", 500.0)], portfolio, "ibkr", 4.0,
        )
        assert len(approved) == 1

    def test_no_cost_model_charges_nothing(self):
        portfolio = make_portfolio([make_position("AAPL", quantity=50, price=100)])
        approved, filtered = filter_by_cost_breakeven(
            [buy("AAPL", 10.0)], portfolio, "", 2.0,
        )
        assert len(approved) == 1 and filtered == []

    def test_xtb_percentage_model_passes_small_orders(self):
        """xtb charges 0.08%, comfortably under 0.5% at any size."""
        portfolio = make_portfolio([make_position("AAPL", quantity=50, price=100)])
        approved, _ = filter_by_cost_breakeven(
            [buy("AAPL", 50.0)], portfolio, "xtb", 2.0,
        )
        assert len(approved) == 1


# ── cost filter: what must never be blocked ──────────────────────────────────

class TestCostFilterExemptions:
    """Blocking an exit strands the position until it reaches zero."""

    @pytest.mark.parametrize("thesis", [
        "STOP-LOSS: Position at -18.0% (threshold: -15%)",
        "ZOMBIE CLEANUP: position worth $3.10 — closing to eliminate dead weight",
        "FORCED REDUCTION: Portfolio drawdown exceeds -20.0% threshold",
        "EXPOSURE FLOOR: invested $4,000 < 60% floor in BULL_TREND",
    ])
    def test_mechanical_actions_bypass_filter(self, thesis):
        portfolio = make_portfolio([make_position("XYZ", quantity=1.5, price=100)])
        action = TradeAction(type="SELL", symbol="XYZ", amount_usd=150.0,
                             urgency="HIGH", thesis=thesis, exit_condition="immediate")
        approved, filtered = filter_by_cost_breakeven(
            [action], portfolio, "ibkr", 2.0,
        )
        assert len(approved) == 1, f"mechanical action blocked: {thesis}"
        assert filtered == []

    def test_full_exit_bypasses_filter(self):
        """A $150 position must stay closable even though $1 is 0.67% of it."""
        portfolio = make_portfolio([make_position("XYZ", quantity=1.5, price=100)])
        approved, filtered = filter_by_cost_breakeven(
            [sell("XYZ", 150.0)], portfolio, "ibkr", 2.0,
        )
        assert len(approved) == 1
        assert filtered == []

    def test_partial_trim_of_small_position_is_still_filtered(self):
        """Half of a $150 position is discretionary, not an exit."""
        portfolio = make_portfolio([make_position("XYZ", quantity=1.5, price=100)])
        approved, filtered = filter_by_cost_breakeven(
            [sell("XYZ", 75.0)], portfolio, "ibkr", 2.0,
        )
        assert approved == []
        assert len(filtered) == 1

    def test_full_exit_of_parked_tbills_bypasses_filter(self):
        """BIL lives in cash_equivalents; the filter must still see it."""
        portfolio = make_portfolio(
            positions=[],
            cash_equivalents=[make_position("BIL", quantity=2.0, price=91.5)],
        )
        approved, filtered = filter_by_cost_breakeven(
            [sell("BIL", 183.0, thesis="CASH SWEEP-OUT")], portfolio, "ibkr", 2.0,
        )
        assert len(approved) == 1
        assert filtered == []

    def test_unknown_price_fails_open(self):
        """No price is no basis for rejection — fabricating one rejects real trades."""
        portfolio = make_portfolio([])
        approved, filtered = filter_by_cost_breakeven(
            [buy("NEWCO", 50.0)], portfolio, "ibkr", 2.0,
        )
        assert len(approved) == 1
        assert filtered == []


# ── SELL clamping in the executor ────────────────────────────────────────────

class TestSellClamp:
    """`amount_usd` is priced off Ghostfolio, the divisor off yfinance."""

    def test_sell_clamped_to_held_quantity(self):
        gf = FakeGhostfolio()
        # Valued at $100/share (10 shares = $1000), executing at $90.
        # 1000/90 = 11.11 shares — more than the account owns.
        executor = TradeExecutor(gf, FakeMarketData(90.0), broker_cost_model="ibkr")
        results = executor.execute_trades(
            [sell("XYZ", 1000.0)], "acct-1", holdings={"XYZ": 10.0},
        )
        assert results[0].success
        assert results[0].quantity == pytest.approx(10.0)
        assert gf.orders[0]["quantity"] == pytest.approx(10.0)

    def test_sell_below_holding_is_untouched(self):
        gf = FakeGhostfolio()
        executor = TradeExecutor(gf, FakeMarketData(100.0), broker_cost_model="ibkr")
        results = executor.execute_trades(
            [sell("XYZ", 500.0)], "acct-1", holdings={"XYZ": 10.0},
        )
        assert results[0].quantity == pytest.approx(5.0)

    def test_sell_without_holding_is_refused(self):
        gf = FakeGhostfolio()
        executor = TradeExecutor(gf, FakeMarketData(100.0), broker_cost_model="ibkr")
        results = executor.execute_trades(
            [sell("GHOST", 500.0)], "acct-1", holdings={"XYZ": 10.0},
        )
        assert results[0].success is False
        assert "holds no shares" in results[0].error
        assert gf.orders == []

    def test_omitting_holdings_disables_clamp(self):
        """Back-compat: callers with no position data must keep working."""
        gf = FakeGhostfolio()
        executor = TradeExecutor(gf, FakeMarketData(90.0), broker_cost_model="ibkr")
        results = executor.execute_trades([sell("XYZ", 1000.0)], "acct-1")
        assert results[0].success
        assert results[0].quantity == pytest.approx(1000.0 / 90.0, rel=1e-4)

    def test_parked_tbills_are_sellable(self):
        """Regression guard: BIL sweep-out must not be read as a zero position."""
        gf = FakeGhostfolio()
        executor = TradeExecutor(gf, FakeMarketData(91.5), broker_cost_model="ibkr")
        portfolio = make_portfolio(
            positions=[],
            cash_equivalents=[make_position("BIL", quantity=20.0, price=91.5)],
        )
        results = executor.execute_trades(
            [sell("BIL", 500.0)], "acct-1", holdings=portfolio.holdings_map(),
        )
        assert results[0].success, results[0].error
        assert results[0].quantity == pytest.approx(500.0 / 91.5, rel=1e-4)

    def test_quantity_is_truncated_not_rounded(self):
        """Rounding up could re-create the short the clamp just removed."""
        gf = FakeGhostfolio()
        executor = TradeExecutor(gf, FakeMarketData(3.0), broker_cost_model="ibkr")
        # 10/3 = 3.3333333... → must floor to 3.333333, never 3.333334
        results = executor.execute_trades(
            [sell("XYZ", 10.0)], "acct-1", holdings={"XYZ": 100.0},
        )
        assert results[0].quantity == pytest.approx(3.333333, abs=1e-9)
        assert results[0].quantity <= 10.0 / 3.0


# ── forced-action dedup and visibility warnings ──────────────────────────────

class TestForcedActionHygiene:
    PROFILE = {
        "max_position_pct": 20, "min_cash_pct": 10, "max_trades_per_cycle": 5,
        "stop_loss_pct": -15, "min_holding_days": 0,
    }

    def test_zombie_and_stop_loss_on_same_symbol_dedupe(self):
        """Both rules fire on a tiny loser; freed_cash must not double-count it."""
        from src.decision_parser import DecisionResult
        pos = make_position("ZOM", quantity=1.0, price=3.0, avg_cost=10.0)
        portfolio = make_portfolio([pos])
        result = RiskManager(self.PROFILE).validate(
            decision=DecisionResult(actions=[]), portfolio=portfolio, quotes={},
        )
        zom_sells = [a for a in result.forced_actions if a.symbol == "ZOM" and a.type == "SELL"]
        assert len(zom_sells) == 1, f"expected one forced sell, got {len(zom_sells)}"
        assert any("DEDUPED" in m for m in result.modifications)

    def test_stale_price_raises_a_warning(self):
        from src.decision_parser import DecisionResult
        portfolio = make_portfolio([make_position("DARK", price_stale=True)])
        result = RiskManager(self.PROFILE).validate(
            decision=DecisionResult(actions=[]), portfolio=portfolio, quotes={},
        )
        assert any("NO MARKET PRICE" in w and "DARK" in w for w in result.warnings)

    def test_phantom_short_raises_a_warning(self):
        from src.decision_parser import DecisionResult
        portfolio = make_portfolio([make_position("AAPL")], oversold={"RDW": 4.7437})
        result = RiskManager(self.PROFILE).validate(
            decision=DecisionResult(actions=[]), portfolio=portfolio, quotes={},
        )
        assert any("PHANTOM SHORT" in w and "RDW" in w for w in result.warnings)


# ── minimum order size ───────────────────────────────────────────────────────

class TestMinOrderUsd:
    def test_small_buy_rejected_when_minimum_set(self):
        from src.decision_parser import DecisionResult
        profile = {"max_position_pct": 50, "min_cash_pct": 10,
                   "max_trades_per_cycle": 5, "min_order_usd": 400}
        portfolio = make_portfolio([], cash=5000.0)
        result = RiskManager(profile).validate(
            decision=DecisionResult(actions=[buy("AAPL", 100.0)]),
            portfolio=portfolio, quotes={},
        )
        assert result.approved_actions == []
        assert any("too small" in r.rejection_reason.lower() for r in result.rejected_actions)

    def test_minimum_does_not_block_sells(self):
        """A position that fell below the minimum must remain exitable."""
        from src.decision_parser import DecisionResult
        profile = {"max_position_pct": 50, "min_cash_pct": 10, "max_trades_per_cycle": 5,
                   "min_order_usd": 400, "stop_loss_pct": -90, "min_holding_days": 0}
        pos = make_position("SMALL", quantity=1.0, price=100.0)
        portfolio = make_portfolio([pos], cash=1000.0)
        result = RiskManager(profile).validate(
            decision=DecisionResult(actions=[sell("SMALL", 100.0)]),
            portfolio=portfolio, quotes={},
        )
        assert len(result.approved_actions) == 1
        assert result.approved_actions[0].type == "SELL"
