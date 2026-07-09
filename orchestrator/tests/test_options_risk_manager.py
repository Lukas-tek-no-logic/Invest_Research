"""Tests for OptionsRiskManager (wheel): collateral reservation and CSP validation."""

from datetime import date, timedelta

from orchestrator.src.options.decision_parser import WheelAction, WheelDecision
from orchestrator.src.options.positions import OptionsPosition
from orchestrator.src.options.risk_manager import OptionsRiskManager
from orchestrator.src.portfolio_state import PortfolioState


RISK_PROFILE = {
    "max_open_csps": 5,
    "max_ccs_per_symbol": 2,
    "min_cash_pct": 5,
    "take_profit_pct": 50,
    "auto_close_dte": 3,
}


def _portfolio(cash: float, total: float | None = None) -> PortfolioState:
    return PortfolioState(
        account_id="acct-1",
        account_name="Wheel Test",
        total_value=total if total is not None else cash,
        cash=cash,
        invested=0,
    )


def _open_csp(id: int, symbol: str, strike: float, contracts: int = 1, dte: int = 30) -> OptionsPosition:
    expiration = (date.today() + timedelta(days=dte)).isoformat()
    return OptionsPosition(
        id=id, account_key="wheel_test", symbol=symbol,
        spread_type="CASH_SECURED_PUT", status="open",
        contracts=contracts, expiration_date=expiration,
        buy_strike=0.0, buy_option_type="put", buy_premium=0.0,
        sell_strike=strike, sell_option_type="put", sell_premium=0.50,
        max_profit=50.0 * contracts, max_loss=strike * 100 * contracts,
        entry_debit=-0.50, entry_date="2026-07-01",
        dte=dte, wheel_state="CSP_OPEN",
    )


def _sell_csp(symbol: str, strike: float, contracts: int = 1) -> WheelAction:
    return WheelAction(
        type="SELL_CSP", symbol=symbol, strike=strike, contracts=contracts,
        reason="high IV, no earnings for 6 weeks",
    )


MARKET_DATA = {
    "F": {"price": 13.50},
    "T": {"price": 21.10},
    "UBER": {"price": 73.60},
    "GM": {"price": 76.20},
}


class TestCollateralReservation:
    def test_open_csp_collateral_reserved(self):
        """Cash committed to an existing CSP cannot back a second one."""
        # $10K cash, open CSP on UBER strike 70 → $7,000 reserved, $3,000 deployable
        mgr = OptionsRiskManager(RISK_PROFILE)
        decision = WheelDecision(actions=[_sell_csp("GM", strike=74.0)])  # needs $7,400

        result = mgr.validate(
            decision=decision,
            active_positions=[_open_csp(1, "UBER", strike=70.0)],
            portfolio=_portfolio(cash=10_000),
            market_data=MARKET_DATA,
        )

        assert result.approved_opens == []
        assert len(result.rejected_opens) == 1
        assert "Collateral" in result.rejected_opens[0]["reason"]

    def test_csp_fits_in_deployable_cash(self):
        """A CSP that fits in cash-minus-reserved is approved."""
        mgr = OptionsRiskManager(RISK_PROFILE)
        decision = WheelDecision(actions=[_sell_csp("F", strike=13.0)])  # needs $1,300

        result = mgr.validate(
            decision=decision,
            active_positions=[_open_csp(1, "UBER", strike=70.0)],  # $7,000 reserved
            portfolio=_portfolio(cash=10_000),
            market_data=MARKET_DATA,
        )

        assert len(result.approved_opens) == 1
        assert result.approved_opens[0].symbol == "F"

    def test_no_reservation_without_open_csps(self):
        mgr = OptionsRiskManager(RISK_PROFILE)
        decision = WheelDecision(actions=[_sell_csp("UBER", strike=70.0)])

        result = mgr.validate(
            decision=decision,
            active_positions=[],
            portfolio=_portfolio(cash=10_000),
            market_data=MARKET_DATA,
        )

        assert len(result.approved_opens) == 1

    def test_csp_being_closed_frees_its_collateral(self):
        """Collateral of a CSP the LLM is closing this cycle is not reserved."""
        mgr = OptionsRiskManager(RISK_PROFILE)
        decision = WheelDecision(actions=[
            WheelAction(type="CLOSE", symbol="UBER", position_id=1,
                        reason="captured 60% of premium"),
            _sell_csp("GM", strike=74.0),  # needs $7,400 — fits once UBER frees $7,000
        ])

        result = mgr.validate(
            decision=decision,
            active_positions=[_open_csp(1, "UBER", strike=70.0)],
            portfolio=_portfolio(cash=10_000),
            market_data=MARKET_DATA,
        )

        assert len(result.approved_closes) == 1
        assert len(result.approved_opens) == 1

    def test_multi_contract_collateral_counted(self):
        """Contracts multiply the reserved collateral."""
        mgr = OptionsRiskManager(RISK_PROFILE)
        decision = WheelDecision(actions=[_sell_csp("GM", strike=74.0)])

        result = mgr.validate(
            decision=decision,
            active_positions=[_open_csp(1, "F", strike=13.0, contracts=4)],  # $5,200
            portfolio=_portfolio(cash=10_000),
            market_data=MARKET_DATA,
        )

        # deployable = 10000 - 5200 = 4800 < 7400 needed
        assert result.approved_opens == []
        assert len(result.rejected_opens) == 1


class TestOffWatchlistRejection:
    def test_no_market_data_gives_explicit_reason(self):
        mgr = OptionsRiskManager(RISK_PROFILE)
        decision = WheelDecision(actions=[
            WheelAction(type="SELL_CSP", symbol="XYZ", contracts=1, reason="looks great"),
        ])

        result = mgr.validate(
            decision=decision,
            active_positions=[],
            portfolio=_portfolio(cash=10_000),
            market_data=MARKET_DATA,
        )

        assert result.approved_opens == []
        assert "not on watchlist" in result.rejected_opens[0]["reason"]
