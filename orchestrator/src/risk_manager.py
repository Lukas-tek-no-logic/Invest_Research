"""Risk manager: validates and modifies trade decisions against hard rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import structlog

from .decision_parser import TradeAction, DecisionResult
from .portfolio_state import PortfolioState
from .market_data import StockQuote
from .transaction_costs import calculate_cost

logger = structlog.get_logger()

MIN_PRICE = 5.0
MIN_AVG_DAILY_VOLUME_USD = 100_000
MAX_PORTFOLIO_DRAWDOWN_PCT = -20.0

# Pairs of highly correlated assets (buying both in same cycle is redundant)
CORRELATED_PAIRS = [
    {"VTI", "VOO"},
    {"SPY", "VOO"},
    {"SPY", "VTI"},
    {"QQQ", "TQQQ"},
    {"SOXL", "NVDA"},
    {"MARA", "COIN"},
    {"IWM", "VB"},
]


@dataclass
class RiskCheckResult:
    """Result of risk validation for a single action."""
    action: TradeAction
    approved: bool = True
    modified: bool = False
    original_amount: float = 0.0
    rejection_reason: str = ""
    modification_reason: str = ""


@dataclass
class RiskManagerResult:
    """Full result of risk validation pass."""
    approved_actions: list[TradeAction] = field(default_factory=list)
    rejected_actions: list[RiskCheckResult] = field(default_factory=list)
    forced_actions: list[TradeAction] = field(default_factory=list)
    modifications: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class RiskManager:
    """Validates trade decisions against account-specific risk rules."""

    def __init__(self, risk_profile: dict, sim_date: str | None = None):
        self.max_position_pct = risk_profile.get("max_position_pct", 20)
        self.min_cash_pct = risk_profile.get("min_cash_pct", 10)
        self.max_trades_per_cycle = risk_profile.get("max_trades_per_cycle", 5)
        self.stop_loss_pct = risk_profile.get("stop_loss_pct", -15)
        self.min_holding_days = risk_profile.get("min_holding_days", 0)
        self.min_holding_hours = risk_profile.get("min_holding_hours", 0)
        self.min_order_usd = risk_profile.get("min_order_usd", 0)
        self.max_sector_exposure_pct = risk_profile.get("max_sector_exposure_pct", 40)
        self._sim_date = sim_date  # None = use datetime.now()

    def validate(
        self,
        decision: DecisionResult,
        portfolio: PortfolioState,
        quotes: dict[str, StockQuote],
        order_history: list[dict] | None = None,
    ) -> RiskManagerResult:
        """Run all risk checks on the decision.

        Order of operations:
          1. Check for forced stop-loss sells
          2. Check portfolio-level drawdown
          3. Validate each action against rules
          4. Trim to max trades (drop lowest urgency first)
        """
        result = RiskManagerResult()

        # 1. Check stop-losses BEFORE model actions
        forced_sells = self._check_stop_losses(portfolio)
        result.forced_actions.extend(forced_sells)
        if forced_sells:
            symbols = [a.symbol for a in forced_sells]
            result.warnings.append(f"STOP-LOSS triggered for: {', '.join(symbols)}")

        # 2. Check portfolio drawdown
        if portfolio.total_pl_pct <= MAX_PORTFOLIO_DRAWDOWN_PCT:
            result.warnings.append(
                f"CRITICAL: Portfolio drawdown {portfolio.total_pl_pct:.1f}% exceeds "
                f"{MAX_PORTFOLIO_DRAWDOWN_PCT}% threshold. Forcing 50% exposure reduction."
            )
            forced_reduce = self._force_reduce_exposure(portfolio, 0.5)
            result.forced_actions.extend(forced_reduce)

        # Bootstrap mode: if mostly cash, allow more trades to deploy capital faster
        effective_max_trades = self.max_trades_per_cycle
        if portfolio.cash_pct > 80:
            effective_max_trades = min(self.max_trades_per_cycle * 2, 10)
            result.warnings.append(
                f"BOOTSTRAP MODE: cash={portfolio.cash_pct:.0f}% > 80% — "
                f"allowing up to {effective_max_trades} trades this cycle"
            )

        # Correlation check: warn if buying highly correlated assets simultaneously
        buy_symbols = {a.symbol for a in decision.actions if a.type == "BUY"}
        for pair in CORRELATED_PAIRS:
            if pair.issubset(buy_symbols):
                result.warnings.append(
                    f"CORRELATION: {' + '.join(sorted(pair))} are highly correlated (~0.99). "
                    f"Consider choosing just one to avoid redundant exposure."
                )

        # 3. Validate each model action
        # Process SELLs first so freed cash is available for BUYs
        forced_sell_symbols = {a.symbol for a in result.forced_actions if a.type == "SELL"}
        sells = [a for a in decision.actions if a.type == "SELL" and a.symbol not in forced_sell_symbols]
        buys = [a for a in decision.actions if a.type == "BUY"]
        skipped_sells = [a for a in decision.actions if a.type == "SELL" and a.symbol in forced_sell_symbols]
        for action in skipped_sells:
            result.modifications.append(
                f"SKIPPED {action.type} {action.symbol}: already covered by forced stop-loss"
            )

        validated = []
        # Track cash freed by approved sells (including forced) for buy validation
        freed_cash = sum(a.amount_usd for a in result.forced_actions if a.type == "SELL")
        for action in sells:
            check = self._validate_action(action, portfolio, quotes, order_history)
            if check.approved:
                validated.append(check)
                freed_cash += check.action.amount_usd
            else:
                result.rejected_actions.append(check)
                result.modifications.append(
                    f"REJECTED {action.type} {action.symbol} ${action.amount_usd:.0f}: "
                    f"{check.rejection_reason}"
                )

        # Validate BUYs with adjusted portfolio (cash + freed_cash from sells)
        if freed_cash > 0:
            # Create adjusted portfolio snapshot for buy validation
            adjusted_portfolio = portfolio.with_extra_cash(freed_cash)
            result.warnings.append(
                f"Sell-then-buy: ${freed_cash:,.0f} freed from sells, "
                f"effective buying power: ${max(0, adjusted_portfolio.cash - adjusted_portfolio.total_value * self.min_cash_pct / 100):,.0f}"
            )
        else:
            adjusted_portfolio = portfolio

        for action in buys:
            check = self._validate_action(action, adjusted_portfolio, quotes, order_history)
            if check.approved:
                validated.append(check)
            else:
                result.rejected_actions.append(check)
                result.modifications.append(
                    f"REJECTED {action.type} {action.symbol} ${action.amount_usd:.0f}: "
                    f"{check.rejection_reason}"
                )

        # 4. Trim to max trades per cycle (keep highest urgency)
        urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        validated.sort(key=lambda c: urgency_order.get(c.action.urgency, 1))

        for check in validated[:effective_max_trades]:
            result.approved_actions.append(check.action)
            if check.modified:
                result.modifications.append(
                    f"MODIFIED {check.action.type} {check.action.symbol}: "
                    f"${check.original_amount:.0f} -> ${check.action.amount_usd:.0f} "
                    f"({check.modification_reason})"
                )

        for check in validated[effective_max_trades:]:
            check.approved = False
            check.rejection_reason = f"Exceeds max {effective_max_trades} trades/cycle"
            result.rejected_actions.append(check)
            result.modifications.append(
                f"REJECTED {check.action.type} {check.action.symbol}: max trades exceeded"
            )

        logger.info(
            "risk_validation_complete",
            approved=len(result.approved_actions),
            rejected=len(result.rejected_actions),
            forced=len(result.forced_actions),
            warnings=len(result.warnings),
        )
        return result

    def _validate_action(
        self,
        action: TradeAction,
        portfolio: PortfolioState,
        quotes: dict[str, StockQuote],
        order_history: list[dict] | None,
    ) -> RiskCheckResult:
        """Validate a single action against all rules."""
        check = RiskCheckResult(action=action, original_amount=action.amount_usd)
        quote = quotes.get(action.symbol)

        # Rule: No penny stocks
        if quote and quote.price < MIN_PRICE:
            check.approved = False
            check.rejection_reason = f"Price ${quote.price:.2f} below ${MIN_PRICE} minimum"
            return check

        # Rule: Minimum liquidity
        if quote and action.type == "BUY":
            avg_vol_usd = quote.avg_volume_10d * quote.price
            if avg_vol_usd < MIN_AVG_DAILY_VOLUME_USD:
                check.approved = False
                check.rejection_reason = (
                    f"Avg daily volume ${avg_vol_usd:,.0f} below "
                    f"${MIN_AVG_DAILY_VOLUME_USD:,.0f} minimum"
                )
                return check

        if action.type == "BUY":
            return self._validate_buy(action, check, portfolio, quote)
        elif action.type == "SELL":
            return self._validate_sell(action, check, portfolio, order_history)

        return check

    def _validate_buy(
        self,
        action: TradeAction,
        check: RiskCheckResult,
        portfolio: PortfolioState,
        quote: StockQuote | None,
    ) -> RiskCheckResult:
        """Validate a BUY action."""
        min_cash = portfolio.total_value * self.min_cash_pct / 100
        max_investable = max(0, portfolio.cash - min_cash)

        # Rule: Cash after BUY >= min_cash_pct
        if action.amount_usd > max_investable:
            if max_investable <= 0:
                check.approved = False
                check.rejection_reason = (
                    f"Insufficient cash. Available: ${portfolio.cash:,.2f}, "
                    f"min reserve: ${min_cash:,.2f}"
                )
                return check
            # Trim amount
            check.action = TradeAction(
                type=action.type,
                symbol=action.symbol,
                amount_usd=max_investable,
                urgency=action.urgency,
                thesis=action.thesis,
                exit_condition=action.exit_condition,
            )
            check.modified = True
            check.modification_reason = f"Trimmed to respect {self.min_cash_pct}% cash reserve"

        # Rule: Position size <= max_position_pct
        existing_position = portfolio.get_position(action.symbol)
        existing_value = existing_position.market_value if existing_position else 0
        new_total = existing_value + check.action.amount_usd
        max_position_value = portfolio.total_value * self.max_position_pct / 100

        if new_total > max_position_value:
            allowed = max(0, max_position_value - existing_value)
            if allowed <= 0:
                check.approved = False
                check.rejection_reason = (
                    f"Position already at {existing_value / portfolio.total_value * 100:.1f}% "
                    f"(max {self.max_position_pct}%)"
                )
                return check
            check.action = TradeAction(
                type=action.type,
                symbol=action.symbol,
                amount_usd=allowed,
                urgency=action.urgency,
                thesis=action.thesis,
                exit_condition=action.exit_condition,
            )
            check.modified = True
            check.modification_reason = f"Trimmed to respect {self.max_position_pct}% max position"

        # Rule: Minimum order size (after all trimming)
        if self.min_order_usd > 0 and check.action.amount_usd < self.min_order_usd:
            check.approved = False
            check.rejection_reason = (
                f"Order too small after trimming: ${check.action.amount_usd:.2f} < ${self.min_order_usd} min"
            )
            return check

        return check

    def _validate_sell(
        self,
        action: TradeAction,
        check: RiskCheckResult,
        portfolio: PortfolioState,
        order_history: list[dict] | None,
    ) -> RiskCheckResult:
        """Validate a SELL action."""
        position = portfolio.get_position(action.symbol)

        # Rule: Must hold the position
        if not position or position.quantity <= 0:
            check.approved = False
            check.rejection_reason = f"No position in {action.symbol} to sell"
            return check

        # Rule: Can't sell more than we have
        if action.amount_usd > position.market_value:
            check.action = TradeAction(
                type=action.type,
                symbol=action.symbol,
                amount_usd=position.market_value,
                urgency=action.urgency,
                thesis=action.thesis,
                exit_condition=action.exit_condition,
            )
            check.modified = True
            check.modification_reason = "Trimmed to actual position value"

        # Rule: Minimum holding period (days + hours combined)
        min_hold_delta = timedelta(
            days=self.min_holding_days,
            hours=self.min_holding_hours,
        )
        if position.first_buy_date and min_hold_delta.total_seconds() > 0:
            try:
                buy_date = datetime.fromisoformat(position.first_buy_date.replace("Z", "+00:00"))
                if self._sim_date:
                    # Backtest: compare dates as naive to avoid timezone issues
                    now = datetime.fromisoformat(self._sim_date.split("T")[0])
                    held_delta = now - buy_date.replace(tzinfo=None)
                else:
                    held_delta = datetime.now(buy_date.tzinfo) - buy_date
                if held_delta < min_hold_delta:
                    check.approved = False
                    check.rejection_reason = (
                        f"Held {held_delta}, minimum is {min_hold_delta}"
                    )
                    return check
            except (ValueError, TypeError):
                pass  # Can't parse date, skip this check

        return check

    def _check_stop_losses(self, portfolio: PortfolioState) -> list[TradeAction]:
        """Check all positions for stop-loss triggers."""
        forced = []
        for position in portfolio.positions:
            if position.unrealized_pl_pct <= self.stop_loss_pct:
                forced.append(TradeAction(
                    type="SELL",
                    symbol=position.symbol,
                    amount_usd=position.market_value,
                    urgency="HIGH",
                    thesis=f"STOP-LOSS: Position at {position.unrealized_pl_pct:+.1f}% "
                           f"(threshold: {self.stop_loss_pct}%)",
                    exit_condition="Immediate stop-loss execution",
                ))
                logger.warning(
                    "stop_loss_triggered",
                    symbol=position.symbol,
                    pl_pct=position.unrealized_pl_pct,
                    threshold=self.stop_loss_pct,
                )
        return forced

    def _force_reduce_exposure(
        self,
        portfolio: PortfolioState,
        reduction_factor: float,
    ) -> list[TradeAction]:
        """Force sell positions to reduce exposure."""
        forced = []
        for position in sorted(portfolio.positions, key=lambda p: p.unrealized_pl_pct):
            sell_amount = position.market_value * reduction_factor
            if sell_amount > 10:
                forced.append(TradeAction(
                    type="SELL",
                    symbol=position.symbol,
                    amount_usd=sell_amount,
                    urgency="HIGH",
                    thesis=f"FORCED REDUCTION: Portfolio drawdown exceeds "
                           f"{MAX_PORTFOLIO_DRAWDOWN_PCT}% threshold",
                    exit_condition="Emergency risk reduction",
                ))
        return forced


def filter_by_cost_breakeven(
    actions: list[TradeAction],
    portfolio: PortfolioState,
    cost_model: str,
    multiplier: float = 2.0,
) -> tuple[list[TradeAction], list[dict]]:
    """Filter out actions where transaction cost is too high to break even.

    For each action, computes the broker fee and rejects those where the fee
    exceeds ``1 / multiplier`` percent of the trade amount.  With the default
    ``multiplier=2.0`` this means: reject if fee > 0.5% of trade value.

    This standalone function is called only from ``run_intraday_cycle`` so
    it does not affect the standard weekly/daily cycle flow.

    Args:
        actions: List of trade actions to filter.
        portfolio: Current portfolio state (used to look up current prices).
        cost_model: Broker identifier passed to ``calculate_cost``.
        multiplier: Minimum required gain-to-cost ratio.  A multiplier of 2
            means the trade must be able to earn at least 2× the fee to be
            worthwhile.

    Returns:
        Tuple ``(approved, filtered_out)`` where ``filtered_out`` is a list of
        dicts ``{action, reason, fee}``.
    """
    # Break-even threshold: reject if fee_pct > threshold
    # e.g. multiplier=2 → reject if fee > 0.5% of trade amount
    threshold_pct = 100.0 / multiplier  # percent

    approved: list[TradeAction] = []
    filtered_out: list[dict] = []

    for action in actions:
        # Estimate price: prefer live portfolio price, fall back to rough calc
        pos = portfolio.get_position(action.symbol)
        price = pos.current_price if pos else 0.0
        if price <= 0 and action.amount_usd > 0:
            price = action.amount_usd / 10  # rough fallback

        quantity = action.amount_usd / price if price > 0 else 0.0
        fee = calculate_cost(cost_model, quantity, price)
        fee_pct = (fee / action.amount_usd * 100) if action.amount_usd > 0 else 0.0

        if fee_pct * multiplier > threshold_pct:
            reason = (
                f"Fee ${fee:.2f} ({fee_pct:.2f}%) × {multiplier} "
                f"exceeds {threshold_pct:.1f}% breakeven threshold"
            )
            filtered_out.append({"action": action, "reason": reason, "fee": fee})
            logger.info(
                "cost_breakeven_filtered",
                symbol=action.symbol,
                fee=round(fee, 4),
                fee_pct=round(fee_pct, 3),
                threshold_pct=threshold_pct,
            )
        else:
            approved.append(action)

    return approved, filtered_out
