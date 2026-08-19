"""Risk manager: validates and modifies trade decisions against hard rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import structlog

from .decision_parser import TradeAction, DecisionResult
from .portfolio_state import PortfolioState, CASH_EQUIVALENT_SYMBOLS
from .market_data import StockQuote
from .transaction_costs import calculate_cost

logger = structlog.get_logger()

MIN_PRICE = 5.0
MIN_AVG_DAILY_VOLUME_USD = 100_000
MAX_PORTFOLIO_DRAWDOWN_PCT = -20.0

# Minimum invested percentage per market regime. The mirror image of the max
# limits: nothing used to protect the account from UNDER-exposure, so LLM
# defensiveness left 40-100% of capital idle through a bull market. Enforced
# like any other hard limit; HIGH_VOLATILITY/BEAR_TREND have no floor.
EXPOSURE_FLOOR_PCT = {"BULL_TREND": 60.0, "SIDEWAYS": 40.0}
# Aligned with the cost-breakeven floor: under the ibkr model a $1 fee is 0.5%
# of $200, so a smaller mechanical top-up would spend more on commission than a
# discretionary buy is allowed to.
FLOOR_TOPUP_MIN_USD = 200.0

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
        self.exposure_floor_pct = risk_profile.get("exposure_floor_pct", EXPOSURE_FLOOR_PCT)
        self._sim_date = sim_date  # None = use datetime.now()

    def validate(
        self,
        decision: DecisionResult,
        portfolio: PortfolioState,
        quotes: dict[str, StockQuote],
        order_history: list[dict] | None = None,
        regime: str | None = None,
        core_symbols: list[str] | None = None,
    ) -> RiskManagerResult:
        """Run all risk checks on the decision.

        Order of operations:
          1. Check for forced stop-loss sells
          2. Check portfolio-level drawdown
          3. Validate each action against rules
          4. Trim to max trades (drop lowest urgency first)
        """
        result = RiskManagerResult()

        # 0. Force-close zombie positions (< $5 value) — not worth the drag
        zombie_threshold = 5.0
        for position in portfolio.positions:
            if 0 < position.market_value < zombie_threshold:
                result.forced_actions.append(TradeAction(
                    type="SELL",
                    symbol=position.symbol,
                    amount_usd=position.market_value,
                    urgency="HIGH",
                    thesis=f"ZOMBIE CLEANUP: position worth ${position.market_value:.2f} — "
                           f"closing to eliminate dead weight",
                    exit_condition="Immediate cleanup",
                ))
                result.modifications.append(
                    f"FORCED ZOMBIE SELL {position.symbol}: ${position.market_value:.2f} < ${zombie_threshold}"
                )

        # 1. Check stop-losses BEFORE model actions
        forced_sells = self._check_stop_losses(portfolio)
        result.forced_actions.extend(forced_sells)
        if forced_sells:
            symbols = [a.symbol for a in forced_sells]
            result.warnings.append(f"STOP-LOSS triggered for: {', '.join(symbols)}")

        # A position valued at avg_cost reports exactly 0.0% P/L, so the
        # stop-loss above cannot see it however far it has actually fallen.
        stale = [p.symbol for p in portfolio.positions if p.price_stale]
        if stale:
            result.warnings.append(
                f"NO MARKET PRICE for {', '.join(stale)} — valued at cost, P/L unknown, "
                f"stop-loss cannot evaluate these positions"
            )

        # Phantom shorts mean Ghostfolio booked sells the account could not cover.
        if portfolio.oversold:
            detail = ", ".join(f"{s} {q:+.4f}" for s, q in portfolio.oversold.items())
            result.warnings.append(
                f"PHANTOM SHORT in Ghostfolio: {detail} — cash was credited for shares "
                f"never held; buying power and total value are overstated"
            )

        # 2. Check portfolio drawdown
        if portfolio.total_pl_pct <= MAX_PORTFOLIO_DRAWDOWN_PCT:
            result.warnings.append(
                f"CRITICAL: Portfolio drawdown {portfolio.total_pl_pct:.1f}% exceeds "
                f"{MAX_PORTFOLIO_DRAWDOWN_PCT}% threshold. Forcing 50% exposure reduction."
            )
            forced_reduce = self._force_reduce_exposure(portfolio, 0.5)
            result.forced_actions.extend(forced_reduce)

        # Deduplicate forced sells. Zombie cleanup, stop-loss and drawdown
        # reduction are independent streams that can all target one symbol; the
        # executor dedupes by (type, symbol) and runs only the first, but the
        # accounting below would otherwise count every duplicate toward
        # `freed_cash` and validate buys against money that never arrives.
        # Worst case is a -20% drawdown, where every position appears twice.
        deduped_forced: list[TradeAction] = []
        seen_forced: set[tuple[str, str]] = set()
        for action in result.forced_actions:
            key = (action.type, action.symbol)
            if key in seen_forced:
                result.modifications.append(
                    f"DEDUPED forced {action.type} {action.symbol}: already queued by an earlier rule"
                )
                continue
            seen_forced.add(key)
            deduped_forced.append(action)
        result.forced_actions = deduped_forced

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

        # Exposure floor: minimum invested % per regime (0 = no floor).
        # Forced sells (stop-loss, zombie, drawdown) are never blocked by it.
        floor_pct = float((self.exposure_floor_pct or {}).get(regime, 0) or 0) if regime else 0.0
        floor_value = portfolio.total_value * floor_pct / 100
        equity_value = sum(p.market_value for p in portfolio.positions)
        projected_equity = equity_value - sum(
            a.amount_usd for a in result.forced_actions if a.type == "SELL"
        )

        validated = []
        # Track cash freed by approved sells (including forced) for buy validation
        freed_cash = sum(a.amount_usd for a in result.forced_actions if a.type == "SELL")
        for action in sells:
            check = self._validate_action(action, portfolio, quotes, order_history)
            if check.approved and floor_value > 0 and \
                    projected_equity - check.action.amount_usd < floor_value:
                check.approved = False
                check.rejection_reason = (
                    f"SELL would drop invested below {floor_pct:.0f}% exposure floor "
                    f"for {regime} (${projected_equity - check.action.amount_usd:,.0f} "
                    f"< ${floor_value:,.0f})"
                )
            if check.approved:
                validated.append(check)
                freed_cash += check.action.amount_usd
                projected_equity -= check.action.amount_usd
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
                f"effective buying power: ${max(0, adjusted_portfolio.available_cash - adjusted_portfolio.total_value * self.min_cash_pct / 100):,.0f}"
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

        # Exposure floor top-up: if the account would still sit below the
        # regime's minimum invested %, mechanically buy core ETFs up to the
        # floor (spread across core symbols to respect per-position caps).
        if floor_value > 0 and core_symbols:
            approved_buy_by_sym: dict[str, float] = {}
            for a in result.approved_actions:
                if a.type == "BUY" and a.symbol not in CASH_EQUIVALENT_SYMBOLS:
                    approved_buy_by_sym[a.symbol] = (
                        approved_buy_by_sym.get(a.symbol, 0) + a.amount_usd
                    )
            buys_total = sum(approved_buy_by_sym.values())
            proj_equity_final = projected_equity + buys_total
            min_reserve = portfolio.total_value * self.min_cash_pct / 100
            cash_avail = portfolio.available_cash + freed_cash - buys_total - min_reserve
            shortfall = floor_value - proj_equity_final
            budget = min(shortfall, cash_avail)

            if budget > FLOOR_TOPUP_MIN_USD:
                max_position_value = portfolio.total_value * self.max_position_pct / 100
                for sym in core_symbols:
                    if budget <= FLOOR_TOPUP_MIN_USD:
                        break
                    if sym in CASH_EQUIVALENT_SYMBOLS:
                        continue
                    existing = portfolio.get_position(sym)
                    existing_val = existing.market_value if existing else 0.0
                    room = max_position_value - existing_val - approved_buy_by_sym.get(sym, 0)
                    amount = min(budget, room)
                    if amount < FLOOR_TOPUP_MIN_USD:
                        continue
                    result.forced_actions.append(TradeAction(
                        type="BUY",
                        symbol=sym,
                        amount_usd=amount,
                        urgency="MEDIUM",
                        thesis=(
                            f"EXPOSURE FLOOR: invested ${proj_equity_final:,.0f} "
                            f"< {floor_pct:.0f}% floor (${floor_value:,.0f}) in {regime} — "
                            f"mechanical top-up into core ETF"
                        ),
                        exit_condition="Managed by strategy cycles",
                    ))
                    budget -= amount
                    proj_equity_final += amount
                result.warnings.append(
                    f"EXPOSURE FLOOR ({regime}): topped up to ${proj_equity_final:,.0f} "
                    f"invested (floor ${floor_value:,.0f} = {floor_pct:.0f}%)"
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
        # available_cash includes T-bill parking (auto-liquidated before buys)
        max_investable = max(0, portfolio.available_cash - min_cash)

        # Rule: Cash after BUY >= min_cash_pct
        if action.amount_usd > max_investable:
            if max_investable <= 0:
                check.approved = False
                check.rejection_reason = (
                    f"Insufficient cash. Available: ${portfolio.available_cash:,.2f}, "
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
        # (cash-equivalents are exempt — parking idle cash is not concentration risk)
        existing_position = portfolio.get_position(action.symbol)
        existing_value = existing_position.market_value if existing_position else 0
        new_total = existing_value + check.action.amount_usd
        max_position_value = portfolio.total_value * self.max_position_pct / 100

        if new_total > max_position_value and action.symbol not in CASH_EQUIVALENT_SYMBOLS:
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


FULL_EXIT_FRACTION = 0.99


def filter_by_cost_breakeven(
    actions: list[TradeAction],
    portfolio: PortfolioState,
    cost_model: str,
    multiplier: float = 2.0,
) -> tuple[list[TradeAction], list[dict]]:
    """Reject trades too small to justify their broker fee.

    A trade is rejected when the fee exceeds ``1 / multiplier`` percent of the
    trade value — with the default ``multiplier=2.0``, when the fee is more than
    0.5% of the amount.  Under the ``ibkr`` model (a flat $1 at these sizes) that
    puts the effective floor at $200 per order.

    Pass ONLY discretionary actions here.  Mechanical ones — stop-loss, drawdown
    reduction, zombie cleanup, exposure-floor top-ups — must bypass the filter
    entirely, because a position too small to sell profitably is still a position
    that has to be closable; blocking its exit strands it until it reaches zero.
    Callers therefore filter ``approved_actions`` and re-attach
    ``forced_actions`` afterwards.

    Full exits are exempt for the same reason: once the decision to leave a
    position is made, the fee is unavoidable and the alternative is holding
    forever.

    Args:
        actions: Discretionary trade actions to filter.
        portfolio: Current portfolio state (used to look up current prices).
        cost_model: Broker identifier passed to ``calculate_cost``.
        multiplier: Inverse of the tolerated fee percentage.  Higher is stricter:
            2.0 → 0.5%, 4.0 → 0.25%.

    Returns:
        Tuple ``(approved, filtered_out)`` where ``filtered_out`` is a list of
        dicts ``{action, reason, fee}``.
    """
    # Reject when fee_pct exceeds this. multiplier=2 → 0.5% of trade value.
    threshold_pct = 1.0 / multiplier if multiplier > 0 else 0.5

    approved: list[TradeAction] = []
    filtered_out: list[dict] = []

    for action in actions:
        thesis = (action.thesis or "").upper()

        # Mechanical actions are never priced out of existence.
        if any(tag in thesis for tag in ("ZOMBIE", "STOP-LOSS", "FORCED REDUCTION", "EXPOSURE FLOOR")):
            approved.append(action)
            continue

        holding = portfolio.get_holding(action.symbol)

        # A SELL that closes (almost) the whole position is an exit, not a
        # discretionary trim — let it through regardless of fee.
        if (
            action.type == "SELL"
            and holding is not None
            and holding.market_value > 0
            and action.amount_usd >= holding.market_value * FULL_EXIT_FRACTION
        ):
            approved.append(action)
            continue

        price = holding.current_price if holding else 0.0
        if price <= 0:
            # No trustworthy price. Fail OPEN rather than fabricate one — a
            # made-up price here silently rejects legitimate trades.
            logger.info(
                "cost_breakeven_skipped_no_price",
                symbol=action.symbol,
                note="no market price available; cost filter not applied",
            )
            approved.append(action)
            continue

        quantity = action.amount_usd / price
        fee = calculate_cost(cost_model, quantity, price)
        fee_pct = (fee / action.amount_usd * 100) if action.amount_usd > 0 else 0.0

        if fee_pct > threshold_pct:
            reason = (
                f"Fee ${fee:.2f} is {fee_pct:.2f}% of ${action.amount_usd:,.0f} "
                f"— above the {threshold_pct:.2f}% limit "
                f"(min viable order ≈ ${fee / (threshold_pct / 100):,.0f})"
            )
            filtered_out.append({"action": action, "reason": reason, "fee": fee})
            logger.info(
                "cost_breakeven_filtered",
                symbol=action.symbol,
                amount_usd=round(action.amount_usd, 2),
                fee=round(fee, 4),
                fee_pct=round(fee_pct, 3),
                threshold_pct=threshold_pct,
            )
        else:
            approved.append(action)

    return approved, filtered_out
