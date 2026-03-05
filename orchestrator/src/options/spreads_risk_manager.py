"""Risk manager for multi-leg option spreads.

Validates SpreadDecision actions against per-account risk rules and adds
auto-close rules for near-expiry or profit-target positions.

Produces a SpreadsRiskResult consumed by main.py's run_spreads_cycle().
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from ..portfolio_state import PortfolioState
from .positions import OptionsPosition
from .spreads_decision_parser import SpreadAction, SpreadDecision

logger = structlog.get_logger()


@dataclass
class SpreadsRiskResult:
    """Validated spread actions ready for execution."""
    approved_opens: list[SpreadAction] = field(default_factory=list)
    rejected_opens: list[dict] = field(default_factory=list)
    approved_closes: list[SpreadAction] = field(default_factory=list)
    forced_closes: list[SpreadAction] = field(default_factory=list)
    approved_rolls: list = field(default_factory=list)
    modifications: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SpreadsRiskManager:
    """Validate spread decisions against per-account risk rules."""

    def __init__(self, risk_profile: dict):
        self.max_open_spreads: int = risk_profile.get("max_open_spreads", 5)
        self.min_cash_pct: float = risk_profile.get("min_cash_pct", 20.0)
        self.max_spread_width: float = risk_profile.get("max_spread_width", 10.0)
        self.take_profit_pct: float = risk_profile.get("take_profit_pct", 50.0)
        self.stop_loss_pct: float = risk_profile.get("stop_loss_pct", 100.0)
        self.auto_close_dte: int = risk_profile.get("auto_close_dte", 3)
        self.target_dte_min: int = risk_profile.get("target_dte_min", 21)
        self.target_dte_max: int = risk_profile.get("target_dte_max", 45)

    def validate(
        self,
        decision: SpreadDecision,
        active_positions: list[OptionsPosition],
        portfolio: PortfolioState,
        portfolio_greeks=None,
        market_data: dict | None = None,
    ) -> SpreadsRiskResult:
        result = SpreadsRiskResult()
        account_value = portfolio.total_value or 1.0
        cash = portfolio.cash

        active_ids = {p.id for p in active_positions}

        # -- Step 1: Auto-close rules --
        llm_close_ids = {
            a.position_id for a in decision.actions
            if a.type == "CLOSE" and a.position_id is not None
        }

        for pos in active_positions:
            if pos.id in llm_close_ids:
                continue
            forced = self._auto_close_check(pos)
            if forced is not None:
                result.forced_closes.append(forced)
                result.modifications.append(
                    f"[AUTO-CLOSE] {pos.symbol} {pos.spread_type} ID:{pos.id}: {forced.reason}"
                )

        forced_close_ids = {a.position_id for a in result.forced_closes}

        # -- Step 2: LLM-requested CLOSE actions --
        for action in decision.actions:
            if action.type != "CLOSE":
                continue
            pid = action.position_id
            if pid is None:
                result.warnings.append("CLOSE action missing position_id - skipped")
                continue
            if pid in forced_close_ids:
                continue
            if pid not in active_ids:
                result.warnings.append(f"CLOSE for unknown position ID {pid} - skipped")
                continue
            result.approved_closes.append(action)

        # -- Step 3: Validate OPEN_SPREAD actions --
        closing_ids = forced_close_ids | {a.position_id for a in result.approved_closes if a.position_id}
        current_spread_count = sum(1 for p in active_positions if p.id not in closing_ids)
        cash_available = cash
        approved_spread_symbols: set[str] = {p.symbol for p in active_positions if p.id not in closing_ids}

        for action in decision.actions:
            if action.type != "OPEN_SPREAD":
                continue

            symbol = action.symbol
            contracts = max(1, action.contracts)

            # 0. Duplicate check within this cycle
            if symbol in approved_spread_symbols:
                reason = f"Spread for {symbol} already open or approved this cycle — skipped"
                result.rejected_opens.append({"instruction": action, "reason": reason})
                result.modifications.append(f"[REJECTED] {symbol} {action.spread_type}: {reason}")
                continue

            # 1. Max open spreads
            if current_spread_count >= self.max_open_spreads:
                reason = (
                    f"Max open spreads ({self.max_open_spreads}) already reached "
                    f"(currently {current_spread_count})"
                )
                result.rejected_opens.append({"instruction": action, "reason": reason})
                result.modifications.append(f"[REJECTED] {symbol} {action.spread_type}: {reason}")
                continue

            # 2. Cash reserve: estimate max loss as max_width * 100 * contracts
            estimated_max_loss = self.max_spread_width * 100 * contracts
            cash_after = cash_available - estimated_max_loss
            cash_after_pct = cash_after / account_value * 100
            if cash_after_pct < self.min_cash_pct:
                reason = (
                    f"Insufficient cash: estimated max loss ~${estimated_max_loss:,.0f} "
                    f"for {symbol} {action.spread_type} but only ${cash_available:,.0f} available "
                    f"(would leave {cash_after_pct:.1f}% < {self.min_cash_pct}% min)"
                )
                result.rejected_opens.append({"instruction": action, "reason": reason})
                result.modifications.append(f"[REJECTED] {symbol} {action.spread_type}: {reason}")
                continue

            # 3. Earnings blackout
            if _earnings_flag_in_reason(action.reason):
                reason = f"Action flagged as near-earnings: '{action.reason}'"
                result.rejected_opens.append({"instruction": action, "reason": reason})
                result.modifications.append(f"[REJECTED] {symbol}: {reason}")
                continue

            # Approved
            result.approved_opens.append(action)
            approved_spread_symbols.add(symbol)
            current_spread_count += 1
            cash_available -= estimated_max_loss

        # -- Step 4: Portfolio warnings --
        if portfolio_greeks is not None and account_value > 0:
            delta_as_pct = abs(portfolio_greeks.total_delta) / account_value * 100
            if delta_as_pct > 15.0:
                result.warnings.append(
                    f"Portfolio delta ({portfolio_greeks.total_delta:+.2f}) exceeds 15% threshold"
                )

        logger.info(
            "spreads_risk_validated",
            approved_opens=len(result.approved_opens),
            approved_closes=len(result.approved_closes),
            forced_closes=len(result.forced_closes),
            rejected_opens=len(result.rejected_opens),
            warnings=len(result.warnings),
        )

        return result

    def _auto_close_check(self, pos: OptionsPosition) -> SpreadAction | None:
        """Return a forced-close SpreadAction if auto-close rules trigger."""

        # DTE expiry threshold
        if pos.dte is not None and pos.dte <= self.auto_close_dte:
            return SpreadAction(
                type="CLOSE",
                symbol=pos.symbol,
                position_id=pos.id,
                reason=f"DTE={pos.dte} <= auto-close threshold ({self.auto_close_dte})",
            )

        # Take-profit
        captured = pos.profit_captured_pct
        if captured is not None and captured >= self.take_profit_pct:
            return SpreadAction(
                type="CLOSE",
                symbol=pos.symbol,
                position_id=pos.id,
                reason=f"Take-profit: {captured:.0f}% of max profit captured (>={self.take_profit_pct}%)",
            )

        # Stop-loss: loss exceeds threshold % of max loss
        if pos.current_pl is not None and pos.max_loss > 0:
            loss_pct = abs(min(pos.current_pl, 0)) / pos.max_loss * 100
            if loss_pct >= self.stop_loss_pct:
                return SpreadAction(
                    type="CLOSE",
                    symbol=pos.symbol,
                    position_id=pos.id,
                    reason=f"Stop-loss: {loss_pct:.0f}% of max loss reached (>={self.stop_loss_pct}%)",
                )

        return None


def _earnings_flag_in_reason(reason: str) -> bool:
    """Return True only if the reason indicates earnings are imminently risky."""
    lower = reason.lower()

    safe_phrases = (
        "no earnings", "no upcoming earnings", "earnings not soon",
        "earnings far", "earnings are not", "earnings aren't",
    )
    if any(p in lower for p in safe_phrases):
        return False

    block_triggers = (
        "before earnings", "near earnings", "earnings soon",
        "earnings this week", "earnings tomorrow", "er soon", "er in ",
        "earnings in 1", "earnings in 2", "earnings in 3",
        "earnings in 4", "earnings in 5",
    )
    return any(t in lower for t in block_triggers)
