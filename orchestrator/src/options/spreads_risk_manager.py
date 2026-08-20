"""Risk manager for multi-leg option spreads.

Validates SpreadDecision actions against per-account risk rules and adds
auto-close rules for near-expiry or profit-target positions.

Produces a SpreadsRiskResult consumed by main.py's run_spreads_cycle().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

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
        # Mark-based exits (take-profit / stop-loss) are suppressed for this
        # many days after entry: option mids are noisy, and a spread designed
        # for DTE 21-45 that gets closed on the first Monday after entry pays
        # two bid-ask spreads for nothing. Structural exits (DTE, short-strike
        # breach) stay active from day one.
        self.min_holding_days: int = risk_profile.get("min_holding_days", 5)
        # VRP-pilot knobs: restrict structures to a whitelist (e.g. ["bull_put"]
        # for systematic index put credit spreads) and allow laddering several
        # spreads on one underlying across expiries (a single-symbol watchlist
        # would otherwise cap the account at one open spread).
        allowed = risk_profile.get("allowed_spread_types")
        self.allowed_spread_types: set[str] | None = (
            {str(t).lower() for t in allowed} if allowed else None
        )
        self.allow_same_symbol_spreads: bool = bool(
            risk_profile.get("allow_same_symbol_spreads", False)
        )

    def validate(
        self,
        decision: SpreadDecision,
        active_positions: list[OptionsPosition],
        portfolio: PortfolioState,
        portfolio_greeks=None,
        market_data: dict | None = None,
        tech_signals: dict | None = None,
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
            spot = (market_data or {}).get(pos.symbol, {}).get("price")
            forced = self.check_position_exits(pos, spot=spot)
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

            # -1. Structure whitelist (VRP pilot: only index put credit spreads)
            if (self.allowed_spread_types is not None
                    and str(action.spread_type).lower() not in self.allowed_spread_types):
                reason = (
                    f"Spread type '{action.spread_type}' not in allowed set "
                    f"{sorted(self.allowed_spread_types)} for this account"
                )
                result.rejected_opens.append({"instruction": action, "reason": reason})
                result.modifications.append(f"[REJECTED] {symbol} {action.spread_type}: {reason}")
                continue

            # 0. Duplicate check within this cycle
            if symbol in approved_spread_symbols and not self.allow_same_symbol_spreads:
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

            # 4. Trend consistency: don't fight a clear trend with a
            # directional spread (repeated bear calls on trending AI names
            # were the single biggest loss source).
            trend_reason = _against_trend_reason(
                action.spread_type, (tech_signals or {}).get(symbol),
            )
            if trend_reason:
                result.rejected_opens.append({"instruction": action, "reason": trend_reason})
                result.modifications.append(
                    f"[REJECTED] {symbol} {action.spread_type}: {trend_reason}"
                )
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

    def check_position_exits(
        self, pos: OptionsPosition, spot: float | None = None,
    ) -> SpreadAction | None:
        """Return a forced-close SpreadAction if exit rules trigger.

        Structural rules (always active): DTE threshold, spot breaching the
        short strike of a credit structure.
        Mark-based rules (only after min_holding_days): take-profit, stop-loss
        as % of max loss.
        """

        # DTE expiry threshold
        if pos.dte is not None and pos.dte <= self.auto_close_dte:
            return SpreadAction(
                type="CLOSE",
                symbol=pos.symbol,
                position_id=pos.id,
                reason=f"DTE={pos.dte} <= auto-close threshold ({self.auto_close_dte})",
            )

        # Short-strike breach on credit structures: the position is under real
        # directional threat regardless of what the (noisy) marks say.
        if (pos.entry_debit or 0) < 0 and spot is not None and pos.sell_strike > 0:
            breached = (
                spot > pos.sell_strike
                if pos.sell_option_type == "call"
                else spot < pos.sell_strike
            )
            if breached:
                return SpreadAction(
                    type="CLOSE",
                    symbol=pos.symbol,
                    position_id=pos.id,
                    reason=(
                        f"Short-strike breach: spot {spot:.2f} beyond short "
                        f"{pos.sell_option_type} {pos.sell_strike}"
                    ),
                )

        # Mark-based exits are unreliable right after entry — skip them until
        # the position has had time to develop.
        if self._held_days(pos) < self.min_holding_days:
            return None

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

    @staticmethod
    def _held_days(pos: OptionsPosition) -> int:
        try:
            entry = datetime.strptime(pos.entry_date, "%Y-%m-%d").date()
            return (date.today() - entry).days
        except (TypeError, ValueError):
            return 0


_BEARISH_SPREADS = {"BEAR_CALL", "BEAR_PUT"}
_BULLISH_SPREADS = {"BULL_CALL", "BULL_PUT"}


def _against_trend_reason(spread_type: str, signals) -> str | None:
    """Return a rejection reason if a directional spread fights a clear trend.

    Uses price vs SMA50 confirmed by RSI; with missing data nothing is
    rejected. Neutral structures (iron condors, butterflies) always pass.
    """
    st = (spread_type or "").upper()
    if st not in _BEARISH_SPREADS and st not in _BULLISH_SPREADS:
        return None
    price = getattr(signals, "price", None)
    sma_50 = getattr(signals, "sma_50", None)
    rsi = getattr(signals, "rsi_14", None)
    if not price or not sma_50 or rsi is None:
        return None

    if st in _BEARISH_SPREADS and price > sma_50 and rsi >= 55:
        return (
            f"Bearish spread against uptrend: price {price:.2f} > SMA50 "
            f"{sma_50:.2f}, RSI {rsi:.0f}"
        )
    if st in _BULLISH_SPREADS and price < sma_50 and rsi <= 45:
        return (
            f"Bullish spread against downtrend: price {price:.2f} < SMA50 "
            f"{sma_50:.2f}, RSI {rsi:.0f}"
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
