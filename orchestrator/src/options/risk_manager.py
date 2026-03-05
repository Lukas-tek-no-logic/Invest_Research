"""Risk manager for the Wheel Strategy.

Validates WheelDecision actions against per-account risk rules and adds
auto-close rules for near-expiry or profit-target positions.

Produces an OptionsRiskResult that main.py already knows how to consume:
  .approved_opens   → SELL_CSP + SELL_CC actions that passed validation
  .approved_closes  → CLOSE actions approved (LLM-requested)
  .forced_closes    → auto-close positions (DTE, take-profit)
  .approved_rolls   → always empty for the Wheel (no roll concept)
  .rejected_opens   → {instruction, reason} dicts for rejected actions
  .modifications    → human-readable log of changes/rejections
  .warnings         → non-blocking risk warnings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import structlog

from ..portfolio_state import PortfolioState
from .positions import OptionsPosition
# Import from the deployed module name.
# When this file is placed in the options package as decision_parser.py (or
# wheel_decision_parser.py), adjust this import to match the actual filename.
from .decision_parser import WheelAction, WheelDecision

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Result type (compatible with what main.py expects from OptionsRiskManager)
# ---------------------------------------------------------------------------

@dataclass
class OptionsRiskResult:
    """Validated wheel actions ready for execution."""
    # approved_opens: SELL_CSP and SELL_CC actions that passed all checks
    approved_opens: list[WheelAction] = field(default_factory=list)
    # rejected_opens: {"instruction": WheelAction, "reason": str}
    rejected_opens: list[dict] = field(default_factory=list)
    # approved_closes: LLM-requested CLOSE actions for known positions
    approved_closes: list[WheelAction] = field(default_factory=list)
    # forced_closes: auto-close rules (DTE expiry, take-profit)
    forced_closes: list[WheelAction] = field(default_factory=list)
    # approved_rolls: always empty for wheel strategy
    approved_rolls: list = field(default_factory=list)
    modifications: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Risk manager
# ---------------------------------------------------------------------------

class OptionsRiskManager:
    """Validate Wheel Strategy decisions against per-account risk rules."""

    def __init__(self, risk_profile: dict):
        self.max_open_csps: int = risk_profile.get("max_open_csps", 3)
        self.max_ccs_per_symbol: int = risk_profile.get("max_ccs_per_symbol", 2)
        self.min_cash_pct: float = risk_profile.get("min_cash_pct", 40.0)
        self.earnings_blackout_days: int = risk_profile.get("earnings_blackout_days", 5)
        self.take_profit_pct: float = risk_profile.get("take_profit_pct", 50.0)
        self.auto_close_dte: int = risk_profile.get("auto_close_dte", 3)
        # For legacy compatibility (greeks-based delta warnings)
        self.max_portfolio_delta_pct: float = risk_profile.get("max_portfolio_delta_pct", 15.0)

    # ── Public interface ───────────────────────────────────────────────────────

    def validate(
        self,
        decision: WheelDecision,
        active_positions: list[OptionsPosition],
        portfolio: PortfolioState,
        portfolio_greeks=None,      # optional; only used for delta warning
        market_data: dict | None = None,   # symbol → {price, ...}
    ) -> OptionsRiskResult:
        """Validate all wheel actions and auto-close rules.

        Args:
            decision:         Parsed LLM wheel decision.
            active_positions: Currently open CSP/CC positions from the tracker.
            portfolio:        Live portfolio state (cash, total_value, …).
            portfolio_greeks: Optional PortfolioGreeks (for delta warning).

        Returns:
            OptionsRiskResult with approved/rejected/forced lists.
        """
        result = OptionsRiskResult()
        account_value = portfolio.total_value or 1.0
        cash = portfolio.cash

        # Build lookup helpers
        active_ids = {p.id for p in active_positions}
        pos_by_id = {p.id: p for p in active_positions}

        # Count existing positions by type
        open_csps = [p for p in active_positions if p.spread_type == "CASH_SECURED_PUT"]
        open_ccs_by_symbol: dict[str, list[OptionsPosition]] = {}
        for p in active_positions:
            if p.spread_type == "COVERED_CALL":
                open_ccs_by_symbol.setdefault(p.symbol, []).append(p)

        # ── Step 1: Auto-close rules (independent of LLM decision) ───────────
        llm_close_ids = {
            a.position_id for a in decision.actions
            if a.type == "CLOSE" and a.position_id is not None
        }

        for pos in active_positions:
            if pos.id in llm_close_ids:
                continue   # LLM already handling it

            forced = self._auto_close_check(pos)
            if forced is not None:
                result.forced_closes.append(forced)
                result.modifications.append(
                    f"[AUTO-CLOSE] {pos.symbol} {pos.spread_type} ID:{pos.id}: {forced.reason}"
                )

        forced_close_ids = {a.position_id for a in result.forced_closes}

        # ── Step 2: LLM-requested CLOSE actions ──────────────────────────────
        for action in decision.actions:
            if action.type != "CLOSE":
                continue
            pid = action.position_id
            if pid is None:
                result.warnings.append("CLOSE action missing position_id — skipped")
                continue
            if pid in forced_close_ids:
                continue   # already in forced_closes
            if pid not in active_ids:
                result.warnings.append(f"CLOSE for unknown position ID {pid} — skipped")
                continue
            result.approved_closes.append(action)

        # ── Step 3: Validate SELL_CSP actions ────────────────────────────────
        # After planned closes, how many CSPs will remain?
        closing_ids = forced_close_ids | {a.position_id for a in result.approved_closes if a.position_id}
        current_csp_count = sum(
            1 for p in open_csps if p.id not in closing_ids
        )
        cash_available = cash   # we will decrement as we approve CSPs
        md = market_data or {}
        # Track symbols already approved this cycle to prevent LLM from opening same symbol twice
        approved_csp_symbols = {p.symbol for p in open_csps if p.id not in closing_ids}

        for action in decision.actions:
            if action.type != "SELL_CSP":
                continue

            symbol = action.symbol
            contracts = max(1, action.contracts)

            # 0. Duplicate check within this cycle
            if symbol in approved_csp_symbols:
                reason = f"CSP for {symbol} already open or approved this cycle — skipped"
                result.rejected_opens.append({"instruction": action, "reason": reason})
                result.modifications.append(f"[REJECTED CSP] {symbol}: {reason}")
                continue

            # 1. Max open CSPs
            if current_csp_count >= self.max_open_csps:
                reason = (
                    f"Max open CSPs ({self.max_open_csps}) already reached "
                    f"(currently {current_csp_count})"
                )
                result.rejected_opens.append({"instruction": action, "reason": reason})
                result.modifications.append(f"[REJECTED CSP] {symbol}: {reason}")
                continue

            # 2. Cash reserve: approximate assignment cost using market price.
            #    A CSP requires strike × 100 in cash as collateral.
            #    We don't know the exact strike yet (executor picks it), so use
            #    current price × 100 as a conservative upper bound.
            estimated_assignment = _estimate_assignment_cost(
                action, portfolio, account_value, md.get(symbol, {})
            )

            # Hard check: collateral must fit in available cash
            if estimated_assignment > cash_available:
                reason = (
                    f"Collateral required (≈${estimated_assignment:,.0f}) exceeds "
                    f"available cash (${cash_available:,.0f}) for {symbol} CSP "
                    f"({contracts} contract{'s' if contracts > 1 else ''})"
                )
                result.rejected_opens.append({"instruction": action, "reason": reason})
                result.modifications.append(f"[REJECTED CSP] {symbol}: {reason}")
                continue

            cash_after = cash_available - estimated_assignment
            cash_after_pct = cash_after / account_value * 100
            if cash_after_pct < self.min_cash_pct:
                reason = (
                    f"Insufficient cash: need ≈${estimated_assignment:,.0f} collateral "
                    f"for {symbol} CSP but only ${cash_available:,.0f} available "
                    f"(would leave {cash_after_pct:.1f}% < {self.min_cash_pct}% min)"
                )
                result.rejected_opens.append({"instruction": action, "reason": reason})
                result.modifications.append(f"[REJECTED CSP] {symbol}: {reason}")
                continue

            # 3. Earnings blackout — only block if earnings are described as IMMINENT
            if _earnings_flag_in_reason(action.reason):
                reason = f"Action flagged as near-earnings: '{action.reason}'"
                result.rejected_opens.append({"instruction": action, "reason": reason})
                result.modifications.append(f"[REJECTED CSP] {symbol}: {reason}")
                continue

            # Approved
            result.approved_opens.append(action)
            approved_csp_symbols.add(symbol)
            current_csp_count += 1
            cash_available -= estimated_assignment

        # ── Step 4: Validate SELL_CC actions ─────────────────────────────────
        for action in decision.actions:
            if action.type != "SELL_CC":
                continue

            symbol = action.symbol
            contracts = max(1, action.contracts)

            # Must reference an existing assigned position (position_id)
            pid = action.position_id
            if pid is not None and pid not in active_ids:
                reason = f"Referenced assigned position ID {pid} not found in active positions"
                result.rejected_opens.append({"instruction": action, "reason": reason})
                result.modifications.append(f"[REJECTED CC] {symbol}: {reason}")
                continue

            # Max CCs per symbol
            symbol_cc_count = len(open_ccs_by_symbol.get(symbol, []))
            if symbol_cc_count >= self.max_ccs_per_symbol:
                reason = (
                    f"Max CCs per symbol ({self.max_ccs_per_symbol}) already reached "
                    f"for {symbol} (currently {symbol_cc_count})"
                )
                result.rejected_opens.append({"instruction": action, "reason": reason})
                result.modifications.append(f"[REJECTED CC] {symbol}: {reason}")
                continue

            # CC strike ≥ cost_basis check is deferred to the selector/executor,
            # which knows the actual strike.  We just pass the action through.

            result.approved_opens.append(action)
            open_ccs_by_symbol.setdefault(symbol, []).append(None)  # type: ignore[arg-type]

        # ── Step 5: Portfolio delta warning ──────────────────────────────────
        if portfolio_greeks is not None and account_value > 0:
            delta_as_pct = abs(portfolio_greeks.total_delta) / account_value * 100
            if delta_as_pct > self.max_portfolio_delta_pct:
                result.warnings.append(
                    f"Portfolio delta ({portfolio_greeks.total_delta:+.2f}) exceeds "
                    f"±{self.max_portfolio_delta_pct}% threshold"
                )

        logger.info(
            "wheel_risk_validated",
            approved_opens=len(result.approved_opens),
            approved_closes=len(result.approved_closes),
            forced_closes=len(result.forced_closes),
            rejected_opens=len(result.rejected_opens),
            warnings=len(result.warnings),
        )

        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _auto_close_check(self, pos: OptionsPosition) -> WheelAction | None:
        """Return a forced-close WheelAction if the position meets an auto-close rule."""

        # DTE expiry threshold (avoid assignment/exercise risk at last moment)
        if pos.dte is not None and pos.dte <= self.auto_close_dte:
            return WheelAction(
                type="CLOSE",
                symbol=pos.symbol,
                position_id=pos.id,
                reason=f"DTE={pos.dte} ≤ auto-close threshold ({self.auto_close_dte})",
            )

        # Take-profit: captured enough premium
        captured = pos.profit_captured_pct
        if captured is not None and captured >= self.take_profit_pct:
            return WheelAction(
                type="CLOSE",
                symbol=pos.symbol,
                position_id=pos.id,
                reason=f"Take-profit: {captured:.0f}% of max premium captured (≥{self.take_profit_pct}%)",
            )

        return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _estimate_assignment_cost(
    action: WheelAction,
    portfolio: PortfolioState,
    account_value: float,
    symbol_market_data: dict | None = None,
) -> float:
    """Estimate cash needed to cover potential assignment for a CSP.

    Priority:
      1. LLM-supplied strike hint (most accurate)
      2. Current market price × 100 (conservative — actual OTM strike will be slightly less)
      3. Fallback: 2× account value (forces rejection — never approve when price is unknown)
    """
    contracts = max(1, action.contracts)
    if action.strike and action.strike > 0:
        return action.strike * 100 * contracts
    if symbol_market_data:
        price = symbol_market_data.get("price", 0) or 0
        if price > 0:
            # Use current price as upper bound; actual OTM strike will be somewhat lower
            return price * 100 * contracts
    # No price data → return 2× account value to force rejection.
    # Better to skip an unknown symbol than to allow an unaffordable position.
    return account_value * 2.0 * contracts


def _earnings_flag_in_reason(reason: str) -> bool:
    """Return True only if the reason indicates earnings are imminently risky.

    Avoids false positives when the LLM says things like
    "no earnings for 6+ weeks" or "earnings far away" — those contain the word
    'earnings' but are NOT a risk flag.
    """
    lower = reason.lower()

    # Explicit safe phrases — LLM confirmed earnings are NOT imminent
    safe_phrases = (
        "no earnings",
        "no upcoming earnings",
        "earnings not soon",
        "earnings far",
        "earnings are not",
        "earnings aren't",
    )
    if any(p in lower for p in safe_phrases):
        return False

    # Risky earnings phrases — earnings are described as close / this week
    block_triggers = (
        "before earnings",
        "near earnings",
        "earnings soon",
        "earnings this week",
        "earnings tomorrow",
        "er soon",
        "er in ",
        "earnings in 1",
        "earnings in 2",
        "earnings in 3",
        "earnings in 4",
        "earnings in 5",
    )
    return any(t in lower for t in block_triggers)
