"""Single-call LLM review: advisory veto layer on rules-engine proposals.

The LLM cannot add trades — only APPROVE or VETO proposed trades.
Vetoes must cite concrete data (earnings date, RSI value, etc.).
If the LLM call fails, all proposals are approved (rules engine is primary).
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from .decision_parser import DecisionResult, TradeAction
from .portfolio_state import PortfolioState
from .rules_engine import ScoredSymbol

logger = structlog.get_logger()


@dataclass
class ReviewVerdict:
    symbol: str
    verdict: str  # "APPROVE" or "VETO"
    reason: str = ""


def build_review_messages(
    proposals: list[TradeAction],
    score_breakdowns: list[ScoredSymbol],
    portfolio: PortfolioState,
    market_context: dict,
    strategy: str,
) -> list[dict]:
    """Build single LLM review prompt with proposed trades + scores."""

    system = (
        "You are a risk-aware investment reviewer for an automated trading system.\n"
        "You receive RULES-ENGINE proposals with quantitative scoring.\n\n"
        "For each proposal: output APPROVE (default) or VETO.\n"
        "Vetoes MUST cite concrete data: earnings date, RSI value, sector %, news event.\n"
        "INVALID vetoes (will be ignored): 'feels risky', 'market might drop', vague concerns.\n\n"
        "You CANNOT add new trades. You can only approve or veto what is proposed.\n"
        "If unsure, APPROVE — the risk manager validates position sizing and limits separately.\n\n"
        'Respond in JSON: {"reviews": [{"symbol": "X", "verdict": "APPROVE|VETO", "reason": "..."}], '
        '"portfolio_outlook": "BULLISH|NEUTRAL|BEARISH", "confidence": 0.0-1.0}'
    )

    # Format proposals
    proposal_lines = []
    score_map = {s.symbol: s for s in score_breakdowns}
    for a in proposals:
        sc = score_map.get(a.symbol)
        score_str = f" (score: {sc.score:.0f}/100, {sc.breakdown})" if sc else ""
        proposal_lines.append(
            f"  {a.type} {a.symbol} ${a.amount_usd:,.0f}{score_str}\n"
            f"    Thesis: {a.thesis}"
        )

    # Format portfolio
    held = ", ".join(
        f"{p.symbol} ${p.market_value:,.0f} ({p.unrealized_pl_pct:+.1f}%)"
        for p in portfolio.positions[:10]
    )

    vix = market_context.get("vix", "N/A")
    tnx = market_context.get("tnx", "N/A")

    user = (
        f"Strategy: {strategy}\n"
        f"Portfolio: ${portfolio.total_value:,.0f} | Cash: {portfolio.cash_pct:.0f}% | "
        f"Positions: {portfolio.position_count}\n"
        f"Holdings: {held or 'none'}\n"
        f"Market: VIX={vix}, 10Y={tnx}\n\n"
        f"PROPOSED TRADES:\n" + "\n".join(proposal_lines)
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_review(raw: dict) -> list[ReviewVerdict]:
    """Parse LLM review response into verdicts."""
    reviews = raw.get("reviews", [])
    verdicts = []
    for r in reviews:
        if not isinstance(r, dict):
            continue
        symbol = r.get("symbol", "")
        verdict = (r.get("verdict") or "APPROVE").upper()
        if verdict not in ("APPROVE", "VETO"):
            verdict = "APPROVE"
        reason = r.get("reason", "")
        verdicts.append(ReviewVerdict(symbol=symbol, verdict=verdict, reason=reason))
    return verdicts


def apply_vetoes(
    proposals: list[TradeAction],
    verdicts: list[ReviewVerdict],
    portfolio_outlook: str = "NEUTRAL",
    confidence: float = 0.7,
) -> DecisionResult:
    """Remove vetoed trades and return a standard DecisionResult."""
    veto_map = {v.symbol: v for v in verdicts if v.verdict == "VETO"}

    approved = []
    for action in proposals:
        veto = veto_map.get(action.symbol)
        if veto:
            logger.info(
                "llm_review_veto",
                symbol=action.symbol, type=action.type,
                reason=veto.reason,
            )
        else:
            approved.append(action)

    reasoning = f"Rules engine proposed {len(proposals)} trades, LLM approved {len(approved)}"
    if veto_map:
        vetoed_str = ", ".join(f"{v.symbol}: {v.reason}" for v in veto_map.values())
        reasoning += f". Vetoed: {vetoed_str}"

    return DecisionResult(
        reasoning=reasoning,
        actions=approved,
        portfolio_outlook=portfolio_outlook,
        confidence=confidence,
        next_cycle_focus="",
        suggest_symbols=[],
    )
