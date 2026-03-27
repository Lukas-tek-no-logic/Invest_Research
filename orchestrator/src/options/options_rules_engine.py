"""Rules-based options selection — replaces LLM Pass 1+2 for wheel and spreads.

Wheel: scores CSP candidates by IV rank, premium yield proxy, and technicals.
Spreads: selects spread type based on ADX/RSI/trend decision tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from ..technical_indicators import TechnicalSignals
from .positions import OptionsPosition

logger = structlog.get_logger()


@dataclass
class ScoredOption:
    symbol: str
    score: float
    action_type: str  # SELL_CSP, OPEN_SPREAD
    spread_type: str = ""  # iron_condor, bull_put, etc.
    reason: str = ""
    breakdown: dict = field(default_factory=dict)


# ── Wheel (CSP) scoring ──────────────────────────────────────────────────


def score_wheel_candidates(
    watchlist: list[str],
    iv_data: dict[str, dict | None],
    tech_signals: dict[str, TechnicalSignals],
    market_data: dict[str, dict],
    portfolio_cash: float,
    active_positions: list[OptionsPosition],
    risk_profile: dict,
) -> list[dict]:
    """Score and rank watchlist symbols for CSP selling.

    Returns list of dicts compatible with WheelAction:
      {"type": "SELL_CSP", "symbol": ..., "contracts": 1, "reason": ...}
    """
    min_iv_rank = risk_profile.get("min_iv_rank", 30)
    max_rsi = risk_profile.get("max_rsi_csp", 75)
    weights = risk_profile.get("scoring_weights", {})
    w_iv = weights.get("iv_rank", 0.4)
    w_premium = weights.get("premium_yield", 0.3)
    w_tech = weights.get("technical", 0.3)

    # Symbols already in open CSP
    open_syms = {p.symbol for p in active_positions if p.spread_type == "CASH_SECURED_PUT" and p.status == "open"}

    scored: list[ScoredOption] = []

    for sym in watchlist:
        # Hard gates
        iv = iv_data.get(sym)
        if not iv or not isinstance(iv, dict):
            continue
        iv_rank = iv.get("rank", 0) or 0
        if iv_rank < min_iv_rank:
            continue
        if sym in open_syms:
            continue

        tech = tech_signals.get(sym)
        rsi = tech.rsi_14 if tech else None
        if rsi is not None and rsi > max_rsi:
            continue

        price = market_data.get(sym, {}).get("price", 0)
        if price <= 0:
            continue
        # Collateral check: strike ~= price, need price * 100 < cash
        if price * 100 > portfolio_cash * 0.9:
            continue

        # Score components (0-1 each)
        iv_score = min(iv_rank / 100, 1.0)

        # Premium yield proxy: higher IV = higher premium. Normalize rank 30-80 → 0-1
        premium_score = max(0, min(1.0, (iv_rank - 30) / 50))

        # Technical: RSI 30-50 best (selling puts on dip), SMA50 support
        tech_score = 0.0
        if rsi is not None:
            if 30 <= rsi <= 50:
                tech_score = 1.0
            elif 50 < rsi <= 65:
                tech_score = 0.6
            elif rsi < 30:
                tech_score = 0.3  # oversold, risky
            else:
                tech_score = 0.2
        if tech and tech.price and tech.sma_50:
            if tech.price > tech.sma_50:
                tech_score = min(1.0, tech_score + 0.2)

        total = iv_score * w_iv + premium_score * w_premium + tech_score * w_tech

        scored.append(ScoredOption(
            symbol=sym, score=total, action_type="SELL_CSP",
            reason=f"IV rank {iv_rank}, RSI {rsi:.0f}" + (f", above SMA50" if tech and tech.price and tech.sma_50 and tech.price > tech.sma_50 else ""),
            breakdown={"iv": round(iv_score, 2), "premium": round(premium_score, 2), "tech": round(tech_score, 2)},
        ))

    scored.sort(key=lambda s: s.score, reverse=True)

    # Return top candidates as WheelAction-compatible dicts
    max_open = risk_profile.get("max_open_csps", 5)
    remaining_slots = max(0, max_open - len(open_syms))

    results = []
    for s in scored[:remaining_slots]:
        if s.score < 0.3:
            break
        results.append({
            "type": "SELL_CSP",
            "symbol": s.symbol,
            "contracts": 1,
            "reason": f"Score {s.score:.2f}: {s.reason} ({s.breakdown})",
        })

    logger.info(
        "wheel_rules_scored",
        candidates=len(scored),
        proposed=len(results),
        top=[(s.symbol, round(s.score, 2)) for s in scored[:5]],
    )
    return results


# ── Spreads type selection ────────────────────────────────────────────────


def score_spread_candidates(
    watchlist: list[str],
    iv_data: dict[str, dict | None],
    tech_signals: dict[str, TechnicalSignals],
    market_data: dict[str, dict],
    active_positions: list[OptionsPosition],
    risk_profile: dict,
) -> list[dict]:
    """Select spread type per symbol based on technicals decision tree.

    Returns list of dicts compatible with SpreadAction:
      {"type": "OPEN_SPREAD", "symbol": ..., "spread_type": ..., "contracts": 1, "reason": ...}
    """
    min_iv_rank = risk_profile.get("min_iv_rank_spreads", risk_profile.get("min_iv_rank", 30))
    max_open = risk_profile.get("max_open_spreads", 5)

    open_syms = {p.symbol for p in active_positions if p.status == "open"}
    remaining_slots = max(0, max_open - len(open_syms))
    if remaining_slots == 0:
        return []

    scored: list[ScoredOption] = []

    for sym in watchlist:
        if sym in open_syms:
            continue

        iv = iv_data.get(sym)
        if not iv or not isinstance(iv, dict):
            continue
        iv_rank = iv.get("rank", 0) or 0
        if iv_rank < min_iv_rank:
            continue

        tech = tech_signals.get(sym)
        if not tech:
            continue

        rsi = tech.rsi_14
        adx = tech.adx_14
        price = tech.price
        sma20 = tech.sma_20
        sma50 = tech.sma_50

        if rsi is None or price is None:
            continue

        # Decision tree for spread type
        spread_type = None
        reason = ""
        score = iv_rank / 100  # base score from IV

        # 1. Iron condor: range-bound + high IV
        if adx is not None and adx < 25 and iv_rank > 50:
            spread_type = "iron_condor"
            reason = f"Range-bound (ADX {adx:.0f}<25) + high IV (rank {iv_rank})"
            score += 0.3

        # 2. Bull put (credit, neutral-bullish): RSI dip + uptrend
        elif rsi < 40 and sma50 and price > sma50:
            spread_type = "bull_put"
            reason = f"RSI {rsi:.0f}<40 + price above SMA50 (support bounce)"
            score += 0.2

        # 3. Bear call (credit, neutral-bearish): RSI high + downtrend
        elif rsi > 65 and sma50 and price < sma50:
            spread_type = "bear_call"
            reason = f"RSI {rsi:.0f}>65 + price below SMA50 (resistance rejection)"
            score += 0.2

        # 4. Bull call (debit, strong uptrend)
        elif rsi < 35 and sma20 and sma50 and price > sma20 > sma50:
            spread_type = "bull_call"
            reason = f"RSI {rsi:.0f}<35 + strong uptrend (price>SMA20>SMA50)"
            score += 0.15

        # 5. Bear put (debit, strong downtrend)
        elif rsi > 70 and sma20 and sma50 and price < sma20 < sma50:
            spread_type = "bear_put"
            reason = f"RSI {rsi:.0f}>70 + strong downtrend (price<SMA20<SMA50)"
            score += 0.15

        if spread_type:
            scored.append(ScoredOption(
                symbol=sym, score=score, action_type="OPEN_SPREAD",
                spread_type=spread_type, reason=reason,
                breakdown={"iv_rank": iv_rank, "rsi": rsi, "adx": adx},
            ))

    scored.sort(key=lambda s: s.score, reverse=True)

    results = []
    for s in scored[:remaining_slots]:
        if s.score < 0.3:
            break
        results.append({
            "type": "OPEN_SPREAD",
            "symbol": s.symbol,
            "spread_type": s.spread_type,
            "contracts": 1,
            "reason": f"Score {s.score:.2f}: {s.reason}",
        })

    logger.info(
        "spreads_rules_scored",
        candidates=len(scored),
        proposed=len(results),
        top=[(s.symbol, s.spread_type, round(s.score, 2)) for s in scored[:5]],
    )
    return results
