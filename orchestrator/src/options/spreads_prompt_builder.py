"""Build LLM prompts for multi-leg option spread decisions.

Pass 1 - Market analysis: IV regime, skew, directional bias per symbol.
Pass 2 - Concrete spread actions: OPEN_SPREAD, CLOSE, or SKIP.
"""

from __future__ import annotations

import json

from ..portfolio_state import PortfolioState
from .greeks import PortfolioGreeks
from .positions import OptionsPosition


# ---------------------------------------------------------------------------
# Pass 1: Market + IV Analysis (reuses formatting from wheel prompt_builder)
# ---------------------------------------------------------------------------

def build_spreads_pass1_messages(
    portfolio: PortfolioState,
    market_data: dict[str, dict],
    technical_signals: dict,
    news_text: str,
    strategy_config: dict,
    active_positions: list[OptionsPosition],
    iv_data: dict[str, float | None],
    portfolio_greeks: PortfolioGreeks,
) -> list[dict]:
    """Pass 1: Market analysis focused on spread suitability."""

    system = (
        "You are an options spread analyst specialising in multi-leg strategies "
        "(iron condors, bull/bear call/put spreads, butterflies). "
        "Analyse market conditions, implied volatility regime, IV skew, "
        "and per-symbol directional bias to determine optimal spread structures. "
        "Do NOT decide specific trades yet - that comes in Pass 2. "
        "Output valid JSON only, no markdown."
    )

    pos_text = _format_active_positions(active_positions)
    mkt_text = _format_market_with_iv(market_data, technical_signals, iv_data)

    user = f"""{portfolio.to_prompt_text()}

{pos_text}

== WATCHLIST ANALYSIS ==
{mkt_text}

== RECENT NEWS ==
{news_text or "No recent news."}

Analyse the above for options spread opportunities and return JSON:
{{
  "market_regime": "BULL_TREND|BEAR_TREND|SIDEWAYS|HIGH_VOLATILITY",
  "regime_reasoning": "brief explanation",
  "iv_regime": "HIGH|NORMAL|LOW",
  "iv_reasoning": "are spreads favoured right now? which type?",
  "sector_analysis": {{"sector_name": "BULLISH|NEUTRAL|BEARISH - reason"}},
  "per_symbol": {{
    "SYMBOL": {{
      "bias": "BULLISH|NEUTRAL|BEARISH",
      "iv_percentile": 45,
      "iv_skew": "PUT_SKEW|CALL_SKEW|FLAT",
      "best_spread_type": "iron_condor|bull_call|bear_put|bull_put|bear_call|butterfly",
      "spread_reasoning": "why this spread type fits",
      "support_level": 150.0,
      "resistance_level": 165.0,
      "earnings_soon": false,
      "reason": "brief"
    }}
  }},
  "portfolio_health": {{
    "open_spreads": 2,
    "cash_deployed_pct": 30.0,
    "issues": ["list of issues, if any"]
  }},
  "threats": [{{"description": "macro or specific threat"}}]
}}"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Pass 2: Concrete Spread Actions
# ---------------------------------------------------------------------------

def build_spreads_pass2_messages(
    analysis_json: dict,
    portfolio: PortfolioState,
    strategy_config: dict,
    risk_profile: dict,
    active_positions: list[OptionsPosition],
    portfolio_greeks: PortfolioGreeks,
    decision_history: str = "",
    market_data: dict | None = None,
) -> list[dict]:
    """Pass 2: Decide concrete spread actions.

    LLM chooses action type, symbol, and spread type.
    The selector module picks exact strikes and expiration dates.
    """

    pos_text = _format_active_positions_detailed(active_positions)
    risk_text = _format_spreads_risk_rules(risk_profile)
    watchlist = strategy_config.get("watchlist", [])
    strategy_desc = strategy_config.get("strategy_description", "Options Spreads")
    max_spreads = risk_profile.get("max_open_spreads", 5)
    min_cash_pct = risk_profile.get("min_cash_pct", 20)
    max_width = risk_profile.get("max_spread_width", 10)

    system = (
        "You are a multi-leg options spread portfolio manager. "
        "Your goal is defined-risk income and directional plays using vertical spreads, "
        "iron condors, and butterflies. "
        "You decide the ACTION TYPE, SYMBOL, and SPREAD TYPE only - the system picks "
        "exact strikes and expiration dates automatically. "
        "Output valid JSON only, no markdown.\n\n"
        "SPREAD TYPES:\n"
        "  iron_condor  - Sell OTM put spread + sell OTM call spread (credit, neutral)\n"
        "                 Best for: sideways markets, high IV, range-bound stocks\n"
        "  bull_call    - Buy lower call, sell higher call (debit, bullish)\n"
        "                 Best for: moderate upside expected, limited risk\n"
        "  bear_put     - Buy higher put, sell lower put (debit, bearish)\n"
        "                 Best for: moderate downside expected, limited risk\n"
        "  bull_put     - Sell higher put, buy lower put (credit, neutral-bullish)\n"
        "                 Best for: support holds, premium selling, high IV\n"
        "  bear_call    - Sell lower call, buy higher call (credit, neutral-bearish)\n"
        "                 Best for: resistance holds, premium selling, high IV\n"
        "  butterfly    - Buy 1 lower + buy 1 upper + sell 2 middle (debit, pinning)\n"
        "                 Best for: low-IV pinning targets, cheap defined-risk\n\n"
        "ACTIONS:\n"
        "  OPEN_SPREAD  - Open a new spread position\n"
        "  CLOSE        - Close an existing spread position (buy back)\n"
        "  SKIP         - Do nothing for a symbol this cycle\n\n"
        "CRITICAL RULES:\n"
        "  - All spreads are defined-risk (max loss is known upfront).\n"
        "  - Prefer credit spreads (iron condors, bull puts, bear calls) in high-IV.\n"
        "  - Prefer debit spreads (bull calls, bear puts) in low-IV with clear direction.\n"
        "  - Avoid spreads within 5 trading days of earnings.\n"
        f"  - Max spread width: ${max_width} between strikes.\n"
        "  - Be selective - quality over quantity.\n"
        "  - LIQUIDITY REQUIREMENT: Only trade options on stocks with avg daily volume > 500K shares.\n"
        "    Small-cap / micro-cap names (< $1B market cap) often have no usable options chains.\n"
        "    Prefer: AAPL, NVDA, TSLA, MSFT, AMZN, META, SPY, QQQ, AMD, GOOGL, BAC, JPM, XOM.\n"
        "    Avoid: YSS, SM, PL, SEDG, KLAR, GSIW, and any name you haven't seen liquid chains for."
    )

    # Build watchlist with prices and max loss estimates
    md = market_data or {}
    watchlist_lines = []
    for sym in watchlist:
        price = md.get(sym, {}).get("price", 0) or 0
        if price:
            est_max_loss = int(max_width * 100)  # width * 100 per contract
            watchlist_lines.append(
                f"  {sym}: ${price:.2f}  -> max loss per spread ~${est_max_loss}"
            )
        else:
            watchlist_lines.append(f"  {sym}")
    watchlist_text = "\n".join(watchlist_lines) if watchlist_lines else "(dynamic from research agent)"

    user = f"""== STRATEGY: {strategy_desc} ==

== MARKET ANALYSIS (Pass 1) ==
{json.dumps(analysis_json, indent=2)}

== CURRENT PORTFOLIO ==
Cash available: ${portfolio.cash:,.2f} ({portfolio.cash_pct:.1f}% of account)
Total value: ${portfolio.total_value:,.2f}
Open spread positions: {len(active_positions)}
Net theta: ${portfolio_greeks.total_theta:+.2f}/day

== ACTIVE SPREAD POSITIONS ==
{pos_text}

== RISK RULES ==
{risk_text}

== AVAILABLE WATCHLIST ==
{watchlist_text}

== YOUR PREVIOUS DECISIONS ==
{decision_history or "No history yet."}

Based on the market analysis, decide what spread actions to take.
Return JSON:
{{
  "market_comment": "brief reasoning about current conditions and spread type selection",
  "outlook": "BULLISH|CAUTIOUSLY_BULLISH|NEUTRAL|CAUTIOUSLY_BEARISH|BEARISH",
  "confidence": 0.0,
  "actions": [
    {{
      "type": "OPEN_SPREAD",
      "symbol": "AAPL",
      "spread_type": "iron_condor",
      "contracts": 1,
      "reason": "IV at 68th pct, range-bound between 170-190, ideal for iron condor"
    }},
    {{
      "type": "OPEN_SPREAD",
      "symbol": "MSFT",
      "spread_type": "bull_call",
      "contracts": 1,
      "reason": "Strong uptrend, low IV makes debit spread attractive"
    }},
    {{
      "type": "CLOSE",
      "symbol": "SPY",
      "position_id": 7,
      "reason": "Captured 65% of max premium, taking profit early"
    }},
    {{
      "type": "SKIP",
      "symbol": "TSLA",
      "reason": "Earnings in 4 days, IV crush risk"
    }}
  ]
}}

Rules:
- Max {max_spreads} open spread positions total; do not open more if at limit
- Keep at least {min_cash_pct}% of account in cash
- For CLOSE: include position_id of the spread to close
- You do NOT pick strikes or expiration dates - the system does that
- Spread type must be one of: iron_condor, bull_call, bear_put, bull_put, bear_call, butterfly
- If market is uncertain or no good setups, output SKIP actions or no OPEN_SPREAD
- Prefer closing positions that have captured ≥50% of max premium
"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _format_active_positions(positions: list[OptionsPosition]) -> str:
    """Brief summary for Pass 1."""
    if not positions:
        return "== ACTIVE SPREAD POSITIONS ==\n(none)"

    lines = [
        "== ACTIVE SPREAD POSITIONS ==",
        f"Open spreads: {len(positions)}",
        "",
    ]

    for p in positions:
        pl_str = f"${p.current_pl:+,.2f}" if p.current_pl is not None else "N/A"
        pct = f"{p.profit_captured_pct:.0f}% captured" if p.profit_captured_pct is not None else ""
        lines.append(
            f"  [{p.id}] {p.symbol} {p.spread_type}  "
            f"strikes={p.buy_strike}/{p.sell_strike}  exp={p.expiration_date}  DTE:{p.dte}  "
            f"P&L:{pl_str}  {pct}"
        )
    return "\n".join(lines)


def _format_active_positions_detailed(positions: list[OptionsPosition]) -> str:
    """Full detail for Pass 2 decision-making."""
    if not positions:
        return "(none)"

    lines = []
    for p in positions:
        pl_pct = f"{p.profit_captured_pct:.0f}%" if p.profit_captured_pct is not None else "?"
        pl_abs = f"${p.current_pl:+,.2f}" if p.current_pl is not None else "?"

        lines.append(
            f"ID:{p.id} | {p.symbol} {p.spread_type} | "
            f"Buy {p.buy_strike} / Sell {p.sell_strike} | "
            f"Exp:{p.expiration_date} DTE:{p.dte} | "
            f"Entry: ${p.entry_debit:.2f} | "
            f"Max profit:${p.max_profit:.2f} Max loss:${p.max_loss:.2f} | "
            f"P&L:{pl_abs} ({pl_pct} of max)"
        )
    return "\n".join(lines)


def _format_market_with_iv(
    market_data: dict,
    technical_signals: dict,
    iv_data: dict,
) -> str:
    """Format market data with IV percentiles — same as wheel prompt_builder."""
    lines = []
    for sym, data in market_data.items():
        price = data.get("price", 0)
        chg = data.get("change_pct", 0)
        w52h = data.get("52w_high", 0)
        w52l = data.get("52w_low", 0)
        iv_pct = iv_data.get(sym)
        if isinstance(iv_pct, dict):
            iv_str = (f"IV-pct:{iv_pct['percentile']:.0f}%"
                      f" IV-rank:{iv_pct['rank']:.0f}%"
                      f" HV:{iv_pct['current_hv']*100:.0f}%"
                      f"(52wH:{iv_pct['hv_52w_high']*100:.0f}%"
                      f"/L:{iv_pct['hv_52w_low']*100:.0f}%)")
        elif iv_pct is not None:
            iv_str = f"IV-pct:{iv_pct:.0f}%"
        else:
            iv_str = "IV:N/A"

        dist_low = ""
        if w52l and price:
            pct_above_low = (price - w52l) / w52l * 100
            dist_low = f"+{pct_above_low:.1f}%^52wLow"

        sig = technical_signals.get(sym)
        tech_str = ""
        if sig:
            summary = sig.to_summary() if hasattr(sig, "to_summary") else {}
            rsi = summary.get("RSI14", "N/A")
            macd_hist = summary.get("MACD_hist")
            interp = summary.get("interpretation", "")
            macd_str = f"{macd_hist:+.4f}" if macd_hist is not None else "N/A"
            tech_str = f"RSI:{rsi} MACDhist:{macd_str}"
            if interp:
                tech_str += f" | {interp}"

        lines.append(
            f"  {sym}: ${price:.2f} ({chg:+.2f}%) {iv_str} {dist_low} | {tech_str}"
        )
    return "\n".join(lines) if lines else "(no data)"


def _format_spreads_risk_rules(risk_profile: dict) -> str:
    return (
        f"Max open spreads: {risk_profile.get('max_open_spreads', 5)}\n"
        f"Min cash reserve: {risk_profile.get('min_cash_pct', 20)}% of account\n"
        f"Max spread width: ${risk_profile.get('max_spread_width', 10)}\n"
        f"Target DTE range: {risk_profile.get('target_dte_min', 21)}-{risk_profile.get('target_dte_max', 45)} days\n"
        f"Take profit at ≥{risk_profile.get('take_profit_pct', 50)}% of max premium captured\n"
        f"Stop loss at {risk_profile.get('stop_loss_pct', 100)}% of max loss\n"
        f"No spreads within 5 trading days of earnings"
    )
