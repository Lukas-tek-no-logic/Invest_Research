"""Build system and user prompts for the 2-pass LLM reasoning process."""

from __future__ import annotations

from datetime import datetime

from .portfolio_state import PortfolioState
from .technical_indicators import TechnicalSignals


def build_pass1_messages(
    portfolio: PortfolioState,
    market_data: dict[str, dict],
    technical_signals: dict[str, TechnicalSignals],
    news_text: str,
    decision_history: str,
    strategy_config: dict,
    earnings_text: str = "",
    fundamentals_text: str = "",
    research_brief: dict | None = None,
) -> list[dict[str, str]]:
    """Build messages for Pass 1: Market Analysis.

    The model analyzes market conditions and portfolio health WITHOUT making trades.
    """
    system_prompt = (
        "You are a senior financial analyst. Your job is to analyze current market "
        "conditions, portfolio health, and identify opportunities and threats.\n\n"
        "IMPORTANT RULES:\n"
        "- Do NOT recommend specific trades yet. Only analyze.\n"
        "- Be specific and data-driven in your analysis.\n"
        "- Consider both technical indicators and fundamental news.\n"
        "- Assess the overall market regime (bull, bear, sideways, high volatility).\n"
        "- Evaluate portfolio diversification and risk concentration.\n"
        "- For HELD positions: assess whether to stay (fundamental thesis intact?) "
        "or exit (earnings miss, analyst downgrades, thesis broken?).\n"
        "- For NEW opportunities: evaluate analyst consensus, growth trajectory, "
        "and whether the setup warrants entry.\n"
        "- ONLY cite data that is EXPLICITLY present in the market data / "
        "technical indicators sections below. Do NOT invent, estimate, or hallucinate "
        "any numbers (IV, SMA, MACD, RSI, analyst targets). If a metric is missing, "
        "say 'N/A' — never fabricate it.\n\n"
        "You MUST respond with valid JSON matching this schema:\n"
        "{\n"
        '  "market_regime": "BULL_TREND" | "BEAR_TREND" | "SIDEWAYS" | "HIGH_VOLATILITY",\n'
        '  "regime_reasoning": "string explaining why",\n'
        '  "sector_analysis": {\n'
        '    "SectorName": {"rating": "OVERWEIGHT|NEUTRAL|UNDERWEIGHT", "score": 2, "reason": "brief"}\n'
        '  },\n'
        '  "_sector_score_scale": "-2=strong underweight, -1=underweight, 0=neutral, +1=overweight, +2=strong overweight",\n'
        '  "portfolio_health": {\n'
        '    "diversification": "GOOD" | "POOR" | "CONCENTRATED",\n'
        '    "risk_level": "LOW" | "MEDIUM" | "HIGH",\n'
        '    "issues": ["list of issues or empty"]\n'
        "  },\n"
        '  "opportunities": [\n'
        '    {"symbol": "X", "signal": "reason this is an opportunity"}\n'
        "  ],\n"
        '  "threats": [\n'
        '    {"description": "macro or specific threat"}\n'
        "  ]\n"
        "}\n"
    )

    # Build market data section (includes VIX/TNX if passed from main.py)
    market_lines = ["== MARKET DATA =="]
    for symbol, data in market_data.items():
        if isinstance(data, dict):
            price = data.get("price", "N/A")
            chg = data.get("change_pct", 0)
            chg_str = f" ({chg:+.2f}%)" if isinstance(chg, (int, float)) and chg != 0 else ""
            label = data.get("label", "")
            label_str = f" [{label}]" if label else ""
            extras = []
            short_pct = data.get("short_pct")
            try:
                pe = float(data.get("pe") or 0) or None
            except (TypeError, ValueError):
                pe = None
            try:
                div = float(data.get("div_yield") or 0) or None
            except (TypeError, ValueError):
                div = None
            if pe:
                extras.append(f"P/E:{pe:.1f}")
            if div:
                extras.append(f"Yield:{div*100:.1f}%")
            if short_pct and short_pct > 0.05:  # only show if >5% (meaningful)
                extras.append(f"Short:{short_pct*100:.1f}%")
            extras_str = f" | {', '.join(extras)}" if extras else ""
            market_lines.append(f"{symbol}{label_str}: ${price}{chg_str}{extras_str}")
    market_text = "\n".join(market_lines)

    # Build technical indicators section
    tech_lines = ["== TECHNICAL INDICATORS =="]
    for symbol, signals in technical_signals.items():
        summary = signals.to_summary()
        interp = summary.pop("interpretation", "")
        parts = [f"{k}={v}" for k, v in summary.items()]
        line = f"{symbol}: {', '.join(parts)}"
        if interp:
            line += f" | {interp}"
        tech_lines.append(line)
    tech_text = "\n".join(tech_lines)

    # Strategy context
    strategy_text = (
        f"== ACCOUNT STRATEGY ==\n"
        f"Strategy: {strategy_config.get('strategy', 'balanced')}\n"
        f"Description: {strategy_config.get('strategy_description', '')}\n"
        f"Horizon: {strategy_config.get('horizon', 'weeks to months')}\n"
        f"Preferred metrics: {', '.join(strategy_config.get('preferred_metrics', []))}\n"
    )

    today = datetime.now().strftime("%A %Y-%m-%d")
    user_parts = [
        f"== TODAY: {today} ==",
        "",
        portfolio.to_prompt_text(),
        "",
        market_text,
        "",
        tech_text,
        "",
        news_text,
    ]
    if fundamentals_text:
        user_parts += ["", fundamentals_text]
    if earnings_text:
        user_parts += ["", earnings_text]
    if research_brief:
        brief_text = format_research_brief(research_brief)
        if brief_text:
            user_parts += ["", brief_text]
    user_parts += [
        "",
        decision_history,
        "",
        strategy_text,
        "",
        "Analyze the current situation and respond with the JSON analysis.",
    ]
    user_prompt = "\n".join(user_parts)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_pass2_messages(
    analysis_json: dict,
    portfolio: PortfolioState,
    strategy_config: dict,
    risk_profile: dict,
) -> list[dict[str, str]]:
    """Build messages for Pass 2: Trading Decision.

    The model receives the analysis and decides specific trades.
    """
    import json

    strategy = strategy_config.get("strategy", "balanced")
    prompt_style = strategy_config.get("prompt_style", "")
    horizon = strategy_config.get("horizon", "weeks to months")
    max_trades = risk_profile.get("max_trades_per_cycle", 5)
    max_position_pct = risk_profile.get("max_position_pct", 20)
    min_cash_pct = risk_profile.get("min_cash_pct", 10)
    stop_loss_pct = risk_profile.get("stop_loss_pct", -15)
    watchlist = strategy_config.get("watchlist", [])

    max_position_usd = portfolio.total_value * max_position_pct / 100

    system_prompt = (
        f"You are a portfolio manager executing a {strategy} strategy.\n"
        f"Style: {prompt_style}\n"
        f"Investment horizon: {horizon}\n\n"
        "Based on the market analysis below, decide specific trades.\n\n"
        "POSITION SIZING RULES:\n"
        f"- Maximum {max_trades} trades per cycle\n"
        f"- No single position > {max_position_pct}% of portfolio "
        f"(= ${max_position_usd:,.0f} at current portfolio value).\n"
        f"  Before each BUY: existing_position_value + amount_usd MUST NOT exceed ${max_position_usd:,.0f}.\n"
        f"- Keep minimum {min_cash_pct}% cash reserve\n"
        f"- Trade any symbol from the market data provided — you are NOT limited to a fixed list.\n"
        f"  Current universe: {', '.join(watchlist)}\n"
        "- You MUST justify every action with a specific thesis\n"
        "- If no good opportunities exist, it's OK to HOLD (empty actions list)\n\n"
        "DATA INTEGRITY:\n"
        "- ONLY reference data explicitly provided in the market analysis and constraints below.\n"
        "- Do NOT fabricate or assume any technical indicators, IV levels, analyst targets, "
        "or fundamentals that are not in the data. If a metric is missing, skip it.\n\n"
        "PORTFOLIO HYGIENE (address BEFORE new trades):\n"
        "- Zombie positions: any position worth < $10 or < 0.1% of portfolio should be "
        "SOLD immediately regardless of P/L — the fee to close now is less than the drag "
        "of holding dead weight.\n"
        "- Overconcentration: if any position exceeds 20% of portfolio, consider trimming "
        "it first — especially winners where you can lock in profit.\n"
        "- Pending sell proceeds: if you SELL in this cycle, that cash becomes available "
        "for BUYs in this same cycle. Plan sell-then-buy sequences.\n\n"
        "SYMBOL DISCOVERY:\n"
        "- In 'suggest_symbols' list up to 5 tickers you want to analyse next cycle.\n"
        "- Use this for any stock, ETF, or asset NOT currently in the universe that\n"
        "  you believe is worth researching (e.g. BRK-B, ARM, CELH, sector ETFs, etc.).\n"
        "- Next cycle those symbols will be fetched with full market data and technicals.\n\n"
        "EXIT CRITERIA — use percentage-based fields, NOT absolute price levels:\n"
        f"  stop_loss_pct: percentage below entry to stop out (e.g. {stop_loss_pct} means exit if down {abs(stop_loss_pct)}%)\n"
        "  take_profit_pct: percentage above entry to take profits (e.g. 25.0 means exit if up 25%)\n"
        "  time_stop_days: days after which to reassess the position (e.g. 30)\n"
        "  DO NOT use historical price levels or absolute dollar targets.\n"
        "  All exit levels are computed by the system from entry price + percentage.\n"
    )

    # Add strategy-specific rules
    if strategy == "value_investing":
        system_prompt += (
            "\nVALUE INVESTING DISCIPLINE:\n"
            "- Only buy if you can cite at least one fundamental metric "
            "(P/E, P/B, dividend yield, or FCF yield).\n"
            "- RSI/MACD alone is NOT sufficient justification for a value trade.\n"
            "- Do NOT buy if RSI > 70 (overbought) without an extraordinary fundamental discount.\n"
        )

    buying_power = max(0, portfolio.cash - portfolio.total_value * min_cash_pct / 100)

    if buying_power <= 0:
        # Show how much could be freed by selling
        total_sellable = sum(p.market_value for p in portfolio.positions)
        system_prompt += (
            f"\n⚠ HARD CONSTRAINT: buying power is $0 (cash ${portfolio.cash:,.2f} "
            f"≤ ${portfolio.total_value * min_cash_pct / 100:,.2f} reserve).\n"
            f"However, you CAN sell positions first to free up cash. "
            f"Total sellable: ${total_sellable:,.2f}. "
            f"Plan: SELL first, then BUY with freed cash in the same cycle.\n"
        )

    system_prompt += (
        "\nYou MUST respond with valid JSON matching this schema:\n"
        "{\n"
        '  "reasoning": "Detailed chain of thought explaining your overall approach",\n'
        '  "actions": [\n'
        "    {\n"
        '      "type": "BUY" | "SELL",\n'
        '      "symbol": "TICKER",\n'
        '      "amount_usd": 1000,\n'
        '      "urgency": "HIGH" | "MEDIUM" | "LOW",\n'
        '      "thesis": "Why this trade makes sense given the analysis",\n'
        '      "stop_loss_pct": -15.0,\n'
        '      "take_profit_pct": 25.0,\n'
        '      "time_stop_days": 30\n'
        "    }\n"
        "  ],\n"
        '  "portfolio_outlook": "BULLISH" | "CAUTIOUSLY_BULLISH" | "NEUTRAL" | "CAUTIOUSLY_BEARISH" | "BEARISH",\n'
        '  "confidence": 0.0 to 1.0,\n'
        '  "next_cycle_focus": "What to watch for in the next decision cycle",\n'
        '  "suggest_symbols": ["TICK1", "TICK2"]  // optional: up to 5 tickers to add to universe next cycle\n'
        "}\n"
    )

    # Build per-symbol trading constraints
    min_reserve = portfolio.total_value * min_cash_pct / 100

    constraints_lines = ["== TRADING CONSTRAINTS =="]

    if buying_power <= 0:
        constraints_lines.append(
            f"⚠ NO BUY BUDGET — cash ${portfolio.cash:,.2f} is below "
            f"${min_reserve:,.2f} min reserve. Only SELL or HOLD allowed."
        )
    else:
        constraints_lines.append(f"Buying power: ${buying_power:,.2f}")

    constraints_lines.append(f"Max position: {max_position_pct}% = ${max_position_usd:,.0f}")
    constraints_lines.append("")

    # Flag zombie positions and overconcentration
    zombies = [p for p in portfolio.positions if p.market_value < 10]
    concentrated = [p for p in portfolio.positions if p.weight_pct > max_position_pct]
    if zombies:
        syms = ", ".join(f"{p.symbol} (${p.market_value:.2f})" for p in zombies)
        constraints_lines.append(f"⚠ ZOMBIE POSITIONS (sell immediately): {syms}")
    if concentrated:
        syms = ", ".join(f"{p.symbol} ({p.weight_pct:.1f}%)" for p in concentrated)
        constraints_lines.append(f"⚠ OVERCONCENTRATED (consider trimming): {syms}")
    constraints_lines.append("")
    constraints_lines.append("Per-symbol room:")

    for p in sorted(portfolio.positions, key=lambda x: x.market_value, reverse=True):
        room = max(0, max_position_usd - p.market_value)
        buy_room = min(room, buying_power)
        status = "AT MAX" if room <= 0 else f"BUY up to ${buy_room:,.0f}"
        zombie_tag = " [ZOMBIE — sell]" if p.market_value < 10 else ""
        over_tag = " [OVERWEIGHT — trim]" if p.weight_pct > max_position_pct else ""
        constraints_lines.append(
            f"  {p.symbol}: {p.weight_pct:.1f}%{zombie_tag}{over_tag} | {status} | SELL up to ${p.market_value:,.0f}"
        )

    # Watchlist symbols without positions
    held = {p.symbol for p in portfolio.positions}
    for sym in watchlist:
        if sym not in held:
            constraints_lines.append(
                f"  {sym}: 0% | BUY up to ${min(max_position_usd, buying_power):,.0f}"
            )

    constraints_text = "\n".join(constraints_lines)

    today = datetime.now().strftime("%A %Y-%m-%d")
    user_prompt = (
        f"== TODAY: {today} ==\n\n"
        f"== MARKET ANALYSIS (from senior analyst) ==\n"
        f"{json.dumps(analysis_json, indent=2)}\n\n"
        f"{portfolio.to_prompt_text()}\n\n"
        f"{constraints_text}\n\n"
        f"Decide your trades and respond with the JSON decision."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def format_research_brief(brief: dict) -> str:
    """Format daily research brief for LLM prompt injection."""
    if not brief:
        return ""
    lines = ["== DAILY RESEARCH BRIEF (pre-market analysis) =="]

    regime = brief.get("market_regime", "")
    themes = brief.get("key_themes", [])
    macro = brief.get("macro_events_today", "")

    if regime:
        lines.append(f"Market regime: {regime}")
    if themes:
        lines.append(f"Key themes: {', '.join(themes)}")
    if macro:
        lines.append(f"Macro events today: {macro}")

    symbols = brief.get("top_symbols", [])
    if symbols:
        lines.append("\nTop research picks:")
        for s in symbols:
            conviction = s.get("conviction", "")
            direction = s.get("direction", "")
            tag = f"[{conviction}/{direction}]" if conviction else ""
            lines.append(
                f"  {s['symbol']} {tag}: {s.get('thesis', '')} "
                f"| catalyst: {s.get('catalyst', 'N/A')}"
            )

    geo_risks = brief.get("geopolitical_risks", [])
    if geo_risks:
        lines.append("\nGeopolitical risks:")
        for r in geo_risks:
            lines.append(
                f"  {r.get('event', '')}: {r.get('market_impact', '')} "
                f"(sectors: {', '.join(r.get('affected_sectors', []))})"
            )

    avoid = brief.get("avoid_today", [])
    if avoid:
        lines.append(f"\nAvoid today: {', '.join(str(a) for a in avoid)}")

    return "\n".join(lines)


def format_decision_history(history: list[dict], max_entries: int = 4) -> str:
    """Format past decisions for prompt injection (model memory)."""
    if not history:
        return "== YOUR PREVIOUS DECISIONS ==\n(No previous decisions - this is your first cycle)"

    lines = ["== YOUR PREVIOUS DECISIONS (last cycles) =="]
    for entry in history[-max_entries:]:
        date = entry.get("date", "Unknown")
        outlook = entry.get("outlook", "Unknown")
        confidence = entry.get("confidence", "N/A")
        lines.append(f"\n[{date}] Outlook: {outlook}, Confidence: {confidence}")

        actions = entry.get("actions", [])
        if actions:
            for action in actions:
                result_str = ""
                if action.get("result_pct") is not None:
                    result_str = f" | Result: {action['result_pct']:+.1f}%"
                lines.append(
                    f"  {action.get('type', '?')} {action.get('symbol', '?')} "
                    f"${action.get('amount_usd', 0):,.0f} "
                    f"(thesis: \"{action.get('thesis', '')}\")"
                    f"{result_str}"
                )
        else:
            lines.append("  HOLD (no trades)")
            reason = entry.get("hold_reason", "")
            if reason:
                lines.append(f"  Reason: \"{reason}\"")

    return "\n".join(lines)
