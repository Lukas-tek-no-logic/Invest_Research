"""Build LLM prompts for Wheel Strategy options trading decisions.

Pass 1 – Market analysis: find quality underlyings with elevated IV and good
         support levels suitable for cash-secured puts.
Pass 2 – Concrete wheel actions: SELL_CSP, SELL_CC, CLOSE, or SKIP.
"""

from __future__ import annotations

import json

from ..portfolio_state import PortfolioState
from .greeks import PortfolioGreeks
from .positions import OptionsPosition


# ---------------------------------------------------------------------------
# Pass 1: Market + IV Analysis
# ---------------------------------------------------------------------------

def build_options_pass1_messages(
    portfolio: PortfolioState,
    market_data: dict[str, dict],
    technical_signals: dict,
    news_text: str,
    strategy_config: dict,
    active_positions: list[OptionsPosition],
    iv_data: dict[str, float | None],       # symbol → IV percentile (0-100)
    portfolio_greeks: PortfolioGreeks,
) -> list[dict]:
    """Pass 1: Market analysis focused on Wheel Strategy suitability.

    Goal: identify which watchlist symbols are good CSP candidates right now
    (elevated IV, solid support, not near earnings) and which assigned positions
    are ready for covered calls.
    """

    system = (
        "You are an options income analyst specialising in the Wheel Strategy "
        "(sell cash-secured puts → if assigned, sell covered calls → repeat). "
        "Analyse market conditions, implied volatility regime, and per-symbol "
        "suitability for selling puts or calls. "
        "Do NOT decide specific trades yet — that comes in Pass 2. "
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

Analyse the above for Wheel Strategy opportunities and return JSON:
{{
  "market_regime": "BULL_TREND|BEAR_TREND|SIDEWAYS|HIGH_VOLATILITY",
  "regime_reasoning": "brief explanation",
  "iv_regime": "HIGH|NORMAL|LOW",
  "iv_reasoning": "is premium selling favoured right now?",
  "sector_analysis": {{"sector_name": "BULLISH|NEUTRAL|BEARISH - reason"}},
  "per_symbol": {{
    "SYMBOL": {{
      "bias": "BULLISH|NEUTRAL|BEARISH",
      "iv_percentile": 45,
      "wheel_suitability": "GOOD|FAIR|POOR",
      "csp_candidate": true,
      "support_level": 150.0,
      "earnings_soon": false,
      "reason": "brief"
    }}
  }},
  "portfolio_health": {{
    "open_csps": 2,
    "open_ccs": 1,
    "assigned_positions": 0,
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
# Pass 2: Concrete Wheel Actions
# ---------------------------------------------------------------------------

def build_options_pass2_messages(
    analysis_json: dict,
    portfolio: PortfolioState,
    strategy_config: dict,
    risk_profile: dict,
    active_positions: list[OptionsPosition],
    portfolio_greeks: PortfolioGreeks,
    decision_history: str = "",
    market_data: dict | None = None,
) -> list[dict]:
    """Pass 2: Decide concrete Wheel Strategy actions.

    LLM chooses action type (SELL_CSP / SELL_CC / CLOSE / SKIP) and symbol.
    The selector module picks exact strikes and expiration dates.
    """

    pos_text = _format_active_positions_detailed(active_positions)
    risk_text = _format_wheel_risk_rules(risk_profile)
    watchlist = strategy_config.get("watchlist", [])
    strategy_desc = strategy_config.get("strategy_description", "Wheel Strategy")
    max_csp = risk_profile.get("max_open_csps", 3)
    min_cash_pct = risk_profile.get("min_cash_pct", 40)

    system = (
        "You are a Wheel Strategy portfolio manager. "
        "Your goal is sustainable income: sell OTM cash-secured puts on quality "
        "underlyings you are comfortable owning; if assigned, sell covered calls "
        "at or above cost basis to reduce it further and eventually exit with profit. "
        "You decide the ACTION TYPE and SYMBOL only — the system picks exact strikes "
        "and expiration dates automatically. "
        "Output valid JSON only, no markdown.\n\n"
        "WHEEL STRATEGY PHASES:\n"
        "  SELL_CSP  → Sell a cash-secured put (OTM, delta ~0.25-0.35, DTE 30-45)\n"
        "              Collect premium. If assigned → own stock at strike.\n"
        "  SELL_CC   → Against assigned stock, sell an OTM covered call\n"
        "              (strike ≥ cost basis, delta ~0.25, DTE 14-30).\n"
        "              Collect premium. If called away → complete the wheel.\n"
        "  CLOSE     → Buy back an existing CSP or CC position early\n"
        "              (e.g. 50%+ of premium captured, or before earnings).\n"
        "  SKIP      → Do nothing for a symbol this cycle (with reason).\n\n"
        "CRITICAL RULES:\n"
        "  • Only sell CSPs on stocks you would genuinely be happy to own.\n"
        "  • Avoid CSPs within 5 trading days of earnings.\n"
        "  • CC strike must be ≥ cost basis of the assigned stock.\n"
        "  • Prefer high-IV environments for premium selling.\n"
        "  • Be selective — quality over quantity."
    )

    # Build watchlist with per-symbol price and CSP collateral so the LLM
    # immediately knows which symbols fit the available cash
    md = market_data or {}
    watchlist_lines = []
    for sym in watchlist:
        price = md.get(sym, {}).get("price", 0) or 0
        if price:
            collateral = int(price * 100)
            fits = "✓ fits" if collateral <= portfolio.cash * 0.95 else "✗ too large for current cash"
            watchlist_lines.append(f"  {sym}: ${price:.2f}  → 1 CSP collateral ≈ ${collateral:,}  {fits}")
        else:
            watchlist_lines.append(f"  {sym}")
    watchlist_text = "\n".join(watchlist_lines) if watchlist_lines else ", ".join(watchlist)

    user = f"""== STRATEGY: {strategy_desc} ==

== MARKET ANALYSIS (Pass 1) ==
{json.dumps(analysis_json, indent=2)}

== CURRENT PORTFOLIO ==
Cash available: ${portfolio.cash:,.2f} ({portfolio.cash_pct:.1f}% of account)
Total value: ${portfolio.total_value:,.2f}
Open CSP/CC positions: {len(active_positions)}
Net theta: ${portfolio_greeks.total_theta:+.2f}/day
NOTE: For a CSP, the full strike × 100 is held as collateral. Only select symbols where the collateral fits your available cash.

== ACTIVE WHEEL POSITIONS ==
{pos_text}

== RISK RULES ==
{risk_text}

== AVAILABLE WATCHLIST (with CSP collateral requirement) ==
{watchlist_text}

== YOUR PREVIOUS DECISIONS ==
{decision_history or "No history yet."}

Based on the market analysis, decide what wheel actions to take.
Return JSON:
{{
  "market_comment": "brief reasoning about current conditions and why you are/are not selling premium",
  "outlook": "BULLISH|CAUTIOUSLY_BULLISH|NEUTRAL|CAUTIOUSLY_BEARISH|BEARISH",
  "confidence": 0.0,
  "actions": [
    {{
      "type": "SELL_CSP",
      "symbol": "AAPL",
      "contracts": 1,
      "reason": "IV at 68th pct, solid support at $170, no earnings for 6 weeks"
    }},
    {{
      "type": "SELL_CC",
      "symbol": "MSFT",
      "position_id": 42,
      "contracts": 1,
      "reason": "Assigned at $380; selling call above cost basis to reduce it"
    }},
    {{
      "type": "CLOSE",
      "symbol": "SPY",
      "position_id": 7,
      "reason": "Captured 72% of max premium, taking profit early"
    }},
    {{
      "type": "SKIP",
      "symbol": "TSLA",
      "reason": "Earnings in 4 days, IV spike too risky"
    }}
  ]
}}

Rules:
- Max {max_csp} open CSP positions total; do not open more if at limit
- Keep at least {min_cash_pct}% of account in cash (for assignment coverage)
- For SELL_CC: include position_id of the assigned stock position
- For CLOSE: include position_id of the CSP or CC to buy back
- You do NOT pick strikes or expiration dates — the system does that
- If market is uncertain or IV is low, output SKIP or no SELL_CSP actions
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
    """Brief summary for Pass 1 (just counts and types)."""
    if not positions:
        return "== ACTIVE WHEEL POSITIONS ==\n(none — no open CSPs or CCs)"

    csps = [p for p in positions if p.spread_type == "CASH_SECURED_PUT" and p.wheel_state != "ASSIGNED"]
    ccs = [p for p in positions if p.spread_type == "COVERED_CALL"]
    assigned = [p for p in positions if p.wheel_state == "ASSIGNED"]
    other = [p for p in positions if p.spread_type not in ("CASH_SECURED_PUT", "COVERED_CALL") and p.wheel_state != "ASSIGNED"]

    lines = [
        "== ACTIVE WHEEL POSITIONS ==",
        f"Open CSPs: {len(csps)}  |  Open CCs: {len(ccs)}  |  Assigned (holding shares): {len(assigned)}",
        "",
    ]

    for p in csps + ccs + other:
        pl_str = f"${p.current_pl:+,.2f}" if p.current_pl is not None else "N/A"
        pct = f"{p.profit_captured_pct:.0f}% captured" if p.profit_captured_pct is not None else ""
        lines.append(
            f"  [{p.id}] {p.symbol} {p.spread_type}  "
            f"strike={p.sell_strike}  exp={p.expiration_date}  DTE:{p.dte}  "
            f"P&L:{pl_str}  {pct}"
        )

    if assigned:
        lines.append("")
        lines.append("== ASSIGNED SHARES (sell Covered Calls on these!) ==")
        for p in assigned:
            lines.append(
                f"  [{p.id}] {p.symbol}: {p.wheel_shares} shares @ cost basis ${p.wheel_cost_basis:.2f}"
                f"  (assigned from CSP strike={p.sell_strike})"
                f"  ⚠ SELL_CC above ${p.wheel_cost_basis:.2f} to complete wheel cycle"
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

        if p.spread_type == "CASH_SECURED_PUT":
            # CSP: we only sold the put (sell_strike is the put strike)
            leg_desc = f"Short {p.sell_strike}P"
            cost_desc = f"Premium collected: ${p.entry_debit:.2f}/share"
        elif p.spread_type == "COVERED_CALL":
            # CC: we own stock and sold a call
            leg_desc = f"Short {p.sell_strike}C"
            cost_desc = f"Stock cost basis: ${p.buy_strike:.2f}  Call premium: ${p.entry_debit:.2f}/share"
        else:
            leg_desc = f"Buy {p.buy_strike} / Sell {p.sell_strike}"
            cost_desc = f"Entry debit: ${p.entry_debit:.2f}"

        lines.append(
            f"ID:{p.id} | {p.symbol} {p.spread_type} | "
            f"{leg_desc} | Exp:{p.expiration_date} DTE:{p.dte} | "
            f"{cost_desc} | "
            f"Max profit:${p.max_profit:.2f} | "
            f"P&L:{pl_abs} ({pl_pct} of max)"
        )
    return "\n".join(lines)


def _format_market_with_iv(
    market_data: dict,
    technical_signals: dict,
    iv_data: dict,
) -> str:
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

        # Distance from 52-week low (potential support proxy)
        dist_low = ""
        if w52l and price:
            pct_above_low = (price - w52l) / w52l * 100
            dist_low = f"+{pct_above_low:.1f}%↑52wLow"

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


def _format_wheel_risk_rules(risk_profile: dict) -> str:
    return (
        f"Max open CSP positions: {risk_profile.get('max_open_csps', 3)}\n"
        f"Max open CC positions per symbol: {risk_profile.get('max_ccs_per_symbol', 2)}\n"
        f"Min cash reserve: {risk_profile.get('min_cash_pct', 40)}% of account\n"
        f"CSP target delta: ~{risk_profile.get('csp_target_delta', 0.30)}\n"
        f"CSP DTE range: {risk_profile.get('csp_dte_min', 21)}-{risk_profile.get('csp_dte_max', 45)} days\n"
        f"CC target delta: ~{risk_profile.get('cc_target_delta', 0.25)}\n"
        f"CC DTE range: {risk_profile.get('cc_dte_min', 14)}-{risk_profile.get('cc_dte_max', 30)} days\n"
        f"No CSP within {risk_profile.get('earnings_blackout_days', 5)} days of earnings\n"
        f"CC strike must be ≥ stock cost basis (no loss-locking)\n"
        f"Early close if ≥{risk_profile.get('take_profit_pct', 50)}% of premium captured"
    )
