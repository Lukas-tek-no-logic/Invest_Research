"""Pass 0 scanner: lightweight LLM check before running full intraday cycle.

The scanner answers one question: has anything changed enough in the last
30 minutes to warrant running a full Pass 1 + Pass 2 analysis?

Signal semantics:
  HOLD → Nothing significant happened; skip cycle (save tokens and time).
  ACT  → Something notable occurred; proceed to full cycle.
"""

from __future__ import annotations

import structlog

from .portfolio_state import PortfolioState

logger = structlog.get_logger()

_SCAN_SYSTEM_PROMPT = """\
You are a fast market scanner for an automated intraday trading system.
Your ONLY job: decide whether conditions changed enough in the last 30 minutes
to warrant a full trading analysis (Pass 1 + Pass 2).

Respond with ONLY valid JSON — no markdown, no explanation:
{"signal": "HOLD" | "ACT", "reason": "<one sentence>", "confidence": 0.0-1.0}

HOLD when: all symbols moved less than 0.5% in 30 min, VIX is stable.
ACT  when: any symbol moved >0.5% in 30 min, VIX spiked, or notable drift.

Favour HOLD to avoid overtrading — only ACT when there is clear evidence.\
"""


def build_scan_messages(
    portfolio: PortfolioState,
    market_data: dict,
    last_cycle_prices: dict,
    strategy_config: dict,
) -> list[dict]:
    """Build Pass 0 scan messages (minimal context for the LLM).

    Intentionally sparse: only current prices vs. 30-min-ago prices, VIX
    delta, and portfolio cash %.  No technicals, news, or fundamentals.

    Args:
        portfolio: Current portfolio state.
        market_data: Current market data dict: symbol → {price, change_pct, …}.
        last_cycle_prices: Prices recorded at the previous cycle (~30 min ago).
        strategy_config: Account configuration dict.

    Returns:
        List with two dicts: [system_message, user_message].
    """
    lines: list[str] = [
        "== PORTFOLIO SNAPSHOT ==",
        f"Cash: {portfolio.cash_pct:.1f}%  |  Positions: {portfolio.position_count}",
        "",
        "== 30-MIN PRICE DELTA ==",
    ]

    data_gaps: list[str] = []
    for sym, data in market_data.items():
        if sym.startswith("^"):
            continue  # skip index symbols in the main table; handled separately below
        current_price = data.get("price", 0.0)
        prev_price = last_cycle_prices.get(sym)
        if current_price <= 0:
            data_gaps.append(sym)
            continue
        if prev_price is None or prev_price <= 0:
            # No prior price (first cycle, or the previous fetch failed).
            # Rendering it as Δ=+0.00% read as "flat market" — the LLM's
            # stated HOLD condition — and silently skipped cycles on no data.
            delta_str = "30m Δ=n/a"
        else:
            delta_str = f"30m Δ={(current_price - prev_price) / prev_price * 100:+.2f}%"
        intraday_pct = data.get("change_pct", 0.0)
        lines.append(
            f"  {sym}: ${current_price:.2f}"
            f"  {delta_str}"
            f"  intraday={intraday_pct:+.2f}%"
        )
    if data_gaps:
        lines.append(f"\n== DATA GAPS (no price this cycle) ==\n  {', '.join(data_gaps)}")

    vix_data = market_data.get("^VIX", {})
    if vix_data:
        vix_price = vix_data.get("price", 0)
        vix_change = vix_data.get("change_pct", 0.0)
        lines.append(f"\nVIX: {vix_price:.1f}  change today: {vix_change:+.2f}%")

    scan_threshold = (
        strategy_config
        .get("risk_profile", {})
        .get("scan_confidence_threshold", 0.6)
    )
    lines.append(f"\nStrategy: {strategy_config.get('strategy', 'unknown')}")
    lines.append(f"Required confidence to ACT: {scan_threshold}")

    return [
        {"role": "system", "content": _SCAN_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def parse_scan_signal(raw: dict) -> tuple[str, str, float]:
    """Parse the LLM scan response.

    Args:
        raw: Parsed JSON dict from the LLM.

    Returns:
        Tuple of (signal, reason, confidence).
        ``signal`` is ``"HOLD"`` or ``"ACT"``; defaults to ``"ACT"`` if
        unrecognised so that we err on the side of running the full cycle.
    """
    signal = str(raw.get("signal", "ACT")).upper().strip()
    if signal not in ("HOLD", "ACT"):
        signal = "ACT"
    reason = str(raw.get("reason", ""))
    confidence = float(raw.get("confidence", 0.5))
    return signal, reason, confidence
