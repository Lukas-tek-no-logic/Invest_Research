"""Deterministic market regime classification from hard data.

The LLM used to classify the regime itself in Pass 1 and chronically labeled
low-VIX days HIGH_VOLATILITY because the screener feeds it outlier movers by
construction. The Pass 2 aggression rules then turned that mislabel into
permanent defensiveness (no BUYs, cash pile-up in a bull market).

Regime is now computed here from VIX and SPY trend data and injected into the
prompts as an authoritative fact. Hysteresis prevents whipsaw around single
thresholds: entering HIGH_VOLATILITY is easy (any spike trigger), leaving it
requires VIX to stay calm for several sessions.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

logger = structlog.get_logger()

# Entry triggers for HIGH_VOLATILITY (any one is enough)
VIX_ENTER_LEVEL = 25.0        # absolute VIX close
VIX_SPIKE_5D_PCT = 40.0       # VIX up >40% over 5 sessions (catches Feb-2020-style ramps)
SPY_DRAWDOWN_5D_PCT = 4.0     # SPY down >4% from its 5-session high

# Exit condition: VIX below this level for N consecutive sessions
VIX_EXIT_LEVEL = 20.0
VIX_EXIT_SESSIONS = 3

# How far back the entry/exit state machine looks
LOOKBACK_SESSIONS = 40


@dataclass
class RegimeResult:
    regime: str          # BULL_TREND | BEAR_TREND | SIDEWAYS | HIGH_VOLATILITY
    reasoning: str       # human-readable justification with the numbers used
    vix: float
    spy: float
    spy_sma50: float | None
    spy_sma200: float | None

    def to_prompt_text(self) -> str:
        return (
            "== MARKET REGIME (AUTHORITATIVE — computed from hard data) ==\n"
            f"Regime: {self.regime}\n"
            f"Basis: {self.reasoning}\n"
            "This classification is computed deterministically from VIX and SPY trend data.\n"
            "Copy it VERBATIM into the market_regime field. Do NOT re-classify.\n"
            "Large single-stock moves in the screener data do NOT change the market "
            "regime — screeners select outliers by construction.\n"
        )


def _sma(closes: list[float], window: int) -> float | None:
    if len(closes) < window:
        return None
    return sum(closes[-window:]) / window


def _entry_triggered(vix: list[float], spy: list[float], i_from_end: int) -> str | None:
    """Return the trigger name if a HIGH_VOLATILITY entry fires at session -i_from_end.

    ``i_from_end`` counts from the end: 1 = latest session. VIX and SPY series
    are aligned from the end (both are daily closes over the same market days).
    """
    v = vix[-i_from_end]
    if v > VIX_ENTER_LEVEL:
        return f"VIX {v:.1f} > {VIX_ENTER_LEVEL:.0f}"

    if len(vix) >= i_from_end + 5:
        v_5d_ago = vix[-(i_from_end + 5)]
        if v_5d_ago > 0 and (v / v_5d_ago - 1) * 100 > VIX_SPIKE_5D_PCT:
            return f"VIX +{(v / v_5d_ago - 1) * 100:.0f}% in 5 sessions"

    if len(spy) >= i_from_end + 4:
        window = spy[-(i_from_end + 4):len(spy) - i_from_end + 1]
        high = max(window)
        last = spy[-i_from_end]
        if high > 0 and (high - last) / high * 100 > SPY_DRAWDOWN_5D_PCT:
            return f"SPY -{(high - last) / high * 100:.1f}% from 5-session high"

    return None


def compute_regime(vix_closes: list[float], spy_closes: list[float]) -> RegimeResult:
    """Classify the market regime from VIX and SPY daily closes.

    Raises ValueError if there is not enough data to classify safely —
    callers should fall back to their previous behavior in that case.
    """
    vix = [float(v) for v in vix_closes if v and v > 0]
    spy = [float(s) for s in spy_closes if s and s > 0]
    if len(vix) < 10 or len(spy) < 60:
        raise ValueError(
            f"not enough data for regime: vix={len(vix)} sessions, spy={len(spy)} sessions"
        )

    vix_now = vix[-1]
    spy_now = spy[-1]
    sma50 = _sma(spy, 50)
    sma200 = _sma(spy, 200)

    # ── HIGH_VOLATILITY state machine with hysteresis over the lookback ──
    # Walk forward from LOOKBACK_SESSIONS ago; enter on any trigger, exit only
    # after VIX_EXIT_SESSIONS consecutive closes below VIX_EXIT_LEVEL.
    span = min(LOOKBACK_SESSIONS, len(vix) - 5)
    in_high_vol = False
    active_trigger = ""
    for i_from_end in range(span, 0, -1):
        trigger = _entry_triggered(vix, spy, i_from_end)
        if trigger:
            in_high_vol = True
            active_trigger = trigger
            continue
        if in_high_vol and i_from_end + VIX_EXIT_SESSIONS - 1 <= len(vix):
            recent = vix[-(i_from_end + VIX_EXIT_SESSIONS - 1):len(vix) - i_from_end + 1]
            if len(recent) >= VIX_EXIT_SESSIONS and all(v < VIX_EXIT_LEVEL for v in recent):
                in_high_vol = False
                active_trigger = ""

    if in_high_vol:
        regime = "HIGH_VOLATILITY"
        reasoning = (
            f"{active_trigger}; exit requires VIX < {VIX_EXIT_LEVEL:.0f} for "
            f"{VIX_EXIT_SESSIONS} consecutive sessions (VIX now {vix_now:.1f})"
        )
    elif sma200 is not None and spy_now < sma200 and sma50 is not None and spy_now < sma50:
        regime = "BEAR_TREND"
        reasoning = (
            f"SPY ${spy_now:.0f} below SMA200 ${sma200:.0f} and SMA50 ${sma50:.0f}, "
            f"VIX {vix_now:.1f}"
        )
    elif sma50 is not None and spy_now > sma50:
        regime = "BULL_TREND"
        sma200_txt = f", above SMA200 ${sma200:.0f}" if sma200 else ""
        reasoning = f"VIX {vix_now:.1f} < {VIX_EXIT_LEVEL:.0f}, SPY ${spy_now:.0f} above SMA50 ${sma50:.0f}{sma200_txt}"
    else:
        regime = "SIDEWAYS"
        sma50_txt = f"below SMA50 ${sma50:.0f}" if sma50 else "no SMA50 available"
        reasoning = f"VIX {vix_now:.1f} calm but SPY ${spy_now:.0f} {sma50_txt} — no confirmed trend"

    logger.info(
        "deterministic_regime",
        regime=regime,
        vix=round(vix_now, 2),
        spy=round(spy_now, 2),
        sma50=round(sma50, 2) if sma50 else None,
        sma200=round(sma200, 2) if sma200 else None,
    )
    return RegimeResult(
        regime=regime,
        reasoning=reasoning,
        vix=vix_now,
        spy=spy_now,
        spy_sma50=sma50,
        spy_sma200=sma200,
    )
