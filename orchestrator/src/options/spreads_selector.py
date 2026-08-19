"""Strike and expiry selector for multi-leg option spreads.

Selects optimal strikes for:
  bull_call  — buy lower call, sell higher call (debit)
  bear_put   — buy higher put, sell lower put (debit)
  bull_put   — sell higher put, buy lower put (credit)
  bear_call  — sell lower call, buy higher call (credit)
  iron_condor — bull_put + bear_call (credit, both OTM)
  butterfly   — buy 1 lower + buy 1 upper + sell 2 middle (debit)

Uses yfinance option chains and Black-Scholes delta targeting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd
import structlog

from .data import get_option_chain
from .greeks import RISK_FREE_RATE, calculate_greeks

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SelectedLeg:
    """A single option leg in a spread."""
    option_type: str        # "call" or "put"
    strike: float
    premium: float          # mid-price per share
    iv: float
    delta: float
    contract_symbol: str = ""
    side: str = ""          # "buy" or "sell"


@dataclass
class SelectedSpread:
    """Result of spread selection."""
    symbol: str
    spread_type: str        # iron_condor, bull_call, etc.
    expiration: str         # YYYY-MM-DD
    dte: int
    underlying_price: float
    legs: list[SelectedLeg]
    net_debit: float        # positive = debit paid, negative = credit received
    max_profit: float       # per contract ($)
    max_loss: float         # per contract ($, always positive)
    contracts: int = 1


# ---------------------------------------------------------------------------
# Public selector
# ---------------------------------------------------------------------------

def select_spread(
    symbol: str,
    spread_type: str,
    contracts: int = 1,
    dte_min: int = 21,
    dte_max: int = 45,
    max_width: float = 10.0,
    target_delta: float = 0.30,
) -> SelectedSpread | None:
    """Select strikes for any supported spread type.

    Args:
        symbol:       Underlying ticker.
        spread_type:  One of: bull_call, bear_put, bull_put, bear_call, iron_condor, butterfly.
        contracts:    Number of contracts.
        dte_min/max:  DTE range for expiration search.
        max_width:    Maximum width between strikes (dollars).
        target_delta: Absolute delta target for the short leg (~0.25-0.35).

    Returns:
        SelectedSpread or None if no suitable chain found.
    """
    chain = get_option_chain(symbol, min_dte=dte_min, max_dte=dte_max)
    if chain is None:
        logger.warning("spread_selector_no_chain", symbol=symbol, spread_type=spread_type)
        return None

    S = chain.underlying_price
    today = date.today()
    exp_date = datetime.strptime(chain.expiration, "%Y-%m-%d").date()
    t = max((exp_date - today).days / 365.0, 0.001)

    selector_map = {
        "bull_call": _select_bull_call,
        "bear_put": _select_bear_put,
        "bull_put": _select_bull_put,
        "bear_call": _select_bear_call,
        "iron_condor": _select_iron_condor,
        "butterfly": _select_butterfly,
    }

    fn = selector_map.get(spread_type)
    if fn is None:
        logger.warning("spread_selector_unknown_type", spread_type=spread_type)
        return None

    result = fn(
        symbol=symbol,
        chain_calls=chain.calls,
        chain_puts=chain.puts,
        S=S,
        t=t,
        expiration=chain.expiration,
        dte=chain.dte,
        max_width=max_width,
        target_delta=target_delta,
        contracts=contracts,
    )

    if result is not None:
        logger.info(
            "spread_selected",
            symbol=symbol, spread_type=spread_type,
            expiration=chain.expiration, dte=chain.dte,
            legs=len(result.legs), net_debit=result.net_debit,
            max_profit=result.max_profit, max_loss=result.max_loss,
        )

    return result


# ---------------------------------------------------------------------------
# Per-type selectors
# ---------------------------------------------------------------------------

def _select_bull_call(
    symbol: str,
    chain_calls: pd.DataFrame,
    chain_puts: pd.DataFrame,
    S: float, t: float,
    expiration: str, dte: int,
    max_width: float, target_delta: float,
    contracts: int,
) -> SelectedSpread | None:
    """Bull call spread: buy lower call (ITM/ATM), sell higher call (OTM). Debit."""
    if chain_calls is None or chain_calls.empty:
        return None

    # Buy leg: ATM or slightly ITM call (delta ~0.50-0.60)
    buy_row = _find_delta_row(chain_calls, "call", S, t, 0.50)
    if buy_row is None:
        return None

    buy_strike = float(buy_row["strike"])

    # Sell leg: OTM call above buy strike, within max_width
    sell_candidates = chain_calls[
        (chain_calls["strike"] > buy_strike) &
        (chain_calls["strike"] <= buy_strike + max_width)
    ]
    if sell_candidates.empty:
        return None

    sell_row = _find_delta_row(sell_candidates, "call", S, t, target_delta)
    if sell_row is None:
        sell_row = sell_candidates.iloc[0]

    return _build_two_leg_spread(
        symbol=symbol, spread_type="bull_call",
        expiration=expiration, dte=dte, S=S, t=t,
        buy_row=buy_row, sell_row=sell_row,
        buy_type="call", sell_type="call",
        contracts=contracts, is_debit=True,
    )


def _select_bear_put(
    symbol: str,
    chain_calls: pd.DataFrame,
    chain_puts: pd.DataFrame,
    S: float, t: float,
    expiration: str, dte: int,
    max_width: float, target_delta: float,
    contracts: int,
) -> SelectedSpread | None:
    """Bear put spread: buy higher put (ITM/ATM), sell lower put (OTM). Debit."""
    if chain_puts is None or chain_puts.empty:
        return None

    # Buy leg: ATM or slightly ITM put (delta ~-0.50)
    buy_row = _find_delta_row(chain_puts, "put", S, t, -0.50)
    if buy_row is None:
        return None

    buy_strike = float(buy_row["strike"])

    # Sell leg: OTM put below buy strike, within max_width
    sell_candidates = chain_puts[
        (chain_puts["strike"] < buy_strike) &
        (chain_puts["strike"] >= buy_strike - max_width)
    ]
    if sell_candidates.empty:
        return None

    sell_row = _find_delta_row(sell_candidates, "put", S, t, -target_delta)
    if sell_row is None:
        sell_row = sell_candidates.iloc[-1]

    return _build_two_leg_spread(
        symbol=symbol, spread_type="bear_put",
        expiration=expiration, dte=dte, S=S, t=t,
        buy_row=buy_row, sell_row=sell_row,
        buy_type="put", sell_type="put",
        contracts=contracts, is_debit=True,
    )


def _select_bull_put(
    symbol: str,
    chain_calls: pd.DataFrame,
    chain_puts: pd.DataFrame,
    S: float, t: float,
    expiration: str, dte: int,
    max_width: float, target_delta: float,
    contracts: int,
) -> SelectedSpread | None:
    """Bull put spread: sell higher put (OTM), buy lower put (further OTM). Credit."""
    if chain_puts is None or chain_puts.empty:
        return None

    # OTM puts (strike < underlying)
    otm_puts = chain_puts[chain_puts["strike"] < S].copy()
    if otm_puts.empty:
        return None

    # Sell leg: OTM put near target delta
    sell_row = _find_delta_row(otm_puts, "put", S, t, -target_delta)
    if sell_row is None:
        return None

    sell_strike = float(sell_row["strike"])

    # Buy leg: further OTM put below sell strike, within max_width
    buy_candidates = otm_puts[
        (otm_puts["strike"] < sell_strike) &
        (otm_puts["strike"] >= sell_strike - max_width)
    ]
    if buy_candidates.empty:
        return None

    # Pick the lowest strike within range for maximum width (defined risk)
    buy_row = buy_candidates.loc[buy_candidates["strike"].idxmin()]

    return _build_two_leg_spread(
        symbol=symbol, spread_type="bull_put",
        expiration=expiration, dte=dte, S=S, t=t,
        buy_row=buy_row, sell_row=sell_row,
        buy_type="put", sell_type="put",
        contracts=contracts, is_debit=False,
    )


def _select_bear_call(
    symbol: str,
    chain_calls: pd.DataFrame,
    chain_puts: pd.DataFrame,
    S: float, t: float,
    expiration: str, dte: int,
    max_width: float, target_delta: float,
    contracts: int,
) -> SelectedSpread | None:
    """Bear call spread: sell lower call (OTM), buy higher call (further OTM). Credit."""
    if chain_calls is None or chain_calls.empty:
        return None

    # OTM calls (strike > underlying)
    otm_calls = chain_calls[chain_calls["strike"] > S].copy()
    if otm_calls.empty:
        return None

    # Sell leg: OTM call near target delta
    sell_row = _find_delta_row(otm_calls, "call", S, t, target_delta)
    if sell_row is None:
        return None

    sell_strike = float(sell_row["strike"])

    # Buy leg: further OTM call above sell strike, within max_width
    buy_candidates = otm_calls[
        (otm_calls["strike"] > sell_strike) &
        (otm_calls["strike"] <= sell_strike + max_width)
    ]
    if buy_candidates.empty:
        return None

    buy_row = buy_candidates.loc[buy_candidates["strike"].idxmax()]

    return _build_two_leg_spread(
        symbol=symbol, spread_type="bear_call",
        expiration=expiration, dte=dte, S=S, t=t,
        buy_row=buy_row, sell_row=sell_row,
        buy_type="call", sell_type="call",
        contracts=contracts, is_debit=False,
    )


def _select_iron_condor(
    symbol: str,
    chain_calls: pd.DataFrame,
    chain_puts: pd.DataFrame,
    S: float, t: float,
    expiration: str, dte: int,
    max_width: float, target_delta: float,
    contracts: int,
) -> SelectedSpread | None:
    """Iron condor: bull_put + bear_call (credit, both OTM wings)."""
    if chain_calls is None or chain_calls.empty or chain_puts is None or chain_puts.empty:
        return None

    # Use slightly lower delta for IC wings (~0.20) for wider range
    wing_delta = min(target_delta, 0.20)

    # ---- Put side (bull put spread) ----
    otm_puts = chain_puts[chain_puts["strike"] < S].copy()
    if otm_puts.empty:
        return None

    put_sell_row = _find_delta_row(otm_puts, "put", S, t, -wing_delta)
    if put_sell_row is None:
        return None
    put_sell_strike = float(put_sell_row["strike"])

    put_buy_cands = otm_puts[
        (otm_puts["strike"] < put_sell_strike) &
        (otm_puts["strike"] >= put_sell_strike - max_width)
    ]
    if put_buy_cands.empty:
        return None
    put_buy_row = put_buy_cands.loc[put_buy_cands["strike"].idxmin()]

    # ---- Call side (bear call spread) ----
    otm_calls = chain_calls[chain_calls["strike"] > S].copy()
    if otm_calls.empty:
        return None

    call_sell_row = _find_delta_row(otm_calls, "call", S, t, wing_delta)
    if call_sell_row is None:
        return None
    call_sell_strike = float(call_sell_row["strike"])

    call_buy_cands = otm_calls[
        (otm_calls["strike"] > call_sell_strike) &
        (otm_calls["strike"] <= call_sell_strike + max_width)
    ]
    if call_buy_cands.empty:
        return None
    call_buy_row = call_buy_cands.loc[call_buy_cands["strike"].idxmax()]

    # Build 4-leg spread
    legs = []
    for row, opt_type, side in [
        (put_buy_row, "put", "buy"),
        (put_sell_row, "put", "sell"),
        (call_sell_row, "call", "sell"),
        (call_buy_row, "call", "buy"),
    ]:
        iv = float(row.get("impliedVolatility", 0.25) or 0.25)
        g = calculate_greeks(opt_type, S, float(row["strike"]), t, iv, RISK_FREE_RATE)
        delta = g.delta if g else 0.0
        legs.append(SelectedLeg(
            option_type=opt_type,
            strike=float(row["strike"]),
            premium=_mid_price(row, side),
            iv=iv,
            delta=delta,
            contract_symbol=str(row.get("contractSymbol", "") or ""),
            side=side,
        ))

    # Net credit = (sell premiums) - (buy premiums)
    credit = sum(l.premium for l in legs if l.side == "sell") - sum(l.premium for l in legs if l.side == "buy")
    put_width = abs(float(put_sell_row["strike"]) - float(put_buy_row["strike"]))
    call_width = abs(float(call_buy_row["strike"]) - float(call_sell_row["strike"]))
    wider_wing = max(put_width, call_width)

    max_profit = round(credit * 100 * contracts, 2)
    max_loss = round((wider_wing - credit) * 100 * contracts, 2)

    if credit <= 0 or max_loss <= 0:
        logger.warning("iron_condor_no_credit", symbol=symbol, credit=credit)
        return None

    return SelectedSpread(
        symbol=symbol,
        spread_type="iron_condor",
        expiration=expiration,
        dte=dte,
        underlying_price=S,
        legs=legs,
        net_debit=-credit,  # negative = credit received
        max_profit=max_profit,
        max_loss=max_loss,
        contracts=contracts,
    )


def _select_butterfly(
    symbol: str,
    chain_calls: pd.DataFrame,
    chain_puts: pd.DataFrame,
    S: float, t: float,
    expiration: str, dte: int,
    max_width: float, target_delta: float,
    contracts: int,
) -> SelectedSpread | None:
    """Long call butterfly: buy 1 lower + buy 1 upper + sell 2 middle. Debit."""
    if chain_calls is None or chain_calls.empty:
        return None

    # Middle strike: ATM
    atm_dist = (chain_calls["strike"] - S).abs()
    mid_idx = atm_dist.idxmin()
    mid_row = chain_calls.loc[mid_idx]
    mid_strike = float(mid_row["strike"])

    # Half-width: use min(max_width/2, available strikes)
    half_width = max_width / 2

    # Lower wing: strike = mid - half_width (closest available)
    lower_cands = chain_calls[
        (chain_calls["strike"] >= mid_strike - half_width - 1) &
        (chain_calls["strike"] < mid_strike)
    ]
    if lower_cands.empty:
        return None
    lower_row = lower_cands.loc[(lower_cands["strike"] - (mid_strike - half_width)).abs().idxmin()]

    # Upper wing: strike = mid + half_width (closest available)
    upper_cands = chain_calls[
        (chain_calls["strike"] > mid_strike) &
        (chain_calls["strike"] <= mid_strike + half_width + 1)
    ]
    if upper_cands.empty:
        return None
    upper_row = upper_cands.loc[(upper_cands["strike"] - (mid_strike + half_width)).abs().idxmin()]

    legs = []
    for row, side, qty_note in [
        (lower_row, "buy", "1x"),
        (mid_row, "sell", "2x"),
        (upper_row, "buy", "1x"),
    ]:
        iv = float(row.get("impliedVolatility", 0.25) or 0.25)
        g = calculate_greeks("call", S, float(row["strike"]), t, iv, RISK_FREE_RATE)
        delta = g.delta if g else 0.0
        legs.append(SelectedLeg(
            option_type="call",
            strike=float(row["strike"]),
            premium=_mid_price(row, side),
            iv=iv,
            delta=delta,
            contract_symbol=str(row.get("contractSymbol", "") or ""),
            side=side,
        ))

    # Net debit: buy lower + buy upper - 2 * sell middle
    debit = legs[0].premium + legs[2].premium - 2 * legs[1].premium
    lower_width = abs(float(mid_row["strike"]) - float(lower_row["strike"]))
    max_profit = round((lower_width - debit) * 100 * contracts, 2)
    max_loss = round(debit * 100 * contracts, 2)

    if debit <= 0 or max_profit <= 0:
        logger.warning("butterfly_bad_pricing", symbol=symbol, debit=debit)
        return None

    return SelectedSpread(
        symbol=symbol,
        spread_type="butterfly",
        expiration=expiration,
        dte=dte,
        underlying_price=S,
        legs=legs,
        net_debit=debit,
        max_profit=max_profit,
        max_loss=max_loss,
        contracts=contracts,
    )


# ---------------------------------------------------------------------------
# Two-leg spread builder
# ---------------------------------------------------------------------------

def _build_two_leg_spread(
    symbol: str,
    spread_type: str,
    expiration: str,
    dte: int,
    S: float,
    t: float,
    buy_row: pd.Series,
    sell_row: pd.Series,
    buy_type: str,
    sell_type: str,
    contracts: int,
    is_debit: bool,
) -> SelectedSpread | None:
    """Build a two-leg vertical spread from selected rows."""
    buy_strike = float(buy_row["strike"])
    sell_strike = float(sell_row["strike"])
    buy_premium = _mid_price(buy_row, "buy")
    sell_premium = _mid_price(sell_row, "sell")

    buy_iv = float(buy_row.get("impliedVolatility", 0.25) or 0.25)
    sell_iv = float(sell_row.get("impliedVolatility", 0.25) or 0.25)

    g_buy = calculate_greeks(buy_type, S, buy_strike, t, buy_iv, RISK_FREE_RATE)
    g_sell = calculate_greeks(sell_type, S, sell_strike, t, sell_iv, RISK_FREE_RATE)

    buy_delta = g_buy.delta if g_buy else 0.0
    sell_delta = g_sell.delta if g_sell else 0.0

    legs = [
        SelectedLeg(
            option_type=buy_type, strike=buy_strike,
            premium=buy_premium, iv=buy_iv, delta=buy_delta,
            contract_symbol=str(buy_row.get("contractSymbol", "") or ""),
            side="buy",
        ),
        SelectedLeg(
            option_type=sell_type, strike=sell_strike,
            premium=sell_premium, iv=sell_iv, delta=sell_delta,
            contract_symbol=str(sell_row.get("contractSymbol", "") or ""),
            side="sell",
        ),
    ]

    width = abs(buy_strike - sell_strike)

    if is_debit:
        # Debit spread: pay (buy - sell), max profit = width - debit
        net_debit = round(buy_premium - sell_premium, 2)
        if net_debit <= 0:
            logger.warning("spread_debit_negative", symbol=symbol, spread_type=spread_type)
            return None
        max_profit = round((width - net_debit) * 100 * contracts, 2)
        max_loss = round(net_debit * 100 * contracts, 2)
    else:
        # Credit spread: receive (sell - buy), max loss = width - credit
        credit = round(sell_premium - buy_premium, 2)
        if credit <= 0:
            logger.warning("spread_credit_negative", symbol=symbol, spread_type=spread_type)
            return None
        net_debit = -credit  # negative = credit received
        max_profit = round(credit * 100 * contracts, 2)
        max_loss = round((width - credit) * 100 * contracts, 2)

    if max_profit <= 0 or max_loss <= 0:
        logger.warning("spread_bad_pnl", symbol=symbol, max_profit=max_profit, max_loss=max_loss)
        return None

    return SelectedSpread(
        symbol=symbol,
        spread_type=spread_type,
        expiration=expiration,
        dte=dte,
        underlying_price=S,
        legs=legs,
        net_debit=net_debit,
        max_profit=max_profit,
        max_loss=max_loss,
        contracts=contracts,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _find_delta_row(
    df: pd.DataFrame,
    option_type: str,
    S: float,
    t: float,
    target_delta: float,
) -> pd.Series | None:
    """Return the row whose BS-calculated delta is closest to target_delta."""
    best_row = None
    best_dist = float("inf")

    for idx, row in df.iterrows():
        iv = float(row.get("impliedVolatility", 0) or 0)
        strike = float(row["strike"])
        if iv <= 0:
            continue
        g = calculate_greeks(option_type, S, strike, t, iv, RISK_FREE_RATE)
        if g is None:
            continue
        dist = abs(g.delta - target_delta)
        if dist < best_dist:
            best_dist = dist
            best_row = row

    return best_row


HAIRCUT_FRACTION = 0.25  # of the half-spread; 1% of lastPrice when no bid/ask


def _mid_price(row: pd.Series, side: str | None = None) -> float:
    """Return bid/ask midpoint with an execution haircut.

    side="sell": fill below mid (premium received is lower than mid);
    side="buy":  fill above mid. side=None: raw mid (marking, not fills).
    Booking fills at raw mid overstated every credit by half the spread —
    on a 4-leg condor, four times over.
    """
    bid = float(row.get("bid", 0) or 0)
    ask = float(row.get("ask", 0) or 0)
    if bid > 0 and ask > 0:
        mid = (bid + ask) / 2
        half = (ask - bid) / 2
        if side == "sell":
            mid -= HAIRCUT_FRACTION * half
        elif side == "buy":
            mid += HAIRCUT_FRACTION * half
        return round(mid, 2)
    last = float(row.get("lastPrice", 0) or 0)
    if side == "sell":
        last *= 0.99
    elif side == "buy":
        last *= 1.01
    return round(last, 2)
