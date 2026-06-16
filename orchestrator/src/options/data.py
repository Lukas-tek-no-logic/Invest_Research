"""Fetch option chains from yfinance, filter by DTE and liquidity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd
import structlog
import yfinance as yf

from ..yf_throttle import paced_call

logger = structlog.get_logger()

# Liquidity minimums
MIN_OPEN_INTEREST = 50
MIN_BID = 0.05
MIN_VOLUME = 5


@dataclass
class OptionChainData:
    symbol: str
    underlying_price: float
    expiration: str          # YYYY-MM-DD
    dte: int                 # days to expiration
    calls: pd.DataFrame      # filtered, liquid calls
    puts: pd.DataFrame       # filtered, liquid puts


def get_option_chain(
    symbol: str,
    min_dte: int = 14,
    max_dte: int = 75,
) -> OptionChainData | None:
    """Fetch and filter option chain for the closest expiry within DTE range.

    Returns the expiry closest to 30-45 DTE within [min_dte, max_dte].
    Returns None if no suitable chain found.
    """
    try:
        ticker = yf.Ticker(symbol)

        # Current underlying price
        underlying_price = float(
            paced_call(lambda: getattr(ticker.fast_info, "last_price", 0), label=f"opt-fast:{symbol}") or 0
        )
        if underlying_price <= 0:
            logger.warning("options_no_price", symbol=symbol)
            return None

        # Available expirations
        expirations = paced_call(lambda: ticker.options, label=f"opt-exp:{symbol}")
        if not expirations:
            logger.warning("options_no_expirations", symbol=symbol)
            return None

        today = date.today()
        target_dte = 35  # ideal DTE

        # Score each expiry by distance from target
        candidates = []
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if min_dte <= dte <= max_dte:
                candidates.append((abs(dte - target_dte), dte, exp_str))

        if not candidates:
            logger.warning("options_no_expiry_in_range", symbol=symbol, min_dte=min_dte, max_dte=max_dte)
            return None

        candidates.sort()
        _, dte, expiration = candidates[0]

        # Fetch chain
        chain = paced_call(lambda: ticker.option_chain(expiration), label=f"opt-chain:{symbol}")

        calls = _filter_chain(chain.calls, underlying_price)
        puts = _filter_chain(chain.puts, underlying_price)

        if calls.empty and puts.empty:
            logger.warning("options_chain_illiquid", symbol=symbol, expiration=expiration)
            return None

        logger.info(
            "options_chain_fetched",
            symbol=symbol,
            expiration=expiration,
            dte=dte,
            calls=len(calls),
            puts=len(puts),
        )
        return OptionChainData(
            symbol=symbol,
            underlying_price=underlying_price,
            expiration=expiration,
            dte=dte,
            calls=calls,
            puts=puts,
        )

    except Exception as e:
        logger.error("options_chain_fetch_failed", symbol=symbol, error=str(e))
        return None


def _filter_chain(df: pd.DataFrame, underlying_price: float) -> pd.DataFrame:
    """Filter option chain for liquid strikes near the money."""
    if df.empty:
        return df

    # Keep strikes within ±30% of underlying
    lo = underlying_price * 0.70
    hi = underlying_price * 1.30
    df = df[(df["strike"] >= lo) & (df["strike"] <= hi)].copy()

    # Liquidity filters
    # Use live bid quotes as the market-open signal:
    # When market is closed (weekends, holidays), market makers stop quoting → bid=0 everywhere.
    # Volume is NOT a reliable signal because yfinance returns historical (cumulative) volume
    # even on non-trading days.
    market_open = (
        "bid" in df.columns and df["bid"].fillna(0).sum() > 0
    )
    if market_open:
        # Live session: require OI, live bid, and daily volume
        if "openInterest" in df.columns:
            df = df[df["openInterest"] >= MIN_OPEN_INTEREST]
        if "bid" in df.columns:
            df = df[df["bid"] >= MIN_BID]
        if "volume" in df.columns:
            df = df[df["volume"].fillna(0) >= MIN_VOLUME]
    else:
        # Market closed (weekend/holiday): use lastPrice as viability proxy
        if "lastPrice" in df.columns:
            df = df[df["lastPrice"].fillna(0) >= MIN_BID]

    return df.reset_index(drop=True)


def get_iv_percentile(symbol: str, lookback_days: int = 252) -> dict | None:
    """Estimate IV percentile and IV rank using historical realized volatility.

    Returns dict with:
      percentile: % of 2y days where HV was lower than current (0-100)
      rank:       (current - 52w_min) / (52w_max - 52w_min) * 100
      current_hv: annualized 21-day realized vol (e.g. 0.35 = 35%)
      hv_52w_high: 52-week maximum HV
      hv_52w_low:  52-week minimum HV
    Uses realized vol as proxy for implied vol.
    """
    try:
        import numpy as np
        ticker = yf.Ticker(symbol)
        hist = paced_call(lambda: ticker.history(period="2y", interval="1d"), label=f"opt-hist:{symbol}")
        if hist.empty or len(hist) < 30:
            return None

        closes = hist["Close"].dropna()
        log_rets = np.log(closes / closes.shift(1)).dropna()
        rolling_vol = log_rets.rolling(21).std() * (252 ** 0.5)
        rolling_vol = rolling_vol.dropna()

        if len(rolling_vol) < 10:
            return None

        current_vol = float(rolling_vol.iloc[-1])
        # IV Percentile (2-year lookback)
        percentile = float((rolling_vol <= current_vol).mean() * 100)
        # IV Rank (52-week high/low)
        vol_52w = rolling_vol.tail(252)
        hv_high = float(vol_52w.max())
        hv_low = float(vol_52w.min())
        rank = ((current_vol - hv_low) / (hv_high - hv_low) * 100) if hv_high > hv_low else 50.0

        return {
            "percentile": round(percentile, 1),
            "rank": round(rank, 1),
            "current_hv": round(current_vol, 3),
            "hv_52w_high": round(hv_high, 3),
            "hv_52w_low": round(hv_low, 3),
        }

    except Exception as e:
        logger.error("iv_percentile_failed", symbol=symbol, error=str(e))
        return None


def get_current_option_price(
    symbol: str,
    option_type: str,   # "call" or "put"
    strike: float,
    expiration: str,    # YYYY-MM-DD
) -> float | None:
    """Fetch mid-price (bid+ask)/2 for a specific option contract."""
    try:
        ticker = yf.Ticker(symbol)
        chain = paced_call(lambda: ticker.option_chain(expiration), label=f"opt-chain:{symbol}")
        df = chain.calls if option_type == "call" else chain.puts
        row = df[df["strike"] == strike]
        if row.empty:
            return None
        bid = float(row["bid"].iloc[0])
        ask = float(row["ask"].iloc[0])
        if bid <= 0 and ask <= 0:
            return float(row["lastPrice"].iloc[0])
        return round((bid + ask) / 2, 2)
    except Exception as e:
        logger.error("option_price_fetch_failed", symbol=symbol, strike=strike, error=str(e))
        return None
