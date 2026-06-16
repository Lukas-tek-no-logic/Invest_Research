"""Market data fetcher using yfinance with caching."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog
import yfinance as yf
import pandas as pd

from .fundamental_data import _is_etf_symbol
from .yf_throttle import paced_call

logger = structlog.get_logger()

QUOTE_CACHE_TTL = 60  # seconds
INFO_CACHE_TTL = 3600  # seconds


@dataclass
class StockQuote:
    symbol: str
    price: float
    change_pct: float
    volume: int
    avg_volume_10d: int
    market_cap: float
    pe_ratio: float | None
    forward_pe: float | None
    pb_ratio: float | None
    dividend_yield: float | None
    week52_high: float
    week52_low: float
    sector: str
    industry: str
    name: str
    short_pct_float: float | None = None  # short interest as % of float


@dataclass
class _CacheEntry:
    data: object
    timestamp: float


class MarketDataProvider:
    """Fetches stock quotes and history from Yahoo Finance with caching."""

    def __init__(self, quote_ttl: int = QUOTE_CACHE_TTL, info_ttl: int = INFO_CACHE_TTL):
        self._quote_cache: dict[str, _CacheEntry] = {}
        self._info_cache: dict[str, _CacheEntry] = {}
        self._quote_ttl = quote_ttl
        self._info_ttl = info_ttl

    def _get_cached(self, cache: dict, key: str, ttl: int):
        entry = cache.get(key)
        if entry and (time.time() - entry.timestamp) < ttl:
            return entry.data
        return None

    def get_quote(self, symbol: str) -> StockQuote:
        """Get current quote with fundamentals for a symbol.

        For known ETFs we skip `ticker.info` (Yahoo's quoteSummary endpoint
        returns 404 for every ETF — wasted round-trip and stack noise) and
        build a minimal StockQuote from `fast_info` plus a brief history
        fallback for change% / 52w range.
        """
        cached = self._get_cached(self._quote_cache, symbol, self._quote_ttl)
        if cached:
            return cached

        fast = paced_call(lambda: dict(yf.Ticker(symbol).fast_info), label=f"fast:{symbol}")

        if _is_etf_symbol(symbol):
            quote = StockQuote(
                symbol=symbol,
                price=fast.get("lastPrice", 0) or 0,
                change_pct=0,  # not exposed via fast_info reliably
                volume=fast.get("lastVolume", 0) or 0,
                avg_volume_10d=fast.get("tenDayAverageVolume", 0) or 0,
                market_cap=fast.get("marketCap", 0) or 0,
                pe_ratio=None,
                forward_pe=None,
                pb_ratio=None,
                dividend_yield=None,
                week52_high=fast.get("yearHigh", 0) or 0,
                week52_low=fast.get("yearLow", 0) or 0,
                sector="ETF",
                industry="ETF",
                name=symbol,
                short_pct_float=None,
            )
        else:
            info = paced_call(lambda: yf.Ticker(symbol).info, label=f"info:{symbol}")
            quote = StockQuote(
                symbol=symbol,
                price=fast.get("lastPrice", 0) or info.get("currentPrice", 0) or info.get("regularMarketPrice", 0),
                change_pct=info.get("regularMarketChangePercent", 0) or 0,
                volume=info.get("regularMarketVolume", 0) or 0,
                avg_volume_10d=info.get("averageDailyVolume10Day", 0) or 0,
                market_cap=info.get("marketCap", 0) or 0,
                pe_ratio=info.get("trailingPE"),
                forward_pe=info.get("forwardPE"),
                pb_ratio=info.get("priceToBook"),
                dividend_yield=info.get("dividendYield"),
                week52_high=info.get("fiftyTwoWeekHigh", 0) or 0,
                week52_low=info.get("fiftyTwoWeekLow", 0) or 0,
                sector=info.get("sector", "Unknown"),
                industry=info.get("industry", "Unknown"),
                name=info.get("shortName", symbol),
                short_pct_float=info.get("shortPercentOfFloat"),
            )

        self._quote_cache[symbol] = _CacheEntry(data=quote, timestamp=time.time())
        logger.debug("market_data_quote_fetched", symbol=symbol, price=quote.price)
        return quote

    def get_quotes_batch(self, symbols: list[str]) -> dict[str, StockQuote]:
        """Get quotes for multiple symbols."""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_quote(symbol)
            except Exception as e:
                logger.warning("market_data_quote_failed", symbol=symbol, error=str(e))
        return results

    def get_history(
        self,
        symbol: str,
        period: str = "6mo",
        interval: str = "1d",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Get historical OHLCV data."""
        if start and end:
            df = paced_call(
                lambda: yf.Ticker(symbol).history(start=start, end=end, interval=interval),
                label=f"hist:{symbol}",
            )
        else:
            df = paced_call(
                lambda: yf.Ticker(symbol).history(period=period, interval=interval),
                label=f"hist:{symbol}",
            )
        if df.empty:
            logger.warning("market_data_empty_history", symbol=symbol, period=period)
        return df

    def get_current_price(self, symbol: str) -> float:
        """Get just the current price (lightweight).

        For ETFs we never fall back to `ticker.info` — that endpoint returns
        404 for every ETF on Yahoo and only wastes a round-trip. If fast_info
        fails for an ETF the caller will get 0 and can handle it.
        """
        try:
            fast = paced_call(lambda: dict(yf.Ticker(symbol).fast_info), label=f"fast:{symbol}")
            price = fast.get("lastPrice", 0) or 0
            if price:
                return price
        except Exception:
            pass
        if _is_etf_symbol(symbol):
            return 0
        info = paced_call(lambda: yf.Ticker(symbol).info, label=f"info:{symbol}")
        return info.get("currentPrice", 0) or info.get("regularMarketPrice", 0) or 0

    def get_upcoming_earnings(self, symbols: list[str], days: int = 14) -> dict[str, str]:
        """Get upcoming earnings dates within the next N days."""
        from datetime import date, timedelta
        cutoff = date.today() + timedelta(days=days)
        result = {}
        for symbol in symbols:
            if _is_etf_symbol(symbol):
                continue  # ETFs don't have earnings
            try:
                cal = paced_call(lambda: yf.Ticker(symbol).calendar, label=f"cal:{symbol}")
                if not isinstance(cal, dict):
                    continue
                dates_raw = cal.get("Earnings Date", [])
                if not isinstance(dates_raw, list):
                    dates_raw = [dates_raw]
                for ed in dates_raw:
                    try:
                        ed_date = ed.date() if hasattr(ed, "date") else ed
                        if isinstance(ed_date, date) and date.today() <= ed_date <= cutoff:
                            days_away = (ed_date - date.today()).days
                            result[symbol] = f"{ed_date} (in {days_away} days)"
                            break
                    except Exception:
                        continue
            except Exception:
                pass
        return result

    def format_upcoming_earnings(self, symbols: list[str], days: int = 14) -> str:
        """Format upcoming earnings for prompt injection."""
        earnings = self.get_upcoming_earnings(symbols, days)
        if not earnings:
            return ""
        lines = [f"== UPCOMING EARNINGS (next {days} days) =="]
        for sym, info in sorted(earnings.items()):
            lines.append(f"  {sym}: {info}")
        return "\n".join(lines)

    def validate_symbol(self, symbol: str) -> bool:
        """Check if a symbol is valid and tradeable."""
        try:
            if _is_etf_symbol(symbol):
                fast = paced_call(lambda: dict(yf.Ticker(symbol).fast_info), label=f"fast:{symbol}")
                return bool(fast.get("lastPrice", 0))
            info = paced_call(lambda: yf.Ticker(symbol).info, label=f"info:{symbol}")
            return bool(info.get("regularMarketPrice") or info.get("currentPrice"))
        except Exception:
            return False

    def get_market_overview(self) -> dict:
        """Get broad market indicators (S&P 500, VIX, 10Y yield, sector ETFs)."""
        benchmarks = {
            "SPY": "S&P 500 ETF",
            "QQQ": "Nasdaq 100 ETF",
            "^VIX": "VIX Volatility",
            "^TNX": "10Y Treasury Yield",
            "XLK": "Tech",
            "XLF": "Financials",
            "XLE": "Energy",
            "XLV": "Healthcare",
            "XLI": "Industrials",
            "XLY": "Consumer Discr.",
        }
        overview = {}
        for sym, label in benchmarks.items():
            try:
                fast = paced_call(lambda: dict(yf.Ticker(sym).fast_info), label=f"fast:{sym}")
                overview[sym] = {
                    "label": label,
                    "price": fast.get("lastPrice", 0),
                    "change_pct": fast.get("regularMarketChangePercent", 0),
                }
            except Exception as e:
                logger.warning("market_overview_failed", symbol=sym, error=str(e))
        return overview
