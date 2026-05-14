"""Fundamental data enrichment: earnings history, analyst consensus, growth metrics.

Fetched separately from real-time quotes (slower, cached 4 hours).
Used to give the LLM context for stay/exit decisions on held positions
and for evaluating new opportunities from the screener / suggestion pool.

All data comes from yfinance (free, no API key required).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import structlog
import yfinance as yf

logger = structlog.get_logger()

# yfinance logs Yahoo's 404 quoteSummary responses at ERROR level for every ETF
# (SPY/QQQ/TQQQ/SOXL/XLE/...) — they have no fundamentals by design. We skip
# .info for known ETFs below, but also raise the floor on the yfinance logger so
# unexpected 404s from other tickers don't spam the cycle log either.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

_CACHE_TTL = 4 * 3600  # 4 hours — fundamentals change slowly
_cache: dict[str, tuple[float, "FundamentalSnapshot"]] = {}  # symbol → (ts, data)

# Known ETF symbols — no EPS / analyst data, so skip the yfinance .info call
# entirely (Yahoo returns 404 for fundamentals on these by design).
_ETF_SYMBOLS = frozenset((
    "SPY", "QQQ", "VTI", "VOO", "IWM", "GLD", "TLT", "VXX",
    "TQQQ", "SOXL", "SOXS", "UVXY", "SCHD", "VYM", "XLE",
    "XLK", "XLF", "XLV", "XLY", "XLI", "XLB", "XLP", "XLU",
    "XLC", "XLRE", "DIA", "EFA", "EEM", "AGG", "BND", "HYG",
    "LQD", "SLV", "USO", "XBI", "ARKK", "ARKG", "SQQQ", "SPXL",
    "SPXS", "TLT", "IEF", "SHY",
))


def _is_etf_symbol(symbol: str) -> bool:
    """Detect ETF symbols by membership in known list or common suffix."""
    return symbol in _ETF_SYMBOLS or symbol.endswith(("ETF", "ETF-USD"))


@dataclass
class EarningsQuarter:
    """One reported quarter: actual EPS vs estimate."""
    period: str          # e.g. "2024-Q3"
    actual_eps: float | None = None
    estimate_eps: float | None = None
    surprise_pct: float | None = None  # (actual - estimate) / |estimate| * 100
    beat: bool | None = None           # True = beat, False = miss


@dataclass
class FundamentalSnapshot:
    """Fundamental data for one symbol."""
    symbol: str

    # Analyst consensus
    rec_label: str = ""           # "STRONG_BUY" | "BUY" | "HOLD" | "SELL" | "STRONG_SELL"
    analyst_count: int = 0
    target_price: float | None = None   # mean analyst price target
    current_price: float | None = None  # needed for upside % calculation

    # Growth metrics (year-over-year)
    eps_growth_yoy: float | None = None      # e.g. 0.15 = +15%
    revenue_growth_yoy: float | None = None

    # EPS absolute values
    trailing_eps: float | None = None
    forward_eps: float | None = None

    # Recent earnings history (most recent first)
    earnings_history: list[EarningsQuarter] = field(default_factory=list)

    # Flags
    is_etf: bool = False
    fetch_error: str = ""

    # -------------------------------------------------------------------
    # Derived helpers
    # -------------------------------------------------------------------

    @property
    def last_quarter(self) -> EarningsQuarter | None:
        return self.earnings_history[0] if self.earnings_history else None

    @property
    def upside_pct(self) -> float | None:
        """Analyst price target upside vs current price."""
        if self.target_price and self.current_price and self.current_price > 0:
            return (self.target_price - self.current_price) / self.current_price * 100
        return None

    def to_prompt_line(self) -> str:
        """Single-line summary for LLM prompt."""
        parts: list[str] = []

        if self.rec_label:
            count = f"({self.analyst_count})" if self.analyst_count else ""
            parts.append(f"analyst={self.rec_label}{count}")

        if self.target_price:
            upside = self.upside_pct
            upside_str = f" [{upside:+.0f}% upside]" if upside is not None else ""
            parts.append(f"target=${self.target_price:.0f}{upside_str}")

        if self.eps_growth_yoy is not None:
            parts.append(f"EPS_growth={self.eps_growth_yoy*100:+.0f}%YoY")

        if self.revenue_growth_yoy is not None:
            parts.append(f"rev_growth={self.revenue_growth_yoy*100:+.0f}%YoY")

        lq = self.last_quarter
        if lq and lq.beat is not None:
            beat_str = "BEAT" if lq.beat else "MISS"
            surp = f" {lq.surprise_pct:+.1f}%" if lq.surprise_pct is not None else ""
            parts.append(f"last_Q={beat_str}{surp}")

        if self.forward_eps and self.trailing_eps and self.trailing_eps > 0:
            pe_fwd_ratio = (self.forward_eps / self.trailing_eps - 1) * 100
            parts.append(f"EPS_fwd_vs_trail={pe_fwd_ratio:+.0f}%")

        return " | ".join(parts) if parts else "(no fundamental data)"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_fundamental(symbol: str) -> FundamentalSnapshot:
    """Fetch fundamental snapshot for one symbol (cached 4h)."""
    symbol = symbol.upper().strip()
    cached = _cache.get(symbol)
    if cached and (time.time() - cached[0]) < _CACHE_TTL:
        return cached[1]

    snap = _fetch(symbol)
    _cache[symbol] = (time.time(), snap)
    return snap


def get_fundamentals_batch(
    symbols: list[str],
    priority_symbols: list[str] | None = None,
    max_symbols: int = 20,
) -> dict[str, FundamentalSnapshot]:
    """Fetch fundamentals for a list of symbols.

    Prioritises ``priority_symbols`` (e.g. currently held positions) so that
    held stocks always get full analysis even when the total universe is large.

    Args:
        symbols: Full list of symbols to fetch.
        priority_symbols: Symbols that MUST be fetched (held positions).
        max_symbols: Cap on total symbols to avoid slow fetches for large universes.

    Returns:
        Dict mapping symbol → FundamentalSnapshot.
    """
    priority = [s.upper() for s in (priority_symbols or [])]
    rest = [s.upper() for s in symbols if s.upper() not in priority]

    # Priority symbols first, then fill up to max_symbols with the rest
    fetch_list = priority + rest
    fetch_list = list(dict.fromkeys(fetch_list))[:max_symbols]  # dedup + cap

    result: dict[str, FundamentalSnapshot] = {}
    for sym in fetch_list:
        try:
            result[sym] = get_fundamental(sym)
        except Exception as e:
            logger.warning("fundamental_fetch_failed", symbol=sym, error=str(e))
            result[sym] = FundamentalSnapshot(symbol=sym, fetch_error=str(e))

    logger.info("fundamentals_fetched", count=len(result),
                priority=len(priority), total_requested=len(symbols))
    return result


def format_fundamentals_for_prompt(
    fundamentals: dict[str, FundamentalSnapshot],
    held_symbols: set[str] | None = None,
) -> str:
    """Format fundamentals as a prompt section.

    Held positions are highlighted and listed first.
    """
    if not fundamentals:
        return ""

    held = held_symbols or set()
    lines: list[str] = ["== FUNDAMENTAL SIGNALS =="]

    # Held positions first (LLM needs these for stay/exit decisions)
    held_snaps = {s: f for s, f in fundamentals.items() if s in held and not f.is_etf}
    other_snaps = {s: f for s, f in fundamentals.items()
                   if s not in held and not f.is_etf and not f.fetch_error}

    if held_snaps:
        lines.append("  [HELD POSITIONS]")
        for sym, snap in sorted(held_snaps.items()):
            lines.append(f"  {sym}: {snap.to_prompt_line()}")

    if other_snaps:
        if held_snaps:
            lines.append("  [WATCHLIST / OPPORTUNITIES]")
        for sym, snap in sorted(other_snaps.items()):
            lines.append(f"  {sym}: {snap.to_prompt_line()}")

    if len(lines) == 1:
        return ""  # nothing useful to show

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal fetch logic
# ---------------------------------------------------------------------------

def _fetch(symbol: str) -> FundamentalSnapshot:
    """Fetch data from yfinance and build FundamentalSnapshot."""
    is_etf = _is_etf_symbol(symbol)
    snap = FundamentalSnapshot(symbol=symbol, is_etf=is_etf)

    # ETFs have no EPS / analyst / earnings data — Yahoo returns 404 on
    # quoteSummary fundamentals for every one of them. The downstream prompt
    # builder filters ETFs out (see format_fundamentals_for_prompt), so a bare
    # snapshot with is_etf=True is all we need.
    if is_etf:
        return snap

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        snap.current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )

        # Analyst consensus
        rec_mean = info.get("recommendationMean")
        snap.rec_label = _rec_mean_to_label(rec_mean)
        snap.analyst_count = info.get("numberOfAnalystOpinions") or 0
        snap.target_price = info.get("targetMeanPrice")

        # Growth metrics
        snap.eps_growth_yoy = info.get("earningsGrowth")
        snap.revenue_growth_yoy = info.get("revenueGrowth")

        # EPS
        snap.trailing_eps = info.get("trailingEps")
        snap.forward_eps = info.get("forwardEps")

        # Quarterly earnings beat/miss
        snap.earnings_history = _fetch_earnings_history(ticker)

    except Exception as e:
        snap.fetch_error = str(e)
        logger.debug("fundamental_info_failed", symbol=symbol, error=str(e))

    return snap


def _fetch_earnings_history(ticker: yf.Ticker) -> list[EarningsQuarter]:
    """Extract last 2-4 quarterly earnings beat/miss records."""
    quarters: list[EarningsQuarter] = []
    try:
        qe = ticker.quarterly_earnings  # DataFrame: index=date, cols=[Actual, Estimate]
        if qe is None or qe.empty:
            return quarters
        # Newest first
        for period, row in qe.iloc[:4].iterrows():
            actual = row.get("Actual") if hasattr(row, "get") else row["Actual"]
            estimate = row.get("Estimate") if hasattr(row, "get") else row["Estimate"]
            try:
                actual = float(actual)
                estimate = float(estimate)
            except (TypeError, ValueError):
                quarters.append(EarningsQuarter(period=str(period)))
                continue

            surprise_pct = None
            if estimate and estimate != 0:
                surprise_pct = (actual - estimate) / abs(estimate) * 100

            quarters.append(EarningsQuarter(
                period=str(period),
                actual_eps=actual,
                estimate_eps=estimate,
                surprise_pct=surprise_pct,
                beat=actual >= estimate,
            ))
    except Exception as e:
        logger.debug("earnings_history_failed", error=str(e))
    return quarters


def _rec_mean_to_label(rec_mean: float | None) -> str:
    """Convert yfinance recommendationMean (1–5) to human label.

    Yahoo Finance scale:
      1.0–1.5 = Strong Buy
      1.5–2.5 = Buy
      2.5–3.5 = Hold
      3.5–4.5 = Underperform / Sell
      4.5–5.0 = Strong Sell
    """
    if rec_mean is None:
        return ""
    if rec_mean <= 1.5:
        return "STRONG_BUY"
    if rec_mean <= 2.5:
        return "BUY"
    if rec_mean <= 3.5:
        return "HOLD"
    if rec_mean <= 4.5:
        return "SELL"
    return "STRONG_SELL"
