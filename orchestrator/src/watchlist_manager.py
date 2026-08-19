"""Dynamic watchlist manager: core symbols + screener results + LLM suggestions.

Each cycle the active symbol universe is:
  1. Core:        symbols defined in account config  (always present)
  2. Screener:    Yahoo Finance top gainers / most active / trending
  3. Suggestions: symbols the LLM asked to watch last cycle  (persisted per account)

The LLM sees all three groups with full market data and technicals, then
adds its own ``suggest_symbols`` to the next cycle's universe.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import structlog
import yfinance as yf

from .yf_throttle import paced_call

logger = structlog.get_logger()

# Screener sources to query (tried in order, failures silently skipped).
# day_gainers/day_losers removed 2026-07-09: they fed the universe with
# one-day-spike small caps (all realized momentum losses came from them) and
# their extreme moves made Pass 1 misread calm markets as HIGH_VOLATILITY.
_SCREENER_SOURCES = ("most_actives",)
MAX_SCREENER_PER_SOURCE = 8   # symbols per screener source
MAX_SUGGESTIONS = 12          # LLM suggestions to carry forward

# Quality gate for screener candidates (core/LLM/research symbols are exempt)
MIN_SCREENER_PRICE = 10.0
MIN_SCREENER_MARKET_CAP = 2e9

# Index / volatility symbols that can't be directly traded
_EXCLUDE_SYMBOLS = frozenset({"^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX", "^TNX",
                               "^FTSE", "^N225", "BTC-USD", "ETH-USD"})

# Valid US ticker: 1–5 uppercase letters, optional dash + letters (e.g. BRK-B)
_TICKER_RE = re.compile(r"^[A-Z]{1,5}(-[A-Z]{1,2})?$")


class WatchlistManager:
    """Per-account dynamic watchlist: core + screener + LLM suggestions."""

    def __init__(
        self,
        account_key: str,
        core: list[str],
        data_dir: str = "data",
    ):
        self.account_key = account_key
        self.core = [s.upper() for s in core]
        self._file = Path(data_dir) / f"suggestions_{account_key}.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_full_watchlist(
        self,
        max_screener_per_source: int = MAX_SCREENER_PER_SOURCE,
    ) -> list[str]:
        """Return deduplicated symbol list: core + screener + suggestions."""
        seen: set[str] = set()
        result: list[str] = []

        def _add(sym: str) -> None:
            s = sym.upper().strip()
            if s and s not in seen and _is_valid_ticker(s):
                seen.add(s)
                result.append(s)

        for s in self.core:
            _add(s)

        screener_syms = self._fetch_screener(max_screener_per_source)
        for s in screener_syms:
            _add(s)

        for s in self._load_suggestions():
            _add(s)

        # Research agent suggestions (today's brief)
        try:
            from .research_agent import ResearchAgent
            brief = ResearchAgent.load_today()
            if brief:
                research_syms = [s["symbol"] for s in brief.get("top_symbols", [])]
                logger.debug("watchlist_research_symbols", count=len(research_syms))
                for s in research_syms:
                    _add(s)
        except Exception as e:
            logger.warning("watchlist_research_load_failed", error=str(e))

        logger.info(
            "watchlist_built",
            account=self.account_key,
            core=len(self.core),
            screener=len(screener_syms),
            total=len(result),
        )
        return result

    def save_suggestions(self, symbols: list[str]) -> None:
        """Persist LLM-suggested tickers for the next cycle."""
        clean = [s.upper().strip() for s in symbols if _is_valid_ticker(s)]
        clean = clean[:MAX_SUGGESTIONS]
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            self._file.write_text(
                json.dumps({"suggestions": clean, "saved_at": time.time()}, indent=2)
            )
            logger.info("suggestions_saved", account=self.account_key, symbols=clean)
        except Exception as e:
            logger.warning("suggestions_save_failed", error=str(e))

    def load_suggestions(self) -> list[str]:
        """Public accessor — returns saved suggestions (e.g. for dashboard)."""
        return self._load_suggestions()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_screener(self, max_per_source: int) -> list[str]:
        """Query Yahoo Finance screeners and return symbol list."""
        symbols: list[str] = []

        for screen in _SCREENER_SOURCES:
            try:
                data = paced_call(lambda: yf.screen(screen, count=max_per_source * 3), label=f"screen:{screen}")
                quotes = data.get("quotes", []) if isinstance(data, dict) else []
                added = 0
                for q in quotes:
                    sym = (q.get("symbol") or "").upper().strip()
                    if not sym or sym in symbols or sym in _EXCLUDE_SYMBOLS:
                        continue
                    # Quality gate: no sub-$10 or micro-cap spike names.
                    # Missing market cap FAILS the gate — the old `mcap and`
                    # short-circuit let every cap-less quote bypass the $2B floor.
                    price = q.get("regularMarketPrice") or 0
                    mcap = q.get("marketCap") or 0
                    if price < MIN_SCREENER_PRICE or mcap < MIN_SCREENER_MARKET_CAP:
                        continue
                    symbols.append(sym)
                    added += 1
                    if added >= max_per_source:
                        break
                logger.debug("screener_ok", screen=screen, found=added)
            except Exception as e:
                # warning, not debug — a silently empty screener looks
                # identical to a normal run in the logs
                logger.warning("screener_failed", screen=screen, error=str(e))

        # Remove core symbols (they're already present at the front)
        return [s for s in symbols if s not in self.core]

    def _load_suggestions(self) -> list[str]:
        """Load persisted LLM suggestions from previous cycle."""
        try:
            if not self._file.exists():
                return []
            data = json.loads(self._file.read_text())
            return [s for s in data.get("suggestions", []) if _is_valid_ticker(s)]
        except Exception:
            return []


def _is_valid_ticker(sym: str) -> bool:
    """Return True for plausible tradeable US ticker symbols."""
    if not sym or not isinstance(sym, str):
        return False
    sym = sym.upper().strip()
    if sym in _EXCLUDE_SYMBOLS:
        return False
    return bool(_TICKER_RE.match(sym))
