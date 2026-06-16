"""Daily market research agent.

Runs once a day (14:00 CET), gathers broad financial + geopolitical data,
sends to LLM for synthesis, saves result to data/daily_research.json.
All trading accounts read this file at the start of their own cycle.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import structlog

logger = structlog.get_logger()

_OUTPUT_FILE = Path("data/daily_research.json")

# yfinance screeners to run (beyond the 3 used in WatchlistManager)
RESEARCH_SCREENERS = [
    "day_gainers",
    "day_losers",
    "most_actives",
    "growth_technology_stocks",
    "undervalued_large_caps",
    "most_shorted_stocks",
]

RESEARCH_SYSTEM_PROMPT = """You are a senior market research analyst preparing a
pre-market intelligence brief for a team of autonomous investment agents.

Your job:
1. Read ALL provided news articles (financial + geopolitical) and market data.
2. Identify 10-15 stocks/ETFs most worth watching today, with clear investment thesis.
3. Assess macro regime and key themes driving markets.
4. Flag geopolitical risks that could move specific sectors.
5. Note any important events (earnings, Fed speakers, economic data releases).

Be concrete and data-driven. Every symbol recommendation must have a specific
catalyst or thesis, not just "momentum" or "looks interesting".

Respond with valid JSON only, no markdown:
{
  "date": "YYYY-MM-DD",
  "market_regime": "BULL_TREND|BEAR_TREND|SIDEWAYS|HIGH_VOLATILITY",
  "key_themes": ["theme1", "theme2"],
  "top_symbols": [
    {
      "symbol": "NVDA",
      "thesis": "Blackwell demand ahead of GTC conference; analysts raised targets",
      "catalyst": "GTC 2026 keynote March 18",
      "sector": "Technology",
      "conviction": "HIGH|MEDIUM|LOW",
      "direction": "BULLISH|BEARISH|NEUTRAL"
    }
  ],
  "sectors": [
    {"name": "Energy", "bias": "OVERWEIGHT|NEUTRAL|UNDERWEIGHT", "reason": "..."}
  ],
  "geopolitical_risks": [
    {
      "event": "Russia-Ukraine energy supply disruption",
      "affected_sectors": ["Energy", "Materials"],
      "market_impact": "Bullish natgas, XOM, CVX; bearish EUR/USD"
    }
  ],
  "macro_events_today": "Use ONLY events from the provided economic calendar. Empty string if none.",
  "avoid_today": ["any sectors or symbols to avoid and why"]
}"""


class ResearchAgent:
    def __init__(self, llm, news, market_data, config: dict):
        self.llm = llm
        self.news = news
        self.market_data = market_data
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Gather data, call LLM, save and return research brief."""
        research_config = self.config.get("accounts", {}).get("research", {})
        model = research_config.get("model", "Nemotron")
        fallback = research_config.get("fallback_model", "Qwen3-Next")
        max_news = research_config.get("max_news_articles", 40)
        max_article_chars = research_config.get("max_article_chars", 3000)
        max_screener = research_config.get("max_screener_per_source", 15)

        logger.info("research_agent_start", model=model)

        news_text = self._gather_news(max_news, max_article_chars)
        screener_text = self._gather_screeners(max_screener)
        market_text = self._gather_market_overview()
        calendar_text = self._gather_economic_calendar()

        from datetime import datetime
        today = datetime.now().strftime("%A %Y-%m-%d %H:%M CET")
        user_content = "\n\n".join([
            f"== TODAY: {today} ==",
            calendar_text,
            market_text,
            screener_text,
            news_text,
            "Analyze all the above and respond with the research brief JSON."
            " Use ONLY the economic calendar events provided above for macro_events_today.",
        ])

        messages = [
            {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        result = self.llm.chat_json(
            messages=messages,
            model=model,
            fallback_model=fallback,
            temperature=0.6,
            max_tokens=8192,
        )

        result["date"] = date.today().isoformat()

        self._save(result)
        logger.info(
            "research_agent_complete",
            themes=result.get("key_themes", []),
            symbols=[s["symbol"] for s in result.get("top_symbols", [])],
        )
        return result

    @staticmethod
    def load_today() -> dict | None:
        """Load today's research brief if it exists. Returns None if stale or missing."""
        if not _OUTPUT_FILE.exists():
            return None
        try:
            with open(_OUTPUT_FILE) as f:
                data = json.load(f)
            if data.get("date") == date.today().isoformat():
                return data
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _gather_news(self, max_items: int, max_article_chars: int) -> str:
        """Fetch full articles from financial + geopolitical RSS feeds."""
        items = self.news.fetch_news_with_articles(
            max_items=max_items,
            max_article_chars=max_article_chars,
        )
        lines = [f"== NEWS ({len(items)} articles) =="]
        for i, item in enumerate(items, 1):
            lines.append(f"\n--- [{i}] {item.source.upper()} | {item.published} ---")
            lines.append(f"TITLE: {item.title}")
            if item.summary:
                lines.append(item.summary)
        return "\n".join(lines)

    def _gather_screeners(self, max_per_source: int) -> str:
        """Run multiple yfinance screeners and return formatted results."""
        import yfinance as yf
        from .yf_throttle import paced_call
        lines = ["== MARKET SCREENERS =="]

        for screener_name in RESEARCH_SCREENERS:
            try:
                result = paced_call(lambda: yf.screen(screener_name), label=f"screen:{screener_name}")
                quotes = result.get("quotes", [])[:max_per_source]
                symbols = [q.get("symbol", "") for q in quotes if q.get("symbol")]
                lines.append(f"\n{screener_name.upper()} ({len(symbols)} symbols):")
                for q in quotes:
                    sym = q.get("symbol", "")
                    price = q.get("regularMarketPrice", "?")
                    chg = q.get("regularMarketChangePercent", 0)
                    name = q.get("shortName", "")
                    lines.append(f"  {sym} ({name}): ${price} {chg:+.2f}%")
            except Exception as e:
                logger.warning("screener_failed", name=screener_name, error=str(e))

        return "\n".join(lines)

    def _gather_market_overview(self) -> str:
        """Fetch indices, sectors, VIX, TNX."""
        try:
            overview = self.market_data.get_market_overview()
            lines = ["== MARKET OVERVIEW =="]
            for sym, data in overview.items():
                price = data.get("price", "?")
                chg = data.get("change_pct", 0)
                label = data.get("label", "")
                label_str = f" [{label}]" if label else ""
                lines.append(f"{sym}{label_str}: {price} ({chg:+.2f}%)")
            return "\n".join(lines)
        except Exception as e:
            logger.warning("market_overview_failed", error=str(e))
            return "== MARKET OVERVIEW ==\n(unavailable)"

    def _gather_economic_calendar(self) -> str:
        """Fetch today's USD economic events from ForexFactory public XML feed."""
        import requests
        import xml.etree.ElementTree as ET
        from datetime import datetime

        try:
            resp = requests.get(
                "https://nfs.faireconomy.media/ff_calendar_thisweek.xml",
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.content)

            today_str = datetime.now().strftime("%m-%d-%Y")
            events = []
            for ev in root.findall("event"):
                country = ev.findtext("country", "")
                impact = ev.findtext("impact", "")
                date_str = ev.findtext("date", "")
                time_str = ev.findtext("time", "")
                title = ev.findtext("title", "")
                forecast = ev.findtext("forecast", "")
                previous = ev.findtext("previous", "")

                if country != "USD":
                    continue
                if impact not in ("High", "Medium"):
                    continue
                if today_str not in date_str:
                    continue

                parts = [f"{time_str} ET: {title} [{impact}]"]
                if forecast:
                    parts.append(f"forecast={forecast}")
                if previous:
                    parts.append(f"prev={previous}")
                events.append(" | ".join(parts))

            if events:
                return "== ECONOMIC CALENDAR TODAY (USD, High/Medium impact) ==\n" + "\n".join(events)
            return "== ECONOMIC CALENDAR TODAY ==\nNo major USD events scheduled."
        except Exception as e:
            logger.warning("economic_calendar_failed", error=str(e))
            return "== ECONOMIC CALENDAR TODAY ==\n(unavailable)"

    @staticmethod
    def _save(data: dict) -> None:
        _OUTPUT_FILE.parent.mkdir(exist_ok=True)
        with open(_OUTPUT_FILE, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("research_saved", path=str(_OUTPUT_FILE))
