"""Main backtest runner: orchestrates the historical simulation loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

import structlog

from ..decision_parser import parse_analysis, parse_decision
from ..llm_client import LLMClient
from ..market_data import StockQuote
from ..prompt_builder import build_pass1_messages, build_pass2_messages
from ..risk_manager import RiskManager
from ..technical_indicators import compute_indicators
from .historical_data import get_history_up_to, get_quotes_at_date, prefetch_history
from .portfolio_sim import SimTrade, SimulatedPortfolio

logger = structlog.get_logger()


@dataclass
class BacktestResult:
    """Full result of a completed backtest simulation."""
    snapshots: list[dict]          # [{date, total_value, cash, pl_pct}] — one per week
    trades: list[SimTrade]
    decisions: list[dict]          # [{week_num, date, outlook, confidence, actions, …}]
    final_value: float
    total_return_pct: float
    max_drawdown_pct: float
    win_rate_pct: float            # % of SELL trades executed above avg_cost
    benchmark_return_pct: float    # SPY buy-and-hold return over same period
    error: str = ""


def run_backtest(
    account_config: dict,
    start_date: str,
    end_date: str,
    llm_client: LLMClient,
    initial_cash: float = 10_000,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> BacktestResult:
    """Run a full historical simulation using the existing LLM pipeline.

    The LLM receives market data and technical signals WITHOUT dates so as to
    reduce (but not fully eliminate) temporal bias.  News is omitted since
    historical news data is not available.

    Args:
        account_config: Full account configuration dict (strategy, watchlist,
            risk_profile, model, etc.).
        start_date: Simulation start date (YYYY-MM-DD).
        end_date: Simulation end date (YYYY-MM-DD).
        llm_client: Instantiated LLMClient.
        initial_cash: Starting capital in USD.
        on_progress: Optional callback(week_num, total_weeks, current_date).

    Returns:
        BacktestResult with performance metrics, trade log, and per-week snapshots.
    """
    watchlist: list[str] = account_config.get("watchlist", [])
    risk_profile: dict = account_config.get("risk_profile", {})
    model: str = account_config.get("model", "Qwen3-Next")
    fallback_model: str | None = account_config.get("fallback_model")

    if not watchlist:
        return BacktestResult(snapshots=[], trades=[], decisions=[],
                              final_value=initial_cash, total_return_pct=0,
                              max_drawdown_pct=0, win_rate_pct=0,
                              benchmark_return_pct=0,
                              error="Empty watchlist in account config")

    # Ensure SPY is fetched for benchmark even if not in watchlist
    symbols_to_fetch = list(set(watchlist + ["SPY"]))

    # Prefetch with extra lookback for indicator warmup (200 trading days ≈ 280 calendar days)
    prefetch_start = (
        datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=300)
    ).strftime("%Y-%m-%d")

    logger.info("backtest_prefetch_start", symbols=len(symbols_to_fetch),
                start=prefetch_start, end=end_date)
    all_history = prefetch_history(symbols_to_fetch, prefetch_start, end_date)

    sim_dates = _get_weekly_dates(start_date, end_date)
    if not sim_dates:
        return BacktestResult(snapshots=[], trades=[], decisions=[],
                              final_value=initial_cash, total_return_pct=0,
                              max_drawdown_pct=0, win_rate_pct=0,
                              benchmark_return_pct=0,
                              error="No trading dates found in the specified range")

    total_weeks = len(sim_dates)
    logger.info("backtest_loop_start", weeks=total_weeks,
                start=sim_dates[0], end=sim_dates[-1])

    # Benchmark: SPY buy-and-hold over the same period
    spy_history = all_history.get("SPY")
    spy_at_start = get_quotes_at_date("SPY", sim_dates[0], spy_history).get("price", 0) if spy_history is not None else 0
    spy_at_end = get_quotes_at_date("SPY", sim_dates[-1], spy_history).get("price", 0) if spy_history is not None else 0
    benchmark_return = (
        (spy_at_end - spy_at_start) / spy_at_start * 100
        if spy_at_start > 0 else 0.0
    )

    sim_portfolio = SimulatedPortfolio(
        initial_cash,
        broker_cost_model=account_config.get("broker_cost_model", ""),
    )
    snapshots: list[dict] = []
    trades: list[SimTrade] = []
    decisions: list[dict] = []

    for week_num, sim_date in enumerate(sim_dates, 1):
        if on_progress:
            try:
                on_progress(week_num, total_weeks, sim_date)
            except Exception:
                pass

        logger.info("backtest_week", week=week_num, date=sim_date)

        # --- Build quotes dict (prompt-compatible dicts) ---
        quotes_dicts: dict[str, dict] = {}
        for sym in watchlist:
            hist = all_history.get(sym)
            if hist is not None:
                quotes_dicts[sym] = get_quotes_at_date(sym, sim_date, hist)

        # --- Build StockQuote objects for RiskManager ---
        stock_quotes: dict[str, StockQuote] = {
            sym: _dict_to_stock_quote(sym, d) for sym, d in quotes_dicts.items()
        }

        # --- Technical indicators (no future leakage) ---
        tech_signals = {}
        for sym in watchlist:
            hist = all_history.get(sym)
            if hist is not None:
                df_slice = get_history_up_to(sym, sim_date, hist, lookback_days=200)
                if not df_slice.empty:
                    tech_signals[sym] = compute_indicators(df_slice, sym)

        # --- Portfolio state ---
        current_prices = {sym: d.get("price", 0.0) for sym, d in quotes_dicts.items()}
        portfolio_state = sim_portfolio.to_portfolio_state(sim_date, "Backtest", current_prices)

        # --- Anonymised decision history (Week N labels, not real dates) ---
        anon_history = _format_anon_history(decisions)

        # --- LLM Pass 1: Market analysis ---
        analysis_raw: dict = {}
        try:
            pass1_msgs = build_pass1_messages(
                portfolio_state,
                quotes_dicts,
                tech_signals,
                news_text="",          # No historical news available
                decision_history=anon_history,
                strategy_config=account_config,
            )
            analysis_raw = llm_client.chat_json(
                pass1_msgs, model=model, fallback_model=fallback_model, temperature=0.7
            )
            analysis = parse_analysis(analysis_raw)
        except Exception as e:
            logger.error("backtest_pass1_failed", week=week_num, date=sim_date, error=str(e))
            # Record snapshot and continue to next week
            snapshots.append(sim_portfolio.snapshot(sim_date, current_prices))
            decisions.append(_empty_decision_entry(week_num, sim_date, str(e)))
            continue

        # --- LLM Pass 2: Trade decision ---
        try:
            pass2_msgs = build_pass2_messages(
                analysis_raw,
                portfolio_state,
                account_config,
                risk_profile,
            )
            decision_raw = llm_client.chat_json(
                pass2_msgs, model=model, fallback_model=fallback_model, temperature=0.5
            )
            decision = parse_decision(decision_raw)
        except Exception as e:
            logger.error("backtest_pass2_failed", week=week_num, date=sim_date, error=str(e))
            snapshots.append(sim_portfolio.snapshot(sim_date, current_prices))
            decisions.append(_empty_decision_entry(week_num, sim_date, str(e)))
            continue

        # --- Risk validation with simulated date ---
        risk_mgr = RiskManager(risk_profile, sim_date=sim_date)
        risk_result = risk_mgr.validate(decision, portfolio_state, stock_quotes)

        # --- Execute approved + forced actions ---
        week_trades: list[SimTrade] = []
        for action in risk_result.approved_actions + risk_result.forced_actions:
            sq = stock_quotes.get(action.symbol)
            if sq is None or sq.price <= 0:
                continue
            if action.type == "BUY":
                trade = sim_portfolio.buy(action.symbol, action.amount_usd, sq.price, sim_date)
            else:
                trade = sim_portfolio.sell(action.symbol, action.amount_usd, sq.price, sim_date)
            if trade.success:
                week_trades.append(trade)
                trades.append(trade)

        # --- Snapshot (prices updated after trades) ---
        updated_prices = {sym: sq.price for sym, sq in stock_quotes.items()}
        snapshots.append(sim_portfolio.snapshot(sim_date, updated_prices))

        # --- Decision log entry ---
        decisions.append({
            "week_num": week_num,
            "date": sim_date,
            "outlook": decision.portfolio_outlook,
            "confidence": decision.confidence,
            "market_regime": analysis.market_regime,
            "actions": [
                {
                    "type": a.type,
                    "symbol": a.symbol,
                    "amount_usd": a.amount_usd,
                    "thesis": a.thesis,
                }
                for a in (risk_result.approved_actions + risk_result.forced_actions)
            ],
            "risk_mods": risk_result.modifications,
            "trades": [
                {
                    "symbol": t.symbol,
                    "type": t.type,
                    "quantity": round(t.quantity, 4),
                    "price": round(t.price, 2),
                    "total": round(t.total, 2),
                }
                for t in week_trades
            ],
        })

    # --- Final metrics ---
    last_prices = {sym: stock_quotes.get(sym, _dummy_quote(sym)).price
                   for sym in watchlist}
    final_value = sim_portfolio.get_total_value(last_prices)
    total_return = ((final_value - initial_cash) / initial_cash * 100) if initial_cash > 0 else 0.0

    return BacktestResult(
        snapshots=snapshots,
        trades=trades,
        decisions=decisions,
        final_value=round(final_value, 2),
        total_return_pct=round(total_return, 2),
        max_drawdown_pct=round(_calc_max_drawdown(snapshots), 2),
        win_rate_pct=round(_calc_win_rate(trades), 1),
        benchmark_return_pct=round(benchmark_return, 2),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_weekly_dates(start_date: str, end_date: str) -> list[str]:
    """Generate weekly simulation dates (Fridays) between start and end."""
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    # Advance to the first Friday on or after start
    days_ahead = (4 - start.weekday()) % 7  # Friday = weekday 4
    current = start + timedelta(days=days_ahead)

    dates: list[str] = []
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(weeks=1)
    return dates


def _dict_to_stock_quote(symbol: str, d: dict) -> StockQuote:
    """Convert a historical quote dict to a StockQuote dataclass."""
    return StockQuote(
        symbol=symbol,
        price=d.get("price", 0.0),
        change_pct=d.get("change_pct", 0.0),
        volume=d.get("volume", 0),
        avg_volume_10d=d.get("avg_volume_10d", 0),
        market_cap=d.get("market_cap", 0),
        pe_ratio=d.get("pe_ratio"),
        forward_pe=d.get("forward_pe"),
        pb_ratio=d.get("pb_ratio"),
        dividend_yield=d.get("dividend_yield"),
        week52_high=d.get("week52_high", 0.0),
        week52_low=d.get("week52_low", 0.0),
        sector=d.get("sector", "Unknown"),
        industry=d.get("industry", "Unknown"),
        name=d.get("name", symbol),
    )


def _dummy_quote(symbol: str) -> StockQuote:
    return StockQuote(symbol=symbol, price=0, change_pct=0, volume=0,
                      avg_volume_10d=0, market_cap=0, pe_ratio=None,
                      forward_pe=None, pb_ratio=None, dividend_yield=None,
                      week52_high=0, week52_low=0, sector="Unknown",
                      industry="Unknown", name=symbol)


def _format_anon_history(decisions: list[dict]) -> str:
    """Format past decisions with relative 'Week N' labels instead of real dates.

    This is the anonymisation step: the LLM sees week numbers rather than
    calendar dates, reducing (though not eliminating) temporal bias.
    """
    if not decisions:
        return "== YOUR PREVIOUS DECISIONS ==\n(No previous decisions — this is your first cycle)"

    lines = ["== YOUR PREVIOUS DECISIONS (last cycles) =="]
    for entry in decisions[-4:]:
        week_label = f"Week {entry['week_num']}"
        outlook = entry.get("outlook", "Unknown")
        confidence = entry.get("confidence", "N/A")
        lines.append(f"\n[{week_label}] Outlook: {outlook}, Confidence: {confidence}")

        actions = entry.get("actions", [])
        if actions:
            for a in actions:
                lines.append(
                    f"  {a.get('type', '?')} {a.get('symbol', '?')} "
                    f"${a.get('amount_usd', 0):,.0f} "
                    f"(thesis: \"{a.get('thesis', '')}\")"
                )
        else:
            lines.append("  HOLD (no trades)")
    return "\n".join(lines)


def _empty_decision_entry(week_num: int, sim_date: str, error: str) -> dict:
    return {
        "week_num": week_num,
        "date": sim_date,
        "outlook": "NEUTRAL",
        "confidence": 0.0,
        "market_regime": "UNKNOWN",
        "actions": [],
        "risk_mods": [],
        "trades": [],
        "error": error,
    }


def _calc_max_drawdown(snapshots: list[dict]) -> float:
    """Calculate maximum peak-to-trough drawdown percentage."""
    if not snapshots:
        return 0.0
    peak = snapshots[0]["total_value"]
    max_dd = 0.0
    for snap in snapshots:
        val = snap["total_value"]
        if val > peak:
            peak = val
        if peak > 0:
            dd = (val - peak) / peak * 100
            if dd < max_dd:
                max_dd = dd
    return max_dd


def _calc_win_rate(trades: list[SimTrade]) -> float:
    """Percentage of SELL trades executed above the position's average cost."""
    sells = [t for t in trades if t.type == "SELL" and t.success]
    if not sells:
        return 0.0
    wins = sum(1 for t in sells if t.price >= t.avg_cost)
    return wins / len(sells) * 100
