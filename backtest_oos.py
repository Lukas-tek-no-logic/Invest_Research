#!/usr/bin/env python3
"""Out-of-sample LLM backtest: 2025-08 -> 2026-08.

Both local models are blind past ~mid-2024 (probed 2026-08-20: Nemotron
declares June 2024 and answers UNKNOWN for 2025 facts; QWEN3.5 declares
early 2024), so this window is genuine out-of-sample for LLM decisions.

Design:
  - Paired arms on IDENTICAL data: QWEN3.5 vs Nemotron, weekly cadence.
  - fallback_model=None in each arm — a silent fallback would contaminate
    the comparison; a failed week is skipped and recorded instead.
  - Static watchlist of names that were already mega/large-cap BEFORE the
    window starts (minimizes survivorship); no screener, no news, no dates
    in prompts (the harness anonymizes history as "Week N").
  - Benchmark: SPY buy-and-hold over the same window (from the harness).

Usage (container 110):
    cd /opt/invest-orchestrator && set -a && . .env && set +a
    .venv/bin/python backtest_oos.py QWEN3.5      # one arm per invocation
    .venv/bin/python backtest_oos.py Nemotron

Results land in backtest_oos_<model>_<window>.json.
"""
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "orchestrator"))

from src.backtest.runner import run_backtest
from src.llm_client import LLMClient

START = "2025-08-01"
END = "2026-08-15"

# Large/mega caps as of mid-2024 (before the window) + broad ETFs.
WATCHLIST = [
    "SPY", "QQQ", "IWM",
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
    "JPM", "V", "UNH", "XOM", "LLY", "HD", "KO", "MRK", "COST", "WMT", "DIS",
]

ACCOUNT_CONFIG = {
    "name": "Backtest OOS",
    "strategy": "core_satellite",
    "strategy_description": "Core-Satellite: 60% ETF core + 30% stock satellites + 10% cash reserve",
    "prompt_style": ("Balance risk and return. Prefer broad ETF exposure as core, "
                     "select individual stocks as satellites for alpha."),
    "preferred_metrics": ["SMA", "RSI", "PE"],
    "horizon": "weeks to months",
    "risk_profile": {
        "max_position_pct": 20,
        "min_cash_pct": 10,
        "max_trades_per_cycle": 5,
        "min_order_usd": 250,
        "stop_loss_pct": -15,
        "min_holding_days": 14,
        "max_sector_exposure_pct": 40,
    },
    "watchlist": WATCHLIST,
}


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: backtest_oos.py <model>")
        sys.exit(2)
    model = sys.argv[1]

    cfg = dict(ACCOUNT_CONFIG)
    cfg["model"] = model
    cfg["fallback_model"] = None  # hard arm separation

    t0 = time.time()

    def progress(week: int, total: int, date: str) -> None:
        elapsed = (time.time() - t0) / 60
        print(f"[{model}] week {week}/{total} {date}  ({elapsed:.0f} min)", flush=True)

    result = run_backtest(
        account_config=cfg,
        start_date=START,
        end_date=END,
        llm_client=LLMClient(),
        initial_cash=10_000,
        on_progress=progress,
    )

    out = Path(__file__).resolve().parent / f"backtest_oos_{model}_{START}_{END}.json"
    payload = {
        "model": model,
        "window": [START, END],
        "watchlist": WATCHLIST,
        "final_value": result.final_value,
        "total_return_pct": result.total_return_pct,
        "benchmark_return_pct": result.benchmark_return_pct,
        "alpha_pct": result.total_return_pct - result.benchmark_return_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "win_rate_pct": result.win_rate_pct,
        "trades": [asdict(t) for t in result.trades],
        "snapshots": result.snapshots,
        "decisions": result.decisions,
        "error": result.error,
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    out.write_text(json.dumps(payload, indent=2, default=str))

    print(f"\n=== {model} {START} -> {END} ===")
    print(f"final:     ${result.final_value:,.2f}  ({result.total_return_pct:+.2f}%)")
    print(f"benchmark: SPY {result.benchmark_return_pct:+.2f}%")
    print(f"alpha:     {result.total_return_pct - result.benchmark_return_pct:+.2f} pp")
    print(f"max DD:    {result.max_drawdown_pct:.2f}%   win rate: {result.win_rate_pct:.0f}%"
          f"   trades: {len(result.trades)}")
    if result.error:
        print(f"ERROR: {result.error}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
