#!/usr/bin/env python3
"""Zero-GPU replay of decision-architecture mechanisms on the finished OOS run.

Inputs: the two backtest_oos_*.json files (identical 55 weekly dates).
Mechanisms:
  1. Hedge meta-allocator (Freund-Schapire multiplicative weights) over four
     experts: QWEN account, Nemotron account, SPY buy-and-hold, CASH. Weights
     move AFTER each observed week (no look-ahead); regret vs the best expert
     is bounded by O(sqrt(T ln K)).
  2. Agreement gating: enter a symbol only in weeks when BOTH models proposed
     a BUY on it; exit when EITHER proposes a SELL on a held symbol.
     $1 commission per fill (IBKR floor).

Usage:
    .venv/bin/python replay_meta.py <qwen.json> <nemo.json>
"""
import json
import math
import sys
from collections import defaultdict

sys.path.insert(0, "orchestrator")
import yfinance as yf

FEE = 1.0
TURNOVER_COST = 0.0005  # 5 bps on weight shifts between experts


def weekly_returns(snapshots):
    vals = [s["total_value"] for s in snapshots]
    return [vals[i + 1] / vals[i] - 1 for i in range(len(vals) - 1)]


def price_matrix(symbols, dates):
    """Adjusted closes for each symbol on (or before) each snapshot date."""
    out = {}
    start, end = dates[0], dates[-1]
    for sym in symbols:
        h = yf.Ticker(sym).history(start=start, end=end, auto_adjust=True)["Close"]
        h.index = h.index.strftime("%Y-%m-%d")
        series = []
        last = None
        hi = list(h.items())
        for d in dates:
            while hi and hi[0][0] <= d:
                last = hi.pop(0)[1]
            series.append(float(last) if last is not None else None)
        out[sym] = series
    return out


def hedge(experts: dict[str, list[float]]):
    names = list(experts)
    T = len(next(iter(experts.values())))
    K = len(names)
    eta = math.sqrt(8 * math.log(K) / T)
    w = {n: 1.0 / K for n in names}
    wealth, weights_path = 1.0, []
    for t in range(T):
        weights_path.append(dict(w))
        r = sum(w[n] * experts[n][t] for n in names)
        prev = dict(w)
        # multiplicative update on observed returns
        for n in names:
            w[n] *= math.exp(eta * experts[n][t])
        z = sum(w.values())
        for n in names:
            w[n] /= z
        turnover = sum(abs(w[n] - prev[n]) for n in names)
        wealth *= (1 + r - TURNOVER_COST * turnover)
    return wealth, w, weights_path


def agreement_replay(qwen, nemo, prices, dates):
    cash, fees = 10_000.0, 0.0
    held: dict[str, float] = defaultdict(float)  # symbol -> qty
    n_trades = 0
    values = []
    def px(sym, i):
        p = prices.get(sym, [None] * len(dates))[i]
        return p if p and p > 0 else None

    dec_q = {d["date"]: d for d in qwen["decisions"]}
    dec_n = {d["date"]: d for d in nemo["decisions"]}
    for i, date in enumerate(dates):
        aq = {(a["type"], a["symbol"]) for a in dec_q.get(date, {}).get("actions", [])}
        an = {(a["type"], a["symbol"]) for a in dec_n.get(date, {}).get("actions", [])}
        # exits first: EITHER model sells a held name
        sells = {s for t, s in (aq | an) if t == "SELL" and held.get(s, 0) > 0}
        for s in sells:
            p = px(s, i)
            if p is None:
                continue
            cash += held[s] * p - FEE
            fees += FEE
            n_trades += 1
            held[s] = 0.0
        # entries: BOTH models buy the same name the same week
        buys = {s for t, s in aq if t == "BUY"} & {s for t, s in an if t == "BUY"}
        for s in sorted(buys):
            p = px(s, i)
            if p is None or cash < 500 + FEE:
                continue
            spend = min(1000.0, cash - FEE)
            held[s] += spend / p
            cash -= spend + FEE
            fees += FEE
            n_trades += 1
        total = cash + sum(q * (px(s, i) or 0) for s, q in held.items() if q > 0)
        values.append(total)
    return values, n_trades, fees


def main():
    qwen = json.load(open(sys.argv[1]))
    nemo = json.load(open(sys.argv[2]))
    dates = [s["date"] for s in qwen["snapshots"]]
    assert dates == [s["date"] for s in nemo["snapshots"]], "windows differ"

    watch = sorted(set(qwen["watchlist"]) | {"SPY"})
    prices = price_matrix(watch, dates)

    spy = prices["SPY"]
    spy_rets = [spy[i + 1] / spy[i] - 1 for i in range(len(spy) - 1)]
    experts = {
        "QWEN": weekly_returns(qwen["snapshots"]),
        "NEMO": weekly_returns(nemo["snapshots"]),
        "SPY": spy_rets,
        "CASH": [0.0] * len(spy_rets),
    }
    wealth, final_w, path = hedge(experts)
    spy_total = (spy[-1] / spy[0] - 1) * 100

    print(f"okno: {dates[0]} -> {dates[-1]}  (T={len(spy_rets)} tygodni)")
    print(f"SPY buy&hold:        {spy_total:+.2f}%")
    print(f"QWEN:                {(qwen['snapshots'][-1]['total_value']/10000-1)*100:+.2f}%")
    print(f"NEMO:                {(nemo['snapshots'][-1]['total_value']/10000-1)*100:+.2f}%")
    print(f"\n[1] HEDGE meta-alokator (K=4, eta=sqrt(8lnK/T), 5bps turnover):")
    print(f"    wynik:           {(wealth-1)*100:+.2f}%   vs SPY: {(wealth-1)*100-spy_total:+.2f} pp")
    print(f"    wagi końcowe:    " + ", ".join(f"{n}={w:.2f}" for n, w in final_w.items()))
    mid = path[len(path)//2]
    print(f"    wagi w połowie:  " + ", ".join(f"{n}={w:.2f}" for n, w in mid.items()))

    vals, n_trades, fees = agreement_replay(qwen, nemo, prices, dates)
    ag_ret = (vals[-1] / 10000 - 1) * 100
    print(f"\n[2] AGREEMENT GATING (obaj kupują -> wejście $1000; ktokolwiek sprzedaje -> wyjście; $1/fill):")
    print(f"    wynik:           {ag_ret:+.2f}%   vs SPY: {ag_ret-spy_total:+.2f} pp")
    print(f"    transakcje:      {n_trades} (QWEN solo: {len(qwen['trades'])}, NEMO solo: {len(nemo['trades'])})  prowizje: ${fees:.0f}")
    invested_weeks = sum(1 for v_i, d_i in zip(vals, dates) if v_i and abs(v_i - vals[0]) > 1e-9)
    print(f"    uwaga: część kapitału zwykle w gotówce — profil ryzyka nizszy niz SPY")


if __name__ == "__main__":
    main()
