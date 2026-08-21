#!/usr/bin/env python3
"""Reflexive theses tested on our own OOS run (round-2 agents S2/S4-style).

Data: the two finished backtest JSONs (identical 55 weekly dates, models blind
to the window). Tests:

1. FADE-YOUR-OWN-MODEL: does the LLM's weekly `confidence` predict the forward
   outcome of that week's BUYs — and in which direction? (Spearman + terciles;
   includes the degeneration check: a confidence histogram stuck at 0.7-0.8
   kills the thesis before any correlation is computed.)
2. CROWDING: BUYs made by BOTH models on the same symbol the same week vs
   solo BUYs — 4-week forward return vs SPY. (Inverse of the failed
   agreement-gating: maybe agreement marks crowding, not quality.)

Usage: .venv/bin/python retro_reflexive.py <qwen.json> <nemo.json>
"""
import json
import sys
from collections import Counter

sys.path.insert(0, "orchestrator")
import numpy as np
import yfinance as yf
from scipy import stats


def price_series(symbols, start, end):
    out = {}
    for s in symbols:
        h = yf.Ticker(s).history(start=start, end=end, auto_adjust=True)["Close"]
        h.index = h.index.strftime("%Y-%m-%d")
        out[s] = h
    return out


def fwd_return(prices, sym, date, horizon_dates):
    """Return over `horizon_dates` snapshot steps ahead (weekly grid)."""
    ser = prices.get(sym)
    if ser is None:
        return None
    idx = [d for d in horizon_dates]
    try:
        i = idx.index(date)
    except ValueError:
        return None
    if i + 4 >= len(idx):
        return None
    def px(d):
        sub = ser[ser.index <= d]
        return float(sub.iloc[-1]) if len(sub) else None
    p0, p4 = px(idx[i]), px(idx[i + 4])
    return (p4 / p0 - 1) * 100 if p0 and p4 else None


def main():
    qwen = json.load(open(sys.argv[1]))
    nemo = json.load(open(sys.argv[2]))
    dates = [s["date"] for s in qwen["snapshots"]]
    watch = sorted(set(qwen["watchlist"]) | {"SPY"})
    prices = price_series(watch, dates[0], "2026-08-22")

    spy_fwd = {d: fwd_return(prices, "SPY", d, dates) for d in dates}

    # ── 1. confidence vs forward outcome of that week's buys ────────────────
    print("=== 1. FADE-YOUR-OWN-MODEL: confidence vs forward 4-tyg. wynik BUY-ów ===")
    for name, arm in (("QWEN", qwen), ("NEMO", nemo)):
        confs, outcomes = [], []
        hist = Counter()
        for dec in arm["decisions"]:
            c = dec.get("confidence")
            if c is None:
                continue
            hist[round(float(c), 1)] += 1
            buys = [a["symbol"] for a in dec.get("actions", []) if a["type"] == "BUY"]
            rets = [fwd_return(prices, s, dec["date"], dates) for s in buys]
            rets = [x - spy_fwd[dec["date"]] for x in rets
                    if x is not None and spy_fwd[dec["date"]] is not None]
            if rets:
                confs.append(float(c))
                outcomes.append(float(np.mean(rets)))
        print(f"  {name}: histogram confidence: "
              + ", ".join(f"{k}:{v}" for k, v in sorted(hist.items())))
        if len(set(confs)) < 3:
            print(f"  {name}: confidence ZDEGENEROWANE — teza nietestowalna na tym ramieniu")
            continue
        rho, p = stats.spearmanr(confs, outcomes)
        print(f"  {name}: n={len(confs)} tygodni z BUY;  Spearman(conf, alfa4t) = "
              f"{rho:+.2f} (p={p:.2f})")
        med = np.median(confs)
        hi = [o for c, o in zip(confs, outcomes) if c > med]
        lo = [o for c, o in zip(confs, outcomes) if c <= med]
        print(f"  {name}: pewne tygodnie: {np.mean(hi):+.2f}pp vs niepewne: "
              f"{np.mean(lo):+.2f}pp (alfa 4-tyg. vs SPY)")

    # ── 2. crowding: consensus buys vs solo buys ─────────────────────────────
    print("\n=== 2. CROWDING: BUY zgodne (oba modele, ten sam tydzień) vs solo ===")
    dq = {d["date"]: {a["symbol"] for a in d.get("actions", []) if a["type"] == "BUY"}
          for d in qwen["decisions"]}
    dn = {d["date"]: {a["symbol"] for a in d.get("actions", []) if a["type"] == "BUY"}
          for d in nemo["decisions"]}
    both_r, solo_r = [], []
    for d in dates:
        b_q, b_n = dq.get(d, set()), dn.get(d, set())
        spy_f = spy_fwd.get(d)
        if spy_f is None:
            continue
        for s in (b_q & b_n):
            x = fwd_return(prices, s, d, dates)
            if x is not None:
                both_r.append(x - spy_f)
        for s in (b_q ^ b_n):
            x = fwd_return(prices, s, d, dates)
            if x is not None:
                solo_r.append(x - spy_f)
    t, p = stats.ttest_ind(both_r, solo_r, equal_var=False)
    print(f"  zgodne BUY:  n={len(both_r)}  alfa 4-tyg. vs SPY = {np.mean(both_r):+.2f}pp")
    print(f"  solo BUY:    n={len(solo_r)}  alfa 4-tyg. vs SPY = {np.mean(solo_r):+.2f}pp")
    print(f"  różnica (zgoda − solo) = {np.mean(both_r)-np.mean(solo_r):+.2f}pp  "
          f"t={t:.2f} p={p:.3f}")
    print("  interpretacja: ujemna różnica wspiera tezę crowdingu; dodatnia — ensemble")


if __name__ == "__main__":
    main()
