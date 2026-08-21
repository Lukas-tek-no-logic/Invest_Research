#!/usr/bin/env python3
"""Deterministic tests of the forced-flow calendar theses (agents' round 2).

1. OPEX week (Stivers-Sun): SPY close-to-close over the week ending on the
   third Friday vs all other weeks.
2. Turn-of-month / Dash-for-Cash (Etula et al.): mean daily SPY return by
   position relative to month end (T-3..T-1 vs T+1..T+3 vs rest).
3. Harvey-Mazzoleni-Melone rebalancing: when SPY-TLT month-to-date spread is
   large by T-3, do the last 2 sessions of the month lean the other way?
4. D+1 reversal after |daily move| > 1.5% (LETF-flow residue).

Data: SPY (1993->) and TLT (2002->) dailies from yfinance. No look-ahead:
all conditioning uses information available at the prior close.
"""
import sys

sys.path.insert(0, "orchestrator")
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats


def third_friday(year, month):
    d = pd.Timestamp(year=year, month=month, day=1)
    fridays = pd.date_range(d, d + pd.offsets.MonthEnd(0), freq="W-FRI")
    return fridays[2]


def main():
    spy = yf.Ticker("SPY").history(start="1993-02-01", auto_adjust=True)["Close"]
    spy.index = spy.index.tz_localize(None)
    tlt = yf.Ticker("TLT").history(start="2002-08-01", auto_adjust=True)["Close"]
    tlt.index = tlt.index.tz_localize(None)
    r = spy.pct_change().dropna()

    # ── 1. OPEX week ─────────────────────────────────────────────────────────
    print("=== 1. TYDZIEŃ OPEX (SPY, close pt. przed -> close 3. piątku) ===")
    wk = spy.resample("W-FRI").last().dropna()
    wret = wk.pct_change().dropna() * 100
    opex_dates = {third_friday(ts.year, ts.month) for ts in wk.index}
    is_opex = pd.Series([ts in opex_dates for ts in wret.index], index=wret.index)
    for label, lo, hi in [("1993-2011", "1993", "2011"), ("2012-2019", "2012", "2019"),
                          ("2020-2026", "2020", "2026"), ("CAŁOŚĆ", "1993", "2026")]:
        m = (wret.index >= lo) & (wret.index <= f"{hi}-12-31")
        a, b = wret[m & is_opex], wret[m & ~is_opex]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        print(f"  {label:10s} OPEX: {a.mean():+.3f}%/tydz. (n={len(a)})  "
              f"inne: {b.mean():+.3f}% (n={len(b)})  diff={a.mean()-b.mean():+.3f}pp  "
              f"t={t:.2f} p={p:.3f}")

    # ── 2. Turn-of-month ─────────────────────────────────────────────────────
    print("\n=== 2. TURN-OF-MONTH (SPY, zwrot dzienny wg pozycji w miesiącu) ===")
    dates = r.index
    month_key = dates.to_period("M")
    pos_from_end, pos_from_start = np.zeros(len(r)), np.zeros(len(r))
    for mk in month_key.unique():
        idx = np.where(month_key == mk)[0]
        pos_from_end[idx] = np.arange(len(idx))[::-1] + 1   # 1 = last session
        pos_from_start[idx] = np.arange(len(idx)) + 1        # 1 = first session
    rr = r * 100
    buckets = {
        "T-3..T-1 (koniec mies.)": (pos_from_end <= 3),
        "T+1..T+3 (początek)": (pos_from_start <= 3),
    }
    rest = ~(buckets["T-3..T-1 (koniec mies.)"] | buckets["T+1..T+3 (początek)"])
    for label, lo in [("1993-2011", "1993"), ("2012-2026", "2012"), ("CAŁOŚĆ", "1993")]:
        m = rr.index >= lo
        base = rr[m & rest]
        line = f"  {label:10s} reszta: {base.mean():+.4f}%/d"
        for bl, mask in buckets.items():
            seg = rr[m & mask]
            t, p = stats.ttest_ind(seg, base, equal_var=False)
            line += f"  | {bl}: {seg.mean():+.4f}%/d (t={t:.2f})"
        print(line)

    # ── 3. Harvey 60/40 rebalans ─────────────────────────────────────────────
    print("\n=== 3. REBALANS 60/40 (spread MTD SPY-TLT do T-3 -> SPY w T-2..T-0) ===")
    both = pd.DataFrame({"spy": spy, "tlt": tlt}).dropna()
    res = []
    for mk, g in both.groupby(both.index.to_period("M")):
        if len(g) < 8:
            continue
        upto = g.iloc[:-2]  # info do T-3 (przed ostatnimi 2 sesjami)
        spread = (upto.spy.iloc[-1] / upto.spy.iloc[0] - 1) - (upto.tlt.iloc[-1] / upto.tlt.iloc[0] - 1)
        last2 = g.spy.iloc[-1] / g.spy.iloc[-3] - 1
        res.append((spread * 100, last2 * 100))
    df = pd.DataFrame(res, columns=["spread", "last2"])
    ter = pd.qcut(df.spread, 3, labels=["TLT>>SPY", "środek", "SPY>>TLT"])
    for name, g in df.groupby(ter, observed=True):
        t, p = stats.ttest_1samp(g.last2, df.last2.mean())
        print(f"  {name:10s} (śr. spread {g.spread.mean():+5.1f}pp): SPY ost. 2 sesje "
              f"= {g.last2.mean():+.3f}% (n={len(g)}, t vs śr.={t:.2f})")
    hi = df[df.spread > 5]
    t, p = stats.ttest_ind(hi.last2, df.last2, equal_var=False)
    print(f"  trigger spread>+5pp: {hi.last2.mean():+.3f}% (n={len(hi)}) vs "
          f"wszystkie {df.last2.mean():+.3f}%  t={t:.2f} p={p:.3f}")

    # ── 4. Rewersja D+1 po dużych dniach ────────────────────────────────────
    print("\n=== 4. REWERSJA D+1 PO |ruch|>1.5% (SPY) ===")
    big_up, big_dn = r.shift(1) > 0.015, r.shift(1) < -0.015
    for label, lo in [("1993-2011", "1993"), ("2012-2026", "2012")]:
        m = r.index >= lo
        up, dn, base = rr[m & big_up], rr[m & big_dn], rr[m & ~(big_up | big_dn)]
        tu, _ = stats.ttest_ind(up, base, equal_var=False)
        td, _ = stats.ttest_ind(dn, base, equal_var=False)
        print(f"  {label:10s} po +1.5%: {up.mean():+.3f}%/d (n={len(up)}, t={tu:.2f})  "
              f"po -1.5%: {dn.mean():+.3f}%/d (n={len(dn)}, t={td:.2f})  "
              f"baza: {base.mean():+.3f}%")


if __name__ == "__main__":
    main()
