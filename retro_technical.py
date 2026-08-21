#!/usr/bin/env python3
"""Deterministic retro-tests: (A) quality stocks + technical timing,
(B) Hedge meta-allocator over a 5-year horizon.

All rules are price-only and deterministic, so full history is legitimate
(no LLM knowledge-cutoff issue). Fees: $1 per fill (IBKR floor).

A) Universe = the 19 pre-window mega caps from the OOS watchlist.
   Weekly decisions on daily data:
     BH    — buy & hold, equal weight
     SMA200— hold a stock only while close > SMA200, else its sleeve sits in cash
     GC    — golden cross: hold while SMA50 > SMA200
     FIB   — mechanical Fibonacci pullback: enter when close > SMA200 AND close
             inside the 38.2%-61.8% retracement of the trailing 126-session
             high-low range; exit when close < 78.6% level or close < SMA200.
   Every knob in FIB (126 sessions, which levels, which exit) is arbitrary —
   that arbitrariness is part of what this test documents.

B) Hedge (multiplicative weights) over weekly returns of 7 asset experts
   {SPY,QQQ,IWM,EFA,GLD,TLT,BIL}, eta = sqrt(8 lnK / T), 5bps turnover,
   plus eta x0.5 / x2 sensitivity. Question: does T~260 weeks fix what
   T=54 could not?

Usage: .venv/bin/python retro_technical.py
"""
import math
import sys
from dataclasses import dataclass

sys.path.insert(0, "orchestrator")
import numpy as np
import pandas as pd
import yfinance as yf

STOCKS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
          "JPM", "V", "UNH", "XOM", "LLY", "HD", "KO", "MRK", "COST", "WMT", "DIS"]
ETFS = ["SPY", "QQQ", "IWM", "EFA", "GLD", "TLT", "BIL"]
FEE = 1.0
CAPITAL = 10_000.0


def fetch(symbols, start="2013-01-01"):
    out = {}
    for s in symbols:
        h = yf.Ticker(s).history(start=start, auto_adjust=True)["Close"].dropna()
        h.index = h.index.tz_localize(None)
        out[s] = h
    return out


@dataclass
class StratResult:
    name: str
    total_pct: float
    maxdd_pct: float
    switches: int
    fees: float


def run_stock_strategy(closes: pd.Series, rule: str, start, end) -> tuple[pd.Series, int]:
    """Return the weekly in/out signal (1=held) and number of switches."""
    sma200 = closes.rolling(200).mean()
    sma50 = closes.rolling(50).mean()
    hi = closes.rolling(126).max()
    lo = closes.rolling(126).min()
    rng = hi - lo
    f382 = hi - 0.382 * rng
    f618 = hi - 0.618 * rng
    f786 = hi - 0.786 * rng

    df = pd.DataFrame({"c": closes, "sma200": sma200, "sma50": sma50,
                       "f382": f382, "f618": f618, "f786": f786}).dropna()
    df = df[(df.index >= start) & (df.index <= end)]
    weekly = df.resample("W-FRI").last().dropna()

    held = False
    sig = []
    for _, r in weekly.iterrows():
        if rule == "BH":
            held = True
        elif rule == "SMA200":
            held = r.c > r.sma200
        elif rule == "GC":
            held = r.sma50 > r.sma200
        elif rule == "FIB":
            if not held:
                held = (r.c > r.sma200) and (r.f618 <= r.c <= r.f382)
            else:
                if r.c < r.f786 or r.c < r.sma200:
                    held = False
        sig.append(1.0 if held else 0.0)
    sig = pd.Series(sig, index=weekly.index)
    switches = int((sig.diff().abs() > 0).sum() + (1 if sig.iloc[0] > 0 else 0))
    return sig, switches, weekly["c"]


def portfolio_test(data, rule, start, end) -> StratResult:
    sleeve = CAPITAL / len(STOCKS)
    curves, total_switches = [], 0
    for s in STOCKS:
        sig, sw, px = run_stock_strategy(data[s], rule, start, end)
        rets = px.pct_change().fillna(0.0)
        # position decided at week t applies to week t+1's return
        strat_rets = rets * sig.shift(1).fillna(0.0)
        curve = sleeve * (1 + strat_rets).cumprod()
        curves.append(curve)
        total_switches += sw
    port = pd.concat(curves, axis=1).sum(axis=1)
    fees = total_switches * FEE
    final = port.iloc[-1] - fees
    peak = port.cummax()
    maxdd = ((port - peak) / peak).min() * 100
    return StratResult(rule, (final / CAPITAL - 1) * 100, maxdd, total_switches, fees)


def hedge_test(data, start, end, eta_mult=1.0):
    px = pd.DataFrame({s: data[s] for s in ETFS}).dropna()
    px = px[(px.index >= start) & (px.index <= end)].resample("W-FRI").last().dropna()
    rets = px.pct_change().dropna()
    T, K = len(rets), len(ETFS)
    eta = eta_mult * math.sqrt(8 * math.log(K) / T)
    w = np.full(K, 1.0 / K)
    wealth = 1.0
    for _, r in rets.iterrows():
        rv = r.values
        wealth *= 1 + float(w @ rv)
        prev = w.copy()
        w = w * np.exp(eta * rv)
        w /= w.sum()
        wealth *= 1 - 0.0005 * float(np.abs(w - prev).sum())
    expert_totals = {s: (px[s].iloc[-1] / px[s].iloc[0] - 1) * 100 for s in ETFS}
    return (wealth - 1) * 100, expert_totals, dict(zip(ETFS, w.round(3))), T, eta


def main():
    data = fetch(STOCKS + ETFS)
    for label, start, end in [("5 LAT", "2021-08-01", "2026-08-15"),
                              ("11 LAT", "2015-08-01", "2026-08-15")]:
        spy = data["SPY"][(data["SPY"].index >= start) & (data["SPY"].index <= end)]
        spy_tot = (spy.iloc[-1] / spy.iloc[0] - 1) * 100
        print(f"\n===== A) TIMING TECHNICZNY NA 19 MEGACAPACH — {label} "
              f"({start[:7]} -> {end[:7]}), SPY B&H: {spy_tot:+.1f}% =====")
        print(f"{'reguła':8s} {'zwrot':>9s} {'maxDD':>8s} {'przełącz.':>10s} {'prowizje':>9s}")
        for rule in ["BH", "SMA200", "GC", "FIB"]:
            r = portfolio_test(data, rule, start, end)
            print(f"{r.name:8s} {r.total_pct:+8.1f}% {r.maxdd_pct:7.1f}% "
                  f"{r.switches:10d} {r.fees:8.0f}$")

    print("\n===== B) HEDGE NA 7 EKSPERTACH-ETF, tygodniowo =====")
    for label, start, end in [("T~54 (rok)", "2025-08-01", "2026-08-15"),
                              ("T~260 (5 lat)", "2021-08-01", "2026-08-15"),
                              ("T~570 (11 lat)", "2015-08-01", "2026-08-15")]:
        tot, experts, w_final, T, eta = hedge_test(data, start, end)
        best = max(experts, key=experts.get)
        print(f"\n{label}: T={T}, eta={eta:.3f}")
        print("  eksperci: " + ", ".join(f"{s} {v:+.0f}%" for s, v in
                                         sorted(experts.items(), key=lambda x: -x[1])))
        print(f"  HEDGE: {tot:+.1f}%   vs SPY: {tot - experts['SPY']:+.1f} pp"
              f"   vs najlepszy ({best}): {tot - experts[best]:+.1f} pp")
        print(f"  wagi końcowe: {w_final}")
        for m in (0.5, 2.0, 5.0):
            tot_m, _, _, _, _ = hedge_test(data, start, end, eta_mult=m)
            print(f"  czułość eta x{m}: {tot_m:+.1f}%")


if __name__ == "__main__":
    main()
