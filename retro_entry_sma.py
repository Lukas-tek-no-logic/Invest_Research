#!/usr/bin/env python3
"""SMA as a STOCK-FINDING filter with buy-and-hold after entry (no exits).

Question: does selecting/timing ENTRIES with SMA50 beat just buying the same
universe outright, when everything is held to the end of the window?

Universe: ~S&P 100 (current members — survivorship hits every arm equally,
so the PAIRED comparison between rules stays valid; absolute levels vs SPY
are inflated and flagged as such).

Arms (equal-weight sleeves, $1 per fill, hold to window end):
  ALL      — buy every stock on the first session of the window
  ABOVE50  — buy on day 1 ONLY stocks with close > SMA50 (capital redistributed
             across qualifiers)
  ABOVE200 — same with SMA200
  CROSS50  — per stock: buy at its first upward SMA50 cross inside the window
             (already above at start = buy at start); idle sleeves sit in cash
  CROSS200 — same with SMA200

Usage: .venv/bin/python retro_entry_sma.py
"""
import sys

sys.path.insert(0, "orchestrator")
import pandas as pd
import yfinance as yf

UNIVERSE = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","BRK-B","JPM",
    "V","MA","UNH","XOM","LLY","HD","KO","PEP","MRK","COST","WMT","DIS","BAC",
    "ABBV","CVX","CRM","AMD","NFLX","TMO","ORCL","ACN","MCD","ABT","CSCO",
    "ADBE","PM","INTC","TXN","QCOM","IBM","GE","CAT","VZ","T","NKE","HON",
    "UNP","LOW","INTU","SPGI","GS","MS","BLK","AXP","BKNG","AMGN","PFE","DHR",
    "RTX","NEE","LIN","SBUX","BMY","UPS","MDT","GILD","CVS","DE","LMT","PLD",
    "AMT","SCHW","C","MO","SO","DUK","CI","BA","MMM","TGT","USB","COP","EMR",
    "F","GM","FDX","GD","CL","KHC","AIG","MET","DOW","WFC","COF","PYPL",
]
FEE = 1.0
CAPITAL = 10_000.0


def fetch(symbols, start="2013-06-01"):
    data = {}
    for s in symbols:
        try:
            h = yf.Ticker(s).history(start=start, auto_adjust=True)["Close"].dropna()
            if len(h) > 300:
                h.index = h.index.tz_localize(None)
                data[s] = h
        except Exception:
            pass
    return data


def run_arm(data, rule, start, end):
    syms = list(data)
    sleeve = CAPITAL / len(syms)
    total_final, fees, entered, delays = 0.0, 0, 0, []
    for s in syms:
        px = data[s]
        n = 50 if "50" in rule else (200 if "200" in rule else 0)
        sma = px.rolling(n).mean() if n else None
        win = px[(px.index >= start) & (px.index <= end)]
        if win.empty:
            total_final += sleeve
            continue
        entry_date = None
        if rule == "ALL":
            entry_date = win.index[0]
        elif rule.startswith("ABOVE"):
            if win.iloc[0] > sma.loc[win.index[0]]:
                entry_date = win.index[0]
        elif rule.startswith("CROSS"):
            above = win > sma.reindex(win.index)
            if above.iloc[0]:
                entry_date = win.index[0]
            else:
                crosses = above & ~above.shift(1, fill_value=True)
                hits = crosses[crosses].index
                entry_date = hits[0] if len(hits) else None
        if entry_date is None:
            total_final += sleeve  # never qualified — sleeve stays cash
            continue
        entered += 1
        delays.append((entry_date - win.index[0]).days)
        ret = win.iloc[-1] / win.loc[entry_date]
        total_final += (sleeve - FEE) * ret
        fees += FEE
    # ABOVE* arms redistribute: rerun with sleeve = CAPITAL / qualifiers
    if rule.startswith("ABOVE") and entered:
        sleeve2 = CAPITAL / entered
        total_final = 0.0
        for s in syms:
            px = data[s]
            n = 50 if "50" in rule else 200
            sma = px.rolling(n).mean()
            win = px[(px.index >= start) & (px.index <= end)]
            if win.empty or not win.iloc[0] > sma.loc[win.index[0]]:
                continue
            total_final += (sleeve2 - FEE) * (win.iloc[-1] / win.iloc[0])
    avg_delay = sum(delays) / len(delays) if delays else 0
    return (total_final / CAPITAL - 1) * 100, entered, len(syms), avg_delay, fees


def fetch_quality(symbols):
    """Top-half by ROE AND profit margin (TODAY'S values — look-ahead that
    FAVOURS this arm; slow-moving metrics, but treat absolute levels warily)."""
    import time
    metrics = {}
    for s in symbols:
        try:
            info = yf.Ticker(s).info
            roe = info.get("returnOnEquity")
            pm = info.get("profitMargins")
            if roe is not None and pm is not None:
                metrics[s] = (roe, pm)
        except Exception:
            pass
        time.sleep(0.2)
    roes = sorted(v[0] for v in metrics.values())
    pms = sorted(v[1] for v in metrics.values())
    med_roe, med_pm = roes[len(roes)//2], pms[len(pms)//2]
    return {s for s, (r, p) in metrics.items() if r >= med_roe and p >= med_pm}


def main():
    data = fetch(UNIVERSE)
    quality = fetch_quality(list(data))
    qdata = {s: v for s, v in data.items() if s in quality}
    print(f"filtr fundamentalny (ROE i marża >= mediany, DZISIEJSZE dane -> look-ahead na korzysc): {len(qdata)} spolek")
    spy = yf.Ticker("SPY").history(start="2013-06-01", auto_adjust=True)["Close"]
    spy.index = spy.index.tz_localize(None)
    print(f"uniwersum: {len(data)} spółek z danymi")
    for label, start, end in [("5 LAT", "2021-08-02", "2026-08-15"),
                              ("11 LAT", "2015-08-03", "2026-08-15")]:
        w = spy[(spy.index >= start) & (spy.index <= end)]
        spy_tot = (w.iloc[-1] / w.iloc[0] - 1) * 100
        print(f"\n===== {label} ({start[:7]} -> {end[:7]}) — SPY: {spy_tot:+.1f}% =====")
        print(f"{'reguła':10s} {'zwrot':>9s} {'kupione':>9s} {'śr. opóźn. wejścia':>20s}")
        for rule in ["ALL", "ABOVE50", "ABOVE200", "CROSS50", "CROSS200"]:
            tot, ent, n, delay, fees = run_arm(data, rule, start, end)
            print(f"{rule:10s} {tot:+8.1f}% {ent:5d}/{n:<3d} {delay:17.0f} dni")
        print("-- z filtrem fundamentalnym (uwaga: look-ahead na korzysc) --")
        for rule in ["ALL", "ABOVE50", "CROSS50"]:
            tot, ent, n, delay, fees = run_arm(qdata, rule, start, end)
            print(f"FUND+{rule:5s} {tot:+8.1f}% {ent:5d}/{n:<3d} {delay:17.0f} dni")


if __name__ == "__main__":
    main()
