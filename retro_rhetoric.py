#!/usr/bin/env python3
"""S5 'rhetoric coroner': does the LANGUAGE of an LLM's trade thesis predict
its realized outcome?

Corpus: BUY theses from the two OOS backtest JSONs (models blind to the
window), outcome = 4-week forward alpha of the bought symbol vs SPY.

Modes:
  lexical  — cheap local features (no LLM): thesis length, digit density,
             hedging words, superlatives, trend-extrapolation words,
             technical-indicator mentions. OLS/rank tests + temporal split.
  label    — batch-label every thesis with Nemotron (llama-swap) on 5 axes;
             writes rhetoric_labels.json next to the input.
  analyze  — join labels with outcomes, temporal-split logistic test.

Usage:
  retro_rhetoric.py lexical <qwen.json> <nemo.json>
  retro_rhetoric.py label   <qwen.json> <nemo.json>
  retro_rhetoric.py analyze <qwen.json> <nemo.json> <labels.json>
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "orchestrator")
import numpy as np
import yfinance as yf
from scipy import stats

HEDGES = re.compile(r"\b(may|might|could|possibly|potentially|appears|seems|"
                    r"likely|cautious|monitor|watch|uncertain|risk)\b", re.I)
SUPERL = re.compile(r"\b(strong(est)?|massive|huge|exceptional|perfect|clear(ly)?|"
                    r"best|surge|soar|explosive|breakout|momentum)\b", re.I)
TREND = re.compile(r"\b(continu\w+|extend\w+|further|keep\w*|remain\w*|sustain\w+|"
                   r"riding|trend)\b", re.I)
TECH = re.compile(r"\b(RSI|MACD|SMA\d*|bollinger|ADX|support|resistance|oversold|"
                  r"overbought)\b", re.I)


def build_corpus(qwen_path, nemo_path):
    rows = []
    arms = [("QWEN", json.load(open(qwen_path))), ("NEMO", json.load(open(nemo_path)))]
    dates = [s["date"] for s in arms[0][1]["snapshots"]]
    watch = sorted(set(arms[0][1]["watchlist"]) | {"SPY"})
    prices = {}
    for s in watch:
        h = yf.Ticker(s).history(start=dates[0], end="2026-08-22", auto_adjust=True)["Close"]
        h.index = h.index.strftime("%Y-%m-%d")
        prices[s] = h

    def px(sym, d):
        ser = prices[sym][prices[sym].index <= d]
        return float(ser.iloc[-1]) if len(ser) else None

    for model, arm in arms:
        for dec in arm["decisions"]:
            d = dec["date"]
            i = dates.index(d)
            if i + 4 >= len(dates):
                continue
            d4 = dates[i + 4]
            spy0, spy4 = px("SPY", d), px("SPY", d4)
            for a in dec.get("actions", []):
                if a.get("type") != "BUY" or not a.get("thesis"):
                    continue
                p0, p4 = px(a["symbol"], d), px(a["symbol"], d4)
                if not all((p0, p4, spy0, spy4)):
                    continue
                alpha = (p4 / p0 - spy4 / spy0) * 100
                rows.append({"model": model, "date": d, "symbol": a["symbol"],
                             "thesis": a["thesis"], "alpha4w": round(alpha, 3)})
    return rows


def lexical_features(t):
    words = max(len(t.split()), 1)
    return {
        "len_words": words,
        "digit_density": sum(c.isdigit() for c in t) / len(t),
        "hedge_rate": len(HEDGES.findall(t)) / words * 100,
        "superl_rate": len(SUPERL.findall(t)) / words * 100,
        "trend_rate": len(TREND.findall(t)) / words * 100,
        "tech_rate": len(TECH.findall(t)) / words * 100,
    }


def mode_lexical(qp, np_):
    rows = build_corpus(qp, np_)
    print(f"korpus: {len(rows)} tez BUY z wynikiem 4-tyg. alfa vs SPY "
          f"(śr. {np.mean([r['alpha4w'] for r in rows]):+.2f}pp)")
    feats = [lexical_features(r["thesis"]) for r in rows]
    y = np.array([r["alpha4w"] for r in rows])
    split = sorted(r["date"] for r in rows)[len(rows) // 2]
    print(f"\n{'cecha':16s} {'Spearman':>9s} {'p':>7s} | walidacja czasowa (po {split})")
    for k in feats[0]:
        x = np.array([f[k] for f in feats])
        rho, p = stats.spearmanr(x, y)
        tr = [i for i, r in enumerate(rows) if r["date"] <= split]
        te = [i for i, r in enumerate(rows) if r["date"] > split]
        rho_tr, _ = stats.spearmanr(x[tr], y[tr])
        rho_te, _ = stats.spearmanr(x[te], y[te])
        flag = " <— spójny znak" if rho_tr * rho_te > 0 and abs(rho_te) > 0.1 else ""
        print(f"{k:16s} {rho:+9.3f} {p:7.3f} | train {rho_tr:+.2f} / test {rho_te:+.2f}{flag}")
    Path("rhetoric_corpus.json").write_text(json.dumps(rows, indent=1))
    print("\nkorpus zapisany: rhetoric_corpus.json")


LABEL_PROMPT = """Oceń poniższe uzasadnienia decyzji kupna akcji. Dla KAŻDEGO zwróć obiekt JSON z polami:
id (liczba), extrapolation (0-2: 0=teza mean-reversion/wyceny, 1=mieszana, 2=czysta ekstrapolacja trwającego trendu),
story (0-2: 0=liczby dominują, 1=mieszane, 2=narracja dominuje),
urgency (0-2: presja natychmiastowości/FOMO),
hedging (0-2: gęstość asekuracji),
crowd (0-2: czy teza powtarza powszechnie znaną narrację rynkową).
Zwróć TYLKO tablicę JSON, bez komentarza.

{items}"""


def mode_label(qp, np_):
    import httpx
    rows = build_corpus(qp, np_)
    out = []
    for i in range(0, len(rows), 5):
        batch = rows[i:i + 5]
        items = "\n".join(f'[id {i+j}] "{r["thesis"][:400]}"' for j, r in enumerate(batch))
        resp = httpx.post("http://192.168.0.169:8080/v1/chat/completions", timeout=600,
                          json={"model": "Nemotron", "temperature": 0, "max_tokens": 8000,
                                "messages": [{"role": "user",
                                              "content": LABEL_PROMPT.format(items=items)}]})
        msg = resp.json()["choices"][0]["message"]
        txt = msg.get("content") or msg.get("reasoning_content") or ""
        m = re.search(r"\[.*\]", txt, re.S)
        if m:
            try:
                out.extend(json.loads(m.group(0)))
            except json.JSONDecodeError:
                print(f"batch {i}: JSON parse fail, pomijam")
        print(f"batch {i//5 + 1}/{(len(rows)+4)//5} ok, etykiet: {len(out)}", flush=True)
    Path("rhetoric_labels.json").write_text(json.dumps(out, indent=1))
    print(f"zapisano rhetoric_labels.json ({len(out)} etykiet)")


def mode_analyze(qp, np_, labels_path):
    rows = build_corpus(qp, np_)
    labels = {int(l["id"]): l for l in json.load(open(labels_path)) if "id" in l}
    y, X, ds = [], [], []
    axes = ["extrapolation", "story", "urgency", "hedging", "crowd"]
    for i, r in enumerate(rows):
        if i in labels:
            y.append(r["alpha4w"])
            X.append([float(labels[i].get(a, 0) or 0) for a in axes])
            ds.append(r["date"])
    y, X = np.array(y), np.array(X)
    print(f"dopasowane etykiety: {len(y)}/{len(rows)}")
    split = sorted(ds)[len(ds) // 2]
    tr = [i for i, d in enumerate(ds) if d <= split]
    te = [i for i, d in enumerate(ds) if d > split]
    print(f"\n{'oś':14s} {'Spearman':>9s} {'p':>7s} | train/test (split {split})")
    for j, a in enumerate(axes):
        rho, p = stats.spearmanr(X[:, j], y)
        rt, _ = stats.spearmanr(X[tr, j], y[tr])
        rv, _ = stats.spearmanr(X[te, j], y[te])
        flag = " <— spójny znak" if rt * rv > 0 and abs(rv) > 0.1 else ""
        print(f"{a:14s} {rho:+9.3f} {p:7.3f} | {rt:+.2f} / {rv:+.2f}{flag}")
    # composite "bad rhetoric" score: extrapolation + story + urgency + crowd - hedging
    comp = X[:, 0] + X[:, 1] + X[:, 2] + X[:, 4] - X[:, 3]
    hi = y[comp >= np.median(comp)]
    lo = y[comp < np.median(comp)]
    t, p = stats.ttest_ind(hi, lo, equal_var=False)
    print(f"\nzła retoryka (górna połowa score): alfa {hi.mean():+.2f}pp vs "
          f"dolna {lo.mean():+.2f}pp  t={t:.2f} p={p:.3f}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "lexical":
        mode_lexical(sys.argv[2], sys.argv[3])
    elif mode == "label":
        mode_label(sys.argv[2], sys.argv[3])
    elif mode == "analyze":
        mode_analyze(sys.argv[2], sys.argv[3], sys.argv[4])
