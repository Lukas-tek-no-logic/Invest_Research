#!/usr/bin/env python3
"""Check today's trades, cycle status and account balances."""
import sqlite3
import yaml
import sys
from datetime import date

sys.path.insert(0, "/opt/invest-orchestrator")
from orchestrator.src.ghostfolio_client import GhostfolioClient

today = date.today().isoformat()

# ── Audit log ──────────────────────────────────────────────────────────────
db = sqlite3.connect("/opt/invest-orchestrator/orchestrator/data/audit.db")
db.row_factory = sqlite3.Row

cycles = db.execute("""
    SELECT account_name, timestamp, success, actions_count, rejected_count, error
    FROM decision_log
    WHERE date(timestamp) = ?
    ORDER BY timestamp
""", (today,)).fetchall()

print(f"=== CYKLE DZIŚ ({today}) ===")
if not cycles:
    print("  Brak cykli dzisiaj")
for c in cycles:
    ts = str(c["timestamp"])[:16]
    name = str(c["account_name"])[:28]
    status = "OK" if c["success"] else "ERR"
    trades = c["actions_count"] or 0
    rejected = c["rejected_count"] or 0
    err = c["error"] or ""
    err_str = f"  ERR:{err[:60]}" if err else ""
    print(f"  {ts}  {name:28s}  {status:5s}  trades={trades}  rej={rejected}{err_str}")

# ── Opcje — otwarte pozycje ─────────────────────────────────────────────────
print()
print("=== OTWARTE POZYCJE OPCYJNE ===")
try:
    positions = db.execute("""
        SELECT account_key, symbol, spread_type, sell_strike, buy_strike,
               expiration_date, contracts, entry_debit, current_value,
               current_pl, dte, status
        FROM options_positions
        WHERE lower(status) = 'open'
        ORDER BY account_key, symbol
    """).fetchall()
    if not positions:
        print("  Brak otwartych pozycji")
    for p in positions:
        strike = p["sell_strike"]
        buy = p["buy_strike"]
        stype = p["spread_type"]
        credit = abs(p["entry_debit"] or 0)
        cur = p["current_value"] or 0
        pl = p["current_pl"] or 0
        dte = p["dte"]
        strikes_str = f"{buy}/{strike}" if buy else str(strike)
        dte_str = f"  DTE={dte}" if dte is not None else ""
        pl_str = f"  P/L=${pl:+.2f}"
        print(f"  [{p['account_key']}] {p['symbol']:6s} {stype:18s} strikes={strikes_str:9s} exp={p['expiration_date']}  entry=${credit:.2f}{pl_str}{dte_str}")
except Exception as e:
    print(f"  Błąd: {e}")

# ── Opcje — transakcje z dziś ───────────────────────────────────────────────
print()
print("=== TRANSAKCJE OPCYJNE DZIŚ ===")
try:
    opt_trades = db.execute("""
        SELECT account_key, symbol, spread_type, sell_strike, buy_strike,
               entry_debit, realized_pl, close_date, entry_date, status
        FROM options_positions
        WHERE date(entry_date) = ? OR date(close_date) = ?
        ORDER BY coalesce(entry_date, close_date)
    """, (today, today)).fetchall()
    if not opt_trades:
        print("  Brak transakcji opcyjnych dziś")
    for t in opt_trades:
        action = "OPEN " if str(t["entry_date"] or "")[:10] == today else "CLOSE"
        buy = t["buy_strike"]
        strikes_str = f"{buy}/{t['sell_strike']}" if buy else str(t["sell_strike"])
        pl_str = f"  P/L=${t['realized_pl']:.2f}" if t["realized_pl"] is not None else ""
        print(f"  {action} [{t['account_key']}] {t['symbol']:6s} {t['spread_type']:18s} strikes={strikes_str:9s}  entry=${abs(t['entry_debit'] or 0):.2f}{pl_str}")
except Exception as e:
    print(f"  Błąd: {e}")

db.close()

# ── Ghostfolio balances ─────────────────────────────────────────────────────
print()
print("=== SALDA KONT (Ghostfolio) ===")
try:
    with open("/opt/invest-orchestrator/orchestrator/data/config.yaml") as f:
        config = yaml.safe_load(f)

    gf = GhostfolioClient()
    accts_raw = gf.list_accounts()
    accts = accts_raw.get("accounts", []) if isinstance(accts_raw, dict) else accts_raw
    bal_by_id = {a["id"]: a for a in accts if isinstance(a, dict)}

    for key, acct in config.get("accounts", {}).items():
        aid = acct.get("ghostfolio_account_id") or acct.get("account_id")
        if not aid:
            continue
        a = bal_by_id.get(aid, {})
        bal   = float(a.get("balance", 0) or 0)
        val   = float(a.get("valueInBaseCurrency", 0) or 0)
        init  = float(acct.get("initial_budget", 10000))
        strat = acct.get("strategy", "")
        if strat in ("wheel", "vertical_spreads"):
            # UWAGA: `balance` to pole ustawiane RĘCZNIE — zamrożone od resetu
            # 2026-07-09, więc `balance - init` zawsze pokazywało $0.00 i wynik
            # 40% kapitału był niewidoczny. Raportujemy dwa niezależne kanały:
            # wycenę Ghostfolio i zrealizowany wynik z audit.db (od baseline).
            pl_gf = val - init
            adb = sqlite3.connect("/opt/invest-orchestrator/orchestrator/data/audit.db")
            row = adb.execute(
                """SELECT COALESCE(SUM(realized_pl), 0), COUNT(*)
                   FROM options_positions
                   WHERE account_key = ? AND status != 'open'
                     AND entry_date >= '2026-08-19'""",
                (key,),
            ).fetchone()
            adb.close()
            realized, n_closed = float(row[0]), int(row[1])
            flag = "  <-- ROZJAZD KANAŁÓW" if abs(pl_gf - realized) > 500 else ""
            print(f"  {key:30s}  gf_val=${val:>10,.2f}  pl_gf=${pl_gf:>+8,.2f}"
                  f"  realized_db=${realized:>+9,.2f} ({n_closed} zamkn.){flag}")
        else:
            pl_val = val - init
            print(f"  {key:30s}  cash=${bal:>10,.2f}  total_val=${val:>10,.2f}  pl=${pl_val:>+8,.2f}")
except Exception as e:
    print(f"  Błąd Ghostfolio: {e}")
