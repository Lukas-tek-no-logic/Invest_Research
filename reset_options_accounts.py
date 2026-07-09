#!/usr/bin/env python3
"""Hard reset of the 4 options accounts to a clean $10K baseline.

History before 2026-07-09 is tainted by the credit-spread P/L sign bug
(fixed in commit 2b50177) — balances show fictional gains. This script:

  1. Backs up audit.db and dumps the accounts' Ghostfolio orders to JSON.
  2. Deletes their rows from options_positions + options_legs in audit.db
     (decision_log stays — it is the audit trail).
  3. Deletes ALL Ghostfolio orders of those accounts (GF_WHEEL-*/GF_SPREAD-*
     synthetics and assignment stock orders alike). Filtering is strictly by
     accountId, never by symbol. Full-account wipe preserves the GF_
     open/close pairing invariant.
  4. Resets each Ghostfolio account balance to $10,000.

DRY-RUN by default — pass --execute to apply.

Usage on LXC 110:
    cd /opt/invest-orchestrator && set -a && source .env && set +a
    .venv/bin/python reset_options_accounts.py            # dry-run
    .venv/bin/python reset_options_accounts.py --execute  # apply
"""
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orchestrator.src.ghostfolio_client import GhostfolioClient

EXECUTE = "--execute" in sys.argv
DB_PATH = Path(__file__).resolve().parent / "orchestrator" / "data" / "audit.db"
RESET_BALANCE = 10_000.0

# account_key → ghostfolio_account_id (from config.yaml)
ACCOUNTS = {
    "options_spreads":          "4edf44b5-8181-4df5-84ad-b6cbd275638b",
    "options_spreads_nemotron": "6c7e408b-90bb-4046-a43f-e2157c652972",
    "spreads_qwen":             "e33ef329-aa85-433e-9658-e19501f1b805",
    "spreads_nemotron":         "dfb1e45f-d274-4129-bda8-010d346a5e3a",
}

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
mode = "EXECUTE" if EXECUTE else "DRY-RUN"
print(f"=== Reset kont opcyjnych [{mode}] ===\n")

# ── 1. Backups ───────────────────────────────────────────────────────────────
gf = GhostfolioClient()
orders = gf.list_orders()
if isinstance(orders, dict):
    orders = orders.get("activities", [])

target_ids = set(ACCOUNTS.values())
target_orders = [o for o in orders if o.get("accountId") in target_ids]

if EXECUTE:
    db_backup = DB_PATH.with_name(f"audit.db.bak-{ts}")
    shutil.copy2(DB_PATH, db_backup)
    orders_backup = DB_PATH.with_name(f"ghostfolio_orders_backup-{ts}.json")
    orders_backup.write_text(json.dumps(target_orders, indent=1, default=str))
    print(f"Backup DB:     {db_backup}")
    print(f"Backup orders: {orders_backup} ({len(target_orders)} zleceń)\n")
else:
    print(f"(dry-run: backupy zostaną utworzone przy --execute)\n")

# ── 2. SQLite: options_positions + options_legs ─────────────────────────────
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
keys = tuple(ACCOUNTS.keys())
ph = ",".join("?" * len(keys))

for key in keys:
    n_pos = conn.execute(
        "SELECT COUNT(*) FROM options_positions WHERE account_key=?", (key,)
    ).fetchone()[0]
    n_legs = conn.execute(
        "SELECT COUNT(*) FROM options_legs WHERE position_id IN "
        "(SELECT id FROM options_positions WHERE account_key=?)", (key,)
    ).fetchone()[0]
    print(f"{key}: {n_pos} pozycji, {n_legs} nóg do usunięcia z audit.db")

if EXECUTE:
    conn.execute(
        f"DELETE FROM options_legs WHERE position_id IN "
        f"(SELECT id FROM options_positions WHERE account_key IN ({ph}))", keys
    )
    cur = conn.execute(
        f"DELETE FROM options_positions WHERE account_key IN ({ph})", keys
    )
    conn.commit()
    print(f"\nUsunięto {cur.rowcount} pozycji z audit.db")
conn.close()

# ── 3. Ghostfolio: delete all orders of the 4 accounts ──────────────────────
id_to_key = {v: k for k, v in ACCOUNTS.items()}
per_account: dict[str, int] = {}
for o in target_orders:
    per_account[id_to_key[o["accountId"]]] = per_account.get(id_to_key[o["accountId"]], 0) + 1

print()
for key in ACCOUNTS:
    print(f"{key}: {per_account.get(key, 0)} zleceń Ghostfolio do usunięcia")

if EXECUTE:
    deleted, failed = 0, 0
    for o in target_orders:
        try:
            gf.delete_order(o["id"])
            deleted += 1
        except Exception as e:
            failed += 1
            print(f"  BŁĄD delete_order {o['id']} ({o.get('SymbolProfile', {}).get('symbol', '?')}): {e}")
    print(f"\nUsunięto {deleted} zleceń, błędów: {failed}")
    if failed:
        print("UWAGA: częściowe usunięcie — sprawdź błędy powyżej przed ponownym uruchomieniem.")

# ── 4. Reset balances ────────────────────────────────────────────────────────
print()
accounts_raw = gf.list_accounts()
if isinstance(accounts_raw, dict):
    accounts_raw = accounts_raw.get("accounts", [])
current = {a["id"]: a.get("balance", 0) for a in accounts_raw if isinstance(a, dict)}

for key, acct_id in ACCOUNTS.items():
    bal = current.get(acct_id, "?")
    print(f"{key}: saldo ${bal} -> ${RESET_BALANCE:,.0f}")
    if EXECUTE:
        gf.update_account(acct_id, balance=RESET_BALANCE)

print(f"\n=== {mode} zakończony ===")
if not EXECUTE:
    print("Uruchom z --execute aby zastosować.")
