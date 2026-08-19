#!/usr/bin/env python3
"""Hard reset of all trading accounts to their initial budget.

Baseline reset 2026-08-19: after the data-gate + measurement-rig deploy the
pre-reset history mixes fee-less and fee-aware accounting, phantom shorts and
three disagreeing options channels. A clean $10k start makes the new numbers
comparable from day one.

What it does per Ghostfolio account found in config.yaml:
  1. Deletes every Ghostfolio order (activities backed up to JSON first).
  2. Sets the account balance to its initial_budget.
  3. Archives open rows in options_positions (status='reset', no fabricated
     close values) and stamps close_reason with the reset tag.

Backups (written before any change, script aborts if either fails):
  - orchestrator/data/audit.db            -> audit.db.bak-<STAMP>
  - all Ghostfolio activities             -> ghostfolio_orders_backup_<STAMP>.json

Usage (inside container 110):
    cd /opt/invest-orchestrator/orchestrator && set -a && . ../.env && set +a
    ../.venv/bin/python ../reset_accounts.py            # dry-run: prints the plan
    ../.venv/bin/python ../reset_accounts.py --apply    # actually resets
"""
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "orchestrator"))
import yaml

from src.ghostfolio_client import GhostfolioClient

APPLY = "--apply" in sys.argv
ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "orchestrator" / "data" / "audit.db"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESET_TAG = f"HARD_RESET_{STAMP}"

with open(ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)

default_budget = float(config.get("defaults", {}).get("initial_budget", 10000))
accounts = {
    key: acct for key, acct in config.get("accounts", {}).items()
    if acct.get("ghostfolio_account_id") and acct["ghostfolio_account_id"] != "TBD"
    and acct.get("cycle_type") != "research"
}
gf_ids = {acct["ghostfolio_account_id"]: key for key, acct in accounts.items()}

gf = GhostfolioClient()
orders = gf.list_orders()
by_account: dict[str, list[dict]] = {}
for o in orders:
    aid = (o.get("account") or {}).get("id") or o.get("accountId")
    if aid in gf_ids:
        by_account.setdefault(aid, []).append(o)

print(f"{'account':32s} {'orders':>7s} {'-> balance':>11s}")
for aid, key in gf_ids.items():
    budget = float(accounts[key].get("initial_budget", default_budget))
    print(f"{key:32s} {len(by_account.get(aid, [])):>7d} {budget:>11,.2f}")

with sqlite3.connect(DB_PATH) as conn:
    open_opts = conn.execute(
        "SELECT COUNT(*) FROM options_positions WHERE status='open'"
    ).fetchone()[0]
print(f"\nopen options_positions to archive: {open_opts}")

if not APPLY:
    print("\nDRY-RUN — nothing changed. Re-run with --apply to execute.")
    sys.exit(0)

# ── backups (abort on failure) ───────────────────────────────────────────────
bak_db = DB_PATH.with_name(f"audit.db.bak-{STAMP}")
shutil.copy2(DB_PATH, bak_db)
bak_orders = ROOT / f"ghostfolio_orders_backup_{STAMP}.json"
bak_orders.write_text(json.dumps(orders, indent=2, default=str))
print(f"\nbackups: {bak_db}\n         {bak_orders} ({len(orders)} orders)")

# ── 1. delete orders ─────────────────────────────────────────────────────────
deleted = failed = 0
for aid, accs_orders in by_account.items():
    for o in accs_orders:
        try:
            gf.delete_order(o["id"])
            deleted += 1
        except Exception as e:
            failed += 1
            print(f"  DELETE FAILED {o.get('id')}: {e}")
print(f"orders deleted: {deleted}, failed: {failed}")

# ── 2. reset balances ────────────────────────────────────────────────────────
for aid, key in gf_ids.items():
    budget = float(accounts[key].get("initial_budget", default_budget))
    try:
        gf.update_account(aid, balance=budget)
        print(f"balance set: {key} -> ${budget:,.2f}")
    except Exception as e:
        print(f"  BALANCE FAILED {key}: {e}")

# ── 3. archive open options positions ────────────────────────────────────────
with sqlite3.connect(DB_PATH) as conn:
    n = conn.execute(
        """UPDATE options_positions
           SET status='reset', close_date=?, close_reason=?
           WHERE status='open'""",
        (datetime.now().strftime("%Y-%m-%d"), RESET_TAG),
    ).rowcount
print(f"options_positions archived: {n}")
print(f"\nDONE — tag {RESET_TAG}. New measurement baseline: {datetime.now():%Y-%m-%d}.")
