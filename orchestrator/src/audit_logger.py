"""Audit logger: saves full decision cycle as JSON + maintains SQLite summary."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()

LOGS_DIR = Path("logs")
DB_PATH = Path("data/audit.db")


class AuditLogger:
    """Logs every decision cycle with full context for auditability."""

    def __init__(self, logs_dir: Path | str = LOGS_DIR, db_path: Path | str = DB_PATH):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite summary table and options positions table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decision_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    account_key TEXT NOT NULL,
                    account_name TEXT NOT NULL,
                    model TEXT NOT NULL,
                    market_regime TEXT,
                    portfolio_outlook TEXT,
                    confidence REAL,
                    actions_count INTEGER DEFAULT 0,
                    forced_actions_count INTEGER DEFAULT 0,
                    rejected_count INTEGER DEFAULT 0,
                    portfolio_value REAL,
                    portfolio_pl_pct REAL,
                    cash REAL,
                    log_file TEXT,
                    success INTEGER DEFAULT 1,
                    error TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS options_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_key TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    spread_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    contracts INTEGER NOT NULL,
                    expiration_date TEXT NOT NULL,
                    buy_strike REAL NOT NULL,
                    buy_option_type TEXT NOT NULL,
                    buy_premium REAL NOT NULL,
                    buy_contract_symbol TEXT,
                    sell_strike REAL NOT NULL,
                    sell_option_type TEXT NOT NULL,
                    sell_premium REAL NOT NULL,
                    sell_contract_symbol TEXT,
                    max_profit REAL NOT NULL,
                    max_loss REAL NOT NULL,
                    entry_debit REAL NOT NULL,
                    entry_date TEXT NOT NULL,
                    current_value REAL,
                    current_pl REAL,
                    current_greeks TEXT,
                    dte INTEGER,
                    close_date TEXT,
                    close_value REAL,
                    realized_pl REAL,
                    close_reason TEXT,
                    ghostfolio_open_order_id TEXT,
                    ghostfolio_close_order_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def log_cycle(
        self,
        account_key: str,
        account_name: str,
        model: str,
        pass1_messages: list[dict] | None = None,
        pass1_response: dict | None = None,
        pass2_messages: list[dict] | None = None,
        pass2_response: dict | None = None,
        risk_modifications: list[str] | None = None,
        risk_warnings: list[str] | None = None,
        forced_actions: list[dict] | None = None,
        rejected_actions: list[dict] | None = None,
        executed_trades: list[dict] | None = None,
        portfolio_before: dict | None = None,
        portfolio_after: dict | None = None,
        error: str | None = None,
        fees_paid: float = 0.0,
    ) -> str:
        """Log a full decision cycle.

        Returns the path to the log file.
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M%S")

        log_entry = {
            "timestamp": now.isoformat(),
            "account_key": account_key,
            "account_name": account_name,
            "model": model,
            "pass1": {
                "messages": pass1_messages,
                "response": pass1_response,
            },
            "pass2": {
                "messages": pass2_messages,
                "response": pass2_response,
            },
            "risk_manager": {
                "modifications": risk_modifications or [],
                "warnings": risk_warnings or [],
                "forced_actions": forced_actions or [],
                "rejected_actions": rejected_actions or [],
            },
            "executed_trades": executed_trades or [],
            "fees_paid": fees_paid,
            "portfolio_before": portfolio_before,
            "portfolio_after": portfolio_after,
            "error": error,
        }

        # Write JSON log file
        log_file = self.logs_dir / f"{date_str}_{account_key}_{time_str}.json"
        with open(log_file, "w") as f:
            json.dump(log_entry, f, indent=2, default=str)

        # Write summary to SQLite
        analysis = pass1_response if isinstance(pass1_response, dict) else {}
        decision = pass2_response if isinstance(pass2_response, dict) else {}
        p_before = portfolio_before if isinstance(portfolio_before, dict) else {}
        # Use portfolio_after for display values — it reflects post-trade state
        # (cash updated; total_value still approximate until next Ghostfolio sync)
        p_after = portfolio_after if isinstance(portfolio_after, dict) else p_before

        # Portfolio valuation can fail (e.g. Ghostfolio valuation endpoints 500 due to a
        # stalled snapshot job). When that happens get_portfolio_state() returns an empty
        # state with total_value=0, which must NOT be persisted — a real account always has
        # positive cash, so a 0/None here means "unknown", not "wiped out". Carry forward
        # the last known good values so the dashboard keeps showing the last valid snapshot
        # instead of $0.00 cards.
        pv = p_after.get("total_value", p_before.get("total_value"))
        pl_pct = p_after.get("total_pl_pct", p_before.get("total_pl_pct"))
        cash_val = p_after.get("cash", p_before.get("cash"))
        if pv is None or pv <= 0:
            last_good = self._last_known_valuation(account_key)
            if last_good is not None:
                pv, pl_pct, cash_val = last_good
                logger.warning(
                    "audit_valuation_unavailable_carried_forward",
                    account=account_key, carried_value=pv,
                )

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO decision_log
                    (timestamp, account_key, account_name, model, market_regime,
                     portfolio_outlook, confidence, actions_count, forced_actions_count,
                     rejected_count, portfolio_value, portfolio_pl_pct, cash,
                     log_file, success, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        now.isoformat(),
                        account_key,
                        account_name,
                        model,
                        analysis.get("market_regime"),
                        decision.get("portfolio_outlook"),
                        decision.get("confidence"),
                        len(decision.get("actions", [])),
                        len(forced_actions or []),
                        len(rejected_actions or []),
                        pv,
                        pl_pct,
                        cash_val,
                        str(log_file),
                        0 if error else 1,
                        error,
                    ),
                )
        except Exception as e:
            logger.error("audit_db_write_failed", error=str(e))

        logger.info("audit_log_written", file=str(log_file), account=account_key)
        return str(log_file)

    def _last_known_valuation(
        self, account_key: str
    ) -> tuple[float, float | None, float | None] | None:
        """Return (portfolio_value, portfolio_pl_pct, cash) from the most recent cycle
        that had a positive portfolio_value, or None if there is no such row.

        Used to carry forward the last valid snapshot when the current cycle could not
        value the portfolio (e.g. Ghostfolio valuation endpoints down)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    """SELECT portfolio_value, portfolio_pl_pct, cash FROM decision_log
                    WHERE account_key = ? AND portfolio_value IS NOT NULL
                      AND portfolio_value > 0
                    ORDER BY timestamp DESC LIMIT 1""",
                    (account_key,),
                ).fetchone()
            if row is None:
                return None
            return float(row[0]), row[1], row[2]
        except Exception as e:
            logger.error("audit_last_valuation_lookup_failed", error=str(e))
            return None

    def get_decision_history(
        self,
        account_key: str,
        limit: int = 4,
    ) -> list[dict]:
        """Get recent decision history for prompt injection."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT timestamp, log_file FROM decision_log
                    WHERE account_key = ? AND success = 1
                    ORDER BY timestamp DESC LIMIT ?""",
                    (account_key, limit),
                ).fetchall()

            history = []
            for row in reversed(rows):
                log_file = row["log_file"]
                try:
                    with open(log_file) as f:
                        entry = json.load(f)
                    decision = entry.get("pass2", {}).get("response", {})
                    if not isinstance(decision, dict):
                        decision = {}
                    actions_raw = decision.get("actions", [])
                    if not isinstance(actions_raw, list):
                        actions_raw = []
                    trades = entry.get("executed_trades", [])

                    # Match results to actions
                    actions = []
                    for a in actions_raw:
                        if not isinstance(a, dict):
                            continue
                        action_data = {
                            "type": a.get("type"),
                            "symbol": a.get("symbol"),
                            "amount_usd": a.get("amount_usd", 0),
                            "thesis": a.get("thesis", ""),
                        }
                        # Find matching trade result
                        for t in trades:
                            if (t.get("symbol") == a.get("symbol") and
                                    t.get("type") == a.get("type")):
                                action_data["result_pct"] = t.get("result_pct")
                                break
                        actions.append(action_data)

                    history.append({
                        "date": row["timestamp"][:10],
                        "outlook": decision.get("portfolio_outlook", "Unknown"),
                        "confidence": decision.get("confidence", "N/A"),
                        "actions": actions,
                        "hold_reason": decision.get("reasoning", "")[:100] if not actions else "",
                    })
                except (FileNotFoundError, json.JSONDecodeError, AttributeError, TypeError, KeyError):
                    continue

            return history
        except Exception as e:
            logger.error("decision_history_fetch_failed", error=str(e))
            return []

    def get_recent_logs(
        self,
        account_key: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Get recent log summaries for dashboard display."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                if account_key:
                    rows = conn.execute(
                        """SELECT * FROM decision_log
                        WHERE account_key = ?
                        ORDER BY timestamp DESC LIMIT ?""",
                        (account_key, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT * FROM decision_log
                        ORDER BY timestamp DESC LIMIT ?""",
                        (limit,),
                    ).fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error("recent_logs_fetch_failed", error=str(e))
            return []

    def get_log_detail(self, log_file: str) -> dict | None:
        """Read full log file for detailed view."""
        try:
            with open(log_file) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error("log_detail_read_failed", file=log_file, error=str(e))
            return None
