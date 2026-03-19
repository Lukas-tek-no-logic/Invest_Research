"""SQLite-based options position tracker — source of truth for all spreads."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()

DB_PATH = Path("data/audit.db")


@dataclass
class OptionsPosition:
    id: int
    account_key: str
    symbol: str
    spread_type: str          # BULL_CALL, BEAR_PUT
    status: str               # open, closed, expired

    contracts: int
    expiration_date: str      # YYYY-MM-DD
    buy_strike: float
    buy_option_type: str      # call, put
    buy_premium: float        # entry mid-price per share
    sell_strike: float
    sell_option_type: str
    sell_premium: float

    max_profit: float         # total $ for all contracts
    max_loss: float           # total $ for all contracts
    entry_debit: float        # net premium paid (positive = debit)
    entry_date: str

    current_value: float | None = None    # current spread value
    current_pl: float | None = None       # unrealized P&L in $
    current_greeks: dict | None = None    # {net_delta, net_gamma, net_theta, net_vega}
    dte: int | None = None

    close_date: str | None = None
    close_value: float | None = None
    realized_pl: float | None = None
    close_reason: str | None = None

    ghostfolio_open_order_id: str | None = None
    ghostfolio_close_order_id: str | None = None

    buy_contract_symbol: str | None = None
    sell_contract_symbol: str | None = None

    # Wheel state machine
    wheel_state: str | None = None  # CSP_OPEN, ASSIGNED, CC_OPEN, CALLED_AWAY, COMPLETE
    wheel_cost_basis: float | None = None  # effective cost basis after assignment
    wheel_shares: int = 0  # shares held after assignment (100 per contract)
    wheel_parent_id: int | None = None  # links CC back to assigned CSP

    @property
    def is_assigned(self) -> bool:
        return self.wheel_state == "ASSIGNED"

    @property
    def pl_pct(self) -> float | None:
        """Unrealized P&L as % of max loss."""
        if self.current_pl is not None and self.max_loss > 0:
            return round(self.current_pl / self.max_loss * 100, 1)
        return None

    @property
    def profit_captured_pct(self) -> float | None:
        """How much of max profit has been captured (for exits)."""
        if self.current_pl is not None and self.max_profit > 0:
            return round(self.current_pl / self.max_profit * 100, 1)
        return None


class OptionsPositionTracker:
    """CRUD operations for options_positions in SQLite."""

    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
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
            # Wheel state columns (added incrementally)
            for col, typedef in [
                ("wheel_state", "TEXT"),
                ("wheel_cost_basis", "REAL"),
                ("wheel_shares", "INTEGER DEFAULT 0"),
                ("wheel_parent_id", "INTEGER"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE options_positions ADD COLUMN {col} {typedef}")
                except sqlite3.OperationalError:
                    pass  # column already exists

    # ── Write operations ────────────────────────────────────────────────────

    def open_position(
        self,
        account_key: str,
        symbol: str,
        spread_type: str,
        contracts: int,
        expiration_date: str,
        buy_strike: float,
        buy_option_type: str,
        buy_premium: float,
        sell_strike: float,
        sell_option_type: str,
        sell_premium: float,
        max_profit: float,
        max_loss: float,
        entry_debit: float,
        buy_contract_symbol: str | None = None,
        sell_contract_symbol: str | None = None,
        ghostfolio_order_id: str | None = None,
    ) -> int:
        """Insert a new open position. Returns new position ID."""
        today = date.today().isoformat()
        dte = (datetime.strptime(expiration_date, "%Y-%m-%d").date() - date.today()).days

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """INSERT INTO options_positions
                (account_key, symbol, spread_type, status, contracts,
                 expiration_date, buy_strike, buy_option_type, buy_premium,
                 buy_contract_symbol, sell_strike, sell_option_type, sell_premium,
                 sell_contract_symbol, max_profit, max_loss, entry_debit,
                 entry_date, dte, ghostfolio_open_order_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    account_key, symbol, spread_type, "open", contracts,
                    expiration_date, buy_strike, buy_option_type, buy_premium,
                    buy_contract_symbol, sell_strike, sell_option_type, sell_premium,
                    sell_contract_symbol, max_profit, max_loss, entry_debit,
                    today, dte, ghostfolio_order_id,
                ),
            )
            pos_id = cur.lastrowid
        logger.info(
            "options_position_opened",
            id=pos_id, symbol=symbol, spread_type=spread_type,
            expiration=expiration_date, dte=dte,
        )
        return pos_id

    def update_position(
        self,
        position_id: int,
        current_value: float,
        current_pl: float,
        greeks: dict,
        dte: int,
    ) -> None:
        """Update a position's market state (called each cycle for open positions)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE options_positions
                SET current_value=?, current_pl=?, current_greeks=?, dte=?
                WHERE id=?""",
                (current_value, current_pl, json.dumps(greeks), dte, position_id),
            )

    def close_position(
        self,
        position_id: int,
        close_value: float,
        reason: str,
        ghostfolio_order_id: str | None = None,
    ) -> float:
        """Mark position as closed. Returns realized P&L."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT entry_debit, contracts FROM options_positions WHERE id=?",
                (position_id,),
            ).fetchone()

            if not row:
                logger.error("options_position_not_found", id=position_id)
                return 0.0

            entry_debit = row["entry_debit"]
            contracts = row["contracts"]
            # entry_debit < 0 → credit trade (CSP, credit spread): P/L = premium_received - close_cost
            # entry_debit > 0 → debit trade (debit spread):        P/L = close_value - debit_paid
            if entry_debit < 0:
                realized_pl = round((-entry_debit - close_value) * contracts * 100, 2)
            else:
                realized_pl = round((close_value - entry_debit) * contracts * 100, 2)

            conn.execute(
                """UPDATE options_positions
                SET status='closed', close_date=?, close_value=?,
                    realized_pl=?, close_reason=?, ghostfolio_close_order_id=?
                WHERE id=?""",
                (
                    date.today().isoformat(),
                    close_value,
                    realized_pl,
                    reason,
                    ghostfolio_order_id,
                    position_id,
                ),
            )

        logger.info(
            "options_position_closed",
            id=position_id, reason=reason, realized_pl=realized_pl,
        )
        return realized_pl

    def expire_position(self, position_id: int) -> None:
        """Mark position as expired OTM (worthless — keep full premium)."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT entry_debit, contracts, spread_type FROM options_positions WHERE id=?",
                (position_id,),
            ).fetchone()
            if row:
                entry_debit, contracts, spread_type = row
                realized_pl = round(-entry_debit * contracts * 100, 2)
                conn.execute(
                    """UPDATE options_positions
                    SET status='expired', close_date=?, close_value=0,
                        realized_pl=?, close_reason='EXPIRED_OTM',
                        wheel_state='COMPLETE'
                    WHERE id=?""",
                    (date.today().isoformat(), realized_pl, position_id),
                )

    def assign_position(self, position_id: int, stock_price: float) -> float:
        """Mark CSP as assigned — we now own 100 shares per contract.

        Returns the effective cost basis per share (strike - premium received).
        Creates a Ghostfolio BUY order for the assigned shares.
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT sell_strike, entry_debit, contracts FROM options_positions WHERE id=?",
                (position_id,),
            ).fetchone()
            if not row:
                logger.error("assign_position_not_found", id=position_id)
                return 0.0

            strike, entry_debit, contracts = row
            premium_received = abs(entry_debit)
            cost_basis = round(strike - premium_received, 2)
            shares = contracts * 100

            conn.execute(
                """UPDATE options_positions
                SET status='assigned',
                    wheel_state='ASSIGNED',
                    wheel_cost_basis=?,
                    wheel_shares=?,
                    close_date=?,
                    close_reason='ASSIGNED_ITM'
                WHERE id=?""",
                (cost_basis, shares, date.today().isoformat(), position_id),
            )

        logger.info(
            "wheel_csp_assigned",
            id=position_id, strike=strike, premium=premium_received,
            cost_basis=cost_basis, shares=shares, stock_price=stock_price,
        )
        return cost_basis

    def call_away_position(self, position_id: int, cc_strike: float, cc_premium: float) -> float:
        """Mark CC as exercised — shares sold at CC strike. Wheel cycle complete.

        Returns total realized P&L for the full wheel cycle:
          (CC strike - cost_basis + CC premium) × shares
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT wheel_cost_basis, wheel_shares, wheel_parent_id FROM options_positions WHERE id=?",
                (position_id,),
            ).fetchone()
            if not row:
                logger.error("call_away_position_not_found", id=position_id)
                return 0.0

            cost_basis, shares, parent_id = row
            cost_basis = cost_basis or 0
            shares = shares or 100
            # Total P&L = (sell price - cost basis) × shares + CC premium × 100
            realized_pl = round((cc_strike - cost_basis) * shares + cc_premium * shares, 2)

            conn.execute(
                """UPDATE options_positions
                SET status='closed',
                    wheel_state='COMPLETE',
                    close_date=?,
                    realized_pl=?,
                    close_reason='CALLED_AWAY'
                WHERE id=?""",
                (date.today().isoformat(), realized_pl, position_id),
            )

            # Also mark parent CSP as complete if exists
            if parent_id:
                conn.execute(
                    "UPDATE options_positions SET wheel_state='COMPLETE' WHERE id=?",
                    (parent_id,),
                )

        logger.info(
            "wheel_called_away",
            id=position_id, cc_strike=cc_strike, cost_basis=cost_basis,
            cc_premium=cc_premium, realized_pl=realized_pl,
        )
        return realized_pl

    def get_assigned_positions(self, account_key: str) -> list["OptionsPosition"]:
        """Return positions in ASSIGNED state (holding shares, ready for CC)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT * FROM options_positions
                    WHERE account_key=? AND wheel_state='ASSIGNED'
                    ORDER BY close_date ASC""",
                    (account_key,),
                ).fetchall()
            return [_row_to_position(dict(row)) for row in rows]
        except Exception as e:
            logger.error("get_assigned_failed", error=str(e))
            return []

    # ── Read operations ─────────────────────────────────────────────────────

    def get_active_positions(self, account_key: str) -> list[OptionsPosition]:
        """Return all open positions for an account."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT * FROM options_positions
                    WHERE account_key=? AND status='open'
                    ORDER BY expiration_date ASC""",
                    (account_key,),
                ).fetchall()
            return [_row_to_position(dict(row)) for row in rows]
        except Exception as e:
            logger.error("options_get_active_failed", error=str(e))
            return []

    def get_position_history(
        self,
        account_key: str,
        limit: int = 20,
        status: str | None = None,
    ) -> list[dict]:
        """Return historical positions for an account."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                if status:
                    rows = conn.execute(
                        """SELECT * FROM options_positions
                        WHERE account_key=? AND status=?
                        ORDER BY entry_date DESC LIMIT ?""",
                        (account_key, status, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT * FROM options_positions
                        WHERE account_key=?
                        ORDER BY entry_date DESC LIMIT ?""",
                        (account_key, limit),
                    ).fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error("options_get_history_failed", error=str(e))
            return []

    def get_position_by_id(self, position_id: int) -> OptionsPosition | None:
        """Fetch a single position by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM options_positions WHERE id=?",
                    (position_id,),
                ).fetchone()
            return _row_to_position(dict(row)) if row else None
        except Exception as e:
            logger.error("options_get_by_id_failed", error=str(e))
            return None

    def get_total_realized_pl(self, account_key: str) -> float:
        """Sum of all realized P&L for closed/expired positions with real (non-dry-run) closes."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    """SELECT COALESCE(SUM(realized_pl), 0) FROM options_positions
                    WHERE account_key=? AND status IN ('closed', 'expired')
                    AND ghostfolio_close_order_id IS NOT NULL
                    AND ghostfolio_close_order_id != 'DRY_RUN'""",
                    (account_key,),
                ).fetchone()
            return float(row[0]) if row else 0.0
        except Exception:
            return 0.0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _row_to_position(row: dict) -> OptionsPosition:
    greeks_raw = row.get("current_greeks")
    greeks = json.loads(greeks_raw) if isinstance(greeks_raw, str) else greeks_raw

    return OptionsPosition(
        id=row["id"],
        account_key=row["account_key"],
        symbol=row["symbol"],
        spread_type=row["spread_type"],
        status=row["status"],
        contracts=row["contracts"],
        expiration_date=row["expiration_date"],
        buy_strike=row["buy_strike"],
        buy_option_type=row["buy_option_type"],
        buy_premium=row["buy_premium"],
        sell_strike=row["sell_strike"],
        sell_option_type=row["sell_option_type"],
        sell_premium=row["sell_premium"],
        max_profit=row["max_profit"],
        max_loss=row["max_loss"],
        entry_debit=row["entry_debit"],
        entry_date=row["entry_date"],
        current_value=row.get("current_value"),
        current_pl=row.get("current_pl"),
        current_greeks=greeks,
        dte=row.get("dte"),
        close_date=row.get("close_date"),
        close_value=row.get("close_value"),
        realized_pl=row.get("realized_pl"),
        close_reason=row.get("close_reason"),
        ghostfolio_open_order_id=row.get("ghostfolio_open_order_id"),
        ghostfolio_close_order_id=row.get("ghostfolio_close_order_id"),
        buy_contract_symbol=row.get("buy_contract_symbol"),
        sell_contract_symbol=row.get("sell_contract_symbol"),
        wheel_state=row.get("wheel_state"),
        wheel_cost_basis=row.get("wheel_cost_basis"),
        wheel_shares=row.get("wheel_shares", 0) or 0,
        wheel_parent_id=row.get("wheel_parent_id"),
    )
