"""Execute Wheel Strategy trades: SQLite position tracking + Ghostfolio cash flows.

Handles:
  execute_sell_csp()  — open a cash-secured put
  execute_sell_cc()   — open a covered call against an assigned stock
  execute_close()     — buy back an existing CSP or CC position
  execute_closes()    — bulk-close convenience wrapper (used by main.py)
  execute_opens()     — bulk-open wrapper; routes SELL_CSP vs SELL_CC
  execute_rolls()     — no-op for wheel (returns empty list, required by main.py)
  update_active_positions() — refresh DTE / P&L for held positions

Ghostfolio integration:
  CSP open  → BUY  "WHEEL-{SYM}-CSP-{YYYYMMDD}-{strike}P"  unit_price=premium
  CC  open  → BUY  "WHEEL-{SYM}-CC-{YYYYMMDD}-{strike}C"   unit_price=premium
  Close     → SELL same symbol, unit_price=close_value
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime

import structlog

from ..ghostfolio_client import GhostfolioClient
from ..market_data import MarketDataProvider
from .data import get_current_option_price
from .positions import OptionsPosition, OptionsPositionTracker
# Import from deployed module names.
# When placed in the options package, adjust these to match actual filenames:
#   wheel_opt_parser.py  → decision_parser.py  (or keep as wheel_decision_parser.py)
#   wheel_opt_selector.py → selector.py        (or keep as wheel_selector.py)
from .decision_parser import WheelAction
from .selector import SelectedCSP, SelectedCC, select_csp, select_cc

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Result dataclass (same shape as original OptionsTradeResult)
# ---------------------------------------------------------------------------

@dataclass
class OptionsTradeResult:
    action: str              # "OPEN_CSP" | "OPEN_CC" | "CLOSE" | "UPDATE" | "ROLL"
    symbol: str
    spread_type: str         # "CASH_SECURED_PUT" | "COVERED_CALL"
    position_id: int | None
    success: bool
    realized_pl: float | None = None
    error: str = ""
    ghostfolio_order_id: str | None = None


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class OptionsExecutor:
    """Execute open/close decisions for wheel strategy positions."""

    def __init__(
        self,
        ghostfolio: GhostfolioClient,
        market_data: MarketDataProvider,
        tracker: OptionsPositionTracker,
        account_id: str,
        risk_profile: dict,
        dry_run: bool = False,
        account_key: str | None = None,
    ):
        self.ghostfolio = ghostfolio
        self.market_data = market_data
        self.tracker = tracker
        self.account_id = account_id
        self.account_key = account_key or account_id
        self.risk_profile = risk_profile
        self.dry_run = dry_run

    # ── Public interface (called by main.py) ──────────────────────────────────

    def execute_opens(
        self,
        opens: list[WheelAction],
        active_positions: list[OptionsPosition] | None = None,
    ) -> list[OptionsTradeResult]:
        """Route each open action to the correct executor (CSP or CC).

        Args:
            opens:            Approved open actions from the risk manager.
            active_positions: Current active positions; passed to execute_sell_cc()
                              so it can look up the parent CSP cost basis.
        """
        seen: set[str] = set()
        results = []
        for action in opens:
            if action.symbol in seen:
                logger.warning("duplicate_options_open_skipped", type=action.type, symbol=action.symbol)
                continue
            seen.add(action.symbol)
            if action.type == "SELL_CSP":
                results.append(self.execute_sell_csp(action))
            elif action.type == "SELL_CC":
                results.append(self.execute_sell_cc(action, active_positions=active_positions))
            else:
                logger.warning("wheel_executor_unknown_open_type", type=action.type, symbol=action.symbol)
        return results

    def execute_closes(
        self,
        closes: list[WheelAction],
        active_positions: list[OptionsPosition],
    ) -> list[OptionsTradeResult]:
        """Close a list of positions (LLM-requested or forced)."""
        results = []
        pos_map = {p.id: p for p in active_positions}
        for action in closes:
            pid = action.position_id
            pos = pos_map.get(pid) if pid is not None else None
            if pos is None:
                results.append(OptionsTradeResult(
                    action="CLOSE", symbol=action.symbol, spread_type="?",
                    position_id=pid, success=False,
                    error=f"Position {pid} not found in active positions",
                ))
                continue
            results.append(self._close_position(pos, action.reason))
        return results

    def execute_rolls(
        self,
        rolls: list,
        active_positions: list[OptionsPosition],
    ) -> list[OptionsTradeResult]:
        """Wheel strategy has no roll concept — always returns empty list."""
        if rolls:
            logger.warning("wheel_executor_rolls_ignored", count=len(rolls))
        return []

    def update_active_positions(
        self,
        active_positions: list[OptionsPosition],
    ) -> list[OptionsTradeResult]:
        """Refresh DTE, current value, and P&L for all held positions."""
        results = []
        today = date.today()
        for pos in active_positions:
            results.append(self._update_position_state(pos, today))
        return results

    # ── CSP execution ─────────────────────────────────────────────────────────

    def execute_sell_csp(self, action: WheelAction) -> OptionsTradeResult:
        """Select strike and record a new cash-secured put."""
        try:
            target_delta = self.risk_profile.get("csp_target_delta", 0.30)
            dte_min = self.risk_profile.get("csp_dte_min", 21)
            dte_max = self.risk_profile.get("csp_dte_max", 45)
            min_yield = self.risk_profile.get("min_premium_yield_pct", 5.0)

            csp = select_csp(
                symbol=action.symbol,
                contracts=action.contracts,
                target_delta=target_delta,
                dte_min=dte_min,
                dte_max=dte_max,
                min_premium_yield_pct=min_yield,
            )
            if csp is None:
                return OptionsTradeResult(
                    action="OPEN_CSP", symbol=action.symbol,
                    spread_type="CASH_SECURED_PUT", position_id=None,
                    success=False, error="CSP strike selection failed (no suitable chain)",
                )

            # Ghostfolio: record premium collected as BUY of synthetic asset
            ghostfolio_order_id = None
            if not self.dry_run:
                ghostfolio_order_id = self._ghostfolio_open_csp(csp, action.contracts)
            else:
                ghostfolio_order_id = "DRY_RUN"
                logger.info(
                    "wheel_dry_run_sell_csp",
                    symbol=csp.symbol, strike=csp.strike,
                    expiration=csp.expiration, premium=csp.premium,
                    contracts=action.contracts,
                )

            # Max profit = premium collected × 100 × contracts
            max_profit = round(csp.premium * 100 * action.contracts, 2)
            # Max loss = strike price × 100 × contracts (stock falls to 0)
            max_loss = round(csp.strike * 100 * action.contracts, 2)

            # SQLite record — we store the CSP as a single-leg "spread":
            #   sell_strike = put strike  (the leg we sold)
            #   sell_option_type = "put"
            #   buy_strike = 0 (no long leg)
            #   entry_debit = negative (we received premium, not paid debit)
            pos_id = self.tracker.open_position(
                account_key=self.account_key,
                symbol=csp.symbol,
                spread_type="CASH_SECURED_PUT",
                contracts=action.contracts,
                expiration_date=csp.expiration,
                buy_strike=0.0,          # no long leg
                buy_option_type="put",
                buy_premium=0.0,
                sell_strike=csp.strike,
                sell_option_type="put",
                sell_premium=csp.premium,
                max_profit=max_profit,
                max_loss=max_loss,
                entry_debit=-csp.premium,   # negative = credit received
                buy_contract_symbol=None,
                sell_contract_symbol=csp.contract_symbol,
                ghostfolio_order_id=ghostfolio_order_id,
            )

            # Initial state update (no P&L yet)
            self.tracker.update_position(
                pos_id,
                current_value=csp.premium,
                current_pl=0.0,
                greeks={"net_delta": csp.delta, "net_gamma": 0.0,
                        "net_theta": 0.0, "net_vega": 0.0},
                dte=csp.dte,
            )
            # Set wheel state
            with sqlite3.connect(self.tracker.db_path) as conn:
                conn.execute(
                    "UPDATE options_positions SET wheel_state='CSP_OPEN' WHERE id=?",
                    (pos_id,),
                )

            logger.info(
                "wheel_csp_opened",
                pos_id=pos_id, symbol=csp.symbol,
                strike=csp.strike, expiration=csp.expiration,
                premium=csp.premium, delta=round(csp.delta, 3),
                contracts=action.contracts, max_profit=max_profit,
            )

            return OptionsTradeResult(
                action="OPEN_CSP", symbol=csp.symbol,
                spread_type="CASH_SECURED_PUT",
                position_id=pos_id, success=True,
                ghostfolio_order_id=ghostfolio_order_id,
            )

        except Exception as e:
            logger.error("wheel_sell_csp_failed", symbol=action.symbol, error=str(e), exc_info=True)
            return OptionsTradeResult(
                action="OPEN_CSP", symbol=action.symbol,
                spread_type="CASH_SECURED_PUT",
                position_id=None, success=False, error=str(e),
            )

    # ── CC execution ──────────────────────────────────────────────────────────

    def execute_sell_cc(
        self,
        action: WheelAction,
        active_positions: list[OptionsPosition] | None = None,
    ) -> OptionsTradeResult:
        """Select call strike and record a new covered call.

        The cost_basis is read from the referenced parent position (position_id).
        If position_id is None or the position is not found, the current stock
        price is used as a conservative proxy.
        """
        try:
            cost_basis = 0.0
            if action.position_id is not None and active_positions:
                parent = next(
                    (p for p in active_positions if p.id == action.position_id), None
                )
                if parent is not None:
                    # For an assigned CSP, sell_strike = the put strike = cost basis
                    cost_basis = parent.sell_strike or 0.0

            target_delta = self.risk_profile.get("cc_target_delta", 0.25)
            dte_min = self.risk_profile.get("cc_dte_min", 14)
            dte_max = self.risk_profile.get("cc_dte_max", 30)

            cc = select_cc(
                symbol=action.symbol,
                contracts=action.contracts,
                cost_basis=cost_basis,
                target_delta=target_delta,
                dte_min=dte_min,
                dte_max=dte_max,
            )
            if cc is None:
                return OptionsTradeResult(
                    action="OPEN_CC", symbol=action.symbol,
                    spread_type="COVERED_CALL", position_id=None,
                    success=False, error="CC strike selection failed (no suitable chain)",
                )

            # Ghostfolio
            ghostfolio_order_id = None
            if not self.dry_run:
                ghostfolio_order_id = self._ghostfolio_open_cc(cc, action.contracts)
            else:
                ghostfolio_order_id = "DRY_RUN"
                logger.info(
                    "wheel_dry_run_sell_cc",
                    symbol=cc.symbol, strike=cc.strike,
                    expiration=cc.expiration, premium=cc.premium,
                    cost_basis=cc.cost_basis, contracts=action.contracts,
                )

            max_profit = round(cc.premium * 100 * action.contracts, 2)
            # Max loss on the CC itself is theoretically unlimited (uncapped upside capped by
            # the call strike).  For record-keeping we store the potential uplift vs cost basis.
            max_loss = 0.0   # stock already owned; CC only caps upside

            # For covered call, buy_strike stores the stock cost_basis for reference
            pos_id = self.tracker.open_position(
                account_key=self.account_key,
                symbol=cc.symbol,
                spread_type="COVERED_CALL",
                contracts=action.contracts,
                expiration_date=cc.expiration,
                buy_strike=cost_basis,       # stock cost basis
                buy_option_type="stock",
                buy_premium=0.0,
                sell_strike=cc.strike,
                sell_option_type="call",
                sell_premium=cc.premium,
                max_profit=max_profit,
                max_loss=max_loss,
                entry_debit=-cc.premium,     # negative = credit received
                buy_contract_symbol=None,
                sell_contract_symbol=cc.contract_symbol,
                ghostfolio_order_id=ghostfolio_order_id,
            )

            self.tracker.update_position(
                pos_id,
                current_value=cc.premium,
                current_pl=0.0,
                greeks={"net_delta": cc.delta, "net_gamma": 0.0,
                        "net_theta": 0.0, "net_vega": 0.0},
                dte=cc.dte,
            )

            # Link CC to parent assigned CSP and update wheel states
            parent_id = action.position_id
            with sqlite3.connect(self.tracker.db_path) as conn:
                conn.execute(
                    "UPDATE options_positions SET wheel_state='CC_OPEN', "
                    "wheel_parent_id=?, wheel_cost_basis=?, wheel_shares=? WHERE id=?",
                    (parent_id, cost_basis, action.contracts * 100, pos_id),
                )
                if parent_id:
                    conn.execute(
                        "UPDATE options_positions SET wheel_state='CC_OPEN' WHERE id=?",
                        (parent_id,),
                    )

            logger.info(
                "wheel_cc_opened",
                pos_id=pos_id, symbol=cc.symbol,
                strike=cc.strike, expiration=cc.expiration,
                premium=cc.premium, cost_basis=cost_basis,
                delta=round(cc.delta, 3), contracts=action.contracts,
                parent_position_id=parent_id,
            )

            return OptionsTradeResult(
                action="OPEN_CC", symbol=cc.symbol,
                spread_type="COVERED_CALL",
                position_id=pos_id, success=True,
                ghostfolio_order_id=ghostfolio_order_id,
            )

        except Exception as e:
            logger.error("wheel_sell_cc_failed", symbol=action.symbol, error=str(e), exc_info=True)
            return OptionsTradeResult(
                action="OPEN_CC", symbol=action.symbol,
                spread_type="COVERED_CALL",
                position_id=None, success=False, error=str(e),
            )

    # ── Close execution ───────────────────────────────────────────────────────

    def _close_position(self, pos: OptionsPosition, reason: str) -> OptionsTradeResult:
        """Buy back an existing CSP or CC position."""
        try:
            # Current mid-price of the sold option
            close_value = get_current_option_price(
                pos.symbol,
                pos.sell_option_type,
                pos.sell_strike,
                pos.expiration_date,
            )
            if close_value is None:
                # Fall back to recorded current value or entry premium
                close_value = pos.current_value or abs(pos.entry_debit or 0)

            # Ghostfolio: record buy-back as SELL of the synthetic asset
            ghostfolio_order_id = None
            if not self.dry_run:
                ghostfolio_order_id = self._ghostfolio_close(pos, close_value)
            else:
                ghostfolio_order_id = "DRY_RUN"
                logger.info(
                    "wheel_dry_run_close",
                    pos_id=pos.id, symbol=pos.symbol,
                    spread_type=pos.spread_type,
                    close_value=close_value, reason=reason,
                )

            realized_pl = self.tracker.close_position(
                pos.id, close_value, reason, ghostfolio_order_id,
            )

            logger.info(
                "wheel_position_closed",
                pos_id=pos.id, symbol=pos.symbol,
                spread_type=pos.spread_type,
                close_value=close_value, realized_pl=realized_pl,
                reason=reason,
            )

            return OptionsTradeResult(
                action="CLOSE", symbol=pos.symbol,
                spread_type=pos.spread_type,
                position_id=pos.id, success=True,
                realized_pl=realized_pl,
                ghostfolio_order_id=ghostfolio_order_id,
            )

        except Exception as e:
            logger.error("wheel_close_failed", pos_id=pos.id, error=str(e), exc_info=True)
            return OptionsTradeResult(
                action="CLOSE", symbol=pos.symbol,
                spread_type=pos.spread_type,
                position_id=pos.id, success=False, error=str(e),
            )

    # ── State update ──────────────────────────────────────────────────────────

    def _update_position_state(
        self, pos: OptionsPosition, today: date
    ) -> OptionsTradeResult:
        """Refresh DTE, current premium value, and P&L for a held position."""
        try:
            exp_date = datetime.strptime(pos.expiration_date, "%Y-%m-%d").date()
            dte = max((exp_date - today).days, 0)

            if dte == 0:
                # Check if CSP expired ITM (assignment) or OTM (worthless)
                if pos.spread_type == "CASH_SECURED_PUT":
                    stock_price = self._get_stock_price(pos.symbol)
                    if stock_price and stock_price < pos.sell_strike:
                        # ITM — assignment! We now own 100 shares at strike price
                        logger.info(
                            "wheel_csp_assignment_detected",
                            pos_id=pos.id, symbol=pos.symbol,
                            strike=pos.sell_strike, stock_price=stock_price,
                        )
                        cost_basis = self.tracker.assign_position(pos.id, stock_price)
                        # Record stock purchase in Ghostfolio
                        self._ghostfolio_assignment(pos, cost_basis)
                        return OptionsTradeResult(
                            action="ASSIGNMENT", symbol=pos.symbol,
                            spread_type=pos.spread_type,
                            position_id=pos.id, success=True,
                        )
                elif pos.spread_type == "COVERED_CALL":
                    stock_price = self._get_stock_price(pos.symbol)
                    if stock_price and stock_price > pos.sell_strike:
                        # CC exercised — shares called away
                        logger.info(
                            "wheel_cc_exercised",
                            pos_id=pos.id, symbol=pos.symbol,
                            strike=pos.sell_strike, stock_price=stock_price,
                        )
                        cc_premium = abs(pos.entry_debit or 0)
                        realized_pl = self.tracker.call_away_position(
                            pos.id, pos.sell_strike, cc_premium,
                        )
                        # Record stock sale in Ghostfolio
                        self._ghostfolio_call_away(pos)
                        return OptionsTradeResult(
                            action="CALLED_AWAY", symbol=pos.symbol,
                            spread_type=pos.spread_type,
                            position_id=pos.id, success=True,
                            realized_pl=realized_pl,
                        )

                # OTM expiry — worthless
                logger.info("wheel_position_expired", pos_id=pos.id, symbol=pos.symbol)
                self.tracker.expire_position(pos.id)
                return OptionsTradeResult(
                    action="UPDATE", symbol=pos.symbol,
                    spread_type=pos.spread_type,
                    position_id=pos.id, success=True,
                )

            current_value = get_current_option_price(
                pos.symbol,
                pos.sell_option_type,
                pos.sell_strike,
                pos.expiration_date,
            )
            if current_value is None:
                return OptionsTradeResult(
                    action="UPDATE", symbol=pos.symbol,
                    spread_type=pos.spread_type,
                    position_id=pos.id, success=False,
                    error="Could not fetch current option price",
                )

            # For short options: P&L = (entry_premium - current_value) × 100 × contracts
            # entry_debit is stored as negative (credit received), so:
            entry_premium = abs(pos.entry_debit or 0)
            current_pl = round(
                (entry_premium - current_value) * pos.contracts * 100, 2
            )

            self.tracker.update_position(
                pos.id,
                current_value=current_value,
                current_pl=current_pl,
                greeks={},   # Greeks update deferred (expensive; done by caller if needed)
                dte=dte,
            )

            return OptionsTradeResult(
                action="UPDATE", symbol=pos.symbol,
                spread_type=pos.spread_type,
                position_id=pos.id, success=True,
            )

        except Exception as e:
            logger.error("wheel_update_failed", pos_id=pos.id, error=str(e))
            return OptionsTradeResult(
                action="UPDATE", symbol=pos.symbol,
                spread_type=pos.spread_type,
                position_id=pos.id, success=False, error=str(e),
            )

    # ── Ghostfolio helpers ────────────────────────────────────────────────────

    def _ghostfolio_open_csp(self, csp: SelectedCSP, contracts: int) -> str | None:
        """Record CSP premium received in Ghostfolio as a SELL of a synthetic asset.

        Selling a put = receiving cash → SELL so Ghostfolio balance increases.
        Unit price × 100: option premium is per share, 1 contract = 100 shares.
        """
        try:
            exp_compact = csp.expiration.replace("-", "")
            symbol = f"WHEEL-{csp.symbol}-CSP-{exp_compact}-{int(csp.strike)}P"
            result = self.ghostfolio.create_order(
                account_id=self.account_id,
                symbol=symbol,
                order_type="SELL",
                quantity=float(contracts),
                unit_price=round(csp.premium * 100, 2),
                data_source="MANUAL",
            )
            return result.get("id") if isinstance(result, dict) else None
        except Exception as e:
            logger.error("ghostfolio_csp_open_failed", symbol=csp.symbol, error=str(e))
            return None

    def _ghostfolio_open_cc(self, cc: SelectedCC, contracts: int) -> str | None:
        """Record CC premium received in Ghostfolio as a SELL of a synthetic asset.

        Selling a call = receiving cash → SELL so Ghostfolio balance increases.
        Unit price × 100: option premium is per share, 1 contract = 100 shares.
        """
        try:
            exp_compact = cc.expiration.replace("-", "")
            symbol = f"WHEEL-{cc.symbol}-CC-{exp_compact}-{int(cc.strike)}C"
            result = self.ghostfolio.create_order(
                account_id=self.account_id,
                symbol=symbol,
                order_type="SELL",
                quantity=float(contracts),
                unit_price=round(cc.premium * 100, 2),
                data_source="MANUAL",
            )
            return result.get("id") if isinstance(result, dict) else None
        except Exception as e:
            logger.error("ghostfolio_cc_open_failed", symbol=cc.symbol, error=str(e))
            return None

    def _get_stock_price(self, symbol: str) -> float | None:
        """Get current stock price for assignment detection."""
        try:
            return self.market_data.get_current_price(symbol)
        except Exception:
            return None

    def _ghostfolio_assignment(self, pos: OptionsPosition, cost_basis: float) -> str | None:
        """Record CSP assignment as stock BUY in Ghostfolio.

        When assigned, we buy 100 shares per contract at the strike price.
        """
        try:
            shares = pos.contracts * 100
            result = self.ghostfolio.create_order(
                account_id=self.account_id,
                symbol=pos.symbol,
                order_type="BUY",
                quantity=float(shares),
                unit_price=pos.sell_strike,
                data_source="YAHOO",
            )
            logger.info(
                "ghostfolio_assignment_recorded",
                symbol=pos.symbol, shares=shares,
                strike=pos.sell_strike, cost_basis=cost_basis,
            )
            return result.get("id") if isinstance(result, dict) else None
        except Exception as e:
            logger.error("ghostfolio_assignment_failed", symbol=pos.symbol, error=str(e))
            return None

    def _ghostfolio_call_away(self, pos: OptionsPosition) -> str | None:
        """Record CC exercise as stock SELL in Ghostfolio.

        When called away, we sell 100 shares per contract at the CC strike price.
        """
        try:
            shares = pos.contracts * 100
            result = self.ghostfolio.create_order(
                account_id=self.account_id,
                symbol=pos.symbol,
                order_type="SELL",
                quantity=float(shares),
                unit_price=pos.sell_strike,
                data_source="YAHOO",
            )
            logger.info(
                "ghostfolio_call_away_recorded",
                symbol=pos.symbol, shares=shares,
                strike=pos.sell_strike,
            )
            return result.get("id") if isinstance(result, dict) else None
        except Exception as e:
            logger.error("ghostfolio_call_away_failed", symbol=pos.symbol, error=str(e))
            return None

    def _ghostfolio_close(self, pos: OptionsPosition, close_value: float) -> str | None:
        """Record position close (buy-back) as a BUY in Ghostfolio.

        Buying back a short option = paying cash → BUY so Ghostfolio balance decreases.
        Unit price × 100: option price is per share, 1 contract = 100 shares.
        """
        try:
            exp_compact = pos.expiration_date.replace("-", "")
            if pos.spread_type == "CASH_SECURED_PUT":
                symbol = f"WHEEL-{pos.symbol}-CSP-{exp_compact}-{int(pos.sell_strike)}P"
            elif pos.spread_type == "COVERED_CALL":
                symbol = f"WHEEL-{pos.symbol}-CC-{exp_compact}-{int(pos.sell_strike)}C"
            else:
                symbol = f"OPT-{pos.symbol}-{pos.spread_type}-{exp_compact}"

            result = self.ghostfolio.create_order(
                account_id=self.account_id,
                symbol=symbol,
                order_type="BUY",
                quantity=float(pos.contracts),
                unit_price=round(close_value * 100, 2),
                data_source="MANUAL",
            )
            return result.get("id") if isinstance(result, dict) else None
        except Exception as e:
            logger.error("ghostfolio_close_failed", pos_id=pos.id, error=str(e))
            return None
