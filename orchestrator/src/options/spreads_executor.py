"""Execute multi-leg spread trades: SQLite position tracking + Ghostfolio cash flows.

Handles:
  execute_opens()   - open new spread positions (select strikes, record in DB + Ghostfolio)
  execute_closes()  - close existing spread positions
  execute_rolls()   - no-op (spreads don't roll; compatibility with main.py)
  update_active_positions() - refresh DTE / P&L for held positions

Ghostfolio integration:
  Open  -> BUY  "SPREAD-{SYM}-{TYPE}-{YYYYMMDD}-{strikes}"  unit_price=net_debit
  Close -> SELL same symbol, unit_price=close_value
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import structlog

from ..ghostfolio_client import GhostfolioClient
from ..market_data import MarketDataProvider
from .data import get_current_option_price
from .positions import OptionsPosition, OptionsPositionTracker
from .spreads_decision_parser import SpreadAction
from .spreads_selector import SelectedSpread, select_spread

logger = structlog.get_logger()


@dataclass
class SpreadsTradeResult:
    action: str              # "OPEN_SPREAD" | "CLOSE" | "UPDATE"
    symbol: str
    spread_type: str
    position_id: int | None
    success: bool
    realized_pl: float | None = None
    error: str = ""
    ghostfolio_order_id: str | None = None


class SpreadsExecutor:
    """Execute open/close decisions for spread positions."""

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

    # -- Public interface --

    def execute_opens(
        self,
        opens: list[SpreadAction],
        active_positions: list[OptionsPosition] | None = None,
    ) -> list[SpreadsTradeResult]:
        seen: set[str] = set()
        results = []
        for action in opens:
            if action.symbol in seen:
                logger.warning("duplicate_spread_open_skipped", symbol=action.symbol, spread_type=action.spread_type)
                continue
            seen.add(action.symbol)
            results.append(self._execute_open_spread(action))
        return results

    def execute_closes(
        self,
        closes: list[SpreadAction],
        active_positions: list[OptionsPosition],
    ) -> list[SpreadsTradeResult]:
        results = []
        pos_map = {p.id: p for p in active_positions}
        for action in closes:
            pid = action.position_id
            pos = pos_map.get(pid) if pid is not None else None
            if pos is None:
                results.append(SpreadsTradeResult(
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
    ) -> list[SpreadsTradeResult]:
        if rolls:
            logger.warning("spreads_executor_rolls_ignored", count=len(rolls))
        return []

    def update_active_positions(
        self,
        active_positions: list[OptionsPosition],
    ) -> list[SpreadsTradeResult]:
        results = []
        today = date.today()
        for pos in active_positions:
            results.append(self._update_position_state(pos, today))
        return results

    # -- Open execution --

    def _execute_open_spread(self, action: SpreadAction) -> SpreadsTradeResult:
        """Select strikes and record a new spread position."""
        try:
            dte_min = self.risk_profile.get("target_dte_min", 21)
            dte_max = self.risk_profile.get("target_dte_max", 45)
            max_width = self.risk_profile.get("max_spread_width", 10.0)
            target_delta = 0.30  # reasonable default for short legs

            spread = select_spread(
                symbol=action.symbol,
                spread_type=action.spread_type,
                contracts=action.contracts,
                dte_min=dte_min,
                dte_max=dte_max,
                max_width=max_width,
                target_delta=target_delta,
            )
            if spread is None:
                return SpreadsTradeResult(
                    action="OPEN_SPREAD", symbol=action.symbol,
                    spread_type=action.spread_type, position_id=None,
                    success=False, error="Spread selection failed (no suitable chain/strikes)",
                )

            # Determine buy/sell legs for DB storage
            # For multi-leg spreads, store the primary buy and sell legs
            buy_legs = [l for l in spread.legs if l.side == "buy"]
            sell_legs = [l for l in spread.legs if l.side == "sell"]

            # Primary legs for DB (first buy, first sell)
            buy_leg = buy_legs[0] if buy_legs else None
            sell_leg = sell_legs[0] if sell_legs else None

            buy_strike = buy_leg.strike if buy_leg else 0.0
            buy_type = buy_leg.option_type if buy_leg else "call"
            buy_premium = buy_leg.premium if buy_leg else 0.0
            buy_contract_sym = buy_leg.contract_symbol if buy_leg else None

            sell_strike = sell_leg.strike if sell_leg else 0.0
            sell_type = sell_leg.option_type if sell_leg else "call"
            sell_premium = sell_leg.premium if sell_leg else 0.0
            sell_contract_sym = sell_leg.contract_symbol if sell_leg else None

            # Ghostfolio
            ghostfolio_order_id = None
            if not self.dry_run:
                ghostfolio_order_id = self._ghostfolio_open(spread)
            else:
                ghostfolio_order_id = "DRY_RUN"
                logger.info(
                    "spreads_dry_run_open",
                    symbol=spread.symbol, spread_type=spread.spread_type,
                    expiration=spread.expiration, legs=len(spread.legs),
                    net_debit=spread.net_debit, contracts=action.contracts,
                )

            # Map spread_type to DB spread_type naming
            db_spread_type = action.spread_type.upper()

            pos_id = self.tracker.open_position(
                account_key=self.account_key,
                symbol=spread.symbol,
                spread_type=db_spread_type,
                contracts=action.contracts,
                expiration_date=spread.expiration,
                buy_strike=buy_strike,
                buy_option_type=buy_type,
                buy_premium=buy_premium,
                sell_strike=sell_strike,
                sell_option_type=sell_type,
                sell_premium=sell_premium,
                max_profit=spread.max_profit,
                max_loss=spread.max_loss,
                entry_debit=spread.net_debit,
                buy_contract_symbol=buy_contract_sym,
                sell_contract_symbol=sell_contract_sym,
                ghostfolio_order_id=ghostfolio_order_id,
            )

            # Store ALL legs for multi-leg spreads (critical for iron condors)
            self.tracker.save_legs(pos_id, [
                {
                    "option_type": l.option_type,
                    "side": l.side,
                    "strike": l.strike,
                    "premium": l.premium,
                    "contract_symbol": l.contract_symbol,
                }
                for l in spread.legs
            ])

            # Compute initial net greeks from legs
            net_delta = sum(
                l.delta * (1 if l.side == "buy" else -1) * 100 * action.contracts
                for l in spread.legs
            )
            net_greeks = {"net_delta": round(net_delta, 2), "net_gamma": 0.0,
                          "net_theta": 0.0, "net_vega": 0.0}

            self.tracker.update_position(
                pos_id,
                current_value=abs(spread.net_debit),
                current_pl=0.0,
                greeks=net_greeks,
                dte=spread.dte,
            )

            logger.info(
                "spread_opened",
                pos_id=pos_id, symbol=spread.symbol,
                spread_type=db_spread_type, expiration=spread.expiration,
                legs=len(spread.legs), net_debit=spread.net_debit,
                max_profit=spread.max_profit, max_loss=spread.max_loss,
                contracts=action.contracts,
            )

            return SpreadsTradeResult(
                action="OPEN_SPREAD", symbol=spread.symbol,
                spread_type=db_spread_type,
                position_id=pos_id, success=True,
                ghostfolio_order_id=ghostfolio_order_id,
            )

        except Exception as e:
            logger.error("spread_open_failed", symbol=action.symbol, error=str(e), exc_info=True)
            return SpreadsTradeResult(
                action="OPEN_SPREAD", symbol=action.symbol,
                spread_type=action.spread_type,
                position_id=None, success=False, error=str(e),
            )

    # -- Close execution --

    def _close_position(self, pos: OptionsPosition, reason: str) -> SpreadsTradeResult:
        """Close an existing spread position."""
        try:
            # Price all legs for accurate close value (critical for iron condors)
            all_legs = self.tracker.get_legs(pos.id)
            if all_legs:
                net_value = 0.0
                all_priced = True
                for leg in all_legs:
                    leg_price = get_current_option_price(
                        pos.symbol, leg.option_type, leg.strike, pos.expiration_date,
                    )
                    if leg_price is None:
                        all_priced = False
                        break
                    sign = 1 if leg.side == "buy" else -1
                    net_value += leg_price * sign
                close_value = round(net_value, 2) if all_priced else None
            else:
                # Legacy: single sell leg pricing
                close_value = get_current_option_price(
                    pos.symbol, pos.sell_option_type, pos.sell_strike, pos.expiration_date,
                )
            if close_value is None:
                close_value = pos.current_value or abs(pos.entry_debit or 0)

            ghostfolio_order_id = None
            if not self.dry_run:
                ghostfolio_order_id = self._ghostfolio_close(pos, close_value)
            else:
                ghostfolio_order_id = "DRY_RUN"
                logger.info(
                    "spreads_dry_run_close",
                    pos_id=pos.id, symbol=pos.symbol,
                    spread_type=pos.spread_type,
                    close_value=close_value, reason=reason,
                )

            realized_pl = self.tracker.close_position(
                pos.id, close_value, reason, ghostfolio_order_id,
            )

            logger.info(
                "spread_position_closed",
                pos_id=pos.id, symbol=pos.symbol,
                spread_type=pos.spread_type,
                close_value=close_value, realized_pl=realized_pl,
                reason=reason,
            )

            return SpreadsTradeResult(
                action="CLOSE", symbol=pos.symbol,
                spread_type=pos.spread_type,
                position_id=pos.id, success=True,
                realized_pl=realized_pl,
                ghostfolio_order_id=ghostfolio_order_id,
            )

        except Exception as e:
            logger.error("spread_close_failed", pos_id=pos.id, error=str(e), exc_info=True)
            return SpreadsTradeResult(
                action="CLOSE", symbol=pos.symbol,
                spread_type=pos.spread_type,
                position_id=pos.id, success=False, error=str(e),
            )

    # -- State update --

    def _update_position_state(
        self, pos: OptionsPosition, today: date
    ) -> SpreadsTradeResult:
        """Refresh DTE, current value, and P&L for a held position."""
        try:
            exp_date = datetime.strptime(pos.expiration_date, "%Y-%m-%d").date()
            dte = max((exp_date - today).days, 0)

            if dte == 0:
                logger.info("spread_position_expired", pos_id=pos.id, symbol=pos.symbol)
                self.tracker.expire_position(pos.id)
                return SpreadsTradeResult(
                    action="UPDATE", symbol=pos.symbol,
                    spread_type=pos.spread_type,
                    position_id=pos.id, success=True,
                )

            # Price all legs from options_legs table (handles iron condors correctly).
            # Falls back to legacy 2-leg pricing if no legs stored.
            entry_debit = pos.entry_debit or 0
            all_legs = self.tracker.get_legs(pos.id)

            if all_legs:
                # Multi-leg pricing: net value = Σ(leg_price * sign)
                # buy legs are assets (positive), sell legs are liabilities (negative)
                net_value = 0.0
                for leg in all_legs:
                    leg_price = get_current_option_price(
                        pos.symbol, leg.option_type, leg.strike, pos.expiration_date,
                    )
                    if leg_price is None:
                        return SpreadsTradeResult(
                            action="UPDATE", symbol=pos.symbol,
                            spread_type=pos.spread_type,
                            position_id=pos.id, success=False,
                            error=f"Could not fetch price for {leg.option_type} {leg.strike}",
                        )
                    sign = 1 if leg.side == "buy" else -1
                    net_value += leg_price * sign
                current_value = round(net_value, 2)
            elif (pos.buy_strike or 0) > 0:
                # Legacy 2-leg fallback (positions opened before options_legs table)
                buy_price = get_current_option_price(
                    pos.symbol, pos.buy_option_type, pos.buy_strike, pos.expiration_date,
                )
                sell_price = get_current_option_price(
                    pos.symbol, pos.sell_option_type, pos.sell_strike, pos.expiration_date,
                )
                if buy_price is None or sell_price is None:
                    return SpreadsTradeResult(
                        action="UPDATE", symbol=pos.symbol,
                        spread_type=pos.spread_type,
                        position_id=pos.id, success=False,
                        error="Could not fetch option leg prices",
                    )
                current_value = round(buy_price - sell_price, 2)
            else:
                # Single-leg (CSP/CC): sell leg only
                current_value = get_current_option_price(
                    pos.symbol, pos.sell_option_type, pos.sell_strike, pos.expiration_date,
                )
                if current_value is None:
                    return SpreadsTradeResult(
                        action="UPDATE", symbol=pos.symbol,
                        spread_type=pos.spread_type,
                        position_id=pos.id, success=False,
                        error="Could not fetch current option price",
                    )

            # P&L depends on whether it's a debit or credit spread
            if entry_debit > 0:
                # Debit spread: P&L = (current_value - entry_debit) * contracts * 100
                current_pl = round((current_value - entry_debit) * pos.contracts * 100, 2)
            else:
                # Credit spread / CSP: P&L = (|entry_credit| - current_value) * contracts * 100
                entry_credit = abs(entry_debit)
                current_pl = round((entry_credit - current_value) * pos.contracts * 100, 2)

            self.tracker.update_position(
                pos.id,
                current_value=current_value,
                current_pl=current_pl,
                greeks={},
                dte=dte,
            )

            return SpreadsTradeResult(
                action="UPDATE", symbol=pos.symbol,
                spread_type=pos.spread_type,
                position_id=pos.id, success=True,
            )

        except Exception as e:
            logger.error("spread_update_failed", pos_id=pos.id, error=str(e))
            return SpreadsTradeResult(
                action="UPDATE", symbol=pos.symbol,
                spread_type=pos.spread_type,
                position_id=pos.id, success=False, error=str(e),
            )

    # -- Ghostfolio helpers --

    def _ghostfolio_open(self, spread: SelectedSpread) -> str | None:
        """Record spread open in Ghostfolio as a BUY of a synthetic asset.

        Debit spread = paying cash to open → BUY (correct direction).
        Unit price × 100: spread price is per share, 1 contract = 100 shares.
        """
        try:
            exp_compact = spread.expiration.replace("-", "")
            strikes = "-".join(f"{int(l.strike)}" for l in spread.legs)
            symbol = f"SPREAD-{spread.symbol}-{spread.spread_type.upper()}-{exp_compact}-{strikes}"
            symbol = symbol[:50]

            raw_debit = abs(spread.net_debit) if spread.net_debit != 0 else 0.01
            unit_price = round(raw_debit * 100, 2)
            result = self.ghostfolio.create_order(
                account_id=self.account_id,
                symbol=symbol,
                order_type="BUY",
                quantity=float(spread.contracts),
                unit_price=unit_price,
                data_source="MANUAL",
            )
            return result.get("id") if isinstance(result, dict) else None
        except Exception as e:
            logger.error("ghostfolio_spread_open_failed", symbol=spread.symbol, error=str(e))
            return None

    def _ghostfolio_close(self, pos: OptionsPosition, close_value: float) -> str | None:
        """Record spread close as a SELL in Ghostfolio.

        Closing a debit spread = receiving cash back → SELL (correct direction).
        Unit price × 100: spread price is per share, 1 contract = 100 shares.
        """
        try:
            exp_compact = pos.expiration_date.replace("-", "")
            symbol = (
                f"SPREAD-{pos.symbol}-{pos.spread_type}-{exp_compact}-"
                f"{int(pos.buy_strike)}-{int(pos.sell_strike)}"
            )
            symbol = symbol[:50]

            result = self.ghostfolio.create_order(
                account_id=self.account_id,
                symbol=symbol,
                order_type="SELL",
                quantity=float(pos.contracts),
                unit_price=round(close_value * 100, 2),
                data_source="MANUAL",
            )
            return result.get("id") if isinstance(result, dict) else None
        except Exception as e:
            logger.error("ghostfolio_spread_close_failed", pos_id=pos.id, error=str(e))
            return None
