"""Execute multi-leg spread trades: SQLite position tracking + Ghostfolio cash flows.

Handles:
  execute_opens()   - open new spread positions (select strikes, record in DB + Ghostfolio)
  execute_closes()  - close existing spread positions
  execute_rolls()   - no-op (spreads don't roll; compatibility with main.py)
  update_active_positions() - refresh DTE / P&L for held positions

Ghostfolio integration:
  Open  -> BUY  "GF_SPREAD-{SYM}-{TYPE}-{YYYYMMDD}-{strikes}"  unit_price=net_debit
  Close -> SELL same symbol, unit_price=close_value

The "GF_" prefix is load-bearing: Ghostfolio rewrites the symbol of every
MANUAL BUY to a random UUID unless it starts with "GF_" (order.service.ts),
which would put open and close on different symbol profiles - the pair would
never net to zero and closed positions would distort the account value forever.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import structlog

from ..ghostfolio_client import GhostfolioClient
from ..market_data import MarketDataProvider
from ..transaction_costs import calculate_option_cost
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
        cash_available: float | None = None,
        broker_cost_model: str = "",
    ):
        self.ghostfolio = ghostfolio
        self.market_data = market_data
        self.tracker = tracker
        self.account_id = account_id
        self.account_key = account_key or account_id
        self.risk_profile = risk_profile
        self.dry_run = dry_run
        self.broker_cost_model = broker_cost_model
        # Post-selection affordability budget: the risk manager pre-approves
        # opens with a crude width-based max-loss proxy; once the selector
        # returns the REAL max_loss we re-check it here. Decremented across
        # opens within one cycle. None = check disabled.
        self._cash_available = cash_available

    def _commission(self, contracts: int, legs: int) -> float:
        return calculate_option_cost(self.broker_cost_model, contracts, max(legs, 1))

    def _leg_count(self, pos: OptionsPosition) -> int:
        legs = self.tracker.get_legs(pos.id)
        return len(legs) if legs else 2

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
            target_delta = float(self.risk_profile.get("target_delta", 0.30))

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

            # Affordability re-check with the selector's REAL max_loss (the risk
            # manager only saw a width-based upper-bound proxy).
            if self._cash_available is not None and spread.max_loss > self._cash_available:
                logger.warning(
                    "spread_open_rejected_max_loss",
                    symbol=action.symbol, spread_type=action.spread_type,
                    max_loss=spread.max_loss,
                    cash_available=round(self._cash_available, 2),
                )
                return SpreadsTradeResult(
                    action="OPEN_SPREAD", symbol=action.symbol,
                    spread_type=action.spread_type, position_id=None,
                    success=False,
                    error=(
                        f"Real max loss ${spread.max_loss:,.0f} exceeds remaining "
                        f"cash budget ${self._cash_available:,.0f}"
                    ),
                )
            if self._cash_available is not None:
                self._cash_available -= spread.max_loss

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
                current_value=spread.net_debit,  # signed: negative = credit structure
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
                    # Closing inverts the side: a leg we sold is bought back
                    # (fill above mid), a leg we bought is sold (fill below mid).
                    leg_price = get_current_option_price(
                        pos.symbol, leg.option_type, leg.strike, pos.expiration_date,
                        side="buy" if leg.side == "sell" else "sell",
                    )
                    if leg_price is None:
                        all_priced = False
                        break
                    sign = 1 if leg.side == "buy" else -1
                    net_value += leg_price * sign
                close_value = round(net_value, 2) if all_priced else None
            else:
                # Legacy: single sell leg pricing (short option = liability,
                # negate to the signed convention used by multi-leg pricing)
                leg_price = get_current_option_price(
                    pos.symbol, pos.sell_option_type, pos.sell_strike, pos.expiration_date,
                    side="buy",
                )
                close_value = -leg_price if leg_price is not None else None
            if close_value is None:
                # current_value and entry_debit are both signed, so falling back
                # to entry_debit records a flat (P/L=0) close.
                close_value = (pos.current_value if pos.current_value is not None
                               else (pos.entry_debit or 0))
            else:
                strikes = ([l.strike for l in all_legs] if all_legs
                           else [pos.buy_strike, pos.sell_strike])
                close_value = self._clamp_value(close_value, strikes, pos.id, "close")

            # Tracker/Ghostfolio convention: debit spreads store the (positive)
            # liquidation value received on close; credit spreads store the
            # (positive) cost paid to buy the spread back — same convention as
            # the wheel executor, which close_position()'s P/L formula expects.
            if (pos.entry_debit or 0) < 0:
                close_value = round(-close_value, 2)

            ghostfolio_order_id = None
            if not self.dry_run:
                ghostfolio_order_id = self._ghostfolio_close(
                    pos, close_value,
                    strikes=[l.strike for l in all_legs] if all_legs else None,
                )
            else:
                ghostfolio_order_id = "DRY_RUN"
                logger.info(
                    "spreads_dry_run_close",
                    pos_id=pos.id, symbol=pos.symbol,
                    spread_type=pos.spread_type,
                    close_value=close_value, reason=reason,
                )

            legs_n = len(all_legs) if all_legs else 2
            realized_pl = self.tracker.close_position(
                pos.id, close_value, reason, ghostfolio_order_id,
                costs=self._commission(pos.contracts, legs_n) * 2,  # open + close
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
                # Record the expiry in Ghostfolio too, or the synthetic holding
                # stays open forever and becomes a phantom position.
                exp_legs = self.tracker.get_legs(pos.id)
                if not self.dry_run:
                    self._ghostfolio_close(
                        pos, 0.0,
                        strikes=[l.strike for l in exp_legs] if exp_legs else None,
                        fee=0.0,  # expiry itself is free
                    )
                # Only the opening commission.
                self.tracker.expire_position(
                    pos.id,
                    costs=self._commission(pos.contracts, len(exp_legs) if exp_legs else 2),
                )
                return SpreadsTradeResult(
                    action="UPDATE", symbol=pos.symbol,
                    spread_type=pos.spread_type,
                    position_id=pos.id, success=True,
                )

            # Price all legs from options_legs table (handles iron condors correctly).
            # Falls back to legacy 2-leg pricing if no legs stored.
            entry_debit = pos.entry_debit or 0
            all_legs = self.tracker.get_legs(pos.id)

            # Spot price for intrinsic-value fallback when a leg quote is missing —
            # one illiquid leg must not freeze the whole position's P/L (that would
            # silently disable mark-based take-profit/stop-loss between cycles).
            try:
                spot = float(self.market_data.get_current_price(pos.symbol) or 0) or None
            except Exception:
                spot = None

            def _leg_price(option_type: str, strike: float) -> float | None:
                price = get_current_option_price(
                    pos.symbol, option_type, strike, pos.expiration_date,
                )
                if price is None and spot:
                    price = (max(spot - strike, 0.0) if option_type == "call"
                             else max(strike - spot, 0.0))
                    logger.warning(
                        "spread_leg_stale_price_intrinsic_fallback",
                        pos_id=pos.id, symbol=pos.symbol,
                        option_type=option_type, strike=strike,
                        spot=round(spot, 2), intrinsic=round(price, 2),
                    )
                return price

            if all_legs:
                # Multi-leg pricing: net value = Σ(leg_price * sign)
                # buy legs are assets (positive), sell legs are liabilities (negative)
                net_value = 0.0
                for leg in all_legs:
                    leg_price = _leg_price(leg.option_type, leg.strike)
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
                buy_price = _leg_price(pos.buy_option_type, pos.buy_strike)
                sell_price = _leg_price(pos.sell_option_type, pos.sell_strike)
                if buy_price is None or sell_price is None:
                    return SpreadsTradeResult(
                        action="UPDATE", symbol=pos.symbol,
                        spread_type=pos.spread_type,
                        position_id=pos.id, success=False,
                        error="Could not fetch option leg prices",
                    )
                current_value = round(buy_price - sell_price, 2)
            else:
                # Single-leg (CSP/CC): sell leg only. Negate to the signed
                # convention — a short option is a liability we pay to close.
                leg_price = _leg_price(pos.sell_option_type, pos.sell_strike)
                if leg_price is None:
                    return SpreadsTradeResult(
                        action="UPDATE", symbol=pos.symbol,
                        spread_type=pos.spread_type,
                        position_id=pos.id, success=False,
                        error="Could not fetch current option price",
                    )
                current_value = -leg_price

            strikes = ([l.strike for l in all_legs] if all_legs
                       else [pos.buy_strike, pos.sell_strike])
            current_value = self._clamp_value(current_value, strikes, pos.id, "update")

            # current_value is the SIGNED liquidation value (negative = we must
            # pay to close, i.e. credit structures), and entry_debit uses the
            # same convention (negative = credit received), so one formula
            # covers both debit and credit spreads.
            current_pl = round((current_value - entry_debit) * pos.contracts * 100, 2)

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

    # -- Value sanity guard --

    @staticmethod
    def _clamp_value(value: float, strikes: list, pos_id, label: str) -> float:
        """Clamp a per-share spread/option value to its structural maximum.

        A defined-risk spread's net per-share value cannot exceed the span
        between its strikes; a single option cannot exceed its strike. Larger
        magnitudes only come from bad leg pricing (e.g. a stale lastPrice on an
        illiquid leg when bid/ask are both 0) and, once multiplied by 100 and
        written to Ghostfolio, corrupt the account cash balance. So we cap the
        value at its structural bound and log it for investigation.
        """
        valid = [float(s) for s in strikes if s and float(s) > 0]
        if not valid:
            return value
        bound = valid[0] if len(valid) == 1 else max(valid) - min(valid)
        if bound > 0 and abs(value) > bound:
            clamped = round(bound if value >= 0 else -bound, 2)
            logger.warning(
                "spread_value_clamped", pos_id=pos_id, label=label,
                raw=round(value, 2), bound=round(bound, 2),
            )
            return clamped
        return value

    # -- Ghostfolio helpers --

    @staticmethod
    def _gf_symbol(underlying: str, spread_type: str, expiration: str, strikes: list) -> str:
        """Synthetic Ghostfolio symbol for a spread.

        Open and close MUST produce the identical symbol or Ghostfolio ends up
        with two profiles (a phantom +1 holding and an orphan -1). Both paths
        build it from the full leg strike list in leg order.
        """
        exp_compact = expiration.replace("-", "")
        strikes_part = "-".join(f"{int(s)}" for s in strikes)
        return f"GF_SPREAD-{underlying}-{spread_type.upper()}-{exp_compact}-{strikes_part}"[:50]

    def _ghostfolio_open(self, spread: SelectedSpread) -> str | None:
        """Record spread open in Ghostfolio.

        Debit spread = paying cash to open → BUY.
        Credit spread = receiving cash at open → SELL (mirrors the wheel
        executor's CSP model; the earlier BUY-for-credit model inverted the
        account's cash P/L on every credit spread).
        Unit price × 100: spread price is per share, 1 contract = 100 shares.
        """
        try:
            symbol = self._gf_symbol(
                spread.symbol, spread.spread_type, spread.expiration,
                [l.strike for l in spread.legs],
            )
            raw_debit = abs(spread.net_debit) if spread.net_debit != 0 else 0.01
            result = self.ghostfolio.create_order(
                account_id=self.account_id,
                symbol=symbol,
                order_type="SELL" if spread.net_debit < 0 else "BUY",
                quantity=float(spread.contracts),
                unit_price=round(raw_debit * 100, 2),
                data_source="MANUAL",
                fee=self._commission(spread.contracts, len(spread.legs)),
            )
            return result.get("id") if isinstance(result, dict) else None
        except Exception as e:
            logger.error("ghostfolio_spread_open_failed", symbol=spread.symbol, error=str(e))
            return None

    def _ghostfolio_close(
        self, pos: OptionsPosition, close_value: float, strikes: list | None = None,
        fee: float | None = None,
    ) -> str | None:
        """Record spread close in Ghostfolio.

        close_value is in tracker convention (positive): debit spread → value
        received on close → SELL; credit spread → cost paid to buy back → BUY.
        Net cash across open+close then equals the realized P/L for both types.
        """
        try:
            symbol = self._gf_symbol(
                pos.symbol, pos.spread_type, pos.expiration_date,
                strikes or [pos.buy_strike, pos.sell_strike],
            )
            result = self.ghostfolio.create_order(
                account_id=self.account_id,
                symbol=symbol,
                order_type="BUY" if (pos.entry_debit or 0) < 0 else "SELL",
                quantity=float(pos.contracts),
                unit_price=max(round(close_value * 100, 2), 0.01),
                data_source="MANUAL",
                fee=self._commission(pos.contracts, self._leg_count(pos)) if fee is None else fee,
            )
            return result.get("id") if isinstance(result, dict) else None
        except Exception as e:
            logger.error("ghostfolio_spread_close_failed", pos_id=pos.id, error=str(e))
            return None
