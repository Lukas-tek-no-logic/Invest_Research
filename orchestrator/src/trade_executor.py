"""Execute validated trades via Ghostfolio API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import structlog

from .decision_parser import TradeAction
from .ghostfolio_client import GhostfolioClient
from .market_data import MarketDataProvider
from .transaction_costs import calculate_cost

logger = structlog.get_logger()


@dataclass
class TradeResult:
    action: TradeAction
    success: bool
    quantity: float = 0.0
    unit_price: float = 0.0
    total_cost: float = 0.0
    ghostfolio_order_id: str = ""
    fee: float = 0.0
    error: str = ""


class TradeExecutor:
    """Executes approved trades by creating orders in Ghostfolio."""

    def __init__(
        self,
        ghostfolio: GhostfolioClient,
        market_data: MarketDataProvider,
        dry_run: bool = False,
        broker_cost_model: str = "",
    ):
        self.ghostfolio = ghostfolio
        self.market_data = market_data
        self.dry_run = dry_run
        self.broker_cost_model = broker_cost_model

    def execute_trades(
        self,
        actions: list[TradeAction],
        account_id: str,
    ) -> list[TradeResult]:
        """Execute a list of trade actions.

        Returns list of TradeResult with success/failure for each.
        """
        # Deduplicate by (type, symbol) — keep first occurrence regardless of source
        seen: set[tuple[str, str]] = set()
        deduped = []
        for action in actions:
            key = (action.type, action.symbol)
            if key in seen:
                logger.warning("duplicate_trade_skipped", type=action.type, symbol=action.symbol)
                continue
            seen.add(key)
            deduped.append(action)

        results = []
        for action in deduped:
            result = self._execute_single(action, account_id)
            results.append(result)
        return results

    def _execute_single(self, action: TradeAction, account_id: str) -> TradeResult:
        """Execute a single trade action."""
        try:
            # Get current price
            price = self.market_data.get_current_price(action.symbol)
            if price <= 0:
                return TradeResult(
                    action=action,
                    success=False,
                    error=f"Could not get price for {action.symbol}",
                )

            # Calculate quantity
            quantity = action.amount_usd / price
            if quantity <= 0:
                return TradeResult(
                    action=action,
                    success=False,
                    error=f"Calculated quantity <= 0 for {action.symbol}",
                )

            total_cost = quantity * price

            # Transaction fee (0.0 when no cost model configured)
            fee = (
                calculate_cost(self.broker_cost_model, quantity, price)
                if self.broker_cost_model else 0.0
            )

            if self.dry_run:
                logger.info(
                    "trade_dry_run",
                    type=action.type,
                    symbol=action.symbol,
                    quantity=round(quantity, 6),
                    price=price,
                    total=round(total_cost, 2),
                    fee=round(fee, 4),
                )
                return TradeResult(
                    action=action,
                    success=True,
                    quantity=quantity,
                    unit_price=price,
                    total_cost=total_cost,
                    ghostfolio_order_id="DRY_RUN",
                    fee=fee,
                )

            # Create order in Ghostfolio
            order = self.ghostfolio.create_order(
                account_id=account_id,
                symbol=action.symbol,
                order_type=action.type,
                quantity=round(quantity, 6),
                unit_price=price,
                date=datetime.now(timezone.utc),
                fee=fee,
            )

            order_id = order.get("id", "unknown")
            logger.info(
                "trade_executed",
                type=action.type,
                symbol=action.symbol,
                quantity=round(quantity, 6),
                price=price,
                total=round(total_cost, 2),
                fee=round(fee, 4),
                order_id=order_id,
            )

            return TradeResult(
                action=action,
                success=True,
                quantity=quantity,
                unit_price=price,
                total_cost=total_cost,
                ghostfolio_order_id=order_id,
                fee=fee,
            )

        except Exception as e:
            logger.error(
                "trade_execution_failed",
                type=action.type,
                symbol=action.symbol,
                amount=action.amount_usd,
                error=str(e),
            )
            return TradeResult(
                action=action,
                success=False,
                error=str(e),
            )

    def verify_orders(self, results: list[TradeResult]) -> list[str]:
        """Verify that executed orders appear in Ghostfolio.

        Returns list of warnings for any unverified orders.
        """
        if self.dry_run:
            return []

        warnings = []
        try:
            orders = self.ghostfolio.list_orders()
            order_ids = {o.get("id") for o in orders if isinstance(o, dict)}
        except Exception as e:
            return [f"Could not verify orders: {e}"]

        for result in results:
            if result.success and result.ghostfolio_order_id != "DRY_RUN":
                if result.ghostfolio_order_id not in order_ids:
                    warnings.append(
                        f"Order {result.ghostfolio_order_id} for "
                        f"{result.action.symbol} not found in Ghostfolio"
                    )
        return warnings
