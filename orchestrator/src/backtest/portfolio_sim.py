"""Simulated in-memory portfolio for backtesting (no Ghostfolio dependency)."""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from ..portfolio_state import PortfolioState, Position
from ..transaction_costs import calculate_cost

logger = structlog.get_logger()


@dataclass
class SimPosition:
    """A single held position in the simulated portfolio."""
    symbol: str
    quantity: float
    avg_cost: float
    buy_date: str  # YYYY-MM-DD — used by RiskManager min_holding_days check


@dataclass
class SimTrade:
    """Record of a single simulated trade execution."""
    date: str
    symbol: str
    type: str          # "BUY" or "SELL"
    quantity: float
    price: float
    total: float
    success: bool
    avg_cost: float = 0.0  # For SELL trades: avg_cost at time of sale
    error: str = ""


class SimulatedPortfolio:
    """In-memory portfolio that tracks cash, positions, and trade history."""

    def __init__(self, initial_cash: float = 10_000, broker_cost_model: str = ""):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: dict[str, SimPosition] = {}
        # "" = frictionless (legacy behaviour). With a model set, every fill
        # pays the same commission the live executor books — a backtest that
        # ignores costs rewards churn, which is exactly the failure mode the
        # OOS run measured (179 vs 114 trades decided the model gap).
        self.broker_cost_model = broker_cost_model
        self.fees_paid = 0.0

    def _fee(self, quantity: float, price: float) -> float:
        if not self.broker_cost_model:
            return 0.0
        return calculate_cost(self.broker_cost_model, quantity, price)

    def buy(
        self,
        symbol: str,
        amount_usd: float,
        price: float,
        sim_date: str,
    ) -> SimTrade:
        """Execute a simulated BUY order."""
        if price <= 0:
            return SimTrade(date=sim_date, symbol=symbol, type="BUY",
                            quantity=0, price=price, total=0, success=False,
                            error="Invalid price")

        # Clamp to available cash (leave room for the commission)
        fee_headroom = 1.0 if self.broker_cost_model else 0.0
        amount_usd = min(amount_usd, self.cash - fee_headroom)
        if amount_usd <= 0:
            return SimTrade(date=sim_date, symbol=symbol, type="BUY",
                            quantity=0, price=price, total=0, success=False,
                            error="Insufficient cash")

        quantity = amount_usd / price
        total = quantity * price

        if symbol in self.positions:
            pos = self.positions[symbol]
            new_qty = pos.quantity + quantity
            pos.avg_cost = (pos.avg_cost * pos.quantity + total) / new_qty
            pos.quantity = new_qty
        else:
            self.positions[symbol] = SimPosition(
                symbol=symbol,
                quantity=quantity,
                avg_cost=price,
                buy_date=sim_date,
            )

        fee = self._fee(quantity, price)
        self.cash -= total + fee
        self.fees_paid += fee
        logger.debug("sim_buy", symbol=symbol, qty=quantity, price=price, total=total, fee=fee)
        return SimTrade(date=sim_date, symbol=symbol, type="BUY",
                        quantity=quantity, price=price, total=total, success=True)

    def sell(
        self,
        symbol: str,
        amount_usd: float,
        price: float,
        sim_date: str,
    ) -> SimTrade:
        """Execute a simulated SELL order."""
        if symbol not in self.positions:
            return SimTrade(date=sim_date, symbol=symbol, type="SELL",
                            quantity=0, price=price, total=0, success=False,
                            error=f"No position in {symbol}")
        if price <= 0:
            return SimTrade(date=sim_date, symbol=symbol, type="SELL",
                            quantity=0, price=price, total=0, success=False,
                            error="Invalid price")

        pos = self.positions[symbol]
        avg_cost = pos.avg_cost
        current_value = pos.quantity * price

        # Clamp to actual position value
        amount_usd = min(amount_usd, current_value)
        quantity = amount_usd / price
        total = quantity * price

        pos.quantity -= quantity
        if pos.quantity <= 0.0001:
            del self.positions[symbol]

        fee = self._fee(quantity, price)
        self.cash += total - fee
        self.fees_paid += fee
        logger.debug("sim_sell", symbol=symbol, qty=quantity, price=price, total=total, fee=fee)
        return SimTrade(date=sim_date, symbol=symbol, type="SELL",
                        quantity=quantity, price=price, total=total,
                        success=True, avg_cost=avg_cost)

    def get_total_value(self, current_prices: dict[str, float]) -> float:
        """Compute total portfolio value (cash + market value of positions)."""
        invested = sum(
            pos.quantity * current_prices.get(sym, pos.avg_cost)
            for sym, pos in self.positions.items()
        )
        return self.cash + invested

    def to_portfolio_state(
        self,
        sim_date: str,
        account_name: str,
        current_prices: dict[str, float] | None = None,
    ) -> PortfolioState:
        """Build a PortfolioState compatible with the existing LLM pipeline."""
        prices = current_prices or {}
        positions: list[Position] = []
        total_invested = 0.0

        for sym, pos in self.positions.items():
            current_price = prices.get(sym, pos.avg_cost)
            market_value = pos.quantity * current_price
            investment = pos.quantity * pos.avg_cost
            unrealized_pl = market_value - investment
            unrealized_pl_pct = (unrealized_pl / investment * 100) if investment > 0 else 0.0

            positions.append(Position(
                symbol=sym,
                name=sym,
                quantity=pos.quantity,
                avg_cost=pos.avg_cost,
                current_price=current_price,
                market_value=market_value,
                unrealized_pl=unrealized_pl,
                unrealized_pl_pct=unrealized_pl_pct,
                sector="Unknown",
                first_buy_date=pos.buy_date,
            ))
            total_invested += investment

        total_market = sum(p.market_value for p in positions)
        total_value = total_market + self.cash

        for p in positions:
            p.weight_pct = (p.market_value / total_value * 100) if total_value > 0 else 0.0

        total_pl = total_market - total_invested
        total_pl_pct = (total_pl / total_invested * 100) if total_invested > 0 else 0.0

        return PortfolioState(
            account_id="backtest",
            account_name=account_name,
            total_value=total_value,
            cash=self.cash,
            invested=total_invested,
            positions=positions,
            total_pl=total_pl,
            total_pl_pct=total_pl_pct,
            timestamp=sim_date,
        )

    def snapshot(
        self,
        sim_date: str,
        current_prices: dict[str, float] | None = None,
    ) -> dict:
        """Return a lightweight snapshot for charting."""
        total = self.get_total_value(current_prices or {})
        pl_pct = ((total - self.initial_cash) / self.initial_cash * 100) if self.initial_cash > 0 else 0.0
        return {
            "date": sim_date,
            "total_value": round(total, 2),
            "cash": round(self.cash, 2),
            "pl_pct": round(pl_pct, 2),
        }
