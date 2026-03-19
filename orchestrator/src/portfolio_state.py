"""Aggregate portfolio state from Ghostfolio for a specific account."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import structlog

from .ghostfolio_client import GhostfolioClient

logger = structlog.get_logger()


@dataclass
class Position:
    symbol: str
    name: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_pl_pct: float
    sector: str
    first_buy_date: str | None = None
    weight_pct: float = 0.0


@dataclass
class PortfolioState:
    account_id: str
    account_name: str
    total_value: float
    cash: float
    invested: float
    positions: list[Position] = field(default_factory=list)
    total_pl: float = 0.0
    total_pl_pct: float = 0.0
    sector_weights: dict[str, float] = field(default_factory=dict)
    timestamp: str = ""

    @property
    def cash_pct(self) -> float:
        if self.total_value <= 0:
            return 100.0
        return (self.cash / self.total_value) * 100

    @property
    def position_count(self) -> int:
        return len(self.positions)

    def get_position(self, symbol: str) -> Position | None:
        for p in self.positions:
            if p.symbol == symbol:
                return p
        return None

    def with_extra_cash(self, extra: float) -> "PortfolioState":
        """Return a copy with extra cash (e.g. from pending sells)."""
        return PortfolioState(
            account_id=self.account_id,
            account_name=self.account_name,
            total_value=self.total_value,
            cash=self.cash + extra,
            invested=self.invested,
            positions=self.positions,
            total_pl=self.total_pl,
            total_pl_pct=self.total_pl_pct,
            sector_weights=self.sector_weights,
            timestamp=self.timestamp,
        )

    def to_prompt_text(self) -> str:
        """Format portfolio state for LLM prompt."""
        lines = [
            f"== PORTFOLIO STATE ({self.account_name}) ==",
            f"Total Value: ${self.total_value:,.2f}",
            f"Cash: ${self.cash:,.2f} ({self.cash_pct:.1f}%)",
            f"Invested: ${self.invested:,.2f}",
            f"Total P/L: ${self.total_pl:,.2f} ({self.total_pl_pct:+.2f}%)",
            f"Positions: {self.position_count}",
            "",
        ]
        if self.positions:
            lines.append("Holdings:")
            for p in sorted(self.positions, key=lambda x: x.market_value, reverse=True):
                lines.append(
                    f"  {p.symbol}: {p.quantity:.4f} shares @ avg ${p.avg_cost:.2f} "
                    f"| now ${p.current_price:.2f} | value ${p.market_value:,.2f} "
                    f"| P/L {p.unrealized_pl_pct:+.1f}% | weight {p.weight_pct:.1f}% "
                    f"| sector: {p.sector}"
                )
        else:
            lines.append("Holdings: (none - cash only)")

        if self.sector_weights:
            lines.append("")
            lines.append("Sector breakdown:")
            for sector, weight in sorted(self.sector_weights.items(), key=lambda x: -x[1]):
                lines.append(f"  {sector}: {weight:.1f}%")

        return "\n".join(lines)


def get_portfolio_state(
    client: GhostfolioClient,
    account_id: str,
    account_name: str,
) -> PortfolioState:
    """Build a PortfolioState from Ghostfolio API data.

    Uses:
      - account list  → cash balance + total value (valueInBaseCurrency)
      - order list    → per-account positions (filtered by accountId)
      - holdings list → current market prices (matched by symbol)
    """
    _empty = PortfolioState(
        account_id=account_id,
        account_name=account_name,
        total_value=0,
        cash=0,
        invested=0,
        timestamp=datetime.utcnow().isoformat(),
    )

    try:
        accounts_raw = client.list_accounts()
        orders_raw = client.list_orders()
        holdings_raw = client.get_portfolio_holdings()
    except Exception as e:
        logger.error("portfolio_state_fetch_failed", account_id=account_id, error=str(e))
        return _empty

    try:
        # ── 1. Account cash + total value ──────────────────────────────────────
        if isinstance(accounts_raw, list):
            accounts = accounts_raw
        elif isinstance(accounts_raw, dict):
            accounts = accounts_raw.get("accounts", []) or []
        else:
            accounts = []

        account_info = next(
            (a for a in accounts if isinstance(a, dict) and a.get("id") == account_id),
            None,
        )

        cash = float(account_info.get("balance", 0) or 0) if account_info else 0.0
        # valueInBaseCurrency = securities market value only (NOT cash).
        # Exception: for accounts with no real positions Ghostfolio echoes the cash
        # balance here, which would double-count if we then added cash again.
        # Detect this by checking if api_total ≈ cash (within 0.5%).
        api_total_raw = float(account_info.get("valueInBaseCurrency", 0) or 0) if account_info else 0.0
        if cash > 0 and abs(api_total_raw - cash) / cash < 0.005:
            api_total = 0.0  # no real securities; total_value = cash only
        else:
            api_total = api_total_raw

        # ── 2. Build price map from holdings list ──────────────────────────────
        # /api/v1/portfolio/holdings returns a list (not a dict) in recent Ghostfolio
        # versions, without per-account filtering.  We use it only as a price source.
        if isinstance(holdings_raw, dict) and "holdings" in holdings_raw:
            raw_list = holdings_raw["holdings"]
        else:
            raw_list = holdings_raw

        price_map: dict[str, dict] = {}
        if isinstance(raw_list, list):
            for h in raw_list:
                if not isinstance(h, dict):
                    continue
                sp = h.get("SymbolProfile") or {}
                sym = sp.get("symbol") or h.get("symbol", "")
                if not sym or len(sym) > 10:  # skip UUIDs / system entries
                    continue
                sectors_raw = h.get("sectors") or []
                first = sectors_raw[0] if sectors_raw else {}
                sector = first.get("name", "Unknown") if isinstance(first, dict) else str(first)
                price_map[sym] = {
                    "price": float(h.get("marketPrice", 0) or 0),
                    "sector": sector,
                    "name": h.get("name", sym),
                }
        elif isinstance(raw_list, dict):
            # Legacy dict format (older Ghostfolio)
            for sym, h in raw_list.items():
                if not isinstance(h, dict):
                    continue
                sectors_raw = h.get("sectors") or []
                first = sectors_raw[0] if sectors_raw else {}
                sector = first.get("name", "Unknown") if isinstance(first, dict) else str(first)
                price_map[sym] = {
                    "price": float(h.get("marketPrice", 0) or 0),
                    "sector": sector,
                    "name": h.get("name", sym),
                }

        # ── 3. Build positions from orders filtered by accountId ───────────────
        if isinstance(orders_raw, list):
            orders = orders_raw
        elif isinstance(orders_raw, dict):
            orders = orders_raw.get("activities", []) or []
        else:
            orders = []

        acct_orders = [o for o in orders if isinstance(o, dict) and o.get("accountId") == account_id]

        # Aggregate BUY / SELL per symbol
        agg: dict[str, dict] = {}  # symbol → {qty, total_cost, first_date}
        for order in acct_orders:
            sp = order.get("SymbolProfile") or {}
            sym = sp.get("symbol") or order.get("symbol", "")
            if not sym:
                continue
            qty = float(order.get("quantity", 0) or 0)
            price = float(order.get("unitPrice", 0) or 0)
            order_type = (order.get("type") or "BUY").upper()
            order_date = str(order.get("date", ""))[:10]

            if sym not in agg:
                agg[sym] = {"qty": 0.0, "total_cost": 0.0, "first_date": order_date}
            if order_type == "BUY":
                agg[sym]["qty"] += qty
                agg[sym]["total_cost"] += qty * price
            elif order_type == "SELL":
                # Reduce qty; proportionally reduce cost basis
                if agg[sym]["qty"] > 0:
                    sell_fraction = min(qty / agg[sym]["qty"], 1.0)
                    agg[sym]["total_cost"] *= (1 - sell_fraction)
                agg[sym]["qty"] = max(0, agg[sym]["qty"] - qty)

        positions: list[Position] = []
        total_market = 0.0
        total_invested = 0.0
        sector_totals: dict[str, float] = {}

        for sym, data in agg.items():
            qty = data["qty"]
            if qty < 0.0001:
                continue

            avg_cost = data["total_cost"] / qty if qty > 0 else 0.0
            info = price_map.get(sym, {})
            current_price = info.get("price", 0.0) or avg_cost  # fallback to cost
            market_value = current_price * qty
            investment = avg_cost * qty
            unrealized_pl = market_value - investment
            unrealized_pl_pct = (unrealized_pl / investment * 100) if investment > 0 else 0.0
            sector = info.get("sector", "Unknown")

            positions.append(Position(
                symbol=sym,
                name=info.get("name", sym),
                quantity=qty,
                avg_cost=avg_cost,
                current_price=current_price,
                market_value=market_value,
                unrealized_pl=unrealized_pl,
                unrealized_pl_pct=unrealized_pl_pct,
                sector=sector,
                first_buy_date=data.get("first_date"),
            ))
            total_market += market_value
            total_invested += investment
            sector_totals[sector] = sector_totals.get(sector, 0) + market_value

        # ── 4. Totals ──────────────────────────────────────────────────────────
        # Prefer self-computed total_market (per-account orders × current prices) over
        # api_total (valueInBaseCurrency) which Ghostfolio reports as the whole-portfolio
        # total, not per-account — causing inflated values for accounts that hold only
        # a fraction of the overall portfolio.
        # Fall back to api_total only if we have no order data at all.
        if acct_orders:
            total_value = total_market + cash
        else:
            total_value = cash  # cash-only, api_total unreliable

        # Compute weights
        for p in positions:
            p.weight_pct = (p.market_value / total_value * 100) if total_value > 0 else 0

        sector_weights = {
            sector: (val / total_value * 100) if total_value > 0 else 0
            for sector, val in sector_totals.items()
        }

        total_pl = total_market - total_invested
        total_pl_pct = (total_pl / total_invested * 100) if total_invested > 0 else 0.0

        state = PortfolioState(
            account_id=account_id,
            account_name=account_name,
            total_value=total_value,
            cash=cash,
            invested=total_invested,
            positions=positions,
            total_pl=total_pl,
            total_pl_pct=total_pl_pct,
            sector_weights=sector_weights,
            timestamp=datetime.utcnow().isoformat(),
        )

        logger.info(
            "portfolio_state_loaded",
            account=account_name,
            total_value=total_value,
            positions=len(positions),
            cash=cash,
        )
        return state

    except Exception as e:
        logger.error("portfolio_state_build_failed", account_id=account_id, error=str(e))
        return _empty


def compute_cash_from_orders(
    ghostfolio: GhostfolioClient,
    account_id: str,
    initial_budget: float,
) -> float | None:
    """Compute correct cash balance from all Ghostfolio orders for an account.

    Self-healing: derives cash from the full order history, not any cached state.
    Formula: cash = initial_budget - Σ(buy_qty × price + fee) + Σ(sell_qty × price - fee)

    Returns None on error so the caller can fall back to a delta-based estimate.
    """
    try:
        orders = ghostfolio.list_orders()
        if isinstance(orders, dict):
            orders = orders.get("activities", [])

        cash = float(initial_budget)
        for o in orders:
            if o.get("accountId") != account_id:
                continue
            qty   = float(o.get("quantity",  0) or 0)
            price = float(o.get("unitPrice", 0) or 0)
            fee   = float(o.get("fee",       0) or 0)
            otype = o.get("type", "")
            if otype == "BUY":
                cash -= qty * price + fee
            elif otype == "SELL":
                cash += qty * price - fee

        result = max(0.0, cash)
        logger.info("cash_computed_from_orders", account_id=account_id, cash=round(result, 2))
        return result
    except Exception as e:
        logger.warning("cash_from_orders_failed", account_id=account_id, error=str(e))
        return None
