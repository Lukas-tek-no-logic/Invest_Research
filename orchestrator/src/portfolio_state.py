"""Aggregate portfolio state from Ghostfolio for a specific account."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import structlog

from .ghostfolio_client import GhostfolioClient

logger = structlog.get_logger()

# Cash-equivalent ETFs (T-bills). These are a mechanical cash parking lot, not
# an investment position: they are split out of `positions`, folded into
# `available_cash`, and never shown to the LLM as holdings.
CASH_EQUIVALENT_SYMBOLS = {"BIL", "SGOV"}


class ValuationUnavailable(RuntimeError):
    """Ghostfolio valuation could not be fetched or built — do not trade.

    Raised instead of returning a zero-value PortfolioState so cycles abort
    (and may retry) rather than acting on a fabricated $0 portfolio.
    """


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
    # True when Ghostfolio had no market price and the valuation fell back to
    # avg_cost.  Such a position reports P/L 0.0% by construction, so every
    # P/L-driven rule (stop-loss, guardian) is blind to it — callers must skip
    # those rules rather than act on the fabricated 0%.
    price_stale: bool = False


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
    # T-bill parking positions (BIL/SGOV) — excluded from `positions`, treated as cash
    cash_equivalents: list[Position] = field(default_factory=list)
    # symbol → share count sold beyond what the account held.  Ghostfolio keeps
    # the phantom short; the ledger clamps it to zero so it stays invisible to
    # every rule.  Surfaced here so cycles can log and refuse to compound it.
    oversold: dict[str, float] = field(default_factory=dict)
    # Return since inception: (cash + securities) vs the account's starting
    # budget.  Distinct from `total_pl_pct`, which measures only currently-held
    # lots and therefore drops realised P/L on every full exit.
    initial_budget: float | None = None
    total_return_pct: float | None = None

    @property
    def cash_equivalent_value(self) -> float:
        return sum(p.market_value for p in self.cash_equivalents)

    @property
    def available_cash(self) -> float:
        """Spendable cash: raw balance + T-bill parking (auto-liquidated on demand)."""
        return self.cash + self.cash_equivalent_value

    @property
    def cash_pct(self) -> float:
        if self.total_value <= 0:
            return 100.0
        return (self.available_cash / self.total_value) * 100

    @property
    def position_count(self) -> int:
        return len(self.positions)

    def get_position(self, symbol: str) -> Position | None:
        for p in self.positions:
            if p.symbol == symbol:
                return p
        return None

    def get_holding(self, symbol: str) -> Position | None:
        """Look up a symbol across investment positions AND T-bill parking.

        `get_position` deliberately hides BIL/SGOV so the LLM never sees them as
        holdings.  Anything that needs the true share count — above all clamping
        a SELL to what the account actually owns — must use this instead, or the
        sweep-out of parked cash looks like a zero-share position.
        """
        for p in self.positions:
            if p.symbol == symbol:
                return p
        for p in self.cash_equivalents:
            if p.symbol == symbol:
                return p
        return None

    def holdings_map(self) -> dict[str, float]:
        """symbol → owned share count, including T-bill parking."""
        return {p.symbol: p.quantity for p in (*self.positions, *self.cash_equivalents)}

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
            cash_equivalents=self.cash_equivalents,
            oversold=self.oversold,
            initial_budget=self.initial_budget,
            total_return_pct=self.total_return_pct,
        )

    def to_prompt_text(self) -> str:
        """Format portfolio state for LLM prompt."""
        lines = [
            f"== PORTFOLIO STATE ({self.account_name}) ==",
            f"Total Value: ${self.total_value:,.2f}",
            f"Cash: ${self.available_cash:,.2f} ({self.cash_pct:.1f}%)",
            f"Invested: ${self.invested:,.2f}",
            f"Open-position P/L: ${self.total_pl:,.2f} ({self.total_pl_pct:+.2f}%)",
        ]
        # Return since inception is the only figure that survives full exits.
        # Without it the model reads "+0.00%" on an account that is down 12%.
        if self.total_return_pct is not None and self.initial_budget:
            lines.append(
                f"TOTAL RETURN since inception: {self.total_return_pct:+.2f}% "
                f"(${self.total_value:,.2f} vs ${self.initial_budget:,.2f} start) "
                f"— this includes realised results from closed trades"
            )
        lines += [
            f"Positions: {self.position_count}",
            "",
        ]
        if self.positions:
            lines.append("Holdings:")
            for p in sorted(self.positions, key=lambda x: x.market_value, reverse=True):
                stale = "  [!] NO MARKET PRICE — P/L unknown" if p.price_stale else ""
                lines.append(
                    f"  {p.symbol}: {p.quantity:.4f} shares @ avg ${p.avg_cost:.2f} "
                    f"| now ${p.current_price:.2f} | value ${p.market_value:,.2f} "
                    f"| P/L {p.unrealized_pl_pct:+.1f}% | weight {p.weight_pct:.1f}% "
                    f"| sector: {p.sector}{stale}"
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
    initial_budget: float | None = None,
) -> PortfolioState:
    """Build a PortfolioState from Ghostfolio API data.

    Uses:
      - account list  → cash balance + total value (valueInBaseCurrency)
      - order list    → per-account positions (filtered by accountId)
      - holdings list → current market prices (matched by symbol)
    """
    try:
        accounts_raw = client.list_accounts()
        orders_raw = client.list_orders()
        holdings_raw = client.get_portfolio_holdings()
    except Exception as e:
        logger.error("portfolio_state_fetch_failed", account_id=account_id, error=str(e))
        raise ValuationUnavailable(
            f"Ghostfolio fetch failed for account {account_name} ({account_id}): {e}"
        ) from e

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
        if account_info is None:
            raise ValuationUnavailable(
                f"Account {account_name} ({account_id}) not found in Ghostfolio account list"
            )

        cash = float(account_info.get("balance", 0) or 0)
        # valueInBaseCurrency = securities market value only (NOT cash).
        # Exception: for accounts with no real positions Ghostfolio echoes the cash
        # balance here, which would double-count if we then added cash again.
        # Detect this by checking if api_total ≈ cash (within 0.5%).
        api_total_raw = float(account_info.get("valueInBaseCurrency", 0) or 0)
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
        # Replaying a ledger demands chronological order: Ghostfolio does not
        # promise one, and a SELL seen before its BUY is silently dropped (qty is
        # still 0, so the reduction is lost and the cost basis stays untouched).
        acct_orders.sort(key=lambda o: str(o.get("date", "")))

        # Aggregate BUY / SELL per symbol
        agg: dict[str, dict] = {}  # symbol → {qty, total_cost, first_buy, oversold}
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
                agg[sym] = {"qty": 0.0, "total_cost": 0.0, "first_buy": None, "oversold": 0.0}
            if order_type == "BUY":
                agg[sym]["qty"] += qty
                agg[sym]["total_cost"] += qty * price
                # Holding period runs from the FIRST buy, not from whichever
                # order Ghostfolio happened to list first.
                if agg[sym]["first_buy"] is None or order_date < agg[sym]["first_buy"]:
                    agg[sym]["first_buy"] = order_date
            elif order_type == "SELL":
                # Reduce qty; proportionally reduce cost basis
                if agg[sym]["qty"] > 0:
                    sell_fraction = min(qty / agg[sym]["qty"], 1.0)
                    agg[sym]["total_cost"] *= (1 - sell_fraction)
                remaining = agg[sym]["qty"] - qty
                if remaining < 0:
                    # Sold more than the account held.  Clamping to zero here is
                    # what let this deficit accumulate unseen across round-trips:
                    # the next BUY starts from 0 and the next full exit oversells
                    # again.  Keep the clamp (a negative "position" would break
                    # every downstream rule) but record the shortfall.
                    agg[sym]["oversold"] += -remaining
                agg[sym]["qty"] = max(0.0, remaining)

        positions: list[Position] = []
        cash_equivalents: list[Position] = []
        total_market = 0.0
        total_invested = 0.0
        sector_totals: dict[str, float] = {}

        # Collect oversell deficits first — a fully-closed symbol is skipped
        # below, and those are exactly the ones carrying phantom shorts.
        # Synthetic GF_ symbols are excluded: credit structures legitimately
        # book their opening SELL before any BUY, so "oversold" is their normal
        # lifecycle, not a ledger fault.
        oversold = {
            sym: round(data["oversold"], 6)
            for sym, data in agg.items()
            if data.get("oversold", 0) > 1e-6 and not sym.startswith("GF_")
        }
        if oversold:
            logger.warning(
                "phantom_short_detected",
                account=account_name,
                symbols=oversold,
                note="Ghostfolio holds SELLs beyond owned quantity; cash was credited for shares never held",
            )

        for sym, data in agg.items():
            qty = data["qty"]
            if qty < 0.0001:
                continue

            avg_cost = data["total_cost"] / qty if qty > 0 else 0.0
            info = price_map.get(sym, {})
            quoted_price = float(info.get("price", 0.0) or 0.0)
            price_stale = quoted_price <= 0
            current_price = quoted_price or avg_cost  # fallback to cost
            market_value = current_price * qty
            investment = avg_cost * qty
            unrealized_pl = market_value - investment
            unrealized_pl_pct = (unrealized_pl / investment * 100) if investment > 0 else 0.0
            sector = info.get("sector", "Unknown")

            pos = Position(
                symbol=sym,
                name=info.get("name", sym),
                quantity=qty,
                avg_cost=avg_cost,
                current_price=current_price,
                market_value=market_value,
                unrealized_pl=unrealized_pl,
                unrealized_pl_pct=unrealized_pl_pct,
                sector=sector,
                first_buy_date=data.get("first_buy"),
                price_stale=price_stale,
            )
            if price_stale and not sym.startswith("GF_"):
                # GF_ synthetics have no market quote by design — their marks
                # live in options_positions, not in Ghostfolio prices.
                logger.warning(
                    "position_price_missing",
                    account=account_name,
                    symbol=sym,
                    note="valued at avg_cost; P/L reads 0% so stop-loss cannot fire",
                )
            total_market += market_value
            if sym in CASH_EQUIVALENT_SYMBOLS:
                # T-bill parking: counts toward total value but is not an
                # investment position — the LLM sees it as cash.
                cash_equivalents.append(pos)
                continue
            positions.append(pos)
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

        # P/L on investment positions only — T-bill parking is cash, not P/L
        equity_market = total_market - sum(p.market_value for p in cash_equivalents)
        total_pl = equity_market - total_invested
        total_pl_pct = (total_pl / total_invested * 100) if total_invested > 0 else 0.0

        # Return since inception. `total_pl_pct` above only sees lots still held,
        # so a rotating account reports ~0% no matter how much it has lost.
        total_return_pct = (
            (total_value / initial_budget - 1) * 100
            if initial_budget and initial_budget > 0
            else None
        )

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
            cash_equivalents=cash_equivalents,
            oversold=oversold,
            initial_budget=initial_budget,
            total_return_pct=total_return_pct,
        )

        logger.info(
            "portfolio_state_loaded",
            account=account_name,
            total_value=total_value,
            positions=len(positions),
            cash=cash,
            total_return_pct=(round(total_return_pct, 2) if total_return_pct is not None else None),
            oversold_symbols=len(oversold),
        )
        return state

    except ValuationUnavailable:
        raise
    except Exception as e:
        logger.error("portfolio_state_build_failed", account_id=account_id, error=str(e))
        raise ValuationUnavailable(
            f"Portfolio state build failed for account {account_name} ({account_id}): {e}"
        ) from e


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
