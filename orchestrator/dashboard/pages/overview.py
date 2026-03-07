"""Overview page: account cards, performance summary, next run times."""

import streamlit as st
from pathlib import Path
from datetime import datetime, timezone
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ghostfolio_client import GhostfolioClient
from src.audit_logger import AuditLogger
from dashboard.components.charts import render_performance_chart
from dashboard.config_utils import load_config


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

_DOW_ISO = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]  # 0=Mon … 6=Sun


def _dow_label(dow_str: str) -> str:
    """Convert APScheduler day_of_week field to readable label."""
    dow_str = dow_str.strip()
    if "-" in dow_str:
        a, b = dow_str.split("-", 1)
        return f"{_DOW_ISO[int(a)]}–{_DOW_ISO[int(b)]}"
    if "," in dow_str:
        return "/".join(_DOW_ISO[int(d)] for d in dow_str.split(","))
    if dow_str == "*":
        return "daily"
    return _DOW_ISO[int(dow_str)]


def cron_to_human(cron: str) -> str:
    """Convert a crontab string to a short human-readable description."""
    if not cron:
        return "no schedule"
    parts = cron.split()
    if len(parts) != 5:
        return cron

    minute, hour, dom, month, dow = parts

    # Build time string
    if "," in minute:
        mins = ", ".join(f":{m.zfill(2)}" for m in minute.split(","))
        time_str = f"{hour}:xx at {mins}"
    elif minute == "*/30":
        time_str = f"every 30 min"
    else:
        time_str = f"{hour}:{minute.zfill(2)}"

    # Build hour range for intraday
    if "-" in hour:
        h_start, h_end = hour.split("-")
        time_str = f"{h_start}:00–{h_end}:59 every 30 min"
        return f"{_dow_label(dow)} · {time_str} CET"

    # Build frequency string
    if dom != "*" and dom.isdigit():
        freq = f"Monthly (day {dom})"
    elif dow == "*":
        freq = "Daily"
    else:
        freq = _dow_label(dow)

    return f"{freq} · {time_str} CET"


def next_run_time(cron: str) -> str:
    """Return 'Next: Weekday Mon DD at HH:MM' using APScheduler."""
    if not cron:
        return ""
    try:
        from apscheduler.triggers.cron import CronTrigger
        parts = cron.split()
        if len(parts) != 5:
            return ""
        minute, hour, dom, month, dow = parts
        trigger = CronTrigger(
            minute=minute, hour=hour,
            day=dom, month=month, day_of_week=dow,
            timezone="Europe/Warsaw",
        )
        now = datetime.now(timezone.utc)
        nxt = trigger.get_next_fire_time(None, now)
        if nxt is None:
            return ""
        # Format as "Mon 25 Feb · 18:00"
        return "Next: " + nxt.strftime("%a %d %b · %H:%M")
    except Exception:
        return ""


config = load_config()
accounts = config.get("accounts", {})

if not accounts:
    st.title("AI Investment Orchestrator")
    st.warning("No accounts configured. Go to Account Management to create one.")
    st.stop()

# Fetch live account values from Ghostfolio.
# Ghostfolio's valueInBaseCurrency in the accounts list is NOT filtered per-account —
# it appears to report the total portfolio value across all accounts, making per-account
# P/L wrong. Instead we compute securities value ourselves:
#   securities = Σ (per-account orders filtered by accountId) × current market prices
# This requires one fetch each of list_accounts, list_orders, get_portfolio_holdings.
_live_values: dict[str, dict] = {}
_ghostfolio_ok = False
try:
    from src.ghostfolio_client import GhostfolioClient
    _gf = GhostfolioClient()

    # ── 1. Account cash balances ───────────────────────────────────────────────
    _acct_list = _gf.list_accounts()
    if isinstance(_acct_list, dict):
        _acct_list = _acct_list.get("accounts", [])
    _cash_by_id: dict[str, float] = {
        _a["id"]: float(_a.get("balance", 0) or 0)
        for _a in _acct_list if isinstance(_a, dict) and _a.get("id")
    }

    # ── 2. Current market prices from holdings ────────────────────────────────
    _holdings_raw = _gf.get_portfolio_holdings()
    if isinstance(_holdings_raw, dict):
        _holdings_raw = _holdings_raw.get("holdings", _holdings_raw)
    _price_map: dict[str, float] = {}
    _h_iter = _holdings_raw if isinstance(_holdings_raw, list) else (
        _holdings_raw.values() if isinstance(_holdings_raw, dict) else []
    )
    for _h in _h_iter:
        if not isinstance(_h, dict):
            continue
        _sp = _h.get("SymbolProfile") or {}
        _sym = _sp.get("symbol") or _h.get("symbol", "")
        if _sym and len(_sym) <= 10:
            _price_map[_sym] = float(_h.get("marketPrice", 0) or 0)

    # ── 3. Per-account order quantities ──────────────────────────────────────
    _all_orders = _gf.list_orders()
    if isinstance(_all_orders, dict):
        _all_orders = _all_orders.get("activities", [])

    # Group orders by accountId
    _orders_by_acct: dict[str, list] = {}
    for _o in _all_orders:
        _oid = _o.get("accountId", "")
        if _oid:
            _orders_by_acct.setdefault(_oid, []).append(_o)

    # ── 4. Compute per-account market value ──────────────────────────────────
    for _aid, _cash in _cash_by_id.items():
        _acct_orders = _orders_by_acct.get(_aid, [])

        # Aggregate net qty per symbol (BUY adds, SELL reduces)
        _agg: dict[str, dict] = {}
        for _o in _acct_orders:
            _sp = _o.get("SymbolProfile") or {}
            _sym = _sp.get("symbol") or _o.get("symbol", "")
            if not _sym:
                continue
            _qty = float(_o.get("quantity", 0) or 0)
            _price = float(_o.get("unitPrice", 0) or 0)
            _otype = (_o.get("type") or "").upper()
            if _sym not in _agg:
                _agg[_sym] = {"qty": 0.0, "avg_cost": 0.0}
            if _otype == "BUY":
                _total_cost = _agg[_sym]["avg_cost"] * _agg[_sym]["qty"] + _qty * _price
                _agg[_sym]["qty"] += _qty
                _agg[_sym]["avg_cost"] = _total_cost / _agg[_sym]["qty"] if _agg[_sym]["qty"] else 0
            elif _otype == "SELL":
                _agg[_sym]["qty"] = max(0.0, _agg[_sym]["qty"] - _qty)

        _securities = sum(
            _data["qty"] * (_price_map.get(_sym) or _data["avg_cost"])
            for _sym, _data in _agg.items()
            if _data["qty"] > 0.0001
        )

        _live_values[_aid] = {"total": _securities + _cash, "cash": _cash}

    _ghostfolio_ok = True
except Exception as _gf_err:
    _ghostfolio_ok = False
    _live_values = {}  # discard any partial data — use audit log consistently

# Account cards — grouped by strategy
trading_accounts = {k: v for k, v in accounts.items() if v.get("cycle_type") != "research"}

# Options strategies use audit DB for P/L (Ghostfolio can't price synthetic option assets)
_OPTIONS_STRATEGIES = {"wheel", "vertical_spreads"}

def _options_pl(account_key: str) -> float:
    """Read cumulative realized P/L from the options positions DB (authoritative source)."""
    try:
        from src.options.positions import OptionsPositionTracker
        tracker = OptionsPositionTracker()
        return tracker.get_total_realized_pl(account_key)
    except Exception:
        return 0.0

_options_pl_cache: dict[str, float] = {
    k: _options_pl(k)
    for k, v in trading_accounts.items()
    if v.get("strategy") in _OPTIONS_STRATEGIES
}

# Calculate total portfolio value across all trading accounts.
# For all accounts (including options) use live Ghostfolio cash balance.
# Options accounts: total = Ghostfolio balance (premiums received/debits paid already
# reflected there after today's fix) + tiny synthetic holdings value.
# Fallback for options: initial_budget + realized_pl if Ghostfolio data unavailable.
initial_budget = config.get("defaults", {}).get("initial_budget", 10000)
_total_value = 0.0
_total_accounts = 0
for _key, _acct in trading_accounts.items():
    _acct_budget = float(_acct.get("initial_budget", initial_budget))
    _aid = _acct.get("ghostfolio_account_id", "")
    _live = _live_values.get(_aid, {})
    if _live.get("total"):
        _total_value += _live["total"]
        _total_accounts += 1
    elif _acct.get("strategy") in _OPTIONS_STRATEGIES:
        # Ghostfolio unavailable — fall back to initial + realized P/L
        _total_value += _acct_budget + _options_pl_cache.get(_key, 0.0)
        _total_accounts += 1

# Title row with total balance
_title_col, _metric_col = st.columns([3, 1])
with _title_col:
    st.title("AI Investment Orchestrator")
with _metric_col:
    if _total_accounts > 0:
        _total_budget = initial_budget * _total_accounts
        _total_pl_pct = (_total_value - _total_budget) / _total_budget * 100 if _total_budget else None
        st.metric(
            "Total Portfolio",
            f"${_total_value:,.2f}",
            delta=f"{_total_pl_pct:+.2f}%" if _total_pl_pct is not None else None,
        )
    else:
        st.metric("Total Portfolio", "—")

if not _ghostfolio_ok:
    st.error("Ghostfolio niedostępny — wartości kont mogą być nieaktualne (dane z ostatniego cyklu)")

audit = AuditLogger()

# Group accounts by strategy
_STRATEGY_LABELS = {
    "core_satellite": "Core-Satellite",
    "value_investing": "Value Investing",
    "momentum": "Momentum",
    "wheel": "Wheel Strategy",
    "vertical_spreads": "Options Spreads",
}
groups: dict[str, list[tuple[str, dict]]] = {}
for key, acct in trading_accounts.items():
    strategy = acct.get("strategy", "other")
    # Separate intraday momentum from daily momentum
    if strategy == "momentum" and acct.get("cycle_type") == "intraday":
        strategy = "intraday"
    groups.setdefault(strategy, []).append((key, acct))

_GROUP_ORDER = ["core_satellite", "value_investing", "momentum", "intraday", "wheel", "vertical_spreads"]
_GROUP_ICONS = {
    "core_satellite": "🏛️",
    "value_investing": "💎",
    "momentum": "🚀",
    "intraday": "⚡",
    "wheel": "🎡",
    "vertical_spreads": "📈",
}

for strategy_key in _GROUP_ORDER:
    acct_list = groups.get(strategy_key)
    if not acct_list:
        continue

    icon = _GROUP_ICONS.get(strategy_key, "📊")
    label = _STRATEGY_LABELS.get(strategy_key, strategy_key.replace("_", " ").title())
    if strategy_key == "intraday":
        label = "Intraday Momentum"
    st.subheader(f"{icon} {label}")

    cols = st.columns(len(acct_list))
    for i, (key, acct) in enumerate(acct_list):
        with cols[i]:
            name = acct.get("name", key)
            model = acct.get("model", "Unknown")
            cron = acct.get("cron", "")

            logs = audit.get_recent_logs(account_key=key, limit=1)
            latest = logs[0] if logs else {}

            last_regime = latest.get("market_regime", "N/A")
            success = latest.get("success", 1)

            acct_id = acct.get("ghostfolio_account_id", "")
            live = _live_values.get(acct_id, {})
            strategy = acct.get("strategy", "")
            acct_budget = float(acct.get("initial_budget", initial_budget))
            if live.get("total"):
                # Use Ghostfolio balance for all accounts (options balance now correct after fix)
                value = live["total"]
                pl_pct = (value - acct_budget) / acct_budget * 100 if acct_budget else None
            elif strategy in _OPTIONS_STRATEGIES:
                # Ghostfolio unavailable — fall back to initial + realized P/L
                realized_pl = _options_pl_cache.get(key, 0.0)
                value = acct_budget + realized_pl
                pl_pct = (realized_pl / acct_budget * 100) if acct_budget else None
            else:
                value = latest.get("portfolio_value")
                pl_pct = latest.get("portfolio_pl_pct")

            st.markdown(f"**{name}**")
            if value is not None:
                delta = f"{pl_pct:+.2f}%" if pl_pct is not None else None
                st.metric("Portfolio Value", f"${value:,.2f}", delta=delta)
            else:
                st.metric("Portfolio Value", f"${initial_budget:,.2f}", delta="New")

            st.caption(f"Model: **{model}**")
            human = cron_to_human(cron)
            nxt = next_run_time(cron)
            st.caption(f"🕐 {human}")
            if nxt:
                st.caption(f"📅 {nxt}")
            st.caption(f"Market: {last_regime}")

            status = "OK" if success else "ERROR"
            if success:
                st.success(f"Status: {status}")
            else:
                st.error(f"Status: {status}")

st.divider()

# Latest decisions
st.subheader("Latest Decisions")
all_logs = audit.get_recent_logs(limit=10)

if all_logs:
    for log in all_logs:
        ts = log.get("timestamp", "")[:16]
        acct_name = log.get("account_name", "Unknown")
        outlook = log.get("portfolio_outlook", "N/A")
        confidence = log.get("confidence")
        n_actions = log.get("actions_count", 0)
        n_forced = log.get("forced_actions_count", 0)
        n_rejected = log.get("rejected_count", 0)
        error = log.get("error")

        if error:
            st.error(f"**{ts}** | {acct_name} | ERROR: {error}")
        else:
            conf_str = f"Confidence: {confidence:.2f}" if confidence else ""
            st.info(
                f"**{ts}** | {acct_name} | "
                f"Outlook: {outlook} | {conf_str} | "
                f"Trades: {n_actions} | Forced: {n_forced} | Rejected: {n_rejected}"
            )
else:
    st.info("No decision logs yet. Waiting for first cycle to run.")

st.divider()

# Performance chart placeholder
st.subheader("Performance Comparison")
render_performance_chart(config, audit)
