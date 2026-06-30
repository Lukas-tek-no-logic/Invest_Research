"""Regression tests for carrying forward the last known portfolio valuation.

When Ghostfolio's valuation endpoints are down, get_portfolio_state() returns an
empty state (total_value=0). That 0 must NOT overwrite the last valid snapshot in
the audit log, otherwise the dashboard shows misleading $0.00 cards.
"""

from src.audit_logger import AuditLogger


def _make_logger(tmp_path):
    return AuditLogger(logs_dir=tmp_path / "logs", db_path=tmp_path / "audit.db")


def test_zero_valuation_carries_forward_last_known(tmp_path):
    audit = _make_logger(tmp_path)

    # Cycle 1: Ghostfolio healthy → real valuation persisted.
    audit.log_cycle(
        account_key="acct1",
        account_name="Account 1",
        model="m",
        portfolio_after={"total_value": 9500.0, "total_pl_pct": -5.0, "cash": 200.0},
    )

    # Cycle 2: Ghostfolio valuation down → empty state with total_value=0.
    audit.log_cycle(
        account_key="acct1",
        account_name="Account 1",
        model="m",
        portfolio_after={"total_value": 0, "total_pl_pct": 0, "cash": 0},
    )

    last = audit._last_known_valuation("acct1")
    assert last is not None
    value, pl_pct, cash = last
    # Cycle 2 must have stored the carried-forward value, not 0.
    assert value == 9500.0
    assert pl_pct == -5.0
    assert cash == 200.0


def test_no_prior_valuation_keeps_zero(tmp_path):
    audit = _make_logger(tmp_path)
    # First ever cycle fails to value → nothing to carry forward.
    audit.log_cycle(
        account_key="fresh",
        account_name="Fresh",
        model="m",
        portfolio_after={"total_value": 0, "total_pl_pct": 0, "cash": 0},
    )
    assert audit._last_known_valuation("fresh") is None


def test_real_valuation_is_not_overwritten(tmp_path):
    audit = _make_logger(tmp_path)
    audit.log_cycle(
        account_key="acct1", account_name="A", model="m",
        portfolio_after={"total_value": 9500.0, "total_pl_pct": -5.0, "cash": 200.0},
    )
    audit.log_cycle(
        account_key="acct1", account_name="A", model="m",
        portfolio_after={"total_value": 10200.0, "total_pl_pct": 2.0, "cash": 300.0},
    )
    value, pl_pct, cash = audit._last_known_valuation("acct1")
    assert value == 10200.0
    assert pl_pct == 2.0
