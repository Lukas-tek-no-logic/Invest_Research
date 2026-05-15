from types import SimpleNamespace
from unittest.mock import MagicMock

import yaml

from orchestrator.src import main as main_module
from orchestrator.src.portfolio_state import PortfolioState


def _write_config(tmp_path, config: dict):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _make_portfolio() -> PortfolioState:
    return PortfolioState(
        account_id="acct-1",
        account_name="Test",
        total_value=10000,
        cash=5000,
        invested=5000,
        positions=[],
        total_pl=0,
        total_pl_pct=0,
        sector_weights={},
        timestamp="2026-04-15T00:00:00",
    )


def _make_orchestrator(config_path, dry_run=False):
    orch = object.__new__(main_module.Orchestrator)
    orch.config_path = str(config_path)
    orch.dry_run = dry_run
    orch._last_cycle_prices = {}
    orch.ghostfolio = MagicMock()
    orch.llm = MagicMock()
    orch.market_data = MagicMock()
    orch.news = MagicMock()
    orch.audit = MagicMock()
    orch.account_mgr = MagicMock()
    orch.reflection = MagicMock()
    orch.rag = MagicMock()
    orch._load_config()
    return orch


class StubWatchlistManager:
    def __init__(self, account_key: str, core: list[str]):
        self.core = list(core)

    def get_full_watchlist(self) -> list[str]:
        return list(self.core)

    def save_suggestions(self, symbols: list[str]) -> None:
        self.saved = list(symbols)


def test_run_cycle_rules_mode_initializes_audit_fields(tmp_path, monkeypatch):
    config = {
        "defaults": {"initial_budget": 10000},
        "accounts": {
            "rules_account": {
                "name": "Rules Account",
                "strategy": "nonexistent",
                "decision_mode": "rules",
                "watchlist": ["SPY"],
                "ghostfolio_account_id": "acct-1",
                "model": "Nemotron",
                "risk_profile": {},
                "rules_params": {},
            }
        },
    }
    config_path = _write_config(tmp_path, config)
    orch = _make_orchestrator(config_path)

    quote = SimpleNamespace(
        price=500.0,
        change_pct=0.0,
        pe_ratio=None,
        dividend_yield=None,
        week52_high=None,
        week52_low=None,
        sector="ETF",
        short_pct_float=None,
    )
    orch.market_data.get_quotes_batch.return_value = {"SPY": quote}
    orch.market_data.get_history.return_value = SimpleNamespace(empty=True)
    orch.market_data.get_market_overview.return_value = {}
    orch.market_data.format_upcoming_earnings.return_value = ""
    orch.market_data.get_upcoming_earnings.return_value = {}
    orch.news.fetch_relevant_news.return_value = []
    orch.news.format_for_prompt.return_value = ""
    orch.audit.get_decision_history.return_value = []

    captured = {}

    def fake_log_cycle(**kwargs):
        captured.update(kwargs)
        return "audit.json"

    orch.audit.log_cycle.side_effect = fake_log_cycle

    monkeypatch.setattr(main_module, "WatchlistManager", StubWatchlistManager)
    monkeypatch.setattr(main_module, "get_portfolio_state", lambda *args, **kwargs: _make_portfolio())
    monkeypatch.setattr(main_module, "get_fundamentals_batch", lambda **kwargs: {})
    monkeypatch.setattr(main_module, "format_fundamentals_for_prompt", lambda *args, **kwargs: "")
    monkeypatch.setattr(main_module, "format_decision_history", lambda history: "")
    monkeypatch.setattr(main_module, "compute_cash_from_orders", lambda *args, **kwargs: 5000.0)
    monkeypatch.setattr(main_module.ResearchAgent, "load_today", lambda: None)

    orch.run_cycle("rules_account")

    assert captured["account_key"] == "rules_account"
    assert captured["model"] == "Nemotron"
    assert captured["pass1_messages"] == []
    assert captured["pass2_messages"] == []
    assert captured["error"] is None


def test_options_maintenance_passes_market_data_and_account_id(tmp_path, monkeypatch):
    config = {
        "accounts": {
            "wheel_account": {
                "name": "Wheel",
                "strategy": "wheel",
                "ghostfolio_account_id": "wheel-id",
                "risk_profile": {},
            },
            "spreads_account": {
                "name": "Spreads",
                "strategy": "vertical_spreads",
                "ghostfolio_account_id": "spreads-id",
                "risk_profile": {},
            },
        }
    }
    config_path = _write_config(tmp_path, config)
    orch = _make_orchestrator(config_path)

    position = SimpleNamespace(
        id=1,
        symbol="SPY",
        profit_captured_pct=None,
        dte=None,
        spread_type="CASH_SECURED_PUT",
        sell_strike=0.0,
    )

    class FakeTracker:
        def get_active_positions(self, account_key):
            return [position]

    option_calls = []
    spread_calls = []

    class FakeOptionsExecutor:
        def __init__(self, **kwargs):
            option_calls.append(kwargs)

        def update_active_positions(self, active_positions):
            return [SimpleNamespace(success=True)]

        def execute_closes(self, close_actions, active_positions):
            return []

    class FakeSpreadsExecutor:
        def __init__(self, **kwargs):
            spread_calls.append(kwargs)

        def update_active_positions(self, active_positions):
            return [SimpleNamespace(success=True)]

        def execute_closes(self, close_actions, active_positions):
            return []

    monkeypatch.setattr(main_module, "OptionsPositionTracker", FakeTracker)
    monkeypatch.setattr(main_module, "OptionsExecutor", FakeOptionsExecutor)
    monkeypatch.setattr(main_module, "SpreadsExecutor", FakeSpreadsExecutor)

    orch.run_options_maintenance()

    assert option_calls[0]["market_data"] is orch.market_data
    assert option_calls[0]["account_id"] == "wheel-id"
    assert spread_calls[0]["market_data"] is orch.market_data
    assert spread_calls[0]["account_id"] == "spreads-id"
