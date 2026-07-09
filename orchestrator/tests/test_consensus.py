"""Tests for Pass 2 self-consistency merging (consensus.py)."""

from unittest.mock import MagicMock

import pytest

from orchestrator.src.consensus import chat_json_consensus, merge_decision_samples


def _sample(actions: list[dict], confidence: float = 0.7, **extra) -> dict:
    return {"actions": actions, "confidence": confidence,
            "portfolio_outlook": "NEUTRAL", **extra}


BUY_SPY = {"type": "BUY", "symbol": "SPY", "amount_usd": 2000, "thesis": "trend up"}
BUY_QQQ = {"type": "BUY", "symbol": "QQQ", "amount_usd": 1500, "thesis": "momentum"}
SELL_VTI = {"type": "SELL", "symbol": "VTI", "amount_usd": 800, "thesis": "weak"}


class TestMergeDecisionSamples:
    def test_majority_action_survives(self):
        merged = merge_decision_samples([
            _sample([BUY_SPY, BUY_QQQ]),
            _sample([BUY_SPY]),
            _sample([BUY_SPY, SELL_VTI]),
        ])
        keys = [(a["type"], a["symbol"]) for a in merged["actions"]]
        assert ("BUY", "SPY") in keys          # 3/3 votes
        assert ("BUY", "QQQ") not in keys      # 1/3 — dropped
        assert ("SELL", "VTI") not in keys     # 1/3 — dropped
        assert merged["_consensus"]["kept"] == 1
        assert len(merged["_consensus"]["dropped"]) == 2

    def test_two_of_three_survives(self):
        merged = merge_decision_samples([
            _sample([BUY_SPY, BUY_QQQ]),
            _sample([BUY_QQQ]),
            _sample([SELL_VTI]),
        ])
        keys = [(a["type"], a["symbol"]) for a in merged["actions"]]
        assert keys == [("BUY", "QQQ")]

    def test_amounts_merged_as_median(self):
        merged = merge_decision_samples([
            _sample([{**BUY_SPY, "amount_usd": 1000}]),
            _sample([{**BUY_SPY, "amount_usd": 2000}]),
            _sample([{**BUY_SPY, "amount_usd": 5000}]),
        ])
        assert merged["actions"][0]["amount_usd"] == 2000

    def test_confidence_averaged(self):
        merged = merge_decision_samples([
            _sample([BUY_SPY], confidence=0.9),
            _sample([BUY_SPY], confidence=0.6),
            _sample([BUY_SPY], confidence=0.6),
        ])
        assert merged["confidence"] == 0.7

    def test_close_actions_keyed_by_position_id(self):
        close_1 = {"type": "CLOSE", "symbol": "SPY", "position_id": 1}
        close_2 = {"type": "CLOSE", "symbol": "SPY", "position_id": 2}
        merged = merge_decision_samples([
            _sample([close_1, close_2]),
            _sample([close_1]),
            _sample([close_2]),
        ])
        ids = {a["position_id"] for a in merged["actions"]}
        assert ids == {1, 2}   # each has 2/3 votes independently

    def test_spread_type_distinguishes_actions(self):
        bull = {"type": "OPEN_SPREAD", "symbol": "SPY", "spread_type": "bull_call", "contracts": 1}
        bear = {"type": "OPEN_SPREAD", "symbol": "SPY", "spread_type": "bear_call", "contracts": 1}
        merged = merge_decision_samples([
            _sample([bull]),
            _sample([bull]),
            _sample([bear]),
        ])
        assert [a["spread_type"] for a in merged["actions"]] == ["bull_call"]

    def test_single_sample_passthrough(self):
        s = _sample([BUY_SPY, BUY_QQQ])
        assert merge_decision_samples([s]) is s

    def test_non_action_fields_preserved(self):
        merged = merge_decision_samples([
            _sample([BUY_SPY], market_comment="cautious"),
            _sample([BUY_SPY]),
            _sample([BUY_SPY]),
        ])
        assert merged["market_comment"] == "cautious"
        assert merged["portfolio_outlook"] == "NEUTRAL"


class TestChatJsonConsensus:
    def test_samples_le_1_is_plain_call(self):
        llm = MagicMock()
        llm.chat_json.return_value = _sample([BUY_SPY])
        result = chat_json_consensus(llm, messages=[], model="m", fallback_model=None, samples=1)
        assert llm.chat_json.call_count == 1
        assert "_consensus" not in result

    def test_three_samples_merged(self):
        llm = MagicMock()
        llm.chat_json.side_effect = [
            _sample([BUY_SPY, BUY_QQQ]),
            _sample([BUY_SPY]),
            _sample([BUY_SPY]),
        ]
        result = chat_json_consensus(llm, messages=[], model="m", fallback_model=None, samples=3)
        assert llm.chat_json.call_count == 3
        assert [(a["type"], a["symbol"]) for a in result["actions"]] == [("BUY", "SPY")]

    def test_failed_samples_tolerated(self):
        llm = MagicMock()
        llm.chat_json.side_effect = [
            _sample([BUY_SPY]),
            RuntimeError("LLM down"),
            _sample([BUY_SPY]),
        ]
        result = chat_json_consensus(llm, messages=[], model="m", fallback_model=None, samples=3)
        # 2 successful samples, min_votes=2, SPY has 2 → survives
        assert [(a["type"], a["symbol"]) for a in result["actions"]] == [("BUY", "SPY")]

    def test_all_samples_failed_raises(self):
        llm = MagicMock()
        llm.chat_json.side_effect = RuntimeError("LLM down")
        with pytest.raises(RuntimeError, match="All 3"):
            chat_json_consensus(llm, messages=[], model="m", fallback_model=None, samples=3)
