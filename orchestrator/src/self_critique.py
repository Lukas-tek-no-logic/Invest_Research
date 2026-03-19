"""Verbal Reinforcement: weekly self-critique that generates investment beliefs.

Inspired by FinCon (NeurIPS 2024) — the agent reviews its own past decisions,
identifies patterns in wins and losses, and distills reusable beliefs that are
injected into future Pass 2 prompts.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import structlog

from .llm_client import LLMClient
from .audit_logger import AuditLogger

logger = structlog.get_logger()

BELIEFS_DIR = Path("data/beliefs")
MAX_BELIEFS = 15  # keep beliefs list focused


class ReflectionEngine:
    """Analyze past decisions and generate investment beliefs."""

    def __init__(self, llm: LLMClient, audit: AuditLogger):
        self.llm = llm
        self.audit = audit
        BELIEFS_DIR.mkdir(parents=True, exist_ok=True)

    def run_reflection(
        self,
        account_key: str,
        model: str = "QWEN3.5",
        fallback_model: str | None = "Nemotron",
        num_cycles: int = 10,
    ) -> dict:
        """Analyze last N cycles and generate/update beliefs."""
        logs = self.audit.get_recent_logs(account_key, limit=num_cycles)
        if len(logs) < 3:
            logger.info("reflection_skipped_too_few_cycles", account=account_key, cycles=len(logs))
            return {}

        # Build reflection context from audit logs
        cycle_summaries = []
        for log in logs:
            summary = {
                "date": log.get("timestamp", "")[:10],
                "outlook": log.get("portfolio_outlook"),
                "confidence": log.get("confidence"),
                "regime": log.get("market_regime"),
                "trades": log.get("actions_count", 0),
                "forced": log.get("forced_actions_count", 0),
                "rejected": log.get("rejected_count", 0),
                "portfolio_value": log.get("portfolio_value"),
                "pl_pct": log.get("portfolio_pl_pct"),
                "cash": log.get("cash"),
                "error": log.get("error"),
            }
            # Load detailed log file for trade details
            log_file = log.get("log_file")
            if log_file:
                try:
                    log_path = Path(log_file)
                    if log_path.exists():
                        with open(log_path) as f:
                            detail = json.load(f)
                        trades = detail.get("executed_trades", [])
                        summary["trades_detail"] = [
                            {
                                "type": t.get("type"),
                                "symbol": t.get("symbol"),
                                "total": t.get("total"),
                                "success": t.get("success"),
                            }
                            for t in trades
                        ]
                        p2 = detail.get("pass2", {}).get("response", {})
                        summary["reasoning"] = (p2.get("reasoning") or "")[:500]
                except Exception:
                    pass
            cycle_summaries.append(summary)

        # Load existing beliefs for context
        existing = self.load_beliefs(account_key)
        existing_text = ""
        if existing.get("beliefs"):
            existing_text = (
                "\n== YOUR CURRENT BELIEFS ==\n"
                + "\n".join(f"- {b}" for b in existing["beliefs"])
                + "\n\nUpdate, remove outdated, or add new beliefs based on the evidence below.\n"
            )

        system_prompt = (
            "You are an investment strategist reviewing your own past trading decisions. "
            "Your goal is to identify PATTERNS — what works, what fails, and why.\n\n"
            "Analyze the cycle history below and produce a set of actionable investment beliefs. "
            "Each belief should be:\n"
            "- Specific and testable (not vague like 'be careful')\n"
            "- Grounded in evidence from the data (cite dates/symbols)\n"
            "- Actionable in future decisions\n\n"
            "Examples of good beliefs:\n"
            "- 'Small-caps with >15% short interest lose money in HIGH_VOLATILITY — avoid'\n"
            "- 'Trimming winners at +30% and redeploying beats holding to +50% target'\n"
            "- 'Energy sector outperforms when VIX >25 — overweight XLE/XOM in high-vol'\n\n"
            f"Return at most {MAX_BELIEFS} beliefs. Drop beliefs that the data contradicts.\n\n"
            "Respond with JSON:\n"
            "{\n"
            '  "reflection": "2-3 sentence summary of what went well and poorly",\n'
            '  "beliefs": ["belief 1", "belief 2", ...],\n'
            '  "dropped_beliefs": ["old belief that data contradicts", ...]\n'
            "}\n"
        )

        user_prompt = (
            f"== ACCOUNT: {account_key} ==\n"
            f"== REVIEW PERIOD: {cycle_summaries[-1]['date']} to {cycle_summaries[0]['date']} ==\n"
            f"== CYCLES REVIEWED: {len(cycle_summaries)} ==\n\n"
            f"{existing_text}\n"
            f"== CYCLE HISTORY (newest first) ==\n"
            f"{json.dumps(cycle_summaries, indent=2, default=str)}\n\n"
            "Analyze patterns and respond with your updated beliefs JSON."
        )

        try:
            result = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
                fallback_model=fallback_model,
                temperature=0.7,
            )

            beliefs = result.get("beliefs", [])[:MAX_BELIEFS]
            reflection = result.get("reflection", "")

            # Save beliefs
            belief_data = {
                "account_key": account_key,
                "updated": datetime.now().isoformat(),
                "reflection": reflection,
                "beliefs": beliefs,
                "dropped": result.get("dropped_beliefs", []),
                "cycles_reviewed": len(cycle_summaries),
            }
            self._save_beliefs(account_key, belief_data)

            logger.info(
                "reflection_complete",
                account=account_key,
                beliefs_count=len(beliefs),
                dropped=len(result.get("dropped_beliefs", [])),
            )
            return belief_data

        except Exception as e:
            logger.error("reflection_failed", account=account_key, error=str(e))
            return {}

    def load_beliefs(self, account_key: str) -> dict:
        """Load saved beliefs for an account."""
        path = BELIEFS_DIR / f"{account_key}.json"
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_beliefs(self, account_key: str, data: dict) -> None:
        path = BELIEFS_DIR / f"{account_key}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
