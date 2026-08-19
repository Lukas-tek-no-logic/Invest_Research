"""Parse and validate LLM responses for Pass 1 (analysis) and Pass 2 (decisions)."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator, model_validator
import structlog

logger = structlog.get_logger()

# Same shape rule as watchlist_manager: 1-5 letters, optional class suffix (BRK-B).
TICKER_RE = re.compile(r"^[A-Z]{1,5}(-[A-Z]{1,2})?$")


# --- Pass 1: Analysis Models ---

class PortfolioHealth(BaseModel):
    diversification: str = "GOOD"
    risk_level: str = "MEDIUM"
    issues: list[str] = Field(default_factory=list)


class Opportunity(BaseModel):
    symbol: str
    signal: str


class Threat(BaseModel):
    description: str


class AnalysisResult(BaseModel):
    """Parsed result from Pass 1 (market analysis)."""
    market_regime: str = "SIDEWAYS"
    regime_reasoning: str = ""
    sector_analysis: dict[str, str] = Field(default_factory=dict)
    portfolio_health: PortfolioHealth = Field(default_factory=PortfolioHealth)
    opportunities: list[Opportunity] = Field(default_factory=list)
    threats: list[Threat] = Field(default_factory=list)

    @field_validator("market_regime")
    @classmethod
    def validate_regime(cls, v: str) -> str:
        valid = {"BULL_TREND", "BEAR_TREND", "SIDEWAYS", "HIGH_VOLATILITY"}
        v_upper = v.upper().replace(" ", "_")
        if v_upper not in valid:
            logger.warning("invalid_market_regime, defaulting", value=v)
            return "SIDEWAYS"
        return v_upper

    @model_validator(mode="before")
    @classmethod
    def normalize_llm_formats(cls, data: dict) -> dict:
        """Normalize common LLM response variations before validation.

        Handles:
        - threats as plain strings: ["risk X"] -> [{"description": "risk X"}]
        - opportunities as plain strings: ["NVDA looks good"] -> [{"symbol": "?", "signal": "..."}]
        - opportunities as single-key dicts: {"NVDA": "reason"} -> [{"symbol": "NVDA", "signal": "reason"}]
        - sector_analysis with nested value: {"Tech": {"rating": "OW"}} -> {"Tech": "OW"}
        """
        if not isinstance(data, dict):
            return data

        # --- Normalize threats ---
        raw_threats = data.get("threats")
        if isinstance(raw_threats, list):
            normalized = []
            for item in raw_threats:
                if isinstance(item, str):
                    normalized.append({"description": item})
                elif isinstance(item, dict) and "description" in item:
                    normalized.append(item)
                elif isinstance(item, dict):
                    # Take first value as description
                    desc = next(iter(item.values()), str(item))
                    normalized.append({"description": str(desc)})
                else:
                    normalized.append({"description": str(item)})
            data["threats"] = normalized

        # --- Normalize opportunities ---
        raw_opps = data.get("opportunities")
        if isinstance(raw_opps, list):
            normalized = []
            for item in raw_opps:
                if isinstance(item, str):
                    # Try "SYMBOL: reason" or "SYMBOL - reason" format
                    for sep in [":", " - ", " — "]:
                        if sep in item:
                            parts = item.split(sep, 1)
                            sym = parts[0].strip().upper()
                            if len(sym) <= 5 and sym.isalpha():
                                normalized.append({"symbol": sym, "signal": parts[1].strip()})
                                break
                    else:
                        normalized.append({"symbol": "?", "signal": item})
                elif isinstance(item, dict) and "symbol" in item and "signal" in item:
                    normalized.append(item)
                elif isinstance(item, dict):
                    # {"NVDA": "reason"} format
                    for k, v in item.items():
                        normalized.append({"symbol": str(k), "signal": str(v)})
                else:
                    normalized.append({"symbol": "?", "signal": str(item)})
            data["opportunities"] = normalized

        # --- Normalize portfolio_health ---
        raw_health = data.get("portfolio_health")
        if isinstance(raw_health, str):
            # LLM returned a plain string like "GOOD" or "HIGH" — coerce to dict
            data["portfolio_health"] = {"risk_level": raw_health if raw_health.upper() in ("LOW", "MEDIUM", "HIGH") else "MEDIUM"}
        elif raw_health is not None and not isinstance(raw_health, dict):
            data["portfolio_health"] = {}

        # --- Normalize sector_analysis values (support both old string and new dict format) ---
        raw_sectors = data.get("sector_analysis")
        if isinstance(raw_sectors, str):
            # LLM returned a plain description string — discard it (can't parse reliably)
            data["sector_analysis"] = {}
        elif isinstance(raw_sectors, dict):
            # Remove meta keys like _sector_score_scale
            raw_sectors.pop("_sector_score_scale", None)
            normalized_sectors: dict[str, str] = {}
            for k, v in raw_sectors.items():
                if k.startswith("_"):
                    continue  # skip meta keys
                if isinstance(v, dict):
                    # New structured format: {"rating": "OVERWEIGHT", "score": 2, "reason": "..."}
                    rating = v.get("rating", "NEUTRAL")
                    score = v.get("score", 0)
                    reason = v.get("reason", "")
                    score_str = f"+{score}" if score > 0 else str(score)
                    normalized_sectors[k] = f"{rating} [{score_str}] - {reason}"
                else:
                    normalized_sectors[k] = str(v)
            data["sector_analysis"] = normalized_sectors

        return data


# --- Pass 2: Decision Models ---

class TradeAction(BaseModel):
    """A single trade action from the LLM."""
    type: str  # BUY or SELL
    symbol: str
    amount_usd: float
    urgency: str = "MEDIUM"
    thesis: str = ""
    stop_loss_pct: float | None = None    # e.g. -15.0 → exit if down 15% from entry
    take_profit_pct: float | None = None  # e.g. 25.0 → exit if up 25% from entry
    time_stop_days: int | None = None     # e.g. 30 → reassess after 30 days
    exit_condition: str = ""              # Legacy free-text (kept for backward compat)

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        v_upper = v.upper()
        if v_upper not in ("BUY", "SELL"):
            raise ValueError(f"Invalid trade type: {v}. Must be BUY or SELL.")
        return v_upper

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("urgency")
    @classmethod
    def validate_urgency(cls, v: str) -> str:
        v_upper = v.upper()
        if v_upper not in ("HIGH", "MEDIUM", "LOW"):
            return "MEDIUM"
        return v_upper

    @field_validator("amount_usd")
    @classmethod
    def validate_amount(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Trade amount must be positive, got {v}")
        return v


class DecisionResult(BaseModel):
    """Parsed result from Pass 2 (trading decisions)."""
    reasoning: str = ""
    actions: list[TradeAction] = Field(default_factory=list)
    portfolio_outlook: str = "NEUTRAL"
    confidence: float = 0.5
    next_cycle_focus: str = ""
    suggest_symbols: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_decision_formats(cls, data: dict) -> dict:
        """Normalize common LLM response variations for Pass 2."""
        if not isinstance(data, dict):
            return data

        # "action" -> "actions"
        if "action" in data and "actions" not in data:
            data["actions"] = data.pop("action")

        # "outlook" -> "portfolio_outlook"
        if "outlook" in data and "portfolio_outlook" not in data:
            data["portfolio_outlook"] = data.pop("outlook")

        # "trades" -> "actions"
        if "trades" in data and "actions" not in data:
            data["actions"] = data.pop("trades")

        # Ensure actions is a list
        if isinstance(data.get("actions"), dict):
            data["actions"] = [data["actions"]]

        # Filter out invalid actions (missing required fields) before validation
        raw_actions = data.get("actions")
        if isinstance(raw_actions, list):
            cleaned = []
            for a in raw_actions:
                if isinstance(a, dict):
                    # Normalize "action" -> "type"
                    if "action" in a and "type" not in a:
                        a["type"] = a.pop("action")
                    # Normalize "ticker" -> "symbol"
                    if "ticker" in a and "symbol" not in a:
                        a["symbol"] = a.pop("ticker")
                    # Normalize "amount" -> "amount_usd"
                    if "amount" in a and "amount_usd" not in a:
                        a["amount_usd"] = a.pop("amount")
                    # Only keep if it has the minimum required fields
                    if a.get("type") and a.get("symbol") and a.get("amount_usd"):
                        sym = str(a["symbol"]).upper().strip()
                        if not TICKER_RE.match(sym):
                            logger.warning("action_invalid_symbol", action=a)
                            continue
                        a["symbol"] = sym
                        cleaned.append(a)
                    else:
                        logger.warning("action_missing_fields", action=a)
                else:
                    # Already a TradeAction or other object - pass through
                    cleaned.append(a)
            data["actions"] = cleaned

        # Normalize suggest_symbols: flatten strings, strip whitespace, uppercase
        raw_suggest = data.get("suggest_symbols")
        if isinstance(raw_suggest, list):
            data["suggest_symbols"] = [
                str(s).upper().strip() for s in raw_suggest
                if isinstance(s, str) and s.strip()
            ]
        elif isinstance(raw_suggest, str):
            # LLM returned a comma-separated string
            data["suggest_symbols"] = [
                s.strip().upper() for s in raw_suggest.split(",") if s.strip()
            ]
        else:
            data["suggest_symbols"] = []

        return data

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @field_validator("portfolio_outlook")
    @classmethod
    def validate_outlook(cls, v: str) -> str:
        valid = {"BULLISH", "CAUTIOUSLY_BULLISH", "NEUTRAL", "CAUTIOUSLY_BEARISH", "BEARISH"}
        v_upper = v.upper().replace(" ", "_")
        if v_upper not in valid:
            return "NEUTRAL"
        return v_upper


def parse_analysis(raw_json: dict) -> AnalysisResult:
    """Parse Pass 1 analysis response into validated AnalysisResult."""
    try:
        result = AnalysisResult.model_validate(raw_json)
        logger.info(
            "analysis_parsed",
            regime=result.market_regime,
            opportunities=len(result.opportunities),
            threats=len(result.threats),
        )
        return result
    except Exception as e:
        logger.error("analysis_parse_failed", error=str(e), raw=str(raw_json)[:500])
        return AnalysisResult()


def parse_decision(raw_json: dict) -> DecisionResult:
    """Parse Pass 2 decision response into validated DecisionResult."""
    try:
        result = DecisionResult.model_validate(raw_json)
        logger.info(
            "decision_parsed",
            actions=len(result.actions),
            outlook=result.portfolio_outlook,
            confidence=result.confidence,
        )
        return result
    except Exception as e:
        logger.error("decision_parse_failed", error=str(e), raw=str(raw_json)[:500])
        return DecisionResult()
