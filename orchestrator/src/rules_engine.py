"""Rules-based trading engine — deterministic, backtestable alternative to LLM decisions.

Replaces LLM Pass 1-4 with quantitative scoring and rebalance logic.
Produces the same TradeAction/DecisionResult interface as the LLM path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import structlog

from .decision_parser import DecisionResult, TradeAction
from .portfolio_state import PortfolioState
from .technical_indicators import TechnicalSignals

logger = structlog.get_logger()


@dataclass
class ScoredSymbol:
    symbol: str
    score: float  # 0-100
    signal: str  # BUY, SELL, HOLD
    breakdown: dict = field(default_factory=dict)
    reason: str = ""


@dataclass
class RulesProposal:
    actions: list[TradeAction]
    scored_universe: list[ScoredSymbol] = field(default_factory=list)
    hold_reason: str = ""
    rebalance_triggered: bool = False


class RulesEngine:
    """Deterministic trading rules per strategy type."""

    def propose(
        self,
        strategy: str,
        portfolio: PortfolioState,
        market_data: dict,
        technical_signals: dict[str, TechnicalSignals],
        fundamentals: dict,
        quotes: dict,
        earnings_data: dict,
        risk_profile: dict,
        rules_params: dict,
        last_rebalance_date: str | None = None,
    ) -> RulesProposal:
        dispatch = {
            "core_satellite": self._propose_core_satellite,
            "value_investing": self._propose_value,
            "momentum": self._propose_momentum,
        }
        fn = dispatch.get(strategy)
        if fn is None:
            logger.warning("rules_engine_unknown_strategy", strategy=strategy)
            return RulesProposal(actions=[], hold_reason=f"Unknown strategy: {strategy}")

        return fn(
            portfolio=portfolio,
            market_data=market_data,
            technical_signals=technical_signals,
            fundamentals=fundamentals,
            quotes=quotes,
            earnings_data=earnings_data,
            risk_profile=risk_profile,
            params=rules_params,
            last_rebalance_date=last_rebalance_date,
        )

    # ── Timing gates ─────────────────────────────────────────────────────────

    def _days_since_rebalance(self, last_rebalance_date: str | None) -> float:
        if not last_rebalance_date:
            return 999  # never rebalanced → always eligible
        try:
            last = datetime.fromisoformat(last_rebalance_date)
            return (datetime.now() - last).total_seconds() / 86400
        except (ValueError, TypeError):
            return 999

    def _has_earnings_soon(self, symbol: str, earnings_data: dict, days: int = 3) -> bool:
        """Check if symbol has earnings within N days. Handles format: '2026-04-22 (in 5 days)'."""
        ear = earnings_data.get(symbol)
        if not ear:
            return False
        try:
            # Extract date from possible formats
            date_str = str(ear).split(" ")[0].strip()
            ear_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            delta = (ear_date - datetime.now().date()).days
            return 0 <= delta <= days
        except (ValueError, TypeError):
            return False

    # ── Scoring helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _linear_score(value: float | None, low: float, high: float, max_pts: float) -> float:
        """Linear interpolation: value at `high` → max_pts, value at `low` → 0."""
        if value is None:
            return 0.0
        clamped = max(low, min(high, value))
        if high == low:
            return max_pts
        return max_pts * (clamped - low) / (high - low)

    @staticmethod
    def _inverse_linear_score(value: float | None, low: float, high: float, max_pts: float) -> float:
        """Inverse: value at `low` → max_pts, value at `high` → 0."""
        if value is None:
            return 0.0
        clamped = max(low, min(high, value))
        if high == low:
            return max_pts
        return max_pts * (1.0 - (clamped - low) / (high - low))

    # ── Core-Satellite strategy ──────────────────────────────────────────────

    def _propose_core_satellite(self, portfolio, market_data, technical_signals,
                                 fundamentals, quotes, earnings_data,
                                 risk_profile, params, last_rebalance_date):
        target_core_pct = params.get("target_core_pct", 60)
        target_satellite_pct = params.get("target_satellite_pct", 25)
        target_cash_pct = params.get("target_cash_pct", 15)
        drift_threshold = params.get("rebalance_drift_pct", 5)
        min_score = params.get("min_satellite_score", 40)
        max_positions = params.get("max_satellite_positions", 5)
        max_position_pct = risk_profile.get("max_position_pct", 20)
        stop_loss_pct = risk_profile.get("stop_loss_pct", -15)

        # Classify current positions as core ETF vs satellite
        core_etfs = {"SPY", "QQQ", "VTI", "VOO", "SCHD", "VYM", "IVV", "IWM"}
        core_value = sum(p.market_value for p in portfolio.positions if p.symbol in core_etfs)
        satellite_value = sum(p.market_value for p in portfolio.positions if p.symbol not in core_etfs)
        total = portfolio.total_value or 1

        current_core_pct = core_value / total * 100
        current_satellite_pct = satellite_value / total * 100
        current_cash_pct = portfolio.cash_pct

        core_drift = abs(current_core_pct - target_core_pct)
        satellite_drift = abs(current_satellite_pct - target_satellite_pct)
        cash_drift = abs(current_cash_pct - target_cash_pct)
        max_drift = max(core_drift, satellite_drift, cash_drift)

        if max_drift < drift_threshold:
            return RulesProposal(
                actions=[], hold_reason=f"Drift {max_drift:.1f}% < {drift_threshold}% threshold",
            )

        # Score satellite universe
        scored = []
        for sym, tech in technical_signals.items():
            if sym in core_etfs:
                continue
            q = quotes.get(sym)
            fund = fundamentals.get(sym)

            pe = getattr(q, "pe_ratio", None) if q else None
            div_y = getattr(q, "dividend_yield", None) if q else None
            rsi = tech.rsi_14
            sma200 = tech.sma_200
            price = tech.price

            # Score components
            pe_score = self._inverse_linear_score(pe, 10, 40, 25)
            div_score = self._linear_score(div_y, 0, 0.03, 15)
            momentum_score = 0.0
            if price and sma200 and sma200 > 0:
                rel = (price - sma200) / sma200
                momentum_score = self._linear_score(rel, -0.1, 0.25, 25)
            analyst_score = 0.0
            if fund:
                rec = getattr(fund, "rec_label", "") or ""
                analyst_map = {"STRONG_BUY": 20, "BUY": 15, "HOLD": 8, "SELL": 2, "STRONG_SELL": 0}
                analyst_score = analyst_map.get(rec.upper().replace(" ", "_"), 8)
            rsi_score = 0.0
            if rsi is not None:
                if 40 <= rsi <= 60:
                    rsi_score = 15
                elif 30 <= rsi < 40 or 60 < rsi <= 70:
                    rsi_score = 8
                # overbought/oversold → 0

            total_score = pe_score + div_score + momentum_score + analyst_score + rsi_score
            scored.append(ScoredSymbol(
                symbol=sym, score=total_score, signal="HOLD",
                breakdown={"pe": pe_score, "div": div_score, "momentum": momentum_score,
                           "analyst": analyst_score, "rsi": rsi_score},
            ))

        scored.sort(key=lambda s: s.score, reverse=True)

        actions = []

        # SELL: satellites not in top rankings or losing
        held_satellites = [p for p in portfolio.positions if p.symbol not in core_etfs]
        top_syms = {s.symbol for s in scored[:max(10, max_positions * 2)]}
        for pos in held_satellites:
            if pos.symbol not in top_syms or (
                pos.unrealized_pl_pct is not None and pos.unrealized_pl_pct <= stop_loss_pct
            ):
                actions.append(TradeAction(
                    type="SELL", symbol=pos.symbol, amount_usd=pos.market_value,
                    urgency="MEDIUM",
                    thesis=f"Dropped from top ranking (score below threshold) or stop-loss",
                    stop_loss_pct=stop_loss_pct,
                ))

        # BUY: deploy cash toward target allocation
        # Core ETFs first if under-allocated
        if current_core_pct < target_core_pct - drift_threshold:
            core_deficit_usd = (target_core_pct - current_core_pct) / 100 * total
            # Buy the first core ETF in watchlist
            for sym in ["VOO", "VTI", "SPY"]:
                if sym in {q for q in quotes}:
                    buy_amt = min(core_deficit_usd, total * max_position_pct / 100)
                    if buy_amt > 50:
                        actions.append(TradeAction(
                            type="BUY", symbol=sym, amount_usd=round(buy_amt, 2),
                            urgency="MEDIUM", thesis=f"Core ETF under-allocated ({current_core_pct:.0f}% vs {target_core_pct}% target)",
                            stop_loss_pct=stop_loss_pct,
                        ))
                    break

        # Satellite buys
        buying_power = max(0, portfolio.cash - total * risk_profile.get("min_cash_pct", 10) / 100)
        for s in scored[:max_positions]:
            if s.score < min_score:
                break
            if self._has_earnings_soon(s.symbol, earnings_data):
                continue
            existing = portfolio.get_position(s.symbol)
            if existing:
                continue  # already held
            alloc = min(buying_power, total * max_position_pct / 100)
            if alloc < 50:
                break
            actions.append(TradeAction(
                type="BUY", symbol=s.symbol, amount_usd=round(alloc, 2),
                urgency="LOW", thesis=f"Satellite score {s.score:.0f}/100: {s.breakdown}",
                stop_loss_pct=stop_loss_pct,
            ))
            buying_power -= alloc
            s.signal = "BUY"

        return RulesProposal(
            actions=actions, scored_universe=scored,
            rebalance_triggered=True,
        )

    # ── Value Investing strategy ─────────────────────────────────────────────

    def _propose_value(self, portfolio, market_data, technical_signals,
                        fundamentals, quotes, earnings_data,
                        risk_profile, params, last_rebalance_date):
        interval = params.get("rebalance_interval_days", 25)
        days = self._days_since_rebalance(last_rebalance_date)
        if days < interval:
            return RulesProposal(
                actions=[], hold_reason=f"Value rebalance: {days:.0f}d since last, need {interval}d",
            )

        pe_max = params.get("pe_max", 20)
        pb_max = params.get("pb_max", 3.0)
        div_min = params.get("div_yield_min", 0.015)
        min_score = params.get("min_value_score", 50)
        max_positions = params.get("max_positions", 5)
        max_position_pct = risk_profile.get("max_position_pct", 15)
        stop_loss_pct = risk_profile.get("stop_loss_pct", -20)

        scored = []
        for sym, q in quotes.items():
            fund = fundamentals.get(sym)
            pe = getattr(q, "pe_ratio", None)
            pb = getattr(q, "pb_ratio", None)
            div_y = getattr(q, "dividend_yield", None)
            eps_g = getattr(fund, "eps_growth_yoy", None) if fund else None

            # Hard screen
            if pe is None or pe <= 0 or pe > pe_max:
                continue
            if pb is not None and pb > pb_max:
                continue
            if div_y is None or div_y < div_min:
                continue
            if eps_g is not None and eps_g < 0:
                continue

            # Score
            pe_score = self._inverse_linear_score(pe, 10, 20, 30)
            pb_score = self._inverse_linear_score(pb, 1.0, 3.0, 15) if pb else 0
            div_score = self._linear_score(div_y, 0, 0.04, 20)
            eps_score = self._linear_score(eps_g, 0, 0.20, 20) if eps_g else 0

            upside_score = 0.0
            if fund and getattr(fund, "target_price", None) and getattr(fund, "current_price", None):
                upside = (fund.target_price - fund.current_price) / fund.current_price
                upside_score = self._linear_score(upside, 0, 0.30, 15)

            total_score = pe_score + pb_score + div_score + eps_score + upside_score
            scored.append(ScoredSymbol(
                symbol=sym, score=total_score, signal="HOLD",
                breakdown={"pe": pe_score, "pb": pb_score, "div": div_score,
                           "eps": eps_score, "upside": upside_score},
            ))

        scored.sort(key=lambda s: s.score, reverse=True)

        actions = []

        # SELL: positions where value thesis broke
        for pos in portfolio.positions:
            q = quotes.get(pos.symbol)
            pe = getattr(q, "pe_ratio", None) if q else None
            div_y = getattr(q, "dividend_yield", None) if q else None
            if pe is not None and pe > 30:
                actions.append(TradeAction(
                    type="SELL", symbol=pos.symbol, amount_usd=pos.market_value,
                    urgency="MEDIUM", thesis=f"Value thesis broken: P/E expanded to {pe:.1f}",
                    stop_loss_pct=stop_loss_pct,
                ))
            elif div_y is not None and div_y < 0.01:
                actions.append(TradeAction(
                    type="SELL", symbol=pos.symbol, amount_usd=pos.market_value,
                    urgency="MEDIUM", thesis=f"Dividend cut below 1%: {div_y*100:.2f}%",
                    stop_loss_pct=stop_loss_pct,
                ))

        # BUY: top value picks
        total = portfolio.total_value or 1
        buying_power = max(0, portfolio.cash - total * risk_profile.get("min_cash_pct", 10) / 100)

        for s in scored[:max_positions]:
            if s.score < min_score:
                break
            if self._has_earnings_soon(s.symbol, earnings_data):
                continue
            existing = portfolio.get_position(s.symbol)
            if existing:
                continue
            alloc = min(buying_power, total * max_position_pct / 100)
            if alloc < 50:
                break
            actions.append(TradeAction(
                type="BUY", symbol=s.symbol, amount_usd=round(alloc, 2),
                urgency="LOW", thesis=f"Value score {s.score:.0f}/100: {s.breakdown}",
                stop_loss_pct=stop_loss_pct,
            ))
            buying_power -= alloc
            s.signal = "BUY"

        return RulesProposal(
            actions=actions, scored_universe=scored,
            rebalance_triggered=True,
        )

    # ── Momentum strategy ────────────────────────────────────────────────────

    def _propose_momentum(self, portfolio, market_data, technical_signals,
                           fundamentals, quotes, earnings_data,
                           risk_profile, params, last_rebalance_date):
        interval = params.get("rebalance_interval_days", 7)
        days = self._days_since_rebalance(last_rebalance_date)
        if days < interval:
            return RulesProposal(
                actions=[],
                hold_reason=f"Momentum rebalance: {days:.0f}d since last, need {interval}d",
            )

        min_score = params.get("min_momentum_score", 45)
        sell_threshold = params.get("sell_score_threshold", 30)
        max_positions = params.get("max_positions", 3)
        max_position_pct = risk_profile.get("max_position_pct", 25)
        stop_loss_pct = risk_profile.get("stop_loss_pct", -12)

        scored = []
        for sym, tech in technical_signals.items():
            price = tech.price
            sma20 = tech.sma_20
            rsi = tech.rsi_14
            vol_ratio = tech.volume_ratio
            macd_hist = tech.macd_histogram

            # Rate of change (20d)
            roc_score = 0.0
            if price and sma20 and sma20 > 0:
                roc = (price - sma20) / sma20
                roc_score = self._linear_score(roc, -0.05, 0.10, 35)

            # RSI zone
            rsi_score = 0.0
            if rsi is not None:
                if 50 <= rsi <= 70:
                    rsi_score = 30
                elif 40 <= rsi < 50:
                    rsi_score = 15
                elif 70 < rsi <= 80:
                    rsi_score = 20
                # else 0

            # Volume confirmation
            vol_score = 0.0
            if vol_ratio is not None:
                if vol_ratio >= 1.5:
                    vol_score = 20
                elif vol_ratio >= 1.2:
                    vol_score = 15
                elif vol_ratio >= 1.0:
                    vol_score = 10

            # MACD histogram
            macd_score = 0.0
            if macd_hist is not None:
                if macd_hist > 0:
                    macd_score = 15
                elif macd_hist > -0.5:
                    macd_score = 5

            total_score = roc_score + rsi_score + vol_score + macd_score
            scored.append(ScoredSymbol(
                symbol=sym, score=total_score, signal="HOLD",
                breakdown={"roc": roc_score, "rsi": rsi_score, "vol": vol_score, "macd": macd_score},
            ))

        scored.sort(key=lambda s: s.score, reverse=True)

        actions = []

        # SELL: held positions with weak momentum
        for pos in portfolio.positions:
            pos_scored = next((s for s in scored if s.symbol == pos.symbol), None)
            score = pos_scored.score if pos_scored else 0
            rsi = technical_signals.get(pos.symbol, TechnicalSignals(symbol=pos.symbol)).rsi_14

            if score < sell_threshold or (rsi is not None and rsi > 80):
                reason = f"Momentum score {score:.0f} < {sell_threshold}" if score < sell_threshold else f"RSI {rsi:.0f} > 80"
                actions.append(TradeAction(
                    type="SELL", symbol=pos.symbol, amount_usd=pos.market_value,
                    urgency="MEDIUM", thesis=reason,
                    stop_loss_pct=stop_loss_pct,
                ))

        # BUY: top momentum symbols
        total = portfolio.total_value or 1
        buying_power = max(0, portfolio.cash - total * risk_profile.get("min_cash_pct", 10) / 100)
        held_syms = {p.symbol for p in portfolio.positions}

        for s in scored[:max_positions]:
            if s.score < min_score:
                break
            if s.symbol in held_syms:
                continue
            if self._has_earnings_soon(s.symbol, earnings_data):
                continue
            alloc = min(buying_power, total * max_position_pct / 100)
            if alloc < 50:
                break
            actions.append(TradeAction(
                type="BUY", symbol=s.symbol, amount_usd=round(alloc, 2),
                urgency="MEDIUM", thesis=f"Momentum score {s.score:.0f}/100: {s.breakdown}",
                stop_loss_pct=stop_loss_pct,
            ))
            buying_power -= alloc
            s.signal = "BUY"

        return RulesProposal(
            actions=actions, scored_universe=scored,
            rebalance_triggered=True,
        )
