"""Self-consistency sampling for Pass 2 trade decisions.

A single LLM generation is noisy: the same analysis can yield a bear-call one
sample and a bull-put the next. Sampling Pass 2 several times and keeping only
the actions that recur in a majority of samples filters out one-off outliers
while preserving the decisions the model is actually confident about.

Works on the RAW decision dicts (before parsing), which all cycle types share:
``{"actions": [{"type": ..., "symbol": ..., ...}], ...}``.
"""

from __future__ import annotations

from statistics import median

import structlog

logger = structlog.get_logger()

# Numeric per-action fields merged as the median across recurring samples
_NUMERIC_FIELDS = ("amount_usd", "contracts", "strike")


def _action_key(action: dict) -> tuple:
    """Identity of an action across samples: what to do with which instrument."""
    return (
        str(action.get("type", "")).upper(),
        str(action.get("symbol", "")).upper(),
        str(action.get("spread_type", "") or "").lower(),
        action.get("position_id"),
    )


def merge_decision_samples(samples: list[dict]) -> dict:
    """Merge N raw decision dicts by majority vote over actions.

    Actions present in >= majority of samples survive; their numeric fields
    become the median across occurrences. Non-action fields come from the
    first sample (confidence: mean). With a single sample this is a no-op.
    """
    samples = [s for s in samples if isinstance(s, dict)]
    if not samples:
        return {}
    if len(samples) == 1:
        return samples[0]

    min_votes = len(samples) // 2 + 1

    # Group occurrences by key, preserving first-seen order
    occurrences: dict[tuple, list[dict]] = {}
    order: list[tuple] = []
    for sample in samples:
        actions = sample.get("actions")
        if not isinstance(actions, list):
            continue
        seen_in_sample: set[tuple] = set()
        for action in actions:
            if not isinstance(action, dict):
                continue
            key = _action_key(action)
            if key in seen_in_sample:
                continue   # count each action once per sample
            seen_in_sample.add(key)
            if key not in occurrences:
                occurrences[key] = []
                order.append(key)
            occurrences[key].append(action)

    merged_actions = []
    dropped = []
    for key in order:
        occ = occurrences[key]
        if len(occ) < min_votes:
            dropped.append({"key": key, "votes": len(occ)})
            continue
        merged = dict(occ[0])   # representative: first occurrence
        for field in _NUMERIC_FIELDS:
            values = [o[field] for o in occ if isinstance(o.get(field), (int, float))]
            if values:
                m = median(values)
                merged[field] = int(round(m)) if field == "contracts" else round(m, 2)
        merged_actions.append(merged)

    result = dict(samples[0])
    result["actions"] = merged_actions

    confidences = [
        s["confidence"] for s in samples
        if isinstance(s.get("confidence"), (int, float))
    ]
    if confidences:
        result["confidence"] = round(sum(confidences) / len(confidences), 2)

    result["_consensus"] = {
        "samples": len(samples),
        "min_votes": min_votes,
        "kept": len(merged_actions),
        "dropped": [
            {"action": f"{k[0]} {k[1]}" + (f" {k[2]}" if k[2] else ""), "votes": v["votes"]}
            for k, v in ((d["key"], d) for d in dropped)
        ],
    }

    logger.info(
        "pass2_consensus_merged",
        samples=len(samples), min_votes=min_votes,
        kept=len(merged_actions), dropped=len(dropped),
    )
    return result


def chat_json_consensus(
    llm,
    messages: list[dict],
    model: str,
    fallback_model: str | None,
    samples: int = 1,
    temperature: float = 0.7,
    max_tokens: int = 16384,
) -> dict:
    """chat_json sampled ``samples`` times, merged by action-majority vote.

    Individual sample failures are tolerated as long as at least one succeeds;
    with one success the decision passes through unmerged. samples <= 1 is a
    plain chat_json call.
    """
    if samples <= 1:
        return llm.chat_json(
            messages=messages, model=model, fallback_model=fallback_model,
            temperature=temperature, max_tokens=max_tokens,
        )

    raw_samples: list[dict] = []
    last_error: Exception | None = None
    for i in range(samples):
        try:
            raw_samples.append(llm.chat_json(
                messages=messages, model=model, fallback_model=fallback_model,
                temperature=temperature, max_tokens=max_tokens,
            ))
        except Exception as e:
            last_error = e
            logger.warning("pass2_sample_failed", sample=i + 1, error=str(e))

    if not raw_samples:
        raise RuntimeError(f"All {samples} Pass 2 samples failed: {last_error}")

    return merge_decision_samples(raw_samples)
