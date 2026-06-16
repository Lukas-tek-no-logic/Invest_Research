"""Global pacing + retry for yfinance HTTP calls.

Yahoo rate-limits bursts: a single decision cycle fires ~15-30 quote /
fundamental / screener calls back-to-back and Yahoo answers HTTP 429 or drops
the connection partway through the burst (visible in scheduler logs every
trading day). This module enforces a minimum interval between consecutive
yfinance calls and retries transient failures with exponential backoff +
jitter, so a cycle no longer flies blind when Yahoo throttles mid-burst.

Usage:

    from .yf_throttle import paced_call
    info = paced_call(lambda: yf.Ticker(sym).info, label=f"info:{sym}")

The thunk should be self-contained so a retry restarts from a clean state.
Non-retryable errors propagate immediately; only rate-limit / connection /
timeout failures are retried.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Callable, TypeVar

import structlog

logger = structlog.get_logger()

MIN_INTERVAL = 0.4   # seconds between consecutive yfinance HTTP calls
MAX_RETRIES = 2      # retries after the first attempt
BASE_BACKOFF = 1.0   # seconds; attempt N waits ~ BASE_BACKOFF * 2**N + jitter

T = TypeVar("T")

_lock = threading.Lock()
_last_call = 0.0

# Substrings marking a transient, retryable Yahoo failure.
_RETRYABLE = (
    "429", "too many requests", "rate limit", "rate-limit",
    "could not connect", "failed to connect", "connection",
    "timed out", "timeout", "temporarily unavailable", "max retries",
)


def _is_retryable(err: Exception) -> bool:
    msg = str(err).lower()
    return any(n in msg for n in _RETRYABLE)


def _pace() -> None:
    """Block until MIN_INTERVAL has elapsed since the previous call.

    Holds the lock across the short spacing sleep so concurrent account
    cycles are serialised into one global, evenly-spaced request stream —
    exactly what avoids tripping Yahoo's burst limiter.
    """
    global _last_call
    with _lock:
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def paced_call(thunk: Callable[[], T], *, label: str = "yf") -> T:
    """Run ``thunk`` with global pacing and retry/backoff on transient errors."""
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        _pace()
        try:
            return thunk()
        except Exception as e:  # broad: yfinance raises bare Exceptions
            last_err = e
            if attempt >= MAX_RETRIES or not _is_retryable(e):
                raise
            backoff = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 0.4)
            logger.debug("yf_retry", label=label, attempt=attempt + 1,
                         backoff=round(backoff, 2), error=str(e)[:120])
            time.sleep(backoff)
    assert last_err is not None  # loop body always sets it before raising
    raise last_err
