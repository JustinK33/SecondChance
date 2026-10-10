"""Token and cost accounting for model calls.

``app.core.metrics`` does not exist yet, so this module owns two things:

1. :class:`UsageAccumulator` - an in-process running total, exposed on the
   client as ``client.usage``. It is what makes "what did this benchmark cost?"
   answerable today, and it is the thing to delete once a real metrics sink
   lands.
2. :func:`record_usage` - a single call site that forwards every
   :class:`~app.schemas.ai.ModelUsage` to ``app.core.metrics.record_model_usage``
   if that function exists, and to the accumulator either way. The lookup is
   done by name through :mod:`importlib` rather than a module-level ``import``
   so that this layer does not have to be edited when the metrics module
   appears, and so a missing module is not an ``ImportError`` at startup.

Nothing here touches user content. ``ModelUsage`` holds ids, counts and
durations only, which is exactly why it is the one AI object that is safe to log
in full.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from app.schemas.ai import ModelUsage

__all__ = ["UsageAccumulator", "UsageTotals", "record_usage"]

_METRICS_MODULE = "app.core.metrics"
_METRICS_FUNCTION = "record_model_usage"


def _metrics_recorder() -> Callable[[ModelUsage], None] | None:
    """Return ``app.core.metrics.record_model_usage`` if it exists yet."""
    try:
        module: Any = importlib.import_module(_METRICS_MODULE)
    except ModuleNotFoundError:
        return None
    recorder = getattr(module, _METRICS_FUNCTION, None)
    if callable(recorder):
        return cast(Callable[[ModelUsage], None], recorder)
    return None


@dataclass(frozen=True, slots=True)
class UsageTotals:
    """An immutable snapshot of what the model layer has spent so far."""

    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    total_tokens: int = 0
    attempts: int = 0
    latency_ms: int = 0
    estimated_cost_usd: float = 0.0

    @property
    def retries(self) -> int:
        """Attempts beyond the first one, summed across calls."""
        return max(self.attempts - self.calls, 0)


@dataclass(slots=True)
class UsageAccumulator:
    """Thread-safe running total of model usage.

    Locked because a workflow fans out explanation calls concurrently and
    ``+=`` on a float is not atomic under the GIL's bytecode boundaries.
    """

    _totals: UsageTotals = field(default_factory=UsageTotals)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, usage: ModelUsage) -> None:
        with self._lock:
            current = self._totals
            self._totals = UsageTotals(
                calls=current.calls + 1,
                prompt_tokens=current.prompt_tokens + usage.prompt_tokens,
                output_tokens=current.output_tokens + usage.output_tokens,
                thought_tokens=current.thought_tokens + usage.thought_tokens,
                total_tokens=current.total_tokens + usage.total_tokens,
                attempts=current.attempts + usage.attempts,
                latency_ms=current.latency_ms + usage.latency_ms,
                estimated_cost_usd=current.estimated_cost_usd + usage.estimated_cost_usd,
            )

    @property
    def totals(self) -> UsageTotals:
        with self._lock:
            return self._totals

    def reset(self) -> None:
        with self._lock:
            self._totals = UsageTotals()


def record_usage(usage: ModelUsage, accumulator: UsageAccumulator | None = None) -> None:
    """Record one model call's usage in every sink that exists."""
    if accumulator is not None:
        accumulator.add(usage)
    recorder = _metrics_recorder()
    if recorder is not None:
        recorder(usage)
