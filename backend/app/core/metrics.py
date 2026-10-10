"""In-process metrics: counters, gauges and histograms with no personal data.

Why a hand-rolled registry rather than ``prometheus_client``: the numbers this
product needs are the ones in ``docs/METRICS.md`` - workflow duration, per-node
duration, model latency, token counts and cost, database timing, retries and
timeouts - and the benchmarks need them as a **structured snapshot**, not only
as scrape text. A registry that can render both from one set of observations is
about two hundred lines, adds no dependency, and makes the one rule that really
matters enforceable in code.

That rule is the label policy. A metric label becomes a time series, and a
label carrying a user id, a county, an income or a narrative would make the
metrics endpoint the single biggest privacy hole in the system - it is
unauthenticated inside the deployment network and scraped forever. So:

* labels are bounded, low-cardinality and come from closed sets: route
  templates (``/api/v1/workflows/{run_id}``, never the id), HTTP methods,
  status codes, node names, model task names, error codes;
* :func:`sanitise_label` rejects anything long, multi-line or digit-shaped
  before it can become a series, and the rejection is a dropped observation
  rather than a leaked label.

Everything is guarded by a ``threading.Lock``. The application is
single-threaded asyncio, but uvicorn's lifespan, a ``run_in_executor`` call and
a test thread can all touch the registry, and a lock is cheaper than reasoning
about which of them will.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Final

__all__ = [
    "DURATION_BUCKETS_S",
    "PROMETHEUS_CONTENT_TYPE",
    "MetricsRegistry",
    "get_metrics",
    "reset_metrics",
    "sanitise_label",
]

PROMETHEUS_CONTENT_TYPE: Final[str] = "text/plain; version=0.0.4; charset=utf-8"

#: Seconds. Dense below one second because that is where an HTTP handler lives,
#: and stretched to two minutes at the top because a workflow run legitimately
#: takes tens of seconds and a histogram that saturates tells you nothing.
DURATION_BUCKETS_S: Final[tuple[float, ...]] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
)

#: The label vocabulary: identifier-ish, plus the punctuation a route template
#: and a model id need. Deliberately excludes whitespace, quotes and newlines.
_LABEL_ALLOWED: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_:./{}\[\]@+-]{1,96}$")
#: A label that is mostly digits is almost always an id that escaped a route
#: template, so it is dropped rather than recorded.
_LABEL_DIGITS: Final[re.Pattern[str]] = re.compile(r"^[0-9-]{6,}$")
_METRIC_NAME: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Replaces a label value that failed the policy. Visible in the output on
#: purpose: a sudden spike in ``other`` is a bug report.
LABEL_FALLBACK: Final[str] = "other"

LabelKey = tuple[tuple[str, str], ...]


def sanitise_label(value: object) -> str:
    """Coerce ``value`` into a safe, low-cardinality label.

    Returns :data:`LABEL_FALLBACK` for anything that is empty, long,
    multi-line, quoted or shaped like an identifier. Never raises: a metric
    must not be able to fail a request.
    """
    text = str(value).strip()
    if not text:
        return LABEL_FALLBACK
    if _LABEL_DIGITS.match(text):
        return LABEL_FALLBACK
    if not _LABEL_ALLOWED.match(text):
        return LABEL_FALLBACK
    return text


def _key(labels: Mapping[str, object] | None) -> LabelKey:
    if not labels:
        return ()
    return tuple(sorted((str(name), sanitise_label(value)) for name, value in labels.items()))


def _render_labels(key: LabelKey, extra: tuple[tuple[str, str], ...] = ()) -> str:
    pairs = [*key, *extra]
    if not pairs:
        return ""
    body = ",".join(f'{name}="{value}"' for name, value in pairs)
    return "{" + body + "}"


def _format(value: float) -> str:
    """Prometheus-safe number rendering."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(round(value, 6))


@dataclass(slots=True)
class _Histogram:
    buckets: tuple[float, ...]
    counts: list[int]
    total: float = 0.0
    count: int = 0

    def observe(self, value: float) -> None:
        self.total += value
        self.count += 1
        for index, edge in enumerate(self.buckets):
            if value <= edge:
                self.counts[index] += 1
        # The +Inf bucket is implicit and equals ``count``.

    def snapshot(self) -> dict[str, Any]:
        cumulative = {_format(edge): self.counts[index] for index, edge in enumerate(self.buckets)}
        return {
            "count": self.count,
            "sum": round(self.total, 6),
            "avg": round(self.total / self.count, 6) if self.count else 0.0,
            "p50": self.quantile(0.5),
            "p95": self.quantile(0.95),
            "p99": self.quantile(0.99),
            "buckets": cumulative,
        }

    def quantile(self, fraction: float) -> float:
        """Upper bucket edge containing the requested quantile.

        A bucketed estimate, not an exact percentile: the registry stores
        counts rather than samples, because keeping every sample for the
        lifetime of a process is how a metrics system becomes the memory leak.
        """
        if self.count == 0:
            return 0.0
        target = fraction * self.count
        for index, edge in enumerate(self.buckets):
            if self.counts[index] >= target:
                return edge
        return math.inf


@dataclass(slots=True)
class _Family:
    name: str
    kind: str
    help_text: str
    unit: str | None = None
    buckets: tuple[float, ...] = DURATION_BUCKETS_S
    values: dict[LabelKey, float] = field(default_factory=dict)
    histograms: dict[LabelKey, _Histogram] = field(default_factory=dict)


class MetricsRegistry:
    """A registry of counters, gauges and duration histograms.

    Families are declared on first use, so a caller records an observation
    without a registration step - but the declared ``kind`` is sticky, and
    recording a counter into a histogram family raises rather than silently
    producing nonsense.
    """

    __slots__ = ("_families", "_lock", "_started_at")

    def __init__(self) -> None:
        self._families: dict[str, _Family] = {}
        self._lock = threading.Lock()
        self._started_at = time.time()

    # ------------------------------------------------------------- internals
    def _family(
        self,
        name: str,
        kind: str,
        help_text: str,
        *,
        buckets: tuple[float, ...] = DURATION_BUCKETS_S,
        unit: str | None = None,
    ) -> _Family:
        if not _METRIC_NAME.match(name):
            msg = f"invalid metric name {name!r}"
            raise ValueError(msg)
        existing = self._families.get(name)
        if existing is None:
            existing = _Family(
                name=name, kind=kind, help_text=help_text, unit=unit, buckets=buckets
            )
            self._families[name] = existing
            return existing
        if existing.kind != kind:
            msg = f"metric {name!r} is already registered as a {existing.kind}"
            raise ValueError(msg)
        return existing

    # ------------------------------------------------------------ primitives
    def increment(
        self,
        name: str,
        *,
        help_text: str = "",
        amount: float = 1.0,
        labels: Mapping[str, object] | None = None,
    ) -> None:
        """Add ``amount`` to a monotonic counter."""
        if amount < 0:
            msg = "a counter cannot decrease"
            raise ValueError(msg)
        with self._lock:
            family = self._family(name, "counter", help_text)
            key = _key(labels)
            family.values[key] = family.values.get(key, 0.0) + amount

    def set_gauge(
        self,
        name: str,
        value: float,
        *,
        help_text: str = "",
        labels: Mapping[str, object] | None = None,
    ) -> None:
        """Set a gauge to an instantaneous value."""
        with self._lock:
            family = self._family(name, "gauge", help_text)
            family.values[_key(labels)] = float(value)

    def adjust_gauge(
        self,
        name: str,
        delta: float,
        *,
        help_text: str = "",
        labels: Mapping[str, object] | None = None,
    ) -> None:
        """Move a gauge by ``delta``. Used for in-flight counts."""
        with self._lock:
            family = self._family(name, "gauge", help_text)
            key = _key(labels)
            family.values[key] = family.values.get(key, 0.0) + delta

    def observe(
        self,
        name: str,
        value: float,
        *,
        help_text: str = "",
        labels: Mapping[str, object] | None = None,
        buckets: tuple[float, ...] = DURATION_BUCKETS_S,
    ) -> None:
        """Record one sample in a histogram."""
        with self._lock:
            family = self._family(name, "histogram", help_text, buckets=buckets)
            key = _key(labels)
            histogram = family.histograms.get(key)
            if histogram is None:
                histogram = _Histogram(buckets=family.buckets, counts=[0] * len(family.buckets))
                family.histograms[key] = histogram
            histogram.observe(value)

    @contextmanager
    def timed(
        self,
        name: str,
        *,
        help_text: str = "",
        labels: Mapping[str, object] | None = None,
    ) -> Iterator[None]:
        """Time the enclosed block into a histogram, even if it raises."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - started, help_text=help_text, labels=labels)

    # -------------------------------------------------- product-level helpers
    def observe_request(
        self, *, method: str, route: str, status_code: int, duration_s: float
    ) -> None:
        """One HTTP request. ``route`` must be the template, never the path."""
        labels = {"method": method.upper(), "route": route, "status": str(status_code)}
        self.observe(
            "http_request_duration_seconds",
            duration_s,
            help_text="HTTP request latency in seconds, by route template and status.",
            labels=labels,
        )
        self.increment(
            "http_requests_total",
            help_text="HTTP requests handled, by route template and status.",
            labels=labels,
        )

    def observe_workflow(self, *, status: str, duration_s: float) -> None:
        """One workflow run, from scheduling to a terminal status."""
        self.observe(
            "workflow_duration_seconds",
            duration_s,
            help_text="End-to-end workflow run duration in seconds, by terminal status.",
            labels={"status": status},
        )
        self.increment(
            "workflow_runs_total",
            help_text="Workflow runs finished, by terminal status.",
            labels={"status": status},
        )

    def observe_workflow_node(self, *, node: str, duration_s: float) -> None:
        """One graph node. Node names are a closed set from the topology."""
        self.observe(
            "workflow_node_duration_seconds",
            duration_s,
            help_text="Per-node workflow duration in seconds.",
            labels={"node": node},
        )

    def observe_model_call(
        self,
        *,
        task: str,
        source: str,
        model: str,
        latency_s: float,
        prompt_tokens: int = 0,
        output_tokens: int = 0,
        thought_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """One model call: latency, token counts and estimated cost.

        ``source`` is ``live`` / ``cache`` / ``fallback``, which is what keeps a
        benchmark honest about whether a model actually ran.
        """
        labels = {"task": task, "source": source, "model": model}
        self.observe(
            "gemini_call_duration_seconds",
            latency_s,
            help_text="Model call latency in seconds, by task and source.",
            labels=labels,
        )
        self.increment(
            "gemini_calls_total",
            help_text="Model calls made, by task and source.",
            labels=labels,
        )
        for kind, count in (
            ("prompt", prompt_tokens),
            ("output", output_tokens),
            ("thought", thought_tokens),
        ):
            if count:
                self.increment(
                    "gemini_tokens_total",
                    help_text="Model tokens consumed, by task and token kind.",
                    amount=float(count),
                    labels={"task": task, "source": source, "kind": kind},
                )
        if cost_usd:
            self.increment(
                "gemini_estimated_cost_usd_total",
                help_text="Estimated model spend in USD at configured prices.",
                amount=cost_usd,
                labels={"task": task, "source": source},
            )

    def observe_db_query(self, *, operation: str, duration_s: float) -> None:
        """One database unit of work, named by operation rather than by SQL."""
        self.observe(
            "db_query_duration_seconds",
            duration_s,
            help_text="Database operation duration in seconds.",
            labels={"operation": operation},
        )

    def increment_retry(self, *, operation: str, reason: str = "transient") -> None:
        self.increment(
            "retries_total",
            help_text="Retried operations, by operation and reason.",
            labels={"operation": operation, "reason": reason},
        )

    def increment_timeout(self, *, operation: str) -> None:
        self.increment(
            "timeouts_total",
            help_text="Operations abandoned after exceeding their deadline.",
            labels={"operation": operation},
        )

    def increment_error(self, *, code: str, route: str = LABEL_FALLBACK) -> None:
        self.increment(
            "errors_total",
            help_text="Errors returned to clients, by error code and route template.",
            labels={"code": code, "route": route},
        )

    def set_pool_gauges(
        self,
        *,
        size: int,
        checked_out: int,
        overflow: int,
        available: int,
    ) -> None:
        """Connection-pool occupancy. Read from the SQLAlchemy pool on scrape."""
        help_text = "Database connection pool slots, by state."
        for state, value in (
            ("size", size),
            ("checked_out", checked_out),
            ("overflow", overflow),
            ("available", available),
        ):
            self.set_gauge(
                "db_pool_connections", float(value), help_text=help_text, labels={"state": state}
            )

    def set_workflows_in_flight(self, value: int) -> None:
        self.set_gauge(
            "workflows_in_flight",
            float(value),
            help_text="Workflow runs currently executing in this process.",
        )

    # ------------------------------------------------------------- rendering
    def render_prometheus(self) -> str:
        """The registry in Prometheus text exposition format."""
        lines: list[str] = []
        with self._lock:
            for name in sorted(self._families):
                family = self._families[name]
                if family.help_text:
                    lines.append(f"# HELP {name} {family.help_text}")
                lines.append(f"# TYPE {name} {family.kind}")
                if family.kind == "histogram":
                    for key in sorted(family.histograms):
                        histogram = family.histograms[key]
                        running = 0
                        for index, edge in enumerate(family.buckets):
                            running = histogram.counts[index]
                            labels = _render_labels(key, (("le", _format(edge)),))
                            lines.append(f"{name}_bucket{labels} {running}")
                        labels = _render_labels(key, (("le", "+Inf"),))
                        lines.append(f"{name}_bucket{labels} {histogram.count}")
                        plain = _render_labels(key)
                        lines.append(f"{name}_sum{plain} {_format(histogram.total)}")
                        lines.append(f"{name}_count{plain} {histogram.count}")
                else:
                    for key in sorted(family.values):
                        lines.append(f"{name}{_render_labels(key)} {_format(family.values[key])}")
        lines.append("")
        return "\n".join(lines)

    def snapshot(self) -> dict[str, Any]:
        """A JSON-serialisable view, for the benchmarks and for ``/readyz``.

        Histograms report count, sum, mean and bucketed p50/p95/p99 so a
        benchmark can assert on a latency budget without re-deriving it from
        raw buckets.
        """
        counters: dict[str, Any] = {}
        gauges: dict[str, Any] = {}
        histograms: dict[str, Any] = {}
        with self._lock:
            for name in sorted(self._families):
                family = self._families[name]
                if family.kind == "histogram":
                    histograms[name] = {
                        _series_name(key): histogram.snapshot()
                        for key, histogram in sorted(family.histograms.items())
                    }
                    continue
                target = counters if family.kind == "counter" else gauges
                target[name] = {
                    _series_name(key): round(value, 6)
                    for key, value in sorted(family.values.items())
                }
        return {
            "uptime_s": round(time.time() - self._started_at, 3),
            "counters": counters,
            "gauges": gauges,
            "histograms": histograms,
        }

    def reset(self) -> None:
        """Drop every observation. For tests and for a benchmark run."""
        with self._lock:
            self._families.clear()
            self._started_at = time.time()


def _series_name(key: LabelKey) -> str:
    """A stable, readable series key for the JSON snapshot."""
    if not key:
        return "_"
    return ",".join(f"{name}={value}" for name, value in key)


_registry = MetricsRegistry()


def get_metrics() -> MetricsRegistry:
    """The process-wide registry."""
    return _registry


def reset_metrics() -> None:
    """Clear the process-wide registry."""
    _registry.reset()
