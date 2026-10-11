"""Latency and cost benchmark over the golden scenarios.

    ./do bench                   both suites, with the defaults
    ./do bench --suite engine    the rule engine only, no database needed
    ./do bench --journey-runs 5  five full journeys instead of three

What is measured, and what is not
---------------------------------
Two suites, because "how fast is this product" has two honest answers.

``engine``
    The deterministic rule engine over the 22 hand-labelled scenarios in
    ``backend/tests/fixtures/eligibility_scenarios.json`` - the same file the
    unit suite scores correctness against. Pure CPU: no database, no network,
    no model. Every status is also checked against its hand-written label, so
    this suite reports accuracy next to latency; a benchmark that got faster by
    getting answers wrong would be a regression, not an improvement.

``journey``
    The whole product end to end over HTTP, by running ``scripts.e2e_journey``
    - the real ASGI app, the real database, the real engine, and the
    fixture-replaying model port. Reports per-step latency and what the run
    would cost at the configured token prices.

The model's own latency is **not** in either number. The journey replays
recorded responses instead of calling Gemini, so its timings are the product's
own work - routing, middleware, retrieval, the engine, persistence, the
background runner - and the model call is a dictionary lookup. That is the
useful figure to track in CI, and it is not a figure to quote as end-to-end
user-perceived latency. The live number is reported as ``NOT MEASURED``, with
the command that would measure it.

The cost figures are real token counts at real prices, and are still estimates:
the counts are what the live model actually reported when
``tests/fixtures/gemini/*.json`` were recorded, and the price is
``GEMINI_PRICE_INPUT_PER_MTOK`` / ``GEMINI_PRICE_OUTPUT_PER_MTOK`` from the
configuration. Sampling, a prompt change or a different model would move them,
so they answer "what does one journey cost at today's prompt sizes", not "what
will the bill be".

Privacy
-------
Nothing personal goes into a result file: the profiles are fictional fixtures,
the journey's person is synthetic, and the written JSON carries timings, token
counts, statuses and configuration *names* only. The database URL is never
recorded, because it carries a password.

Results are written to ``benchmarks/results/<timestamp>.local.json`` by
default. That suffix is gitignored on purpose - a local run on a laptop is not
a number anybody else should inherit. Pass ``--out`` to write a result that is
meant to be committed and compared.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import app
from app.core.config import Settings, get_settings
from app.eligibility.engine import ProgramRuleset, evaluate_all
from app.schemas import RuleSet, UserProfile, program_uuid
from scripts.e2e_journey import JourneyFailed, journey_settings, run_journey

__all__ = ["BenchmarkError", "main", "run_engine_suite", "run_journey_suite"]

BACKEND_ROOT: Final[Path] = Path(app.__path__[0]).resolve().parent
SCENARIOS_PATH: Final[Path] = BACKEND_ROOT / "tests" / "fixtures" / "eligibility_scenarios.json"
RESULTS_DIR: Final[Path] = Path(__file__).resolve().parent / "results"

DEFAULT_ENGINE_REPEAT: Final[int] = 50
DEFAULT_JOURNEY_RUNS: Final[int] = 3

#: Numbers this benchmark cannot produce, and the command that would produce
#: them. Emitted verbatim into the result file: an absent number that looks
#: absent is safe, and an absent number that looks present is not.
NOT_MEASURED: Final[dict[str, str]] = {
    "live_model_latency": (
        "NOT MEASURED. Every suite here replays recorded model responses, so no "
        "measurement includes Gemini's own latency. To measure it, point a "
        "server at a working key and run './do smoke --base-url ...', which "
        "drives the same flow over real HTTP against whatever model access that "
        "process has, and read the gemini_call_duration_seconds histogram from "
        "/internal/metrics.json."
    ),
    "live_model_cost": (
        "NOT MEASURED. The cost below is recorded token counts at the configured "
        "price, not a billed amount. To measure the real thing, run the flow "
        "against a live key and read gemini_tokens_total and "
        "gemini_estimated_cost_usd_total from /internal/metrics.json."
    ),
    "concurrency_and_load": (
        "NOT MEASURED. These suites run one request at a time, so nothing here "
        "says anything about behaviour under concurrent load, pool saturation or "
        "queueing. locust is pinned in the backend dev dependencies; no load "
        "profile has been written yet, so no load number exists."
    ),
    "frontend_latency": (
        "NOT MEASURED. No browser timing is collected here. './do e2e-web' is "
        "the hook for it once a Playwright suite exists."
    ),
}


class BenchmarkError(RuntimeError):
    """A suite could not produce a trustworthy number."""


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def _percentile(ordered_ms: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile.

    Deliberately not interpolated: an interpolated p95 of nine samples invents
    a value that was never observed, and these sample counts are small enough
    that it would matter.
    """
    if not ordered_ms:
        raise BenchmarkError("no samples")
    rank = max(1, math.ceil(fraction * len(ordered_ms)))
    return ordered_ms[rank - 1]


def _stats(samples_ms: Sequence[float]) -> dict[str, float | int]:
    """The distribution of one set of timings, in milliseconds."""
    if not samples_ms:
        raise BenchmarkError("no samples")
    ordered = sorted(samples_ms)
    return {
        "samples": len(ordered),
        "min_ms": round(ordered[0], 4),
        "p50_ms": round(_percentile(ordered, 0.50), 4),
        "p95_ms": round(_percentile(ordered, 0.95), 4),
        "p99_ms": round(_percentile(ordered, 0.99), 4),
        "max_ms": round(ordered[-1], 4),
        "mean_ms": round(statistics.fmean(ordered), 4),
    }


# ---------------------------------------------------------------------------
# the rule engine suite
# ---------------------------------------------------------------------------
def _load_scenarios() -> tuple[list[ProgramRuleset], list[dict[str, Any]], datetime, str]:
    """The golden file: its rule sets, its scenarios and the date it is labelled for."""
    if not SCENARIOS_PATH.exists():  # pragma: no cover - a broken checkout
        raise BenchmarkError(f"the golden scenario file is missing: {SCENARIOS_PATH}")
    with SCENARIOS_PATH.open(encoding="utf-8") as handle:
        fixture: dict[str, Any] = json.load(handle)
    rulesets = [RuleSet.model_validate(item) for item in fixture["rulesets"]]
    entries = [
        ProgramRuleset(program_uuid(ruleset.program_slug), ruleset.program_slug, ruleset)
        for ruleset in rulesets
    ]
    evaluated_on = datetime.fromisoformat(str(fixture["evaluated_on"]))
    return entries, list(fixture["scenarios"]), evaluated_on, str(fixture["fixture_version"])


def run_engine_suite(*, repeat: int = DEFAULT_ENGINE_REPEAT) -> dict[str, Any]:
    """Time the rule engine over every golden scenario, and score its answers.

    One sample is one scenario evaluated against every rule set in the file,
    which is the unit of work a single results page needs. Profile parsing is
    done once, outside the timed region, because deserialising JSON is not the
    engine's work.
    """
    entries, scenarios, evaluated_on, fixture_version = _load_scenarios()
    profiles = [UserProfile.model_validate(scenario["profile"]) for scenario in scenarios]

    per_scenario: list[dict[str, Any]] = []
    all_samples: list[float] = []
    mismatches: list[dict[str, str]] = []

    for scenario, profile in zip(scenarios, profiles, strict=True):
        samples: list[float] = []
        observed: dict[str, str] = {}
        for _ in range(repeat):
            started = time.perf_counter_ns()
            evaluations = evaluate_all(entries, profile, now=evaluated_on)
            samples.append((time.perf_counter_ns() - started) / 1_000_000)
            observed = {item.program_slug: item.status.value for item in evaluations}
        all_samples.extend(samples)

        expected: dict[str, str] = dict(scenario["expected"])
        wrong = {
            slug: (expected[slug], observed.get(slug, "MISSING"))
            for slug in expected
            if observed.get(slug) != expected[slug]
        }
        for slug, (want, got) in sorted(wrong.items()):
            mismatches.append(
                {
                    "scenario": str(scenario["name"]),
                    "program": slug,
                    "expected": want,
                    "observed": got,
                }
            )
        per_scenario.append(
            {
                "id": str(scenario["id"]),
                "name": str(scenario["name"]),
                "programs": len(expected),
                "labels_matched": len(expected) - len(wrong),
                **_stats(samples),
            }
        )

    labels = sum(int(entry["programs"]) for entry in per_scenario)
    matched = sum(int(entry["labels_matched"]) for entry in per_scenario)
    return {
        "unit": "one scenario evaluated against every rule set in the golden file",
        "fixture": {
            "path": str(SCENARIOS_PATH.relative_to(BACKEND_ROOT.parent)),
            "version": fixture_version,
            "scenarios": len(scenarios),
            "rule_sets": len(entries),
            "labelled_on": evaluated_on.isoformat(),
        },
        "repeat": repeat,
        "latency": _stats(all_samples),
        "accuracy": {
            "labels": labels,
            "matched": matched,
            "mismatched": labels - matched,
            "mismatches": mismatches,
        },
        "per_scenario": per_scenario,
    }


# ---------------------------------------------------------------------------
# the end-to-end journey suite
# ---------------------------------------------------------------------------
async def run_journey_suite(
    *, runs: int = DEFAULT_JOURNEY_RUNS, database_url: str | None = None
) -> dict[str, Any]:
    """Run the whole product journey ``runs`` times and time every step.

    The journey asserts every transition as it goes, so a benchmark run that
    completes is also a correctness run: if the product stopped working, this
    raises instead of reporting a fast number for a broken flow.

    The application logs to stdout, which is where this report goes, so stdout
    is redirected to stderr for the duration of the runs. The logging still
    happens - it is part of what is being measured - it just does not land in
    the middle of the table. ``./do bench > report.txt`` therefore captures the
    report and leaves the log lines on the terminal.
    """
    settings = journey_settings(database_url=database_url) if database_url else journey_settings()
    step_samples: dict[str, list[float]] = {}
    step_order: list[str] = []
    totals: list[float] = []
    costs: list[float] = []
    token_totals: list[dict[str, int]] = []

    for index in range(runs):
        sink = io.StringIO()
        started = time.perf_counter()
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = await run_journey(settings, out=sink)
        except JourneyFailed as exc:
            # The trace is the only readable account of what broke, and it was
            # captured rather than printed, so print it before giving up.
            print(sink.getvalue(), file=sys.stderr)
            raise BenchmarkError(f"journey run {index + 1} of {runs} failed: {exc}") from exc
        totals.append((time.perf_counter() - started) * 1000)
        for step in result.steps:
            if step.name not in step_samples:
                step_samples[step.name] = []
                step_order.append(step.name)
            step_samples[step.name].append(float(step.duration_ms))
        usage = result.usage
        costs.append(usage.estimated_cost_usd)
        token_totals.append(
            {
                "calls": usage.calls,
                "attempts": usage.attempts,
                "prompt_tokens": usage.prompt_tokens,
                "output_tokens": usage.output_tokens,
                "thought_tokens": usage.thought_tokens,
                "total_tokens": usage.total_tokens,
            }
        )

    live = get_settings()
    return {
        "unit": "one full journey: 14 asserted steps over HTTP against the real app and database",
        "runs": runs,
        "model_port": "fixture replay (app.ai.fake.FakeGeminiClient); no network call is made",
        "app_log_output": "the application's own stdout is redirected to stderr during the runs",
        "journey_total": _stats(totals),
        "per_step": [
            {"step": name, **_stats(step_samples[name])}
            for name in step_order
            if len(step_samples[name]) == runs
        ],
        "cost_per_journey": {
            "basis": (
                "token counts recorded in backend/tests/fixtures/gemini/*.json, priced at "
                "the configured rate; an estimate, not a billed amount"
            ),
            "price_input_usd_per_mtok": live.gemini_price_input_per_mtok,
            "price_output_usd_per_mtok": live.gemini_price_output_per_mtok,
            "estimated_usd_mean": round(statistics.fmean(costs), 8) if costs else 0.0,
            "estimated_usd_max": round(max(costs), 8) if costs else 0.0,
            "tokens_first_run": token_totals[0] if token_totals else {},
        },
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _environment() -> dict[str, Any]:
    """What the numbers below are numbers *about*.

    Configuration names and prices only. The database URL carries a password
    and the API key is a secret, so neither is read here, let alone written.
    """
    settings = get_settings()
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "processor": platform.machine(),
        "configured_model": settings.gemini_model,
        "model_access_configured": settings.gemini_enabled,
        "note": "timings are from this machine and are not comparable across machines",
    }


def _render(report: dict[str, Any], out: Any) -> None:
    """The human-readable form. The JSON file is the machine-readable one."""

    def write(line: str) -> None:
        print(line, file=out, flush=True)

    write("")
    write("=" * 78)
    write("  SecondChance benchmark")
    write("=" * 78)
    environment = report["environment"]
    write(f"  {environment['implementation']} {environment['python']} on {environment['platform']}")
    write(f"  started {report['started_at']}")

    engine = report["suites"].get("engine")
    if engine:
        write("")
        write("-- rule engine over the golden scenarios -------------------------------------")
        fixture = engine["fixture"]
        write(
            f"  {fixture['scenarios']} scenarios x {fixture['rule_sets']} rule sets, "
            f"{engine['repeat']} repeats each"
        )
        latency = engine["latency"]
        write(
            f"  per scenario: p50 {latency['p50_ms']:.3f} ms  p95 {latency['p95_ms']:.3f} ms  "
            f"p99 {latency['p99_ms']:.3f} ms  max {latency['max_ms']:.3f} ms"
        )
        accuracy = engine["accuracy"]
        write(f"  labels matched: {accuracy['matched']}/{accuracy['labels']}")
        for mismatch in accuracy["mismatches"]:
            write(
                f"  MISMATCH {mismatch['scenario']}/{mismatch['program']}: "
                f"expected {mismatch['expected']}, observed {mismatch['observed']}"
            )
        slowest = max(engine["per_scenario"], key=lambda entry: float(entry["p95_ms"]))
        write(f"  slowest scenario: {slowest['name']} at p95 {slowest['p95_ms']:.3f} ms")

    journey = report["suites"].get("journey")
    if journey:
        write("")
        write("-- the whole journey over HTTP (fixture model port) --------------------------")
        total = journey["journey_total"]
        write(
            f"  {journey['runs']} runs: p50 {total['p50_ms']:.0f} ms  "
            f"p95 {total['p95_ms']:.0f} ms  max {total['max_ms']:.0f} ms"
        )
        width = max((len(str(step["step"])) for step in journey["per_step"]), default=0)
        for step in journey["per_step"]:
            write(f"    {step['step']!s:<{width}}  p50 {step['p50_ms']:>7.0f} ms")
        cost = journey["cost_per_journey"]
        write(
            f"  estimated model cost per journey: ${cost['estimated_usd_mean']:.6f} "
            f"(mean), ${cost['estimated_usd_max']:.6f} (max)"
        )
        write(f"  tokens in the first run: {cost['tokens_first_run']}")

    write("")
    write("-- not measured --------------------------------------------------------------")
    for name, reason in report["not_measured"].items():
        write(f"  {name}:")
        for chunk in _wrap(reason, 72):
            write(f"    {chunk}")
    write("")


def _wrap(text: str, width: int) -> list[str]:
    """Greedy wrap. A dependency on textwrap is not worth the import here."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.run",
        description="Latency and cost benchmark over the golden scenarios.",
    )
    parser.add_argument(
        "--suite",
        choices=("engine", "journey", "all"),
        default="all",
        help="Which suites to run. 'engine' needs no database. Default: all.",
    )
    parser.add_argument(
        "--engine-repeat",
        type=int,
        default=DEFAULT_ENGINE_REPEAT,
        help=f"Evaluations per scenario. Default {DEFAULT_ENGINE_REPEAT}.",
    )
    parser.add_argument(
        "--journey-runs",
        type=int,
        default=DEFAULT_JOURNEY_RUNS,
        help=f"Full journeys to run. Default {DEFAULT_JOURNEY_RUNS}.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres URL for the journey suite. Default: TEST_DATABASE_URL.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Where to write the JSON result. Default: "
            "benchmarks/results/<timestamp>.local.json (gitignored). "
            "Use '-' to print the JSON instead of writing it."
        ),
    )
    parser.add_argument(
        "--no-write", action="store_true", help="Do not write a result file at all."
    )
    return parser.parse_args(argv)


def _default_out(started: datetime) -> Path:
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    return RESULTS_DIR / f"{stamp}.local.json"


def _database_url(args: argparse.Namespace) -> str:
    if args.database_url:
        url: str = args.database_url
        return url
    settings: Settings = get_settings()
    return settings.test_database_url


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.engine_repeat < 1 or args.journey_runs < 1:
        print("error: --engine-repeat and --journey-runs must be at least 1", file=sys.stderr)
        return 2

    started = datetime.now(UTC)
    report: dict[str, Any] = {
        "benchmark_version": 1,
        "started_at": started.isoformat(),
        "environment": _environment(),
        "suites": {},
        "not_measured": NOT_MEASURED,
    }

    try:
        if args.suite in ("engine", "all"):
            report["suites"]["engine"] = run_engine_suite(repeat=args.engine_repeat)
        if args.suite in ("journey", "all"):
            report["suites"]["journey"] = asyncio.run(
                run_journey_suite(runs=args.journey_runs, database_url=_database_url(args))
            )
    except BenchmarkError as exc:
        print(f"\nBENCHMARK FAILED: {exc}\n", file=sys.stderr)
        return 1
    except OSError as exc:
        # Almost always "the database is not up" for the journey suite.
        print(
            f"\nBENCHMARK FAILED: {type(exc).__name__} while running the suites.\n"
            "The journey suite needs Postgres and a migrated test database: "
            "./do db-up && ./do migrate\n",
            file=sys.stderr,
        )
        return 1

    report["finished_at"] = datetime.now(UTC).isoformat()
    _render(report, sys.stdout)

    payload = json.dumps(report, indent=2, sort_keys=False)
    if args.no_write:
        pass
    elif args.out == "-":
        print(payload)
    else:
        destination = Path(args.out) if args.out else _default_out(started)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload + "\n", encoding="utf-8")
        print(f"result written to {destination}")

    engine = report["suites"].get("engine")
    if engine and engine["accuracy"]["mismatched"]:
        print(
            f"BENCHMARK FAILED: {engine['accuracy']['mismatched']} golden label(s) did not match "
            "- the engine disagrees with the hand-labelled scenarios.",
            file=sys.stderr,
        )
        return 1
    print("BENCHMARK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
