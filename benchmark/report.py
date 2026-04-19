"""Generate Markdown benchmark reports from structured job artifacts."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median, pstdev
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
REPORTS_ROOT = PROJECT_ROOT / "reports"
TASKS_DIR = PROJECT_ROOT / "tasks"


@dataclass(frozen=True)
class MetricSummary:
    """Optional runtime metrics derived from metrics.csv."""

    sample_count: int
    avg_cpu_percent: float | None
    max_cpu_percent: float | None
    avg_memory_percent: float | None
    max_memory_percent: float | None


@dataclass(frozen=True)
class ModelReport:
    """Report-ready data for a single model run."""

    summary: dict[str, Any]
    metrics: MetricSummary | None

    @property
    def model_id(self) -> str:
        return str(self.summary["model_id"])


@dataclass(frozen=True)
class JobReport:
    """Report-ready data for one benchmark job."""

    job_id: str
    job_dir: Path
    task_label: str
    agent_label: str
    job_metadata: dict[str, Any]
    models: list[ModelReport]


def resolve_job_dir(job_id: str, results_root: Path = RESULTS_ROOT) -> Path:
    """Resolve a job id to an on-disk results directory."""
    job_dir = results_root / job_id
    if not job_dir.exists():
        raise FileNotFoundError(f"Benchmark job not found: {job_dir}")
    if not job_dir.is_dir():
        raise FileNotFoundError(f"Benchmark job path is not a directory: {job_dir}")
    return job_dir


def load_job_report(
    job_dir: Path,
    *,
    task_label: str | None = None,
    agent_label: str | None = None,
) -> JobReport:
    """Load and normalize structured result files for one job."""
    summary_paths = sorted(job_dir.glob("*/summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No summary.json files found under {job_dir}")

    models: list[ModelReport] = []
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text())
        metrics = _load_metric_summary(summary_path.parent / "metrics.csv")
        models.append(ModelReport(summary=summary, metrics=metrics))

    models.sort(key=_model_sort_key)

    return JobReport(
        job_id=job_dir.name,
        job_dir=job_dir,
        task_label=task_label or _infer_task_label(job_dir),
        agent_label=agent_label or _infer_agent_label(job_dir),
        job_metadata=_load_job_metadata(job_dir),
        models=models,
    )


def generate_markdown_report(
    job_dir: Path,
    *,
    task_label: str | None = None,
    agent_label: str | None = None,
    include_placeholders: bool = True,
) -> str:
    """Render a Markdown report for one benchmark job."""
    report = load_job_report(job_dir, task_label=task_label, agent_label=agent_label)
    grouped_models = _group_models(report)
    total = len(report.models)
    unique_models = len(grouped_models)
    passed = sum(1 for model in report.models if _is_pass(model.summary))
    pass_rate = _format_percent(passed, total)
    rounds = report.job_metadata.get("rounds")
    order = report.job_metadata.get("order")
    repeated_job = any(len(runs) > 1 for _, runs in grouped_models)

    lines: list[str] = [
        f"# Benchmark Results - Job `{report.job_id}`",
        "",
        f"**Task:** {report.task_label}",
        f"**Agent:** {report.agent_label}",
        f"**Models tested:** {unique_models}",
        f"**Runs executed:** {total}",
        f"**Pass rate:** {passed}/{total} ({pass_rate})",
    ]
    if rounds is not None:
        lines.append(f"**Rounds:** {rounds}")
    if order:
        lines.append(f"**Order:** {order}")
    lines.extend(
        [
            f"**Results dir:** `results/{report.job_id}/`",
            "",
            "_Generated from structured run artifacts. Fill in qualitative commentary manually._",
            "",
            "---",
            "",
            "## Overview",
            "",
        ]
    )
    lines.extend(_render_overview(report.models))
    lines.extend(
        [
            "---",
            "",
            "## Results",
            "",
        ]
    )

    if repeated_job:
        for index, (model_id, runs) in enumerate(grouped_models, start=1):
            lines.extend(_render_group_section(index, model_id, runs, include_placeholders))
    else:
        for index, model in enumerate(report.models, start=1):
            lines.extend(_render_model_section(index, model, include_placeholders))

    lines.extend(["## Summary Table", ""])
    if repeated_job:
        lines.extend(
            [
                "| # | Model | Runs | Passes | Median | Stddev | P90 | App Hashes | Shapes |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for index, (model_id, runs) in enumerate(grouped_models, start=1):
            lines.append(
                "| "
                f"{index} | {model_id} | {len(runs)} | {_count_passes(runs)}/{len(runs)} | "
                f"{_format_seconds(_median_wall_time(runs))} | {_format_seconds(_wall_time_stddev(runs))} | "
                f"{_format_seconds(_wall_time_p90(runs))} | {_distinct_app_hash_count(runs)} | "
                f"{_distinct_solution_shape_count(runs)} |"
            )
    else:
        lines.extend(
            [
                "| # | Model | Pass | Wall Time | Finish | HTTP | Eval |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for index, model in enumerate(report.models, start=1):
            summary = model.summary
            eval_result = summary.get("eval", {})
            lines.append(
                "| "
                f"{index} | {summary['model_id']} | {_pass_label(summary)} | {_format_seconds(summary.get('wall_seconds'))} | "
                f"{summary.get('finish_reason') or '-'} | {eval_result.get('http_status') or '-'} | "
                f"{eval_result.get('error') or '-'} |"
            )

    if repeated_job:
        lines.extend(_render_order_effects(report.models))

    finish_reasons = Counter((model.summary.get("finish_reason") or "unknown") for model in report.models)
    eval_errors = Counter(
        (model.summary.get("eval", {}).get("error") or "none") for model in report.models
    )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Aggregate Counts",
            "",
            "### Finish Reasons",
            "",
            "| Finish reason | Count |",
            "|---|---|",
        ]
    )
    for finish_reason, count in sorted(finish_reasons.items()):
        lines.append(f"| {finish_reason} | {count} |")

    lines.extend(
        [
            "",
            "### Evaluator Outcomes",
            "",
            "| Evaluator result | Count |",
            "|---|---|",
        ]
    )
    for eval_error, count in sorted(eval_errors.items()):
        lines.append(f"| {eval_error} | {count} |")

    return "\n".join(lines).rstrip() + "\n"


def report_output_path(job_id: str, reports_root: Path | None = None) -> Path:
    """Return the default on-disk path for a generated Markdown report."""
    return (reports_root or REPORTS_ROOT) / f"results_{job_id}.md"


def write_markdown_report(
    job_dir: Path,
    *,
    output_path: Path | None = None,
    task_label: str | None = None,
    agent_label: str | None = None,
    include_placeholders: bool = True,
) -> Path:
    """Render and write a Markdown report, returning the output path."""
    path = output_path or report_output_path(job_dir.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown = generate_markdown_report(
        job_dir,
        task_label=task_label,
        agent_label=agent_label,
        include_placeholders=include_placeholders,
    )
    path.write_text(markdown)
    return path


def _render_overview(models: list[ModelReport]) -> list[str]:
    total = len(models)
    if total == 0:
        return ["_No model runs recorded._", ""]

    passes = [m for m in models if _is_pass(m.summary)]
    fails = [m for m in models if not _is_pass(m.summary)]
    pass_rate = _format_percent(len(passes), total)
    unique_models = len({m.model_id for m in models})
    subject = "runs" if unique_models != total else "models"

    lines: list[str] = []

    if len(passes) == total:
        headline = f"All {total} {subject} produced a working app (pass rate {pass_rate})."
    elif not passes:
        headline = f"None of the {total} {subject} produced a working app."
    else:
        headline = (
            f"{len(passes)} of {total} {subject} produced a working app "
            f"(pass rate {pass_rate})."
        )
    lines.extend([headline, ""])

    if passes:
        wall_times = [
            (m, float(m.summary["wall_seconds"]))
            for m in passes
            if isinstance(m.summary.get("wall_seconds"), (int, float))
        ]
        if wall_times:
            fastest = min(wall_times, key=lambda kv: kv[1])
            slowest = max(wall_times, key=lambda kv: kv[1])
            avg = sum(t for _, t in wall_times) / len(wall_times)
            if fastest[0].model_id == slowest[0].model_id:
                lines.append(
                    f"- Fastest pass: **{fastest[0].model_id}** in {fastest[1]:.1f} s."
                )
            else:
                lines.append(
                    f"- Fastest pass: **{fastest[0].model_id}** ({fastest[1]:.1f} s); "
                    f"slowest pass: **{slowest[0].model_id}** ({slowest[1]:.1f} s); "
                    f"mean pass time {avg:.1f} s."
                )

    if fails:
        buckets = _bucket_failures(fails)
        reason_summary = ", ".join(
            f"{count} {label}" for label, count in buckets.most_common()
        )
        lines.append(f"- Failure breakdown: {reason_summary}.")
        for model in fails:
            lines.append(f"  - `{model.model_id}` — {_describe_failure(model.summary)}")

    lines.append("")
    return lines


def _bucket_failures(fails: list[ModelReport]) -> Counter:
    buckets: Counter = Counter()
    for model in fails:
        buckets[_failure_bucket(model.summary)] += 1
    return buckets


def _failure_bucket(summary: dict[str, Any]) -> str:
    category = summary.get("failure_category")
    if category:
        return str(category)
    agent_stop = summary.get("agent_stop") or {}
    if agent_stop.get("category"):
        return str(agent_stop["category"])
    if summary.get("timed_out"):
        return "timeout"
    finish_reason = summary.get("finish_reason")
    if finish_reason and finish_reason != "completed":
        return str(finish_reason)
    eval_error = summary.get("eval", {}).get("error")
    if eval_error:
        return f"eval_{eval_error}"
    if summary.get("error"):
        return "agent_error"
    http_status = summary.get("eval", {}).get("http_status")
    if http_status and http_status != 200:
        return f"http_{http_status}"
    return "unknown"


def _describe_failure(summary: dict[str, Any]) -> str:
    parts: list[str] = []
    category = summary.get("failure_category") or (summary.get("agent_stop") or {}).get("category")
    if category:
        parts.append(f"**{category}**")
    agent_stop_detail = (summary.get("agent_stop") or {}).get("detail")
    if agent_stop_detail:
        parts.append(str(agent_stop_detail)[:140])
    else:
        finish_reason = summary.get("finish_reason")
        if finish_reason and finish_reason != "completed" and finish_reason != category:
            parts.append(f"finish `{finish_reason}`")
        eval_error = summary.get("eval", {}).get("error")
        if eval_error:
            parts.append(f"eval `{eval_error}`")
        http_status = summary.get("eval", {}).get("http_status")
        if http_status and http_status != 200:
            parts.append(f"HTTP {http_status}")
        agent_error = summary.get("error")
        if agent_error:
            short = str(agent_error).splitlines()[0][:120]
            parts.append(short)
    wall = summary.get("wall_seconds")
    if isinstance(wall, (int, float)):
        parts.append(f"{wall:.1f} s")
    return " — ".join(parts) or "unknown failure"


def _render_model_section(
    index: int,
    model: ModelReport,
    include_placeholders: bool,
) -> list[str]:
    summary = model.summary
    eval_result = summary.get("eval", {})

    rows: list[tuple[str, str]] = [
        ("Result", "PASS" if _is_pass(summary) else "FAIL"),
        ("Wall time", _format_seconds(summary.get("wall_seconds"))),
        ("Finish reason", str(summary.get("finish_reason") or "-")),
        ("Timed out", "yes" if summary.get("timed_out") else "no"),
        ("HTTP status", str(eval_result.get("http_status") or "-")),
        ("Entry point", _code_or_dash(eval_result.get("entry_point"))),
        ("Eval error", str(eval_result.get("error") or "-")),
        ("Response bytes", str(eval_result.get("response_bytes") or "-")),
        ("Tokens in", _format_counter(summary.get("tokens_in"))),
        ("Tokens out", _format_counter(summary.get("tokens_out"))),
        ("API calls", _format_counter(summary.get("api_calls"))),
        ("Tool calls", _format_counter(summary.get("tool_calls"))),
    ]

    agent_stop = summary.get("agent_stop")
    if isinstance(agent_stop, dict) and agent_stop.get("category"):
        detail = agent_stop.get("detail") or ""
        rows.append(("Agent stop", f"`{agent_stop['category']}`" + (f" — {detail}" if detail else "")))

    agent_warning = summary.get("agent_warning")
    if isinstance(agent_warning, dict) and agent_warning.get("category"):
        detail = agent_warning.get("detail") or ""
        rows.append(("Agent warning", f"`{agent_warning['category']}`" + (f" — {detail}" if detail else "")))

    failure_category = summary.get("failure_category")
    if failure_category:
        rows.append(("Failure category", f"`{failure_category}`"))

    aider_stagnation_timeout = summary.get("aider_stagnation_timeout_seconds")
    if aider_stagnation_timeout is not None:
        rows.append(("Aider stagnation timeout", f"{aider_stagnation_timeout} s"))

    if summary.get("error"):
        rows.append(("Agent error", str(summary["error"])))

    if model.metrics is not None:
        rows.extend(
            [
                ("Metric samples", str(model.metrics.sample_count)),
                ("Avg CPU", _format_metric_percent(model.metrics.avg_cpu_percent)),
                ("Max CPU", _format_metric_percent(model.metrics.max_cpu_percent)),
                ("Avg memory", _format_metric_percent(model.metrics.avg_memory_percent)),
                ("Max memory", _format_metric_percent(model.metrics.max_memory_percent)),
            ]
        )

    lines = [
        f"### {index}. {summary['model_id']}",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for label, value in rows:
        lines.append(f"| {label} | {value} |")

    if include_placeholders:
        lines.extend(
            [
                "",
                "**Commentary:** _Add manual notes here._",
            ]
        )

    lines.append("")
    return lines


def _render_group_section(
    index: int,
    model_id: str,
    runs: list[ModelReport],
    include_placeholders: bool,
) -> list[str]:
    pass_count = _count_passes(runs)
    total_runs = len(runs)
    finish_reasons = Counter((run.summary.get("finish_reason") or "unknown") for run in runs)
    eval_errors = Counter((run.summary.get("eval", {}).get("error") or "none") for run in runs)

    rows: list[tuple[str, str]] = [
        ("Runs", str(total_runs)),
        ("Passes", f"{pass_count}/{total_runs} ({_format_percent(pass_count, total_runs)})"),
        ("Median wall time", _format_seconds(_median_wall_time(runs))),
        ("Wall time stddev", _format_seconds(_wall_time_stddev(runs))),
        ("Wall time p90", _format_seconds(_wall_time_p90(runs))),
        ("Fastest pass", _format_seconds(_fastest_pass_time(runs))),
        ("Slowest pass", _format_seconds(_slowest_pass_time(runs))),
        ("Distinct app hashes", str(_distinct_app_hash_count(runs))),
        ("Distinct solution shapes", str(_distinct_solution_shape_count(runs))),
        ("Finish reasons", _format_counter_summary(finish_reasons)),
        ("Evaluator outcomes", _format_counter_summary(eval_errors)),
    ]

    agent_times = _numeric_summary_values(runs, "agent_seconds")
    if agent_times:
        rows.append(("Median agent time", _format_seconds(_median(agent_times))))

    eval_times = _numeric_summary_values(runs, "eval_seconds")
    if eval_times:
        rows.append(("Median eval time", _format_seconds(_median(eval_times))))

    startup_times = _numeric_summary_values(runs, "startup_seconds")
    if startup_times:
        rows.append(("Median startup time", _format_seconds(_median(startup_times))))

    first_edit_times = _numeric_summary_values(runs, "first_edit_seconds")
    if first_edit_times:
        rows.append(("Median first edit", _format_seconds(_median(first_edit_times))))

    sample_counts = [run.metrics.sample_count for run in runs if run.metrics is not None]
    if sample_counts:
        rows.append(("Median metric samples", _format_median_int(sample_counts)))

    memory_peaks = [
        run.metrics.max_memory_percent
        for run in runs
        if run.metrics is not None and run.metrics.max_memory_percent is not None
    ]
    if memory_peaks:
        rows.append(("Median max memory", _format_metric_percent(_median(memory_peaks))))

    lines = [
        f"### {index}. {model_id}",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for label, value in rows:
        lines.append(f"| {label} | {value} |")

    lines.extend(
        [
            "",
            "| Run | Round | Pos | Result | Wall Time | Finish | HTTP | Eval |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for run in runs:
        summary = run.summary
        eval_result = summary.get("eval", {})
        lines.append(
            "| "
            f"{summary.get('run_index') or '-'} | {summary.get('round_index') or '-'} | "
            f"{summary.get('position_in_round') or '-'} | {_pass_label(summary)} | "
            f"{_format_seconds(summary.get('wall_seconds'))} | {summary.get('finish_reason') or '-'} | "
            f"{eval_result.get('http_status') or '-'} | {eval_result.get('error') or '-'} |"
        )

    if include_placeholders:
        lines.extend(["", "**Commentary:** _Add manual notes here._"])

    lines.append("")
    return lines


def _load_metric_summary(metrics_path: Path) -> MetricSummary | None:
    if not metrics_path.exists():
        return None

    cpu_values: list[float] = []
    memory_values: list[float] = []
    sample_count = 0

    with metrics_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_count += 1
            _append_float(cpu_values, row.get("cpu_percent"))
            _append_float(memory_values, row.get("memory_percent"))

    if sample_count == 0:
        return None

    return MetricSummary(
        sample_count=sample_count,
        avg_cpu_percent=_average(cpu_values),
        max_cpu_percent=max(cpu_values) if cpu_values else None,
        avg_memory_percent=_average(memory_values),
        max_memory_percent=max(memory_values) if memory_values else None,
    )


def _append_float(values: list[float], raw_value: str | None) -> None:
    if not raw_value:
        return
    try:
        values.append(float(raw_value))
    except ValueError:
        return


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _model_sort_key(model: ModelReport) -> tuple[int, int, float, str]:
    summary = model.summary
    finish_reason = summary.get("finish_reason")
    wall_seconds = summary.get("wall_seconds")
    if not isinstance(wall_seconds, (int, float)):
        wall_seconds = float("inf")
    return (
        0 if _is_pass(summary) else 1,
        0 if finish_reason == "completed" else 1,
        float(wall_seconds),
        model.model_id,
    )


def _group_models(report: JobReport) -> list[tuple[str, list[ModelReport]]]:
    grouped: dict[str, list[ModelReport]] = {}
    for model in report.models:
        grouped.setdefault(model.model_id, []).append(model)

    for runs in grouped.values():
        runs.sort(key=_run_sort_key)

    ordered_ids: list[str] = []
    for model_id in report.job_metadata.get("model_ids", []):
        if model_id in grouped and model_id not in ordered_ids:
            ordered_ids.append(model_id)
    for model_id in sorted(grouped):
        if model_id not in ordered_ids:
            ordered_ids.append(model_id)
    return [(model_id, grouped[model_id]) for model_id in ordered_ids]


def _run_sort_key(model: ModelReport) -> tuple[int, int, int, str]:
    summary = model.summary
    return (
        int(summary.get("round_index") or 0),
        int(summary.get("position_in_round") or 0),
        int(summary.get("run_index") or 0),
        str(summary.get("started_at") or ""),
    )


def _infer_agent_label(job_dir: Path) -> str:
    job_metadata = _load_job_metadata(job_dir)
    if job_metadata.get("agent_type"):
        return str(job_metadata["agent_type"])

    agent_types: list[str] = []
    for trace_path in job_dir.glob("*/trace.jsonl"):
        with trace_path.open() as f:
            for line in f:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "agent_start" and event.get("agent_type"):
                    agent_types.append(str(event["agent_type"]))
                    break
    if agent_types:
        return Counter(agent_types).most_common(1)[0][0]
    return "unknown"


def _infer_task_label(job_dir: Path) -> str:
    job_metadata = _load_job_metadata(job_dir)
    if job_metadata.get("task_name"):
        return str(job_metadata["task_name"])

    task_files = sorted(TASKS_DIR.glob("*.md"))
    if len(task_files) == 1:
        return task_files[0].stem
    return "unknown"


def _load_job_metadata(job_dir: Path) -> dict[str, Any]:
    metadata_path = job_dir / "job.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return {}


def _is_pass(summary: dict[str, Any]) -> bool:
    if "passed" in summary:
        return bool(summary.get("passed"))
    return summary.get("eval", {}).get("http_status") == 200


def _pass_label(summary: dict[str, Any]) -> str:
    return "PASS" if _is_pass(summary) else "FAIL"


def _format_seconds(seconds: Any) -> str:
    if isinstance(seconds, (int, float)):
        return f"{seconds:.1f} s"
    return "-"


def _format_percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0%"
    percent = (numerator / denominator) * 100
    whole = round(percent)
    if abs(percent - whole) < 0.05:
        return f"{whole}%"
    return f"{percent:.1f}%"


def _format_metric_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}%"


def _format_counter(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _format_counter_summary(counter: Counter) -> str:
    if not counter:
        return "-"
    return ", ".join(f"`{label}` x{count}" for label, count in sorted(counter.items()))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(median(values))


def _stddev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(pstdev(values))


def _p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * 0.9) - 1)
    return float(ordered[index])


def _numeric_summary_values(runs: list[ModelReport], key: str) -> list[float]:
    values: list[float] = []
    for run in runs:
        value = run.summary.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _median_wall_time(runs: list[ModelReport]) -> float | None:
    return _median(_numeric_summary_values(runs, "wall_seconds"))


def _wall_time_stddev(runs: list[ModelReport]) -> float | None:
    return _stddev(_numeric_summary_values(runs, "wall_seconds"))


def _wall_time_p90(runs: list[ModelReport]) -> float | None:
    return _p90(_numeric_summary_values(runs, "wall_seconds"))


def _fastest_pass_time(runs: list[ModelReport]) -> float | None:
    values = [
        float(run.summary["wall_seconds"])
        for run in runs
        if _is_pass(run.summary) and isinstance(run.summary.get("wall_seconds"), (int, float))
    ]
    if not values:
        return None
    return min(values)


def _slowest_pass_time(runs: list[ModelReport]) -> float | None:
    values = [
        float(run.summary["wall_seconds"])
        for run in runs
        if _is_pass(run.summary) and isinstance(run.summary.get("wall_seconds"), (int, float))
    ]
    if not values:
        return None
    return max(values)


def _count_passes(runs: list[ModelReport]) -> int:
    return sum(1 for run in runs if _is_pass(run.summary))


def _distinct_app_hash_count(runs: list[ModelReport]) -> int:
    return len(
        {
            str(run.summary["app_py_sha256"])
            for run in runs
            if run.summary.get("app_py_sha256")
        }
    )


def _solution_shape_key(summary: dict[str, Any]) -> tuple[Any, ...] | None:
    fields = (
        summary.get("uses_render_template_string"),
        summary.get("uses_inline_html"),
        summary.get("route_count"),
        summary.get("dependency_count"),
    )
    if all(value is None for value in fields):
        return None
    return fields


def _distinct_solution_shape_count(runs: list[ModelReport]) -> int:
    return len(
        {
            shape
            for run in runs
            if (shape := _solution_shape_key(run.summary)) is not None
        }
    )


def _render_order_effects(models: list[ModelReport]) -> list[str]:
    by_position: dict[int, list[ModelReport]] = {}
    for model in models:
        position = model.summary.get("position_in_round")
        if not isinstance(position, int):
            continue
        by_position.setdefault(position, []).append(model)

    if not by_position:
        return []

    lines = [
        "",
        "## Order Effects",
        "",
        "| Position | Runs | Pass Rate | Median Wall Time |",
        "|---|---|---|---|",
    ]
    for position in sorted(by_position):
        runs = by_position[position]
        pass_count = _count_passes(runs)
        lines.append(
            "| "
            f"{position} | {len(runs)} | {_format_percent(pass_count, len(runs))} | "
            f"{_format_seconds(_median_wall_time(runs))} |"
        )
    return lines


def _format_median_int(values: list[int]) -> str:
    if not values:
        return "-"
    return str(int(round(float(median(values)))))


def _code_or_dash(value: Any) -> str:
    if not value:
        return "-"
    return f"`{value}`"
