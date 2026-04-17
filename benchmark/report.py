"""Generate Markdown benchmark reports from structured job artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
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
    total = len(report.models)
    passed = sum(1 for model in report.models if _is_pass(model.summary))
    pass_rate = _format_percent(passed, total)

    lines: list[str] = [
        f"# Benchmark Results - Job `{report.job_id}`",
        "",
        f"**Task:** {report.task_label}",
        f"**Agent:** {report.agent_label}",
        f"**Models tested:** {total}",
        f"**Pass rate:** {passed}/{total} ({pass_rate})",
        f"**Results dir:** `results/{report.job_id}/`",
        "",
        "_Generated from structured run artifacts. Fill in qualitative commentary manually._",
        "",
        "---",
        "",
        "## Overview",
        "",
    ]
    lines.extend(_render_overview(report.models))
    lines.extend(
        [
            "---",
            "",
            "## Results",
            "",
        ]
    )

    for index, model in enumerate(report.models, start=1):
        lines.extend(_render_model_section(index, model, include_placeholders))

    lines.extend(
        [
            "## Summary Table",
            "",
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

    lines: list[str] = []

    if len(passes) == total:
        headline = f"All {total} models produced a working app (pass rate {pass_rate})."
    elif not passes:
        headline = f"None of the {total} models produced a working app."
    else:
        headline = (
            f"{len(passes)} of {total} models produced a working app "
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
    if summary.get("timed_out"):
        return "timed out"
    finish_reason = summary.get("finish_reason")
    if finish_reason and finish_reason != "completed":
        return str(finish_reason)
    eval_error = summary.get("eval", {}).get("error")
    if eval_error:
        return str(eval_error)
    if summary.get("error"):
        return "agent error"
    http_status = summary.get("eval", {}).get("http_status")
    if http_status and http_status != 200:
        return f"http {http_status}"
    return "unknown failure"


def _describe_failure(summary: dict[str, Any]) -> str:
    parts: list[str] = []
    if summary.get("timed_out"):
        parts.append("timed out")
    finish_reason = summary.get("finish_reason")
    if finish_reason and finish_reason != "completed":
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
        parts.append(f"error: {short}")
    wall = summary.get("wall_seconds")
    if isinstance(wall, (int, float)):
        parts.append(f"{wall:.1f} s")
    return "; ".join(parts) or "unknown failure"


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
        ("Tokens in", str(summary.get("tokens_in", 0))),
        ("Tokens out", str(summary.get("tokens_out", 0))),
        ("API calls", str(summary.get("api_calls", 0))),
        ("Tool calls", str(summary.get("tool_calls", 0))),
    ]

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


def _code_or_dash(value: Any) -> str:
    if not value:
        return "-"
    return f"`{value}`"
