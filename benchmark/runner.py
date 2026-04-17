"""Orchestrates benchmark runs serially across a list of models."""

import asyncio
import json
import logging
import re
import subprocess
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import AIDER_STAGNATION_SECONDS, TIMEOUT_SECONDS, run_agent
from .evaluator import PORT, evaluate
from .metrics import MetricsCollector
from .process_utils import assert_port_available, sanitized_subprocess_env
from .report import write_markdown_report

logger = logging.getLogger(__name__)

RESULTS_ROOT = Path(__file__).parent.parent / "results"
TASKS_DIR = Path(__file__).parent.parent / "tasks"

# Workspaces live OUTSIDE the repo so `uv init` inside them can't walk up
# and auto-register as a workspace member in our root pyproject.toml.
WORKSPACE_BASE = Path.home() / ".limerick-benchmark" / "workspaces"


def _slug(model_id: str) -> str:
    """Convert a model ID to a filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", model_id)


def _new_job_id() -> str:
    """Build a human-sortable job id (one per `run_benchmark` invocation)."""
    return datetime.now().strftime("%Y%m%d.%H%M%S")


def _run_dir(job_id: str, model_id: str) -> Path:
    """Per-model results directory under the job's collation dir."""
    return RESULTS_ROOT / job_id / _slug(model_id)


def _load_task(task_name: str) -> str:
    path = TASKS_DIR / f"{task_name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")
    return path.read_text()


def _prepare_workspace(workspace: Path) -> None:
    """Initialize the workspace as a uv project with Flask preinstalled."""
    pyproject = workspace / "pyproject.toml"
    if not pyproject.exists():
        subprocess.run(
            ["uv", "init", ".", "--python", "3.12", "--name", workspace.name.replace("_", "-")],
            cwd=workspace,
            check=True,
            capture_output=True,
            env=sanitized_subprocess_env(),
            text=True,
        )

    if _workspace_has_dependency(workspace, "flask"):
        return

    subprocess.run(
        ["uv", "add", "flask"],
        cwd=workspace,
        check=True,
        capture_output=True,
        env=sanitized_subprocess_env(),
        text=True,
    )


def _workspace_has_dependency(workspace: Path, dependency_name: str) -> bool:
    """Return True if the pyproject already declares the dependency."""
    pyproject = workspace / "pyproject.toml"
    if not pyproject.exists():
        return False

    try:
        data = tomllib.loads(pyproject.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return False

    normalized = dependency_name.lower().replace("_", "-")
    for spec in data.get("project", {}).get("dependencies", []) or []:
        token = spec.split(";")[0].strip()
        for marker in ("[", "=", "<", ">", "!", "~"):
            if marker in token:
                token = token.split(marker, 1)[0]
        if token.strip().lower().replace("_", "-") == normalized:
            return True
    return False


def _task_prompt_with_workspace_note(task_prompt: str) -> str:
    """Add a stable environment note that the workspace is already prepared."""
    return (
        "Environment note:\n"
        "- The current directory is already initialized as a uv project with Python 3.12.\n"
        "- Do not run `uv init` again.\n"
        "- Flask is already installed. Do not run `uv add flask`.\n"
        "- Just create the application files and ensure the server starts on port 8181.\n\n"
        f"{task_prompt}"
    )


def _classify_failure(summary: dict[str, Any]) -> str:
    """Pick a stable category label for a failed run."""
    if summary.get("timed_out"):
        return "timeout"
    agent_stop = summary.get("agent_stop") or {}
    if agent_stop.get("category"):
        return str(agent_stop["category"])
    finish_reason = summary.get("finish_reason")
    if finish_reason and finish_reason != "completed":
        return str(finish_reason)
    eval_result = summary.get("eval") or {}
    eval_error = eval_result.get("error")
    if eval_error:
        return f"eval_{eval_error}"
    http_status = eval_result.get("http_status")
    if http_status and http_status != 200:
        return f"http_{http_status}"
    if summary.get("error"):
        return "agent_error"
    return "unknown"


def _should_evaluate(agent_stats: dict[str, Any]) -> bool:
    """Return True when post-run evaluation can still produce meaningful data."""
    finish_reason = agent_stats.get("finish_reason")
    if finish_reason in {"redundant_uv_init_loop", "invalid_tool_loop", "repeated_command_loop", "repeated_file_write_loop"}:
        return False
    if agent_stats.get("error"):
        return False
    return True


async def _run_one(
    model: dict[str, Any],
    task_prompt: str,
    timeout: int,
    aider_stagnation_timeout: int,
    enable_hardware_metrics: bool,
    job_id: str,
    agent_type: str = "react",
    run_label: str = "aider",
) -> dict[str, Any]:
    """Run the full benchmark pipeline for a single model."""
    model_id: str = model["id"]
    provider: str = model.get("provider", "ollama")

    assert_port_available(PORT, f"starting run for {model_id}")

    run_dir = _run_dir(job_id, model_id)
    run_dir.mkdir(parents=True)

    # Workspace is outside the repo to prevent uv from treating our
    # pyproject.toml as a parent workspace when the model runs `uv init`.
    # Nested under the job id so per-job cleanup is one `rm -rf`.
    workspace = WORKSPACE_BASE / job_id / _slug(model_id)
    workspace.mkdir(parents=True)
    _prepare_workspace(workspace)

    # Symlink for convenience so results dir is self-contained for browsing
    (run_dir / "workspace").symlink_to(workspace)

    logger.info("=" * 60)
    logger.info("Model   : %s", model_id)
    logger.info("Provider: %s", provider)
    logger.info("Agent   : %s", agent_type)
    logger.info("Run dir : %s", run_dir)
    logger.info("=" * 60)

    token_state: dict[str, Any] = {
        "tokens_in": 0,
        "tokens_out": 0,
        "api_calls": 0,
        "tool_calls": 0,
    }

    collector = MetricsCollector(
        run_dir / "metrics.csv",
        enable_hardware_metrics=enable_hardware_metrics,
    )
    collector.start(token_state)

    wall_start = time.time()

    agent_stats = await run_agent(
        model_id=model_id,
        provider=provider,
        task_prompt=_task_prompt_with_workspace_note(task_prompt),
        workspace=workspace,
        trace_path=run_dir / "trace.jsonl",
        token_state=token_state,
        timeout=timeout,
        aider_stagnation_timeout=aider_stagnation_timeout,
        agent_type=agent_type,
        run_label=run_label,
    )

    wall_elapsed = round(time.time() - wall_start, 1)
    collector.stop()

    if _should_evaluate(agent_stats):
        assert_port_available(PORT, f"evaluating {model_id}")
        logger.info("Agent done in %.1fs — evaluating…", wall_elapsed)
        eval_result = await evaluate(workspace, run_dir)
    else:
        logger.info("Agent done in %.1fs — skipping evaluation (%s)", wall_elapsed, agent_stats.get("finish_reason"))
        eval_result = {
            "entry_point": None,
            "entry_point_candidates": [],
            "server_started": False,
            "http_status": None,
            "response_bytes": None,
            "error": "evaluation_skipped",
        }

    summary = {
        "model_id": model_id,
        "provider": provider,
        "run_dir": str(run_dir),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": wall_elapsed,
        "timeout_seconds": timeout,
        "aider_stagnation_timeout_seconds": aider_stagnation_timeout,
        **token_state,
        **agent_stats,
        "eval": eval_result,
    }
    summary["passed"] = eval_result.get("http_status") == 200
    summary["failure_category"] = None if summary["passed"] else _classify_failure(summary)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _print_summary(summary)
    return summary


def _print_summary(s: dict[str, Any]) -> None:
    ev = s.get("eval", {})
    status = ev.get("http_status")
    started = ev.get("server_started", False)

    logger.info("─" * 60)
    logger.info("  Model      : %s", s["model_id"])
    logger.info("  Wall time  : %.1fs", s["wall_seconds"])
    logger.info("  Tokens in  : %d", s["tokens_in"])
    logger.info("  Tokens out : %d", s["tokens_out"])
    logger.info("  API calls  : %d", s["api_calls"])
    logger.info("  Tool calls : %d", s["tool_calls"])
    logger.info("  Timed out  : %s", s.get("timed_out", False))
    logger.info("  Server up  : %s", started)
    logger.info("  HTTP status: %s", status)
    if ev.get("error"):
        logger.info("  Eval error : %s", ev["error"])
    logger.info("─" * 60)


async def run_benchmark(
    models: list[dict[str, Any]],
    task_name: str = "limerick",
    timeout: int = TIMEOUT_SECONDS,
    aider_stagnation_timeout: int = AIDER_STAGNATION_SECONDS,
    enable_hardware_metrics: bool = False,
    agent_type: str = "react",
) -> list[dict[str, Any]]:
    """Run all models serially. Returns list of summary dicts."""
    task_prompt = _load_task(task_name)
    RESULTS_ROOT.mkdir(exist_ok=True)

    job_id = _new_job_id()
    job_dir = RESULTS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "task_name": task_name,
                "agent_type": agent_type,
                "timeout_seconds": timeout,
                "aider_stagnation_timeout_seconds": aider_stagnation_timeout,
                "enable_hardware_metrics": enable_hardware_metrics,
                "model_ids": [model["id"] for model in models],
            },
            indent=2,
        )
    )

    logger.info(
        "Starting benchmark job %s: %d model(s), task=%s, timeout=%ds, aider_stagnation_timeout=%ds, hardware_metrics=%s, agent=%s",
        job_id,
        len(models),
        task_name,
        timeout,
        aider_stagnation_timeout,
        enable_hardware_metrics,
        agent_type,
    )
    logger.info("Job dir: %s", job_dir)

    summaries = []
    total = len(models)
    for i, model in enumerate(models, 1):
        logger.info("Run %d/%d: %s", i, total, model["id"])
        slug = model["id"].replace(":", "-")
        run_label = f"{i}/{total}:{slug}:{agent_type}"
        summary = await _run_one(
            model,
            task_prompt,
            timeout,
            aider_stagnation_timeout=aider_stagnation_timeout,
            enable_hardware_metrics=enable_hardware_metrics,
            job_id=job_id,
            agent_type=agent_type,
            run_label=run_label,
        )
        summaries.append(summary)

    report_path = write_markdown_report(job_dir)
    logger.info("Generated report: %s", report_path)

    return summaries
