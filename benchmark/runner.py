"""Orchestrates benchmark runs serially across a list of models."""

import asyncio
import json
import logging
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import run_agent, TIMEOUT_SECONDS
from .evaluator import PORT, evaluate
from .metrics import MetricsCollector
from .process_utils import assert_port_available, sanitized_subprocess_env

logger = logging.getLogger(__name__)

RESULTS_ROOT = Path(__file__).parent.parent / "results"
TASKS_DIR = Path(__file__).parent.parent / "tasks"

# Workspaces live OUTSIDE the repo so `uv init` inside them can't walk up
# and auto-register as a workspace member in our root pyproject.toml.
WORKSPACE_BASE = Path.home() / ".limerick-benchmark" / "workspaces"


def _slug(model_id: str) -> str:
    """Convert a model ID to a filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", model_id)


def _run_dir(model_id: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RESULTS_ROOT / f"{ts}_{_slug(model_id)}"


def _load_task(task_name: str) -> str:
    path = TASKS_DIR / f"{task_name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")
    return path.read_text()


def _prepare_workspace(workspace: Path) -> None:
    """Initialize the workspace as a uv project before handing it to the model."""
    if (workspace / "pyproject.toml").exists():
        return

    subprocess.run(
        ["uv", "init", ".", "--name", workspace.name.replace("_", "-")],
        cwd=workspace,
        check=True,
        capture_output=True,
        env=sanitized_subprocess_env(),
        text=True,
    )


def _task_prompt_with_workspace_note(task_prompt: str) -> str:
    """Add a stable environment note that the workspace is already initialized."""
    return (
        "Environment note:\n"
        "- The current directory is already initialized as a uv project.\n"
        "- Do not run `uv init`.\n"
        "- Use `uv add ...`, create the application files, and start the server with `uv run ...`.\n\n"
        f"{task_prompt}"
    )


def _should_evaluate(agent_stats: dict[str, Any]) -> bool:
    """Return True when post-run evaluation can still produce meaningful data."""
    finish_reason = agent_stats.get("finish_reason")
    if finish_reason in {"redundant_uv_init_loop", "invalid_tool_loop", "repeated_command_loop"}:
        return False
    if agent_stats.get("error"):
        return False
    return True


async def _run_one(
    model: dict[str, Any],
    task_prompt: str,
    timeout: int,
    enable_hardware_metrics: bool,
) -> dict[str, Any]:
    """Run the full benchmark pipeline for a single model."""
    model_id: str = model["id"]
    provider: str = model.get("provider", "ollama")

    assert_port_available(PORT, f"starting run for {model_id}")

    run_dir = _run_dir(model_id)
    run_dir.mkdir(parents=True)

    # Workspace is outside the repo to prevent uv from treating our
    # pyproject.toml as a parent workspace when the model runs `uv init`.
    workspace = WORKSPACE_BASE / run_dir.name
    workspace.mkdir(parents=True)
    _prepare_workspace(workspace)

    # Symlink for convenience so results dir is self-contained for browsing
    (run_dir / "workspace").symlink_to(workspace)

    logger.info("=" * 60)
    logger.info("Model   : %s", model_id)
    logger.info("Provider: %s", provider)
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
        **token_state,
        **agent_stats,
        "eval": eval_result,
    }

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
    enable_hardware_metrics: bool = False,
) -> list[dict[str, Any]]:
    """Run all models serially. Returns list of summary dicts."""
    task_prompt = _load_task(task_name)
    RESULTS_ROOT.mkdir(exist_ok=True)

    logger.info(
        "Starting benchmark: %d model(s), task=%s, timeout=%ds, hardware_metrics=%s",
        len(models),
        task_name,
        timeout,
        enable_hardware_metrics,
    )

    summaries = []
    for i, model in enumerate(models, 1):
        logger.info("Run %d/%d: %s", i, len(models), model["id"])
        summary = await _run_one(
            model,
            task_prompt,
            timeout,
            enable_hardware_metrics=enable_hardware_metrics,
        )
        summaries.append(summary)

    return summaries
