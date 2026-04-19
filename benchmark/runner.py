"""Orchestrates benchmark runs serially across a list of models."""

import asyncio
import json
import logging
import random
import re
import subprocess
import time
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
RUN_ORDER_CHOICES = ("balanced", "random", "fixed")
_WORKSPACE_IGNORE_DIR_NAMES = {".venv", ".git", "__pycache__", "node_modules"}
_WORKSPACE_IGNORE_PREFIXES = (".aider.",)


def _slug(model_id: str) -> str:
    """Convert a model ID to a filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", model_id)


def _new_job_id() -> str:
    """Build a human-sortable job id (one per `run_benchmark` invocation)."""
    return datetime.now().strftime("%Y%m%d.%H%M%S")


def _run_dir(job_id: str, run_dir_name: str) -> Path:
    """Per-run results directory under the job's collation dir."""
    return RESULTS_ROOT / job_id / run_dir_name


def _run_dir_name(
    model_id: str,
    *,
    run_index: int,
    total_runs: int,
    round_index: int,
    position_in_round: int,
) -> str:
    """Build a stable per-run directory name.

    Single-run jobs keep the historical layout of `results/<job>/<model-slug>/`.
    Repeated jobs prefix each run with stable indices so repeated model IDs do not
    collide on disk and sort in execution order.
    """
    model_slug = _slug(model_id)
    if total_runs == 1:
        return model_slug
    return (
        f"{run_index:02d}_{model_slug}"
        f"__r{round_index:02d}_p{position_in_round:02d}"
    )


def _round_seconds(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 1)


def _first_meaningful_edit_seconds(workspace: Path, started_epoch_ns: int) -> float | None:
    """Return elapsed seconds until the first non-cache workspace file edit."""
    first_edit_ns: int | None = None
    if not workspace.exists():
        return None

    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(workspace)
        except ValueError:
            continue
        if any(part in _WORKSPACE_IGNORE_DIR_NAMES for part in rel.parts):
            continue
        if any(part.startswith(_WORKSPACE_IGNORE_PREFIXES) for part in rel.parts):
            continue
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            continue
        if mtime_ns < started_epoch_ns:
            continue
        if first_edit_ns is None or mtime_ns < first_edit_ns:
            first_edit_ns = mtime_ns

    if first_edit_ns is None:
        return None
    return _round_seconds(max(0.0, (first_edit_ns - started_epoch_ns) / 1_000_000_000))


def _ordered_models_for_round(
    models: list[dict[str, Any]],
    *,
    round_index: int,
    order: str,
    rng: random.Random | None,
) -> list[dict[str, Any]]:
    """Return the execution order for one round."""
    if order == "fixed" or len(models) <= 1:
        return list(models)
    if order == "balanced":
        offset = (round_index - 1) % len(models)
        return list(models[offset:] + models[:offset])
    if order == "random":
        shuffled = list(models)
        assert rng is not None
        rng.shuffle(shuffled)
        return shuffled
    raise ValueError(f"Unknown run order: {order}")


def _build_run_plan(
    models: list[dict[str, Any]],
    *,
    rounds: int,
    order: str,
    seed: int | None,
) -> list[dict[str, Any]]:
    """Expand a unique model list into concrete per-run schedule entries."""
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if order not in RUN_ORDER_CHOICES:
        raise ValueError(f"Unknown run order: {order}")

    rng = random.Random(seed) if order == "random" else None
    plan: list[dict[str, Any]] = []
    total_runs = len(models) * rounds
    run_index = 0

    for round_index in range(1, rounds + 1):
        round_models = _ordered_models_for_round(
            models,
            round_index=round_index,
            order=order,
            rng=rng,
        )
        for position_in_round, model in enumerate(round_models, start=1):
            run_index += 1
            plan.append(
                {
                    "model": model,
                    "run_index": run_index,
                    "total_runs": total_runs,
                    "round_index": round_index,
                    "position_in_round": position_in_round,
                    "run_dir_name": _run_dir_name(
                        model["id"],
                        run_index=run_index,
                        total_runs=total_runs,
                        round_index=round_index,
                        position_in_round=position_in_round,
                    ),
                }
            )
    return plan


def _load_task(task_name: str) -> str:
    path = TASKS_DIR / f"{task_name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")
    return path.read_text()


def _prepare_workspace(
    workspace: Path,
    task_name: str | None = None,
    *,
    agent_type: str = "react",
) -> None:
    """Seed the workspace with task resources and, for file-only agents,
    the project scaffold.

    ReAct agents have a bash tool and are expected to run setup themselves
    (`uv init`, `uv add flask`, etc.) — that is part of what the benchmark
    measures for those agents. Aider runs in headless `--exit` mode with no
    shell execution, so it can only edit files; for Aider we pre-create a
    uv project with Flask installed, otherwise every run would fail with
    `ModuleNotFoundError: flask`.
    """
    if task_name:
        _seed_task_resources(workspace, task_name)

    if agent_type == "aider":
        _bootstrap_uv_project_with_flask(workspace)


def _bootstrap_uv_project_with_flask(workspace: Path) -> None:
    """Run `uv init` and `uv add flask` in the workspace."""
    if not (workspace / "pyproject.toml").exists():
        subprocess.run(
            ["uv", "init", ".", "--python", "3.12", "--name", workspace.name.replace("_", "-")],
            cwd=workspace,
            check=True,
            capture_output=True,
            env=sanitized_subprocess_env(),
            text=True,
        )
        # Drop the stock main.py uv creates so it can't shadow app.py at eval.
        stock_main = workspace / "main.py"
        if stock_main.exists():
            stock_main.unlink()

    subprocess.run(
        ["uv", "add", "flask"],
        cwd=workspace,
        check=True,
        capture_output=True,
        env=sanitized_subprocess_env(),
        text=True,
    )


def _seed_task_resources(workspace: Path, task_name: str) -> None:
    """Copy any task-specific data files into the workspace."""
    if task_name == "limerick":
        src = TASKS_DIR / "limericks.txt"
        if src.exists():
            (workspace / "limericks.txt").write_bytes(src.read_bytes())


def _task_prompt_with_workspace_note(
    task_prompt: str,
    *,
    task_name: str | None = None,
    agent_type: str = "react",
) -> str:
    """Add a stable environment note describing the workspace state."""
    notes: list[str] = []
    if agent_type == "aider":
        notes.extend(
            [
                "- The current directory is already a uv project (Python 3.12) with Flask installed.",
                "- Do not run `uv init` or `uv add flask` — they are already done.",
                "- Create `app.py` as the entry point. Do not create a `main.py`.",
            ]
        )
    else:
        notes.extend(
            [
                "- The current directory is empty except for any task data files listed below.",
                "- Python 3.12 and `uv` are available on PATH. Setting up the project "
                "(`uv init . && uv add flask`, or `pip install flask` in a venv) is your job.",
                "- Create `app.py` as the entry point; do not leave a stock `main.py` behind.",
            ]
        )
    if task_name == "limerick":
        notes.append(
            "- A file `limericks.txt` with 20 pre-written limericks is already in "
            "this directory. Read from it instead of generating limericks in your reply."
        )
    notes.append(
        "- Write code (and run shell commands if you have that tool). Do NOT output limericks, "
        "poems, or long literal data in your chat replies — put data in files."
    )
    return "Environment note:\n" + "\n".join(notes) + "\n\n" + task_prompt


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
    if finish_reason in {
        "redundant_uv_init_loop",
        "invalid_tool_loop",
        "repeated_command_loop",
        "repeated_file_write_loop",
        "stuck_loop",
    }:
        return False
    if agent_stats.get("error"):
        return False
    return True


def _normalize_agent_stats_for_eval(
    agent_stats: dict[str, Any],
    eval_result: dict[str, Any],
) -> dict[str, Any]:
    """Convert late non-fatal Aider parser rejects into warnings on passing runs."""
    normalized = dict(agent_stats)
    if (
        eval_result.get("passed")
        and normalized.get("finish_reason") == "aider_edit_format_reject"
    ):
        warning = normalized.get("agent_stop")
        if warning:
            normalized["agent_warning"] = warning
        normalized["finish_reason"] = "completed"
        normalized["agent_stop"] = None
    return normalized


async def _run_one(
    model: dict[str, Any],
    task_prompt: str,
    timeout: int,
    aider_stagnation_timeout: int,
    enable_hardware_metrics: bool,
    job_id: str,
    run_index: int,
    total_runs: int,
    round_index: int,
    position_in_round: int,
    total_rounds: int,
    run_dir_name: str,
    agent_type: str = "react",
    run_label: str = "aider",
    task_name: str | None = None,
) -> dict[str, Any]:
    """Run the full benchmark pipeline for a single model."""
    model_id: str = model["id"]
    provider: str = model.get("provider", "ollama")

    assert_port_available(PORT, f"starting run for {model_id}")

    run_dir = _run_dir(job_id, run_dir_name)
    run_dir.mkdir(parents=True)

    # Workspace is outside the repo to prevent uv from treating our
    # pyproject.toml as a parent workspace when the model runs `uv init`.
    # Nested under the job id so per-job cleanup is one `rm -rf`.
    workspace = WORKSPACE_BASE / job_id / run_dir_name
    workspace.mkdir(parents=True)
    _prepare_workspace(workspace, task_name=task_name, agent_type=agent_type)

    # Symlink for convenience so results dir is self-contained for browsing
    (run_dir / "workspace").symlink_to(workspace)

    logger.info("=" * 60)
    logger.info("Model   : %s", model_id)
    logger.info("Provider: %s", provider)
    logger.info("Agent   : %s", agent_type)
    logger.info("Run dir : %s", run_dir)
    logger.info("=" * 60)

    if agent_type == "aider":
        token_state: dict[str, Any] = {
            "tokens_in": None,
            "tokens_out": None,
            "api_calls": None,
            "tool_calls": None,
        }
    else:
        token_state = {
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

    started_at = datetime.now(timezone.utc).isoformat()
    run_started_epoch_ns = time.time_ns()
    run_started_monotonic = time.monotonic()

    agent_stats = await run_agent(
        model_id=model_id,
        provider=provider,
        task_prompt=_task_prompt_with_workspace_note(task_prompt, task_name=task_name, agent_type=agent_type),
        workspace=workspace,
        trace_path=run_dir / "trace.jsonl",
        token_state=token_state,
        timeout=timeout,
        aider_stagnation_timeout=aider_stagnation_timeout,
        agent_type=agent_type,
        run_label=run_label,
    )

    agent_finished_at = datetime.now(timezone.utc).isoformat()
    agent_seconds = _round_seconds(time.monotonic() - run_started_monotonic)
    first_edit_seconds = _first_meaningful_edit_seconds(workspace, run_started_epoch_ns)

    eval_started_at: str | None = None
    eval_finished_at: str | None = None
    eval_seconds: float | None = None
    if _should_evaluate(agent_stats):
        assert_port_available(PORT, f"evaluating {model_id}")
        logger.info("Agent done in %.1fs — evaluating…", agent_seconds or 0.0)
        eval_started_at = datetime.now(timezone.utc).isoformat()
        eval_started_monotonic = time.monotonic()
        eval_result = await evaluate(workspace, run_dir)
        eval_seconds = _round_seconds(time.monotonic() - eval_started_monotonic)
        eval_finished_at = datetime.now(timezone.utc).isoformat()
    else:
        logger.info(
            "Agent done in %.1fs — skipping evaluation (%s)",
            agent_seconds or 0.0,
            agent_stats.get("finish_reason"),
        )
        eval_result = {
            "entry_point": None,
            "entry_point_candidates": [],
            "entry_point_mismatch": False,
            "server_started": False,
            "http_status": None,
            "response_bytes": None,
            "body_has_refresh_mechanism": False,
            "body_has_limerick_shape": False,
            "startup_seconds": None,
            "passed": False,
            "error": "evaluation_skipped",
        }

    finished_at = datetime.now(timezone.utc).isoformat()
    wall_elapsed = _round_seconds(time.monotonic() - run_started_monotonic) or 0.0
    collector.stop()

    normalized_agent_stats = _normalize_agent_stats_for_eval(agent_stats, eval_result)

    summary = {
        "model_id": model_id,
        "provider": provider,
        "run_dir": str(run_dir),
        "started_at": started_at,
        "agent_finished_at": agent_finished_at,
        "eval_started_at": eval_started_at,
        "eval_finished_at": eval_finished_at,
        "finished_at": finished_at,
        "run_index": run_index,
        "total_runs": total_runs,
        "round_index": round_index,
        "position_in_round": position_in_round,
        "total_rounds": total_rounds,
        "wall_seconds": wall_elapsed,
        "agent_seconds": agent_seconds,
        "eval_seconds": eval_seconds,
        "first_edit_seconds": first_edit_seconds,
        "startup_seconds": eval_result.get("startup_seconds"),
        "timeout_seconds": timeout,
        "aider_stagnation_timeout_seconds": aider_stagnation_timeout,
        **token_state,
        **normalized_agent_stats,
        "eval": eval_result,
    }
    summary["passed"] = bool(eval_result.get("passed"))
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
    if isinstance(s.get("agent_seconds"), (int, float)):
        logger.info("  Agent time : %.1fs", s["agent_seconds"])
    if isinstance(s.get("eval_seconds"), (int, float)):
        logger.info("  Eval time  : %.1fs", s["eval_seconds"])
    if isinstance(s.get("startup_seconds"), (int, float)):
        logger.info("  Startup    : %.1fs", s["startup_seconds"])
    logger.info("  Tokens in  : %s", _format_counter(s.get("tokens_in")))
    logger.info("  Tokens out : %s", _format_counter(s.get("tokens_out")))
    logger.info("  API calls  : %s", _format_counter(s.get("api_calls")))
    logger.info("  Tool calls : %s", _format_counter(s.get("tool_calls")))
    logger.info("  Timed out  : %s", s.get("timed_out", False))
    logger.info("  Server up  : %s", started)
    logger.info("  HTTP status: %s", status)
    if ev.get("error"):
        logger.info("  Eval error : %s", ev["error"])
    logger.info("─" * 60)


def _format_counter(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


async def run_benchmark(
    models: list[dict[str, Any]],
    task_name: str = "limerick",
    timeout: int = TIMEOUT_SECONDS,
    aider_stagnation_timeout: int = AIDER_STAGNATION_SECONDS,
    enable_hardware_metrics: bool = False,
    agent_type: str = "react",
    rounds: int = 1,
    order: str = "balanced",
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Run all models serially. Returns list of summary dicts."""
    task_prompt = _load_task(task_name)
    RESULTS_ROOT.mkdir(exist_ok=True)
    run_plan = _build_run_plan(models, rounds=rounds, order=order, seed=seed)

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
                "rounds": rounds,
                "order": order,
                "seed": seed,
                "total_runs": len(run_plan),
                "run_plan": [
                    {
                        "run_index": entry["run_index"],
                        "round_index": entry["round_index"],
                        "position_in_round": entry["position_in_round"],
                        "model_id": entry["model"]["id"],
                        "run_dir_name": entry["run_dir_name"],
                    }
                    for entry in run_plan
                ],
            },
            indent=2,
        )
    )

    logger.info(
        "Starting benchmark job %s: %d model(s), %d round(s), %d total run(s), order=%s, seed=%s, task=%s, timeout=%ds, aider_stagnation_timeout=%ds, hardware_metrics=%s, agent=%s",
        job_id,
        len(models),
        rounds,
        len(run_plan),
        order,
        seed,
        task_name,
        timeout,
        aider_stagnation_timeout,
        enable_hardware_metrics,
        agent_type,
    )
    logger.info("Job dir: %s", job_dir)

    summaries = []
    total = len(run_plan)
    for entry in run_plan:
        model = entry["model"]
        logger.info(
            "Run %d/%d: round %d/%d pos %d/%d %s",
            entry["run_index"],
            total,
            entry["round_index"],
            rounds,
            entry["position_in_round"],
            len(models),
            model["id"],
        )
        slug = model["id"].replace(":", "-")
        run_label = f"{entry['run_index']}/{total}:r{entry['round_index']}p{entry['position_in_round']}:{slug}:{agent_type}"
        summary = await _run_one(
            model,
            task_prompt,
            timeout,
            aider_stagnation_timeout=aider_stagnation_timeout,
            enable_hardware_metrics=enable_hardware_metrics,
            job_id=job_id,
            run_index=entry["run_index"],
            total_runs=total,
            round_index=entry["round_index"],
            position_in_round=entry["position_in_round"],
            total_rounds=rounds,
            run_dir_name=entry["run_dir_name"],
            agent_type=agent_type,
            run_label=run_label,
            task_name=task_name,
        )
        summaries.append(summary)
        result = "PASS" if summary.get("passed") else "FAIL"
        detail = "" if summary.get("passed") else f" ({summary.get('failure_category') or 'unknown'})"
        logger.info(
            "Run %d/%d done: round %d pos %d %s — %s%s in %.1fs",
            entry["run_index"],
            total,
            entry["round_index"],
            entry["position_in_round"],
            model["id"],
            result,
            detail,
            summary.get("wall_seconds", 0.0),
        )

    report_path = write_markdown_report(job_dir)
    logger.info("Generated report: %s", report_path)

    return summaries
