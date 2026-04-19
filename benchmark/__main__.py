"""
CLI entry point: python -m benchmark <command>

Usage:
    uv run python -m benchmark list
    uv run python -m benchmark run --set {poc,v1,recommended,qwen-coding,local,reference}
    uv run python -m benchmark run --model gemma4:e2b
    uv run python -m benchmark run --set poc --timeout 300
    uv run python -m benchmark run --set qwen-coding
    uv run python -m benchmark run --set poc --agent aider --aider-stagnation-timeout 420
    uv run python -m benchmark report --job-id 20260417.083818
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from .agent import AIDER_STAGNATION_SECONDS, TIMEOUT_SECONDS
from .model_sets import BENCHMARK_SET_CHOICES, NAMED_MODEL_SETS, format_set_metavar
from .ollama_utils import get_local_models, get_pulled_names
from .report import generate_markdown_report, resolve_job_dir
from .runner import run_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
# LiteLLM is very chatty at INFO level
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
console = Console()

MODELS_YAML = Path(__file__).parent.parent / "models.yaml"


def load_catalog() -> dict:
    with open(MODELS_YAML) as f:
        data = yaml.safe_load(f)

    catalog: dict = {}
    for _family, entries in data.items():
        if isinstance(entries, list):
            for entry in entries:
                catalog[entry["id"]] = entry
    return catalog


def models_for_set(catalog: dict, model_set: str, pulled: set[str]) -> list[dict]:
    def is_runnable(e: dict) -> bool:
        return not e.get("exclude") and e.get("provider") in ("ollama", "anthropic")

    if model_set == "reference":
        return [e for e in catalog.values() if is_runnable(e) and e.get("provider") == "anthropic"]

    if model_set == "local":
        # All catalog entries whose tag is currently pulled, plus unknown locals
        catalog_matches = [e for e in catalog.values() if is_runnable(e) and e["id"] in pulled]
        catalog_ids = {e["id"] for e in catalog_matches}
        local_only = [
            {"id": m.name, "provider": "ollama", "size_gb": round(m.size_gb, 1)}
            for m in get_local_models()
            if m.name not in catalog_ids
        ]
        return catalog_matches + local_only

    if model_set not in NAMED_MODEL_SETS:
        logger.error(
            "Unknown set '%s'. Choose: %s",
            model_set,
            ", ".join(BENCHMARK_SET_CHOICES),
        )
        sys.exit(1)

    key = NAMED_MODEL_SETS[model_set]
    return [e for e in catalog.values() if is_runnable(e) and e.get(key)]


def cmd_list(catalog: dict) -> None:
    """Show all locally-pulled models cross-referenced with the catalog."""
    local_models = get_local_models()
    if not local_models:
        console.print("[yellow]No models found in local Ollama store (is ollama running?)[/yellow]")
        return

    pulled_names = {m.name for m in local_models}
    local_by_name = {m.name: m for m in local_models}

    table = Table(title="Local Ollama models", show_header=True, header_style="bold")
    table.add_column("Model", style="cyan", max_width=52, no_wrap=False)
    table.add_column("Size", justify="right")
    table.add_column("Modified")
    table.add_column("In catalog", justify="center")
    table.add_column("Sets")

    for m in sorted(local_models, key=lambda x: x.name):
        entry = catalog.get(m.name)
        if entry:
            in_catalog = "[green]✓[/green]"
            sets = []
            for set_name, field_name in NAMED_MODEL_SETS.items():
                if entry.get(field_name):
                    sets.append(set_name)
            if entry.get("exclude"):
                sets.append(f"[red]excluded[/red]")
            sets_str = ", ".join(sets) if sets else "—"
        else:
            in_catalog = "[yellow]not in catalog[/yellow]"
            sets_str = "—"

        table.add_row(
            m.name,
            f"{m.size_gb:.1f} GB",
            m.modified,
            in_catalog,
            sets_str,
        )

    console.print()
    console.print(table)

    not_pulled = [
        e for e in catalog.values()
        if e.get("provider") == "ollama"
        and not e.get("exclude")
        and e["id"] not in pulled_names
        and any(e.get(field_name) for field_name in NAMED_MODEL_SETS.values())
    ]
    if not_pulled:
        console.print(
            f"\n[dim]{len(not_pulled)} named-set model(s) not yet pulled "
            f"(use `uv run prefetch --set <set>` or `uv run prefetch --model <id>`)[/dim]"
        )


def preflight_check(models: list[dict], pulled: set[str]) -> bool:
    """
    Print a pre-run table showing local availability. Returns False if any
    Ollama model is missing (caller decides whether to abort).
    """
    table = Table(title="Run plan", show_header=True, header_style="bold")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Provider", justify="center")
    table.add_column("Size", justify="right")
    table.add_column("Status", justify="center")

    missing = []
    for m in models:
        mid = m["id"]
        provider = m.get("provider", "ollama")
        size = m.get("size_gb")
        size_str = f"{size:.1f} GB" if size else "—"

        if provider == "anthropic":
            status = "[blue]cloud[/blue]"
        elif mid in pulled:
            status = "[green]✓ ready[/green]"
        else:
            status = "[red]not pulled[/red]"
            missing.append(mid)

        table.add_row(mid, provider, size_str, status)

    console.print()
    console.print(table)

    if missing:
        console.print(f"\n[red]Missing {len(missing)} model(s) — run prefetch first:[/red]")
        for mid in missing:
            console.print(f"  uv run python prefetch.py --model {mid}")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark",
        description="LLM coding benchmark.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list subcommand
    sub.add_parser("list", help="Show local Ollama models vs catalog")

    # run subcommand
    run_p = sub.add_parser("run", help="Run the benchmark")
    group = run_p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--set",
        dest="model_set",
        choices=list(BENCHMARK_SET_CHOICES),
        metavar=format_set_metavar(BENCHMARK_SET_CHOICES),
        help="Named model set (local = whatever is already pulled)",
    )
    group.add_argument(
        "--model",
        nargs="+",
        metavar="MODEL_ID",
        help="One or more specific model IDs",
    )
    run_p.add_argument("--task", default="limerick", help="Task name (default: limerick)")
    run_p.add_argument(
        "--timeout",
        type=int,
        default=TIMEOUT_SECONDS,
        help=f"Per-model timeout in seconds (default: {TIMEOUT_SECONDS})",
    )
    run_p.add_argument(
        "--aider-stagnation-timeout",
        type=int,
        default=AIDER_STAGNATION_SECONDS,
        help=(
            "Abort an aider run if the workspace stays unchanged for this many "
            f"seconds (default: {AIDER_STAGNATION_SECONDS})"
        ),
    )
    run_p.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip models not yet pulled instead of aborting",
    )
    run_p.add_argument(
        "--agent",
        choices=["react", "aider"],
        default="react",
        help="Agent type (default: react)",
    )
    run_p.add_argument(
        "--enable-hardware-metrics",
        action="store_true",
        help="Collect GPU/thermal/fan metrics via powermetrics (may prompt for sudo)",
    )

    # report subcommand
    report_p = sub.add_parser("report", help="Generate a Markdown report from an existing job")
    report_p.add_argument(
        "--job-id",
        required=True,
        help="Existing benchmark job id (for example 20260417.083818)",
    )
    report_p.add_argument(
        "--output",
        type=Path,
        help="Write the report to a file instead of stdout",
    )
    report_p.add_argument(
        "--task",
        dest="task_label",
        help="Override the task label shown in the report header",
    )
    report_p.add_argument(
        "--agent",
        choices=["react", "aider"],
        dest="agent_label",
        help="Override the agent label shown in the report header",
    )
    report_p.add_argument(
        "--no-placeholders",
        action="store_true",
        help="Omit placeholder commentary lines from the generated Markdown",
    )

    args = parser.parse_args()
    catalog = load_catalog()

    if args.command == "list":
        cmd_list(catalog)
        return

    if args.command == "report":
        job_dir = resolve_job_dir(args.job_id)
        markdown = generate_markdown_report(
            job_dir,
            task_label=args.task_label,
            agent_label=args.agent_label,
            include_placeholders=not args.no_placeholders,
        )
        if args.output:
            args.output.write_text(markdown)
            logger.info("Wrote report to %s", args.output)
        else:
            sys.stdout.write(markdown)
        return

    # --- run ---
    pulled = get_pulled_names()

    if args.model_set:
        models = models_for_set(catalog, args.model_set, pulled)
        if not models:
            logger.error("No runnable models found for set '%s'", args.model_set)
            sys.exit(1)
    else:
        models = []
        for mid in args.model:
            entry = catalog.get(mid)
            if entry is None:
                logger.warning("%s not in catalog — adding with provider=ollama", mid)
                entry = {"id": mid, "provider": "ollama"}
            elif entry.get("exclude"):
                logger.warning("Skipping %s: %s", mid, entry["exclude"])
                continue
            models.append(entry)

        if not models:
            logger.error("No valid models to run")
            sys.exit(1)

    ok = preflight_check(models, pulled)
    if not ok:
        if args.skip_missing:
            models = [m for m in models if m.get("provider") == "anthropic" or m["id"] in pulled]
            console.print(f"\n[yellow]--skip-missing: running {len(models)} available model(s)[/yellow]")
            if not models:
                sys.exit(1)
        else:
            console.print("\n[dim]Use --skip-missing to run only the available models.[/dim]")
            sys.exit(1)

    console.print()
    summaries = asyncio.run(
        run_benchmark(
            models,
            task_name=args.task,
            timeout=args.timeout,
            aider_stagnation_timeout=args.aider_stagnation_timeout,
            enable_hardware_metrics=args.enable_hardware_metrics,
            agent_type=args.agent,
        )
    )

    passed = sum(1 for s in summaries if s.get("eval", {}).get("http_status") == 200)
    logger.info("Done. %d/%d returned HTTP 200.", passed, len(summaries))


if __name__ == "__main__":
    main()
