"""
Prefetch Ollama models needed for a benchmark run.

Usage:
    uv run python prefetch.py --set poc
    uv run python prefetch.py --set v1
    uv run python prefetch.py --set recommended
    uv run python prefetch.py --model gemma4:e2b qwen3.5:9b
    uv run python prefetch.py --set recommended --dry-run
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
console = Console()

MODELS_YAML = Path(__file__).parent / "models.yaml"


def load_catalog(path: Path) -> dict[str, dict]:
    """Return flat {model_id: entry} dict from models.yaml."""
    with open(path) as f:
        data = yaml.safe_load(f)

    catalog: dict[str, dict] = {}
    for _family, entries in data.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            catalog[entry["id"]] = entry
    return catalog


def models_for_set(catalog: dict[str, dict], model_set: str) -> list[dict]:
    """Return Ollama-only, non-excluded models for the given named set."""
    def is_local_ollama(e: dict) -> bool:
        return e.get("provider") == "ollama" and not e.get("exclude")

    key_map = {"poc": "poc", "v1": "v1", "recommended": "recommended"}

    if model_set == "all":
        return [e for e in catalog.values() if is_local_ollama(e)]

    if model_set not in key_map:
        console.print(f"[red]Unknown set '{model_set}'. Choose from: poc, v1, recommended, all[/red]")
        sys.exit(1)

    key = key_map[model_set]
    return [e for e in catalog.values() if is_local_ollama(e) and e.get(key)]


def get_pulled_models() -> set[str]:
    """Return set of model IDs already present in local Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=15,
        )
    except FileNotFoundError:
        console.print("[red]Error: 'ollama' not found. Install Ollama from https://ollama.ai[/red]")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        console.print("[yellow]Warning: 'ollama list' timed out — is ollama running? (try: ollama serve)[/yellow]")
        return set()

    if result.returncode != 0:
        msg = result.stderr.strip() or "unknown error"
        console.print(f"[yellow]Warning: 'ollama list' failed ({msg}) — assuming nothing is pulled[/yellow]")
        return set()

    pulled: set[str] = set()
    for line in result.stdout.splitlines()[1:]:  # first line is header
        parts = line.split()
        if parts:
            pulled.add(parts[0])
    return pulled


def free_space_gb() -> float:
    return shutil.disk_usage(Path.home()).free / (1024 ** 3)


def fmt_size(gb: float | None) -> str:
    return f"{gb:.1f} GB" if gb is not None else "unknown"


def time_estimate(total_gb: float) -> str:
    """Bracket download time between 100 Mbps and 1 Gbps."""
    if total_gb == 0:
        return "nothing to download"
    total_mb = total_gb * 1024

    def fmt(secs: float) -> str:
        if secs < 90:
            return f"{secs:.0f}s"
        elif secs < 3600:
            return f"{secs / 60:.0f}m"
        else:
            return f"{secs / 3600:.1f}h"

    fast = total_mb / 125   # 1 Gbps  = 125 MB/s
    slow = total_mb / 12.5  # 100 Mbps = 12.5 MB/s
    return f"~{fmt(fast)} at 1 Gbps  /  ~{fmt(slow)} at 100 Mbps"


def pull_model(model_id: str) -> bool:
    """Pull one model, streaming ollama's own progress output. Returns True on success."""
    proc = subprocess.run(["ollama", "pull", model_id])
    return proc.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prefetch Ollama models for the benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--set", dest="model_set",
        choices=["poc", "v1", "recommended", "all"],
        metavar="{poc,v1,recommended,all}",
        help="Named model set to prefetch",
    )
    group.add_argument(
        "--model", nargs="+", metavar="MODEL_ID",
        help="One or more specific Ollama model IDs",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show plan without downloading")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    catalog = load_catalog(MODELS_YAML)

    if args.model_set:
        targets = models_for_set(catalog, args.model_set)
        if not targets:
            console.print(f"[yellow]No Ollama models found for set '{args.model_set}'.[/yellow]")
            sys.exit(0)
    else:
        targets = []
        for mid in args.model:
            entry = catalog.get(mid)
            if entry is None:
                console.print(f"[yellow]{mid} not in catalog — will attempt pull anyway[/yellow]")
                targets.append({"id": mid, "provider": "ollama", "size_gb": None})
            elif entry.get("exclude"):
                console.print(f"[yellow]Skipping {mid}: {entry['exclude']}[/yellow]")
            elif entry.get("provider") != "ollama":
                console.print(f"[yellow]Skipping {mid}: not an Ollama model (provider={entry.get('provider')})[/yellow]")
            else:
                targets.append(entry)

    pulled = get_pulled_models()
    have = [t for t in targets if t["id"] in pulled]
    need = [t for t in targets if t["id"] not in pulled]

    # ── Summary table ──────────────────────────────────────────────────────
    table = Table(title="Model prefetch plan", show_header=True, header_style="bold")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Size", justify="right")
    table.add_column("Status", justify="center")

    for e in sorted(have, key=lambda x: x["id"]):
        table.add_row(e["id"], fmt_size(e.get("size_gb")), "[green]✓ already pulled[/green]")
    for e in sorted(need, key=lambda x: x["id"]):
        table.add_row(e["id"], fmt_size(e.get("size_gb")), "[yellow]needs download[/yellow]")

    console.print()
    console.print(table)

    if not need:
        console.print("\n[green]All models already pulled. Nothing to do.[/green]")
        return

    # ── Space / time warnings ──────────────────────────────────────────────
    known_gb = sum(e["size_gb"] for e in need if e.get("size_gb") is not None)
    unknown_count = sum(1 for e in need if e.get("size_gb") is None)
    free_gb = free_space_gb()

    console.print()
    console.print(
        f"  To download : [bold]{len(need)} model(s)[/bold]  "
        f"([bold]{known_gb:.1f} GB[/bold] known"
        + (f" + {unknown_count} size unknown)" if unknown_count else ")")
    )
    console.print(f"  Free space  : [bold]{free_gb:.1f} GB[/bold]", end="")

    if known_gb > free_gb:
        console.print("  [bold red]⚠  INSUFFICIENT SPACE — download will likely fail[/bold red]")
    elif known_gb > free_gb * 0.85:
        console.print("  [bold red]⚠  WARNING: less than 15% free space remaining after download[/bold red]")
    elif known_gb > free_gb * 0.70:
        console.print("  [yellow]⚠  cutting it close — monitor disk space during download[/yellow]")
    else:
        console.print()

    console.print(f"  Est. time   : {time_estimate(known_gb)}")
    console.print()

    if args.dry_run:
        console.print("[dim]Dry run — no downloads performed.[/dim]")
        return

    if not args.yes:
        answer = console.input("Proceed with download? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            console.print("Aborted.")
            return

    # ── Pull ───────────────────────────────────────────────────────────────
    failed: list[str] = []
    for i, entry in enumerate(need, 1):
        mid = entry["id"]
        console.rule(f"[bold blue]{i}/{len(need)}  {mid}[/bold blue]")
        if pull_model(mid):
            console.print(f"[green]✓ {mid}[/green]")
        else:
            console.print(f"[red]✗ {mid} — pull failed[/red]")
            failed.append(mid)

    console.rule()
    if failed:
        console.print(f"[red]Failed ({len(failed)}): {', '.join(failed)}[/red]")
        sys.exit(1)
    else:
        console.print(f"[green]All {len(need)} model(s) pulled successfully.[/green]")


if __name__ == "__main__":
    main()
