# Repository Guidelines

This file is the single source of instructions for any coding agent working in
this repo. `CLAUDE.md` and `GEMINI.md` are symlinks to this file, so Claude
Code, Codex, Gemini, and Aider all read the same guidance. **Edit `AGENTS.md`
directly** — do not create per-agent variants.

## Project Structure & Module Organization

`benchmark/` contains the core runtime code:

- `__main__.py`: CLI entry point (`uv run benchmark {list,run}`). Parses args,
  loads `models.yaml`, prints the preflight table, dispatches to `runner`.
- `runner.py`: Orchestrates serial model runs. Each `run_benchmark`
  invocation gets a single job id (`YYYYMMDD.HHMMSS`, produced by
  `_new_job_id`) and every per-model run is collated under it:
  results at `results/<job_id>/<slug>/` and workspaces at
  `~/.limerick-benchmark/workspaces/<job_id>/<slug>/` (with a `workspace`
  symlink back to the out-of-tree workspace). Also builds a per-run label
  like `3/10:qwen3.5-9b:aider` that agent backends use as their log prefix.
  Cleaning up old work is `rm -rf results/<job_id>` plus the corresponding
  workspace subtree.
- `agent.py`: Hosts both agent backends.
  - `_run_react`: ReAct loop using `litellm` with a single `bash` tool (60 s
    per-command timeout, 15-minute overall hard limit). Loop-detection guards
    cover repeated commands, redundant `uv init`, repeated full-file rewrites,
    and unknown / malformed tool calls.
  - `_run_aider`: Wraps the Aider CLI (`--agent aider`) with:
    - sliding-window log-line repeat detection (`AIDER_REPEAT_WINDOW`,
      `AIDER_UNIQUE_THRESHOLD`),
    - tail-cycle detection (`_aider_has_repeating_cycle`),
    - per-file edit cap (`AIDER_MAX_EDITS_PER_FILE`),
    - workspace-hash stagnation watch (abort if the workspace tree is
      unchanged for `AIDER_STAGNATION_SECONDS`, currently 300 s by default
      and configurable via `--aider-stagnation-timeout`).
    Every stdout line is printed with the `run_label` prefix so multi-model
    runs remain grep-friendly.
- `evaluator.py`: Starts the generated app and validates it against the task
  requirements. For the limerick task it requires the canonical entry point
  `app.py` and treats any other discovered start command as an
  `entry_point_mismatch` failure. It starts `uv run python app.py`, requires
  HTTP 200 on port 8181, and checks the response body for both a 5-line
  limerick and either a refresh `<meta>` tag or a `setInterval(` call. It
  still records alternative entry point candidates and writes a convenience
  `run.sh` into the results directory.
- `metrics.py`: Background 5-second sampler for CPU/memory and token counts.
  When `--enable-hardware-metrics` is set it also collects GPU utilization,
  GPU power (mW), die temperature, and fan RPM via `powermetrics` (requires
  `sudo`).
- `ollama_utils.py`: Local Ollama model store helpers.
- `process_utils.py`: Port ownership checks, sanitized subprocess env, and
  process-group teardown.
- `report.py`: Generates Markdown benchmark reports from structured job
  artifacts (summary.json, metrics.csv, job.json). It aggregates results,
  calculates pass rates, and provides failure breakdowns.

`prefetch.py` is a separate CLI (exposed as the `prefetch` project script) for
pulling Ollama models. Benchmark inputs live in `tasks/` (Markdown files,
currently only `limerick.md`) and `models.yaml` (the model catalog with
`poc` / `v1` / `recommended` / `qwen_coding` / `exclude` flags). `reports/` contains the
automatically generated Markdown reports for each job. `tests/` mirrors the
`benchmark/` modules one-for-one.

Generated workspaces live **outside** the repo tree at
`~/.limerick-benchmark/workspaces/` so `uv init` inside them cannot walk up
and auto-register as a member of this project's `pyproject.toml`. Never move
them back under the repo. `runner._prepare_workspace` seeds task-specific
data files (for the limerick task, `tasks/limericks.txt` is copied in as
`limericks.txt`).

For **ReAct agents** (default), it does NOT run `uv init` or install
dependencies — project setup is part of what the benchmark measures. For
**Aider**, it pre-bootstraps a `uv` project with `flask` installed, as Aider
cannot run setup commands itself. The environment note prefixed to the task
prompt tells the model what is expected; keep that note in sync if you
change the prep.

## Build, Test, and Development Commands

Use `uv` for all Python workflows — never call `pip`/`python` directly.

- `uv sync`: Install dependencies from `pyproject.toml` / `uv.lock`.
- `uv run benchmark list`: Show locally-pulled Ollama models vs. the catalog.
- `uv run benchmark run --set poc`: Smallest proof-of-concept run.
- `uv run benchmark run --set qwen-coding`: Focused comparison batch for
  `gemma4:e4b` plus the Qwen 3.5 and 3.6 coding models.
- `uv run benchmark run --set recommended --skip-missing`: Full run, skipping
  models that aren't pulled.
- `uv run benchmark run --set local`: Run everything currently in the local
  Ollama store (useful when you just want to score what you have on disk).
- `uv run benchmark run --set reference`: Run only the Anthropic reference
  models (requires `ANTHROPIC_API_KEY`).
- `uv run benchmark run --model gemma4:e2b`: Single model.
- `uv run benchmark run --model gemma4:e2b qwen3.5:9b`: Multiple explicit
  models.
- `uv run benchmark run --set poc --agent aider`: Use the Aider backend
  instead of the default ReAct loop.
- `uv run benchmark run --set poc --agent aider --aider-stagnation-timeout 420`:
  Override the Aider workspace-stagnation watchdog.
- `uv run benchmark run --set poc --timeout 600`: Override the default
  15-minute per-model hard limit.
- `uv run benchmark run --set poc --enable-hardware-metrics`: Include GPU /
  thermal / fan metrics (prompts for `sudo`).
- `uv run benchmark report --job-id 20260417.083818`: Re-generate or view a
  Markdown report for a specific job id. Reports are automatically written
  to `reports/results_<job_id>.md` at the end of every `run`.
- `uv run prefetch --set recommended --dry-run`: Preview required downloads.
- `uv run prefetch --set qwen-coding --dry-run`: Preview downloads for the
  focused Qwen coding comparison batch.
- `uv run prefetch --model gemma4:e2b qwen3.5:9b`: Pull specific models.
- `uv run python -m unittest discover tests`: Run the test suite.

## Coding Style & Naming Conventions

Target Python 3.11+ and keep code stdlib-first unless a dependency (like
`litellm`, `rich`, `pyyaml`, `aiohttp`, `psutil`, or `aider-chat`) is already
declared. 4-space indentation, snake_case for functions/variables, CapWords
for classes, explicit type hints on public helpers. Keep modules focused;
new benchmark pipeline logic belongs in `benchmark/`, not top-level scripts.

Use the `logging` module (`logger = logging.getLogger(__name__)`) for all
output. Do not use `print()` in library code except where the agent loop
already streams live user-facing output (status lines in `_run_react`,
prefixed aider stdout in `_run_aider`, and the `$ command` / `->` echo of
bash tool activity). `rich.console` is acceptable for user-facing CLI tables
in `__main__.py` and `prefetch.py`.

## Testing Guidelines
The `tests/` directory contains `unittest` suites mirroring the `benchmark/`
modules (`test_agent.py`, `test_runner.py`, `test_evaluator.py`,
`test_metrics.py`, `test_process_utils.py`, `test_prefetch.py`,
`test_report.py`). For any change, add or update a test case. Run the full
...
`uv run python -m unittest discover tests` before submitting. At minimum,
exercise the affected CLI path with `uv run benchmark ...` or
`uv run prefetch ...`.

## Commit & Pull Request Guidelines

Recent commits use short, imperative subjects such as
`Add infinite loop detection to Aider agent`,
`Show model index and name in aider log prefix`, and
`Strengthen Aider infinite-loop detection`. Lead with the action, keep the
summary specific, no noisy prefixes. PRs should explain the behavioral
change, list verification commands, and note any environment assumptions
(`ollama serve`, `ANTHROPIC_API_KEY`, `sudo powermetrics`).

## Environment & Safety Notes

This project assumes macOS on Apple Silicon. Do not commit generated
workspaces, `results/` artifacts, or large binaries — the `.gitignore`
already excludes `results/*` (with a `.gitkeep` passthrough) and the
out-of-tree `~/.limerick-benchmark/` workspace root, but don't relax those
rules. Published score writeups (e.g. `results_YYYYMMDD_HHMMSS.md`) live at
the repo root and are intentionally tracked.

Treat `models.yaml` edits carefully: they affect benchmark selection,
download size, and reproducibility. The `exclude` field is the gate for
models that can't run on Apple Silicon (NVIDIA FP4) or won't fit in 64 GB
unified memory — keep it honest.

The ReAct loop runs arbitrary shell commands from model output — only run
it inside workspaces under `~/.limerick-benchmark/workspaces/` and never in
the repo root. The Aider backend runs a subprocess without the bash tool
shim, but still operates on the workspace directory, so the same rule
applies.
