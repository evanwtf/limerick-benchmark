# limerick-benchmark

A benchmark suite for evaluating LLM coding ability on local Apple Silicon hardware. Each model gets a coding task, a tool to act on a workspace, and a hard 15-minute timebox to produce a working result — no hints, no hand-holding.

## Quick start

```bash
# Install dependencies
uv sync

# See what models are already pulled locally
uv run benchmark list

# Preview or pull the recommended local models
uv run prefetch --set recommended --dry-run
uv run prefetch --set recommended

# Pull a specific set of models
uv run prefetch --model gemma4:e2b qwen3.5:9b

# Run a fast proof-of-concept pass
uv run benchmark run --set poc

# Run the recommended benchmark set with the default ReAct agent
uv run benchmark run --set recommended

# Run the recommended benchmark set with Aider
uv run benchmark run --agent aider --set recommended

# Skip any missing Ollama models instead of aborting
uv run benchmark run --agent aider --set recommended --skip-missing

# Run only models already present in your local Ollama store
uv run benchmark run --set local

# Run only the Anthropic reference models
uv run benchmark run --set reference

# Increase the aider stagnation watchdog from the 300s default
uv run benchmark run --agent aider --set recommended --aider-stagnation-timeout 420

# Generate a Markdown report for an existing job
uv run benchmark report --job-id 20260417.083818

# Write the report to a file
uv run benchmark report --job-id 20260417.083818 --output results_20260417.083818.md

# Re-run a generated app manually from a finished result directory
cd results/20260417.083818/gemma4_e2b && ./run.sh
```

## How it works

1. The runner starts a fresh agent session for each model, serially (never two at once, so the GPU is never contested).
2. The workspace is pre-initialized as a `uv` project on Python 3.12 with Flask already installed, and lives **outside** the repo at `~/.limerick-benchmark/workspaces/<timestamp>_<slug>/`. The task prompt is prefixed with an environment note telling the model to skip `uv init` / `uv add`.
3. One of two agent backends drives the run:
   - **ReAct (default)** — `litellm` + a single `bash` tool. Loop-detection guards abort on repeated commands, redundant `uv init`, or the same file being overwritten in a tight loop.
   - **Aider (`--agent aider`)** — the Aider CLI in headless mode, wrapped with log-repeat detection, per-file edit caps, and a workspace-hash stagnation watch that kills the run if the tree is unchanged for 300 seconds by default (`--aider-stagnation-timeout` to override).
4. The run is **hard-killed after 15 minutes** by default (override with `--timeout`).
5. System metrics (CPU, memory, token counts, and optionally GPU / thermal / fan data) are sampled every 5 seconds throughout the run.
6. After the run, the evaluator auto-discovers entry points (`run.sh`, `[project.scripts]`, `app.py` / `main.py` / `server.py` / `web.py`, Flask-containing `.py` files, and `python -m <pkg>`), starts the server, and checks for HTTP 200 on port 8181. A convenience `run.sh` is written to the results directory for manual re-evaluation.

## Current task: Limerick Generator

See [`tasks/limerick.md`](tasks/limerick.md) for the full prompt.

> Create a Python web application using Flask and `uv` for package management. The app should generate a random limerick and display it in the browser, show a new one every 5 seconds, and listen on port 8181.

## Scoring

The first two checks are automated pass/fail gates — if the server doesn't start, there's nothing to evaluate.

| Check | Type | Points |
|---|---|---|
| Server starts without error | Automated pass/fail | Gate |
| `GET /` returns HTTP 200 | Automated pass/fail | Gate |
| Page displays a recognizable limerick | Human | 0–40 |
| Auto-refreshes with a new limerick every ~5 seconds | Human | 0–30 |
| Code quality / approach (uv usage, structure, etc.) | Human | 0–30 |

**Reference**: Claude Opus 4.7 = 100 points. All other scores are relative.

See [`results_20260417_073034.md`](results_20260417_073034.md) for an example multi-model scoring writeup.

## Prerequisites

- macOS with Apple Silicon (developed on M5 Max, 64 GB)
- [Ollama](https://ollama.ai) installed and running (`ollama serve`)
- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- `ANTHROPIC_API_KEY` set (for reference model runs)
- `sudo` access only if you choose `--enable-hardware-metrics`

## More commands

```bash
# Install dependencies
uv sync

# See what models are locally available and how they map to the catalog
uv run benchmark list

# Prefetch models you don't have yet
uv run prefetch --set poc
uv run prefetch --set recommended --dry-run   # preview first
uv run prefetch --model gemma4:e2b qwen3.5:9b # pull specific models

# Run the POC (single model used for quick harness validation)
uv run benchmark run --set poc

# Opt in to privileged Apple Silicon hardware metrics
uv run benchmark run --set poc --enable-hardware-metrics

# Run only models already in your local Ollama store
uv run benchmark run --set local

# Run the full recommended set (top 10 local + 3 Anthropic reference)
uv run benchmark run --set recommended

# Run a single model
uv run benchmark run --model gemma4:e2b

# Run multiple explicit models
uv run benchmark run --model gemma4:e2b qwen3.5:9b

# Run Anthropic reference models only (requires ANTHROPIC_API_KEY)
uv run benchmark run --set reference

# Use the Aider backend instead of the default ReAct loop
uv run benchmark run --set poc --agent aider

# Override the default 300-second Aider stagnation watchdog
uv run benchmark run --set poc --agent aider --aider-stagnation-timeout 420

# Override the default 15-minute per-model hard limit
uv run benchmark run --set poc --timeout 600

# Skip models that aren't pulled instead of aborting
uv run benchmark run --set recommended --skip-missing
```

After each run, open the result directory and run `./run.sh` to start the generated server, then browse to http://localhost:8181 to evaluate manually.

## CLI flags

### `uv run benchmark run`

| Flag | Default | Notes |
|---|---|---|
| `--set {poc,v1,recommended,local,reference}` | — | Named model set. `local` = whatever is already pulled; `reference` = Anthropic cloud models. Mutually exclusive with `--model`. |
| `--model MODEL_ID [MODEL_ID …]` | — | One or more specific model IDs. Unknown IDs are treated as Ollama models. Mutually exclusive with `--set`. |
| `--task NAME` | `limerick` | Task file name (without `.md`) in `tasks/`. |
| `--timeout SECONDS` | 900 | Per-model hard limit. |
| `--agent {react,aider}` | `react` | Agent backend. |
| `--aider-stagnation-timeout SECONDS` | 300 | Abort an Aider run if the workspace tree stays unchanged this long. |
| `--skip-missing` | off | Skip Ollama models that aren't pulled instead of aborting the run plan. |
| `--enable-hardware-metrics` | off | Collect GPU / thermal / fan metrics via `powermetrics` (prompts for `sudo`). |

### `uv run prefetch`

| Flag | Default | Notes |
|---|---|---|
| `--set {poc,v1,recommended,all}` | — | Named model set. Mutually exclusive with `--model`. |
| `--model MODEL_ID [MODEL_ID …]` | — | Specific model IDs to pull. |
| `--dry-run` | off | Show the plan without downloading. |
| `--yes` / `-y` | off | Skip the confirmation prompt. |

## Model sets

### Reference models (Anthropic cloud)

| Model ID | Label | Role |
|---|---|---|
| `claude-opus-4-7` | Claude Opus 4.7 | Reference — 100 pts |
| `claude-sonnet-4-6` | Claude Sonnet 4.6 | Reference — better |
| `claude-haiku-4-5-20251001` | Claude Haiku 4.5 | Reference — good |

### POC model

For initial testing of the harness, use this one model while debugging the benchmark loop:

| Model ID | Size | Why |
|---|---|---|
| `gemma4:e2b` | 7.2 GB | Current POC target while debugging the `ollama_chat/gemma4:e2b` run behavior |

### Recommended benchmark set (top 10)

Chosen to answer specific questions: does size help? does a coding fine-tune beat a larger general model? does Apple MLX runtime differ from GGUF? does Qwen 3.6 beat 3.5 at the same weight class?

| # | Model ID | Size | Family | Question it answers |
|---|---|---|---|---|
| 1 | `qwen3.5:9b` | 6.6 GB | Qwen 3.5 | Small/fast Qwen baseline |
| 2 | `gemma4:e2b` | 7.2 GB | Gemma 4 | Small/fast Gemma baseline |
| 3 | `gemma4:e4b` | 9.6 GB | Gemma 4 | One size up from e2b |
| 4 | `gemma4:e2b-mlx-bf16` | 10 GB | Gemma 4 MLX | Same model as #2, Apple MLX runtime — runtime comparison |
| 5 | `qwen3.5:27b-coding-mxfp8` | 31 GB | Qwen 3.5 Coder | Does coding fine-tune beat a bigger general model? |
| 6 | `qwen3.5:35b-a3b` | 24 GB | Qwen 3.5 | Gen 3.5 large MoE baseline |
| 7 | `gemma4:26b` | 18 GB | Gemma 4 | Large Gemma — quality jump from small? |
| 8 | `qwen3.5:35b-a3b-coding-mxfp8` | 38 GB | Qwen 3.5 Coder | Coding fine-tune at 35B scale |
| 9 | `qwen3.6:latest` | 24 GB | Qwen 3.6 | Same size as #6 — does gen 3.6 matter? |
| 10 | `qwen3.6:35b-a3b-q8_0` | 39 GB | Qwen 3.6 | Gen 3.6 at higher precision |

Full catalog with all variants and exclusion notes: [`models.yaml`](models.yaml)

## Result directory layout

Each `benchmark run` invocation gets a single **job ID** (`YYYYMMDD.HHMMSS`) and all per-model results are collated under it, so cleaning up an old run is `rm -rf results/<job_id>`:

```
results/
└── 20260416.143022/                             ← one job ID per invocation
    ├── gemma4_e2b/
    │   ├── workspace -> ~/.limerick-benchmark/workspaces/20260416.143022/gemma4_e2b/
    │   │                 (symlink — generated code lives here, outside the repo)
    │   ├── run.sh            ← run this to start the server for manual eval
    │   ├── trace.jsonl       ← full agent message history (prompts, tool calls, outputs)
    │   ├── metrics.csv       ← timestamped system metrics
    │   └── summary.json      ← run stats (tokens, timing, eval result)
    └── qwen3.5_9b/
        └── …
```

Workspaces are kept outside the repo tree so that `uv init` inside them cannot auto-register as a parent workspace member in this project's `pyproject.toml`. They are pre-initialized with Python 3.12 and Flask before the agent starts, and are nested under the same job ID so dropping both the results and workspace side of a run is a two-command cleanup.

## metrics.csv columns

| Column | Source | Notes |
|---|---|---|
| `timestamp` | wall clock | ISO 8601 |
| `elapsed_s` | benchmark | seconds since run start |
| `cpu_percent` | psutil | all cores averaged |
| `memory_percent` | psutil | system RAM |
| `gpu_utilization_percent` | powermetrics | Apple GPU busy % |
| `gpu_power_mw` | powermetrics | GPU power draw in mW |
| `die_temp_c` | powermetrics | CPU/GPU die temperature |
| `fan_rpm` | powermetrics SMC | first fan, if present |
| `tokens_in` | LiteLLM | cumulative prompt tokens |
| `tokens_out` | LiteLLM | cumulative completion tokens |
| `api_calls` | benchmark | cumulative model API calls |
| `tool_calls` | benchmark | cumulative bash tool invocations |

GPU/thermal/fan columns are populated only when `--enable-hardware-metrics` is passed. That mode uses `powermetrics`, which requires `sudo`. To avoid repeated prompts, add to `/etc/sudoers` (via `visudo`):
```
yourusername ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
```

## Loop detection

Both agent backends abort early if they get stuck, so a stalled model doesn't burn the full 15-minute budget:

- **ReAct** — aborts on 5 consecutive identical commands, 5 consecutive redundant `uv init` attempts, 3+ consecutive rewrites of the same file, or 5 malformed / unknown tool calls. The `finish_reason` in `summary.json` records which guard tripped.
- **Aider** — aborts on low log-line uniqueness over a rolling window, a detectable repeating log cycle, any single file being edited more than `AIDER_MAX_EDITS_PER_FILE` times, or the workspace tree hash not changing for 300 seconds by default. Use `--aider-stagnation-timeout` to change that threshold. Aider stdout is prefixed with `[N/total:model-id:agent]` so multi-model runs stay grep-friendly.

When either guard trips, the run is recorded with `finish_reason: stuck_loop` and the post-run HTTP evaluation is skipped.

## Hardware

Benchmarks are run serially to avoid GPU contention. One model runs at a time. Results include thermal data (when `--enable-hardware-metrics` is on) so you can see if earlier runs affect later ones (thermal throttling).

Tested on: Apple M5 Max, 64 GB unified memory, macOS Sequoia.

## Agent-facing docs

`AGENTS.md` is the single source of coding guidelines for any agent working in this repo. `CLAUDE.md` and `GEMINI.md` are symlinks to it, so Claude Code, Codex, Gemini, and Aider all read the same file. Edit `AGENTS.md` directly — do not create per-agent variants.
