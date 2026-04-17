# limerick-benchmark

A benchmark suite for evaluating LLM coding ability on local Apple Silicon hardware. Each model gets a coding task, a `bash` tool, and a hard 15-minute timebox to produce a working result — no hints, no hand-holding.

## How it works

1. The runner starts a fresh agent session for each model, serially (never two at once, so GPU is never contested)
2. The agent receives the task prompt and can call a `bash` tool to create files, install packages, run the server, read output, and fix errors — just like a developer at a terminal
3. The run is **hard-killed after 15 minutes** if not complete
4. System metrics (CPU, memory, token counts, and optionally GPU/thermal/fan data) are sampled every 5 seconds throughout the run
5. After the run, you manually evaluate by running `./run.sh` in the result directory

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

## Prerequisites

- macOS with Apple Silicon (developed on M5 Max, 64 GB)
- [Ollama](https://ollama.ai) installed and running (`ollama serve`)
- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- `ANTHROPIC_API_KEY` set (for reference model runs)
- `sudo` access only if you choose `--enable-hardware-metrics`

## Quick start

```bash
# Install dependencies
uv sync

# See what models are locally available and how they map to the catalog
uv run benchmark list

# Prefetch models you don't have yet
uv run prefetch --set poc
uv run prefetch --set recommended --dry-run   # preview first

# Run the POC (2 smallest models — good for validating the harness)
uv run benchmark run --set poc

# Opt in to privileged Apple Silicon hardware metrics
uv run benchmark run --set poc --enable-hardware-metrics

# Run only models already in your local Ollama store
uv run benchmark run --set local

# Run the full recommended set (top 10 local + 3 Anthropic reference)
uv run benchmark run --set recommended

# Run a single model
uv run benchmark run --model gemma4:e2b

# Run Anthropic reference models only
uv run benchmark run --set reference

# Skip models that aren't pulled instead of aborting
uv run benchmark run --set recommended --skip-missing
```

After each run, open the result directory and run `./run.sh` to start the generated server, then browse to http://localhost:8181 to evaluate manually.

## Model sets

### Reference models (Anthropic cloud)

| Model ID | Label | Role |
|---|---|---|
| `claude-opus-4-7` | Claude Opus 4.7 | Reference — 100 pts |
| `claude-sonnet-4-6` | Claude Sonnet 4.6 | Reference — better |
| `claude-haiku-4-5-20251001` | Claude Haiku 4.5 | Reference — good |

### POC models

For initial testing of the harness, use these two — smallest in the recommended set, complete quickly:

| Model ID | Size | Why |
|---|---|---|
| `qwen3.5:9b` | 6.6 GB | Smallest Qwen with real coding ability |
| `gemma4:e2b` | 7.2 GB | Smallest Gemma 4 IT variant |

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

```
results/
└── 20260416_143022_gemma4_e2b/
    ├── workspace -> ~/.limerick-benchmark/workspaces/20260416_143022_gemma4_e2b/
    │                 (symlink — generated code lives here, outside the repo)
    ├── run.sh            ← run this to start the server for manual eval
    ├── trace.jsonl       ← full agent message history (prompts, tool calls, outputs)
    ├── metrics.csv       ← timestamped system metrics
    └── summary.json      ← run stats (tokens, timing, eval result)
```

Workspaces are kept outside the repo tree so that `uv init` inside them cannot auto-register as a parent workspace member in this project's `pyproject.toml`.

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

## Hardware

Benchmarks are run serially to avoid GPU contention. One model runs at a time. Results include thermal data so you can see if earlier runs affect later ones (thermal throttling).

Tested on: Apple M5 Max, 64 GB unified memory, macOS Sequoia.
