# limerick-benchmark

**How good is your Mac at running LLM coding agents?**
A reproducible, single-command benchmark for local coding models on Apple
Silicon. Each model gets one real coding task, one tool (a shell), and a
hard 15-minute clock to produce a working Flask app. Pass/fail is verified
by an actual HTTP request, not a vibe check.

```bash
uv sync
uv run benchmark run --set recommended
```

That's it. The harness spins up each model in turn, runs it against the
task in an isolated workspace, watches for infinite loops, boots the app
the model generated, `curl`s it, and writes a Markdown scoreboard to
`reports/results_<job_id>.md`. Past reports are checked into this repo
(`reports/*.md`) so you can see exactly what you'd be signing up for.

---

## Why this exists

Most "local LLM leaderboards" measure perplexity on a frozen test set, or
pass rates on toy one-line puzzles. That doesn't tell you whether a model
can actually **build a thing** on your hardware in a reasonable amount of
time. This project does the dumb-but-honest version of that:

- One real, end-to-end coding task per run (currently: a Flask web app).
- Two agent harnesses — a minimal **ReAct** loop (`litellm` + a `bash`
  tool) and the real **Aider** CLI — so you can compare how a model
  behaves under different scaffolding.
- **Hardware-aware metrics**: CPU, memory, token counts, and optionally
  GPU utilisation, GPU power (mW), die temp, and fan RPM from Apple's
  `powermetrics`. See if your M-series chip is throttling before you
  publish a "Qwen is slow" take.
- **Loop detection** on both backends, so a stuck model gets killed in
  seconds instead of eating the full 15-minute timeout.
- **Deterministic file layout**: one job ID per run, all artefacts
  collated under it, cleanup is a single `rm -rf`.
- **No network sandbox tricks** — generated code runs on your box. Only
  run this in the workspaces the harness creates (it already keeps them
  outside the repo tree so `uv init` can't climb back in).

If you want to contribute a task, a model, or a different agent harness,
see [Contributing](#contributing) below — we genuinely want PRs.

## Quick start

```bash
# 1. Install
uv sync

# 2. See what Ollama models you already have (vs. the catalog)
uv run benchmark list

# 3. Pull the recommended benchmark set (or a subset)
uv run prefetch --set recommended --dry-run   # preview downloads
uv run prefetch --set recommended             # actually pull

# 4. Smallest possible sanity run (~30s)
uv run benchmark run --set poc

# 5. Full recommended set — 10 local models
uv run benchmark run --set recommended

# 6. Open the auto-generated report
open reports/results_$(ls -1 reports/ | tail -1)
```

Each completed run writes `reports/results_<job_id>.md` automatically.
Open a finished result directory and run `./run.sh` to serve the
model-generated app on <http://localhost:8181> for manual inspection.

## What you'll see

Sample headline row from a recent `--set recommended` run:

| Model | Pass | Wall time | Notes |
|---|---|---|---|
| `qwen3.5:35b-a3b-coding-mxfp8` | ✅ | 12.5 s | Real rhyme-group generator. |
| `gemma4:e4b` | ✅ | 16.8 s | Structured tuples, clean HTML. |
| `gemma4:e2b` | ✅ | 13.0 s | Tiny, static-list, fastest. |
| `qwen3.6:latest` | ❌ | 180.3 s | Infinite planning loop, wrote nothing. |
| `gemma4:26b` | ❌ | 70.5 s | Analysis paralysis, then loop-guard fired. |

Full detailed example: [`reports/results_20260417_073034.md`](reports/results_20260417_073034.md). A cross-run
meta-analysis across 10 jobs from one day is in
[`claude_report.md`](claude_report.md).

**Stable findings so far:**
- Smaller models finish more often than larger ones on this task — the
  2B–4B Gemmas ship static-list apps in 12–30 s; base 26B–35B models
  routinely burn the full time budget "planning" rhyme dictionaries they
  never write to disk.
- Run-to-run variance is large. One sample per model is not a
  leaderboard. Plan for `n ≥ 3`.
- Aider's SEARCH/REPLACE diff format is a real failure surface for local
  models — a model that fails under Aider may pass under the ReAct
  harness and vice versa.

## How it works

1. **One job per invocation.** Each `benchmark run` gets a single job ID
   (`YYYYMMDD.HHMMSS`); every per-model run is collated under it.
   Results live in `results/<job_id>/<slug>/`, workspaces in
   `~/.limerick-benchmark/workspaces/<job_id>/<slug>/` with a symlink
   back. Cleanup is `rm -rf results/<job_id>` plus the matching
   workspace tree.
2. **Serial runs, no GPU contention.** One model at a time so the numbers
   are comparable.
3. **Workspace prep.** Workspaces are pre-seeded with task data files
   (for the limerick task, `tasks/limericks.txt` gets copied in).
   - **ReAct (default):** the workspace is **not** pre-initialised —
     running `uv init` and installing Flask is part of what the model is
     being graded on.
   - **Aider (`--agent aider`):** the workspace **is** pre-bootstrapped
     as a `uv` project with Flask installed, because Aider can't run
     setup commands. The task prompt has an environment note that
     reflects whichever mode you picked.
4. **One of two agent backends drives the run.**
   - **ReAct** — `litellm` + a single `bash` tool (60s per-command
     timeout). Loop guards abort on 5 identical commands in a row,
     repeated `uv init`, repeated whole-file rewrites, or 5 malformed
     tool calls.
   - **Aider** — the real Aider CLI in headless mode, plus a harness
     wrapper that watches for low log-line uniqueness, repeating log
     cycles, per-file edit caps, and workspace-hash stagnation (kills
     the run if the file tree hasn't changed for 300 s by default —
     override with `--aider-stagnation-timeout`).
5. **Hard 15-minute clock** on the whole run (`--timeout` to change).
6. **Metrics sampled every 5 seconds.** CPU / memory / tokens always;
   GPU utilisation / power / die temp / fan RPM if you pass
   `--enable-hardware-metrics` (needs `sudo` for `powermetrics`).
7. **Automated evaluator.** Requires canonical entry point `app.py`,
   runs `uv run python app.py`, expects HTTP 200 on port 8181, checks
   the response body for a 5-line limerick shape **and** either a
   refresh `<meta>` tag or a `setInterval(` call. Writes a `run.sh`
   into the result dir so you can re-boot the server manually.
8. **Auto-report.** At the end of the job, a Markdown summary lands in
   `reports/results_<job_id>.md`.

## The current task: "Limerick Generator"

See [`tasks/limerick.md`](tasks/limerick.md) for the full prompt.

> Build a Python web app using Flask and `uv` that shows a limerick in
> the browser, rotates to a new one every 5 seconds, and listens on
> port 8181.

Tasks are just Markdown files in `tasks/`. Adding a new one is as simple
as dropping a new `.md` in there and (if needed) a corresponding data
seed and an evaluator branch. PRs welcome — see [Contributing](#contributing).

## Scoring

The first two checks are automated gates: if the server doesn't start,
there is nothing to evaluate.

| Check | Type | Points |
|---|---|---|
| Server starts without error | Automated gate | — |
| `GET /` returns HTTP 200 | Automated gate | — |
| Page displays a recognisable 5-line limerick | Human | 0–40 |
| Auto-refreshes every ~5 s (meta refresh or `setInterval`) | Human | 0–30 |
| Code quality / approach (uv usage, structure, etc.) | Human | 0–30 |

**Reference**: Claude Opus 4.7 = 100 points. All other scores are
relative to that.

## Prerequisites

- macOS with Apple Silicon (developed on M5 Max, 64 GB). Tested on
  macOS Sequoia.
- [Ollama](https://ollama.ai) installed and running (`ollama serve`).
- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).
- `ANTHROPIC_API_KEY` **only** if you want to run the Anthropic
  reference models (`--set reference`).
- `sudo` access **only** if you pass `--enable-hardware-metrics`.

Nothing else. No Docker, no cloud, no sign-up.

## Every CLI flag you'll actually use

### `uv run benchmark run`

| Flag | Default | Notes |
|---|---|---|
| `--set {poc,v1,recommended,local,reference}` | — | Named model set. `local` = whatever is already in your Ollama store; `reference` = Anthropic cloud. Mutually exclusive with `--model`. |
| `--model MODEL_ID […]` | — | One or more explicit model IDs. Unknown IDs are treated as Ollama models. |
| `--task NAME` | `limerick` | Task file name (without `.md`) in `tasks/`. |
| `--timeout SECONDS` | 900 | Per-model hard limit. |
| `--agent {react,aider}` | `react` | Agent backend. |
| `--aider-stagnation-timeout SECONDS` | 300 | Abort an Aider run if the workspace tree stays unchanged this long. |
| `--skip-missing` | off | Skip Ollama models that aren't pulled, instead of aborting the run plan. |
| `--enable-hardware-metrics` | off | Collect GPU / thermal / fan metrics via `powermetrics` (prompts for `sudo`). |

### `uv run benchmark list` / `report`

```bash
uv run benchmark list                           # local models vs. catalog
uv run benchmark report --job-id 20260417.083818
uv run benchmark report --job-id <id> --output reports/mine.md
```

### `uv run prefetch`

| Flag | Default | Notes |
|---|---|---|
| `--set {poc,v1,recommended,all}` | — | Named model set. |
| `--model MODEL_ID […]` | — | Specific model IDs to pull. |
| `--dry-run` | off | Show the plan without downloading. |
| `--yes`, `-y` | off | Skip the confirmation prompt. |

## Result directory layout

```
results/
└── 20260417.193159/                         ← one job ID per invocation
    ├── job.json                             ← what was requested
    ├── gemma4_e2b/
    │   ├── workspace -> ~/.limerick-benchmark/workspaces/…   ← generated code
    │   ├── run.sh          ← boot the generated server manually
    │   ├── summary.json    ← tokens, timing, evaluator result, failure class
    │   ├── metrics.csv     ← 5-second samples throughout the run
    │   └── trace.jsonl     ← full agent message history
    └── qwen3.5_9b/
        └── …
```

Workspaces live **outside** the repo tree at
`~/.limerick-benchmark/workspaces/` so `uv init` inside them can't walk
up and auto-register as a member of this project's `pyproject.toml`.
Keep it that way.

`metrics.csv` columns:

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

GPU/thermal/fan columns are populated only when
`--enable-hardware-metrics` is on. To skip repeated `sudo` prompts, add
to `/etc/sudoers` (via `visudo`):

```
yourusername ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
```

## Loop detection

Both backends abort early if they're stuck, so a bad run costs seconds,
not the full 15-minute budget:

- **ReAct** — aborts on 5 consecutive identical commands, 5 consecutive
  redundant `uv init` attempts, 3+ rewrites of the same file in a row,
  or 5 malformed / unknown tool calls. `summary.json` records which
  guard tripped.
- **Aider** — aborts on low log-line uniqueness over a rolling window,
  any detectable repeating log cycle, any single file edited more than
  `AIDER_MAX_EDITS_PER_FILE` times, or the workspace tree hash not
  changing for 300 s by default. Override with
  `--aider-stagnation-timeout`. Aider stdout is prefixed with
  `[N/total:model-id:agent]` so multi-model runs stay grep-friendly.

A stuck run is recorded with `finish_reason: stuck_loop` and the HTTP
evaluation is skipped.

## Model sets

### Reference (Anthropic cloud)

| Model ID | Role |
|---|---|
| `claude-opus-4-7` | Reference — 100 pts |
| `claude-sonnet-4-6` | Reference — better |
| `claude-haiku-4-5-20251001` | Reference — good |

### Recommended local set (top 10)

Chosen to answer specific questions: does size help? does a coding
fine-tune beat a larger general model? does Apple MLX runtime differ
from GGUF? does Qwen 3.6 beat 3.5 at the same weight class?

| # | Model ID | Size | Family | Question |
|---|---|---|---|---|
| 1 | `qwen3.5:9b` | 6.6 GB | Qwen 3.5 | Small/fast Qwen baseline |
| 2 | `gemma4:e2b` | 7.2 GB | Gemma 4 | Small/fast Gemma baseline |
| 3 | `gemma4:e4b` | 9.6 GB | Gemma 4 | One size up from e2b |
| 4 | `gemma4:e2b-mlx-bf16` | 10 GB | Gemma 4 MLX | MLX vs. GGUF runtime |
| 5 | `qwen3.5:27b-coding-mxfp8` | 31 GB | Qwen 3.5 Coder | Coding fine-tune vs. bigger general? |
| 6 | `qwen3.5:35b-a3b` | 24 GB | Qwen 3.5 | Large MoE baseline |
| 7 | `gemma4:26b` | 18 GB | Gemma 4 | Quality jump from small Gemmas? |
| 8 | `qwen3.5:35b-a3b-coding-mxfp8` | 38 GB | Qwen 3.5 Coder | Coding fine-tune at 35B |
| 9 | `qwen3.6:latest` | 24 GB | Qwen 3.6 | Same size as #6 — does 3.6 matter? |
| 10 | `qwen3.6:35b-a3b-q8_0` | 39 GB | Qwen 3.6 | Gen 3.6 at higher precision |

Full catalog with every variant and exclusion notes: [`models.yaml`](models.yaml).

## Contributing

PRs are very welcome. The low-friction ways to contribute:

- **Run the benchmark on your hardware and open a PR with the result
  Markdown.** Different Apple Silicon generations (M1/M2/M3/M4/M5 and
  their Pro/Max/Ultra variants) produce different numbers; real data
  points are the most useful thing this repo can collect.
- **Propose a new task.** Drop a new `tasks/<name>.md` and a
  corresponding evaluator branch. Good candidates: "build a CLI that
  parses a CSV," "implement a job queue with Redis," "generate an
  interactive data viz." Tasks should be small enough to finish in 15
  minutes but have a verifiable artefact.
- **Add a model family.** Edit `models.yaml` — mark anything that won't
  fit in 64 GB unified memory or that's NVIDIA-only (FP4, etc.) with
  `exclude: true` and a reason.
- **Add a third agent backend.** Current backends live in
  `benchmark/agent.py` (`_run_react` and `_run_aider`). A Codex / Aider
  architect mode / your-own-agent plug-in would be a great PR.
- **Harden the evaluator.** See the "Recommendations" section of
  [`claude_report.md`](claude_report.md) for a concrete list of known
  gaps (per-job harness pinning, `n ≥ 3` sampling, partial credit for
  `body_missing_limerick`, …).

Agent-facing repo conventions live in [`AGENTS.md`](AGENTS.md).
`CLAUDE.md` and `GEMINI.md` are symlinks to it — edit `AGENTS.md`
directly so Claude Code, Codex, Gemini, and Aider all read the same
guidance. Run the test suite before submitting:

```bash
uv run python -m unittest discover tests
```

## Safety notes

- The ReAct loop executes arbitrary shell commands from model output.
  The harness runs it only inside workspaces under
  `~/.limerick-benchmark/workspaces/` (outside the repo). **Do not**
  run the benchmark in the repo root, and **do not** point it at a
  workspace you don't want a model writing to.
- The Aider backend runs a subprocess without the bash-tool shim, but
  still operates on the workspace directory, so the same rule applies.
- Generated workspaces and `results/` artefacts are gitignored on
  purpose (with a `.gitkeep` passthrough in `results/`). Published
  score writeups (`results_YYYYMMDD_HHMMSS.md`, `claude_report.md`,
  etc.) are intentionally tracked.

## Acknowledgements

Built on top of [Ollama](https://ollama.ai), [Aider](https://aider.chat),
[LiteLLM](https://github.com/BerriAI/litellm),
[uv](https://docs.astral.sh/uv/), and a lot of Apple Silicon heat.
