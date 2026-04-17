# Claude's Cross-Run Analysis — Limerick Benchmark (2026-04-17)

Synthesis of all 10 jobs under `reports/` from a single day (07:30 → 19:31).
Every run used the Aider backend against the same 10-model catalog and the
same limerick task. The goal of this write-up is to separate **model
behaviour** from **harness drift**, and to extract findings that are stable
across jobs rather than chasing any single noisy sample.

## Timeline at a glance

| Job ID | Time | Pass rate | Dominant failure mode |
|---|---|---|---|
| `20260417_073034` | 07:30 | 5/10 | stuck_loop (Qwen planning paralysis, 180 s stagnation) |
| `20260417.083818` | 08:38 | 4/10 | stuck_loop (log cycle + stagnation) + 1 port_never_opened |
| `20260417.110017` | 11:00 | 4/10 | stuck_loop (6), port_never_opened (1) |
| `20260417.134803` | 13:48 | 4/10 | repeating_log_cycle (3), workspace_stagnation (2) |
| `20260417.145044` | 14:50 | **0/10** | port_never_opened (7), no_entry_point (3) — harness regression |
| `20260417.155108` | 15:51 | 7/10 | no_entry_point (3, large Qwens) |
| `20260417.165047` | 16:50 | 4/10 | aider_edit_format_reject (4), entry_point_mismatch (1) |
| `20260417.172042` | 17:20 | 8/10 | body_missing_limerick (2) |
| `20260417.173614` | 17:36 | 7/10 | body_missing_limerick (3) |
| `20260417.193159` | 19:31 | **8/10** | body_missing_limerick (2) |

**Net trajectory:** pass rates drift upward across the day (5 → 4 → 4 → 4 →
**0** → 7 → 4 → 8 → 7 → **8**). The pattern is not monotonic model
improvement — it's the harness getting stricter while, in parallel, becoming
more forgiving of the long-tail failure shapes that earlier killed large
Qwen models.

## Harness evolution, read from the data

The failure-category vocabulary changes mid-day in ways that clearly
correspond to code changes rather than model behaviour:

1. **07:30 → 13:48 — the "stuck_loop" era.** Aider's own repeating-log-cycle
   guard and workspace-stagnation watchdog dominate failure reasons. Large
   Qwens (27b/35b, 3.6:latest) routinely burn 180 s or 300 s without writing a
   file. Gemma models mostly pass.
2. **14:50 — the collapse.** Every single run "completes" but nothing binds
   port 8181 or exposes a detectable entry point. 7 `port_never_opened` + 3
   `no_entry_point` across 10 models — including `gemma4:e2b`, which had
   passed in every prior job — is not a model-behaviour story. Most likely
   cause: a harness change that stopped pre-initialising the workspace as a
   `uv` project with Flask installed (matching the commit
   `Stop pre-installing deps; make project setup part of the task`), which
   aider can't recover from on its own.
3. **15:51 → 16:50 — stricter evaluator.** `entry_point_mismatch` and
   `aider_edit_format_reject` appear for the first time. `entry_point_mismatch`
   lines up with the evaluator tightening to require the canonical `app.py`
   (AGENTS.md confirms this). The 16:50 job is the worst since 14:50
   specifically because four models wrote valid apps under the wrong filename
   and/or broke aider's SEARCH/REPLACE format under the new constraints.
4. **17:20 → 19:31 — steady state.** Once the entry-point requirement is
   honored (and, presumably, the prompt re-tuned to match), pass rates settle
   at 7–8 / 10. The residual failures are all `body_missing_limerick`: the
   server runs, but the response body doesn't contain five distinct lines the
   evaluator will accept.

**Takeaway:** four of the ten jobs are measuring the benchmark itself, not
the models. A leaderboard that averages them uncritically will be dominated
by the 14:50 and 16:50 harness shocks. Any published score needs to either
pin the harness version or average only across jobs with a consistent
failure-category vocabulary.

## Per-model behavioural profile (stable across jobs)

| Model | Pass / 10 | Typical time | Stable behaviour |
|---|---|---|---|
| `gemma4:e2b` | 8/10 | 11–16 s | Fastest; static-list strategy. Fails via `body_missing_limerick` when the refresh JS renders client-side and the initial HTML ships empty. |
| `gemma4:e4b` | 9/10 | 16–29 s | Most reliable model in the set. Slight quality bump over e2b (structured tuples, basic styling). |
| `gemma4:e2b-mlx-bf16` | 6/10 | 18–47 s | Produces code nearly identical to base `e2b` at ~2–3× the wall-clock. MLX runtime loss on this task is real. |
| `gemma4:26b` | 7/10 | 22–55 s | High-variance: either analysis-paralysis at 70 s or the job's best structured implementation. No middle ground. |
| `qwen3.5:9b` | 5/10 | 21–95 s | Fragile. Same model produces a passing fetch/setInterval app in one job and a repeating-log-cycle failure in the next. |
| `qwen3.5:27b-coding-mxfp8` | 3/10 | 42–300 s | "Coding" tag doesn't rescue the base-model planning paralysis in the first half of the day; recovers once the prompt/env change lands. |
| `qwen3.5:35b-a3b` | 4/10 | 21–180 s | Same dual-mode shape as the 9b — either ~20 s or stuck_loop. |
| `qwen3.5:35b-a3b-coding-mxfp8` | 7/10 | 12–180 s | Best-in-class *when* it ships — highest code quality of any passing run (real rhyme-group generator vs. static lists). |
| `qwen3.6:latest` | 3/10 | 46–300 s | Worst cost/value: burns the full stagnation window planning phonetic dictionaries. |
| `qwen3.6:35b-a3b-q8_0` | 3/10 | 66–300 s | Same pattern as `qwen3.6:latest`; Q8 precision doesn't change the behaviour. |

## Stable findings (robust across the harness changes)

1. **Size is not monotonic with success.** The 2B–9B models clear the bar
   faster and more often than the 26B–35B base variants. The pattern replays
   across every job: the smallest Gemma is the fastest passer; the base-Qwen
   26B–35B variants are the least reliable.
2. **Coding fine-tunes flip the large-Qwen story.** `qwen3.5:35b-a3b-coding-mxfp8`
   is the only large local model that produces a real generative
   implementation (rhyme-sound dictionaries + AABBA templates) and does so
   in under 80 s. Whenever it passes, it scores highest.
3. **Aider's SEARCH/REPLACE format is a real failure surface for local
   models.** The 16:50 job produced four `aider_edit_format_reject` failures
   without a single underlying bug in the generated code — the models lost
   their grip on the diff framing. A ReAct-only run of the same models would
   likely show a different failure distribution.
4. **Run-to-run variance is higher than the gap between most adjacent
   models.** `qwen3.5:9b` has passing times ranging from 21 s to 95 s within
   one day. No single-sample ranking survives a second run. The benchmark
   needs `n ≥ 3` samples per model per harness version to be trustworthy.
5. **`body_missing_limerick` is the new floor.** Once the harness stabilised,
   the only recurring failure is evaluator-rejected HTML. Spot-checking the
   generated `app.py` in `results/20260417.193159/gemma4_e2b/workspace/`
   shows the actual cause: the model reads `limericks.txt` as one limerick
   per **line** and renders a single random line, while the seeded file
   actually stores 5-line limericks separated by **blank lines** (so the
   correct split is on `\n\n`). The passing `qwen3.5:35b-a3b-coding-mxfp8`
   implementation in the same job splits on `\n\n` and is rewarded for it.
   This is a real task-comprehension miss by the small Gemmas — not a
   tooling glitch — and suggests the task prompt should call out the file
   format explicitly, or the seed file should be replaced with one the
   naive "one per line" read still resolves to a valid limerick.

## Notes from the workspace artefacts (`results/`)

- `results/` contains **10 job directories**, one per reported run, plus an
  **orphan `20260417.164817/`** that never made it to `reports/`. It only
  has two per-model dirs (`gemma4_e2b`, `gemma4_e2b-mlx-bf16`) and a
  `job.json` — almost certainly an interrupted / killed run. It's not
  reflected in the published timeline.
- Each job dir has the structure AGENTS.md advertises: `job.json`,
  per-model subdirs with `summary.json`, `metrics.csv`, `trace.jsonl`,
  `run.sh`, and a `workspace` symlink out to
  `~/.limerick-benchmark/workspaces/<job_id>/<slug>/`. The symlinks in
  older jobs still resolve, which means workspaces aren't being cleaned up
  — a `du -sh ~/.limerick-benchmark/workspaces/` check is worth adding to
  the housekeeping docs.
- `summary.json` is rich: it records `failure_category` separately from
  `eval.error` and `finish_reason`, which is what made the cross-run
  failure-mode comparison in this report possible at all. Keep that
  structure — it's the ground truth.
- Token counts are zero on local runs in the 11:00 and 13:48 jobs but
  non-zero from 17:20 onward (e.g. `tokens_in: 1400, tokens_out: 871` for
  `gemma4:e2b` in the 19:31 job). That shift looks like LiteLLM token
  accounting getting wired up for the Ollama path mid-day. Before drawing
  any "tokens per pass" conclusions, filter to jobs where
  `tokens_out > 0` — otherwise early-run rows will pull the averages to
  zero.

## Recommendations for the harness

- **Pin the evaluator version in the job artefacts.** A harness-change
  audit trail would make 14:50-style cliffs obvious rather than
  indistinguishable from model-family collapses.
- **Run `n ≥ 3` per model per job** (or introduce a `--samples N` flag) so
  the dominant signal is not whichever side of the variance a model landed
  on.
- **Split the leaderboard by agent backend.** The
  `aider_edit_format_reject` failures are not telling us much about coding
  ability — they're telling us which models are fragile under Aider's
  strict diff protocol. ReAct numbers on the same catalog would be a
  valuable second column.
- **Treat `body_missing_limerick` as a scoring class, not a failure.** It's
  evidence the server works but the content is wrong; that's useful signal
  for a partial-credit score rather than a zero.
- **Consider a "second task" to break the limerick-static-list ceiling.**
  Every passing Gemma run ships the same static-list implementation. A
  task that actively requires generation (e.g. "generate a limerick about
  the user-supplied topic") would better separate the models whose current
  90-point scores are rewarding pattern-matched scaffolding.
