# Comparison: Jobs `20260417.232608`, `20260417.233306`, `20260417.233950`

This note compares:

- `reports/results_20260417.232608.md`
- `reports/results_20260417.233306.md`
- `reports/results_20260417.233950.md`
- their associated artifacts under `results/20260417.232608/`, `results/20260417.233306/`, and `results/20260417.233950/`

## Executive Summary

The strongest overall run is `20260417.233950`. It reached `8/10` passes, improved on the earlier `6/10` and `7/10` runs, and was the only run where `gemma4:31b` converted from a loop failure into a real pass.

The weakest run is `20260417.232608`. It had the lowest pass rate at `6/10`, although it also had the fastest average successful completions. Across the three runs, reliability improved, but average successful wall time increased as more marginal models finished.

## Run Ranking

1. `20260417.233950`: best pass rate (`8/10`), only two failures, and no loop termination.
2. `20260417.233306`: middle result (`7/10`), still blocked by the same two chronic failures plus `gemma4:31b` looping.
3. `20260417.232608`: lowest pass rate (`6/10`) and the broadest failure spread.

## Model Winners

### Clear winner

`qwen3.5:35b-a3b-coding-mxfp8` is the benchmark leader.

- `3/3` passes
- fastest pass in every run
- average successful wall time: `15.5s`
- no warnings or unstable behavior observed

### Strong runner-up

`qwen3.5:35b-a3b` is the safest non-coding variant.

- `3/3` passes
- average successful wall time: `20.4s`
- lower memory usage than several other reliable passers

### Reliable but less compelling

`gemma4:e4b` passed `3/3`, but got slower across the runs and regularly pushed memory into the mid/high `90%` range.

`qwen3.6:35b-a3b-coding-mxfp8` also passed `3/3`, but it was materially slower than both qwen3.5 variants without showing better robustness.

`gemma4:26b` passed `3/3`, but it was the slowest consistent passer by a large margin, averaging `48.0s`.

## Mixed Results

`qwen3-coder-next:latest` passed `3/3`, but every run carried the same `aider_edit_format_reject` warning. It is effective, but the formatting behavior looks brittle rather than cleanly aligned with the Aider workflow.

`glm-4.7-flash:latest` improved meaningfully. It failed in `20260417.232608`, then passed in both later runs. It also had the lightest memory profile of the field. I would treat it as an improver, not a top-tier winner.

`gemma4:31b` is the clearest flaky model. It failed twice with `repeating_log_cycle`, then passed once in `20260417.233950`. The traces show prompt-format fixation rather than a normal implementation mistake, so this looks unstable rather than solved.

## Model Losers

`gemma4:e2b` is a consistent loser.

- `0/3` passes
- same failure every time: `eval_body_missing_limerick`
- generated pages refreshed correctly and returned HTTP 200, but the app read `limericks.txt` line-by-line instead of as blank-line-separated poems, so the evaluator never saw a valid limerick shape

`ShreyanGondaliya/gemma-4-claude-opus-4.6-thinking-s7-multimodal:latest` is the worst model in this set.

- `0/3` passes
- same failure every time: `eval_no_entry_point`
- `0` output tokens in every run
- never produced `app.py` or any runnable entry point

## Bottom Line

If you want one model to represent the current winner, use `qwen3.5:35b-a3b-coding-mxfp8`.

If you want a conservative shortlist, use:

- `qwen3.5:35b-a3b-coding-mxfp8`
- `qwen3.5:35b-a3b`
- `gemma4:e4b`

If you want models to avoid for now, use:

- `gemma4:e2b`
- `ShreyanGondaliya/gemma-4-claude-opus-4.6-thinking-s7-multimodal:latest`

`gemma4:31b` should stay on probation until it passes repeatedly rather than once.
