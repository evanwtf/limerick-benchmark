# Benchmark Analysis Report: Limerick Task (2026-04-17)

This report synthesizes findings from nine benchmark jobs conducted on April 17, 2026, evaluating various local LLMs using the **Aider** agent on a Python/Flask limerick generation task.

---

## 1. The "Small Model" Paradox
The most significant finding across all reports is that **smaller models (2B-4B) consistently outperformed larger models (26B-35B)** in both reliability and speed.

*   **Gemma 2B (e2b):** The undisputed champion of the benchmark. It consistently delivered working apps in **12–15 seconds**. 
    *   *Why it succeeds:* It adopts a "minimalist" approach, often hardcoding a static list of limericks and using simple `window.location.reload()`. It avoids the over-engineering traps that sink larger models.
*   **Larger Models (26B+):** Frequently hit **Stuck Loops** or **Workspace Stagnation**. They often spent their entire 180s-300s time budget designing elaborate phonetic rhyme dictionaries in their "thoughts" without ever writing a single file to disk.

## 2. Model Family Behavioral Profiles

### Gemma Family (Gemma 4)
*   **High Reliability:** The Gemma family (e2b, e4b, 26b) maintained the most stable pass rates.
*   **Consistency:** Unlike Qwen, if a Gemma model passed once, it likely passed in subsequent jobs.
*   **Runtime Variance:** MLX/BF16 variants were notably slower than GGUF equivalents (e.g., 46s vs 15s for the 2B model) while producing nearly identical code.

### Qwen Family (Qwen 3.5 / 3.6)
*   **The "Coding" Edge:** Base Qwen models were prone to "planning paralysis." However, the `35b-a3b-coding-mxfp8` variant was the standout exception, often delivering the highest-quality generative logic (randomized template filling) when it succeeded.
*   **Fragility:** Small Qwen models (9b) showed high variance, passing with a 68/100 in one run and failing with a log-cycle error the next.

## 3. Failure Mode Evolution
Analysis of the job sequence (07:30 to 17:36) shows shifting failure patterns:

1.  **Morning (Planning Failures):** Dominated by `stuck_loop` and `workspace_stagnation`. Models designed but did not build.
2.  **Mid-day (Environmental/Logic Errors):** A spike in `port_never_opened` failures suggests models were either crashing at startup or binding to incorrect ports (e.g., 5000 instead of 8181).
3.  **Evening (Content Failures):** Pass rates rose to 80%, but a new "Lazy" failure appeared: `body_missing_limerick`. Models built the server correctly but failed to populate it with the required 5-line poem.

## 4. Technical Insights for Benchmark Improvement

*   **Aider Agent Sensitivity:** A significant number of failures were triggered by `aider_edit_format_reject`. Local models frequently struggle to maintain the strict SEARCH/REPLACE block formatting required by Aider under pressure.
*   **Stochasticity:** The high run-to-run variance (especially in Qwen 9b and Gemma 26b) confirms that a single sample is insufficient for a "leaderboard" ranking.
*   **Evaluation Rigor:** The shift from "Server didn't start" to "Limerick missing from body" shows that the `evaluator.py` is effectively catching logic regressions even when the infrastructure (Flask) is correct.

---

**Summary Conclusion:** For high-speed, utility-grade task execution on Apple Silicon, **Gemma 4:e2b** is the most efficient choice. For complex, generative coding tasks where time is not a factor, **Qwen 3.5:35b-coding** provides the best architectural depth, provided it can be nudged past the planning phase.
