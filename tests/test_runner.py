import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from benchmark.runner import (
    RUN_ORDER_CHOICES,
    RESULTS_ROOT,
    _build_run_plan,
    _first_meaningful_edit_seconds,
    _new_job_id,
    _normalize_agent_stats_for_eval,
    _prepare_workspace,
    _run_dir,
    _run_dir_name,
    _run_one,
    _should_evaluate,
    _slug,
    _task_prompt_with_workspace_note,
    run_benchmark,
)


class RunnerEvaluationPolicyTests(unittest.TestCase):
    def test_skips_evaluation_for_redundant_setup_loop(self) -> None:
        self.assertFalse(_should_evaluate({"finish_reason": "redundant_uv_init_loop", "error": None}))

    def test_skips_evaluation_for_invalid_tool_loop(self) -> None:
        self.assertFalse(_should_evaluate({"finish_reason": "invalid_tool_loop", "error": None}))

    def test_skips_evaluation_for_repeated_command_loop(self) -> None:
        self.assertFalse(_should_evaluate({"finish_reason": "repeated_command_loop", "error": None}))

    def test_skips_evaluation_for_repeated_file_write_loop(self) -> None:
        self.assertFalse(_should_evaluate({"finish_reason": "repeated_file_write_loop", "error": None}))

    def test_skips_evaluation_when_agent_errors(self) -> None:
        self.assertFalse(_should_evaluate({"finish_reason": "error", "error": "boom"}))

    def test_keeps_evaluation_for_timeout(self) -> None:
        self.assertTrue(_should_evaluate({"finish_reason": "timeout", "error": None}))

    def test_skips_evaluation_for_stuck_loop(self) -> None:
        self.assertFalse(_should_evaluate({"finish_reason": "stuck_loop", "error": None}))

    def test_normalizes_passing_aider_reject_into_warning(self) -> None:
        normalized = _normalize_agent_stats_for_eval(
            {
                "finish_reason": "aider_edit_format_reject",
                "agent_stop": {
                    "category": "aider_edit_format_reject",
                    "detail": "No filename provided before file listing",
                },
                "timed_out": False,
                "error": None,
            },
            {"passed": True},
        )

        self.assertEqual(normalized["finish_reason"], "completed")
        self.assertIsNone(normalized["agent_stop"])
        self.assertEqual(
            normalized["agent_warning"],
            {
                "category": "aider_edit_format_reject",
                "detail": "No filename provided before file listing",
            },
        )


class RunnerWorkspacePreparationTests(unittest.TestCase):
    def test_prepare_workspace_is_a_noop_for_react_without_task_resources(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _prepare_workspace(workspace)
            self.assertEqual(list(workspace.iterdir()), [])

    def test_prepare_workspace_seeds_limericks_file(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            tasks_dir = root / "tasks"
            tasks_dir.mkdir()
            (tasks_dir / "limericks.txt").write_text("seed\n")

            with mock.patch("benchmark.runner.TASKS_DIR", tasks_dir):
                _prepare_workspace(workspace, task_name="limerick")

            self.assertEqual((workspace / "limericks.txt").read_text(), "seed\n")

    def test_prepare_workspace_bootstraps_uv_project_for_aider(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            with mock.patch("benchmark.runner.subprocess.run") as run_mock:
                _prepare_workspace(workspace, agent_type="aider")

        self.assertEqual(run_mock.call_count, 2)
        self.assertEqual(run_mock.call_args_list[0].args[0][:3], ["uv", "init", "."])
        self.assertEqual(run_mock.call_args_list[1].args[0], ["uv", "add", "flask"])

    def test_prepare_workspace_skips_bootstrap_for_react(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            with mock.patch("benchmark.runner.subprocess.run") as run_mock:
                _prepare_workspace(workspace, agent_type="react")
            run_mock.assert_not_called()

    def test_task_prompt_for_react_agent(self) -> None:
        prompt = _task_prompt_with_workspace_note(
            "Build the app.", task_name="limerick", agent_type="react"
        )
        self.assertIn("Setting up the project", prompt)
        self.assertIn("`app.py`", prompt)
        self.assertIn("limericks.txt", prompt)
        self.assertTrue(prompt.endswith("Build the app."))

    def test_task_prompt_for_aider_agent(self) -> None:
        prompt = _task_prompt_with_workspace_note(
            "Build the app.", task_name="limerick", agent_type="aider"
        )
        self.assertIn("Flask installed", prompt)
        self.assertIn("Do not run `uv init`", prompt)
        self.assertIn("limericks.txt", prompt)

    def test_run_dir_nests_model_under_job_id(self) -> None:
        job_id = "20260417.073034"
        run_dir = _run_dir(job_id, _run_dir_name("gemma4:e2b", run_index=1, total_runs=1, round_index=1, position_in_round=1))
        self.assertEqual(run_dir, RESULTS_ROOT / job_id / _slug("gemma4:e2b"))
        self.assertEqual(run_dir.parent.name, job_id)
        self.assertNotIn(":", run_dir.name)

    def test_new_job_id_matches_expected_shape(self) -> None:
        job_id = _new_job_id()
        date_part, _, time_part = job_id.partition(".")
        self.assertEqual(len(date_part), 8)
        self.assertEqual(len(time_part), 6)
        self.assertTrue(date_part.isdigit() and time_part.isdigit())


class RunnerPlanTests(unittest.TestCase):
    def test_run_order_choices_cover_expected_values(self) -> None:
        self.assertEqual(RUN_ORDER_CHOICES, ("balanced", "random", "fixed"))

    def test_build_run_plan_balanced_rotates_each_round(self) -> None:
        models = [
            {"id": "gemma4:e4b"},
            {"id": "qwen3.5:35b-a3b-coding-mxfp8"},
            {"id": "qwen3.6:35b-a3b-coding-mxfp8"},
        ]

        plan = _build_run_plan(models, rounds=3, order="balanced", seed=None)

        self.assertEqual(
            [(entry["round_index"], entry["position_in_round"], entry["model"]["id"]) for entry in plan],
            [
                (1, 1, "gemma4:e4b"),
                (1, 2, "qwen3.5:35b-a3b-coding-mxfp8"),
                (1, 3, "qwen3.6:35b-a3b-coding-mxfp8"),
                (2, 1, "qwen3.5:35b-a3b-coding-mxfp8"),
                (2, 2, "qwen3.6:35b-a3b-coding-mxfp8"),
                (2, 3, "gemma4:e4b"),
                (3, 1, "qwen3.6:35b-a3b-coding-mxfp8"),
                (3, 2, "gemma4:e4b"),
                (3, 3, "qwen3.5:35b-a3b-coding-mxfp8"),
            ],
        )

    def test_build_run_plan_random_is_seeded(self) -> None:
        models = [
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ]

        plan_a = _build_run_plan(models, rounds=2, order="random", seed=7)
        plan_b = _build_run_plan(models, rounds=2, order="random", seed=7)

        self.assertEqual(
            [entry["model"]["id"] for entry in plan_a],
            [entry["model"]["id"] for entry in plan_b],
        )

    def test_build_run_plan_uses_plain_slug_for_single_run(self) -> None:
        plan = _build_run_plan([{"id": "gemma4:e4b"}], rounds=1, order="balanced", seed=None)
        self.assertEqual(plan[0]["run_dir_name"], "gemma4_e4b")


class RunnerTimingTests(unittest.TestCase):
    def test_first_meaningful_edit_ignores_cache_directories(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            cache_dir = workspace / ".venv"
            cache_dir.mkdir()
            (cache_dir / "ignored.txt").write_text("ignore\n")
            app_path = workspace / "app.py"
            app_path.write_text("print('ok')\n")
            started_ns = app_path.stat().st_mtime_ns - 1_000_000

            edit_seconds = _first_meaningful_edit_seconds(workspace, started_ns)

        self.assertEqual(edit_seconds, 0.0)


class RunnerPropagationTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_benchmark_passes_aider_stagnation_timeout_to_each_run(self) -> None:
        model = {"id": "qwen3.5:9b", "provider": "ollama"}
        with TemporaryDirectory() as tmp:
            results_root = Path(tmp) / "results"
            with (
                mock.patch("benchmark.runner.RESULTS_ROOT", results_root),
                mock.patch("benchmark.runner._load_task", return_value="Build the app"),
                mock.patch("benchmark.runner._new_job_id", return_value="20260417.083818"),
                mock.patch(
                    "benchmark.runner._run_one",
                    new=mock.AsyncMock(return_value={"model_id": model["id"]}),
                ) as run_one_mock,
                mock.patch(
                    "benchmark.runner.write_markdown_report",
                    return_value=Path(tmp) / "reports" / "results_20260417.083818.md",
                ),
            ):
                summaries = await run_benchmark(
                    [model],
                    agent_type="aider",
                    aider_stagnation_timeout=420,
                )

        self.assertEqual(summaries, [{"model_id": model["id"]}])
        self.assertEqual(
            run_one_mock.await_args.kwargs["aider_stagnation_timeout"],
            420,
        )
        self.assertEqual(run_one_mock.await_args.kwargs["round_index"], 1)
        self.assertEqual(run_one_mock.await_args.kwargs["position_in_round"], 1)
        self.assertEqual(run_one_mock.await_args.kwargs["total_rounds"], 1)

    async def test_run_benchmark_writes_job_metadata_and_generates_report(self) -> None:
        model = {"id": "gemma4:e2b", "provider": "ollama"}
        with TemporaryDirectory() as tmp:
            results_root = Path(tmp) / "results"
            report_path = Path(tmp) / "reports" / "results_20260417.083818.md"
            with (
                mock.patch("benchmark.runner.RESULTS_ROOT", results_root),
                mock.patch("benchmark.runner._load_task", return_value="Build the app"),
                mock.patch("benchmark.runner._new_job_id", return_value="20260417.083818"),
                mock.patch(
                    "benchmark.runner._run_one",
                    new=mock.AsyncMock(return_value={"model_id": model["id"]}),
                ),
                mock.patch(
                    "benchmark.runner.write_markdown_report",
                    return_value=report_path,
                ) as write_report_mock,
            ):
                await run_benchmark(
                    [model],
                    task_name="limerick",
                    agent_type="react",
                    timeout=600,
                    aider_stagnation_timeout=420,
                    enable_hardware_metrics=True,
                    rounds=3,
                    order="balanced",
                )

            job_metadata = json.loads((results_root / "20260417.083818" / "job.json").read_text())
            self.assertEqual(job_metadata["job_id"], "20260417.083818")
            self.assertEqual(job_metadata["task_name"], "limerick")
            self.assertEqual(job_metadata["agent_type"], "react")
            self.assertEqual(job_metadata["timeout_seconds"], 600)
            self.assertEqual(job_metadata["aider_stagnation_timeout_seconds"], 420)
            self.assertTrue(job_metadata["enable_hardware_metrics"])
            self.assertEqual(job_metadata["model_ids"], ["gemma4:e2b"])
            self.assertEqual(job_metadata["rounds"], 3)
            self.assertEqual(job_metadata["order"], "balanced")
            self.assertIsNone(job_metadata["seed"])
            self.assertEqual(job_metadata["total_runs"], 3)
            self.assertEqual(len(job_metadata["run_plan"]), 3)
            write_report_mock.assert_called_once_with(results_root / "20260417.083818")


class RunnerPortGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_one_checks_port_before_starting_model_run(self) -> None:
        model = {"id": "gemma4:e2b", "provider": "ollama"}
        with TemporaryDirectory() as tmp:
            results_root = Path(tmp) / "results"
            workspace_base = Path(tmp) / "workspaces"
            with (
                mock.patch("benchmark.runner.RESULTS_ROOT", results_root),
                mock.patch("benchmark.runner.WORKSPACE_BASE", workspace_base),
                mock.patch(
                    "benchmark.runner.assert_port_available",
                    side_effect=RuntimeError("Port 8181 is already in use before starting run for gemma4:e2b."),
                ) as assert_mock,
                mock.patch("benchmark.runner.run_agent", new=mock.AsyncMock()) as run_agent_mock,
            ):
                with self.assertRaisesRegex(RuntimeError, "starting run for gemma4:e2b"):
                    await _run_one(
                        model,
                        "Build the app.",
                        timeout=900,
                        aider_stagnation_timeout=420,
                        enable_hardware_metrics=False,
                        job_id="20260417.083818",
                        run_index=1,
                        total_runs=1,
                        round_index=1,
                        position_in_round=1,
                        total_rounds=1,
                        run_dir_name="gemma4_e2b",
                        agent_type="react",
                        run_label="1/1:gemma4-e2b:react",
                        task_name="limerick",
                    )

        assert_mock.assert_called_once()
        run_agent_mock.assert_not_called()


class RunnerSummaryTimingTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_one_records_timing_breakdown_and_lifecycle_fields(self) -> None:
        model = {"id": "gemma4:e2b", "provider": "ollama"}
        with TemporaryDirectory() as tmp:
            results_root = Path(tmp) / "results"
            workspace_base = Path(tmp) / "workspaces"
            collector = mock.Mock()

            async def fake_run_agent(*args, **kwargs):
                workspace = kwargs["workspace"]
                (workspace / "app.py").write_text("print('ok')\n")
                return {
                    "finish_reason": "completed",
                    "timed_out": False,
                    "error": None,
                    "agent_stop": None,
                }

            eval_result = {
                "entry_point": "uv run python app.py",
                "entry_point_candidates": ["uv run python app.py"],
                "entry_point_mismatch": False,
                "server_started": True,
                "http_status": 200,
                "response_bytes": 123,
                "body_has_refresh_mechanism": True,
                "body_has_limerick_shape": True,
                "startup_seconds": 2.7,
                "passed": True,
                "error": None,
            }

            with (
                mock.patch("benchmark.runner.RESULTS_ROOT", results_root),
                mock.patch("benchmark.runner.WORKSPACE_BASE", workspace_base),
                mock.patch("benchmark.runner.MetricsCollector", return_value=collector),
                mock.patch("benchmark.runner.run_agent", side_effect=fake_run_agent),
                mock.patch("benchmark.runner.evaluate", new=mock.AsyncMock(return_value=eval_result)),
                mock.patch(
                    "benchmark.runner.time.monotonic",
                    side_effect=[100.0, 112.4, 112.4, 115.1, 115.1],
                ),
            ):
                summary = await _run_one(
                    model,
                    "Build the app.",
                    timeout=900,
                    aider_stagnation_timeout=420,
                    enable_hardware_metrics=False,
                    job_id="20260417.083818",
                    run_index=1,
                    total_runs=1,
                    round_index=1,
                    position_in_round=1,
                    total_rounds=1,
                    run_dir_name="gemma4_e2b",
                    agent_type="react",
                    run_label="1/1:gemma4-e2b:react",
                    task_name="limerick",
                )

            saved_summary = json.loads((results_root / "20260417.083818" / "gemma4_e2b" / "summary.json").read_text())

        collector.start.assert_called_once()
        collector.stop.assert_called_once()
        self.assertEqual(summary["wall_seconds"], 15.1)
        self.assertEqual(summary["agent_seconds"], 12.4)
        self.assertEqual(summary["eval_seconds"], 2.7)
        self.assertEqual(summary["startup_seconds"], 2.7)
        self.assertIsInstance(summary["first_edit_seconds"], float)
        self.assertGreaterEqual(summary["first_edit_seconds"], 0.0)
        self.assertIsNotNone(summary["started_at"])
        self.assertIsNotNone(summary["agent_finished_at"])
        self.assertIsNotNone(summary["eval_started_at"])
        self.assertIsNotNone(summary["eval_finished_at"])
        self.assertIsNotNone(summary["finished_at"])
        self.assertTrue(summary["passed"])
        self.assertEqual(saved_summary["wall_seconds"], 15.1)
        self.assertEqual(saved_summary["agent_seconds"], 12.4)
        self.assertEqual(saved_summary["eval_seconds"], 2.7)
        self.assertEqual(saved_summary["startup_seconds"], 2.7)

    async def test_run_one_leaves_eval_timing_empty_when_evaluation_is_skipped(self) -> None:
        model = {"id": "qwen3.5:9b", "provider": "ollama"}
        with TemporaryDirectory() as tmp:
            results_root = Path(tmp) / "results"
            workspace_base = Path(tmp) / "workspaces"
            collector = mock.Mock()
            with (
                mock.patch("benchmark.runner.RESULTS_ROOT", results_root),
                mock.patch("benchmark.runner.WORKSPACE_BASE", workspace_base),
                mock.patch("benchmark.runner.MetricsCollector", return_value=collector),
                mock.patch(
                    "benchmark.runner.run_agent",
                    new=mock.AsyncMock(
                        return_value={
                            "finish_reason": "stuck_loop",
                            "timed_out": False,
                            "error": None,
                            "agent_stop": {
                                "category": "repeating_log_cycle",
                                "detail": "repeating log cycle detected",
                            },
                        }
                    ),
                ),
                mock.patch("benchmark.runner.evaluate", new=mock.AsyncMock()) as evaluate_mock,
                mock.patch(
                    "benchmark.runner.time.monotonic",
                    side_effect=[200.0, 205.4, 205.4],
                ),
            ):
                summary = await _run_one(
                    model,
                    "Build the app.",
                    timeout=900,
                    aider_stagnation_timeout=420,
                    enable_hardware_metrics=False,
                    job_id="20260417.083818",
                    run_index=1,
                    total_runs=1,
                    round_index=1,
                    position_in_round=1,
                    total_rounds=1,
                    run_dir_name="qwen3.5_9b",
                    agent_type="react",
                    run_label="1/1:qwen3.5-9b:react",
                    task_name="limerick",
                )

        evaluate_mock.assert_not_awaited()
        self.assertEqual(summary["wall_seconds"], 5.4)
        self.assertEqual(summary["agent_seconds"], 5.4)
        self.assertIsNone(summary["eval_seconds"])
        self.assertIsNone(summary["eval_started_at"])
        self.assertIsNone(summary["eval_finished_at"])
        self.assertIsNone(summary["startup_seconds"])
        self.assertFalse(summary["passed"])
