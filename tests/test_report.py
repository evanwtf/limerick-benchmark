import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from benchmark.report import generate_markdown_report, load_job_report, report_output_path, write_markdown_report


class ReportGenerationTests(unittest.TestCase):
    def test_generate_markdown_report_groups_repeated_runs_by_model(self) -> None:
        with TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "20260419.010203"
            job_dir.mkdir()
            (job_dir / "job.json").write_text(
                json.dumps(
                    {
                        "job_id": "20260419.010203",
                        "task_name": "limerick",
                        "agent_type": "aider",
                        "model_ids": ["gemma4:e4b", "qwen3.5:35b-a3b-coding-mxfp8"],
                        "rounds": 2,
                        "order": "balanced",
                        "total_runs": 4,
                    }
                )
            )

            self._write_model(
                job_dir,
                slug="01_gemma4_e4b__r01_p01",
                summary={
                    "model_id": "gemma4:e4b",
                    "run_index": 1,
                    "round_index": 1,
                    "position_in_round": 1,
                    "started_at": "2026-04-19T01:02:03+00:00",
                    "wall_seconds": 18.0,
                    "agent_seconds": 16.0,
                    "eval_seconds": 2.0,
                    "startup_seconds": 1.2,
                    "first_edit_seconds": 4.0,
                    "app_py_sha256": "hash-gemma-a",
                    "uses_render_template_string": True,
                    "uses_inline_html": True,
                    "route_count": 1,
                    "dependency_count": 1,
                    "tokens_in": 1000,
                    "tokens_out": 700,
                    "api_calls": None,
                    "tool_calls": None,
                    "finish_reason": "completed",
                    "timed_out": False,
                    "error": None,
                    "passed": True,
                    "eval": {
                        "entry_point": "uv run python app.py",
                        "server_started": True,
                        "http_status": 200,
                        "response_bytes": 300,
                        "error": None,
                    },
                },
            )
            self._write_model(
                job_dir,
                slug="02_qwen3.5_35b-a3b-coding-mxfp8__r01_p02",
                summary={
                    "model_id": "qwen3.5:35b-a3b-coding-mxfp8",
                    "run_index": 2,
                    "round_index": 1,
                    "position_in_round": 2,
                    "started_at": "2026-04-19T01:02:23+00:00",
                    "wall_seconds": 16.0,
                    "agent_seconds": 14.5,
                    "eval_seconds": 1.5,
                    "startup_seconds": 1.0,
                    "first_edit_seconds": 3.0,
                    "app_py_sha256": "hash-qwen-a",
                    "uses_render_template_string": False,
                    "uses_inline_html": True,
                    "route_count": 2,
                    "dependency_count": 1,
                    "tokens_in": 1000,
                    "tokens_out": 600,
                    "api_calls": None,
                    "tool_calls": None,
                    "finish_reason": "completed",
                    "timed_out": False,
                    "error": None,
                    "passed": True,
                    "eval": {
                        "entry_point": "uv run python app.py",
                        "server_started": True,
                        "http_status": 200,
                        "response_bytes": 280,
                        "error": None,
                    },
                },
            )
            self._write_model(
                job_dir,
                slug="03_qwen3.5_35b-a3b-coding-mxfp8__r02_p01",
                summary={
                    "model_id": "qwen3.5:35b-a3b-coding-mxfp8",
                    "run_index": 3,
                    "round_index": 2,
                    "position_in_round": 1,
                    "started_at": "2026-04-19T01:02:43+00:00",
                    "wall_seconds": 17.0,
                    "agent_seconds": 15.2,
                    "eval_seconds": 1.8,
                    "startup_seconds": 1.1,
                    "first_edit_seconds": 2.5,
                    "app_py_sha256": "hash-qwen-b",
                    "uses_render_template_string": True,
                    "uses_inline_html": True,
                    "route_count": 2,
                    "dependency_count": 2,
                    "tokens_in": 1000,
                    "tokens_out": 610,
                    "api_calls": None,
                    "tool_calls": None,
                    "finish_reason": "completed",
                    "timed_out": False,
                    "error": None,
                    "passed": True,
                    "eval": {
                        "entry_point": "uv run python app.py",
                        "server_started": True,
                        "http_status": 200,
                        "response_bytes": 290,
                        "error": None,
                    },
                },
            )
            self._write_model(
                job_dir,
                slug="04_gemma4_e4b__r02_p02",
                summary={
                    "model_id": "gemma4:e4b",
                    "run_index": 4,
                    "round_index": 2,
                    "position_in_round": 2,
                    "started_at": "2026-04-19T01:03:03+00:00",
                    "wall_seconds": 19.0,
                    "agent_seconds": 16.8,
                    "eval_seconds": 2.2,
                    "startup_seconds": 1.4,
                    "first_edit_seconds": 4.5,
                    "app_py_sha256": "hash-gemma-a",
                    "uses_render_template_string": True,
                    "uses_inline_html": True,
                    "route_count": 1,
                    "dependency_count": 1,
                    "tokens_in": 1000,
                    "tokens_out": 710,
                    "api_calls": None,
                    "tool_calls": None,
                    "finish_reason": "completed",
                    "timed_out": False,
                    "error": None,
                    "passed": True,
                    "eval": {
                        "entry_point": "uv run python app.py",
                        "server_started": True,
                        "http_status": 200,
                        "response_bytes": 310,
                        "error": None,
                    },
                },
            )

            markdown = generate_markdown_report(job_dir, task_label="limerick", include_placeholders=False)

        self.assertIn("**Models tested:** 2", markdown)
        self.assertIn("**Runs executed:** 4", markdown)
        self.assertIn("**Rounds:** 2", markdown)
        self.assertIn("**Order:** balanced", markdown)
        self.assertIn("| Runs | 2 |", markdown)
        self.assertIn("| Passes | 2/2 (100%) |", markdown)
        self.assertIn("| Wall time stddev | 0.5 s |", markdown)
        self.assertIn("| Wall time p90 | 19.0 s |", markdown)
        self.assertIn("| Distinct app hashes | 1 |", markdown)
        self.assertIn("| Distinct solution shapes | 1 |", markdown)
        self.assertIn("| Median agent time | 16.4 s |", markdown)
        self.assertIn("| Median startup time | 1.3 s |", markdown)
        self.assertIn("| Run | Round | Pos | Result | Wall Time | Finish | HTTP | Eval |", markdown)
        self.assertIn("| 1 | gemma4:e4b | 2 | 2/2 | 18.5 s | 0.5 s | 19.0 s | 1 | 1 |", markdown)
        self.assertIn("| 2 | qwen3.5:35b-a3b-coding-mxfp8 | 2 | 2/2 | 16.5 s | 0.5 s | 17.0 s | 2 | 2 |", markdown)
        self.assertIn("## Order Effects", markdown)
        self.assertIn("| Position | Runs | Pass Rate | Median Wall Time |", markdown)
        self.assertIn("| 1 | 2 | 100% | 17.5 s |", markdown)
        self.assertIn("| 2 | 2 | 100% | 17.5 s |", markdown)

    def test_generate_markdown_report_with_mixed_results(self) -> None:
        with TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "20260417.083818"
            job_dir.mkdir()

            self._write_model(
                job_dir,
                slug="gemma4_e2b",
                summary={
                    "model_id": "gemma4:e2b",
                    "started_at": "2026-04-17T12:38:35.781344+00:00",
                    "wall_seconds": 15.7,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "api_calls": 0,
                    "tool_calls": 0,
                    "finish_reason": "completed",
                    "timed_out": False,
                    "error": None,
                    "eval": {
                        "entry_point": "uv run python app.py",
                        "server_started": True,
                        "http_status": 200,
                        "response_bytes": 2064,
                        "error": None,
                    },
                },
                metrics_csv=(
                    "timestamp,elapsed_s,cpu_percent,memory_percent,gpu_utilization_percent,gpu_power_mw,"
                    "die_temp_c,fan_rpm,tokens_in,tokens_out,api_calls,tool_calls\n"
                    "2026-04-17T08:39:59,5.0,10.0,74.5,,,,,0,0,0,0\n"
                    "2026-04-17T08:40:04,10.0,20.0,75.0,,,,,0,0,0,0\n"
                ),
            )
            self._write_model(
                job_dir,
                slug="qwen3.5_9b",
                summary={
                    "model_id": "qwen3.5:9b",
                    "started_at": "2026-04-17T12:46:34.097863+00:00",
                    "wall_seconds": 71.8,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "api_calls": 0,
                    "tool_calls": 0,
                    "finish_reason": "stuck_loop",
                    "timed_out": False,
                    "error": "Detected infinite loop: repeating log cycle detected",
                    "eval": {
                        "entry_point": None,
                        "server_started": False,
                        "http_status": None,
                        "response_bytes": None,
                        "error": "evaluation_skipped",
                    },
                },
            )

            markdown = generate_markdown_report(job_dir, task_label="limerick")

        self.assertIn("# Benchmark Results - Job `20260417.083818`", markdown)
        self.assertIn("**Task:** limerick", markdown)
        self.assertIn("**Agent:** aider", markdown)
        self.assertIn("**Pass rate:** 1/2 (50%)", markdown)
        self.assertIn("| Result | PASS |", markdown)
        self.assertIn("| Result | FAIL |", markdown)
        self.assertIn("| Eval error | evaluation_skipped |", markdown)
        self.assertIn("| Avg CPU | 15.0% |", markdown)
        self.assertIn("| Max memory | 75.0% |", markdown)
        self.assertIn("| 1 | gemma4:e2b | PASS | 15.7 s | completed | 200 | - |", markdown)
        self.assertIn("| 2 | qwen3.5:9b | FAIL | 71.8 s | stuck_loop | - | evaluation_skipped |", markdown)
        self.assertIn("| completed | 1 |", markdown)
        self.assertIn("| stuck_loop | 1 |", markdown)
        self.assertIn("| evaluation_skipped | 1 |", markdown)
        self.assertIn("**Commentary:** _Add manual notes here._", markdown)

    def test_render_missing_agent_telemetry_as_na(self) -> None:
        with TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "20260417.155108"
            job_dir.mkdir()

            self._write_model(
                job_dir,
                slug="qwen3.5_9b",
                summary={
                    "model_id": "qwen3.5:9b",
                    "started_at": "2026-04-17T12:46:34.097863+00:00",
                    "wall_seconds": 71.8,
                    "tokens_in": None,
                    "tokens_out": None,
                    "api_calls": None,
                    "tool_calls": None,
                    "finish_reason": "aider_edit_format_reject",
                    "timed_out": False,
                    "error": None,
                    "passed": False,
                    "eval": {
                        "entry_point": None,
                        "server_started": False,
                        "http_status": None,
                        "response_bytes": None,
                        "error": "no_entry_point",
                    },
                },
            )

            markdown = generate_markdown_report(job_dir, task_label="limerick", include_placeholders=False)

        self.assertIn("| Tokens in | n/a |", markdown)
        self.assertIn("| Tokens out | n/a |", markdown)
        self.assertIn("| API calls | n/a |", markdown)
        self.assertIn("| Tool calls | n/a |", markdown)

    def test_render_passing_aider_warning_without_counting_it_as_failure_finish(self) -> None:
        with TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "20260417.165047"
            job_dir.mkdir()

            self._write_model(
                job_dir,
                slug="qwen3.5_35b-a3b",
                summary={
                    "model_id": "qwen3.5:35b-a3b",
                    "started_at": "2026-04-17T16:50:47.000000+00:00",
                    "wall_seconds": 88.4,
                    "tokens_in": 5200,
                    "tokens_out": 820,
                    "api_calls": None,
                    "tool_calls": None,
                    "finish_reason": "completed",
                    "timed_out": False,
                    "error": None,
                    "passed": True,
                    "agent_warning": {
                        "category": "aider_edit_format_reject",
                        "detail": "No filename provided before ``` in file listing",
                    },
                    "eval": {
                        "entry_point": "uv run python app.py",
                        "server_started": True,
                        "http_status": 200,
                        "response_bytes": 2048,
                        "error": None,
                    },
                },
            )

            markdown = generate_markdown_report(job_dir, task_label="limerick", include_placeholders=False)

        self.assertIn("| Agent warning | `aider_edit_format_reject` — No filename provided before ``` in file listing |", markdown)
        self.assertIn("| completed | 1 |", markdown)
        self.assertNotIn("| aider_edit_format_reject | 1 |", markdown)

    def test_load_job_report_infers_task_from_single_task_file_and_no_placeholders(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_dir = root / "20260417.083818"
            task_dir = root / "tasks"
            job_dir.mkdir()
            task_dir.mkdir()
            (task_dir / "limerick.md").write_text("# task\n")
            self._write_model(
                job_dir,
                slug="gemma4_e4b",
                summary={
                    "model_id": "gemma4:e4b",
                    "started_at": "2026-04-17T12:39:54.021703+00:00",
                    "wall_seconds": 29.1,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "api_calls": 0,
                    "tool_calls": 0,
                    "finish_reason": "completed",
                    "timed_out": False,
                    "error": None,
                    "eval": {
                        "entry_point": "uv run python app.py",
                        "server_started": True,
                        "http_status": 200,
                        "response_bytes": 3059,
                        "error": None,
                    },
                },
                agent_type="react",
            )

            with mock.patch("benchmark.report.TASKS_DIR", task_dir):
                report = load_job_report(job_dir)
                markdown = generate_markdown_report(job_dir, include_placeholders=False)

        self.assertEqual(report.task_label, "limerick")
        self.assertEqual(report.agent_label, "react")
        self.assertNotIn("**Commentary:** _Add manual notes here._", markdown)

    def test_write_markdown_report_uses_reports_directory_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_dir = root / "results" / "20260417.083818"
            job_dir.mkdir(parents=True)
            self._write_model(
                job_dir,
                slug="gemma4_e2b",
                summary={
                    "model_id": "gemma4:e2b",
                    "started_at": "2026-04-17T12:38:35.781344+00:00",
                    "wall_seconds": 15.7,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "api_calls": 0,
                    "tool_calls": 0,
                    "finish_reason": "completed",
                    "timed_out": False,
                    "error": None,
                    "eval": {
                        "entry_point": "uv run python app.py",
                        "server_started": True,
                        "http_status": 200,
                        "response_bytes": 2064,
                        "error": None,
                    },
                },
            )

            with mock.patch("benchmark.report.REPORTS_ROOT", root / "reports"):
                output_path = write_markdown_report(job_dir, include_placeholders=False)
                self.assertEqual(output_path, report_output_path("20260417.083818", root / "reports"))
                self.assertTrue(output_path.exists())
                self.assertIn("# Benchmark Results - Job `20260417.083818`", output_path.read_text())
                self.assertNotIn("**Commentary:** _Add manual notes here._", output_path.read_text())

    def _write_model(
        self,
        job_dir: Path,
        *,
        slug: str,
        summary: dict,
        metrics_csv: str | None = None,
        agent_type: str = "aider",
    ) -> None:
        run_dir = job_dir / slug
        run_dir.mkdir()
        (run_dir / "summary.json").write_text(json.dumps(summary))
        if metrics_csv is not None:
            (run_dir / "metrics.csv").write_text(metrics_csv)
        (run_dir / "trace.jsonl").write_text(
            json.dumps(
                {
                    "type": "agent_start",
                    "model": f"ollama_chat/{summary['model_id']}",
                    "agent_type": agent_type,
                }
            )
            + "\n"
        )
