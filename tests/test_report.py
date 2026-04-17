import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from benchmark.report import generate_markdown_report, load_job_report


class ReportGenerationTests(unittest.TestCase):
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
