import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmark.evaluator import (
    _candidate_entry_points,
    _classify_http_response,
    _python_file_contains_entrypoint_markers,
    _try_entry_point,
    evaluate,
)


class CandidateEntryPointTests(unittest.TestCase):
    def test_discovers_project_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "pyproject.toml").write_text(
                "[project]\n"
                "name = 'demo'\n"
                "version = '0.1.0'\n"
                "[project.scripts]\n"
                "serve-demo = 'demo:main'\n"
            )

            self.assertIn("uv run serve-demo", _candidate_entry_points(workspace))

    def test_discovers_src_package_main_modules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            package_dir = workspace / "src" / "demoapp"
            package_dir.mkdir(parents=True)
            (package_dir / "__main__.py").write_text("print('hello')\n")

            self.assertIn("uv run python -m demoapp", _candidate_entry_points(workspace))

    def test_discovers_src_flask_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            src_dir = workspace / "src"
            src_dir.mkdir()
            (src_dir / "server.py").write_text("from flask import Flask\napp = Flask(__name__)\n")

            self.assertIn("uv run python src/server.py", _candidate_entry_points(workspace))

    def test_candidate_entry_points_scans_python_files_without_read_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "demo.py").write_text("from flask import Flask\n" + ("# filler\n" * 10000))

            with mock.patch("pathlib.Path.read_text", side_effect=AssertionError("read_text should not be used")):
                candidates = _candidate_entry_points(workspace)

        self.assertIn("uv run python demo.py", candidates)

    def test_python_file_marker_scan_stops_after_byte_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            py = Path(tmp) / "demo.py"
            py.write_text(("# filler\n" * 9000) + "app.run()\n")

            self.assertFalse(_python_file_contains_entrypoint_markers(py))


class EvaluatorBodyChecksTests(unittest.TestCase):
    def test_classify_http_response_passes_valid_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "limericks.txt").write_text(
                "There once was a coder from Leeds\n"
                "Who worked at remarkable speeds\n"
                "She shipped with delight\n"
                "And tested each night\n"
                "While pruning her duplicate reads\n"
            )

            result = _classify_http_response(
                200,
                (
                    b"<html><head><meta http-equiv='refresh' content='5'></head>"
                    b"<body><pre>One line\nTwo line\nThree line\nFour line\nFive line</pre></body></html>"
                ),
                workspace,
            )

        self.assertTrue(result["body_has_refresh_mechanism"])
        self.assertTrue(result["body_has_limerick_shape"])
        self.assertTrue(result["passed"])
        self.assertIsNone(result["error"])

    def test_classify_http_response_rejects_missing_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            result = _classify_http_response(
                200,
                b"<html><body><pre>One\nTwo\nThree\nFour\nFive</pre></body></html>",
                workspace,
            )

        self.assertFalse(result["body_has_refresh_mechanism"])
        self.assertTrue(result["body_has_limerick_shape"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["error"], "body_missing_refresh")

    def test_classify_http_response_rejects_missing_limerick(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            result = _classify_http_response(
                200,
                (
                    b"<html><head><script>setInterval(function(){}, 5000)</script></head>"
                    b"<body><p>Hello world</p></body></html>"
                ),
                workspace,
            )

        self.assertTrue(result["body_has_refresh_mechanism"])
        self.assertFalse(result["body_has_limerick_shape"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["error"], "body_missing_limerick")


class EvaluatorPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_evaluate_uses_app_py_even_when_other_candidates_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            results_dir = workspace / "results"
            results_dir.mkdir()
            (workspace / "app.py").write_text("print('app')\n")
            (workspace / "main.py").write_text("print('main')\n")

            with (
                mock.patch(
                    "benchmark.evaluator._try_entry_point",
                    new=mock.AsyncMock(
                        return_value={
                            "entry_point": "uv run python app.py",
                            "entry_point_candidates": [],
                            "entry_point_mismatch": False,
                            "server_started": True,
                            "http_status": 200,
                            "response_bytes": 123,
                            "body_has_refresh_mechanism": True,
                            "body_has_limerick_shape": True,
                            "passed": True,
                            "error": None,
                        }
                    ),
                ) as try_mock,
                mock.patch("benchmark.evaluator._write_run_sh") as write_run_sh_mock,
            ):
                result = await evaluate(workspace, results_dir)

        try_mock.assert_awaited_once_with(workspace, "uv run python app.py")
        write_run_sh_mock.assert_called_once_with(results_dir, workspace, "uv run python app.py")
        self.assertFalse(result["entry_point_mismatch"])
        self.assertEqual(result["entry_point"], "uv run python app.py")
        self.assertIn("uv run python app.py", result["entry_point_candidates"])
        self.assertIn("uv run python main.py", result["entry_point_candidates"])

    async def test_evaluate_marks_non_app_py_candidates_as_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            results_dir = workspace / "results"
            results_dir.mkdir()
            (workspace / "main.py").write_text("print('main')\n")

            with (
                mock.patch(
                    "benchmark.evaluator._try_entry_point",
                    new=mock.AsyncMock(),
                ) as try_mock,
                mock.patch("benchmark.evaluator._write_run_sh") as write_run_sh_mock,
            ):
                result = await evaluate(workspace, results_dir)

        try_mock.assert_not_called()
        write_run_sh_mock.assert_called_once_with(results_dir, workspace, "uv run python main.py")
        self.assertTrue(result["entry_point_mismatch"])
        self.assertEqual(result["error"], "entry_point_mismatch")
        self.assertEqual(result["entry_point"], "uv run python main.py")
        self.assertEqual(result["entry_point_candidates"], ["uv run python main.py"])

    async def test_try_entry_point_checks_port_before_starting_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            with (
                mock.patch(
                    "benchmark.evaluator.assert_port_available",
                    side_effect=RuntimeError("Port 8181 is already in use before starting evaluator command 'uv run python app.py'."),
                ) as assert_mock,
                mock.patch("benchmark.evaluator.asyncio.create_subprocess_shell") as create_proc_mock,
            ):
                with self.assertRaisesRegex(RuntimeError, "starting evaluator command 'uv run python app.py'"):
                    await _try_entry_point(workspace, "uv run python app.py")

        assert_mock.assert_called_once()
        create_proc_mock.assert_not_called()

    async def test_try_entry_point_records_startup_seconds_after_http_response(self) -> None:
        class FakeResponse:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def read(self) -> bytes:
                return (
                    b"<html><head><meta http-equiv='refresh' content='5'></head>"
                    b"<body><pre>One\nTwo\nThree\nFour\nFive</pre></body></html>"
                )

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def get(self, *args, **kwargs):
                return FakeResponse()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            proc = mock.Mock(pid=123, returncode=0)
            with (
                mock.patch("benchmark.evaluator.assert_port_available"),
                mock.patch(
                    "benchmark.evaluator.asyncio.create_subprocess_shell",
                    new=mock.AsyncMock(return_value=proc),
                ),
                mock.patch(
                    "benchmark.evaluator._wait_for_port",
                    new=mock.AsyncMock(return_value=True),
                ),
                mock.patch(
                    "benchmark.evaluator.listener_belongs_to_process_tree",
                    return_value=True,
                ),
                mock.patch(
                    "benchmark.evaluator.terminate_process_group",
                    new=mock.AsyncMock(),
                ),
                mock.patch("benchmark.evaluator.aiohttp.ClientSession", return_value=FakeSession()),
                mock.patch("benchmark.evaluator.time.monotonic", side_effect=[10.0, 12.6]),
            ):
                result = await _try_entry_point(workspace, "uv run python app.py")

        self.assertEqual(result["http_status"], 200)
        self.assertEqual(result["startup_seconds"], 2.6)
        self.assertTrue(result["passed"])
