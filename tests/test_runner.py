import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from benchmark.runner import _prepare_workspace, _should_evaluate, _task_prompt_with_workspace_note


class RunnerEvaluationPolicyTests(unittest.TestCase):
    def test_skips_evaluation_for_redundant_setup_loop(self) -> None:
        self.assertFalse(_should_evaluate({"finish_reason": "redundant_uv_init_loop", "error": None}))

    def test_skips_evaluation_for_invalid_tool_loop(self) -> None:
        self.assertFalse(_should_evaluate({"finish_reason": "invalid_tool_loop", "error": None}))

    def test_skips_evaluation_for_repeated_command_loop(self) -> None:
        self.assertFalse(_should_evaluate({"finish_reason": "repeated_command_loop", "error": None}))

    def test_skips_evaluation_when_agent_errors(self) -> None:
        self.assertFalse(_should_evaluate({"finish_reason": "error", "error": "boom"}))

    def test_keeps_evaluation_for_timeout(self) -> None:
        self.assertTrue(_should_evaluate({"finish_reason": "timeout", "error": None}))


class RunnerWorkspacePreparationTests(unittest.TestCase):
    def test_prepare_workspace_runs_uv_init_once(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            with mock.patch("benchmark.runner.subprocess.run") as run_mock:
                _prepare_workspace(workspace)

        run_mock.assert_called_once()
        args, kwargs = run_mock.call_args
        self.assertEqual(args[0][:3], ["uv", "init", "."])
        self.assertEqual(kwargs["cwd"], workspace)
        self.assertTrue(kwargs["check"])

    def test_prepare_workspace_skips_existing_project(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.1.0'\n")
            with mock.patch("benchmark.runner.subprocess.run") as run_mock:
                _prepare_workspace(workspace)

        run_mock.assert_not_called()

    def test_task_prompt_includes_workspace_note(self) -> None:
        prompt = _task_prompt_with_workspace_note("Build the app.")
        self.assertIn("already initialized as a uv project", prompt)
        self.assertIn("Do not run `uv init`", prompt)
        self.assertTrue(prompt.endswith("Build the app."))
