import unittest

from benchmark.runner import _should_evaluate


class RunnerEvaluationPolicyTests(unittest.TestCase):
    def test_skips_evaluation_for_redundant_setup_loop(self) -> None:
        self.assertFalse(_should_evaluate({"finish_reason": "redundant_uv_init_loop", "error": None}))

    def test_skips_evaluation_for_invalid_tool_loop(self) -> None:
        self.assertFalse(_should_evaluate({"finish_reason": "invalid_tool_loop", "error": None}))

    def test_skips_evaluation_when_agent_errors(self) -> None:
        self.assertFalse(_should_evaluate({"finish_reason": "error", "error": "boom"}))

    def test_keeps_evaluation_for_timeout(self) -> None:
        self.assertTrue(_should_evaluate({"finish_reason": "timeout", "error": None}))
