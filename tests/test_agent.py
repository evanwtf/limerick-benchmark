import unittest

from benchmark.agent import _format_status_line, _parse_tool_arguments, _summarize_command_output


class ParseToolArgumentsTests(unittest.TestCase):
    def test_rejects_invalid_json(self) -> None:
        with self.assertRaises(ValueError):
            _parse_tool_arguments('{"command": ')

    def test_rejects_non_object_json(self) -> None:
        with self.assertRaises(ValueError):
            _parse_tool_arguments('["pwd"]')

    def test_accepts_valid_object_json(self) -> None:
        self.assertEqual(_parse_tool_arguments('{"command": "pwd"}'), {"command": "pwd"})


class AgentConsoleFormattingTests(unittest.TestCase):
    def test_formats_compact_status_line(self) -> None:
        line = _format_status_line(
            "ollama_chat/qwen3.5:9b",
            elapsed_s=134.0,
            phase="thinking",
            api_calls=3,
            tool_calls=7,
            output_tokens=1842,
            tokens_per_second=21.6,
        )

        self.assertIn("[ollama_chat/qwen3.5:9b] 02:14 | thinking", line)
        self.assertIn("api=3 tool=7", line)
        self.assertIn("1842", line)
        self.assertIn("21.6 tok/s", line)

    def test_summarizes_multiline_command_output(self) -> None:
        summary = _summarize_command_output("line one\nline two\nline three\n")
        self.assertIn("3 lines", summary)
        self.assertIn("line one", summary)

    def test_preserves_short_single_line_output(self) -> None:
        self.assertEqual(_summarize_command_output("installed"), "installed")
