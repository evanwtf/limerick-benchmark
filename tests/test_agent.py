import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmark.agent import (
    _declared_dependencies,
    _contains_redundant_uv_init,
    _format_status_line,
    _normalize_dependency_name,
    _parse_tool_arguments,
    _prepare_command,
    _summarize_command_output,
    _workspace_has_started_work,
    _written_file_target,
)


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


class AgentWorkspaceDetectionTests(unittest.TestCase):
    def test_empty_workspace_is_not_started(self) -> None:
        with TemporaryDirectory() as tmp:
            self.assertFalse(_workspace_has_started_work(Path(tmp)))

    def test_initialized_workspace_counts_as_started_without_python_files(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.1.0'\n")
            self.assertTrue(_workspace_has_started_work(workspace))

    def test_prepare_command_skips_redundant_uv_init(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.1.0'\n")

            command, note = _prepare_command("uv init .\nuv add flask", workspace)

        self.assertEqual(command, "uv add flask")
        self.assertIn("skipped redundant `uv init`", note)

    def test_prepare_command_returns_note_when_only_uv_init_remains(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.1.0'\n")

            command, note = _prepare_command("uv init .", workspace)

        self.assertIsNone(command)
        self.assertIn("Do not run `uv init` again", note)

    def test_detects_redundant_uv_init_only_after_initialization(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self.assertFalse(_contains_redundant_uv_init("uv init .", workspace))
            (workspace / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.1.0'\n")
            self.assertTrue(_contains_redundant_uv_init("uv init .", workspace))

    def test_prepare_command_skips_redundant_uv_add(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "pyproject.toml").write_text(
                "[project]\nname='demo'\nversion='0.1.0'\ndependencies=['flask>=3.0']\n"
            )

            command, note = _prepare_command("uv add flask", workspace)

        self.assertIsNone(command)
        self.assertIn("skipped redundant `uv add`", note)

    def test_declared_dependencies_normalize_names(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "pyproject.toml").write_text(
                "[project]\nname='demo'\nversion='0.1.0'\ndependencies=['Flask[async]>=3.0; python_version >= \"3.11\"']\n"
            )

            deps = _declared_dependencies(workspace)

        self.assertEqual(deps, {"flask"})

    def test_normalize_dependency_name(self) -> None:
        self.assertEqual(_normalize_dependency_name("Flask[async]>=3.0"), "flask")

    def test_written_file_target_detects_redirect_target(self) -> None:
        self.assertEqual(_written_file_target("cat <<EOF > app.py\nhello\nEOF"), "app.py")
        self.assertEqual(_written_file_target("printf foo > src/app.py"), "src/app.py")
        self.assertIsNone(_written_file_target("uv run python app.py"))
