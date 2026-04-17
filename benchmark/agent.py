"""Agent loop: sends a task to a model, executes bash tool calls, records trace."""

import asyncio
import hashlib
import json
import logging
import re
import shlex
import sys
import time
import tomllib
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import litellm

from .process_utils import (
    listener_matches_process_groups,
    port_accepts_connections,
    process_group_exists,
    sanitized_subprocess_env,
    terminate_process_groups,
)

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 900  # 15 minutes hard limit
CMD_TIMEOUT_SECONDS = 60  # per bash command
MAX_OUTPUT_CHARS = 8000  # truncate long command output
STATUS_REFRESH_SECONDS = 0.25
MAX_REDUNDANT_UV_INIT_STREAK = 5
MAX_REPEATED_COMMAND_STREAK = 5
MAX_REPEATED_FILE_WRITE_STREAK = 3

SYSTEM_PROMPT = """\
You are an expert Python developer. Complete the coding task by calling the bash tool.

You have exactly ONE tool: bash. Call it to run shell commands.
Do NOT call any other tool name — only "bash" exists.

Rules:
- Work in the current directory only.
- The workspace is already initialized as a uv project. Do NOT run `uv init`.
- Use `uv` for all Python package management.
- Test that your work actually runs before declaring done.
- When the task is complete and working, stop and summarize what you built.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command in the workspace directory. "
                "stdout and stderr are combined and returned. "
                "Long-running servers should be started in the background with '&'. "
                f"Commands time out after {CMD_TIMEOUT_SECONDS}s."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run"},
                },
                "required": ["command"],
            },
        },
    }
]

_FILE_WRITE_RE = re.compile(r">\s*([A-Za-z0-9_./-]+)")


def _parse_tool_arguments(raw_args: str | None) -> dict[str, Any]:
    """Decode function-call JSON arguments into a dict."""
    if raw_args is None:
        return {}

    try:
        data = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON arguments: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise ValueError("tool arguments must decode to a JSON object")
    return data


def _format_status_line(
    model_id: str,
    *,
    elapsed_s: float,
    phase: str,
    api_calls: int,
    tool_calls: int,
    output_tokens: int,
    tokens_per_second: float,
) -> str:
    """Render one compact live status line for terminal output."""
    minutes = int(elapsed_s // 60)
    seconds = int(elapsed_s % 60)
    return (
        f"\r\033[2K[{model_id}] {minutes:02d}:{seconds:02d} | "
        f"{phase:<11} | api={api_calls} tool={tool_calls} | "
        f"out~={output_tokens:4d} tok | {tokens_per_second:4.1f} tok/s"
    )


def _summarize_command_output(output: str) -> str:
    """Return a short human-readable summary of command output."""
    if output in {"(no output)", ""}:
        return "(no output)"
    if output.startswith("[timeout") or output.startswith("[error:"):
        return output

    stripped = output.strip()
    if "\n" not in stripped and len(stripped) <= 120:
        return stripped

    lines = [line for line in output.splitlines() if line.strip()]
    line_count = len(output.splitlines())
    preview = lines[0][:120] if lines else ""
    if preview:
        return f"{line_count} lines, {len(output)} chars | {preview}"
    return f"{line_count} lines, {len(output)} chars"


def _workspace_has_started_work(workspace: Path) -> bool:
    """Return True once the model has created any workspace artifact."""
    return any(workspace.iterdir())


def _contains_redundant_uv_init(command: str, workspace: Path) -> bool:
    """Return True if the command retries `uv init` after initialization."""
    if not (workspace / "pyproject.toml").exists():
        return False
    return any(line.strip().startswith("uv init") for line in command.splitlines())


def _normalize_dependency_name(spec: str) -> str:
    """Normalize a dependency specifier to its package name."""
    token = spec.strip()
    if not token:
        return ""
    token = token.split(";")[0].strip()
    for marker in ("[", "=", "<", ">", "!", "~"):
        if marker in token:
            token = token.split(marker, 1)[0]
    return token.strip().lower().replace("_", "-")


def _declared_dependencies(workspace: Path) -> set[str]:
    """Return normalized dependency names already declared in pyproject.toml."""
    pyproject = workspace / "pyproject.toml"
    if not pyproject.exists():
        return set()

    try:
        data = tomllib.loads(pyproject.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return set()

    declared: set[str] = set()
    for spec in data.get("project", {}).get("dependencies", []) or []:
        name = _normalize_dependency_name(spec)
        if name:
            declared.add(name)
    return declared


def _written_file_target(command: str) -> str | None:
    """Return the file path being overwritten, when detectable."""
    matches = _FILE_WRITE_RE.findall(command)
    if not matches:
        return None
    return matches[-1]


def _prepare_command(command: str, workspace: Path) -> tuple[str | None, str | None]:
    """
    Rewrite redundant setup commands before execution.

    Returns (command_to_run, note). When command_to_run is None, the caller
    should skip execution and return note directly as the tool output.
    """
    if not (workspace / "pyproject.toml").exists():
        return command, None

    lines = command.splitlines()
    kept_lines: list[str] = []
    notes: list[str] = []
    declared_dependencies = _declared_dependencies(workspace)

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("uv init"):
            notes.append("Project already initialized; skipped redundant `uv init`.")
            continue
        if stripped.startswith("uv add"):
            try:
                tokens = shlex.split(stripped)
            except ValueError:
                kept_lines.append(line)
                continue
            packages = [token for token in tokens[2:] if not token.startswith("-")]
            if packages and all(_normalize_dependency_name(pkg) in declared_dependencies for pkg in packages):
                notes.append("Dependencies already declared; skipped redundant `uv add`.")
                continue
            for pkg in packages:
                normalized = _normalize_dependency_name(pkg)
                if normalized:
                    declared_dependencies.add(normalized)
            kept_lines.append(line)
            continue
        kept_lines.append(line)

    if not notes:
        return command, None

    note = " ".join(notes) + " Do not run `uv init` again. Proceed with creating files and starting the server."
    if kept_lines:
        return "\n".join(kept_lines), note
    return None, note


async def _run_bash(command: str, workspace: Path, active_process_groups: set[int]) -> str:
    logger.debug("bash: %s", command[:120])
    command_to_run, note = _prepare_command(command, workspace)
    if command_to_run is None:
        return note or "(no output)"

    proc: asyncio.subprocess.Process | None = None
    pgid: int | None = None
    try:
        proc = await asyncio.create_subprocess_shell(
            command_to_run,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=workspace,
            env=sanitized_subprocess_env(),
            start_new_session=True,
        )
        pgid = proc.pid
        active_process_groups.add(pgid)
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=CMD_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            await terminate_process_groups({pgid})
            return f"[timeout after {CMD_TIMEOUT_SECONDS}s]"
        except asyncio.CancelledError:
            await terminate_process_groups({pgid})
            raise

        output = stdout.decode("utf-8", errors="replace")
        if len(output) > MAX_OUTPUT_CHARS:
            half = MAX_OUTPUT_CHARS // 2
            output = output[:half] + f"\n...[{len(output) - MAX_OUTPUT_CHARS} chars truncated]...\n" + output[-half:]
        if note:
            output = f"{note}\n\n{output}" if output != "(no output)" else note
        if pgid is not None and not process_group_exists(pgid):
            active_process_groups.discard(pgid)
        return output or "(no output)"
    except Exception as exc:
        return f"[error: {exc}]"


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Aider loop-detection helpers ---------------------------------------------

AIDER_REPEAT_WINDOW = 60
AIDER_UNIQUE_THRESHOLD = 8
AIDER_CYCLE_MIN_PERIOD = 2
AIDER_CYCLE_MAX_PERIOD = 20
AIDER_CYCLE_MIN_REPEATS = 3
AIDER_MAX_EDITS_PER_FILE = 6
AIDER_STAGNATION_SECONDS = 300
AIDER_STAGNATION_POLL_SECONDS = 20
AIDER_NORMALIZED_HISTORY_MAX = 400

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_NUMERIC_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_PATH_LIKE_RE = re.compile(r"(?:[A-Za-z]:)?(?:[\w.-]+[/\\])+[\w.-]+")
_HEX_RE = re.compile(r"\b[0-9a-f]{8,}\b", re.IGNORECASE)

_AIDER_EDIT_PATTERNS = [
    re.compile(r"Applied edit to\s+(\S+)"),
    re.compile(r"Edited\s+(\S+)"),
    re.compile(r"Wrote\s+changes\s+to\s+(\S+)"),
    re.compile(r"Writing\s+(?:to\s+)?(\S+)"),
]

_AIDER_TREE_IGNORE_DIR_NAMES = {".venv", ".git", "__pycache__", "node_modules"}
_AIDER_TREE_IGNORE_PREFIXES = (".aider.",)


def _normalize_aider_line(text: str) -> str:
    """Collapse noise (ANSI, paths, numbers, hex) so repeats with minor variation still match."""
    s = _ANSI_RE.sub("", text)
    s = _HEX_RE.sub("<hex>", s)
    s = _PATH_LIKE_RE.sub("<path>", s)
    s = _NUMERIC_RE.sub("<n>", s)
    return " ".join(s.split())


def _aider_low_uniqueness(
    normalized: list[str],
    window: int = AIDER_REPEAT_WINDOW,
    threshold: int = AIDER_UNIQUE_THRESHOLD,
) -> bool:
    """True if the last `window` normalized lines have fewer than `threshold` unique values."""
    if len(normalized) < window:
        return False
    return len(set(normalized[-window:])) < threshold


def _aider_has_repeating_cycle(
    normalized: list[str],
    min_period: int = AIDER_CYCLE_MIN_PERIOD,
    max_period: int = AIDER_CYCLE_MAX_PERIOD,
    min_repeats: int = AIDER_CYCLE_MIN_REPEATS,
) -> bool:
    """True if the tail contains a k-line block repeating at least `min_repeats` times."""
    n = len(normalized)
    upper = min(max_period, n // min_repeats)
    for period in range(min_period, upper + 1):
        needed = period * min_repeats
        if n < needed:
            continue
        tail = normalized[-needed:]
        block = tail[:period]
        if all(tail[i * period : (i + 1) * period] == block for i in range(min_repeats)):
            return True
    return False


def _extract_aider_edit_target(line: str) -> str | None:
    for pat in _AIDER_EDIT_PATTERNS:
        m = pat.search(line)
        if m:
            return m.group(1).strip(".,:;\"'`")
    return None


def _hash_workspace_tree(workspace: Path) -> str:
    """Content hash of the workspace, skipping caches and virtualenvs."""
    h = hashlib.sha256()
    if not workspace.exists():
        return h.hexdigest()
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(workspace)
        except ValueError:
            continue
        parts = rel.parts
        if any(p in _AIDER_TREE_IGNORE_DIR_NAMES for p in parts):
            continue
        if any(p.startswith(_AIDER_TREE_IGNORE_PREFIXES) for p in parts):
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        h.update(str(rel).encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(data).digest())
        h.update(b"\0")
    return h.hexdigest()


async def _run_aider(
    *,
    model_id: str,
    provider: str,
    task_prompt: str,
    workspace: Path,
    trace_path: Path,
    token_state: dict[str, Any],
    timeout: int,
    aider_stagnation_timeout: int = AIDER_STAGNATION_SECONDS,
    run_label: str = "aider",
) -> dict[str, Any]:
    """
    Run aider-chat in headless mode for the task.
    """
    if provider == "ollama":
        # Aider uses ollama/ prefix but LiteLLM-style names work via ollama_chat/
        aider_model = f"ollama_chat/{model_id}"
    else:
        aider_model = model_id

    # Aider likes to have a git repo, but we don't want it to commit.
    # We use --yes-always to avoid prompts.
    cmd = [
        "uv", "run", "aider",
        "--model", aider_model,
        "--message", task_prompt,
        "--yes-always",
        "--no-auto-commits",
        "--no-git",
        "--no-suggest-shell-commands",
        "--no-check-update",
        "--exit",
    ]

    env = sanitized_subprocess_env()
    if provider == "ollama":
        env["OLLAMA_API_BASE"] = "http://localhost:11434"

    stats: dict[str, Any] = {
        "finish_reason": "completed",
        "timed_out": False,
        "error": None,
        "agent_stop": None,
    }

    trace: list[dict] = []
    def append_trace(entry: dict) -> None:
        entry["ts"] = _ts()
        trace.append(entry)
        with open(trace_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    logger.info("Starting aider with model %s", aider_model)
    append_trace(
        {
            "type": "agent_start",
            "model": aider_model,
            "agent_type": "aider",
            "aider_stagnation_timeout_seconds": aider_stagnation_timeout,
        }
    )

    pgid: int | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=workspace,
            env=env,
            start_new_session=True,
        )
        pgid = proc.pid

        normalized_lines: list[str] = []
        edit_counts: dict[str, int] = {}
        abort_reason: str | None = None
        abort_category: str | None = None

        def trip(category: str, reason: str) -> None:
            nonlocal abort_reason, abort_category
            if abort_reason is None:
                abort_reason = reason
                abort_category = category
                logger.error("Aborting aider [%s]: %s", category, reason)

        async def read_output() -> None:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                if abort_reason:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if not text:
                    continue
                print(f"[{run_label}] {text}", flush=True)
                append_trace({"type": "aider_log", "content": text})

                normalized = _normalize_aider_line(text)
                if normalized:
                    normalized_lines.append(normalized)
                    if len(normalized_lines) > AIDER_NORMALIZED_HISTORY_MAX:
                        del normalized_lines[:-AIDER_NORMALIZED_HISTORY_MAX]

                if _aider_low_uniqueness(normalized_lines):
                    trip(
                        "low_log_uniqueness",
                        f"<{AIDER_UNIQUE_THRESHOLD} unique normalized "
                        f"lines in last {AIDER_REPEAT_WINDOW}",
                    )
                    break
                if _aider_has_repeating_cycle(normalized_lines):
                    trip("repeating_log_cycle", "repeating log cycle detected")
                    break

                edited = _extract_aider_edit_target(text)
                if edited:
                    edit_counts[edited] = edit_counts.get(edited, 0) + 1
                    if edit_counts[edited] > AIDER_MAX_EDITS_PER_FILE:
                        trip(
                            "file_edit_cap",
                            f"file {edited!r} edited {edit_counts[edited]} times "
                            f"(> {AIDER_MAX_EDITS_PER_FILE})",
                        )
                        break

        async def watch_stagnation() -> None:
            last_hash = _hash_workspace_tree(workspace)
            last_change = time.monotonic()
            while True:
                await asyncio.sleep(AIDER_STAGNATION_POLL_SECONDS)
                if abort_reason:
                    return
                current = _hash_workspace_tree(workspace)
                now = time.monotonic()
                if current != last_hash:
                    last_hash = current
                    last_change = now
                    continue
                idle = now - last_change
                if idle > aider_stagnation_timeout:
                    trip(
                        "workspace_stagnation",
                        f"workspace unchanged for {int(idle)}s",
                    )
                    return

        reader = asyncio.create_task(read_output())
        watcher = asyncio.create_task(watch_stagnation())

        async def supervise() -> None:
            done, pending = await asyncio.wait(
                {reader, watcher}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in pending:
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        try:
            await asyncio.wait_for(supervise(), timeout=timeout)
        except asyncio.TimeoutError:
            stats["timed_out"] = True
            stats["finish_reason"] = "timeout"
            stats["agent_stop"] = {"category": "timeout", "detail": f"no completion within {timeout}s"}
            logger.warning("Aider timed out after %ds", timeout)
            for task in (reader, watcher):
                task.cancel()

        if abort_reason is not None:
            stats["error"] = f"Detected infinite loop: {abort_reason}"
            stats["finish_reason"] = "stuck_loop"
            stats["agent_stop"] = {"category": abort_category, "detail": abort_reason}

        if proc.returncode is None:
            await terminate_process_groups({pgid})

        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
        
        if (
            proc.returncode not in (0, None)
            and not stats["timed_out"]
            and stats["finish_reason"] == "completed"
        ):
            stats["error"] = f"aider exited with code {proc.returncode}"
            stats["finish_reason"] = "error"
            stats["agent_stop"] = {
                "category": "nonzero_exit",
                "detail": f"aider exited with code {proc.returncode}",
            }

    except Exception as exc:
        stats["error"] = str(exc)
        stats["finish_reason"] = "error"
        stats["agent_stop"] = {"category": "agent_exception", "detail": str(exc)}
        logger.error("Aider error: %s", exc, exc_info=True)
        if pgid:
            await terminate_process_groups({pgid})
    
    append_trace({"type": "agent_end", "stats": stats})
    return stats


async def run_agent(
    *,
    model_id: str,
    provider: str,
    task_prompt: str,
    workspace: Path,
    trace_path: Path,
    token_state: dict[str, Any],
    timeout: int = TIMEOUT_SECONDS,
    aider_stagnation_timeout: int = AIDER_STAGNATION_SECONDS,
    agent_type: str = "react",
    run_label: str = "aider",
) -> dict[str, Any]:
    if agent_type == "aider":
        return await _run_aider(
            model_id=model_id,
            provider=provider,
            task_prompt=task_prompt,
            workspace=workspace,
            trace_path=trace_path,
            token_state=token_state,
            timeout=timeout,
            aider_stagnation_timeout=aider_stagnation_timeout,
            run_label=run_label,
        )
    return await _run_react(
        model_id=model_id,
        provider=provider,
        task_prompt=task_prompt,
        workspace=workspace,
        trace_path=trace_path,
        token_state=token_state,
        timeout=timeout,
    )


async def _run_react(
    *,
    model_id: str,
    provider: str,
    task_prompt: str,
    workspace: Path,
    trace_path: Path,
    token_state: dict[str, Any],
    timeout: int = TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """
    Run the agent loop for one model. Returns a stats dict.

    token_state is mutated in-place so the metrics collector can read live counts.
    """
    if provider == "ollama":
        # ollama_chat/ uses /api/chat endpoint which supports native tool calling.
        # ollama/ uses /api/generate and falls back to JSON-format tool emulation,
        # which breaks for thinking models (Qwen3, etc.) that stream reasoning tokens.
        litellm_model = f"ollama_chat/{model_id}"
        extra_kwargs: dict[str, Any] = {"api_base": "http://localhost:11434"}
    else:
        litellm_model = model_id
        extra_kwargs = {}

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
    ]

    trace: list[dict] = []
    stats: dict[str, Any] = {
        "finish_reason": None,
        "timed_out": False,
        "error": None,
        "agent_stop": None,
    }
    nudge_count = 0
    MAX_NUDGES = 2
    invalid_tool_count = 0
    MAX_INVALID_TOOLS = 5
    active_process_groups: set[int] = set()
    agent_started_at = asyncio.get_running_loop().time()
    redundant_uv_init_streak = 0
    last_command_signature: str | None = None
    repeated_command_streak = 0
    last_written_file: str | None = None
    repeated_file_write_streak = 0

    def append_trace(entry: dict) -> None:
        entry["ts"] = _ts()
        trace.append(entry)

    def save_trace() -> None:
        with open(trace_path, "w") as f:
            for entry in trace:
                f.write(json.dumps(entry) + "\n")

    append_trace({"type": "task", "content": task_prompt})

    async def _loop() -> None:
        nonlocal nudge_count, invalid_tool_count, redundant_uv_init_streak
        nonlocal last_command_signature, repeated_command_streak
        nonlocal last_written_file, repeated_file_write_streak
        while True:
            token_state["api_calls"] += 1
            logger.info("API call #%d to %s", token_state["api_calls"], litellm_model)

            # Stream the response so the user can see tokens as they arrive
            stream = await litellm.acompletion(
                model=litellm_model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                stream=True,
                **extra_kwargs,
            )

            full_content = ""
            tool_call_chunks: dict[int, dict] = {}  # index → {id, name, args}
            finish_reason = None
            stream_started_at = asyncio.get_running_loop().time()
            stream_text_chars = 0
            stream_completion_tokens = 0
            current_phase = "requesting"
            last_status_update = 0.0

            def render_status(phase: str, *, force: bool = False) -> None:
                nonlocal current_phase, last_status_update
                now = asyncio.get_running_loop().time()
                if not force and (now - last_status_update) < STATUS_REFRESH_SECONDS:
                    return
                current_phase = phase
                elapsed = now - agent_started_at
                stream_elapsed = max(now - stream_started_at, 0.001)
                output_tokens = max(stream_completion_tokens, stream_text_chars // 4)
                tokens_per_second = output_tokens / stream_elapsed
                print(
                    _format_status_line(
                        litellm_model,
                        elapsed_s=elapsed,
                        phase=current_phase,
                        api_calls=token_state["api_calls"],
                        tool_calls=token_state["tool_calls"],
                        output_tokens=output_tokens,
                        tokens_per_second=tokens_per_second,
                    ),
                    end="",
                    flush=True,
                )
                last_status_update = now

            render_status(current_phase, force=True)
            async for chunk in stream:
                c = chunk.choices[0]
                delta = c.delta

                thinking = getattr(delta, "reasoning_content", None)
                if thinking:
                    stream_text_chars += len(thinking)
                    current_phase = "thinking"

                if delta.content:
                    stream_text_chars += len(delta.content)
                    full_content += delta.content
                    current_phase = "responding"

                for tc_chunk in delta.tool_calls or []:
                    idx = tc_chunk.index
                    if idx not in tool_call_chunks:
                        tool_call_chunks[idx] = {"id": "", "name": "", "args": ""}
                    entry = tool_call_chunks[idx]
                    if tc_chunk.id:
                        entry["id"] += tc_chunk.id
                    if tc_chunk.function:
                        if tc_chunk.function.name:
                            entry["name"] += tc_chunk.function.name
                        if tc_chunk.function.arguments:
                            entry["args"] += tc_chunk.function.arguments
                    current_phase = "tool call"

                if c.finish_reason:
                    finish_reason = c.finish_reason

                if hasattr(chunk, "usage") and chunk.usage:
                    token_state["tokens_in"] += chunk.usage.prompt_tokens or 0
                    token_state["tokens_out"] += chunk.usage.completion_tokens or 0
                    stream_completion_tokens = max(stream_completion_tokens, chunk.usage.completion_tokens or 0)

                render_status(current_phase)

            render_status(current_phase, force=True)
            print()

            # Reconstruct tool_calls from chunks
            tool_calls = [
                types.SimpleNamespace(
                    id=v["id"],
                    function=types.SimpleNamespace(name=v["name"], arguments=v["args"]),
                )
                for v in tool_call_chunks.values()
            ] if tool_call_chunks else []

            append_trace({
                "type": "assistant",
                "content": full_content,
                "tool_calls": [
                    {"id": tc.id, "name": tc.function.name, "args": tc.function.arguments}
                    for tc in tool_calls
                ],
            })

            # Fake message/choice objects for the rest of the loop
            msg = types.SimpleNamespace(content=full_content, tool_calls=tool_calls or None)
            choice = types.SimpleNamespace(finish_reason=finish_reason, message=msg)

            # LiteLLM message objects need to be converted for re-use in messages list
            msg_dict: dict = {"role": "assistant"}
            if msg.content:
                msg_dict["content"] = msg.content
            if msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(msg_dict)

            if not msg.tool_calls:
                # Check if the task looks complete: server running on port 8181
                if port_accepts_connections(8181) and listener_matches_process_groups(8181, active_process_groups):
                    # Server is up — task is done
                    stats["finish_reason"] = choice.finish_reason
                    logger.info("Agent finished with server running: %s", choice.finish_reason)
                    break

                if nudge_count < MAX_NUDGES:
                    nudge_count += 1
                    workspace_has_files = _workspace_has_started_work(workspace)
                    if workspace_has_files:
                        if (workspace / "pyproject.toml").exists():
                            nudge_msg = (
                                "The project is already initialized, so do not run `uv init` again. "
                                "Use the bash tool to finish creating the application files, install any missing packages, "
                                "and start the server on port 8181 with `uv run ...`."
                            )
                        else:
                            nudge_msg = (
                                "The workspace already contains files. "
                                "Use the bash tool to finish the application and start the server on port 8181. "
                                "Do not restart setup from scratch."
                            )
                    else:
                        nudge_msg = (
                            "You haven't created any files yet. "
                            "Use the bash tool RIGHT NOW to start building the application. "
                            "Begin with: uv init . && uv add flask"
                        )
                    logger.warning("Model returned without completing task — nudging (%d/%d)", nudge_count, MAX_NUDGES)
                    messages.append({"role": "user", "content": nudge_msg})
                    continue

                stats["finish_reason"] = choice.finish_reason
                logger.info("Agent finished: %s", choice.finish_reason)
                break

            # Execute tool calls
            tool_results: list[dict] = []
            for tc in msg.tool_calls:
                token_state["tool_calls"] += 1
                fn_name = tc.function.name.strip()
                try:
                    fn_args = _parse_tool_arguments(tc.function.arguments)
                except ValueError as exc:
                    invalid_tool_count += 1
                    logger.warning("Model sent invalid tool arguments (%d/%d): %s",
                                   invalid_tool_count, MAX_INVALID_TOOLS, exc)
                    if invalid_tool_count >= MAX_INVALID_TOOLS:
                        logger.error("Too many invalid tool calls — aborting agent")
                        stats["finish_reason"] = "invalid_tool_loop"
                        stats["agent_stop"] = {
                            "category": "invalid_tool_loop",
                            "detail": f"{invalid_tool_count} invalid tool calls; last: {exc}",
                        }
                        return
                    output = f"Error: invalid tool arguments. {exc}. Use the bash tool with JSON like {{\"command\": \"pwd\"}}."
                    append_trace({
                        "type": "tool_result",
                        "call_id": tc.id,
                        "command": "[invalid tool arguments]",
                        "output": output,
                    })
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                    })
                    continue

                if fn_name != "bash":
                    invalid_tool_count += 1
                    logger.warning("Model called unknown tool '%s' (%d/%d) — returning error",
                                   fn_name, invalid_tool_count, MAX_INVALID_TOOLS)
                    if invalid_tool_count >= MAX_INVALID_TOOLS:
                        logger.error("Too many invalid tool calls — aborting agent")
                        stats["finish_reason"] = "invalid_tool_loop"
                        stats["agent_stop"] = {
                            "category": "invalid_tool_loop",
                            "detail": f"{invalid_tool_count} calls to unknown tools; last: {fn_name!r}",
                        }
                        return
                    output = f"Error: unknown tool '{fn_name}'. The only available tool is 'bash'. Use bash to run shell commands."
                    append_trace({
                        "type": "tool_result",
                        "call_id": tc.id,
                        "command": f"[invalid tool: {fn_name}]",
                        "output": output,
                    })
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                    })
                    continue

                command = fn_args.get("command", "")
                if not isinstance(command, str) or not command.strip():
                    invalid_tool_count += 1
                    logger.warning("Model omitted string 'command' argument (%d/%d)",
                                   invalid_tool_count, MAX_INVALID_TOOLS)
                    if invalid_tool_count >= MAX_INVALID_TOOLS:
                        logger.error("Too many invalid tool calls — aborting agent")
                        stats["finish_reason"] = "invalid_tool_loop"
                        stats["agent_stop"] = {
                            "category": "invalid_tool_loop",
                            "detail": f"{invalid_tool_count} invalid tool calls; last: missing command arg",
                        }
                        return
                    output = "Error: bash tool requires a non-empty string 'command' argument."
                    append_trace({
                        "type": "tool_result",
                        "call_id": tc.id,
                        "command": "[missing command]",
                        "output": output,
                    })
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                    })
                    continue
                if _contains_redundant_uv_init(command, workspace):
                    redundant_uv_init_streak += 1
                else:
                    redundant_uv_init_streak = 0
                command_signature = " ".join(command.split())
                if command_signature == last_command_signature:
                    repeated_command_streak += 1
                else:
                    last_command_signature = command_signature
                    repeated_command_streak = 1
                written_file = _written_file_target(command)
                if written_file is not None:
                    if written_file == last_written_file:
                        repeated_file_write_streak += 1
                    else:
                        last_written_file = written_file
                        repeated_file_write_streak = 1
                else:
                    last_written_file = None
                    repeated_file_write_streak = 0

                if redundant_uv_init_streak > MAX_REDUNDANT_UV_INIT_STREAK:
                    output = (
                        "Error: detected a redundant `uv init` loop more than 5 times in a row. "
                        "The project is already initialized, and this run is being aborted as stuck."
                    )
                    append_trace({
                        "type": "tool_result",
                        "call_id": tc.id,
                        "command": command,
                        "output": output,
                    })
                    logger.error("Aborting run after %d consecutive redundant uv init attempts",
                                 redundant_uv_init_streak)
                    stats["finish_reason"] = "redundant_uv_init_loop"
                    stats["agent_stop"] = {
                        "category": "redundant_uv_init_loop",
                        "detail": f"{redundant_uv_init_streak} consecutive redundant `uv init` attempts",
                    }
                    return
                if repeated_command_streak > MAX_REPEATED_COMMAND_STREAK:
                    output = (
                        "Error: detected the same command more than 5 times in a row. "
                        "This run is being aborted as stuck."
                    )
                    append_trace({
                        "type": "tool_result",
                        "call_id": tc.id,
                        "command": command,
                        "output": output,
                    })
                    logger.error("Aborting run after %d consecutive identical commands: %s",
                                 repeated_command_streak, command_signature)
                    stats["finish_reason"] = "repeated_command_loop"
                    stats["agent_stop"] = {
                        "category": "repeated_command_loop",
                        "detail": f"{repeated_command_streak}x `{command_signature[:120]}`",
                    }
                    return
                if repeated_file_write_streak > MAX_REPEATED_FILE_WRITE_STREAK:
                    output = (
                        f"Error: detected the same file being overwritten more than {MAX_REPEATED_FILE_WRITE_STREAK} times in a row "
                        f"({written_file}). This run is being aborted as stuck."
                    )
                    append_trace({
                        "type": "tool_result",
                        "call_id": tc.id,
                        "command": command,
                        "output": output,
                    })
                    logger.error(
                        "Aborting run after %d consecutive overwrites of %s",
                        repeated_file_write_streak,
                        written_file,
                    )
                    stats["finish_reason"] = "repeated_file_write_loop"
                    stats["agent_stop"] = {
                        "category": "repeated_file_write_loop",
                        "detail": f"{repeated_file_write_streak}x overwrite of `{written_file}`",
                    }
                    return
                print(f"$ {command}", flush=True)
                output = await _run_bash(command, workspace, active_process_groups)
                print(f"-> {_summarize_command_output(output)}", flush=True)
                logger.info("tool[%d] bash done", token_state["tool_calls"])

                append_trace({
                    "type": "tool_result",
                    "call_id": tc.id,
                    "command": command,
                    "output": output[:2000],
                })

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output,
                })

            messages.extend(tool_results)

    try:
        await asyncio.wait_for(_loop(), timeout=timeout)
    except asyncio.TimeoutError:
        stats["timed_out"] = True
        stats["finish_reason"] = "timeout"
        stats["agent_stop"] = {"category": "timeout", "detail": f"no completion within {timeout}s"}
        logger.warning("Agent timed out after %ds", timeout)
    except Exception as exc:
        stats["error"] = str(exc)
        stats["finish_reason"] = f"error"
        stats["agent_stop"] = {"category": "agent_exception", "detail": str(exc)}
        logger.error("Agent error: %s", exc, exc_info=True)
    finally:
        await terminate_process_groups(active_process_groups)
        save_trace()

    return stats
