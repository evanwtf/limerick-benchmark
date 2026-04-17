"""Agent loop: sends a task to a model, executes bash tool calls, records trace."""

import asyncio
import json
import logging
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import litellm

from .process_utils import (
    listener_matches_process_groups,
    port_accepts_connections,
    process_group_exists,
    terminate_process_groups,
)

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 900  # 15 minutes hard limit
CMD_TIMEOUT_SECONDS = 60  # per bash command
MAX_OUTPUT_CHARS = 8000  # truncate long command output
STATUS_REFRESH_SECONDS = 0.25
MAX_REDUNDANT_UV_INIT_STREAK = 5

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
    removed_uv_init = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("uv init"):
            removed_uv_init = True
            continue
        kept_lines.append(line)

    if not removed_uv_init:
        return command, None

    note = (
        "Project already initialized; skipped redundant `uv init`. "
        "Do not run `uv init` again. Proceed with `uv add ...`, create files, and start the server."
    )
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


async def run_agent(
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
    }
    nudge_count = 0
    MAX_NUDGES = 2
    invalid_tool_count = 0
    MAX_INVALID_TOOLS = 5
    active_process_groups: set[int] = set()
    agent_started_at = asyncio.get_running_loop().time()
    redundant_uv_init_streak = 0

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
        logger.warning("Agent timed out after %ds", timeout)
    except Exception as exc:
        stats["error"] = str(exc)
        stats["finish_reason"] = f"error"
        logger.error("Agent error: %s", exc, exc_info=True)
    finally:
        await terminate_process_groups(active_process_groups)
        save_trace()

    return stats
