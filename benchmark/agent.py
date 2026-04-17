"""Agent loop: sends a task to a model, executes bash tool calls, records trace."""

import asyncio
import json
import logging
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import litellm

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 900  # 15 minutes hard limit
CMD_TIMEOUT_SECONDS = 60  # per bash command
MAX_OUTPUT_CHARS = 8000  # truncate long command output

SYSTEM_PROMPT = """\
You are an expert Python developer. Complete the coding task by calling the bash tool.

You have exactly ONE tool: bash. Call it to run shell commands.
Do NOT call any other tool name — only "bash" exists.

Rules:
- Work in the current directory only.
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


async def _run_bash(command: str, workspace: Path) -> str:
    logger.debug("bash: %s", command[:120])
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=workspace,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=CMD_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            proc.kill()
            return f"[timeout after {CMD_TIMEOUT_SECONDS}s]"

        output = stdout.decode("utf-8", errors="replace")
        if len(output) > MAX_OUTPUT_CHARS:
            half = MAX_OUTPUT_CHARS // 2
            output = output[:half] + f"\n...[{len(output) - MAX_OUTPUT_CHARS} chars truncated]...\n" + output[-half:]
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

    def append_trace(entry: dict) -> None:
        entry["ts"] = _ts()
        trace.append(entry)

    def save_trace() -> None:
        with open(trace_path, "w") as f:
            for entry in trace:
                f.write(json.dumps(entry) + "\n")

    append_trace({"type": "task", "content": task_prompt})

    async def _loop() -> None:
        nonlocal nudge_count, invalid_tool_count
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

            print(f"\n[{litellm_model}] ", end="", flush=True)
            async for chunk in stream:
                c = chunk.choices[0]
                delta = c.delta

                thinking = getattr(delta, "reasoning_content", None)
                if thinking:
                    print(thinking, end="", flush=True)

                if delta.content:
                    print(delta.content, end="", flush=True)
                    full_content += delta.content

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

                if c.finish_reason:
                    finish_reason = c.finish_reason

                if hasattr(chunk, "usage") and chunk.usage:
                    token_state["tokens_in"] += chunk.usage.prompt_tokens or 0
                    token_state["tokens_out"] += chunk.usage.completion_tokens or 0

            print()  # newline after streamed content

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
                import socket
                def _port_open() -> bool:
                    try:
                        with socket.create_connection(("localhost", 8181), timeout=1):
                            return True
                    except OSError:
                        return False

                if _port_open():
                    # Server is up — task is done
                    stats["finish_reason"] = choice.finish_reason
                    logger.info("Agent finished with server running: %s", choice.finish_reason)
                    break

                if nudge_count < MAX_NUDGES:
                    nudge_count += 1
                    workspace_has_files = any(workspace.rglob("*.py"))
                    if workspace_has_files:
                        nudge_msg = (
                            "The server is not running yet on port 8181. "
                            "Use the bash tool to finish the application and start it. "
                            "Run `uv run python app.py` or equivalent to start the server."
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
                fn_args = json.loads(tc.function.arguments)

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
                print(f"\n$ {command}", flush=True)
                output = await _run_bash(command, workspace)
                print(output[:1000] if output else "(no output)", flush=True)
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
        save_trace()

    return stats
