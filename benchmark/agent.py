"""Agent loop: sends a task to a model, executes bash tool calls, records trace."""

import asyncio
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import litellm

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 900  # 15 minutes hard limit
CMD_TIMEOUT_SECONDS = 60  # per bash command
MAX_OUTPUT_CHARS = 8000  # truncate long command output

SYSTEM_PROMPT = """\
You are an expert Python developer working in a terminal. Your goal is to complete \
the given coding task using the bash tool. You can run any shell commands: create \
files, install packages, start and test servers, fix errors, etc.

Rules:
- Work in the current directory only.
- Use `uv` for all Python package management.
- Test that your work actually runs before declaring done.
- When you are confident the task is complete and working, stop calling tools \
and write a brief summary of what you built.
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
        litellm_model = f"ollama/{model_id}"
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

    def append_trace(entry: dict) -> None:
        entry["ts"] = _ts()
        trace.append(entry)

    def save_trace() -> None:
        with open(trace_path, "w") as f:
            for entry in trace:
                f.write(json.dumps(entry) + "\n")

    append_trace({"type": "task", "content": task_prompt})

    async def _loop() -> None:
        while True:
            token_state["api_calls"] += 1
            logger.info("API call #%d to %s", token_state["api_calls"], litellm_model)

            response = await litellm.acompletion(
                model=litellm_model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                **extra_kwargs,
            )

            if response.usage:
                token_state["tokens_in"] += response.usage.prompt_tokens or 0
                token_state["tokens_out"] += response.usage.completion_tokens or 0

            choice = response.choices[0]
            msg = choice.message

            append_trace({
                "type": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {"id": tc.id, "name": tc.function.name, "args": tc.function.arguments}
                    for tc in (msg.tool_calls or [])
                ],
            })

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
                # Model returned without using tools. If the workspace is still
                # empty this is a non-starter — push it once, then give up.
                workspace_has_files = any(workspace.rglob("*"))
                if not workspace_has_files and token_state["tool_calls"] == 0:
                    logger.warning("Model returned without using tools and workspace is empty — nudging")
                    messages.append({
                        "role": "user",
                        "content": (
                            "You haven't created any files yet. "
                            "Use the bash tool RIGHT NOW to start building the application. "
                            "Begin with: uv init . && uv add flask"
                        ),
                    })
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
                    # Model called a nonexistent tool — tell it what's available
                    logger.warning("Model called unknown tool '%s' — returning error", fn_name)
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
                logger.info("tool[%d] bash: %s", token_state["tool_calls"], command[:100])
                output = await _run_bash(command, workspace)
                logger.debug("tool output: %s", output[:200])

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
