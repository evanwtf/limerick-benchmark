"""Post-run evaluation: start the generated server, check HTTP 200, write run.sh."""

from __future__ import annotations

import asyncio
import contextlib
import html
import logging
import re
import stat
import time
import tomllib
from pathlib import Path
from typing import Any

import aiohttp

from .process_utils import (
    assert_port_available,
    listener_belongs_to_process_tree,
    sanitized_subprocess_env,
    terminate_process_group,
)

logger = logging.getLogger(__name__)

PORT = 8181
STARTUP_TIMEOUT = 30  # seconds to wait for server to come up
POLL_INTERVAL = 1.0
CANONICAL_ENTRY_POINT = "uv run python app.py"
PYTHON_SCAN_MAX_BYTES = 64 * 1024

_REFRESH_RE = re.compile(r"<meta[^>]+http-equiv\s*=\s*[\"']?refresh\b|setInterval\s*\(", re.IGNORECASE)
_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style)\b.*?</\1>")
_BLOCK_BREAK_RE = re.compile(r"(?i)<br\s*/?>|</p>|</div>|</li>|</h[1-6]>|</pre>|</tr>")
_TAG_RE = re.compile(r"<[^>]+>")


def _empty_eval_result(
    *,
    entry_point: str | None = None,
    entry_point_candidates: list[str] | None = None,
    entry_point_mismatch: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "entry_point": entry_point,
        "entry_point_candidates": entry_point_candidates or [],
        "entry_point_mismatch": entry_point_mismatch,
        "server_started": False,
        "http_status": None,
        "response_bytes": None,
        "body_has_refresh_mechanism": False,
        "body_has_limerick_shape": False,
        "startup_seconds": None,
        "passed": False,
        "error": error,
    }


def _script_commands_from_pyproject(workspace: Path) -> list[str]:
    pyproject = workspace / "pyproject.toml"
    if not pyproject.exists():
        return []

    try:
        data = tomllib.loads(pyproject.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return []

    scripts = data.get("project", {}).get("scripts", {})
    if not isinstance(scripts, dict):
        return []
    return [f"uv run {name}" for name in scripts]


def _candidate_entry_points(workspace: Path) -> list[str]:
    """Return plausible commands to start the generated app."""
    candidates: list[str] = []

    def add(command: str) -> None:
        if command not in candidates:
            candidates.append(command)

    run_sh = workspace / "run.sh"
    if run_sh.exists():
        add("bash run.sh")

    for command in _script_commands_from_pyproject(workspace):
        add(command)

    search_roots = [workspace]
    src_dir = workspace / "src"
    if src_dir.exists():
        search_roots.append(src_dir)

    for root in search_roots:
        for name in ("app.py", "main.py", "server.py", "web.py"):
            py = root / name
            if py.exists():
                add(f"uv run python {py.relative_to(workspace)}")

        for package_main in sorted(root.glob("*/__main__.py")):
            package_dir = package_main.parent
            if not package_dir.name.isidentifier():
                continue
            if root == src_dir:
                add(f"uv run python -m {package_dir.name}")
            else:
                module = package_dir.relative_to(workspace).as_posix().replace("/", ".")
                add(f"uv run python -m {module}")

        python_files = sorted(root.glob("*.py"))
        for py in python_files:
            if _python_file_contains_entrypoint_markers(py):
                add(f"uv run python {py.relative_to(workspace)}")

        if len(python_files) == 1:
            add(f"uv run python {python_files[0].relative_to(workspace)}")

    return candidates


def _python_file_contains_entrypoint_markers(py: Path) -> bool:
    """Scan just the start of a Python file for common Flask entry-point markers."""
    bytes_read = 0
    try:
        with py.open("r", errors="replace") as f:
            while bytes_read < PYTHON_SCAN_MAX_BYTES:
                line = f.readline(PYTHON_SCAN_MAX_BYTES - bytes_read)
                if not line:
                    break
                bytes_read += len(line.encode("utf-8", errors="replace"))
                if "Flask" in line or "app.run" in line:
                    return True
    except OSError:
        return False
    return False


def _extract_body_text_lines(body_text: str) -> list[str]:
    cleaned = _SCRIPT_STYLE_RE.sub(" ", body_text)
    cleaned = _BLOCK_BREAK_RE.sub("\n", cleaned)
    cleaned = _TAG_RE.sub(" ", cleaned)
    cleaned = html.unescape(cleaned)
    lines = [" ".join(line.split()) for line in cleaned.splitlines()]
    return [line for line in lines if line]


def _limerick_first_lines(workspace: Path) -> list[str]:
    limericks = workspace / "limericks.txt"
    if not limericks.exists():
        return []
    try:
        text = limericks.read_text()
    except OSError:
        return []

    first_lines: list[str] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if lines:
            first_lines.append(lines[0])
    return first_lines


def _body_has_refresh_mechanism(body_text: str) -> bool:
    return bool(_REFRESH_RE.search(body_text))


def _body_has_limerick_shape(body_text: str, workspace: Path) -> bool:
    lines = _extract_body_text_lines(body_text)
    if len(lines) >= 5:
        return True

    normalized_body = " ".join(lines)
    return any(first_line in normalized_body for first_line in _limerick_first_lines(workspace))


def _classify_http_response(http_status: int | None, body: bytes | None, workspace: Path) -> dict[str, Any]:
    result = {
        "body_has_refresh_mechanism": False,
        "body_has_limerick_shape": False,
        "passed": False,
        "error": None,
    }
    if http_status != 200 or body is None:
        return result

    body_text = body.decode("utf-8", errors="replace")
    has_refresh = _body_has_refresh_mechanism(body_text)
    has_limerick = _body_has_limerick_shape(body_text, workspace)
    result["body_has_refresh_mechanism"] = has_refresh
    result["body_has_limerick_shape"] = has_limerick

    if not has_refresh:
        result["error"] = "body_missing_refresh"
    elif not has_limerick:
        result["error"] = "body_missing_limerick"
    else:
        result["passed"] = True
    return result


async def _wait_for_port(port: int, timeout: float) -> bool:
    """Poll localhost:port until it accepts connections. Returns True if up."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        try:
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            await writer.wait_closed()
            return True
        except (ConnectionRefusedError, OSError):
            await asyncio.sleep(POLL_INTERVAL)
    return False


async def _try_entry_point(workspace: Path, entry_cmd: str) -> dict[str, Any]:
    """Start one candidate entry point and return the evaluation result."""
    assert_port_available(PORT, f"starting evaluator command '{entry_cmd}'")

    result = _empty_eval_result(entry_point=entry_cmd)

    proc = await asyncio.create_subprocess_shell(
        entry_cmd,
        cwd=workspace,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env=sanitized_subprocess_env(),
        start_new_session=True,
    )

    try:
        startup_started = time.monotonic()
        up = await _wait_for_port(PORT, STARTUP_TIMEOUT)
        if not up:
            logger.warning("Server did not come up on port %d within %ds for '%s'", PORT, STARTUP_TIMEOUT, entry_cmd)
            result["error"] = "port_never_opened"
            return result

        if not listener_belongs_to_process_tree(PORT, proc.pid):
            logger.warning("Port %d listener was not started by '%s'", PORT, entry_cmd)
            result["error"] = "unexpected_listener"
            return result

        result["server_started"] = True
        logger.info("Server up on port %d via '%s'", PORT, entry_cmd)

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"http://localhost:{PORT}/",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    body = await resp.read()
                    result["http_status"] = resp.status
                    result["response_bytes"] = len(body)
                    result["startup_seconds"] = round(time.monotonic() - startup_started, 1)
                    result.update(_classify_http_response(resp.status, body, workspace))
                    logger.info("GET / → %d (%d bytes)", resp.status, len(body))
            except Exception as exc:
                result["error"] = f"http_error: {exc}"
                logger.warning("HTTP check failed for '%s': %s", entry_cmd, exc)
    finally:
        await terminate_process_group(proc.pid)
        with contextlib.suppress(RuntimeError):
            assert_port_available(PORT, f"cleaning up evaluator command '{entry_cmd}'")

    return result


async def evaluate(workspace: Path, results_dir: Path) -> dict[str, Any]:
    """
    Try to start the generated server and check it responds with HTTP 200.
    Writes run.sh to results_dir for later manual evaluation.
    Returns an evaluation dict.
    """
    entry_points = _candidate_entry_points(workspace)
    if not entry_points:
        logger.warning("No entry point found in workspace")
        result = _empty_eval_result(error="no_entry_point")
        _write_run_sh(results_dir, workspace, None)
        return result

    canonical_entry_point = CANONICAL_ENTRY_POINT if (workspace / "app.py").exists() else None
    if canonical_entry_point is None:
        logger.warning("Workspace has non-canonical entry points but no app.py: %s", entry_points)
        result = _empty_eval_result(
            entry_point=entry_points[0],
            entry_point_candidates=entry_points,
            entry_point_mismatch=True,
            error="entry_point_mismatch",
        )
        _write_run_sh(results_dir, workspace, entry_points[0])
        return result

    result = await _try_entry_point(workspace, canonical_entry_point)
    result["entry_point_candidates"] = entry_points
    _write_run_sh(results_dir, workspace, canonical_entry_point)
    return result


def _write_run_sh(results_dir: Path, workspace: Path, entry_cmd: str | None) -> None:
    """Write a convenience run.sh to the results directory."""
    run_sh = results_dir / "run.sh"
    if entry_cmd:
        content = f"#!/bin/sh\ncd '{workspace}'\n{entry_cmd}\n"
    else:
        content = (
            "#!/bin/sh\n"
            "# No entry point was detected — inspect the workspace manually.\n"
            f"echo 'Workspace: {workspace}'\n"
            f"ls '{workspace}'\n"
        )
    run_sh.write_text(content)
    run_sh.chmod(run_sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
