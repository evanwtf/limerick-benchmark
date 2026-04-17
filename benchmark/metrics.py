"""Background system metrics collector: CPU, memory, GPU, thermals, fans."""

import csv
import logging
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)

SAMPLE_INTERVAL = 5.0  # seconds between samples

FIELDNAMES = [
    "timestamp",
    "elapsed_s",
    "cpu_percent",
    "memory_percent",
    "gpu_utilization_percent",
    "gpu_power_mw",
    "die_temp_c",
    "fan_rpm",
    "tokens_in",
    "tokens_out",
    "api_calls",
    "tool_calls",
]

# Patterns for powermetrics text output (Apple Silicon)
_RE_GPU_UTIL = re.compile(r"GPU HW active residency:\s+([\d.]+)%")
_RE_GPU_POWER = re.compile(r"GPU Power:\s+([\d.]+)\s*mW")
_RE_CPU_TEMP = re.compile(r"CPU die temperature:\s+([\d.]+)")
_RE_GPU_TEMP = re.compile(r"GPU die temperature:\s+([\d.]+)")
_RE_FAN = re.compile(r"Fan:\s+([\d.]+)\s*rpm", re.IGNORECASE)


def _sample_powermetrics() -> dict[str, float | None]:
    """Run one powermetrics sample and parse GPU/thermal data. Returns {} on failure."""
    try:
        result = subprocess.run(
            ["sudo", "powermetrics", "-n", "1", "-i", "500",
             "--samplers", "gpu_power,thermal,smc"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError) as exc:
        logger.debug("powermetrics unavailable: %s", exc)
        return {}

    if result.returncode != 0:
        logger.debug("powermetrics exited %d: %s", result.returncode, result.stderr[:200])
        return {}

    out = result.stdout
    data: dict[str, float | None] = {}

    m = _RE_GPU_UTIL.search(out)
    data["gpu_utilization_percent"] = float(m.group(1)) if m else None

    m = _RE_GPU_POWER.search(out)
    data["gpu_power_mw"] = float(m.group(1)) if m else None

    # Prefer CPU die temp; fall back to GPU die temp
    m = _RE_CPU_TEMP.search(out) or _RE_GPU_TEMP.search(out)
    data["die_temp_c"] = float(m.group(1)) if m else None

    m = _RE_FAN.search(out)
    data["fan_rpm"] = float(m.group(1)) if m else None

    return data


class MetricsCollector:
    """
    Samples system metrics every SAMPLE_INTERVAL seconds in a background thread
    and writes rows to a CSV file.

    Pass token_state (a shared dict with tokens_in/tokens_out/api_calls/tool_calls)
    so each CSV row captures live token progress.
    """

    def __init__(self, csv_path: Path, *, enable_hardware_metrics: bool = False) -> None:
        self._csv_path = csv_path
        self._enable_hardware_metrics = enable_hardware_metrics
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time: float = 0.0
        self._token_state: dict[str, Any] = {}

    def start(self, token_state: dict[str, Any]) -> None:
        self._token_state = token_state
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="metrics")
        self._thread.start()
        logger.debug("MetricsCollector started → %s", self._csv_path)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)
        logger.debug("MetricsCollector stopped")

    def _loop(self) -> None:
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

            while not self._stop.wait(SAMPLE_INTERVAL):
                row = self._sample()
                writer.writerow(row)
                f.flush()

            # Final sample after stop
            writer.writerow(self._sample())

    def _sample(self) -> dict[str, Any]:
        elapsed = round(time.time() - self._start_time, 1)
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        pm = _sample_powermetrics() if self._enable_hardware_metrics else {}

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s": elapsed,
            "cpu_percent": cpu,
            "memory_percent": mem,
            "gpu_utilization_percent": pm.get("gpu_utilization_percent"),
            "gpu_power_mw": pm.get("gpu_power_mw"),
            "die_temp_c": pm.get("die_temp_c"),
            "fan_rpm": pm.get("fan_rpm"),
            "tokens_in": self._token_state.get("tokens_in", 0),
            "tokens_out": self._token_state.get("tokens_out", 0),
            "api_calls": self._token_state.get("api_calls", 0),
            "tool_calls": self._token_state.get("tool_calls", 0),
        }
