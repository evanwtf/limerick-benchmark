"""Shared named model-set definitions for benchmark and prefetch CLIs."""

from __future__ import annotations

NAMED_MODEL_SETS: dict[str, str] = {
    "poc": "poc",
    "v1": "v1",
    "recommended": "recommended",
    "qwen-coding": "qwen_coding",
}

BENCHMARK_SET_CHOICES: tuple[str, ...] = (*NAMED_MODEL_SETS.keys(), "local", "reference")
PREFETCH_SET_CHOICES: tuple[str, ...] = (*NAMED_MODEL_SETS.keys(), "all")


def format_set_metavar(choices: tuple[str, ...]) -> str:
    return "{" + ",".join(choices) + "}"
