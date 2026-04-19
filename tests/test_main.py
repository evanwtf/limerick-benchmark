import sys
import unittest
from unittest import mock

import benchmark.__main__ as benchmark_main
from benchmark.agent import AIDER_STAGNATION_SECONDS


class MainCliTests(unittest.TestCase):
    def test_models_for_set_returns_qwen_coding_matchup(self) -> None:
        catalog = {
            "gemma4:e4b": {
                "id": "gemma4:e4b",
                "provider": "ollama",
                "qwen_coding": True,
            },
            "qwen3.6:35b-a3b-coding-mxfp8": {
                "id": "qwen3.6:35b-a3b-coding-mxfp8",
                "provider": "ollama",
                "qwen_coding": True,
            },
            "qwen3.5:35b-a3b-coding-mxfp8": {
                "id": "qwen3.5:35b-a3b-coding-mxfp8",
                "provider": "ollama",
                "qwen_coding": True,
            },
            "qwen3.5:35b-a3b": {
                "id": "qwen3.5:35b-a3b",
                "provider": "ollama",
                "recommended": True,
            },
        }

        models = benchmark_main.models_for_set(catalog, "qwen-coding", pulled=set())

        self.assertEqual(
            [model["id"] for model in models],
            [
                "gemma4:e4b",
                "qwen3.6:35b-a3b-coding-mxfp8",
                "qwen3.5:35b-a3b-coding-mxfp8",
            ],
        )

    def test_run_uses_default_aider_stagnation_timeout(self) -> None:
        with (
            mock.patch("benchmark.__main__.load_catalog", return_value={}),
            mock.patch("benchmark.__main__.get_pulled_names", return_value=set()),
            mock.patch("benchmark.__main__.preflight_check", return_value=True),
            mock.patch(
                "benchmark.__main__.run_benchmark",
                new=mock.Mock(return_value=[]),
            ) as run_benchmark_mock,
            mock.patch("benchmark.__main__.asyncio.run", return_value=[]),
            mock.patch.object(sys, "argv", ["benchmark", "run", "--model", "gemma4:e2b"]),
        ):
            benchmark_main.main()

        self.assertEqual(
            run_benchmark_mock.call_args.kwargs["aider_stagnation_timeout"],
            AIDER_STAGNATION_SECONDS,
        )

    def test_run_accepts_explicit_aider_stagnation_timeout_flag(self) -> None:
        with (
            mock.patch("benchmark.__main__.load_catalog", return_value={}),
            mock.patch("benchmark.__main__.get_pulled_names", return_value=set()),
            mock.patch("benchmark.__main__.preflight_check", return_value=True),
            mock.patch(
                "benchmark.__main__.run_benchmark",
                new=mock.Mock(return_value=[]),
            ) as run_benchmark_mock,
            mock.patch("benchmark.__main__.asyncio.run", return_value=[]),
            mock.patch.object(
                sys,
                "argv",
                [
                    "benchmark",
                    "run",
                    "--model",
                    "gemma4:e2b",
                    "--agent",
                    "aider",
                    "--aider-stagnation-timeout",
                    "420",
                ],
            ),
        ):
            benchmark_main.main()

        self.assertEqual(
            run_benchmark_mock.call_args.kwargs["aider_stagnation_timeout"],
            420,
        )

    def test_run_accepts_qwen_coding_set(self) -> None:
        catalog = {
            "gemma4:e4b": {
                "id": "gemma4:e4b",
                "provider": "ollama",
                "qwen_coding": True,
            },
            "qwen3.6:35b-a3b-coding-mxfp8": {
                "id": "qwen3.6:35b-a3b-coding-mxfp8",
                "provider": "ollama",
                "qwen_coding": True,
            },
            "qwen3.5:35b-a3b-coding-mxfp8": {
                "id": "qwen3.5:35b-a3b-coding-mxfp8",
                "provider": "ollama",
                "qwen_coding": True,
            },
            "gemma4:e2b": {
                "id": "gemma4:e2b",
                "provider": "ollama",
                "poc": True,
            },
        }
        pulled = {
            "gemma4:e4b",
            "qwen3.6:35b-a3b-coding-mxfp8",
            "qwen3.5:35b-a3b-coding-mxfp8",
        }

        with (
            mock.patch("benchmark.__main__.load_catalog", return_value=catalog),
            mock.patch("benchmark.__main__.get_pulled_names", return_value=pulled),
            mock.patch("benchmark.__main__.preflight_check", return_value=True),
            mock.patch(
                "benchmark.__main__.run_benchmark",
                new=mock.Mock(return_value=[]),
            ) as run_benchmark_mock,
            mock.patch("benchmark.__main__.asyncio.run", return_value=[]),
            mock.patch.object(sys, "argv", ["benchmark", "run", "--set", "qwen-coding"]),
        ):
            benchmark_main.main()

        self.assertEqual(
            [model["id"] for model in run_benchmark_mock.call_args.args[0]],
            [
                "gemma4:e4b",
                "qwen3.6:35b-a3b-coding-mxfp8",
                "qwen3.5:35b-a3b-coding-mxfp8",
            ],
        )
