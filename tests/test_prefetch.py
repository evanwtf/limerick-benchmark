from types import SimpleNamespace
from unittest import TestCase, mock

import prefetch


class PrefetchTests(TestCase):
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
            "claude-opus-4-7": {
                "id": "claude-opus-4-7",
                "provider": "anthropic",
                "qwen_coding": True,
            },
        }

        models = prefetch.models_for_set(catalog, "qwen-coding")

        self.assertEqual(
            [model["id"] for model in models],
            [
                "gemma4:e4b",
                "qwen3.6:35b-a3b-coding-mxfp8",
                "qwen3.5:35b-a3b-coding-mxfp8",
            ],
        )

    def test_pull_model_uses_subprocess(self) -> None:
        with mock.patch("prefetch.subprocess.run", return_value=SimpleNamespace(returncode=0)) as run_mock:
            self.assertTrue(prefetch.pull_model("gemma4:e2b"))
            run_mock.assert_called_once_with(["ollama", "pull", "gemma4:e2b"])
