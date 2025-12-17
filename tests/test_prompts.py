"""Snapshot tests for llmwalk prompts."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import pytest
from syrupy.assertion import SnapshotAssertion
from syrupy.extensions.json import JSONSnapshotExtension

from llmwalk.cli import main

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
DEFAULT_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
MISTRAL_MODEL = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"

try:
    from huggingface_hub.errors import (
        LocalEntryNotFoundError as HFLocalEntryNotFoundError,
    )
except Exception:  # pragma: no cover - optional dependency details
    HFLocalEntryNotFoundError = None  # type: ignore[assignment]


class JSONSnapshotExtensionPretty(JSONSnapshotExtension):
    """JSON snapshot extension with pretty printing."""

    def serialize(self, data, **kwargs):
        return json.dumps(data, indent=2, sort_keys=True)


@pytest.fixture
def snapshot_json(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(JSONSnapshotExtensionPretty)


def get_prompt_files() -> list[Path]:
    """Get all prompt files from the prompts directory."""
    return sorted(PROMPTS_DIR.glob("*.txt"))


def get_models() -> list[str]:
    return [DEFAULT_MODEL, MISTRAL_MODEL]


def _should_skip_missing_model_error(exc: Exception) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True
    if HFLocalEntryNotFoundError is not None and isinstance(
        exc, HFLocalEntryNotFoundError
    ):
        return True
    return False


@pytest.fixture(autouse=True)
def purge_model_memory():
    yield
    gc.collect()
    try:
        import mlx.core as mx

        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass


@pytest.mark.parametrize(
    "prompt_file",
    get_prompt_files(),
    ids=[p.stem for p in get_prompt_files()],
)
@pytest.mark.parametrize(
    "model",
    get_models(),
    ids=["default", "mistral"],
)
def test_prompt_snapshot(
    prompt_file: Path,
    model: str,
    snapshot_json: SnapshotAssertion,
    capsys,
) -> None:
    """Test that running llmwalk on each prompt produces consistent output."""
    # Run with JSON format for deterministic output
    try:
        main(
            [
                "-p",
                str(prompt_file),
                "--format",
                "json",
                "--model",
                model,
                "--offline",
                "-n",
                "3",
                "--top-p",
                "0.5",
                "--top-k",
                "10",
            ]
        )
    except Exception as e:
        if _should_skip_missing_model_error(e):
            pytest.skip(f"Model not available locally: {model}")
        raise

    captured = capsys.readouterr()
    output = json.loads(captured.out)

    # Snapshot the structured output
    assert output == snapshot_json
