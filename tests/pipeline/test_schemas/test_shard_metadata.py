"""Behavioral tests for the ShardMetadata pydantic model.

ShardMetadata is the strict sidecar payload for wds tar shards (member
``metadata.json``). It lives in a leaf module with no project imports so
consumers on either side of the ``src/`` ↔ ``src/pipeline/`` boundary can
import it without picking up transitive dependencies that would form an
import cycle.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

import pytest
from pydantic import ValidationError

from src.pipeline.schemas.shard_metadata import ShardMetadata


def _valid_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "velocity": 100,
        "signal_duration_seconds": 4.0,
        "sample_rate": 16000,
        "channels": 2,
        "min_loudness": -55.0,
    }
    kwargs.update(overrides)
    return kwargs


class TestShardMetadataConstruction:
    """Tests for ShardMetadata model construction and round-trip."""

    def test_valid_payload_constructs(self) -> None:
        """A payload mirroring the audio HDF5 attrs constructs cleanly."""
        meta = ShardMetadata(**_valid_kwargs())
        assert meta.velocity == 100
        assert meta.signal_duration_seconds == 4.0
        assert meta.sample_rate == 16000
        assert meta.channels == 2
        assert meta.min_loudness == -55.0

    def test_json_round_trip_preserves_values(self) -> None:
        """``model_dump_json`` → ``model_validate_json`` round-trips identity."""
        original = ShardMetadata(**_valid_kwargs())
        rebuilt = ShardMetadata.model_validate_json(original.model_dump_json())
        assert rebuilt == original


class TestShardMetadataStrictness:
    """Tests for the strict / frozen / extra=forbid model config."""

    def test_is_frozen(self) -> None:
        """Mutating a field after construction raises ValidationError."""
        meta = ShardMetadata(**_valid_kwargs())
        with pytest.raises(ValidationError):
            meta.velocity = 99  # type: ignore[misc]

    def test_extra_fields_rejected(self) -> None:
        """Unknown keys raise — sidecar shape is fixed."""
        with pytest.raises(ValidationError):
            ShardMetadata(**_valid_kwargs(), extra="oops")  # type: ignore[call-arg]

    def test_missing_required_field_raises(self) -> None:
        """A missing required field raises rather than defaulting."""
        kwargs = _valid_kwargs()
        del kwargs["velocity"]
        with pytest.raises(ValidationError):
            ShardMetadata(**kwargs)

    def test_strict_mode_rejects_string_for_int_field(self) -> None:
        """Strict mode forbids string→int coercion at the trust boundary."""
        with pytest.raises(ValidationError):
            ShardMetadata(**_valid_kwargs(velocity="100"))

    def test_malformed_sidecar_json_raises(self) -> None:
        """A malformed ``metadata.json`` (missing field) fails loudly on validate."""
        payload = json.dumps({"velocity": 100, "channels": 2})  # incomplete
        with pytest.raises(ValidationError):
            ShardMetadata.model_validate_json(payload)


class TestShardMetadataLeafImport:
    """The model lives in a leaf module so consumers can import it without cycles."""

    def test_module_has_no_project_imports(self) -> None:
        """Parse the module's AST and assert no ``src.*`` imports exist.

        The leaf-module guarantee matters because ``generate_vst_dataset`` (a
        src→pipeline consumer) imports this; if the module pulled in
        ``src.pipeline.schemas.spec`` or another non-leaf, the import graph
        would form a cycle through ``param_specs`` / pedalboard. Parsing the
        AST rather than substring-matching the source defends against
        false negatives (a docstring mentioning ``from src.``) and false
        positives (alternative import phrasings).
        """
        import ast
        from pathlib import Path

        module = importlib.import_module("src.pipeline.schemas.shard_metadata")
        source = module.__file__
        assert source is not None
        tree = ast.parse(Path(source).read_text(encoding="utf-8"))
        project_imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                project_imports.extend(
                    alias.name for alias in node.names if alias.name.startswith("src.")
                )
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("src."):
                    project_imports.append(node.module)
        assert project_imports == [], f"leaf module pulled project imports: {project_imports}"
