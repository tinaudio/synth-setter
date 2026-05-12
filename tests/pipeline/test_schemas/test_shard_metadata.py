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


class TestShardMetadataRangeValidators:
    """Tests for the ``_ranges_must_be_sane`` model_validator (trust-boundary defense)."""

    @pytest.mark.parametrize("bad_velocity", [-1, 128, 200])
    def test_velocity_outside_midi_range_raises(self, bad_velocity: int) -> None:
        """Velocity outside [0, 127] is rejected — mirrors RenderConfig."""
        with pytest.raises(ValidationError, match="velocity must be in"):
            ShardMetadata(**_valid_kwargs(velocity=bad_velocity))

    @pytest.mark.parametrize("bad_duration", [0.0, -1.0])
    def test_non_positive_signal_duration_raises(self, bad_duration: float) -> None:
        """signal_duration_seconds must be > 0."""
        with pytest.raises(ValidationError, match="signal_duration_seconds must be positive"):
            ShardMetadata(**_valid_kwargs(signal_duration_seconds=bad_duration))

    @pytest.mark.parametrize("bad_sample_rate", [0, -16000])
    def test_non_positive_sample_rate_raises(self, bad_sample_rate: int) -> None:
        """sample_rate must be > 0."""
        with pytest.raises(ValidationError, match="sample_rate must be positive"):
            ShardMetadata(**_valid_kwargs(sample_rate=bad_sample_rate))

    @pytest.mark.parametrize("bad_channels", [0, -1])
    def test_channels_less_than_one_raises(self, bad_channels: int) -> None:
        """Channels must be >= 1."""
        with pytest.raises(ValidationError, match="channels must be >= 1"):
            ShardMetadata(**_valid_kwargs(channels=bad_channels))


class TestShardMetadataLeafImport:
    """The model lives in a leaf module so consumers can import it without cycles."""

    def test_module_has_no_project_imports(self) -> None:
        """Parse the module's AST and assert it has no project-internal imports.

        The leaf-module guarantee matters because the wds writer side
        (``src.data.vst.generate_vst_dataset``, to be wired in PR-13) will
        import this model; if the module pulled in ``src.pipeline.schemas.spec``
        or another non-leaf, the import graph would form a cycle through
        ``param_specs`` / pedalboard. The check flags every form Python
        supports for reaching project code: ``import src``/``import
        src.x.y``, ``from src import x`` (module == "src"), ``from src.x.y
        import z`` (module starts with "src."), and any relative ``from .x
        import y`` (``node.level > 0``).
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
                    alias.name
                    for alias in node.names
                    if alias.name == "src" or alias.name.startswith("src.")
                )
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    project_imports.append(f"<relative-level-{node.level}>")
                elif node.module == "src" or (node.module or "").startswith("src."):
                    project_imports.append(node.module or "")
        assert project_imports == [], f"leaf module pulled project imports: {project_imports}"
