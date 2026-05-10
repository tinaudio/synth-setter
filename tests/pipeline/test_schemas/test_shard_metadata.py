from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import pipeline.schemas.shard_metadata as shard_metadata_module
from pipeline.schemas.shard_metadata import ShardMetadata


def _valid_kwargs() -> dict[str, Any]:
    """Return a baseline kwarg dict that constructs a valid ShardMetadata."""
    return {
        "velocity": 100,
        "signal_duration_seconds": 4.0,
        "sample_rate": 16000.0,
        "channels": 2,
        "min_loudness": -55.0,
    }


class TestShardMetadata:
    """Behavioral contracts for the ShardMetadata model that pins shard sidecar JSON.

    The class lives in ``pipeline.schemas.shard_metadata`` (a leaf module with no
    project imports) so the renderer in ``src/data/vst/generate_vst_dataset.py`` can
    import it without forming a cycle with ``pipeline.schemas.spec`` (which imports
    ``src.data.vst.param_specs``).
    """

    def test_valid_kwargs_construct(self) -> None:
        """A complete kwargs dict constructs a model with the supplied field values."""
        meta = ShardMetadata(**_valid_kwargs())
        assert meta.velocity == 100
        assert meta.channels == 2

    def test_strict_rejects_string_for_int_velocity(self) -> None:
        """Strict=True rejects coercion of "100" → int for the velocity field."""
        kwargs = _valid_kwargs()
        kwargs["velocity"] = "100"
        with pytest.raises(ValidationError):
            ShardMetadata(**kwargs)

    def test_missing_required_field_raises(self) -> None:
        """Omitting a required field raises ValidationError (no defaults exist)."""
        kwargs = _valid_kwargs()
        del kwargs["channels"]
        with pytest.raises(ValidationError):
            ShardMetadata(**kwargs)

    def test_extra_fields_rejected(self) -> None:
        """Extra="forbid" rejects unknown sidecar keys so the schema stays pinned."""
        kwargs = _valid_kwargs()
        kwargs["bonus"] = "field"
        with pytest.raises(ValidationError):
            ShardMetadata(**kwargs)

    def test_json_round_trip(self) -> None:
        """model_dump_json() and model_validate_json() are exact inverses."""
        original = ShardMetadata(**_valid_kwargs())
        restored = ShardMetadata.model_validate_json(original.model_dump_json())
        assert restored == original

    @pytest.mark.parametrize("velocity", [-1, 128, 200])
    def test_velocity_outside_midi_range_rejected(self, velocity: int) -> None:
        """A metadata.json carrying a velocity outside [0, 127] fails on read."""
        kwargs = _valid_kwargs()
        kwargs["velocity"] = velocity
        with pytest.raises(ValidationError, match="velocity"):
            ShardMetadata(**kwargs)

    @pytest.mark.parametrize("duration", [0.0, -1.0, -100.0])
    def test_non_positive_signal_duration_rejected(self, duration: float) -> None:
        """A non-positive signal_duration_seconds fails on read."""
        kwargs = _valid_kwargs()
        kwargs["signal_duration_seconds"] = duration
        with pytest.raises(ValidationError, match="signal_duration_seconds"):
            ShardMetadata(**kwargs)

    @pytest.mark.parametrize("sample_rate", [0.0, -1.0, -16000.0])
    def test_non_positive_sample_rate_rejected(self, sample_rate: float) -> None:
        """A non-positive sample_rate fails on read."""
        kwargs = _valid_kwargs()
        kwargs["sample_rate"] = sample_rate
        with pytest.raises(ValidationError, match="sample_rate"):
            ShardMetadata(**kwargs)

    @pytest.mark.parametrize("channels", [0, -1, -2])
    def test_channels_below_one_rejected(self, channels: int) -> None:
        """A channels value below 1 fails on read."""
        kwargs = _valid_kwargs()
        kwargs["channels"] = channels
        with pytest.raises(ValidationError, match="channels"):
            ShardMetadata(**kwargs)


class TestShardMetadataLeafModuleInvariant:
    """The leaf module must stay leaf — no project imports allowed — so the renderer in
    ``src/data/vst/generate_vst_dataset.py`` can import ``ShardMetadata`` without pulling in
    ``pipeline.schemas.config`` and ``src.data.vst.param_specs`` (which would form an import cycle
    with the renderer itself)."""

    def test_leaf_module_imports_no_project_modules(self) -> None:
        """The leaf module's source AST contains no ``pipeline.``/``src.``/``scripts.`` imports."""
        module_file = shard_metadata_module.__file__
        assert module_file is not None
        module_path = Path(module_file)
        tree = ast.parse(module_path.read_text())

        project_prefixes = ("pipeline.", "src.", "scripts.")
        offenders: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(project_prefixes):
                    offenders.append(f"from {node.module} import …")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(project_prefixes):
                        offenders.append(f"import {alias.name}")

        assert offenders == [], f"leaf module imports project modules: {offenders}"

    def test_spec_re_exports_shard_metadata(self) -> None:
        """``pipeline.schemas.spec.ShardMetadata`` is the same class as the leaf import."""
        import pipeline.schemas.spec as spec_module
        from pipeline.schemas.shard_metadata import ShardMetadata as Leaf

        assert spec_module.ShardMetadata is Leaf
