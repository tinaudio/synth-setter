from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

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


class TestShardMetadataLeafModule:
    """ShardMetadata lives in its own leaf module so pipeline-side and src-side consumers can
    import it without dragging spec.py's ``src.data.vst`` import (which would create a cycle for
    the src->pipeline import in ``generate_vst_dataset``)."""

    def test_imports_from_leaf_module(self) -> None:
        """ShardMetadata imports successfully from the leaf module path."""
        meta = ShardMetadata(**_valid_kwargs())
        assert meta.velocity == 100

    def test_leaf_module_has_no_project_imports(self) -> None:
        """``pipeline.schemas.shard_metadata`` must not depend on src/, pipeline.schemas.config, or
        any other project module — it's a leaf so the renderer can import it cycle-free."""
        import pipeline.schemas.shard_metadata as leaf

        # Filter for project imports only, not stdlib/3p.
        forbidden_prefixes = ("pipeline.", "src.", "scripts.")
        offenders = [
            name
            for name in dir(leaf)
            if not name.startswith("_")
            and hasattr(getattr(leaf, name, None), "__module__")
            and getattr(getattr(leaf, name), "__module__", "").startswith(forbidden_prefixes)
            and getattr(leaf, name).__module__ != "pipeline.schemas.shard_metadata"
        ]
        assert offenders == [], f"leaf module imports project modules: {offenders}"

    def test_strict_rejects_string_for_int_velocity(self) -> None:
        """Strict=True rejects coercion of "100" → int for the velocity field."""
        kwargs = _valid_kwargs()
        kwargs["velocity"] = "100"
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


class TestShardMetadataReExportFromSpec:
    """Backward-compatible re-export so existing callers don't break."""

    def test_spec_re_exports_shard_metadata(self) -> None:
        """`pipeline.schemas.spec.ShardMetadata` is the same class as the leaf import."""
        from pipeline.schemas.shard_metadata import ShardMetadata as Leaf
        from pipeline.schemas.spec import ShardMetadata as ReExported

        assert ReExported is Leaf
