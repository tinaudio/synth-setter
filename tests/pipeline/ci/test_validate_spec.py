"""Tests for synth_setter.pipeline.ci.validate_spec validation functions."""

from __future__ import annotations

from synth_setter.pipeline.ci.validate_spec import (
    _REQUIRED_RENDER_FIELDS,
    _REQUIRED_TOP_LEVEL_FIELDS,
    validate_structure,
    validate_test_values,
)
from synth_setter.pipeline.schemas.spec import DatasetSpec, RenderConfig


def _make_valid_spec(*, output_format: str = "hdf5", **overrides: object) -> dict:
    """Build a minimal valid spec dict mirroring DatasetSpec.model_dump output."""
    ext = ".h5" if output_format == "hdf5" else ".tar"
    spec: dict = {
        "task_name": "test",
        "run_id": "test-20260328T120000000Z",
        "created_at": "2026-03-28T12:00:00+00:00",
        "git_sha": "a" * 40,
        "is_repo_dirty": False,
        "output_format": output_format,
        "train_val_test_sizes": [32, 32, 32],
        "train_val_test_seeds": None,
        "base_seed": 42,
        "num_params": 92,
        "num_shards": 3,
        "r2": {
            "bucket": "intermediate-data",
            "prefix_root": "data",
            "prefix": "data/test/test-20260328T120000000Z/",
        },
        "render": {
            "plugin_path": "plugins/Surge XT.vst3",
            "preset_path": "presets/surge-base.vstpreset",
            "param_spec_name": "surge_simple",
            "renderer_version": "1.3.4",
            "sample_rate": 16000,
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "samples_per_render_batch": 32,
            "samples_per_shard": 32,
            "plugin_reload_cadence": "render",
            "gui_toggle_cadence": "never",
        },
        "shards": [
            {"shard_id": i, "filename": f"shard-{i:06d}{ext}", "seed": 42 + i} for i in range(3)
        ],
    }
    if "render" in overrides:
        # Merge nested render overrides instead of replacing the whole sub-dict.
        spec["render"] = {**spec["render"], **overrides.pop("render")}  # type: ignore[dict-item]
    spec.update(overrides)
    return spec


class TestValidateStructure:
    """Tests for validate_structure."""

    def test_valid_spec_returns_no_errors(self) -> None:
        """Valid spec with all required fields passes validation."""
        spec = _make_valid_spec()
        assert validate_structure(spec) == []

    def test_missing_field_returns_error(self) -> None:
        """Spec missing a required field returns a 'missing' error."""
        spec = _make_valid_spec()
        del spec["base_seed"]
        errors = validate_structure(spec)
        assert len(errors) == 1
        assert "missing" in errors[0]

    def test_invalid_git_sha_returns_error(self) -> None:
        """Non-hex git_sha returns a git_sha error."""
        spec = _make_valid_spec(git_sha="not-a-sha")
        assert any("git_sha" in e for e in validate_structure(spec))

    def test_empty_renderer_version_returns_error(self) -> None:
        """Empty render.renderer_version returns a renderer_version error."""
        spec = _make_valid_spec(render={"renderer_version": ""})
        assert any("renderer_version" in e for e in validate_structure(spec))

    def test_empty_shards_returns_error(self) -> None:
        """Empty shards list returns a shards error."""
        spec = _make_valid_spec(shards=[])
        assert any("shards" in e for e in validate_structure(spec))

    def test_unknown_output_format_returns_error(self) -> None:
        """An output_format outside the known mapping returns a structural error."""
        spec = _make_valid_spec(output_format="parquet")
        errors = validate_structure(spec)
        assert any("output_format" in e and "parquet" in e for e in errors)

    def test_missing_r2_block_returns_error(self) -> None:
        """Spec missing the top-level ``r2`` block (a DatasetSpec model field) is rejected.

        Pins the model-derived required set: ``r2`` is the nested
        ``R2Location`` field that replaced the flat ``r2_bucket`` /
        ``r2_prefix_root`` / ``r2_prefix`` keys. The structural validator
        derives the required set from the model, so adding/removing fields
        on ``DatasetSpec`` automatically tightens/loosens the check.
        """
        spec = _make_valid_spec()
        del spec["r2"]
        errors = validate_structure(spec)
        assert any("missing" in e and "r2" in e for e in errors)

    def test_required_top_level_fields_match_dataset_spec_model(self) -> None:
        """Required top-level set is derived from DatasetSpec, not hand-mirrored."""
        expected = set(DatasetSpec.model_fields) | set(DatasetSpec.model_computed_fields)
        assert set(_REQUIRED_TOP_LEVEL_FIELDS) == expected

    def test_required_render_fields_match_render_config_model(self) -> None:
        """Required render set is derived from RenderConfig, not hand-mirrored."""
        assert set(_REQUIRED_RENDER_FIELDS) == set(RenderConfig.model_fields)


class TestValidateTestValues:
    """Tests for validate_test_values."""

    def test_valid_test_spec_returns_no_errors(self) -> None:
        """Spec matching generate_dataset/ci-materialize-test.yaml expectations passes."""
        spec = _make_valid_spec()
        assert validate_test_values(spec) == []
        assert all(s["filename"].endswith(".h5") for s in spec["shards"])

    def test_wrong_shard_count_returns_error(self) -> None:
        """Spec with 2 shards instead of 3 returns a shard count error."""
        spec = _make_valid_spec(
            shards=[
                {"shard_id": 0, "filename": "shard-000000.h5", "seed": 42},
                {"shard_id": 1, "filename": "shard-000001.h5", "seed": 43},
            ]
        )
        errors = validate_test_values(spec)
        assert any("3 shards" in e for e in errors)

    def test_wrong_seeds_returns_error(self) -> None:
        """Spec with wrong seeds returns a seed error."""
        spec = _make_valid_spec(
            shards=[
                {"shard_id": 0, "filename": "shard-000000.h5", "seed": 1},
                {"shard_id": 1, "filename": "shard-000001.h5", "seed": 2},
                {"shard_id": 2, "filename": "shard-000002.h5", "seed": 3},
            ]
        )
        errors = validate_test_values(spec)
        assert any("seed" in e for e in errors)

    def test_unknown_output_format_returns_error_not_keyerror(self) -> None:
        """Unknown output_format produces a graceful error rather than a KeyError crash."""
        spec = _make_valid_spec(output_format="parquet")
        errors = validate_test_values(spec)
        assert any("output_format" in e and "parquet" in e for e in errors)
