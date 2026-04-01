"""Tests for pipeline.ci.validate_spec validation functions."""

from __future__ import annotations

from pipeline.ci.validate_spec import validate_structure, validate_test_values


def _make_valid_spec(**overrides: object) -> dict:
    """Build a minimal valid spec dict for testing."""
    spec: dict = {
        "run_id": "test-20260328T120000Z",
        "created_at": "2026-03-28T12:00:00+00:00",
        "code_version": "a" * 40,
        "is_repo_dirty": False,
        "param_spec": "surge_simple",
        "renderer_version": "1.3.4",
        "output_format": "hdf5",
        "sample_rate": 16000,
        "shard_size": 32,
        "base_seed": 42,
        "num_params": 92,
        "splits": {"train": 1, "val": 1, "test": 1},
        "plugin_path": "plugins/Surge XT.vst3",
        "preset_path": "presets/surge-base.vstpreset",
        "channels": 2,
        "r2_prefix": "data/test/test-20260328T120000Z/",
        "velocity": 100,
        "signal_duration_seconds": 4.0,
        "min_loudness": -55.0,
        "sample_batch_size": 32,
        "shards": [
            {"shard_id": 0, "filename": "shard-000000.h5", "seed": 42},
            {"shard_id": 1, "filename": "shard-000001.h5", "seed": 43},
            {"shard_id": 2, "filename": "shard-000002.h5", "seed": 44},
        ],
    }
    spec.update(overrides)
    return spec


class TestValidateStructure:
    """Tests for validate_structure."""

    def test_valid_spec_returns_no_errors(self) -> None:
        # plumb:req-74aa845b
        # plumb:req-2b453be4
        # plumb:req-9c3bfede
        # plumb:req-bc388485
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

    def test_invalid_code_version_returns_error(self) -> None:
        # plumb:req-bd3e7a97
        """Non-hex code_version returns a code_version error."""
        spec = _make_valid_spec(code_version="not-a-sha")
        assert any("code_version" in e for e in validate_structure(spec))

    def test_empty_renderer_version_returns_error(self) -> None:
        # plumb:req-9c9e072c
        """Empty renderer_version returns a renderer_version error."""
        spec = _make_valid_spec(renderer_version="")
        assert any("renderer_version" in e for e in validate_structure(spec))

    def test_empty_shards_returns_error(self) -> None:
        # plumb:req-9f751e55
        """Empty shards list returns a shards error."""
        spec = _make_valid_spec(shards=[])
        assert any("shards" in e for e in validate_structure(spec))


class TestValidateTestValues:
    """Tests for validate_test_values."""

    def test_valid_test_spec_returns_no_errors(self) -> None:
        """Spec matching ci-materialize-test.yaml expectations passes."""
        spec = _make_valid_spec()
        assert validate_test_values(spec) == []

    def test_wrong_shard_count_returns_error(self) -> None:
        # plumb:req-c1d2a20f
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
        # plumb:req-37250f13
        # plumb:req-455c526a
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
