"""Tests for pipeline.ci.validate_spec validation functions."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.ci.validate_spec import _read_spec_text, validate_structure, validate_test_values


def _make_valid_spec(*, output_format: str = "hdf5", **overrides: object) -> dict:
    """Build a minimal valid spec dict for testing."""
    ext = ".h5" if output_format == "hdf5" else ".tar"
    spec: dict = {
        "run_id": "test-20260328T120000Z",
        "created_at": "2026-03-28T12:00:00+00:00",
        "code_version": "a" * 40,
        "is_repo_dirty": False,
        "param_spec": "surge_simple",
        "renderer_version": "1.3.4",
        "output_format": output_format,
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
            {"shard_id": i, "filename": f"shard-{i:06d}{ext}", "seed": 42 + i} for i in range(3)
        ],
    }
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

    def test_invalid_code_version_returns_error(self) -> None:
        """Non-hex code_version returns a code_version error."""
        spec = _make_valid_spec(code_version="not-a-sha")
        assert any("code_version" in e for e in validate_structure(spec))

    def test_empty_renderer_version_returns_error(self) -> None:
        """Empty renderer_version returns a renderer_version error."""
        spec = _make_valid_spec(renderer_version="")
        assert any("renderer_version" in e for e in validate_structure(spec))

    def test_empty_shards_returns_error(self) -> None:
        """Empty shards list returns a shards error."""
        spec = _make_valid_spec(shards=[])
        assert any("shards" in e for e in validate_structure(spec))


class TestValidateTestValues:
    """Tests for validate_test_values."""

    @pytest.mark.parametrize(
        ("output_format", "ext"),
        [("hdf5", ".h5"), ("wds", ".tar")],
    )
    def test_valid_test_spec_returns_no_errors(self, output_format: str, ext: str) -> None:
        """Spec matching ci-materialize-test.yaml expectations passes for both formats."""
        spec = _make_valid_spec(output_format=output_format)
        assert validate_test_values(spec) == []
        assert all(s["filename"].endswith(ext) for s in spec["shards"])

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

    def test_wds_spec_with_h5_filenames_returns_error(self) -> None:
        """A wds spec with hdf5-style filenames fails the extension check."""
        spec = _make_valid_spec(
            output_format="wds",
            shards=[
                {"shard_id": i, "filename": f"shard-{i:06d}.h5", "seed": 42 + i} for i in range(3)
            ],
        )
        errors = validate_test_values(spec)
        assert any("filenames" in e for e in errors)


class TestReadSpecText:
    """Tests for _read_spec_text — local path or r2:// URI dispatch."""

    def test_local_path_reads_file_directly(self, tmp_path: Path) -> None:
        """A non-URI argument is read directly as a filesystem path."""
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps({"hello": "world"}))
        assert json.loads(_read_spec_text(str(spec_path))) == {"hello": "world"}

    def test_r2_uri_downloads_via_r2_io(self) -> None:
        """R2:// URI dispatches through pipeline.r2_io.downloaded_to_tempfile."""

        def fake_check_call(args: list[str]) -> None:
            Path(args[-1]).write_text(json.dumps({"hello": "from-r2"}))

        with patch("pipeline.r2_io.subprocess.check_call", side_effect=fake_check_call):
            text = _read_spec_text("r2://bucket/spec.json")
        assert json.loads(text) == {"hello": "from-r2"}
