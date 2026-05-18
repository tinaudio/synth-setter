"""Tests for synth_setter.pipeline.ci.validate_shard."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from synth_setter.pipeline.ci.validate_shard import (
    validate_all_shards_from_r2,
    validate_shard,
)
from synth_setter.pipeline.schemas.spec import DatasetSpec

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


_AUDIO_CHANNELS = 2
_AUDIO_SAMPLES_PER_ROW = 64000
_MEL_SHAPE_PER_ROW = (2, 128, 401)
_PARAM_LENGTH = 92


def _create_shard(
    path: Path, shard_size: int, datasets: dict[str, tuple[int, ...]] | None = None
) -> None:
    """Create a minimal HDF5 shard with given datasets and shapes."""
    defaults: dict[str, tuple[int, ...]] = {
        "audio": (shard_size, _AUDIO_CHANNELS, _AUDIO_SAMPLES_PER_ROW),
        "mel_spec": (shard_size, *_MEL_SHAPE_PER_ROW),
        "param_array": (shard_size, _PARAM_LENGTH),
    }
    with h5py.File(path, "w") as f:
        for name, shape in (datasets or defaults).items():
            f.create_dataset(name, shape=shape, dtype=np.float32)


@pytest.fixture()
def real_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DatasetSpec:
    """Build a real DatasetSpec with mocked git/timestamp factories."""
    fixed_now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._get_git_sha", lambda: "a" * 40)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._is_repo_dirty", lambda: False)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._utc_now", lambda: fixed_now)

    contents = tmp_path / "FakePlugin.vst3" / "Contents"
    contents.mkdir(parents=True)
    (contents / "moduleinfo.json").write_text('{"Version": "1.3.4"}')

    return DatasetSpec(
        task_name="test-dataset",
        output_format="hdf5",
        train_val_test_sizes=(10, 0, 0),
        base_seed=42,
        r2_bucket="intermediate-data",
        render={
            "plugin_path": str(tmp_path / "FakePlugin.vst3"),
            "preset_path": "presets/surge-base.vstpreset",
            "param_spec_name": "surge_simple",
            "renderer_version": "1.3.4",
            "sample_rate": 16000,
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "samples_per_render_batch": 32,
            "samples_per_shard": 10,
            "open_gui_every_render": False,
        },  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Tests for validate_shard()
# ---------------------------------------------------------------------------


class TestValidateShard:
    """Tests for validate_shard() function."""

    def test_valid_shard_returns_no_errors(self, real_spec: DatasetSpec, tmp_path: Path) -> None:
        """Correct HDF5 with all expected datasets and correct row counts returns []."""
        shard_path = tmp_path / "shard-000000.h5"
        _create_shard(shard_path, shard_size=real_spec.render.samples_per_shard)

        errors = validate_shard(shard_path, real_spec)

        assert errors == []

    def test_missing_dataset_returns_error(self, real_spec: DatasetSpec, tmp_path: Path) -> None:
        """HDF5 missing one of the expected datasets returns an error."""
        shard_path = tmp_path / "shard-000000.h5"
        _create_shard(
            shard_path,
            shard_size=real_spec.render.samples_per_shard,
            datasets={
                "audio": (real_spec.render.samples_per_shard, 2, 64000),
                "mel_spec": (real_spec.render.samples_per_shard, 2, 128, 401),
                # param_array intentionally omitted
            },
        )

        errors = validate_shard(shard_path, real_spec)

        assert len(errors) == 1
        assert "param_array" in errors[0]

    def test_wrong_row_count_returns_error(self, real_spec: DatasetSpec, tmp_path: Path) -> None:
        """Dataset with wrong shape[0] returns an error mentioning that dataset."""
        shard_path = tmp_path / "shard-000000.h5"
        wrong_size = real_spec.render.samples_per_shard + 5
        _create_shard(
            shard_path,
            shard_size=wrong_size,
            datasets={
                "audio": (wrong_size, 2, 64000),
                "mel_spec": (real_spec.render.samples_per_shard, 2, 128, 401),
                "param_array": (real_spec.render.samples_per_shard, 92),
            },
        )

        errors = validate_shard(shard_path, real_spec)

        assert len(errors) == 1
        assert "audio" in errors[0]

    def test_not_hdf5_returns_error(self, real_spec: DatasetSpec, tmp_path: Path) -> None:
        """File that is not valid HDF5 returns an error."""
        shard_path = tmp_path / "not-an-hdf5.h5"
        shard_path.write_bytes(b"this is not an hdf5 file\n")

        errors = validate_shard(shard_path, real_spec)

        assert len(errors) == 1
        assert "HDF5" in errors[0] or "hdf5" in errors[0].lower()

    def test_file_not_found_returns_error(self, real_spec: DatasetSpec, tmp_path: Path) -> None:
        """Path that does not exist returns an error."""
        shard_path = tmp_path / "nonexistent.h5"

        errors = validate_shard(shard_path, real_spec)

        assert len(errors) == 1
        assert "not found" in errors[0].lower() or "does not exist" in errors[0].lower()

    def test_extra_datasets_ignored(self, real_spec: DatasetSpec, tmp_path: Path) -> None:
        """Extra datasets in HDF5 beyond the required three do not cause errors."""
        shard_path = tmp_path / "shard-000000.h5"
        shard_size = real_spec.render.samples_per_shard
        _create_shard(
            shard_path,
            shard_size=shard_size,
            datasets={
                "audio": (shard_size, 2, 64000),
                "mel_spec": (shard_size, 2, 128, 401),
                "param_array": (shard_size, 92),
                "extra_dataset": (shard_size, 7),
            },
        )

        errors = validate_shard(shard_path, real_spec)

        assert errors == []


# ---------------------------------------------------------------------------
# Tests for main() CLI entry point
# ---------------------------------------------------------------------------


class TestValidateAllShardsFromR2:
    """Tests for validate_all_shards_from_r2 — iterates spec.shards via R2."""

    def test_all_valid_returns_no_errors(self, real_spec: DatasetSpec, tmp_path: Path) -> None:
        """When every shard downloads valid HDF5, returns []."""
        spec = real_spec

        def fake_check_call(args: list[str]) -> None:
            # Simulate rclone copyto: write a valid shard to dest path
            _create_shard(Path(args[-1]), shard_size=spec.render.samples_per_shard)

        with patch(
            "synth_setter.pipeline.r2_io.subprocess.check_call", side_effect=fake_check_call
        ):
            errors = validate_all_shards_from_r2(spec)

        assert errors == []

    def test_invalid_shard_error_carries_shard_filename(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """Validation errors are prefixed with the shard filename."""
        spec = real_spec

        def fake_check_call(args: list[str]) -> None:
            # Write garbage so shard fails HDF5 open
            Path(args[-1]).write_bytes(b"garbage")

        with patch(
            "synth_setter.pipeline.r2_io.subprocess.check_call", side_effect=fake_check_call
        ):
            errors = validate_all_shards_from_r2(spec)

        assert errors  # at least one error
        # First spec.shards filename appears in the error string
        assert any(spec.shards[0].filename in e for e in errors)


class TestMain:
    """Tests for the CLI entry point main() with the new single-arg shape."""

    def test_cli_rejects_two_args(
        self, real_spec: DatasetSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The legacy 2-arg shape (spec + shard) is rejected."""
        from synth_setter.pipeline.ci.validate_shard import main

        spec_json_path = tmp_path / "spec.json"
        spec_json_path.write_text(real_spec.model_dump_json())

        monkeypatch.setattr(sys, "argv", ["validate_shard", str(spec_json_path), "ignored.h5"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    def test_cli_exits_zero_when_all_shards_valid(
        self, real_spec: DatasetSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Valid spec + R2-served valid shards → exit 0."""
        from synth_setter.pipeline.ci.validate_shard import main

        spec = real_spec
        spec_json_path = tmp_path / "spec.json"
        spec_json_path.write_text(spec.model_dump_json())

        def fake_check_call(args: list[str]) -> None:
            _create_shard(Path(args[-1]), shard_size=spec.render.samples_per_shard)

        monkeypatch.setattr(sys, "argv", ["validate_shard", str(spec_json_path)])
        with patch(
            "synth_setter.pipeline.r2_io.subprocess.check_call", side_effect=fake_check_call
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0

    def test_cli_exits_one_when_a_shard_is_invalid(
        self, real_spec: DatasetSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If any shard in spec.shards fails validation, exit 1."""
        from synth_setter.pipeline.ci.validate_shard import main

        spec = real_spec
        spec_json_path = tmp_path / "spec.json"
        spec_json_path.write_text(spec.model_dump_json())

        def fake_check_call(args: list[str]) -> None:
            Path(args[-1]).write_bytes(b"garbage")

        monkeypatch.setattr(sys, "argv", ["validate_shard", str(spec_json_path)])
        with patch(
            "synth_setter.pipeline.r2_io.subprocess.check_call", side_effect=fake_check_call
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
