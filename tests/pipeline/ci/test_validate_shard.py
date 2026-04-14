"""Tests for pipeline.ci.validate_shard."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest

from pipeline.ci.validate_shard import validate_shard

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _create_shard(path: Path, shard_size: int, datasets: dict[str, tuple] | None = None) -> None:
    """Create a minimal HDF5 shard with given datasets and shapes."""
    defaults = {
        "audio": (shard_size, 2, 64000),
        "mel_spec": (shard_size, 2, 128, 401),
        "param_array": (shard_size, 92),
    }
    with h5py.File(path, "w") as f:
        for name, shape in (datasets or defaults).items():
            f.create_dataset(name, shape=shape, dtype=np.float32)


@pytest.fixture()
def real_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[return]
    """Create a real DatasetPipelineSpec with mocked I/O."""
    from pipeline.schemas.config import DatasetConfig, SplitsConfig
    from pipeline.schemas.prefix import DatasetConfigId
    from pipeline.schemas.spec import materialize_spec

    monkeypatch.setattr("pipeline.schemas.spec._get_git_sha", lambda: "a" * 40)
    monkeypatch.setattr("pipeline.schemas.spec._is_repo_dirty", lambda: False)
    fixed_now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        "pipeline.schemas.spec.datetime",
        type(
            "FakeDatetime",
            (),
            {
                "now": staticmethod(lambda tz: fixed_now),
                "fromisoformat": datetime.fromisoformat,
            },
        )(),
    )

    contents = tmp_path / "FakePlugin.vst3" / "Contents"
    contents.mkdir(parents=True)
    (contents / "moduleinfo.json").write_text('{"Version": "1.3.4"}')

    config = DatasetConfig(
        param_spec="surge_simple",
        plugin_path=str(tmp_path / "FakePlugin.vst3"),
        output_format="hdf5",
        sample_rate=16000,
        shard_size=10,
        num_shards=1,
        base_seed=42,
        splits=SplitsConfig(train=1, val=0, test=0),
        preset_path="presets/surge-base.vstpreset",
        channels=2,
        velocity=100,
        signal_duration_seconds=4.0,
        min_loudness=-55.0,
        sample_batch_size=32,
    )
    return materialize_spec(config, DatasetConfigId("test-dataset"))


# ---------------------------------------------------------------------------
# Tests for validate_shard()
# ---------------------------------------------------------------------------


class TestValidateShard:
    """Tests for validate_shard() function."""

    def test_valid_shard_returns_no_errors(self, real_spec: object, tmp_path: Path) -> None:
        """Correct HDF5 with all expected datasets and correct row counts returns []."""
        shard_path = tmp_path / "shard-000000.h5"
        _create_shard(shard_path, shard_size=real_spec.shard_size)  # type: ignore[union-attr]

        errors = validate_shard(shard_path, real_spec)  # type: ignore[arg-type]

        assert errors == []

    def test_missing_dataset_returns_error(self, real_spec: object, tmp_path: Path) -> None:
        """HDF5 missing one of the expected datasets returns an error."""
        shard_path = tmp_path / "shard-000000.h5"
        _create_shard(
            shard_path,
            shard_size=real_spec.shard_size,  # type: ignore[union-attr]
            datasets={
                "audio": (real_spec.shard_size, 2, 64000),  # type: ignore[union-attr]
                "mel_spec": (real_spec.shard_size, 2, 128, 401),  # type: ignore[union-attr]
                # param_array intentionally omitted
            },
        )

        errors = validate_shard(shard_path, real_spec)  # type: ignore[arg-type]

        assert len(errors) == 1
        assert "param_array" in errors[0]

    def test_wrong_row_count_returns_error(self, real_spec: object, tmp_path: Path) -> None:
        """Dataset with wrong shape[0] returns an error mentioning that dataset."""
        shard_path = tmp_path / "shard-000000.h5"
        wrong_size = real_spec.shard_size + 5  # type: ignore[union-attr]
        _create_shard(
            shard_path,
            shard_size=wrong_size,
            datasets={
                "audio": (wrong_size, 2, 64000),
                "mel_spec": (real_spec.shard_size, 2, 128, 401),  # type: ignore[union-attr]
                "param_array": (real_spec.shard_size, 92),  # type: ignore[union-attr]
            },
        )

        errors = validate_shard(shard_path, real_spec)  # type: ignore[arg-type]

        assert len(errors) == 1
        assert "audio" in errors[0]

    def test_not_hdf5_returns_error(self, real_spec: object, tmp_path: Path) -> None:
        """File that is not valid HDF5 returns an error."""
        shard_path = tmp_path / "not-an-hdf5.h5"
        shard_path.write_bytes(b"this is not an hdf5 file\n")

        errors = validate_shard(shard_path, real_spec)  # type: ignore[arg-type]

        assert len(errors) == 1
        assert "HDF5" in errors[0] or "hdf5" in errors[0].lower()

    def test_file_not_found_returns_error(self, real_spec: object, tmp_path: Path) -> None:
        """Path that does not exist returns an error."""
        shard_path = tmp_path / "nonexistent.h5"

        errors = validate_shard(shard_path, real_spec)  # type: ignore[arg-type]

        assert len(errors) == 1
        assert "not found" in errors[0].lower() or "does not exist" in errors[0].lower()

    def test_extra_datasets_ignored(self, real_spec: object, tmp_path: Path) -> None:
        """Extra datasets in HDF5 beyond the required three do not cause errors."""
        shard_path = tmp_path / "shard-000000.h5"
        shard_size = real_spec.shard_size  # type: ignore[union-attr]
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

        errors = validate_shard(shard_path, real_spec)  # type: ignore[arg-type]

        assert errors == []


# ---------------------------------------------------------------------------
# Tests for main() CLI entry point
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the CLI entry point main()."""

    def test_cli_exits_zero_on_valid_shard(
        self, real_spec: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Valid shard and valid spec JSON produce exit code 0."""
        from pipeline.ci.validate_shard import main

        shard_path = tmp_path / "shard-000000.h5"
        _create_shard(shard_path, shard_size=real_spec.shard_size)  # type: ignore[union-attr]

        spec_json_path = tmp_path / "spec.json"
        spec_json_path.write_text(real_spec.model_dump_json())  # type: ignore[union-attr]

        monkeypatch.setattr(sys, "argv", ["validate_shard", str(spec_json_path), str(shard_path)])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0

    def test_cli_exits_one_on_invalid_shard(
        self, real_spec: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid shard (not HDF5) produces exit code 1."""
        from pipeline.ci.validate_shard import main

        shard_path = tmp_path / "bad.h5"
        shard_path.write_bytes(b"garbage")

        spec_json_path = tmp_path / "spec.json"
        spec_json_path.write_text(real_spec.model_dump_json())  # type: ignore[union-attr]

        monkeypatch.setattr(sys, "argv", ["validate_shard", str(spec_json_path), str(shard_path)])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
