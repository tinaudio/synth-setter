"""Tests for pipeline.ci.validate_shard."""

from __future__ import annotations

import io
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import webdataset as wds

from pipeline.ci.validate_shard import (
    _shard_uri,
    validate_all_shards_from_r2,
    validate_shard,
)
from pipeline.schemas.spec import DatasetSpec

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


def _create_tar_shard(
    path: Path,
    shard_size: int,
    arrays: dict[str, tuple[int, ...]] | None = None,
    *,
    omit_fields: tuple[str, ...] = (),
    omit_metadata: bool = False,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Create a wds tar shard with per-batch keyed members and a trailing metadata.json.

    The single batch holds ``shard_size`` rows so the validator's summed-row check passes.
    ``arrays`` overrides per-field shapes; ``omit_fields`` drops named fields entirely;
    ``omit_metadata`` skips the trailing metadata sample.
    """
    defaults: dict[str, tuple[int, ...]] = {
        "audio": (shard_size, _AUDIO_CHANNELS, _AUDIO_SAMPLES_PER_ROW),
        "mel_spec": (shard_size, *_MEL_SHAPE_PER_ROW),
        "param_array": (shard_size, _PARAM_LENGTH),
    }
    chosen = arrays or defaults
    meta = (
        metadata
        if metadata is not None
        else {
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "sample_rate": 16000.0,
            "channels": 2,
            "min_loudness": -55.0,
        }
    )

    sample: dict[str, object] = {"__key__": "00000000"}
    for field, shape in chosen.items():
        if field in omit_fields:
            continue
        sample[f"{field}.npy"] = np.zeros(shape, dtype=np.float32)

    with wds.TarWriter(str(path)) as writer:  # pyright: ignore[reportAttributeAccessIssue]
        writer.write(sample)
        if not omit_metadata:
            writer.write({"__key__": "metadata", "json": meta})


@pytest.fixture()
def real_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DatasetSpec:
    """Build a real DatasetSpec with mocked git/timestamp factories."""
    fixed_now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("pipeline.schemas.spec._get_git_sha", lambda: "a" * 40)
    monkeypatch.setattr("pipeline.schemas.spec._is_repo_dirty", lambda: False)
    monkeypatch.setattr("pipeline.schemas.spec._utc_now", lambda: fixed_now)

    contents = tmp_path / "FakePlugin.vst3" / "Contents"
    contents.mkdir(parents=True)
    (contents / "moduleinfo.json").write_text('{"Version": "1.3.4"}')

    return DatasetSpec(
        task_name="test-dataset",
        output_format="hdf5",
        train_val_test_sizes=(10, 0, 0),
        train_val_test_seeds=(42, 43, 44),
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
            "sample_batch_size": 32,
            "batch_per_shard": 10,
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
        _create_shard(shard_path, shard_size=real_spec.render.batch_per_shard)

        errors = validate_shard(shard_path, real_spec)

        assert errors == []

    def test_missing_dataset_returns_error(self, real_spec: DatasetSpec, tmp_path: Path) -> None:
        """HDF5 missing one of the expected datasets returns an error."""
        shard_path = tmp_path / "shard-000000.h5"
        _create_shard(
            shard_path,
            shard_size=real_spec.render.batch_per_shard,
            datasets={
                "audio": (real_spec.render.batch_per_shard, 2, 64000),
                "mel_spec": (real_spec.render.batch_per_shard, 2, 128, 401),
                # param_array intentionally omitted
            },
        )

        errors = validate_shard(shard_path, real_spec)

        assert len(errors) == 1
        assert "param_array" in errors[0]

    def test_wrong_row_count_returns_error(self, real_spec: DatasetSpec, tmp_path: Path) -> None:
        """Dataset with wrong shape[0] returns an error mentioning that dataset."""
        shard_path = tmp_path / "shard-000000.h5"
        wrong_size = real_spec.render.batch_per_shard + 5
        _create_shard(
            shard_path,
            shard_size=wrong_size,
            datasets={
                "audio": (wrong_size, 2, 64000),
                "mel_spec": (real_spec.render.batch_per_shard, 2, 128, 401),
                "param_array": (real_spec.render.batch_per_shard, 92),
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
        shard_size = real_spec.render.batch_per_shard
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


class TestShardUri:
    """Tests for _shard_uri helper."""

    def test_builds_r2_uri_from_spec_and_filename(self, real_spec: DatasetSpec) -> None:
        """The constructed URI embeds bucket, prefix, and filename."""
        spec = real_spec  # type: ignore[assignment]
        uri = _shard_uri(spec, "shard-000007.h5")
        assert uri.startswith("r2://")
        assert "shard-000007.h5" in uri
        assert spec.r2_bucket in uri
        assert spec.r2_prefix in uri


class TestValidateAllShardsFromR2:
    """Tests for validate_all_shards_from_r2 — iterates spec.shards via R2."""

    def test_all_valid_returns_no_errors(self, real_spec: DatasetSpec, tmp_path: Path) -> None:
        """When every shard downloads valid HDF5, returns []."""
        spec = real_spec  # type: ignore[assignment]

        def fake_check_call(args: list[str]) -> None:
            # Simulate rclone copyto: write a valid shard to dest path
            _create_shard(Path(args[-1]), shard_size=spec.render.batch_per_shard)

        with patch("pipeline.r2_io.subprocess.check_call", side_effect=fake_check_call):
            errors = validate_all_shards_from_r2(spec)

        assert errors == []

    def test_invalid_shard_error_carries_shard_filename(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """Validation errors are prefixed with the shard filename."""
        spec = real_spec  # type: ignore[assignment]

        def fake_check_call(args: list[str]) -> None:
            # Write garbage so shard fails HDF5 open
            Path(args[-1]).write_bytes(b"garbage")

        with patch("pipeline.r2_io.subprocess.check_call", side_effect=fake_check_call):
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
        from pipeline.ci.validate_shard import main

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
        from pipeline.ci.validate_shard import main

        spec = real_spec  # type: ignore[assignment]
        spec_json_path = tmp_path / "spec.json"
        spec_json_path.write_text(spec.model_dump_json())

        def fake_check_call(args: list[str]) -> None:
            _create_shard(Path(args[-1]), shard_size=spec.render.batch_per_shard)

        monkeypatch.setattr(sys, "argv", ["validate_shard", str(spec_json_path)])
        with patch("pipeline.r2_io.subprocess.check_call", side_effect=fake_check_call):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0

    def test_cli_exits_one_when_a_shard_is_invalid(
        self, real_spec: DatasetSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If any shard in spec.shards fails validation, exit 1."""
        from pipeline.ci.validate_shard import main

        spec = real_spec  # type: ignore[assignment]
        spec_json_path = tmp_path / "spec.json"
        spec_json_path.write_text(spec.model_dump_json())

        def fake_check_call(args: list[str]) -> None:
            Path(args[-1]).write_bytes(b"garbage")

        monkeypatch.setattr(sys, "argv", ["validate_shard", str(spec_json_path)])
        with patch("pipeline.r2_io.subprocess.check_call", side_effect=fake_check_call):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Tests for the tar (wds) shard validation path
# ---------------------------------------------------------------------------


class TestValidateTarShard:
    """Tests for validate_shard() on .tar shards."""

    def test_valid_tar_shard_returns_no_errors(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """Tar with all expected members and correct row counts returns []."""
        shard_path = tmp_path / "shard-000000.tar"
        _create_tar_shard(shard_path, shard_size=real_spec.render.batch_per_shard)

        errors = validate_shard(shard_path, real_spec)

        assert errors == []

    def test_missing_mel_member_returns_error(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """Tar missing all mel_spec members returns a missing-member error."""
        shard_path = tmp_path / "shard-000000.tar"
        _create_tar_shard(
            shard_path,
            shard_size=real_spec.render.batch_per_shard,
            omit_fields=("mel_spec",),
        )

        errors = validate_shard(shard_path, real_spec)

        assert len(errors) == 1
        assert "mel_spec" in errors[0]

    def test_missing_metadata_member_returns_error(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """Tar missing metadata.json returns a missing-member error."""
        shard_path = tmp_path / "shard-000000.tar"
        _create_tar_shard(
            shard_path,
            shard_size=real_spec.render.batch_per_shard,
            omit_metadata=True,
        )

        errors = validate_shard(shard_path, real_spec)

        assert any("metadata.json" in e for e in errors)

    def test_wrong_row_count_returns_error(self, real_spec: DatasetSpec, tmp_path: Path) -> None:
        """Tar with summed audio rows != shard_size returns an error mentioning the field."""
        shard_path = tmp_path / "shard-000000.tar"
        wrong_size = real_spec.render.batch_per_shard + 5
        _create_tar_shard(
            shard_path,
            shard_size=real_spec.render.batch_per_shard,
            arrays={
                "audio": (wrong_size, 2, 64000),
                "mel_spec": (real_spec.render.batch_per_shard, 2, 128, 401),
                "param_array": (real_spec.render.batch_per_shard, 92),
            },
        )

        errors = validate_shard(shard_path, real_spec)

        assert len(errors) == 1
        assert "audio" in errors[0]

    def test_not_a_tar_returns_error(self, real_spec: DatasetSpec, tmp_path: Path) -> None:
        """File that is not a valid tar returns an error."""
        shard_path = tmp_path / "shard-000000.tar"
        shard_path.write_bytes(b"definitely not a tar file")

        errors = validate_shard(shard_path, real_spec)

        assert len(errors) == 1
        assert "tar" in errors[0].lower()

    def test_gzipped_tar_returns_error(self, real_spec: DatasetSpec, tmp_path: Path) -> None:
        """A .tar.gz file masquerading as .tar fails validation rather than being silently
        accepted.

        The wds writer only emits uncompressed tars; pinning ``mode="r:"`` keeps the validator
        from auto-detecting and accepting a recompressed shard whose layout would still pass
        downstream checks but disagree with the writer-side contract.
        """
        shard_path = tmp_path / "shard-000000.tar"
        with tarfile.open(shard_path, mode="w:gz") as gz:
            payload = io.BytesIO(b"junk")
            info = tarfile.TarInfo(name="00000000.audio.npy")
            info.size = len(payload.getvalue())
            gz.addfile(info, payload)

        errors = validate_shard(shard_path, real_spec)

        assert len(errors) == 1
        assert "tar" in errors[0].lower()

    def test_metadata_missing_required_field_returns_error(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """Tar with metadata.json that fails ShardMetadata validation returns an error."""
        shard_path = tmp_path / "shard-000000.tar"
        _create_tar_shard(
            shard_path,
            shard_size=real_spec.render.batch_per_shard,
            metadata={"sample_rate": 16000.0},  # missing velocity, channels, ...
        )

        errors = validate_shard(shard_path, real_spec)

        assert any("metadata.json" in e for e in errors)

    def test_metadata_wrong_type_returns_error(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """Tar metadata.json with a strict-typed field violation returns an error."""
        shard_path = tmp_path / "shard-000000.tar"
        _create_tar_shard(
            shard_path,
            shard_size=real_spec.render.batch_per_shard,
            metadata={
                "velocity": "100",  # strict=True rejects str-for-int
                "signal_duration_seconds": 4.0,
                "sample_rate": 16000.0,
                "channels": 2,
                "min_loudness": -55.0,
            },
        )

        errors = validate_shard(shard_path, real_spec)

        assert any("metadata.json" in e for e in errors)

    def test_malformed_npy_payload_returns_error_does_not_raise(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """Garbage bytes inside an ``audio.npy`` member surface as an error string, not an
        exception.

        np.load on malformed bytes can raise ``ValueError`` / ``EOFError``;
        the validator must trap and convert to an error so a corrupt shard
        from a worker doesn't crash the validate job before reporting.
        """
        shard_path = tmp_path / "shard-000000.tar"
        with tarfile.open(shard_path, "w") as tar:
            for field in ("mel_spec", "param_array"):
                shape = (
                    (real_spec.render.batch_per_shard, 2, 64000)
                    if field == "audio"
                    else (real_spec.render.batch_per_shard, 92)
                )
                buf = io.BytesIO()
                np.save(buf, np.zeros(shape, dtype=np.float32))
                buf.seek(0)
                payload = buf.read()
                info = tarfile.TarInfo(name=f"00000000.{field}.npy")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))

            audio_garbage = b"definitely not a valid npy header"
            audio_info = tarfile.TarInfo(name="00000000.audio.npy")
            audio_info.size = len(audio_garbage)
            tar.addfile(audio_info, io.BytesIO(audio_garbage))

            meta_payload = (
                b'{"velocity":100,"signal_duration_seconds":4.0,"sample_rate":16000.0,'
                b'"channels":2,"min_loudness":-55.0}'
            )
            meta_info = tarfile.TarInfo(name="metadata.json")
            meta_info.size = len(meta_payload)
            tar.addfile(meta_info, io.BytesIO(meta_payload))

        errors = validate_shard(shard_path, real_spec)

        assert any("malformed npy payload" in e and "audio" in e for e in errors)
