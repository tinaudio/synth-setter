"""Tests for ``synth_setter.pipeline.spec_io`` — ``input_spec.json`` write/upload helpers."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from synth_setter.pipeline import r2_io, spec_io
from synth_setter.pipeline.constants import INPUT_SPEC_FILENAME
from synth_setter.pipeline.schemas.spec import DatasetSpec


def _spec_kwargs() -> dict[str, object]:
    """Return DatasetSpec kwargs deterministic enough for path/URI assertions.

    :returns: kwargs dict suitable for ``DatasetSpec(**kwargs)``.
    """
    return {
        "task_name": "test-task",
        "run_id": "test-task-20260519T120000000Z",
        "created_at": datetime(2026, 5, 19, 12, 0, 0, tzinfo=timezone.utc),
        "git_sha": "a" * 40,
        "is_repo_dirty": False,
        "output_format": "hdf5",
        "train_val_test_sizes": [10000, 0, 0],
        "base_seed": 42,
        "r2": {
            "bucket": "intermediate-data",
            "prefix": "data/test-task/test-task-20260519T120000000Z/",
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
            "samples_per_shard": 10000,
            "gui_toggle_cadence": "never",
        },
    }


@pytest.fixture()
def spec() -> DatasetSpec:
    """Return a deterministic DatasetSpec for path/URI assertions.

    :returns: A pinned-prefix ``DatasetSpec`` built from ``_spec_kwargs``.
    """
    return DatasetSpec(**_spec_kwargs())  # type: ignore[arg-type]


class TestLocalSpecPath:
    """``local_spec_path`` anticipates §3a's target ``metadata/`` layout."""

    def test_targets_metadata_layout(self, spec: DatasetSpec, tmp_path: Path) -> None:
        """Path is ``<output_dir>/data/<task_name>/<run_id>/metadata/input_spec.json``.

        :param spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Pytest tmp dir used as ``output_dir``.
        """
        result = spec_io.local_spec_path(spec, tmp_path)
        expected = (
            tmp_path / "data" / spec.task_name / spec.run_id / "metadata" / INPUT_SPEC_FILENAME
        )
        assert result == expected

    def test_returns_path_object(self, spec: DatasetSpec, tmp_path: Path) -> None:
        """Return type is ``pathlib.Path`` (not str).

        :param spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Pytest tmp dir used as ``output_dir``.
        """
        assert isinstance(spec_io.local_spec_path(spec, tmp_path), Path)


class TestWriteSpecLocally:
    """``write_spec_locally`` writes spec JSON to its local path."""

    def test_creates_parent_dirs(self, spec: DatasetSpec, tmp_path: Path) -> None:
        """Missing parent directories are created.

        :param spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Pytest tmp dir used as ``output_dir``.
        """
        result = spec_io.write_spec_locally(spec, tmp_path)
        assert result.is_file()
        assert result.parent.is_dir()

    def test_returns_local_spec_path(self, spec: DatasetSpec, tmp_path: Path) -> None:
        """Returned path equals ``local_spec_path(spec, output_dir)``.

        :param spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Pytest tmp dir used as ``output_dir``.
        """
        result = spec_io.write_spec_locally(spec, tmp_path)
        assert result == spec_io.local_spec_path(spec, tmp_path)

    def test_serializes_with_indent_2(self, spec: DatasetSpec, tmp_path: Path) -> None:
        """File content equals ``spec.model_dump_json(indent=2)``.

        :param spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Pytest tmp dir used as ``output_dir``.
        """
        result = spec_io.write_spec_locally(spec, tmp_path)
        assert result.read_text(encoding="utf-8") == spec.model_dump_json(indent=2)

    def test_is_idempotent(self, spec: DatasetSpec, tmp_path: Path) -> None:
        """Calling twice succeeds and leaves the same content on disk.

        :param spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Pytest tmp dir used as ``output_dir``.
        """
        first = spec_io.write_spec_locally(spec, tmp_path)
        first_content = first.read_text(encoding="utf-8")
        second = spec_io.write_spec_locally(spec, tmp_path)
        assert second == first
        assert second.read_text(encoding="utf-8") == first_content


class TestUploadSpec:
    """``upload_spec`` uploads the spec to its R2 URI."""

    def test_invokes_rclone_with_input_spec_uri(self, spec: DatasetSpec) -> None:
        """The rclone destination is the rclone-form of ``spec.r2.input_spec_uri()``.

        :param spec: Fixture-provided ``DatasetSpec``.
        """
        with patch.object(r2_io.subprocess, "check_call") as mock_call:
            spec_io.upload_spec(spec)
        args = mock_call.call_args[0][0]
        assert args[0] == "rclone"
        assert args[1] == "copyto"
        assert "--checksum" in args
        assert args[-1] == r2_io.to_rclone_path(spec.r2.input_spec_uri())

    def test_uses_reliability_flags(self, spec: DatasetSpec) -> None:
        """Rclone command carries the same flag set as ``r2_io.upload_to_uri``.

        :param spec: Fixture-provided ``DatasetSpec``.
        """
        with patch.object(r2_io.subprocess, "check_call") as mock_call:
            spec_io.upload_spec(spec)
        args = mock_call.call_args[0][0]
        assert "--contimeout=30s" in args
        assert "--timeout=300s" in args
        assert "--retries=3" in args

    def test_returns_input_spec_uri(self, spec: DatasetSpec) -> None:
        """Return value equals ``spec.r2.input_spec_uri()``.

        :param spec: Fixture-provided ``DatasetSpec``.
        """
        with patch.object(r2_io.subprocess, "check_call"):
            result = spec_io.upload_spec(spec)
        assert result == spec.r2.input_spec_uri()

    def test_cleans_up_tempfile_on_rclone_failure(self, spec: DatasetSpec) -> None:
        """Tempfile is removed even when rclone raises CalledProcessError.

        :param spec: Fixture-provided ``DatasetSpec``.
        """
        captured_src: list[str] = []

        def _raise(args: list[str]) -> None:
            captured_src.append(args[-2])
            raise subprocess.CalledProcessError(1, args)

        with patch.object(r2_io.subprocess, "check_call", side_effect=_raise):
            with pytest.raises(subprocess.CalledProcessError):
                spec_io.upload_spec(spec)

        assert captured_src, "rclone was not invoked"
        assert not Path(captured_src[0]).exists()

    def test_cleans_up_tempfile_on_success(self, spec: DatasetSpec) -> None:
        """Tempfile is removed after a successful rclone copy.

        :param spec: Fixture-provided ``DatasetSpec``.
        """
        captured_src: list[str] = []

        def _record(args: list[str]) -> None:
            captured_src.append(args[-2])

        with patch.object(r2_io.subprocess, "check_call", side_effect=_record):
            spec_io.upload_spec(spec)

        assert captured_src, "rclone was not invoked"
        assert not Path(captured_src[0]).exists()

    def test_uploaded_content_matches_spec_dump(self, spec: DatasetSpec) -> None:
        """The bytes handed to rclone are ``spec.model_dump_json(indent=2)``.

        :param spec: Fixture-provided ``DatasetSpec``.
        """
        captured: dict[str, str] = {}

        def _record(args: list[str]) -> None:
            captured["content"] = Path(args[-2]).read_text(encoding="utf-8")

        with patch.object(r2_io.subprocess, "check_call", side_effect=_record):
            spec_io.upload_spec(spec)

        assert captured["content"] == spec.model_dump_json(indent=2)
