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
    """``upload_spec`` uploads the spec to its R2 URI.

    State-based tests run rclone against the ``fake_r2_remote`` fixture (local-
    typed backend rooted at a tmp dir) and assert on the materialized object's
    bytes; one mock-based test below pins the rclone reliability-flag set,
    which state-based tests cannot observe (see issue #1124).
    """

    def test_lands_spec_at_input_spec_uri(self, spec: DatasetSpec, fake_r2_remote: Path) -> None:
        """Upload writes the spec to the path implied by ``spec.r2.input_spec_uri()``.

        :param spec: Fixture-provided ``DatasetSpec``.
        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        """
        spec_io.upload_spec(spec)

        landed = fake_r2_remote / spec.r2.bucket / spec.r2.prefix / INPUT_SPEC_FILENAME
        assert landed.is_file()

    def test_uploaded_content_matches_spec_dump(
        self, spec: DatasetSpec, fake_r2_remote: Path
    ) -> None:
        """The bytes landed at the R2 URI equal ``spec.model_dump_json(indent=2)``.

        :param spec: Fixture-provided ``DatasetSpec``.
        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        """
        spec_io.upload_spec(spec)

        landed = fake_r2_remote / spec.r2.bucket / spec.r2.prefix / INPUT_SPEC_FILENAME
        assert landed.read_text(encoding="utf-8") == spec.model_dump_json(indent=2)

    def test_returns_input_spec_uri(self, spec: DatasetSpec, fake_r2_remote: Path) -> None:
        """Return value equals ``spec.r2.input_spec_uri()``.

        :param spec: Fixture-provided ``DatasetSpec``.
        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        """
        result = spec_io.upload_spec(spec)
        assert result == spec.r2.input_spec_uri()

    def test_cleans_up_tempfile_on_success(self, spec: DatasetSpec, fake_r2_remote: Path) -> None:
        """The exact tempfile passed to rclone is unlinked after a successful copy.

        Captures the source path via a pass-through wrapper around the real
        ``subprocess.check_call`` (so the rclone copy still lands the spec on the
        fake-local remote), then asserts the captured path no longer exists.
        The capture is per-test so it's safe under ``pytest -n auto`` — unlike a
        ``tempfile.gettempdir()`` glob, which races with sibling workers writing
        their own ``tmp*.json`` files in the shared system tempdir.

        :param spec: Fixture-provided ``DatasetSpec``.
        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        """
        captured_src: list[str] = []
        real_check_call = subprocess.check_call

        def _capture_then_passthrough(args: list[str]) -> None:
            captured_src.append(args[-2])
            real_check_call(args)

        with patch.object(r2_io.subprocess, "check_call", side_effect=_capture_then_passthrough):
            spec_io.upload_spec(spec)

        assert captured_src, "rclone was not invoked"
        assert not Path(captured_src[0]).exists()

    def test_cleans_up_tempfile_on_rclone_failure(self, spec: DatasetSpec) -> None:
        """Tempfile is removed even when rclone raises CalledProcessError.

        Stays mock-based: ``upload_spec``'s success path covers the cleanup-on-
        success branch (state-based above), but the error path needs a forced
        non-zero rclone exit and there's no clean way to make local-backend
        rclone fail without also breaking the captured tempfile path.

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

    def test_command_carries_rclone_reliability_flags(self, spec: DatasetSpec) -> None:
        """Pin the rclone reliability-flag set + the input_spec_uri destination.

        State-based tests cover the spec-lands-at-URI contract but cannot
        observe ``--checksum / --contimeout / --timeout / --retries`` or the
        ``rclone copyto`` verb. Losing any of those is a silent correctness
        regression — one mock-based argv assertion guards them all.

        :param spec: Fixture-provided ``DatasetSpec``.
        """
        with patch.object(r2_io.subprocess, "check_call") as mock_call:
            spec_io.upload_spec(spec)
        args = mock_call.call_args[0][0]
        assert args[0] == "rclone"
        assert args[1] == "copyto"
        assert "--checksum" in args
        assert "--contimeout=30s" in args
        assert "--timeout=300s" in args
        assert "--retries=3" in args
        assert args[-1] == r2_io.to_rclone_path(spec.r2.input_spec_uri())


def _touch_spec(data_dir: Path, task: str, run_id: str) -> Path:
    """Create an empty ``input_spec.json`` under ``data_dir/<task>/<run_id>/metadata/``.

    :param data_dir: Parent ``data/`` directory.
    :param task: Task name segment.
    :param run_id: Run id segment.
    :returns: Path to the created file.
    """
    spec_path = data_dir / task / run_id / "metadata" / INPUT_SPEC_FILENAME
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("{}", encoding="utf-8")
    return spec_path


class TestFindInputSpecs:
    """``find_input_specs`` discovers ``data/<task>/<run>/metadata/input_spec.json`` files."""

    def test_returns_empty_list_when_data_dir_missing(self, tmp_path: Path) -> None:
        """A non-existent ``data_dir`` returns ``[]`` (no exception).

        :param tmp_path: Pytest tmp dir.
        """
        assert spec_io.find_input_specs(tmp_path / "does-not-exist") == []

    def test_returns_empty_list_when_no_specs_present(self, tmp_path: Path) -> None:
        """An existing but empty ``data_dir`` returns ``[]``.

        :param tmp_path: Pytest tmp dir.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        assert spec_io.find_input_specs(data_dir) == []

    def test_finds_single_spec(self, tmp_path: Path) -> None:
        """One matching spec file is returned in a single-element list.

        :param tmp_path: Pytest tmp dir.
        """
        data_dir = tmp_path / "data"
        expected = _touch_spec(data_dir, "task-a", "task-a-20260519T120000000Z")
        assert spec_io.find_input_specs(data_dir) == [expected]

    def test_returns_sorted_matches(self, tmp_path: Path) -> None:
        """Multiple matches are returned in sorted order for determinism.

        :param tmp_path: Pytest tmp dir.
        """
        data_dir = tmp_path / "data"
        b = _touch_spec(data_dir, "task-b", "task-b-20260519T120000000Z")
        a = _touch_spec(data_dir, "task-a", "task-a-20260519T120000000Z")
        c = _touch_spec(data_dir, "task-a", "task-a-20260519T130000000Z")
        assert spec_io.find_input_specs(data_dir) == [a, c, b]

    def test_ignores_specs_outside_glob_depth(self, tmp_path: Path) -> None:
        """Specs not at ``data/<task>/<run>/metadata/input_spec.json`` depth are skipped.

        :param tmp_path: Pytest tmp dir.
        """
        data_dir = tmp_path / "data"
        # Too shallow — directly under data_dir.
        shallow = data_dir / INPUT_SPEC_FILENAME
        shallow.parent.mkdir(parents=True, exist_ok=True)
        shallow.write_text("{}", encoding="utf-8")
        # Too deep — extra level under metadata/.
        deep = data_dir / "task" / "run" / "metadata" / "nested" / INPUT_SPEC_FILENAME
        deep.parent.mkdir(parents=True, exist_ok=True)
        deep.write_text("{}", encoding="utf-8")
        # Missing the ``metadata/`` segment.
        sibling = data_dir / "task" / "run" / INPUT_SPEC_FILENAME
        sibling.parent.mkdir(parents=True, exist_ok=True)
        sibling.write_text("{}", encoding="utf-8")

        assert spec_io.find_input_specs(data_dir) == []

    def test_ignores_non_input_spec_files_in_metadata(self, tmp_path: Path) -> None:
        """Other JSON files under ``metadata/`` are not returned.

        :param tmp_path: Pytest tmp dir.
        """
        data_dir = tmp_path / "data"
        metadata_dir = data_dir / "task" / "run" / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / "report.json").write_text("{}", encoding="utf-8")
        assert spec_io.find_input_specs(data_dir) == []
