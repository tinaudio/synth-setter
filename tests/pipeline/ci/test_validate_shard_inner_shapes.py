"""Inner-shape validation tests for ``synth_setter.pipeline.ci.validate_shard``.

These tests pin the per-dataset full-shape check the validator gained when
the HDF5 path was tightened to compare each dataset's ``.shape`` tuple to
the writer's source-of-truth shape helpers in
``synth_setter.data.vst.shapes`` (instead of only its ``shape[0]`` row
count). The older row-count tests live alongside in
``test_validate_shard.py``; this file isolates the new failure modes (wrong
channels, wrong time samples, wrong mel ``n_frames``, wrong ``num_params``)
in a sibling that is *not* on the pydoclint exclude list, so the new
helpers get full sphinx-docstring coverage from day one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest

from synth_setter.pipeline.ci.validate_shard import validate_shard
from synth_setter.pipeline.schemas.spec import DatasetSpec

_VALID_AUDIO_CHANNELS = 2
_VALID_AUDIO_SAMPLES_PER_ROW = 64000
_VALID_MEL_INNER_SHAPE: tuple[int, int, int] = (2, 128, 401)
_VALID_PARAM_LENGTH = 92


def _write_h5_with_shapes(path: Path, shapes: dict[str, tuple[int, ...]]) -> None:
    """Create a minimal HDF5 file with one zero-filled ``float32`` dataset per entry.

    :param path: Filesystem path where the HDF5 file will be written.
    :param shapes: Mapping from dataset name to its exact ``.shape`` tuple. Each entry
        becomes a top-level dataset; no attributes or groups are written.
    :returns: ``None``.
    :rtype: None
    """
    with h5py.File(path, "w") as f:
        for name, shape in shapes.items():
            f.create_dataset(name, shape=shape, dtype=np.float32)


def _valid_default_shapes(spec: DatasetSpec) -> dict[str, tuple[int, ...]]:
    """Return the canonical ``(N, ...)`` shapes the writer would emit for ``spec``.

    Mirrors the shapes ``_expected_dataset_shapes`` computes in the validator under test
    so that each test case can override exactly one field while leaving the other two
    correct, isolating the failure mode under test.

    :param spec: The dataset spec whose ``render`` and ``num_params`` drive the shapes.
    :returns: Mapping with one entry per writer-emitted dataset, each value matching the
        full ``(N, ...)`` shape the HDF5 writer is expected to produce.
    :rtype: dict[str, tuple[int, ...]]
    """
    n = spec.render.samples_per_shard
    return {
        "audio": (n, _VALID_AUDIO_CHANNELS, _VALID_AUDIO_SAMPLES_PER_ROW),
        "mel_spec": (n, *_VALID_MEL_INNER_SHAPE),
        "param_array": (n, _VALID_PARAM_LENGTH),
    }


@pytest.fixture()
def real_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DatasetSpec:
    """Build a ``DatasetSpec`` whose render config produces known-good inner shapes.

    The sample rate (``16000``) and duration (``4.0``) make audio time-samples
    ``64000`` and mel ``n_frames`` ``401``, lining up with ``_VALID_*`` module
    constants so that overriding a single dimension in a test is unambiguous.

    :param tmp_path: pytest-provided temp directory used for the fake VST3 bundle.
    :param monkeypatch: pytest monkeypatch to freeze git/timestamp factories the spec
        validators consult — keeps spec construction deterministic on machines without
        the repo's git metadata available to the schema layer.
    :returns: A spec whose render's samples_per_shard / channels / sample_rate /
        signal_duration_seconds match this module's ``_VALID_*`` constants.
    :rtype: DatasetSpec
    """
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
        r2={"bucket": "intermediate-data"},  # type: ignore[arg-type]
        render={
            "plugin_path": str(tmp_path / "FakePlugin.vst3"),
            "preset_path": "presets/surge-base.vstpreset",
            "param_spec_name": "surge_simple",
            "renderer_version": "1.3.4",
            "sample_rate": 16000,
            "channels": _VALID_AUDIO_CHANNELS,
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "samples_per_render_batch": 32,
            "samples_per_shard": 10,
            "gui_toggle_cadence": "never",
        },  # type: ignore[arg-type]
    )


class TestInnerShapeValidation:
    """Inner-shape (``(N, C, time)`` / ``(N, C, n_mels, n_frames)`` / ``(N, P)``) checks."""

    def test_valid_full_shapes_pass(self, real_spec: DatasetSpec, tmp_path: Path) -> None:
        """All three datasets at canonical writer shapes produce no errors.

        :param real_spec: The standard test spec.
        :param tmp_path: pytest-provided temp directory for the shard file.
        :returns: ``None``.
        :rtype: None
        """
        shard_path = tmp_path / "shard-000000.h5"
        _write_h5_with_shapes(shard_path, _valid_default_shapes(real_spec))

        assert validate_shard(shard_path, real_spec) == []

    def test_validate_shard_rejects_wrong_channels(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """Audio written with three channels instead of two surfaces a shape-mismatch error.

        :param real_spec: The standard test spec.
        :param tmp_path: pytest-provided temp directory for the shard file.
        :returns: ``None``.
        :rtype: None
        """
        shapes = _valid_default_shapes(real_spec)
        n = real_spec.render.samples_per_shard
        shapes["audio"] = (n, 3, _VALID_AUDIO_SAMPLES_PER_ROW)
        shard_path = tmp_path / "shard-000000.h5"
        _write_h5_with_shapes(shard_path, shapes)

        errors = validate_shard(shard_path, real_spec)

        assert len(errors) == 1
        assert "audio" in errors[0]
        assert "shape" in errors[0]
        assert str((n, 3, _VALID_AUDIO_SAMPLES_PER_ROW)) in errors[0]
        assert str((n, _VALID_AUDIO_CHANNELS, _VALID_AUDIO_SAMPLES_PER_ROW)) in errors[0]

    def test_validate_shard_rejects_wrong_time_samples(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """Audio with the wrong trailing time-samples dim surfaces a shape-mismatch error.

        :param real_spec: The standard test spec.
        :param tmp_path: pytest-provided temp directory for the shard file.
        :returns: ``None``.
        :rtype: None
        """
        shapes = _valid_default_shapes(real_spec)
        n = real_spec.render.samples_per_shard
        wrong_time = _VALID_AUDIO_SAMPLES_PER_ROW + 1
        shapes["audio"] = (n, _VALID_AUDIO_CHANNELS, wrong_time)
        shard_path = tmp_path / "shard-000000.h5"
        _write_h5_with_shapes(shard_path, shapes)

        errors = validate_shard(shard_path, real_spec)

        assert len(errors) == 1
        assert "audio" in errors[0]
        assert str((n, _VALID_AUDIO_CHANNELS, wrong_time)) in errors[0]
        assert str((n, _VALID_AUDIO_CHANNELS, _VALID_AUDIO_SAMPLES_PER_ROW)) in errors[0]

    def test_validate_shard_rejects_wrong_n_frames_mel(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """Mel with the wrong trailing ``n_frames`` dim surfaces a shape-mismatch error.

        :param real_spec: The standard test spec.
        :param tmp_path: pytest-provided temp directory for the shard file.
        :returns: ``None``.
        :rtype: None
        """
        shapes = _valid_default_shapes(real_spec)
        n = real_spec.render.samples_per_shard
        valid_channels, valid_n_mels, valid_n_frames = _VALID_MEL_INNER_SHAPE
        wrong_n_frames = valid_n_frames + 1
        shapes["mel_spec"] = (n, valid_channels, valid_n_mels, wrong_n_frames)
        shard_path = tmp_path / "shard-000000.h5"
        _write_h5_with_shapes(shard_path, shapes)

        errors = validate_shard(shard_path, real_spec)

        assert len(errors) == 1
        assert "mel_spec" in errors[0]
        assert str((n, valid_channels, valid_n_mels, wrong_n_frames)) in errors[0]
        assert str((n, valid_channels, valid_n_mels, valid_n_frames)) in errors[0]

    def test_validate_shard_rejects_wrong_num_params(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """Param-array with the wrong width surfaces a shape-mismatch error.

        :param real_spec: The standard test spec.
        :param tmp_path: pytest-provided temp directory for the shard file.
        :returns: ``None``.
        :rtype: None
        """
        shapes = _valid_default_shapes(real_spec)
        n = real_spec.render.samples_per_shard
        wrong_width = _VALID_PARAM_LENGTH + 1
        shapes["param_array"] = (n, wrong_width)
        shard_path = tmp_path / "shard-000000.h5"
        _write_h5_with_shapes(shard_path, shapes)

        errors = validate_shard(shard_path, real_spec)

        assert len(errors) == 1
        assert "param_array" in errors[0]
        assert str((n, wrong_width)) in errors[0]
        assert str((n, _VALID_PARAM_LENGTH)) in errors[0]

    def test_validate_shard_row_count_mismatch_uses_full_shape_format(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """A wrong row count is now reported as a full-shape mismatch, not ``"X rows"``.

        Pins the new error-message format so callers grepping for old ``"rows, expected"``
        wording have a single place to update. Verifies that the format change applies to
        the row-count failure mode as well — it's the first-dim case of the same check.

        :param real_spec: The standard test spec.
        :param tmp_path: pytest-provided temp directory for the shard file.
        :returns: ``None``.
        :rtype: None
        """
        shapes = _valid_default_shapes(real_spec)
        wrong_n = real_spec.render.samples_per_shard + 5
        shapes["audio"] = (wrong_n, _VALID_AUDIO_CHANNELS, _VALID_AUDIO_SAMPLES_PER_ROW)
        shard_path = tmp_path / "shard-000000.h5"
        _write_h5_with_shapes(shard_path, shapes)

        errors = validate_shard(shard_path, real_spec)

        assert len(errors) == 1
        assert "audio" in errors[0]
        assert "shape" in errors[0]
        assert "rows" not in errors[0]
