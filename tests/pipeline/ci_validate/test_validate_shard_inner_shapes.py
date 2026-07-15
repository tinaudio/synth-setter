"""Inner-shape validation tests for ``synth_setter.pipeline.ci.validate_shard``.

These pin the per-column inner-shape check the Lance validator performs: each
fixed-shape tensor column's inner dims must match the writer's source-of-truth
shape helpers in ``synth_setter.data.vst.shapes``. This file isolates the
field-specific failure modes (wrong channels, wrong time samples, wrong mel
``n_frames``, wrong ``num_params``); ``test_validate_shard_lance.py`` carries the
row-count / dtype / metadata cases.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from synth_setter.data.vst.shapes import DATASET_FIELD_DTYPES
from synth_setter.pipeline.ci.validate_shard import validate_shard
from synth_setter.pipeline.data.lance_shard import (
    lance_schema,
    record_batch_from_arrays,
    write_lance_dataset,
)
from synth_setter.pipeline.schemas.spec import DatasetSpec, OutputFormat
from tests.helpers.finalize_shards import smoke_shard_metadata

_VALID_AUDIO_CHANNELS = 2
_VALID_AUDIO_SAMPLES_PER_ROW = 176400
_VALID_MEL_INNER_SHAPE: tuple[int, int, int] = (2, 128, 401)
_VALID_PARAM_LENGTH = 92


def _write_lance_with_shapes(
    path: Path, spec: DatasetSpec, shapes: dict[str, tuple[int, ...]]
) -> None:
    """Write a Lance shard with one zero-filled tensor column per ``shapes`` entry.

    :param path: Filesystem path where the Lance dataset is written.
    :param spec: Spec whose first shard's seed the embedded metadata must match.
    :param shapes: Mapping from column name to its full ``(N, ...)`` shape.
    :returns: ``None``.
    :rtype: None
    """
    render = spec.render.model_copy(update={"base_seed": spec.shards[0].seed})
    schema = lance_schema(shapes, smoke_shard_metadata(render))
    arrays = {
        field: np.zeros(shape, dtype=DATASET_FIELD_DTYPES[field])
        for field, shape in shapes.items()
    }
    write_lance_dataset(path, schema, [record_batch_from_arrays(arrays, schema)])


def _valid_default_shapes(spec: DatasetSpec) -> dict[str, tuple[int, ...]]:
    """Return the canonical ``(N, ...)`` shapes the writer would emit for ``spec``.

    Mirrors the shapes ``_expected_dataset_shapes`` computes in the validator under test
    so that each test case can override exactly one field while leaving the other two
    correct, isolating the failure mode under test.

    :param spec: The dataset spec whose ``render`` and ``num_params`` drive the shapes.
    :returns: Mapping with one entry per writer-emitted dataset, each value matching the
        full ``(N, ...)`` shape the Lance writer is expected to produce.
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

    The sample rate (``44100``) and duration (``4.0``) make audio time-samples
    ``176400`` and mel ``n_frames`` ``401``, lining up with ``_VALID_*`` module
    constants so that overriding a single dimension in a test is unambiguous.

    :param tmp_path: pytest-provided temp directory used for the fake VST3 bundle.
    :param monkeypatch: pytest monkeypatch to freeze git/timestamp factories the spec
        validators consult — keeps spec construction deterministic on machines without
        the repo's git metadata available to the schema layer.
    :returns: A spec whose render's samples_per_shard / channels / sample_rate /
        signal_duration_seconds match this module's ``_VALID_*`` constants.
    :rtype: DatasetSpec
    """
    fixed_now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._get_git_sha", lambda: "a" * 40)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._is_repo_dirty", lambda: False)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._utc_now", lambda: fixed_now)

    contents = tmp_path / "FakePlugin.vst3" / "Contents"
    contents.mkdir(parents=True)
    (contents / "moduleinfo.json").write_text('{"Version": "1.3.4"}')

    return DatasetSpec(
        task_name="test-dataset",
        output_format=OutputFormat.LANCE,
        train_val_test_sizes=(10, 0, 0),
        base_seed=42,
        r2={"bucket": "intermediate-data"},  # type: ignore[arg-type]
        render={
            "plugin_path": str(tmp_path / "FakePlugin.vst3"),
            "plugin_state_path": "presets/surge-base.vstpreset",
            "param_spec_name": "surge_simple",
            "renderer_version": "1.3.4",
            "sample_rate": 44100,
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
        """All three columns at canonical writer shapes produce no errors.

        :param real_spec: The standard test spec.
        :param tmp_path: pytest-provided temp directory for the shard.
        :returns: ``None``.
        :rtype: None
        """
        shard_path = tmp_path / "shard-000000.lance"
        _write_lance_with_shapes(shard_path, real_spec, _valid_default_shapes(real_spec))

        assert validate_shard(shard_path, real_spec) == []

    def test_validate_shard_rejects_wrong_channels(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """Audio written with three channels instead of two surfaces an inner-shape error.

        :param real_spec: The standard test spec.
        :param tmp_path: pytest-provided temp directory for the shard.
        :returns: ``None``.
        :rtype: None
        """
        shapes = _valid_default_shapes(real_spec)
        n = real_spec.render.samples_per_shard
        shapes["audio"] = (n, 3, _VALID_AUDIO_SAMPLES_PER_ROW)
        shard_path = tmp_path / "shard-000000.lance"
        _write_lance_with_shapes(shard_path, real_spec, shapes)

        errors = validate_shard(shard_path, real_spec)

        assert len(errors) == 1
        assert "audio" in errors[0]
        assert str((3, _VALID_AUDIO_SAMPLES_PER_ROW)) in errors[0]
        assert str((_VALID_AUDIO_CHANNELS, _VALID_AUDIO_SAMPLES_PER_ROW)) in errors[0]

    def test_validate_shard_rejects_wrong_time_samples(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """Audio with the wrong trailing time-samples dim surfaces an inner-shape error.

        :param real_spec: The standard test spec.
        :param tmp_path: pytest-provided temp directory for the shard.
        :returns: ``None``.
        :rtype: None
        """
        shapes = _valid_default_shapes(real_spec)
        n = real_spec.render.samples_per_shard
        wrong_time = _VALID_AUDIO_SAMPLES_PER_ROW + 1
        shapes["audio"] = (n, _VALID_AUDIO_CHANNELS, wrong_time)
        shard_path = tmp_path / "shard-000000.lance"
        _write_lance_with_shapes(shard_path, real_spec, shapes)

        errors = validate_shard(shard_path, real_spec)

        assert len(errors) == 1
        assert "audio" in errors[0]
        assert str((_VALID_AUDIO_CHANNELS, wrong_time)) in errors[0]
        assert str((_VALID_AUDIO_CHANNELS, _VALID_AUDIO_SAMPLES_PER_ROW)) in errors[0]

    def test_validate_shard_rejects_wrong_n_frames_mel(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """Mel with the wrong trailing ``n_frames`` dim surfaces an inner-shape error.

        :param real_spec: The standard test spec.
        :param tmp_path: pytest-provided temp directory for the shard.
        :returns: ``None``.
        :rtype: None
        """
        shapes = _valid_default_shapes(real_spec)
        n = real_spec.render.samples_per_shard
        valid_channels, valid_n_mels, valid_n_frames = _VALID_MEL_INNER_SHAPE
        wrong_n_frames = valid_n_frames + 1
        shapes["mel_spec"] = (n, valid_channels, valid_n_mels, wrong_n_frames)
        shard_path = tmp_path / "shard-000000.lance"
        _write_lance_with_shapes(shard_path, real_spec, shapes)

        errors = validate_shard(shard_path, real_spec)

        assert len(errors) == 1
        assert "mel_spec" in errors[0]
        assert str((valid_channels, valid_n_mels, wrong_n_frames)) in errors[0]
        assert str((valid_channels, valid_n_mels, valid_n_frames)) in errors[0]

    def test_validate_shard_rejects_wrong_num_params(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """Param-array with the wrong width surfaces an inner-shape error.

        :param real_spec: The standard test spec.
        :param tmp_path: pytest-provided temp directory for the shard.
        :returns: ``None``.
        :rtype: None
        """
        shapes = _valid_default_shapes(real_spec)
        n = real_spec.render.samples_per_shard
        wrong_width = _VALID_PARAM_LENGTH + 1
        shapes["param_array"] = (n, wrong_width)
        shard_path = tmp_path / "shard-000000.lance"
        _write_lance_with_shapes(shard_path, real_spec, shapes)

        errors = validate_shard(shard_path, real_spec)

        assert len(errors) == 1
        assert "param_array" in errors[0]
        assert str((wrong_width,)) in errors[0]
        assert str((_VALID_PARAM_LENGTH,)) in errors[0]
