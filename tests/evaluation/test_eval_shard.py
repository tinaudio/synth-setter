"""Behavioral tests for rendering one eval-table shard from the predictions table."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
from pedalboard.io import AudioFile

from synth_setter.data.vst import param_specs
from synth_setter.data.vst.param_spec import ParameterValue
from synth_setter.data.vst.renderers import NonFiniteAudioError
from synth_setter.evaluation.eval_shard import main, render_eval_rows
from synth_setter.evaluation.predict_vst_audio import RenderFn
from synth_setter.evaluation.prediction_table import prediction_rows, write_prediction_table
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.data.lance_shard import read_shard_metadata
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.synth_spec import SynthName, SynthSpec

_PARAM_SPEC_NAME = ParamSpecName("torchsynth_simple")
_SPEC = param_specs[_PARAM_SPEC_NAME]
_SAMPLE_RATE = 22_050
_CHANNELS = 2
# The spec's note window spans four seconds, so shorter renders would truncate the note.
_DURATION_SECONDS = 4.0
_SAMPLES = int(_SAMPLE_RATE * _DURATION_SECONDS)


def _render_config(
    *,
    sample_offset: int = 0,
    samples_per_shard: int = 1,
    signal_duration_seconds: float = _DURATION_SECONDS,
) -> RenderConfig:
    """Build an in-process TorchSynth render config, so no plugin host is needed.

    :param sample_offset: Row id of the shard's first row.
    :param samples_per_shard: Number of rows the shard renders.
    :param signal_duration_seconds: Length of every rendered clip.
    :returns: Render config for the ``torchsynth_simple`` spec.
    """
    return RenderConfig(
        synth=SynthSpec(
            name=SynthName("torchsynth_simple"),
            param_spec_name=_PARAM_SPEC_NAME,
            plugin_path="torchsynth",
            plugin_state_path="",
            synth_version="1.0.2",
        ),
        renderer_backend="torchsynth",
        sample_rate=_SAMPLE_RATE,
        channels=_CHANNELS,
        velocity=100,
        signal_duration_seconds=signal_duration_seconds,
        min_loudness=-55.0,
        samples_per_render_batch=1,
        samples_per_shard=samples_per_shard,
        sample_offset=sample_offset,
        gui_toggle_cadence="never",
    )


# Fake-renderer tests use a tenth-of-a-second clip: the clip content is scripted, not rendered.
_SHORT_CONFIG = _render_config(signal_duration_seconds=0.1)


def _short_clip(value: float) -> np.ndarray:
    """Build a constant clip in the shape ``_SHORT_CONFIG`` renders.

    :param value: Sample value filling the clip.
    :returns: Clip shaped ``(channels, samples)``.
    """
    return np.full((_CHANNELS, int(_SAMPLE_RATE * 0.1)), value, dtype=np.float32)


def _scripted_render(*outcomes: np.ndarray | Exception) -> RenderFn:
    r"""Build a renderer that replays ``outcomes`` call by call, raising the exceptions.

    :param \*outcomes: Clip to return, or exception to raise, for each successive render.
    :returns: Renderer standing in for a synth at the render boundary.
    """
    remaining = iter(outcomes)

    def render(
        synth_params: Mapping[str, ParameterValue], pitch: int, window: tuple[float, float]
    ) -> np.ndarray:
        outcome = next(remaining)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return render


def _prediction_table(predictions: np.ndarray, targets: np.ndarray | None) -> pa.Table:
    """Build prediction-table rows numbered from 0.

    :param predictions: Model-space predictions, one row per sample.
    :param targets: Model-space ground truth, or ``None`` for a table without targets.
    :returns: Rows as a reader of ``predictions.lance`` sees them.
    """
    rows = prediction_rows(predictions, targets, _SPEC)
    return rows.add_column(0, "row_id", pa.array(range(rows.num_rows), pa.int64()))


def _zero_rows(num_rows: int) -> pa.Table:
    """Build table rows whose predictions and targets are all the midpoint patch.

    :param num_rows: Number of rows.
    :returns: Prediction-table rows with matching targets.
    """
    zeros = np.zeros((num_rows, _SPEC.encoded_width), dtype=np.float32)
    return _prediction_table(zeros, zeros.copy())


def _column(rows: pa.Table | pa.RecordBatch, name: str) -> np.ndarray:
    """Read one tensor column back as a ``(rows, ...)`` array.

    :param rows: Arrow rows holding the column.
    :param name: Column to read.
    :returns: Stacked row values.
    """
    table = rows if isinstance(rows, pa.Table) else pa.Table.from_batches([rows])
    return table.column(name).combine_chunks().to_numpy_ndarray()


def _write_table(root: Path, predictions: np.ndarray, targets: np.ndarray) -> Path:
    """Write a predictions table the way the predict step does.

    :param root: Directory hosting the table.
    :param predictions: Model-space predictions, one row per sample.
    :param targets: Model-space ground truth for the same rows.
    :returns: Location of the written table.
    """
    table_uri = root / "predictions.lance"
    write_prediction_table(table_uri, [prediction_rows(predictions, targets, _SPEC)])
    return table_uri


def _cli_args(table_uri: Path, shard_path: Path, config: RenderConfig) -> list[str]:
    """Build the worker argv for one shard.

    :param table_uri: Predictions table to render from.
    :param shard_path: Destination shard dataset.
    :param config: Render settings, including the shard's row range.
    :returns: Arguments for :func:`main`.
    """
    args = [str(table_uri), str(shard_path)]
    # Same flag encoding as the generate-dataset dispatcher's worker argv.
    for key, value in config.model_dump().items():
        encoded = str(value) if isinstance(value, str | int | float) else json.dumps(value)
        args.extend([f"--{key}", encoded])
    return args


@pytest.fixture(scope="module")
def rendered_shard_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Render rows 1-2 of a three-row predictions table through the real CLI and renderer.

    Row 1 predicts its target exactly; row 2 predicts a different patch.

    :param tmp_path_factory: Hosts the predictions table and the rendered shard.
    :returns: Location of the rendered shard.
    """
    root = tmp_path_factory.mktemp("eval-shard")
    width = _SPEC.encoded_width
    predictions = np.zeros((3, width), dtype=np.float32)
    predictions[2] = 0.3
    table_uri = _write_table(root, predictions, np.zeros((3, width), dtype=np.float32))
    shard_path = root / "shard-000001.lance"

    main(_cli_args(table_uri, shard_path, _render_config(sample_offset=1, samples_per_shard=2)))

    return shard_path


@pytest.fixture(scope="module")
def rendered_shard(rendered_shard_path: Path) -> pa.Table:
    """Read the rendered shard's rows.

    :param rendered_shard_path: Location of the rendered shard.
    :returns: Rows 1-2 as a reader sees them.
    """
    return lance.dataset(str(rendered_shard_path)).to_table()


def test_main_renders_only_the_shards_row_range(rendered_shard: pa.Table) -> None:
    """A shard holds exactly its row range, keyed by the predictions table's row ids.

    :param rendered_shard: Rows 1-2 rendered from a three-row table.
    """
    assert rendered_shard.column("row_id").to_pylist() == [1, 2]


def test_main_renders_audible_audio_for_both_sides(rendered_shard: pa.Table) -> None:
    """Prediction and target audio are full-length clips with signal in them.

    :param rendered_shard: Rows 1-2 rendered from a three-row table.
    """
    pred_audio = _column(rendered_shard, "pred_audio")
    target_audio = _column(rendered_shard, "target_audio")

    assert pred_audio.shape == (2, _CHANNELS, _SAMPLES)
    assert target_audio.shape == (2, _CHANNELS, _SAMPLES)
    assert np.abs(pred_audio[0]).max() > 0.0
    assert np.abs(target_audio[0]).max() > 0.0


def test_main_exact_prediction_renders_the_same_audio_as_its_target(
    rendered_shard: pa.Table,
) -> None:
    """A prediction equal to its target goes through the same renderer to the same clip.

    :param rendered_shard: Rows 1-2 rendered from a three-row table.
    """
    np.testing.assert_array_equal(
        _column(rendered_shard, "pred_audio")[0], _column(rendered_shard, "target_audio")[0]
    )


def test_main_wrong_prediction_renders_different_audio_from_its_target(
    rendered_shard: pa.Table,
) -> None:
    """Each side is rendered from its own params, so a wrong prediction sounds different.

    :param rendered_shard: Rows 1-2 rendered from a three-row table.
    """
    pred_audio = _column(rendered_shard, "pred_audio")[1]
    target_audio = _column(rendered_shard, "target_audio")[1]

    assert not np.array_equal(pred_audio, target_audio)
    np.testing.assert_array_equal(target_audio, _column(rendered_shard, "target_audio")[0])


def test_main_mel_spectrograms_follow_their_own_audio(rendered_shard: pa.Table) -> None:
    """Mel columns are per-side: equal where the audio is equal, different where it differs.

    :param rendered_shard: Rows 1-2 rendered from a three-row table.
    """
    pred_mel = _column(rendered_shard, "pred_mel_spec")
    target_mel = _column(rendered_shard, "target_mel_spec")

    assert pred_mel.shape[:3] == (2, _CHANNELS, 128)
    np.testing.assert_array_equal(pred_mel[0], target_mel[0])
    assert not np.array_equal(pred_mel[1], target_mel[1])


def test_main_copies_param_columns_from_the_predictions_table(rendered_shard: pa.Table) -> None:
    """The shard is self-contained: it carries the params each clip was rendered from.

    :param rendered_shard: Rows 1-2 rendered from a three-row table.
    """
    width = _SPEC.encoded_width

    np.testing.assert_array_equal(
        _column(rendered_shard, "pred_params_raw")[1], np.full(width, 0.3, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        _column(rendered_shard, "target_params"), np.zeros((2, width), dtype=np.float32)
    )
    assert _column(rendered_shard, "pred_params_effective").shape == (2, width)


def test_main_previews_decode_as_audio_of_the_clip_length(rendered_shard: pa.Table) -> None:
    """Preview bytes are a playable MP3 of the rendered clip.

    :param rendered_shard: Rows 1-2 rendered from a three-row table.
    """
    preview = rendered_shard.column("pred_audio_mp3")[0].as_py()

    with AudioFile(io.BytesIO(preview)) as decoded:
        assert decoded.samplerate == _SAMPLE_RATE
        assert decoded.frames == pytest.approx(_SAMPLES, abs=_SAMPLE_RATE // 10)


def test_main_preview_columns_are_tagged_as_mp3_audio(rendered_shard: pa.Table) -> None:
    """Table viewers find playable previews by the column's MIME type.

    :param rendered_shard: Rows 1-2 rendered from a three-row table.
    """
    for name in ("pred_audio_mp3", "target_audio_mp3"):
        assert rendered_shard.schema.field(name).metadata == {b"mime_type": b"audio/mpeg"}


def test_main_shard_schema_carries_the_render_metadata(rendered_shard: pa.Table) -> None:
    """Shard readers recover the render settings from the shard itself.

    :param rendered_shard: Rows 1-2 rendered from a three-row table.
    """
    metadata = read_shard_metadata(rendered_shard.schema)

    assert metadata.sample_rate == _SAMPLE_RATE
    assert metadata.sample_offset == 1


def test_main_real_renders_are_not_flagged_non_finite(rendered_shard: pa.Table) -> None:
    """Ordinary renders leave both non-finite flags clear.

    :param rendered_shard: Rows 1-2 rendered from a three-row table.
    """
    assert rendered_shard.column("pred_non_finite").to_pylist() == [False, False]
    assert rendered_shard.column("target_non_finite").to_pylist() == [False, False]


def test_render_eval_rows_keeps_clipped_audio_unchanged() -> None:
    """A clipping prediction is evidence about the model, so it is stored as rendered."""
    render = _scripted_render(_short_clip(2.0), _short_clip(0.5))

    shard = render_eval_rows(_zero_rows(1), _SPEC, render, _SHORT_CONFIG)

    assert _column(shard, "pred_audio").max() == 2.0


def test_render_eval_rows_silent_render_is_kept_as_a_row() -> None:
    """A silent prediction still yields a row; nothing is dropped or re-drawn."""
    render = _scripted_render(*[_short_clip(0.0)] * 4)

    shard = render_eval_rows(_zero_rows(2), _SPEC, render, _SHORT_CONFIG)

    assert shard.column("row_id").to_pylist() == [0, 1]
    assert np.isfinite(_column(shard, "pred_mel_spec")).all()


def test_render_eval_rows_non_finite_prediction_is_stored_as_flagged_silence() -> None:
    """One unrenderable prediction cannot fail its shard: it becomes silence with a flag."""
    render = _scripted_render(NonFiniteAudioError("nan"), _short_clip(0.5))

    shard = render_eval_rows(_zero_rows(1), _SPEC, render, _SHORT_CONFIG)

    assert _column(shard, "pred_audio").max() == 0.0
    assert _column(shard, "target_audio").max() == 0.5
    assert shard.column("pred_non_finite").to_pylist() == [True]
    assert shard.column("target_non_finite").to_pylist() == [False]


def test_render_eval_rows_non_finite_target_is_flagged_on_the_target_side() -> None:
    """The flag names the side that failed, so a bad target is not blamed on the model."""
    render = _scripted_render(_short_clip(0.5), NonFiniteAudioError("nan"))

    shard = render_eval_rows(_zero_rows(1), _SPEC, render, _SHORT_CONFIG)

    assert shard.column("pred_non_finite").to_pylist() == [False]
    assert shard.column("target_non_finite").to_pylist() == [True]


def test_render_eval_rows_table_without_targets_raises_value_error() -> None:
    """Both sides are re-rendered, so a table with no ground-truth patch cannot be used."""
    rows = _prediction_table(np.zeros((1, _SPEC.encoded_width), dtype=np.float32), None)

    with pytest.raises(ValueError, match="target_params"):
        render_eval_rows(rows, _SPEC, _scripted_render(), _SHORT_CONFIG)


def test_main_row_range_past_the_table_end_raises_value_error(tmp_path: Path) -> None:
    """A shard asking for rows the table does not hold fails instead of rendering fewer.

    :param tmp_path: Hosts the predictions table.
    """
    rows = np.zeros((1, _SPEC.encoded_width), dtype=np.float32)
    table_uri = _write_table(tmp_path, rows, rows)

    with pytest.raises(ValueError, match="rows 0-1"):
        main(_cli_args(table_uri, tmp_path / "shard.lance", _render_config(samples_per_shard=2)))


def test_main_rerendering_a_shard_replaces_the_earlier_attempt(tmp_path: Path) -> None:
    """A retried shard overwrites its earlier attempt instead of failing or doubling rows.

    :param tmp_path: Hosts the predictions table and the shard.
    """
    rows = np.zeros((1, _SPEC.encoded_width), dtype=np.float32)
    shard_path = tmp_path / "shard-000000.lance"
    cli_args = _cli_args(_write_table(tmp_path, rows, rows), shard_path, _render_config())
    main(cli_args)

    main(cli_args)

    assert lance.dataset(str(shard_path)).to_table().column("row_id").to_pylist() == [0]
