"""Render one shard of the eval table from a row range of the predictions table.

Each output row is a complete prediction/target pair — both clips, their mel spectrograms,
previews, and the params they were rendered from — so the metrics step reads one table and joins
nothing.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import NamedTuple

import lance
import numpy as np
import pyarrow as pa
from pydantic_settings import BaseSettings, CliApp, CliPositionalArg, SettingsConfigDict

from synth_setter.data.vst import param_specs
from synth_setter.data.vst.audio_preview import DEFAULT_MP3_BITRATE_KBPS, encode_audio_to_mp3
from synth_setter.data.vst.param_spec import (
    ParameterValue,
    ParamSpec,
    decode_model_output,
    require_note_params,
)
from synth_setter.data.vst.renderers import NonFiniteAudioError
from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    AUDIO_MP3_FIELD_METADATA,
    MEL_SPEC_FIELD,
    dataset_field_dtypes,
    dataset_field_shapes,
    make_spectrogram,
)
from synth_setter.evaluation.predict_vst_audio import (
    RenderFn,
    hold_render_session,
    prediction_render_request,
)
from synth_setter.evaluation.prediction_table import (
    PRED_PARAMS_EFFECTIVE_FIELD,
    PRED_PARAMS_RAW_FIELD,
    ROW_ID_FIELD,
    TARGET_PARAMS_FIELD,
)
from synth_setter.pipeline.data.lance_shard import (
    SHARD_METADATA_SCHEMA_KEY,
    tensor_array,
    write_lance_dataset,
)
from synth_setter.pipeline.schemas.spec import RenderConfig

PRED_AUDIO_FIELD = "pred_audio"
TARGET_AUDIO_FIELD = "target_audio"
PRED_MEL_SPEC_FIELD = "pred_mel_spec"
TARGET_MEL_SPEC_FIELD = "target_mel_spec"
PRED_AUDIO_MP3_FIELD = "pred_audio_mp3"
TARGET_AUDIO_MP3_FIELD = "target_audio_mp3"
PRED_NON_FINITE_FIELD = "pred_non_finite"
TARGET_NON_FINITE_FIELD = "target_non_finite"


class _ClipFields(NamedTuple):
    """Column names holding one side's rendered clip.

    .. attribute :: audio

        Rendered clip.

    .. attribute :: mel_spec

        Mel spectrogram of the clip.

    .. attribute :: audio_mp3

        MP3 preview of the clip.

    .. attribute :: non_finite

        Whether the render produced NaN or infinity.
    """

    audio: str
    mel_spec: str
    audio_mp3: str
    non_finite: str


_PRED_CLIP_FIELDS = _ClipFields(
    PRED_AUDIO_FIELD, PRED_MEL_SPEC_FIELD, PRED_AUDIO_MP3_FIELD, PRED_NON_FINITE_FIELD
)
_TARGET_CLIP_FIELDS = _ClipFields(
    TARGET_AUDIO_FIELD, TARGET_MEL_SPEC_FIELD, TARGET_AUDIO_MP3_FIELD, TARGET_NON_FINITE_FIELD
)
_PARAM_FIELDS = (PRED_PARAMS_RAW_FIELD, PRED_PARAMS_EFFECTIVE_FIELD, TARGET_PARAMS_FIELD)


class _EvalShardCliArgs(RenderConfig, BaseSettings):
    """Render configuration plus the predictions table a shard renders from.

    The shard's rows are ``sample_offset`` to ``sample_offset + samples_per_shard - 1``;
    row ids equal sample indices because the eval table is a single split.

    .. attribute :: model_config

        Pydantic settings and CLI parsing policy; snake-case flags match the argv the
        generate-dataset dispatcher builds.

    .. attribute :: predictions_uri

        Predictions table to render from.

    .. attribute :: shard_path

        Destination Lance dataset for the rendered rows.
    """

    model_config = SettingsConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        cli_kebab_case=False,
    )

    predictions_uri: CliPositionalArg[str]
    shard_path: CliPositionalArg[Path]


def eval_shard_schema(config: RenderConfig, num_params: int) -> pa.Schema:
    """Build the Arrow schema every eval shard of a run shares.

    :param config: Render settings fixing clip shapes, storage dtypes, and shard metadata.
    :param num_params: Encoded width of the parameter spec.
    :returns: Schema carrying the shard metadata that shard readers require.
    """
    shapes = dataset_field_shapes(config, num_params)
    dtypes = dataset_field_dtypes(config)

    def tensor(name: str, source_field: str, inner_shape: tuple[int, ...]) -> pa.Field:
        scalar = pa.from_numpy_dtype(dtypes[source_field])
        return pa.field(name, pa.fixed_shape_tensor(scalar, inner_shape), nullable=False)

    fields = [pa.field(ROW_ID_FIELD, pa.int64(), nullable=False)]
    fields += [
        pa.field(name, pa.fixed_shape_tensor(pa.float32(), (num_params,)), nullable=False)
        for name in _PARAM_FIELDS
    ]
    for clip in (_PRED_CLIP_FIELDS, _TARGET_CLIP_FIELDS):
        fields += [
            tensor(clip.audio, AUDIO_FIELD, shapes[AUDIO_FIELD][1:]),
            tensor(clip.mel_spec, MEL_SPEC_FIELD, shapes[MEL_SPEC_FIELD][1:]),
            pa.field(
                clip.audio_mp3, pa.binary(), nullable=False, metadata=AUDIO_MP3_FIELD_METADATA
            ),
            pa.field(clip.non_finite, pa.bool_(), nullable=False),
        ]
    metadata = config.shard_metadata().model_dump_json().encode("utf-8")
    return pa.schema(fields, metadata={SHARD_METADATA_SCHEMA_KEY: metadata})


def _read_prediction_rows(uri: str, config: RenderConfig) -> pa.Table:
    """Read a shard's consecutive rows from the predictions table, in row order.

    :param uri: Predictions table location.
    :param config: Render settings naming the shard's first row and row count.
    :returns: The shard's rows sorted by row id.
    :raises ValueError: The table does not hold every row of the shard.
    """
    first_row, end_row = config.sample_offset, config.sample_offset + config.samples_per_shard
    rows = (
        lance.dataset(uri)
        .to_table(filter=f"{ROW_ID_FIELD} >= {first_row} AND {ROW_ID_FIELD} < {end_row}")
        .sort_by(ROW_ID_FIELD)
    )
    if rows.column(ROW_ID_FIELD).to_pylist() != list(range(first_row, end_row)):
        raise ValueError(
            f"predictions table at {uri} does not hold rows {first_row}-{end_row - 1}; "
            f"it returned {rows.num_rows} of {config.samples_per_shard}"
        )
    return rows


def render_eval_rows(
    predictions: pa.Table, spec: ParamSpec, render: RenderFn, config: RenderConfig
) -> pa.RecordBatch:
    """Render prediction and target audio for every row, rejecting none.

    Silent and clipped renders are stored as rendered because they are evidence
    about the prediction; a non-finite render is stored as silence and flagged so
    one bad row cannot fail its shard.

    :param predictions: Predictions-table rows, including target params.
    :param spec: Parameter decoder for the rows.
    :param render: Row renderer from :func:`hold_render_session`.
    :param config: Render settings fixing clip shape and storage dtypes.
    :returns: Eval-table rows in the order given, in :func:`eval_shard_schema` layout.
    :raises ValueError: The rows carry no target params to re-render.
    """
    if TARGET_PARAMS_FIELD not in predictions.column_names:
        raise ValueError(
            f"predictions table has no {TARGET_PARAMS_FIELD} column; "
            "the eval table re-renders every target"
        )
    schema = eval_shard_schema(config, spec.encoded_width)
    clip_shape = tuple(schema.field(PRED_AUDIO_FIELD).type.shape)

    def render_clip(
        synth_params: Mapping[str, ParameterValue], pitch: int, note_window: tuple[float, float]
    ) -> tuple[np.ndarray, bool]:
        try:
            return render(synth_params, pitch, note_window), False
        except NonFiniteAudioError:
            return np.zeros(clip_shape, dtype=np.float32), True

    pred_clips: list[tuple[np.ndarray, bool]] = []
    target_clips: list[tuple[np.ndarray, bool]] = []
    for effective_row, target_row in zip(
        _tensor_rows(predictions, PRED_PARAMS_EFFECTIVE_FIELD),
        _tensor_rows(predictions, TARGET_PARAMS_FIELD),
        strict=True,
    ):
        pred_synth, pred_note, pred_window = prediction_render_request(effective_row, spec, config)
        target_synth, target_note_values = decode_model_output(target_row, spec)
        target_note = require_note_params(target_note_values)
        pred_clips.append(render_clip(pred_synth, int(pred_note["pitch"]), pred_window))
        target_clips.append(
            render_clip(target_synth, int(target_note["pitch"]), target_note["note_start_and_end"])
        )

    columns = {
        ROW_ID_FIELD: predictions.column(ROW_ID_FIELD).combine_chunks(),
        **{
            name: tensor_array(
                _tensor_rows(predictions, name), np.dtype(np.float32), (spec.encoded_width,)
            )
            for name in _PARAM_FIELDS
        },
        **_clip_columns(_PRED_CLIP_FIELDS, pred_clips, config),
        **_clip_columns(_TARGET_CLIP_FIELDS, target_clips, config),
    }
    return pa.record_batch([columns[name] for name in schema.names], schema=schema)


def _clip_columns(
    fields: _ClipFields, clips: list[tuple[np.ndarray, bool]], config: RenderConfig
) -> dict[str, pa.Array]:
    """Build one side's audio, mel-spectrogram, preview, and non-finite columns.

    :param fields: Column names for the side being built.
    :param clips: Per row, the rendered ``(channels, samples)`` clip and whether the
        render was non-finite.
    :param config: Render settings fixing sample rate and storage dtypes.
    :returns: The side's columns keyed by field name.
    """
    dtypes = dataset_field_dtypes(config)
    rendered = [clip for clip, _ in clips]
    audio = np.stack(rendered).astype(dtypes[AUDIO_FIELD], copy=False)
    mel = np.stack([make_spectrogram(clip, config.sample_rate) for clip in rendered])
    previews = [
        encode_audio_to_mp3(clip, config.sample_rate, DEFAULT_MP3_BITRATE_KBPS) for clip in audio
    ]
    return {
        fields.audio: tensor_array(audio, dtypes[AUDIO_FIELD], audio.shape[1:]),
        fields.mel_spec: tensor_array(mel, dtypes[MEL_SPEC_FIELD], mel.shape[1:]),
        fields.audio_mp3: pa.array(previews, pa.binary()),
        fields.non_finite: pa.array([non_finite for _, non_finite in clips], pa.bool_()),
    }


def _tensor_rows(table: pa.Table, name: str) -> np.ndarray:
    """Read a fixed-shape tensor column as a ``(rows, width)`` array.

    :param table: Table holding the column.
    :param name: Tensor column to read.
    :returns: Stacked row values.
    """
    return table.column(name).combine_chunks().to_numpy_ndarray()


def render_eval_shard(args: _EvalShardCliArgs) -> None:
    """Render one shard's rows and write them, replacing any earlier attempt.

    Rows are rendered and written ``samples_per_render_batch`` at a time, so memory
    is bounded by one render batch rather than the shard.

    :param args: Validated render configuration and predictions table.
    """
    spec = param_specs[args.param_spec_name]
    predictions = _read_prediction_rows(args.predictions_uri, args)

    def write_shard(render: RenderFn) -> None:
        def batches() -> Iterator[pa.RecordBatch]:
            for start in range(0, predictions.num_rows, args.samples_per_render_batch):
                rows = predictions.slice(start, args.samples_per_render_batch)
                yield render_eval_rows(rows, spec, render, args)

        write_lance_dataset(
            args.shard_path, eval_shard_schema(args, spec.encoded_width), batches()
        )

    hold_render_session(args, write_shard)


def main(cli_args: list[str] | None = None) -> None:
    """Parse the process request and render one eval shard.

    :param cli_args: Explicit arguments for tests; ``None`` reads ``sys.argv``.
    """
    render_eval_shard(CliApp.run(_EvalShardCliArgs, cli_args=cli_args))


if __name__ == "__main__":
    main()
