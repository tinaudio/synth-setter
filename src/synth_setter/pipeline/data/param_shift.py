"""Measure how much one parameter changes the sound, one parameter per dataset row.

Each row is assigned exactly one parameter of the render's param spec, that parameter is
redrawn from its own distribution, the patch is re-rendered through the ordinary
``AudioRenderer`` contract, and the perturbed audio is scored against the row's stored
audio. Selection is keyed on the Lance row id so the assignment is balanced across the
spec and survives a resume-cache replay unchanged.

The seven facets land in one nested ``shift`` struct column — ``shift.param``,
``shift.amount``, ``shift.audio``, and one subfield per metric.

Consumed through the ``param_shift`` entry of ``add_embeddings``'s registry, not directly.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pyarrow as pa
import structlog

from synth_setter.data.vst.seeding import rng_for_sample
from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    PARAM_ARRAY_FIELD,
    SHIFT_AMOUNT_SUBFIELD,
    SHIFT_AUDIO_SUBFIELD,
    SHIFT_FIELD,
    SHIFT_MSS_SUBFIELD,
    SHIFT_PARAM_SUBFIELD,
    SHIFT_RMS_SUBFIELD,
    SHIFT_SOT_SUBFIELD,
    SHIFT_SUBFIELD_NAMES,
    SHIFT_WMFCC_SUBFIELD,
)

if TYPE_CHECKING:
    from synth_setter.data.vst.param_spec import ParamSpec
    from synth_setter.data.vst.renderers import AudioRenderer
    from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig
    from synth_setter.pipeline.schemas.spec import RenderConfig

logger = structlog.get_logger(__name__)

ROW_ID_FIELD: str = "_rowid"
PARAM_SHIFT_INPUT_FIELDS: tuple[str, ...] = (AUDIO_FIELD, PARAM_ARRAY_FIELD, ROW_ID_FIELD)

_METRIC_SUBFIELDS: tuple[str, ...] = (
    SHIFT_RMS_SUBFIELD,
    SHIFT_SOT_SUBFIELD,
    SHIFT_WMFCC_SUBFIELD,
    SHIFT_MSS_SUBFIELD,
)
# ``audio`` is absent by design: its tensor type is taken from the dataset's own audio
# column, so the re-render is stored exactly like the render it is compared against.
_SCALAR_TYPES: dict[str, pa.DataType] = {
    SHIFT_PARAM_SUBFIELD: pa.string(),
    SHIFT_AMOUNT_SUBFIELD: pa.float32(),
    **dict.fromkeys(_METRIC_SUBFIELDS, pa.float32()),
}


@dataclass(frozen=True)
class ShiftedRow:
    """One row's redrawn parameter and the encoded patch that expresses it.

    .. attribute :: param_name

        Name of the single parameter that was redrawn.

    .. attribute :: amount

        Euclidean size of the change over the parameter's encoded span.

    .. attribute :: encoded

        Full encoded row carrying the redrawn value.
    """

    param_name: str
    amount: float
    encoded: np.ndarray


def assigned_param_index(row_id: int, num_params: int) -> int:
    """Select which parameter a row shifts, balanced across the spec.

    Lance row ids are ``fragment_id << 32 | offset``, so the modulo cycles through the
    spec once per fragment and every parameter ends up owning the same row count give or
    take one per fragment.

    :param row_id: Lance row id of the row being shifted.
    :param num_params: Positive parameter count of the render's param spec.
    :returns: Index into ``ParamSpec.encoded_slices()``.
    :raises ValueError: The spec has no parameters.
    """
    if num_params < 1:
        raise ValueError(f"param spec must have at least one parameter, got {num_params}")
    return int(row_id) % num_params


def shift_encoded_row(
    encoded: np.ndarray, spec: ParamSpec, *, param_index: int, rng: np.random.Generator
) -> ShiftedRow:
    """Redraw one parameter of an encoded row and report the size of the change.

    The replacement comes from the parameter's own ``sample``/``encode`` pair, so bounds,
    constant-value probabilities, categorical sets, and note-duration ordering all hold.
    Measuring the amount in encoded space keeps one comparable scale across continuous,
    onehot, and multi-column parameters.

    :param encoded: One encoded row, ``(spec.encoded_width,)`` with values in ``[0, 1]``.
    :param spec: Param spec the row was encoded against.
    :param param_index: Index into ``spec.encoded_slices()`` selecting the parameter.
    :param rng: Generator the replacement value is drawn from.
    :returns: The redrawn parameter's name, the change size, and the shifted row.
    :raises ValueError: The row width does not match the spec.
    """
    if encoded.shape != (spec.encoded_width,):
        raise ValueError(
            f"encoded row has shape {encoded.shape}, expected ({spec.encoded_width},)"
        )
    param, span = list(spec.encoded_slices())[param_index]
    replacement = np.asarray(param.encode(param.sample(rng)), dtype=np.float32).reshape(-1)
    shifted = np.array(encoded, dtype=np.float32)
    amount = float(np.linalg.norm(replacement - shifted[span]))
    shifted[span] = replacement
    return ShiftedRow(param_name=param.name, amount=amount, encoded=shifted)


def _render_encoded_row(
    renderer: AudioRenderer, encoded: np.ndarray, spec: ParamSpec, velocity: int
) -> np.ndarray:
    """Decode one encoded row and render it through the configured backend.

    :param renderer: Renderer session for the run's synth identity.
    :param encoded: Encoded row to render.
    :param spec: Param spec the row was encoded against.
    :param velocity: MIDI velocity the source dataset was rendered with.
    :returns: Rendered audio shaped ``(channels, samples)``.
    """
    synth_params, note_params = spec.decode(encoded)
    return renderer.render(
        synth_params,
        note_params["pitch"],
        velocity,
        note_params["note_start_and_end"],
    )


@contextmanager
def _quiet_metric_logs() -> Iterator[None]:
    """Silence the metric module's per-call loguru lines for the duration of a pass.

    The four metric helpers each log on every call, which at one call per row per metric would bury
    the pipeline's own throttled progress output.

    :yields: Control to the caller's scoring loop, with that module's per-call logging suppressed.
    :ytype: None
    """
    from loguru import logger as loguru_logger

    module = "synth_setter.evaluation.compute_audio_metrics"
    loguru_logger.disable(module)
    try:
        yield
    finally:
        loguru_logger.enable(module)


def _shift_metrics(original: np.ndarray, shifted: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Score a perturbed render against the row's stored audio.

    ``rms`` is an envelope cosine *similarity* (1.0 means unchanged); the other three are
    distances that grow with the size of the audible change.

    :param original: Stored audio, shape ``(C, T)``.
    :param shifted: Re-rendered audio, same shape as ``original``.
    :param sample_rate: Dataset sample rate in Hz.
    :returns: One finite score per metric subfield.
    :raises ValueError: A metric produced a non-finite score.
    """
    from synth_setter.evaluation.compute_audio_metrics import (
        compute_mss,
        compute_rms,
        compute_sot,
        compute_wmfcc,
    )

    target = np.ascontiguousarray(original, dtype=np.float32)
    pred = np.ascontiguousarray(shifted, dtype=np.float32)
    scores = {
        SHIFT_RMS_SUBFIELD: float(compute_rms(target, pred, sample_rate)),
        SHIFT_SOT_SUBFIELD: float(compute_sot(target, pred, sample_rate)),
        SHIFT_WMFCC_SUBFIELD: float(compute_wmfcc(target, pred, sample_rate)),
        SHIFT_MSS_SUBFIELD: float(compute_mss(target, pred, sample_rate)),
    }
    non_finite = sorted(name for name, score in scores.items() if not np.isfinite(score))
    if non_finite:
        raise ValueError(f"shift metrics {non_finite} are non-finite")
    return scores


@dataclass(frozen=True)
class ShiftedBatch:
    """One batch's shift results, already aligned row-for-row with the source batch.

    .. attribute :: audio

        ``(B, C, T)`` re-rendered audio in the source column's dtype.

    .. attribute :: scalars

        Non-audio subfield values keyed by subfield name.
    """

    audio: np.ndarray
    scalars: dict[str, list[str] | list[float]]


@dataclass(frozen=True)
class ParamShifter:
    """Re-render one batch of rows with a single parameter redrawn per row.

    .. attribute :: renderer

        Renderer session shared across every row of the run.

    .. attribute :: spec

        Param spec the dataset's ``param_array`` is encoded against.

    .. attribute :: velocity

        MIDI velocity the source dataset was rendered with.

    .. attribute :: seed

        Master seed the per-row replacement draws derive from.
    """

    renderer: AudioRenderer
    spec: ParamSpec
    velocity: int
    seed: int

    def _shift_row(self, row_id: int, encoded: np.ndarray) -> ShiftedRow:
        """Redraw this row's assigned parameter under its own deterministic seed.

        :param row_id: Lance row id, driving both the assignment and the draw.
        :param encoded: The row's stored encoded param vector.
        :returns: The redrawn parameter, its change size, and the shifted row.
        """
        return shift_encoded_row(
            encoded,
            self.spec,
            param_index=assigned_param_index(row_id, len(self.spec.names)),
            rng=rng_for_sample(self.seed, row_id),
        )

    def _render_shift(self, shift: ShiftedRow, original: np.ndarray) -> np.ndarray:
        """Render a shifted patch into the stored audio's exact shape and dtype.

        :param shift: The row's shifted encoded patch.
        :param original: The row's stored audio, whose layout the render must match.
        :returns: Re-rendered audio matching ``original``'s shape and dtype.
        :raises ValueError: The re-render's shape differs from the stored audio's.
        """
        rendered = _render_encoded_row(self.renderer, shift.encoded, self.spec, self.velocity)
        if rendered.shape != original.shape:
            raise ValueError(
                f"re-rendered audio has shape {rendered.shape}, but the dataset's audio rows "
                f"are {original.shape}; the render config does not match the dataset it is "
                "being applied to"
            )
        return rendered

    def __call__(self, sources: Mapping[str, np.ndarray], sample_rate: int) -> ShiftedBatch:
        """Shift, re-render, and score every row of one batch.

        :param sources: Decoded ``audio``, ``param_array``, and ``_rowid`` columns.
        :param sample_rate: Dataset sample rate in Hz.
        :returns: Re-rendered audio and non-audio subfields aligned with the batch's rows.
        """
        rows = zip(
            sources[ROW_ID_FIELD], sources[PARAM_ARRAY_FIELD], sources[AUDIO_FIELD], strict=True
        )
        names: list[str] = []
        amounts: list[float] = []
        rendered_rows: list[np.ndarray] = []
        metrics: dict[str, list[float]] = {field: [] for field in _METRIC_SUBFIELDS}
        with _quiet_metric_logs():
            for row_id, encoded, original in rows:
                shift = self._shift_row(int(row_id), encoded)
                rendered = self._render_shift(shift, original)
                names.append(shift.param_name)
                amounts.append(shift.amount)
                rendered_rows.append(rendered.astype(original.dtype))
                for field, score in _shift_metrics(original, rendered, sample_rate).items():
                    metrics[field].append(score)
        return ShiftedBatch(
            audio=np.stack(rendered_rows),
            scalars={
                SHIFT_PARAM_SUBFIELD: names,
                SHIFT_AMOUNT_SUBFIELD: amounts,
                **metrics,
            },
        )


def load_param_shifter(config: AddEmbeddingsConfig) -> ParamShifter:
    """Build the run's renderer and param spec from the composed render config.

    :param config: Run config carrying the composed ``render``/``synth`` selection.
    :returns: Shifter bound to one renderer session.
    :raises ValueError: No render config is composed.
    """
    from synth_setter.data.vst.param_spec_registry import resolve_param_spec
    from synth_setter.renderer_factory import make_audio_renderer

    if config.render is None:
        raise ValueError(
            f"{SHIFT_FIELD} embeddings re-render every row and need a composed render "
            "config; pass `render=<group> synth=<group>`"
        )
    render: RenderConfig = config.render
    logger.info(
        "loading_param_shift_renderer",
        backend=render.renderer_backend,
        synth=render.synth.name,
        param_spec=render.param_spec_name,
        seed=config.param_shift_seed,
    )
    return ParamShifter(
        renderer=make_audio_renderer(render),
        spec=resolve_param_spec(render.param_spec_name),
        velocity=render.velocity,
        seed=config.param_shift_seed,
    )


def encode_param_shift_column(
    sources: Mapping[str, np.ndarray], sample_rate: int, encoder: object
) -> pa.Array:
    """Encode one batch's shift results as the nested ``shift`` struct column.

    :param sources: Decoded ``audio``, ``param_array``, and ``_rowid`` columns.
    :param sample_rate: Dataset sample rate in Hz.
    :param encoder: Renderer-bound :class:`ParamShifter` for this run.
    :returns: Struct array whose subfields are ``SHIFT_SUBFIELD_NAMES``.
    """
    from synth_setter.pipeline.data.lance_shard import tensor_array

    audio = sources[AUDIO_FIELD]
    batch = cast("ParamShifter", encoder)(sources, sample_rate)
    subfields: dict[str, pa.Array] = {
        name: pa.array(batch.scalars[name], arrow_type)
        for name, arrow_type in _SCALAR_TYPES.items()
    }
    subfields[SHIFT_AUDIO_SUBFIELD] = tensor_array(batch.audio, audio.dtype, audio.shape[1:])
    return pa.StructArray.from_arrays(
        [subfields[name] for name in SHIFT_SUBFIELD_NAMES], names=list(SHIFT_SUBFIELD_NAMES)
    )


def param_shift_policy_values(config: AddEmbeddingsConfig) -> Sequence[str]:
    """Return every run setting that changes this embedder's persisted output.

    :param config: Run config carrying the render selection and seed.
    :returns: Values folded into the column's artifact identity.
    """
    render = config.render
    return (
        "" if render is None else render.model_dump_json(),
        str(config.param_shift_seed),
    )
