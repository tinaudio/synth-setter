"""Real production-path coverage for the ``param_shift`` embedder — no fakes, no mocks.

The real writer renders a real torchsynth Lance shard, the real
``synth-setter-add-embeddings`` CLI runs against it in a subprocess, and the committed
columns are checked against a fresh render of the patch each row claims to describe.
torchsynth is the backend because it renders in-process, so the whole production path
runs on a stock CI runner with nothing substituted.
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.data.vst.seeding import rng_for_sample
from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    AUDIO_SHIFT_FIELD,
    MSS_SHIFT_FIELD,
    PARAM_AMOUNT_SHIFT_FIELD,
    PARAM_ARRAY_FIELD,
    PARAM_SHIFT_FIELD,
    PARAM_SHIFT_FIELD_NAMES,
    RMS_SHIFT_FIELD,
    SOT_SHIFT_FIELD,
    WMFCC_SHIFT_FIELD,
)
from synth_setter.data.vst.writers import make_lance_dataset
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.data.param_shift import (
    assigned_param_index,
    shift_encoded_row,
)
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.renderer_factory import make_audio_renderer
from synth_setter.synth_spec import SynthName, SynthSpec

pytestmark = pytest.mark.slow

_SYNTH = "torchsynth_adsr"
_SAMPLE_RATE = 22_050
_DURATION_SECONDS = 0.5
_CHANNELS = 2
_VELOCITY = 100
_ROWS = 28
_SEED = 4242
_METRIC_FIELDS = (RMS_SHIFT_FIELD, SOT_SHIFT_FIELD, WMFCC_SHIFT_FIELD, MSS_SHIFT_FIELD)


def _render_config() -> RenderConfig:
    """Build the render config both the source shard and the re-render use.

    :returns: Tiny real torchsynth render config.
    """
    return RenderConfig(
        synth=SynthSpec(
            name=SynthName(_SYNTH),
            param_spec_name=ParamSpecName(_SYNTH),
            plugin_path="torchsynth",
            plugin_state_path="",
            synth_version="1.0.2",
        ),
        renderer_backend="torchsynth",
        sample_rate=_SAMPLE_RATE,
        channels=_CHANNELS,
        velocity=_VELOCITY,
        signal_duration_seconds=_DURATION_SECONDS,
        min_loudness=-70.0,
        samples_per_render_batch=4,
        samples_per_shard=_ROWS,
        base_seed=1757,
        plugin_reload_cadence="once",
        gui_toggle_cadence="never",
    )


@pytest.fixture(name="shifted_dataset", scope="module")
def _shifted_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Render a real shard and run the real add-embeddings CLI over it once.

    :param tmp_path_factory: Session-scoped temp directory factory.
    :returns: Path to the augmented Lance dataset.
    """
    shard = tmp_path_factory.mktemp("param-shift") / "shard-000000.lance"
    make_lance_dataset(shard, _render_config())
    subprocess.run(  # noqa: S603 — sys.executable and every argument are test-owned
        [
            sys.executable,
            "-m",
            "synth_setter.pipeline.data.add_embeddings",
            f"lance_uri={shard}",
            "embeddings=[param_shift]",
            "render=torchsynth",
            f"synth={_SYNTH}",
            f"render.sample_rate={_SAMPLE_RATE}",
            f"render.channels={_CHANNELS}",
            f"render.velocity={_VELOCITY}",
            f"render.signal_duration_seconds={_DURATION_SECONDS}",
            f"param_shift_seed={_SEED}",
            "batch_size=8",
            "build_index=false",
        ],
        check=True,
    )
    return shard


def _column(dataset: lance.LanceDataset, field: str) -> pa.Array:
    """Read one committed column in row order.

    :param dataset: Augmented Lance dataset.
    :param field: Column to read.
    :returns: Combined Arrow array for the column.
    """
    return dataset.to_table(columns=[field]).column(field).combine_chunks()


def test_param_shift_cli_writes_every_declared_column(shifted_dataset: Path) -> None:
    """A real CLI run commits all seven columns with usable Arrow types.

    :param shifted_dataset: Augmented Lance dataset.
    """
    dataset = lance.dataset(str(shifted_dataset))

    assert set(PARAM_SHIFT_FIELD_NAMES) <= set(dataset.schema.names)
    assert dataset.schema.field(PARAM_SHIFT_FIELD).type == pa.string()
    for field in (PARAM_AMOUNT_SHIFT_FIELD, *_METRIC_FIELDS):
        assert dataset.schema.field(field).type == pa.float32()


def test_param_shift_audio_matches_the_source_audio_layout(shifted_dataset: Path) -> None:
    """The re-render is stored exactly like the render it is compared against.

    :param shifted_dataset: Augmented Lance dataset.
    """
    dataset = lance.dataset(str(shifted_dataset))

    audio = _column(dataset, AUDIO_FIELD).to_numpy_ndarray()
    shifted = _column(dataset, AUDIO_SHIFT_FIELD).to_numpy_ndarray()

    assert shifted.shape == audio.shape == (_ROWS, _CHANNELS, int(_SAMPLE_RATE * _DURATION_SECONDS))
    assert shifted.dtype == audio.dtype
    assert np.isfinite(shifted.astype(np.float32)).all()


def test_param_shift_spreads_rows_evenly_across_the_param_spec(shifted_dataset: Path) -> None:
    """Every parameter of the spec owns a share of the rows, none more than one apart.

    :param shifted_dataset: Augmented Lance dataset.
    """
    spec = resolve_param_spec(ParamSpecName(_SYNTH))
    names = _column(lance.dataset(str(shifted_dataset)), PARAM_SHIFT_FIELD).to_pylist()

    counts = Counter(names)

    assert set(counts) == set(spec.names)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_param_shift_records_a_move_and_finite_metrics(shifted_dataset: Path) -> None:
    """Rows record a real move and every metric scored the result finitely.

    A redraw may legitimately land back on the original value — ``pitch`` is discrete — so
    a zero amount is valid; a run where nothing moved would not be.

    :param shifted_dataset: Augmented Lance dataset.
    """
    dataset = lance.dataset(str(shifted_dataset))

    amounts = np.asarray(_column(dataset, PARAM_AMOUNT_SHIFT_FIELD).to_pylist())
    assert amounts.shape == (_ROWS,)
    assert np.isfinite(amounts).all()
    assert (amounts >= 0.0).all()
    assert np.count_nonzero(amounts) >= _ROWS // 2
    for field in _METRIC_FIELDS:
        scores = np.asarray(_column(dataset, field).to_pylist())
        assert scores.shape == (_ROWS,)
        assert np.isfinite(scores).all()
    # An RMS-envelope cosine similarity is bounded; the guard would catch a wired-up
    # distance masquerading as a similarity. The tolerance covers float32 rounding of a
    # similarity of exactly 1.0, which an unchanged redraw produces.
    rms = np.asarray(_column(dataset, RMS_SHIFT_FIELD).to_pylist())
    assert np.all((rms >= -1.0 - 1e-6) & (rms <= 1.0 + 1e-6))


def test_param_shift_audio_is_the_patch_the_row_claims(shifted_dataset: Path) -> None:
    """Re-deriving each row's shift from its row id reproduces the committed audio.

    This is the load-bearing check: it proves ``param_shift``/``param_amount_shift`` are a
    truthful description of ``audio_shift`` rather than three independently-computed
    columns that happen to sit in the same row.

    :param shifted_dataset: Augmented Lance dataset.
    """
    dataset = lance.dataset(str(shifted_dataset))
    spec = resolve_param_spec(ParamSpecName(_SYNTH))
    renderer = make_audio_renderer(_render_config())

    table = dataset.to_table(
        columns=[PARAM_ARRAY_FIELD, PARAM_SHIFT_FIELD, PARAM_AMOUNT_SHIFT_FIELD, AUDIO_SHIFT_FIELD],
        with_row_id=True,
    )
    params = table.column(PARAM_ARRAY_FIELD).combine_chunks().to_numpy_ndarray()
    committed_audio = table.column(AUDIO_SHIFT_FIELD).combine_chunks().to_numpy_ndarray()
    committed_names = table.column(PARAM_SHIFT_FIELD).to_pylist()
    committed_amounts = table.column(PARAM_AMOUNT_SHIFT_FIELD).to_pylist()
    row_ids = table.column("_rowid").to_pylist()

    for index, row_id in enumerate(row_ids):
        expected = shift_encoded_row(
            params[index],
            spec,
            param_index=assigned_param_index(row_id, len(spec.names)),
            rng=rng_for_sample(_SEED, row_id),
        )
        assert committed_names[index] == expected.param_name
        assert committed_amounts[index] == pytest.approx(expected.amount, rel=1e-6)

        synth_params, note_params = spec.decode(expected.encoded)
        rendered = renderer.render(
            synth_params, note_params["pitch"], _VELOCITY, note_params["note_start_and_end"]
        )
        assert np.array_equal(
            rendered.astype(committed_audio.dtype), committed_audio[index]
        ), f"row {index} audio does not match a re-render of its recorded shift"
