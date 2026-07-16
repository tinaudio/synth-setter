"""Real torchsynth Lance dataset generation — no plugin host, CPU-fast."""

from __future__ import annotations

from pathlib import Path

import lance
import numpy as np

from synth_setter.data.vst.shapes import AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD
from synth_setter.data.vst.torchsynth_param_spec import TORCHSYNTH_ADSR_PARAM_SPEC
from synth_setter.data.vst.writers import make_lance_dataset
from synth_setter.pipeline.schemas.spec import RenderConfig

_SAMPLE_RATE = 22_050
_DURATION_SECONDS = 0.5
_SAMPLES_PER_SHARD = 5


def _torchsynth_render_cfg() -> RenderConfig:
    """Build a tiny real torchsynth shard config.

    :returns: Render config for a five-row ADSR-spec shard.
    """
    return RenderConfig(
        plugin_path="torchsynth",
        plugin_state_path="",
        param_spec_name="torchsynth_adsr",  # type: ignore[arg-type]
        renderer_version="1.0.2",
        renderer_backend="torchsynth",
        sample_rate=_SAMPLE_RATE,
        channels=2,
        velocity=100,
        signal_duration_seconds=_DURATION_SECONDS,
        min_loudness=-70.0,
        samples_per_render_batch=2,
        samples_per_shard=_SAMPLES_PER_SHARD,
        base_seed=1757,
        plugin_reload_cadence="once",
        gui_toggle_cadence="never",
    )


def _read_lance_column(path: Path, field: str) -> np.ndarray:
    """Materialize one fixed-shape tensor column from a Lance shard.

    :param path: Rendered ``.lance`` shard directory.
    :param field: Column name to read.
    :returns: The column stacked into a ``(num_rows, *shape)`` array.
    """
    chunk = lance.dataset(str(path)).to_table(columns=[field]).column(field).combine_chunks()
    return chunk.to_numpy_ndarray()


def test_make_lance_dataset_renders_a_real_torchsynth_shard(tmp_path: Path) -> None:
    """The writer produces a complete, loud, well-shaped Lance shard from torchsynth.

    Drives the real path end-to-end — registry lookup, renderer construction, sampling, rendering,
    batching, fragment commit — with no host or fake.

    :param tmp_path: Destination directory for the rendered shard.
    """
    shard = tmp_path / "shard-000000.lance"

    make_lance_dataset(shard, _torchsynth_render_cfg())

    audio = _read_lance_column(shard, AUDIO_FIELD)
    mel = _read_lance_column(shard, MEL_SPEC_FIELD)
    params = _read_lance_column(shard, PARAM_ARRAY_FIELD)
    samples = int(_SAMPLE_RATE * _DURATION_SECONDS)
    assert audio.shape == (_SAMPLES_PER_SHARD, 2, samples)
    assert mel.shape[0] == _SAMPLES_PER_SHARD
    assert params.shape == (_SAMPLES_PER_SHARD, len(TORCHSYNTH_ADSR_PARAM_SPEC))
    assert np.isfinite(params).all()
    assert np.all((params >= 0.0) & (params <= 1.0))
    # Every accepted row passed the loudness gate, so no row may be silent.
    assert (np.abs(audio.astype(np.float32)).max(axis=(1, 2)) > 0.0).all()


def test_make_lance_dataset_same_seed_reproduces_the_shard(tmp_path: Path) -> None:
    """Two renders with one seed produce identical rows across every column.

    Params pin the per-sample seeding; audio and mel equality additionally pin the render path
    itself (a stray global-RNG dependency would diverge here).

    :param tmp_path: Destination directory for the two rendered shards.
    """
    first, second = tmp_path / "a.lance", tmp_path / "b.lance"

    make_lance_dataset(first, _torchsynth_render_cfg())
    make_lance_dataset(second, _torchsynth_render_cfg())

    for field in (PARAM_ARRAY_FIELD, AUDIO_FIELD, MEL_SPEC_FIELD):
        assert np.array_equal(
            _read_lance_column(first, field),
            _read_lance_column(second, field),
        ), field
