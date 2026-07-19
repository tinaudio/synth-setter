"""Real torchsynth Lance dataset generation — no plugin host, CPU-fast."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path

import lance
import numpy as np
import pytest

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


@pytest.mark.slow
def test_generate_vst_dataset_cli_renders_a_torchsynth_lance_shard(tmp_path: Path) -> None:
    """The public CLI parses torchsynth config and writes readable audio rows.

    Distinct from the in-process writer test above: this pins the
    pydantic-settings CLI arg surface end-to-end, at real-subprocess cost.

    :param tmp_path: Destination directory for the rendered shard.
    """
    shard = tmp_path / "cli-shard.lance"
    subprocess.run(  # noqa: S603 — sys.executable and every CLI argument are test-owned
        [
            sys.executable,
            "-m",
            "synth_setter.data.vst.generate_vst_dataset",
            str(shard),
            "--plugin_path",
            "torchsynth",
            "--plugin_state_path",
            "",
            "--param_spec_name",
            "torchsynth_adsr",
            "--renderer_version",
            importlib.metadata.version("torchsynth"),
            "--renderer_backend",
            "torchsynth",
            "--sample_rate",
            str(_SAMPLE_RATE),
            "--channels",
            "2",
            "--velocity",
            "100",
            "--signal_duration_seconds",
            str(_DURATION_SECONDS),
            "--min_loudness",
            "-70.0",
            "--samples_per_shard",
            "2",
            "--samples_per_render_batch",
            "1",
            "--base_seed",
            "42",
            "--plugin_reload_cadence",
            "once",
            "--gui_toggle_cadence",
            "never",
        ],
        check=True,
        timeout=120,
    )

    dataset = lance.dataset(str(shard))
    audio = _read_lance_column(shard, AUDIO_FIELD)
    assert dataset.count_rows() == 2
    assert audio.shape == (2, 2, int(_SAMPLE_RATE * _DURATION_SECONDS))
    assert (np.abs(audio.astype(np.float32)).max(axis=(1, 2)) > 0.0).all()


def test_make_lance_dataset_compacts_shard_to_one_fragment_and_version(tmp_path: Path) -> None:
    """The committed shard holds one compacted fragment and no stale versions.

    With ``samples_per_render_batch=2`` over five rows the writer stages three
    fragments; the final dataset must compact them into one fragment, keep only
    the post-compaction manifest, and hold exactly the data files that manifest
    references (pre-compaction files would double the shard's footprint).

    :param tmp_path: Destination directory for the rendered shard.
    """
    shard = tmp_path / "shard-000000.lance"

    make_lance_dataset(shard, _torchsynth_render_cfg())

    dataset = lance.dataset(str(shard))
    fragments = dataset.get_fragments()
    assert len(fragments) == 1
    assert dataset.count_rows() == _SAMPLES_PER_SHARD
    assert len(dataset.versions()) == 1
    referenced = {Path(f.path).name for frag in fragments for f in frag.metadata.files}
    on_disk = {p.name for p in (shard / "data").iterdir()}
    assert on_disk == referenced


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
