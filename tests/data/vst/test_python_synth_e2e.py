"""End-to-end shard writes driven by the real Python synth backends.

No fakes anywhere: ``make_hdf5_dataset`` resolves ``plugin_path="torchsynth"`` /
``"synthax"`` through ``core.load_plugin`` and renders real audio, proving the
backends are interchangeable with a ``.vst3`` through the whole writer path
(batch loop, loudness gate, mel-spec computation, HDF5 writer).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  side-effect: registers Blosc2 filter for h5py reads
import numpy as np
import pytest

from synth_setter.data.vst.core import render_params
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.data.vst.shapes import AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD
from synth_setter.data.vst.writers import make_hdf5_dataset, make_lance_dataset, make_wds_dataset
from synth_setter.pipeline.schemas.spec import RenderConfig

_SAMPLE_RATE = 44100
_CHANNELS = 2
_DURATION_S = 1.0
_NUM_SAMPLES = 3


def _python_synth_render_cfg(name: str) -> RenderConfig:
    """Build a small all-real RenderConfig for a Python synth backend.

    The loudness gate is disabled (``-inf``) so unlucky quiet random patches
    don't retry — retry behaviour is covered by the generic writer tests.

    :param name: ``"torchsynth"`` or ``"synthax"``.
    :returns: A 3-sample single-batch shard config.
    """
    return RenderConfig(
        plugin_path=name,
        preset_path="",
        param_spec_name=name,
        renderer_version="unchecked",
        sample_rate=_SAMPLE_RATE,
        channels=_CHANNELS,
        velocity=100,
        signal_duration_seconds=_DURATION_S,
        min_loudness=float("-inf"),
        samples_per_render_batch=_NUM_SAMPLES,
        samples_per_shard=_NUM_SAMPLES,
        plugin_reload_cadence="once",
        gui_toggle_cadence="never",
    )


@pytest.mark.parametrize("name", ["torchsynth", pytest.param("synthax", marks=pytest.mark.slow)])
def test_make_hdf5_dataset_python_synth_writes_complete_shard(name: str, tmp_path: Path) -> None:
    """A real shard write through the writer path produces complete, finite data.

    :param name: Python synth backend under test.
    :param tmp_path: Pytest fixture providing the shard destination.
    """
    out = tmp_path / f"{name}-shard.h5"
    make_hdf5_dataset(str(out), _python_synth_render_cfg(name))

    spec = param_specs[name]
    with h5py.File(out, "r") as f:
        audio_ds, mel_ds, params_ds = f[AUDIO_FIELD], f[MEL_SPEC_FIELD], f[PARAM_ARRAY_FIELD]
        assert isinstance(audio_ds, h5py.Dataset)
        assert isinstance(mel_ds, h5py.Dataset)
        assert isinstance(params_ds, h5py.Dataset)
        audio, mel, params = audio_ds[...], mel_ds[...], params_ds[...]
    assert audio.shape == (_NUM_SAMPLES, _CHANNELS, int(_SAMPLE_RATE * _DURATION_S))
    assert params.shape == (_NUM_SAMPLES, len(spec))
    assert mel.shape[0] == _NUM_SAMPLES
    assert np.all(np.isfinite(audio))
    assert np.all(np.isfinite(params))


@pytest.mark.slow
def test_wds_and_lance_writers_accept_python_synth_plugin(tmp_path: Path) -> None:
    """The wds and Lance writer paths also resolve a Python synth end-to-end.

    :param tmp_path: Pytest fixture providing the shard destinations.
    """
    cfg = _python_synth_render_cfg("torchsynth")
    make_wds_dataset(str(tmp_path / "torchsynth-shard.tar"), cfg)
    make_lance_dataset(tmp_path / "torchsynth-shard.lance", cfg)
    assert (tmp_path / "torchsynth-shard.tar").stat().st_size > 0
    assert (tmp_path / "torchsynth-shard.lance").is_dir()


def test_render_params_torchsynth_end_to_end_returns_audio() -> None:
    """``core.render_params`` drives a torchsynth render exactly like a VST3."""
    spec = param_specs["torchsynth"]
    synth_params, _ = spec.sample(np.random.default_rng(3))
    audio = render_params(
        "torchsynth",
        synth_params,
        midi_note=60,
        velocity=100,
        note_start_and_end=(0.0, 0.5),
        signal_duration_seconds=_DURATION_S,
        sample_rate=float(_SAMPLE_RATE),
        channels=_CHANNELS,
        preset_path=None,
    )
    assert audio.shape == (_CHANNELS, int(_SAMPLE_RATE * _DURATION_S))
    assert audio.dtype == np.float32
    assert np.all(np.isfinite(audio))
