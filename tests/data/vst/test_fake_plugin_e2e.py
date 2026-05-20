"""End-to-end shard-write proven against a duck-typed ``FakeVST3Plugin``.

No real ``.vst3`` bundle, no X11, no Surge XT — ``install_fake_plugin``
swaps the loader for the fake, so the whole ``make_hdf5_dataset`` path
(batch loop, held-open editor, HDF5 writer, mel-spec computation) runs
on every PR. The real-plugin counterpart at
``test_always_on_integration.py`` stays as the "does Surge XT still
work" gate; this is the "does our pipeline still work" gate.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import h5py
import hdf5plugin  # noqa: F401  side-effect: registers Blosc2 filter for h5py reads
import numpy as np
import pytest

from synth_setter.data.vst import core
from synth_setter.data.vst.writers import make_hdf5_dataset
from synth_setter.pipeline.schemas.spec import RenderConfig

_ = hdf5plugin  # keep type checkers from flagging the side-effect import

from tests.data.vst._fake_plugin import FakeVST3Plugin  # noqa: E402
from tests.data.vst.test_generate_vst_dataset import (  # noqa: E402  pinned canonical patch
    _HARDCODED_NOTE_PARAMS,
    _HARDCODED_SYNTH_PARAMS,
)

_PLUGIN_PATH = "plugins/fake.vst3"  # never touched on disk — load_plugin is patched
_PRESET_PATH = "presets/fake.vstpreset"
_SAMPLE_RATE = 44100
_CHANNELS = 2
_DURATION = 4.0
_VELOCITY = 100
_MIN_LOUDNESS = -55.0
_SPEC_NAME = "surge_xt"
_RENDERER_VERSION = "fake-0.0.0"


def _render_cfg(num_samples: int, samples_per_render_batch: int) -> RenderConfig:
    """Build a held-open ``RenderConfig`` pinned to the fake-plugin defaults.

    :param num_samples: Total samples in the shard.
    :param samples_per_render_batch: Per-batch size; setting smaller than
        ``num_samples`` forces multiple flush callbacks inside the held-open
        scope (the cross-flush invariant the test is here to defend).
    :return: A ``RenderConfig`` pinned to ``always_on`` + ``once``, the only
        pairing the schema validator allows for held-open editor runs.
    """
    return RenderConfig(
        plugin_path=_PLUGIN_PATH,
        preset_path=_PRESET_PATH,
        param_spec_name=_SPEC_NAME,
        renderer_version=_RENDERER_VERSION,
        sample_rate=_SAMPLE_RATE,
        channels=_CHANNELS,
        velocity=_VELOCITY,
        signal_duration_seconds=_DURATION,
        min_loudness=_MIN_LOUDNESS,
        samples_per_render_batch=samples_per_render_batch,
        samples_per_shard=num_samples,
        plugin_reload_cadence="once",
        gui_toggle_cadence="always_on",
    )


@pytest.mark.fake_vst
def test_make_hdf5_dataset_writes_valid_shard_under_fake_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    install_fake_plugin: FakeVST3Plugin,
) -> None:
    """The dataset pipeline produces a valid shard with no real VST3 or X11.

    Four samples in two batches force a mid-shard flush inside the
    held-open editor scope. Asserts shape/dtype/finiteness and that the
    editor-thread crash log never fires — the fake's ``show_editor``
    just blocks on the close event, so there is no realistic failure
    mode here, but keeping the assertion pins the contract for when
    the held-open editor surface evolves.

    :param tmp_path: Destination directory for the shard HDF5 file under test.
    :param monkeypatch: Stubs ``core.logger`` so the crash-gate assertion
        is observable (loguru does not propagate to ``caplog``).
    :param install_fake_plugin: Swaps ``core.load_plugin`` /
        ``core.VST3Plugin`` for the fake before the writer fires.
    """
    num_samples = 4
    render_cfg = _render_cfg(num_samples=num_samples, samples_per_render_batch=2)
    out = tmp_path / "shard-000000.h5"
    fixed_synth = [_HARDCODED_SYNTH_PARAMS] * num_samples
    fixed_note = [_HARDCODED_NOTE_PARAMS] * num_samples
    fake_logger = MagicMock(wraps=core.logger)
    monkeypatch.setattr(core, "logger", fake_logger)

    make_hdf5_dataset(
        hdf5_file=out,
        render_cfg=render_cfg,
        fixed_synth_params_list=fixed_synth,
        fixed_note_params_list=fixed_note,
    )

    assert out.exists()
    with h5py.File(out, "r") as f:
        for key in ("audio", "mel_spec", "param_array"):
            assert key in f, f"missing expected dataset: {key}"
        audio_ds = f["audio"]
        assert isinstance(audio_ds, h5py.Dataset)
        assert audio_ds.shape[0] == num_samples
        audio = audio_ds[...]
    assert np.isfinite(audio).all(), "rendered audio contains NaN/Inf"
    assert (np.abs(audio) <= 1.0).all(), "rendered audio exceeds [-1, 1] bounds"

    crash_log_calls = [
        call
        for call in fake_logger.exception.call_args_list
        if "vst-editor-window crashed" in str(call.args[0])
    ]
    assert not crash_log_calls, f"editor-thread crash logged: {crash_log_calls}"
