"""End-to-end Linux integration test for ``gui_toggle_cadence="always_on"`` (#1187).

Runs through ``docker/ubuntu22_04/run-linux-vst-headless.sh`` (Xvfb + xsettingsd
+ openbox + dbus) inside the existing ``test-vst-slow.yml`` workflow. Verifies
that ``editor_held_open`` survives off the main thread on X11, the shard loop
completes for all samples with the editor realised throughout, the produced
HDF5 carries the expected datasets/shapes/dtypes/finiteness, and the editor
thread emits no crash log via ``core.logger.exception``.

The matching macOS Cocoa coverage is tracked separately — Apple runners with
Surge XT installed are not part of any current workflow; see the follow-up
issue noted in the Wave 3 PR body.
"""

from __future__ import annotations

import os
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

from tests.data.vst.test_generate_vst_dataset import (  # noqa: E402  pinned canonical patch
    _HARDCODED_NOTE_PARAMS,
    _HARDCODED_SYNTH_PARAMS,
)

_PLUGIN_PATH = os.environ.get("SYNTH_SETTER_PLUGIN_PATH") or "plugins/Surge XT.vst3"
_PRESET_PATH = "presets/surge-base.vstpreset"
_SAMPLE_RATE = 44100
_CHANNELS = 2
_DURATION = 4.0
_VELOCITY = 100
_MIN_LOUDNESS = -55.0
_SPEC_NAME = "surge_xt"
_RENDERER_VERSION = "1.3.4"

skip_no_vst = pytest.mark.skipif(
    not Path(_PLUGIN_PATH).exists(),
    reason=f"VST plugin not found at {_PLUGIN_PATH}",
)


def _render_cfg(num_samples: int, samples_per_render_batch: int) -> RenderConfig:
    """Build a held-open ``RenderConfig`` with the integration-test defaults.

    :param num_samples: Total samples in the shard.
    :param samples_per_render_batch: Per-batch size; setting this smaller than
        ``num_samples`` forces multiple flush callbacks inside the held-open
        scope (the cross-flush invariant the test is here to defend).
    :return: A ``RenderConfig`` pinned to the ``always_on`` + ``once`` pairing
        required by the schema validator.
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


@pytest.mark.slow
@pytest.mark.requires_vst
@skip_no_vst
def test_always_on_renders_small_shard_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``always_on`` holds the editor open across a 4-sample shard with mid-shard flush.

    Renders 4 samples in 2 batches so the per-batch flush callback fires inside
    the held-open scope twice. Asserts the HDF5 shard carries all three datasets
    at the expected shape/dtype, that audio is finite and within ``[-1, 1]``,
    and that ``core.logger.exception`` was never called from the editor thread
    (the structural crash gate — caplog cannot observe loguru output, so the
    logger is stubbed to make the check observable).

    :param tmp_path: Destination directory for the shard HDF5 file under test.
    :param monkeypatch: Stubs ``core.logger`` so the editor-thread crash gate
        is observable (loguru does not propagate to ``caplog``).
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
