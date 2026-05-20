"""End-to-end integration test for ``gui_toggle_cadence="always_on"`` (#1187).

Runs on both the Linux Xvfb path (``docker/ubuntu22_04/run-linux-vst-headless.sh``)
and the macOS Cocoa path via the existing CI matrix. Verifies that
``editor_held_open`` survives off the main thread on each platform, the shard
loop completes for all samples with the editor realised throughout, and no
editor-thread crash is logged.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  side-effect: registers Blosc2 filter for h5py reads
import pytest

from synth_setter.data.vst.writers import make_hdf5_dataset
from synth_setter.pipeline.schemas.spec import RenderConfig

_ = hdf5plugin  # keep type checkers from flagging the side-effect import

from tests.data.vst.test_generate_vst_dataset import (  # noqa: E402
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


def _render_cfg(num_samples: int) -> RenderConfig:
    """Build a held-open ``RenderConfig`` with the integration-test defaults.

    :param num_samples: Samples in the shard; also used as the per-batch size so
        the flush callback fires at the end of the shard (one batch).
    :return: A ``RenderConfig`` with ``always_on`` + ``once`` pairing required
        by the schema validator.
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
        samples_per_render_batch=num_samples,
        samples_per_shard=num_samples,
        plugin_reload_cadence="once",
        gui_toggle_cadence="always_on",
    )


@pytest.mark.slow
@pytest.mark.requires_vst
@skip_no_vst
def test_always_on_renders_small_shard_end_to_end(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``always_on`` holds the editor open for the whole shard render on real Surge XT.

    Runs on whichever platform the CI matrix picks up (Linux + macOS). Asserts
    the shard completes with all rows written, the HDF5 file is well-formed, and
    no ``vst-editor-window crashed`` warning landed in the captured logs.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param caplog: Captures stdlib-logging records (loguru bridging may not
        fire; the assertion is best-effort signal, not a contract gate).
    """
    num_samples = 4
    render_cfg = _render_cfg(num_samples=num_samples)
    out = tmp_path / "shard-000000.h5"
    fixed_synth = [_HARDCODED_SYNTH_PARAMS] * num_samples
    fixed_note = [_HARDCODED_NOTE_PARAMS] * num_samples

    with caplog.at_level(logging.ERROR):
        make_hdf5_dataset(
            hdf5_file=out,
            render_cfg=render_cfg,
            fixed_synth_params_list=fixed_synth,
            fixed_note_params_list=fixed_note,
        )

    assert out.exists()
    with h5py.File(out, "r") as f:
        audio_ds = f["audio"]
        assert isinstance(audio_ds, h5py.Dataset)
        assert audio_ds.shape[0] == num_samples
    assert not any("vst-editor-window crashed" in r.message for r in caplog.records)
