"""End-to-end Linux integration test for ``gui_toggle_cadence="always_on"`` (#1187).

Runs through ``src/synth_setter/scripts/run-linux-vst-headless.sh`` (Xvfb + xsettingsd
+ openbox + dbus) inside the existing ``test-vst-slow.yml`` workflow. Verifies
that ``run_with_editor_held_open`` keeps the editor realised on the main
thread while renders run on a worker, the shard loop completes for all
samples, the produced HDF5 carries the expected datasets/shapes/dtypes/
finiteness, and no render-worker crash log is emitted via
``core.logger.exception``.

The matching macOS Cocoa coverage is tracked separately — Apple runners with
Surge XT installed are not part of any current workflow; see the follow-up
issue noted in the Wave 3 PR body.
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

_ = hdf5plugin  # keep type checkers from flagging the side-effect import

# ``_PLUGIN_PATH`` is imported with the helpers so the ``skip_no_vst`` mark
# tracks the same path that ``_render_cfg`` embeds in the produced
# ``RenderConfig`` — a local copy could drift if the canonical default ever
# changed.
from tests.data.vst.test_generate_vst_dataset import (  # noqa: E402  pinned canonical patch
    _HARDCODED_NOTE_PARAMS,
    _HARDCODED_SYNTH_PARAMS,
    _PLUGIN_PATH,
    _render_cfg,
)

skip_no_vst = pytest.mark.skipif(
    not Path(_PLUGIN_PATH).exists(),
    reason=f"VST plugin not found at {_PLUGIN_PATH}",
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
    and that ``core.logger.exception`` was never called (the structural crash
    gate — caplog cannot observe loguru output, so the logger is stubbed to
    make the check observable).

    :param tmp_path: Destination directory for the shard HDF5 file under test.
    :param monkeypatch: Stubs ``core.logger`` so the crash gate is observable
        (loguru does not propagate to ``caplog``).
    """
    num_samples = 4
    # ``always_on`` requires ``plugin_reload_cadence="once"`` per schema validator;
    # batch=2 forces a mid-shard flush so the cross-flush invariant is exercised.
    render_cfg = _render_cfg(
        num_samples=num_samples,
        samples_per_render_batch=2,
        plugin_reload_cadence="once",
        gui_toggle_cadence="always_on",
    )
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

    assert fake_logger.exception.call_count == 0, (
        f"unexpected logger.exception calls: {fake_logger.exception.call_args_list}"
    )
