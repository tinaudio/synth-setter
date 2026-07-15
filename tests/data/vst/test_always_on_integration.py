"""End-to-end Linux integration test for ``gui_toggle_cadence="always_on"`` (#1187).

Runs through ``src/synth_setter/scripts/run-linux-vst-headless.sh`` (Xvfb + xsettingsd
+ openbox + dbus) inside the existing ``test-vst-slow.yml`` workflow. Verifies
that ``run_with_editor_held_open`` keeps the editor realised on the main
thread while renders run on a worker, the shard loop completes for all
samples, the produced Lance dataset carries the expected columns/shapes/
finiteness, and no render-worker crash log is emitted via
``core.logger.exception``.

The matching macOS Cocoa coverage is tracked separately — Apple runners with
Surge XT installed are not part of any current workflow; see the follow-up
issue noted in the Wave 3 PR body.
"""

from __future__ import annotations

from pathlib import Path

import lance
import numpy as np
import pytest

from synth_setter.data.vst import core
from synth_setter.data.vst.writers import make_lance_dataset
from tests.data.vst.test_generate_vst_dataset import (  # noqa: E402  pinned canonical patch
    _HARDCODED_NOTE_PARAMS,
    _HARDCODED_SYNTH_PARAMS,
    _render_cfg,
)
from tests.helpers.logger_assertions import assert_no_logger_exceptions  # noqa: E402


@pytest.mark.slow
@pytest.mark.requires_vst
def test_always_on_renders_small_shard_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``always_on`` holds the editor open across a 4-sample shard with mid-shard flush.

    Renders 4 samples in 2 batches so the per-batch flush callback fires inside
    the held-open scope twice. Asserts the Lance shard carries all three columns
    at the expected row count, that audio is finite and within ``[-1, 1]``,
    and that ``core.logger.exception`` was never called (the structural crash
    gate — caplog cannot observe loguru output, so the logger is stubbed to
    make the check observable).

    :param tmp_path: Destination directory for the shard Lance dataset under test.
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
    out = tmp_path / "shard-000000.lance"
    fixed_synth = [_HARDCODED_SYNTH_PARAMS] * num_samples
    fixed_note = [_HARDCODED_NOTE_PARAMS] * num_samples

    with assert_no_logger_exceptions(monkeypatch, core):
        make_lance_dataset(
            out,
            render_cfg,
            fixed_synth_params_list=fixed_synth,
            fixed_note_params_list=fixed_note,
        )

    assert out.exists()
    dataset = lance.dataset(str(out))
    for key in ("audio", "mel_spec", "param_array"):
        assert key in dataset.schema.names, f"missing expected column: {key}"
    assert dataset.count_rows() == num_samples
    audio = np.stack(dataset.to_table(columns=["audio"]).to_pydict()["audio"]).astype(np.float32)
    assert audio.shape[0] == num_samples
    assert np.isfinite(audio).all(), "rendered audio contains NaN/Inf"
    assert (np.abs(audio) <= 1.0).all(), "rendered audio exceeds [-1, 1] bounds"
