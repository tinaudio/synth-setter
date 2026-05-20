"""Pin the pedalboard ``show_editor`` main-thread invariant (#1187, #1204).

Pedalboard 0.9.x enforces the JUCE rule that ``VST3Plugin.show_editor`` can
only be invoked from the process main thread; calling it from any other thread
raises ``RuntimeError("Plugin UI windows can only be shown from the main
thread.")``. This contract test pins that behaviour so:

1. A future pedalboard upgrade that lifts the restriction fails this test
   loud — at which point ``editor_held_open``'s original "editor on a daemon
   thread" design becomes viable again.
2. Anyone proposing a similar thread-the-editor design has to confront the
   invariant here before re-introducing the regression.

Runs through ``docker/ubuntu22_04/run-linux-vst-headless.sh`` inside the
existing ``test-vst-slow.yml`` workflow.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest
from pedalboard import VST3Plugin

_PLUGIN_PATH = os.environ.get("SYNTH_SETTER_PLUGIN_PATH") or "plugins/Surge XT.vst3"

skip_no_vst = pytest.mark.skipif(
    not Path(_PLUGIN_PATH).exists(),
    reason=f"VST plugin not found at {_PLUGIN_PATH}",
)


@pytest.mark.slow
@pytest.mark.requires_vst
@skip_no_vst
def test_show_editor_rejects_non_main_thread() -> None:
    """``VST3Plugin.show_editor`` raises ``RuntimeError`` from a non-main thread.

    Drives ``show_editor`` on a worker thread with a pre-set close event so the
    call returns promptly even on the (regression) path where pedalboard
    accepts the off-thread invocation. Asserts the captured exception is a
    ``RuntimeError`` whose message names the main-thread requirement — string
    match is intentional because pedalboard exposes no typed subclass for this
    contract.
    """
    plugin = VST3Plugin(_PLUGIN_PATH)
    captured: list[BaseException] = []
    close = threading.Event()
    close.set()  # pre-arm so a regression path (no raise) returns immediately

    def run() -> None:
        try:
            plugin.show_editor(close)
        except BaseException as exc:  # noqa: BLE001 — pin whatever pedalboard raises
            captured.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(timeout=3.0)
    assert not worker.is_alive(), "show_editor worker did not return within 3s"
    assert len(captured) == 1, f"expected one captured exception, got: {captured!r}"
    err = captured[0]
    assert isinstance(err, RuntimeError), f"expected RuntimeError, got {type(err).__name__}: {err}"
    assert "main thread" in str(err).lower(), f"unexpected message: {err}"
