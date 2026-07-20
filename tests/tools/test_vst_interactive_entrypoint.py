"""Public entrypoint contract for the synth-neutral interactive VST tool."""

import importlib.util
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from pedalboard.io import AudioFile

from synth_setter.resources import as_file, vst_headless_wrapper
from tests._vst import PLUGIN_PATH


def test_surge_xt_interactive_module_when_imported_is_not_found() -> None:
    """The tool has no compatibility shim because no serialized import path depends on it."""
    assert importlib.util.find_spec("synth_setter.tools.surge_xt_interactive") is None


def test_vst_interactive_help_when_invoked_uses_synth_neutral_summary() -> None:
    """The canonical module runs and describes a VST3 plugin rather than one synth."""
    result = subprocess.run(  # noqa: S603 — fixed module entrypoint
        [sys.executable, "-m", "synth_setter.tools.vst_interactive", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "Open a VST3 plugin GUI" in result.stdout
    assert "--experiment" in result.stdout


@pytest.mark.requires_vst
def test_vst_interactive_entrypoint_when_recording_session_writes_audible_wav(
    tmp_path: Path,
) -> None:
    """Drive the real module, plugin, editor shutdown, and recording output.

    :param tmp_path: Per-test temporary directory.
    """
    output_path = tmp_path / "session.wav"
    command = [
        sys.executable,
        "-m",
        "synth_setter.tools.vst_interactive",
        "--plugin-path",
        PLUGIN_PATH,
        "--param-spec-name",
        "surge_simple",
        "--session-recording-path",
        str(output_path),
    ]
    output_lines: list[str] = []
    with as_file(vst_headless_wrapper()) as wrapper_path:
        if sys.platform == "linux":
            command.insert(0, str(wrapper_path))
        process = subprocess.Popen(  # noqa: S603 — real fixed VST entrypoint
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
        )
        selector = selectors.DefaultSelector()
        assert process.stdout is not None
        assert process.stderr is not None
        selector.register(process.stdout, selectors.EVENT_READ, data=process.stdout)
        selector.register(process.stderr, selectors.EVENT_READ, data=process.stderr)
        deadline = time.monotonic() + 60
        try:
            while time.monotonic() < deadline and not output_path.is_file():
                remaining = deadline - time.monotonic()
                for key, _ in selector.select(timeout=max(remaining, 0)):
                    line = key.data.readline()
                    if line:
                        output_lines.append(line)
                if process.poll() is not None:
                    break
        finally:
            selector.close()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)

    assert output_path.is_file(), "".join(output_lines)

    with AudioFile(str(output_path)) as audio_file:
        audio = audio_file.read(audio_file.frames)
    assert float(np.abs(audio).max()) > 1e-4
