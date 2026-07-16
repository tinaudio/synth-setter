"""Public entrypoint contract for the synth-neutral interactive VST tool."""

import importlib.util
import subprocess
import sys


def test_old_surge_module_name_is_removed() -> None:
    """The tool has no compatibility shim because no serialized import path depends on it."""
    assert importlib.util.find_spec("synth_setter.tools.surge_xt_interactive") is None


def test_help_from_vst_module_uses_synth_neutral_summary() -> None:
    """The canonical module runs and describes a VST3 plugin rather than one synth."""
    result = subprocess.run(  # noqa: S603 — fixed module entrypoint
        [sys.executable, "-m", "synth_setter.tools.vst_interactive", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "Open a VST3 plugin GUI" in result.stdout
