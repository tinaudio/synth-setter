"""Offline structural assertion for ``sweeps/generate_dataset_cadence.yaml``.

Pins the operator-facing sweep YAML's ``command:`` shape and parameter
grid so the contract that ``wandb agent`` executes (subprocess launch
via ``${interpreter} ${program} ... ${args_no_hyphens}``) cannot drift
silently. Complements any live-backend sweep e2e by catching shape
breakage without touching the W&B API.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_SWEEP_YAML = Path(__file__).resolve().parent.parent / "sweeps" / "generate_dataset_cadence.yaml"


def test_generate_dataset_cadence_yaml_shape() -> None:
    """Sweep YAML pins the program path, command template, and cadence grid."""
    cfg = yaml.safe_load(_SWEEP_YAML.read_text())

    assert cfg["program"] == "src/synth_setter/cli/generate_dataset.py"

    cmd = cfg["command"]
    assert cmd[:2] == ["${interpreter}", "${program}"]
    assert "${args_no_hyphens}" in cmd
    assert any(isinstance(t, str) and t.startswith("experiment=") for t in cmd)

    assert set(cfg["parameters"].keys()) == {
        "render.plugin_reload_cadence",
        "render.gui_toggle_cadence",
    }
