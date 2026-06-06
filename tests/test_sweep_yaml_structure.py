"""Offline structural assertions for the operator-facing ``sweeps/*.yaml``.

Pins each sweep YAML's ``command:`` shape and parameter grid so the
contract that ``wandb agent`` executes (subprocess launch via
``${interpreter} ${program} ... ${args_no_hyphens}``) cannot drift
silently. Complements any live-backend sweep e2e by catching shape
breakage without touching the W&B API.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_SWEEPS_DIR = Path(__file__).resolve().parent.parent / "sweeps"
_SWEEP_YAML = _SWEEPS_DIR / "generate_dataset_cadence.yaml"
_SHUFFLE_PROBE_YAML = _SWEEPS_DIR / "generate_dataset_shuffle_probe_surge_xt.yaml"


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


def test_generate_dataset_shuffle_probe_yaml_shape() -> None:
    """Shuffle-probe sweep pins shard param cadence (the probe trigger) and the surge_xt set."""
    cfg = yaml.safe_load(_SHUFFLE_PROBE_YAML.read_text())

    assert cfg["program"] == "src/synth_setter/cli/generate_dataset.py"

    cmd = cfg["command"]
    assert cmd[:2] == ["${interpreter}", "${program}"]
    assert "${args_no_hyphens}" in cmd
    assert any(isinstance(t, str) and t.startswith("experiment=") for t in cmd)
    # These three overrides are the probe's load-bearing contract: shard cadence gates the shuffle
    # probe on uniform params, and surge_xt params are meaningless without the matching base preset.
    assert "render.param_sample_cadence=shard" in cmd
    assert "render.param_spec_name=surge_xt" in cmd
    assert "render.preset_path=presets/surge-base.vstpreset" in cmd

    assert set(cfg["parameters"].keys()) == {
        "render.plugin_reload_cadence",
        "render.gui_toggle_cadence",
    }
    # Gridding param_sample_cadence would un-gate the probe on the `sample` cells; it must stay a
    # fixed command override, never a swept dimension.
    assert "render.param_sample_cadence" not in cfg["parameters"]
