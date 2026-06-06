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
_REUSE_DEPTH_YAML = _SWEEPS_DIR / "generate_dataset_reuse_depth_surge_xt.yaml"


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


def test_generate_dataset_reuse_depth_yaml_shape() -> None:
    """Reuse-depth sweep pins the once+shard reuse path and grids samples_per_shard."""
    cfg = yaml.safe_load(_REUSE_DEPTH_YAML.read_text())

    assert cfg["program"] == "src/synth_setter/cli/generate_dataset.py"

    cmd = cfg["command"]
    assert cmd[:2] == ["${interpreter}", "${program}"]
    assert "${args_no_hyphens}" in cmd
    assert any(isinstance(t, str) and t.startswith("experiment=") for t in cmd)
    # These four pins ARE the experiment — drift on any silently voids the probe.
    assert "render.param_sample_cadence=shard" in cmd
    assert "render.plugin_reload_cadence=once" in cmd
    assert "render.gui_toggle_cadence=never" in cmd
    assert "render.param_spec_name=surge_xt" in cmd

    assert set(cfg["parameters"].keys()) == {"render.samples_per_shard"}
    depths = cfg["parameters"]["render.samples_per_shard"]["values"]
    # Each split size must be a multiple of every swept depth, else that depth's grid cell fails
    # DatasetSpec validation. Check against the pinned split, not a literal, so the two stay coupled.
    (sizes_arg,) = [t for t in cmd if isinstance(t, str) and t.startswith("train_val_test_sizes=")]
    sizes = [int(x) for x in sizes_arg.split("=", 1)[1].strip("[]").split(",")]
    assert all(size % depth == 0 for size in sizes for depth in depths)
    assert max(depths) >= 40  # past PR #706's ~12-reuse junk threshold
