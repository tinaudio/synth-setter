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
_COPY_GUI_YAML = _SWEEPS_DIR / "generate_dataset_copy_paired_gui_surge_xt.yaml"

_REFERENCE_URI = "copy_dataset_root_uri=r2://intermediate-data/data/ref-surge-xt-489/paired-ref-v1"


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


def test_generate_dataset_copy_paired_gui_yaml_shape() -> None:
    """Paired-copy GUI sweep pins the copy base + reference URI and grids gui_toggle_cadence."""
    cfg = yaml.safe_load(_COPY_GUI_YAML.read_text())

    assert cfg["program"] == "src/synth_setter/cli/generate_dataset.py"
    assert cfg["method"] == "grid"

    cmd = cfg["command"]
    assert cmd[:2] == ["${interpreter}", "${program}"]
    assert "${args_no_hyphens}" in cmd
    # The copy base fixes the match set + gate-off; the reference URI is the patch set every
    # arm replays. Drift on either voids the per-patch pairing.
    assert "experiment=generate_dataset/copy-paired-surge-xt" in cmd
    assert _REFERENCE_URI in cmd

    assert set(cfg["parameters"].keys()) == {"render.gui_toggle_cadence"}
    # always_on is excluded — it requires plugin_reload_cadence=once and would confound the contrast.
    assert cfg["parameters"]["render.gui_toggle_cadence"]["values"] == ["never", "once", "render"]
