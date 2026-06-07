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
_COPY_RELOAD_YAML = _SWEEPS_DIR / "generate_dataset_copy_paired_reload_surge_xt.yaml"


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


def test_generate_dataset_copy_paired_reload_yaml_shape() -> None:
    """Paired-copy reload sweep pins the copy base + reference URI and grids reload cadence."""
    cfg = yaml.safe_load(_COPY_RELOAD_YAML.read_text())

    assert cfg["program"] == "src/synth_setter/cli/generate_dataset.py"
    assert cfg["method"] == "grid"  # both arms must be exhausted deterministically

    cmd = cfg["command"]
    assert cmd[:2] == ["${interpreter}", "${program}"]
    assert "${args_no_hyphens}" in cmd
    # The copy base fixes the match set + gate-off; the reference URI is the patch set every
    # arm replays. Drift on either voids the per-patch pairing.
    assert "experiment=generate_dataset/copy-paired-surge-xt" in cmd
    assert (
        "copy_dataset_root_uri=r2://intermediate-data/data/ref-surge-xt-489/paired-ref-v1" in cmd
    )

    assert set(cfg["parameters"].keys()) == {"render.plugin_reload_cadence"}
    assert cfg["parameters"]["render.plugin_reload_cadence"]["values"] == ["once", "render"]
