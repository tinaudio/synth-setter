"""Offline structural assertions for the operator-facing sweeps covered below.

Each test pins one ``sweeps/*.yaml`` file's ``command:`` shape and
parameter grid (this is a curated set, not every file under ``sweeps/``)
so the contract that ``wandb agent`` executes (subprocess launch via
``${interpreter} ${program} ... ${args_no_hyphens}``) cannot drift
silently. Complements any live-backend sweep e2e by catching shape
breakage without touching the W&B API.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_SWEEPS_DIR = Path(__file__).resolve().parent.parent / "sweeps"
_SWEEP_YAML = _SWEEPS_DIR / "generate_dataset_cadence.yaml"
_COPY_REPRO_YAML = _SWEEPS_DIR / "generate_dataset_copy_paired_repro_surge_xt.yaml"

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


def test_generate_dataset_copy_paired_repro_yaml_shape() -> None:
    """Reproducibility-floor sweep replays the reference under run_id-only trial replicates."""
    cfg = yaml.safe_load(_COPY_REPRO_YAML.read_text())

    assert cfg["program"] == "src/synth_setter/cli/generate_dataset.py"
    assert cfg["method"] == "grid"

    cmd = cfg["command"]
    assert cmd[:2] == ["${interpreter}", "${program}"]
    assert "${args_no_hyphens}" in cmd
    # The copy base fixes the match set + gate-off; the reference URI is the patch set every
    # trial replays. Drift on either voids the floor measurement.
    assert "experiment=generate_dataset/copy-paired-surge-xt" in cmd
    assert _REFERENCE_URI in cmd

    # run_id is the only knob: each trial is an identical replay (distinct W&B run + R2 prefix), so
    # the cross-trial spread isolates phase-init nondeterminism. Need >= 2 distinct trials, else two
    # trials collapse onto one R2 prefix and silently halve the sample.
    assert set(cfg["parameters"].keys()) == {"run_id"}
    trials = cfg["parameters"]["run_id"]["values"]
    assert len(trials) >= 2
    assert len(set(trials)) == len(trials)
