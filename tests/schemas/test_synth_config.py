"""Contract tests for the top-level ``synth`` identity config group (#2565).

Pins the hoisted synth-identity contracts: the ``configs/synth/`` group stays a
bijective projection of the ``SYNTHS`` registry, the train root declares the
group (absent until selected), and no config interpolates identity out of the
datamodule anymore. Lives here rather than beside ``TestSynthConfigGroup`` in
``tests/test_synth_spec.py`` because the schemas suite must stay importable on
a minimal install (no ``param_spec_registry`` import).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from synth_setter.resources import configs_dir
from synth_setter.synth_spec import SYNTHS
from tests.schemas.conftest import compose_train_cfg

_CONFIGS_DIR = Path(str(configs_dir()))
_SYNTH_CONFIG_DIR = _CONFIGS_DIR / "synth"


def test_synth_group_files_and_registry_are_bijective() -> None:
    """Every ``SYNTHS`` row has a ``configs/synth/<name>.yaml`` and vice versa."""
    file_names = sorted(path.stem for path in _SYNTH_CONFIG_DIR.iterdir())
    assert file_names == sorted(SYNTHS)


def test_synth_group_files_mirror_registry_identity() -> None:
    """Each group file carries exactly the registry row's five fields."""
    for name, spec in SYNTHS.items():
        content = yaml.safe_load((_SYNTH_CONFIG_DIR / f"{name}.yaml").read_text())
        assert content == spec.model_dump(), name


def test_train_root_selects_synth_group_by_name() -> None:
    """``synth=<name>`` composes the full identity node at the config root."""
    cfg = compose_train_cfg(overrides=["synth=surge_4"])
    assert cfg["synth"] == {
        "name": "surge_4",
        "param_spec_name": "surge_4",
        "plugin_path": "plugins/Surge XT.vst3",
        "plugin_state_path": "presets/surge-mini.vstpreset",
        "synth_version": "1.3.4",
    }


def test_train_root_defaults_synth_to_absent() -> None:
    """Without a selection no ``synth`` node exists, so consumers fail loudly."""
    cfg = compose_train_cfg()
    assert cfg.get("synth") is None


def test_no_config_interpolates_datamodule_param_spec_name() -> None:
    """Ownership contract: identity flows from ``synth``, never the datamodule."""
    offenders = [
        str(path)
        for path in _CONFIGS_DIR.rglob("*.yaml")
        if "${datamodule.param_spec_name}" in path.read_text()
    ]
    assert offenders == []
