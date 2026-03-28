from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

VALID_CONFIG = {
    "param_spec": "surge_simple",
    "plugin_path": "plugins/Surge XT.vst3",
    "output_format": "hdf5",
    "sample_rate": 16000,
    "shard_size": 10000,
    "num_shards": 48,
    "base_seed": 42,
    "splits": {"train": 44, "val": 2, "test": 2},
    "preset_path": "presets/surge-base.vstpreset",
    "channels": 2,
    "velocity": 100,
    "signal_duration_seconds": 4.0,
    "min_loudness": -55.0,
    "sample_batch_size": 32,
}


@pytest.fixture()
def valid_config_dict() -> dict:
    """Return a fresh copy of the valid config dict for mutation in tests."""
    return copy.deepcopy(VALID_CONFIG)


@pytest.fixture()
def write_config_yaml(tmp_path: Path):
    """Return a factory that writes a config YAML file with optional overrides."""

    def _write(overrides: dict | None = None) -> Path:
        data = VALID_CONFIG.copy()
        if overrides:
            data.update(overrides)
        path = tmp_path / "test-config.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return path

    return _write
