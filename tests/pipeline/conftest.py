from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

# Legacy flat YAML shape used by the load_dataset_spec_yaml bridge (removed
# in Phase A.3). Tests that round-trip a YAML file through the bridge use
# this shape; tests that build DatasetSpec directly should use ``valid_dataset_spec_kwargs``.
LEGACY_VALID_YAML: dict[str, Any] = {
    "param_spec": "surge_simple",
    "plugin_path": "plugins/Surge XT.vst3",
    "output_format": "hdf5",
    "sample_rate": 16000,
    "shard_size": 10000,
    "num_shards": 48,
    "base_seed": 42,
    "r2_bucket": "intermediate-data",
    "splits": {"train": 44, "val": 2, "test": 2},
    "preset_path": "presets/surge-base.vstpreset",
    "channels": 2,
    "velocity": 100,
    "signal_duration_seconds": 4.0,
    "min_loudness": -55.0,
    "sample_batch_size": 32,
}


def _make_dataset_spec_kwargs(plugin_path: str = "plugins/Surge XT.vst3") -> dict[str, Any]:
    """Return DatasetSpec kwargs that build a 48-shard hdf5 spec by default."""
    return {
        "task_name": "ci-smoke-test",
        "output_format": "hdf5",
        "train_val_test_sizes": (440000, 20000, 20000),
        "train_val_test_seeds": (42, 43, 44),
        "base_seed": 42,
        "r2_bucket": "intermediate-data",
        "render": {
            "plugin_path": plugin_path,
            "preset_path": "presets/surge-base.vstpreset",
            "param_spec_name": "surge_simple",
            "renderer_version": "1.3.4",
            "sample_rate": 16000,
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "sample_batch_size": 32,
            "batch_per_shard": 10000,
        },
    }


@pytest.fixture()
def valid_dataset_spec_kwargs() -> dict[str, Any]:
    """Return a fresh deep-copied DatasetSpec kwargs dict for mutation in tests."""
    return copy.deepcopy(_make_dataset_spec_kwargs())


@pytest.fixture()
def write_legacy_config_yaml(tmp_path: Path):
    """Return a factory that writes a legacy-shape config YAML with optional overrides."""

    def _write(overrides: dict | None = None) -> Path:
        data = LEGACY_VALID_YAML.copy()
        if overrides:
            data.update(overrides)
        path = tmp_path / "ci-smoke-test.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return path

    return _write
