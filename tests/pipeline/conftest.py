"""Shared pytest fixtures for the ``tests/pipeline`` package."""

from __future__ import annotations

import copy
from typing import Any

import pytest


def _make_dataset_spec_kwargs(plugin_path: str = "plugins/Surge XT.vst3") -> dict[str, Any]:
    """Return DatasetSpec kwargs that build a 48-shard hdf5 spec by default."""
    return {
        "task_name": "ci-smoke-test",
        "output_format": "hdf5",
        "train_val_test_sizes": [440000, 20000, 20000],
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
            "samples_per_render_batch": 32,
            "samples_per_shard": 10000,
            # Pin off so DatasetSpec construction stays darwin-portable —
            # see ``_open_gui_every_render_forbidden_on_darwin`` (#714).
            "open_gui_every_render": False,
        },
    }


@pytest.fixture()
def valid_dataset_spec_kwargs() -> dict[str, Any]:
    """Return a fresh deep-copied DatasetSpec kwargs dict for mutation in tests."""
    return copy.deepcopy(_make_dataset_spec_kwargs())
