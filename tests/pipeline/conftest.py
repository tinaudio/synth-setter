"""Shared pytest fixtures for the ``tests/pipeline`` package."""

from __future__ import annotations

import copy
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture()
def fake_r2_remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Yield a tmp_path that backs the ``r2:`` rclone remote as a local filesystem.

    Sets ``RCLONE_CONFIG_R2_TYPE=local`` so rclone resolves ``r2:`` against the
    local filesystem, and chdirs into ``tmp_path`` so a URI of the form
    ``r2://<bucket>/<key>`` materializes at ``<tmp_path>/<bucket>/<key>``. Tests
    read/write the yielded path to inspect the "uploaded" objects, exercising
    the real rclone binary end-to-end instead of patching ``subprocess.check_call``.

    Skips the test if ``rclone`` is not on ``PATH`` (the binary is required in
    the CI image but might be absent on a contributor's bare clone).

    :param tmp_path: Pytest tmp dir used as the fake R2 root.
    :param monkeypatch: Pytest fixture used to set env vars and cwd.
    :yields Path: The fake R2 root path (``<tmp_path>``). A URI ``r2://<bucket>/<key>``
        materializes at ``<root>/<bucket>/<key>``.
    """
    if shutil.which("rclone") is None:
        pytest.skip("rclone binary not available on PATH")
    monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "local")
    monkeypatch.chdir(tmp_path)
    yield tmp_path


def _make_dataset_spec_kwargs(plugin_path: str = "plugins/Surge XT.vst3") -> dict[str, Any]:
    """Return DatasetSpec kwargs that build a 48-shard hdf5 spec by default."""
    return {
        "task_name": "ci-smoke-test",
        "output_format": "hdf5",
        "train_val_test_sizes": [440000, 20000, 20000],
        "base_seed": 42,
        "r2": {"bucket": "intermediate-data"},
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
            # Darwin-portable (#714).
            "gui_toggle_cadence": "never",
        },
    }


@pytest.fixture()
def valid_dataset_spec_kwargs() -> dict[str, Any]:
    """Return a fresh deep-copied DatasetSpec kwargs dict for mutation in tests."""
    return copy.deepcopy(_make_dataset_spec_kwargs())
