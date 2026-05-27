"""Stress integration test for parallel shard renders under the Linux Xvfb wrapper.

When ``render.parallel=True`` and we run on Linux, ``_render_and_upload_shard``
forks N concurrent invocations of ``src/synth_setter/scripts/run-linux-vst-headless.sh``
(resolved via :func:`synth_setter.resources.vst_headless_wrapper`). The wrapper bootstraps Xvfb +
xsettingsd + dbus for the pedalboard ``VST3Plugin`` to construct against an
X11 display. If the wrapper does not isolate Xvfb per invocation, concurrent
calls race the X server and the renderer raises ``CalledProcessError``.

This test drives the configured parallel-dispatch path through the real
wrapper against the Surge XT VST3, with R2 I/O stubbed local, and asserts
every shard's HDF5 file lands with the configured sample count. A failure
here gates whether per-thread ``DISPLAY`` provisioning (or wrapper edits)
is needed; a pass closes the open question from the design doc.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import pytest

from synth_setter.cli.generate_dataset import generate
from synth_setter.data.vst.shapes import AUDIO_FIELD
from synth_setter.pipeline.schemas.spec import DatasetSpec

pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_vst,
    pytest.mark.skipif(sys.platform != "linux", reason="X11 wrapper is Linux-only"),
]

PLUGIN_PATH = os.environ.get("SYNTH_SETTER_PLUGIN_PATH") or "plugins/Surge XT.vst3"
PRESET_PATH = "presets/surge-base.vstpreset"
RENDERER_VERSION = os.environ.get("SYNTH_SETTER_RENDERER_VERSION", "")

_NUM_SHARDS = 4
_SAMPLES_PER_SHARD = 8


@pytest.mark.skipif(
    not Path(PLUGIN_PATH).exists(),
    reason=f"VST plugin not found at {PLUGIN_PATH}",
)
@pytest.mark.skipif(
    not RENDERER_VERSION,
    reason="SYNTH_SETTER_RENDERER_VERSION must be set to the baked plugin version",
)
def test_parallel_renders_under_xvfb_wrapper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """4 concurrent renders through the real wrapper + real plugin must all complete.

    Fails (with the wrapper's ``CalledProcessError``) if the Xvfb / dbus
    bootstrap is not concurrency-safe; on pass, every shard's HDF5 file
    contains exactly ``_SAMPLES_PER_SHARD`` rows in the audio dataset.

    :param monkeypatch: Pytest fixture used to stub R2 calls and partition env.
    :param tmp_path: Pytest tmp dir used as the launcher's repo root.
    """
    spec = _build_real_surge_spec(tmp_path)
    monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
    monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")

    rclone_srcs: list[Path] = []

    def _fake_rclone(src: str, _dest: str) -> None:
        rclone_srcs.append(Path(src))

    monkeypatch.setattr("synth_setter.cli.generate_dataset._rclone_copy", _fake_rclone)
    monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda *_a, **_k: None)

    generate(spec, tmp_path, [])

    assert len(rclone_srcs) == _NUM_SHARDS
    for path in rclone_srcs:
        assert path.exists(), f"rclone source missing: {path}"
        with h5py.File(path, "r") as h5:
            dataset = h5[AUDIO_FIELD]
            assert dataset.shape[0] == _SAMPLES_PER_SHARD  # type: ignore[attr-defined]


def _build_real_surge_spec(_tmp_path: Path) -> DatasetSpec:
    """Build a ``DatasetSpec`` pinned to the real Surge XT plugin for stress testing.

    :param _tmp_path: Reserved for tests that need a per-run scratch root; the
        spec itself carries no filesystem state beyond the plugin/preset paths.
    :returns: ``DatasetSpec`` with ``render.parallel=True``, ``_NUM_SHARDS``
        shards, and ``_SAMPLES_PER_SHARD`` samples per shard.
    """
    kwargs: dict[str, object] = {
        "task_name": "parallel-xvfb-stress",
        "run_id": "parallel-xvfb-stress-20260520T000000000Z",
        "created_at": datetime(2026, 5, 20, 0, 0, 0, tzinfo=timezone.utc),
        "git_sha": "0" * 40,
        "is_repo_dirty": False,
        "output_format": "hdf5",
        "train_val_test_sizes": [_SAMPLES_PER_SHARD * _NUM_SHARDS, 0, 0],
        "base_seed": 42,
        "r2": {
            "bucket": "stress-bucket",
            "prefix": "data/parallel-xvfb-stress/parallel-xvfb-stress-20260520T000000000Z/",
        },
        "render": {
            "plugin_path": PLUGIN_PATH,
            "preset_path": PRESET_PATH,
            "param_spec_name": "surge_simple",
            "renderer_version": RENDERER_VERSION,
            "sample_rate": 16000,
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 1.0,
            "min_loudness": -60.0,
            "samples_per_render_batch": _SAMPLES_PER_SHARD,
            "samples_per_shard": _SAMPLES_PER_SHARD,
            "parallel": True,
            "gui_toggle_cadence": "never",
        },
    }
    return DatasetSpec(**kwargs)  # type: ignore[arg-type]
