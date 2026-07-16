"""Stress integration test for parallel shard renders under the Linux Xvfb wrapper.

When ``render.parallel=True`` and we run on Linux, ``_render_and_upload_shard``
forks N concurrent invocations of ``src/synth_setter/scripts/run-linux-vst-headless.sh``
(resolved via :func:`synth_setter.resources.vst_headless_wrapper`). The wrapper bootstraps Xvfb +
xsettingsd + dbus for the pedalboard ``VST3Plugin`` to construct against an
X11 display. If the wrapper does not isolate Xvfb per invocation, concurrent
calls race the X server and the renderer raises ``CalledProcessError``.

This test drives the configured parallel-dispatch path through the real
wrapper against the Surge XT VST3, with R2 I/O stubbed local, and asserts
every shard's Lance dataset lands with the configured sample count. A failure
here gates whether per-thread ``DISPLAY`` provisioning (or wrapper edits)
is needed; a pass closes the open question from the design doc.

It is parametrized over representative ``gui_toggle_cadence`` /
``plugin_reload_cadence`` pairs so a real render exercises each structurally
distinct toggle/reload loop. The dispatch branches are covered fast in
``tests/data/vst/test_writers.py``. The cadence cross-field *validation* (which
pairs are accepted / rejected, e.g. ``always_on`` requires
``plugin_reload_cadence="once"``) is unit-tested in
``tests/pipeline/schemas/test_dataset_spec.py`` (#1354).
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import lance
import pytest

from synth_setter.cli.generate_dataset import generate
from synth_setter.data.vst.core import extract_renderer_version
from synth_setter.pipeline.schemas.spec import DatasetSpec
from tests._vst import PLUGIN_PATH

pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_vst,
    pytest.mark.skipif(sys.platform != "linux", reason="X11 wrapper is Linux-only"),
]

PRESET_PATH = "presets/surge-base.vstpreset"

_NUM_SHARDS = 4
_SAMPLES_PER_SHARD = 8


# (gui_toggle_cadence, plugin_reload_cadence) pairs whose render loop differs:
# reload-every-render with no toggle, max churn (toggle + reload every render),
# and the always-on hold (which the validator pins to a single per-shard load).
_CADENCE_CELLS = [
    ("never", "render"),
    ("render", "render"),
    ("always_on", "once"),
]


@pytest.mark.parametrize(
    ("gui_toggle_cadence", "plugin_reload_cadence"),
    _CADENCE_CELLS,
    ids=[f"{g}-{p}" for g, p in _CADENCE_CELLS],
)
def test_parallel_renders_under_xvfb_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    gui_toggle_cadence: str,
    plugin_reload_cadence: str,
) -> None:
    """4 concurrent renders through the real wrapper + real plugin must all complete.

    Fails (with the wrapper's ``CalledProcessError``) if the Xvfb / dbus
    bootstrap is not concurrency-safe; on pass, every shard's Lance dataset
    contains exactly ``_SAMPLES_PER_SHARD`` rows.

    :param monkeypatch: Pytest fixture used to stub R2 calls and partition env.
    :param tmp_path: Pytest tmp dir used as the launcher's repo root.
    :param gui_toggle_cadence: ``render.gui_toggle_cadence`` under test.
    :param plugin_reload_cadence: ``render.plugin_reload_cadence`` paired with it.
    """
    spec = _build_real_surge_spec(tmp_path, gui_toggle_cadence, plugin_reload_cadence)
    monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
    monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")

    local_shard_paths: list[Path] = []

    def _record_lance_shard_attempt(
        _spec: object, _shard: object, local_shard_path: Path, **_kwargs: object
    ) -> None:
        local_shard_paths.append(local_shard_path)

    monkeypatch.setattr(
        "synth_setter.cli.generate_dataset.shard_has_complete_attempt",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        "synth_setter.cli.generate_dataset.write_rendering_marker",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "synth_setter.cli.generate_dataset.stage_lance_shard_attempt",
        _record_lance_shard_attempt,
    )

    generate(spec, tmp_path, [])

    assert len(local_shard_paths) == _NUM_SHARDS
    for path in local_shard_paths:
        assert path.is_dir(), f"shard dataset missing: {path}"
        assert lance.dataset(str(path)).count_rows() == _SAMPLES_PER_SHARD


def _build_real_surge_spec(
    _tmp_path: Path, gui_toggle_cadence: str, plugin_reload_cadence: str
) -> DatasetSpec:
    """Build a ``DatasetSpec`` pinned to the real Surge XT plugin for stress testing.

    :param _tmp_path: Reserved for tests that need a per-run scratch root; the
        spec itself carries no filesystem state beyond the plugin/preset paths.
    :param gui_toggle_cadence: ``render.gui_toggle_cadence`` to pin on the spec.
    :param plugin_reload_cadence: ``render.plugin_reload_cadence`` paired with it.
    :returns: ``DatasetSpec`` with ``render.parallel=True``, ``_NUM_SHARDS``
        shards, and ``_SAMPLES_PER_SHARD`` samples per shard.
    """
    # Resolve the version at runtime because fallback plugin loading requires Xvfb.
    renderer_version = os.environ.get("SYNTH_SETTER_RENDERER_VERSION") or (
        extract_renderer_version(Path(PLUGIN_PATH))
    )
    task_name = f"parallel-xvfb-stress-{gui_toggle_cadence}-{plugin_reload_cadence}"
    run_id = f"{task_name}-20260520T000000000Z"
    kwargs: dict[str, object] = {
        "task_name": task_name,
        "run_id": run_id,
        "created_at": datetime(2026, 5, 20, 0, 0, 0, tzinfo=UTC),
        "git_sha": "0" * 40,
        "is_repo_dirty": False,
        "output_format": "lance",
        "train_val_test_sizes": [_SAMPLES_PER_SHARD * _NUM_SHARDS, 0, 0],
        "base_seed": 42,
        "r2": {
            "bucket": "stress-bucket",
            "prefix": f"data/{task_name}/{run_id}/",
        },
        "render": {
            "plugin_path": PLUGIN_PATH,
            "plugin_state_path": PRESET_PATH,
            "param_spec_name": "surge_simple",
            "renderer_version": renderer_version,
            "sample_rate": 44100,
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 1.0,
            "min_loudness": -60.0,
            "samples_per_render_batch": _SAMPLES_PER_SHARD,
            "samples_per_shard": _SAMPLES_PER_SHARD,
            "parallel": True,
            "gui_toggle_cadence": gui_toggle_cadence,
            "plugin_reload_cadence": plugin_reload_cadence,
        },
    }
    return DatasetSpec(**kwargs)  # type: ignore[arg-type]
