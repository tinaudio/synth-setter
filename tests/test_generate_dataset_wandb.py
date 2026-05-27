"""Offline integration tests for ``generate_dataset`` wandb tracking.

Drives ``generate(spec, work_dir, loggers)`` against a real
``WandbLogger(offline=True)`` so spec ingestion (``log_hyperparams`` +
artifact upload) is exercised through the live wandb client without touching
the network. Shards are short-circuited via a stubbed ``object_size`` probe
so the test never invokes the renderer subprocess or rclone.
"""

from __future__ import annotations

import glob
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger

from synth_setter.cli.generate_dataset import generate
from synth_setter.pipeline.schemas.spec import DatasetSpec


def _build_spec() -> DatasetSpec:
    """Hand-built spec with a fixed ``run_id`` and 2 shards; renderer paths are placeholders.

    ``extract_renderer_version`` is stubbed in the test so the placeholder
    ``plugin_path`` never has to resolve.

    :returns: Spec with ``num_shards == 2`` and a deterministic ``run_id``.
    """
    kwargs: dict[str, object] = {
        "task_name": "wandb-track-test",
        "run_id": "wandb-track-test-20260520T000000000Z",
        "created_at": datetime(2026, 5, 20, 0, 0, 0, tzinfo=timezone.utc),
        "git_sha": "0" * 40,
        "is_repo_dirty": False,
        "output_format": "hdf5",
        "train_val_test_sizes": [8, 0, 0],
        "base_seed": 42,
        "r2": {
            "bucket": "wandb-track-bucket",
            "prefix": "data/wandb-track-test/wandb-track-test-20260520T000000000Z/",
        },
        "render": {
            "plugin_path": "plugins/fake.vst3",
            "preset_path": "presets/fake.vstpreset",
            "param_spec_name": "surge_simple",
            "renderer_version": "0.0.0-fake",
            "sample_rate": 16000,
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 1.0,
            "min_loudness": -60.0,
            "samples_per_render_batch": 4,
            "samples_per_shard": 4,
            "gui_toggle_cadence": "never",
        },
    }
    return DatasetSpec(**kwargs)  # type: ignore[arg-type]


def _scrub_wandb_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every ``WANDB_*`` env var the host may have exported.

    Operator dotenvs and CI shells typically export ``WANDB_API_KEY`` /
    ``WANDB_PROJECT`` / ``WANDB_ENTITY`` / ``WANDB_MODE``; any of these
    would steer the offline run toward a different project or trigger
    network calls and defeat the hermetic guarantee.

    :param monkeypatch: Pytest fixture used to ``delenv`` ambient
        ``WANDB_*`` keys for the duration of the calling test.
    """
    for key in [k for k in os.environ if k.startswith("WANDB_")]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _reset_wandb_session_state() -> None:
    """Drop any wandb session cached by an earlier test.

    ``wandb`` caches its session at first ``wandb.init``; without an
    explicit ``wandb.teardown`` between runs, a subsequent ``WandbLogger``
    reuses the cached session and silently ignores ``offline=True`` (the
    library logs a warning that env changes are ignored). Tearing down
    here keeps each test hermetic regardless of the runtime order
    ``pytest-randomly`` picks.
    """
    wandb.teardown()


def test_generate_logs_spec_as_hyperparams_and_artifact_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``generate`` pushes the spec as hyperparams + uploads a spec artifact.

    Skips every shard via a stubbed R2 probe so the renderer subprocess and
    rclone never run; the wandb client is the only side effect under test.

    :param tmp_path: Per-test tmp dir; the offline run lands at
        ``tmp_path/wandb/offline-run-*-<run_id>``.
    :param monkeypatch: Used to scrub ambient ``WANDB_*`` env and stub
        ``object_size`` + ``extract_renderer_version``.
    """
    _scrub_wandb_env(monkeypatch)
    monkeypatch.setenv("WANDB_MODE", "offline")

    spec = _build_spec()

    monkeypatch.setattr(
        "synth_setter.cli.generate_dataset.extract_renderer_version",
        lambda _path: spec.render.renderer_version,
    )
    # Force every shard onto the R2-skip branch in ``_render_one_owned_shard``
    # so no renderer subprocess fires.
    monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda *_a, **_k: 1_024)

    wandb_logger = WandbLogger(
        offline=True,
        save_dir=str(tmp_path),
        id=spec.run_id,
        project="wandb-track-test-project",
    )

    generate(spec, tmp_path, [wandb_logger])
    # ``generate``'s finally block calls ``wandb.finish()``, so the offline
    # ``.wandb`` binary is already flushed before the assertions run.
    assert wandb.run is None, "generate() did not close the wandb run on return"

    offline_dirs = list((tmp_path / "wandb").glob(f"offline-run-*-{spec.run_id}"))
    assert len(offline_dirs) == 1, (
        f"expected one offline-run dir for {spec.run_id}, found {offline_dirs}"
    )

    binary_files = glob.glob(str(offline_dirs[0] / "run-*.wandb"))
    assert len(binary_files) == 1, (
        f"expected one .wandb binary in {offline_dirs[0]}, found {binary_files}"
    )
    payload = Path(binary_files[0]).read_bytes()
    artifact_name = f"{spec.task_name}-input-spec"
    assert artifact_name.encode() in payload, (
        f"artifact name {artifact_name!r} not recorded in offline run binary"
    )
    assert b"dataset-spec" in payload, "artifact type 'dataset-spec' not recorded"
