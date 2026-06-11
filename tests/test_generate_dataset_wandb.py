"""Offline integration tests for ``generate_dataset`` wandb tracking.

Drives ``generate`` against a real ``WandbLogger(offline=True)`` so spec
ingestion and per-shard / summary ``log_metrics`` exercise the live wandb
client without network. ``object_size`` is stubbed so every shard hits the
R2-skip branch — no renderer subprocess or rclone.
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections.abc import Callable
from pathlib import Path

import pytest
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger

from synth_setter.cli.generate_dataset import generate
from synth_setter.pipeline.schemas.spec import DatasetSpec
from tests.helpers.wandb_offline import read_history_rows, read_run_binary, read_run_config

_RUN_ID = "wandb-track-test-20260520T000000000Z"


def _build_spec(dataset_spec_factory: Callable[..., DatasetSpec]) -> DatasetSpec:
    """Build a 2-shard wandb-tracking spec from the shared factory.

    ``extract_renderer_version`` is stubbed in the test so the placeholder
    ``plugin_path`` never has to resolve.

    :param dataset_spec_factory: Shared ``conftest`` factory.
    :returns: Spec with ``num_shards == 2`` and a deterministic ``run_id``.
    """
    return dataset_spec_factory(
        task_name="wandb-track-test",
        run_id=_RUN_ID,
        train_val_test_sizes=[8, 0, 0],
        r2={
            "bucket": "wandb-track-bucket",
            "prefix": f"data/wandb-track-test/{_RUN_ID}/",
        },
        render={"samples_per_render_batch": 4, "samples_per_shard": 4},
    )


def _offline_wandb_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pin a hermetic offline wandb env so a host dotenv or home dir can't steer the run.

    Scrubs ambient ``WANDB_*`` env, forces ``WANDB_MODE=offline``, and points
    ``WANDB_DATA_DIR`` at ``tmp_path`` so artifact staging never falls back to a
    (possibly read-only) ``~/.local/share/wandb``.

    :param monkeypatch: ``delenv`` / ``setenv`` are applied for the calling test.
    :param tmp_path: Per-test tmp dir hosting the offline run and artifact staging.
    """
    for key in [k for k in os.environ if k.startswith("WANDB_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "wandb-data"))


class TestLoggersPinnedToSpec:
    """``_loggers_pinned_to_spec`` writes the run identity onto the wandb logger cfg."""

    def test_pins_run_id_and_data_generation_job_type(
        self, dataset_spec_factory: Callable[..., DatasetSpec]
    ) -> None:
        """A wandb logger cfg inherits ``spec.run_id`` and ``job_type=data-generation``.

        :param dataset_spec_factory: Shared ``conftest`` ``DatasetSpec`` factory.
        """
        from omegaconf import OmegaConf

        from synth_setter.cli.generate_dataset import _loggers_pinned_to_spec

        spec = _build_spec(dataset_spec_factory)
        # No ``_target_`` so ``instantiate_loggers`` skips construction and the test
        # isolates the cfg pinning rather than building a real WandbLogger.
        cfg = OmegaConf.create({"logger": {"wandb": {"id": None, "job_type": ""}}})

        _loggers_pinned_to_spec(cfg, spec)

        assert cfg.logger.wandb.id == spec.run_id
        assert cfg.logger.wandb.job_type == "data-generation"


@pytest.fixture(autouse=True)
def _reset_wandb_session_state() -> None:
    """Tear down any cached wandb session so each test's ``offline=True`` takes effect."""
    wandb.teardown()


def test_generate_logs_spec_as_hyperparams_and_artifact_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_spec_factory: Callable[..., DatasetSpec],
) -> None:
    """``generate`` pushes the spec as hyperparams + uploads a spec artifact.

    Skips every shard via a stubbed R2 probe so the renderer subprocess and
    rclone never run; the wandb client is the only side effect under test.

    :param tmp_path: Per-test tmp dir; the offline run lands at
        ``tmp_path/wandb/offline-run-*-<run_id>``.
    :param monkeypatch: Used to pin a hermetic offline ``WANDB_*`` env and stub
        ``object_size`` + ``extract_renderer_version``.
    :param dataset_spec_factory: Shared ``conftest`` ``DatasetSpec`` factory.
    """
    _offline_wandb_env(monkeypatch, tmp_path)

    spec = _build_spec(dataset_spec_factory)

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
    # ``generate``'s finally block calls ``wandb.finish()``, which closes the
    # run; the offline writer still flushes the ``.wandb`` binary asynchronously
    # (handled by the polling read below).
    assert wandb.run is None, "generate() did not close the wandb run on return"

    offline_dirs = list((tmp_path / "wandb").glob(f"offline-run-*-{spec.run_id}"))
    assert len(offline_dirs) == 1, (
        f"expected one offline-run dir for {spec.run_id}, found {offline_dirs}"
    )

    binary_files = glob.glob(str(offline_dirs[0] / "run-*.wandb"))
    assert len(binary_files) == 1, (
        f"expected one .wandb binary in {offline_dirs[0]}, found {binary_files}"
    )
    # The offline writer flushes the artifact record asynchronously, so poll
    # the binary until both markers land rather than reading once and racing.
    artifact_name = f"{spec.task_name}-input-spec"
    payload = read_run_binary(
        Path(binary_files[0]),
        until=lambda data: artifact_name.encode() in data and b"dataset-spec" in data,
    )
    assert artifact_name.encode() in payload, (
        f"artifact name {artifact_name!r} not recorded in offline run binary"
    )
    assert b"dataset-spec" in payload, "artifact type 'dataset-spec' not recorded"


def test_generate_logs_per_shard_and_summary_metrics_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_spec_factory: Callable[..., DatasetSpec],
) -> None:
    """``generate`` emits one history row per shard plus a terminal summary row.

    Per-shard rows carry ``shard/bytes`` (from the R2 skip-probe's
    ``existing_size``) and ``shard/render_seconds`` (``0.0`` for skips).
    The terminal row carries the ``shards/{rendered,skipped,total}``
    counters and the e2e generation triple
    (``generation/{elapsed_seconds,samples,samples_per_second}``).

    :param tmp_path: Per-test tmp dir; the offline run lands at
        ``tmp_path/wandb/offline-run-*-<run_id>``.
    :param monkeypatch: Used to pin a hermetic offline ``WANDB_*`` env and stub
        ``object_size`` + ``extract_renderer_version``.
    :param dataset_spec_factory: Shared ``conftest`` ``DatasetSpec`` factory.
    """
    _offline_wandb_env(monkeypatch, tmp_path)

    spec = _build_spec(dataset_spec_factory)

    monkeypatch.setattr(
        "synth_setter.cli.generate_dataset.extract_renderer_version",
        lambda _path: spec.render.renderer_version,
    )
    # Every shard hits the R2-skip branch so ``shard/bytes`` is the stubbed
    # ``existing_size`` and ``shard/render_seconds`` is 0.0.
    monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda *_a, **_k: 1_024)

    wandb_logger = WandbLogger(
        offline=True,
        save_dir=str(tmp_path),
        id=spec.run_id,
        project="wandb-track-test-project",
    )

    generate(spec, tmp_path, [wandb_logger])

    binary_files = glob.glob(
        str(tmp_path / "wandb" / f"offline-run-*-{spec.run_id}" / "run-*.wandb")
    )
    assert len(binary_files) == 1, f"expected exactly one .wandb binary, found {binary_files}"

    rows = read_history_rows(
        Path(binary_files[0]),
        until=lambda scanned: (
            sum("shard/bytes" in r for r in scanned) >= spec.num_shards
            and any("shards/rendered" in r for r in scanned)
        ),
    )
    shard_rows = [r for r in rows if "shard/bytes" in r]
    assert len(shard_rows) == spec.num_shards, (
        f"expected {spec.num_shards} per-shard history rows, got {len(shard_rows)}: {shard_rows}"
    )
    for r in shard_rows:
        assert json.loads(r["shard/bytes"]) == 1024, r
        assert json.loads(r["shard/render_seconds"]) == 0.0, r

    summary_rows = [r for r in rows if "shards/rendered" in r]
    assert len(summary_rows) == 1, (
        f"expected exactly one summary history row, got {len(summary_rows)}: {summary_rows}"
    )
    summary = summary_rows[0]
    assert json.loads(summary["shards/rendered"]) == 0, summary
    assert json.loads(summary["shards/skipped"]) == spec.num_shards, summary
    assert json.loads(summary["shards/total"]) == spec.num_shards, summary
    for key in (
        "generation/elapsed_seconds",
        "generation/samples",
        "generation/samples_per_second",
    ):
        assert key in summary, (key, summary)
    assert json.loads(summary["generation/samples"]) == 0, summary
    assert json.loads(summary["generation/samples_per_second"]) == 0.0, summary


def test_generate_stamps_wandb_provenance_into_run_config_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_spec_factory: Callable[..., DatasetSpec],
) -> None:
    """``generate`` stamps ``github_sha``/``image_tag``/``command`` into the run config.

    Parity with train.py/eval.py: provenance lands in ``wandb.config`` (not
    history). ``image_tag`` is pinned via ``IMAGE_TAG`` and ``command``/argv via a
    stubbed ``sys.argv`` so the stamped values — not just the keys — are asserted;
    ``github_sha`` is a 40-char hex SHA or the ``"unknown"`` git-unavailable
    sentinel. The run is still closed on return (``wandb.run is None``).

    :param tmp_path: Per-test tmp dir; the offline run lands at
        ``tmp_path/wandb/offline-run-*-<run_id>``.
    :param monkeypatch: Used to pin a hermetic offline ``WANDB_*`` env, ``IMAGE_TAG``,
        ``sys.argv``, and stub ``object_size`` + ``extract_renderer_version``.
    :param dataset_spec_factory: Shared ``conftest`` ``DatasetSpec`` factory.
    """
    _offline_wandb_env(monkeypatch, tmp_path)
    monkeypatch.setenv("IMAGE_TAG", "test-image:abc123")
    pinned_argv = ["synth-setter-generate-dataset", "experiment=smoke-shard"]
    monkeypatch.setattr("sys.argv", pinned_argv)

    spec = _build_spec(dataset_spec_factory)

    monkeypatch.setattr(
        "synth_setter.cli.generate_dataset.extract_renderer_version",
        lambda _path: spec.render.renderer_version,
    )
    monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda *_a, **_k: 1_024)

    wandb_logger = WandbLogger(
        offline=True,
        save_dir=str(tmp_path),
        id=spec.run_id,
        project="wandb-track-test-project",
    )

    generate(spec, tmp_path, [wandb_logger])
    assert wandb.run is None, "generate() did not close the wandb run on return"

    binary_files = glob.glob(
        str(tmp_path / "wandb" / f"offline-run-*-{spec.run_id}" / "run-*.wandb")
    )
    assert len(binary_files) == 1, f"expected exactly one .wandb binary, found {binary_files}"

    config = read_run_config(
        Path(binary_files[0]),
        until=lambda c: {"github_sha", "image_tag", "command"} <= c.keys(),
    )
    assert json.loads(config["command"]) == " ".join(pinned_argv), config
    assert json.loads(config["image_tag"]) == "test-image:abc123", config
    sha = json.loads(config["github_sha"])
    assert sha == "unknown" or re.fullmatch(r"[0-9a-f]{40}", sha), config
