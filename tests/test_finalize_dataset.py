"""Tests for the ``synth-setter-finalize-dataset`` CLI entrypoint.

The tests drive ``finalize(cfg)`` and ``main()`` against a local-typed rclone
remote. Branch-level tests live under ``tests/pipeline/entrypoints``; shared
seeders and spec builders live in ``tests/helpers/finalize_shards.py``.
"""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NoReturn, cast

import lance
import pytest
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

from synth_setter.cli import finalize_dataset
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.lance_staging import stage_lance_shard_attempt
from synth_setter.pipeline.schemas.spec import DatasetSpec
from tests.helpers.finalize_shards import (
    build_finalize_cfg,
    build_lance_smoke_spec,
    install_finalize_setup_stubs,
    stub_finalize_lance_io,
    uri_to_local_path,
    write_minimal_lance_shard,
    write_spec_to_root,
)
from tests.helpers.wandb_offline import read_history_rows, read_run_binary


@pytest.fixture()
def stub_finalize_setup(monkeypatch: pytest.MonkeyPatch) -> Callable[[int | None], None]:
    """Install the auth + marker-probe stubs and expose a marker-size setter.

    Thin wrapper over :func:`tests.helpers.finalize_shards.install_finalize_setup_stubs`
    so the entrypoint and branch lanes share one stub set.

    :param monkeypatch: Pytest fixture used to install the stubs.
    :returns: A setter that overrides the marker-probe's "size in R2"
        response — ``None`` (default) makes ``finalize()`` proceed; an
        ``int`` triggers the idempotency short-circuit.
    """
    return install_finalize_setup_stubs(monkeypatch)


def test_finalize_uploads_stats_then_marker_at_canonical_uris(
    tmp_path: Path,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_finalize_setup: Callable[[int | None], None],  # noqa: ARG001 — installs stubs only
) -> None:
    """Spy on ``r2_io.upload`` to assert ``stats.npz`` is uploaded before ``dataset.complete``.

    Order via a spy is filesystem-invariant — mtime granularity on fast
    filesystems can tie two writes inside a single ``finalize`` call. The Lance
    fragment commit is stubbed, so ``stats.npz``, ``dataset.json``, and the
    marker are the objects routed through ``r2_io.upload``.

    :param tmp_path: Pytest tmp dir; hosts the on-disk spec JSON + Hydra-style output_dir.
    :param fake_r2_remote: Local-typed rclone remote; both artifacts land here.
    :param monkeypatch: Pytest fixture used to wrap ``synth_setter.pipeline.r2_io.upload``
        with an order-recording spy that still delegates to the real helper.
    :param stub_finalize_setup: Fixture-activation only — installs the
        ``ensure_r2_env_loaded`` / ``object_size`` stubs.
    """
    spec = build_lance_smoke_spec(task_name="finalize-marker-last-lance")
    stub_finalize_lance_io(monkeypatch)
    output_dir = tmp_path / "hydra_output"
    output_dir.mkdir()
    cfg = build_finalize_cfg(write_spec_to_root(spec, tmp_path), output_dir)

    real_upload = r2_io.upload
    upload_order: list[str] = []

    def spy_upload(src: str | Path, dst: str) -> None:
        upload_order.append(dst)
        real_upload(src, dst)

    monkeypatch.setattr("synth_setter.pipeline.r2_io.upload", spy_upload)

    finalize_dataset.finalize(cfg)

    stats_uri = spec.r2.stats_uri()
    marker_uri = spec.r2.dataset_complete_marker_uri()
    assert uri_to_local_path(fake_r2_remote, stats_uri).is_file()
    assert uri_to_local_path(fake_r2_remote, marker_uri).is_file()
    assert upload_order.count(stats_uri) == 1
    assert upload_order.count(marker_uri) == 1
    assert upload_order.index(marker_uri) == len(upload_order) - 1
    assert upload_order.index(stats_uri) < upload_order.index(marker_uri)


def test_finalize_is_idempotent_when_marker_already_exists(
    tmp_path: Path,
    fake_r2_remote: Path,
    stub_finalize_setup: Callable[[int | None], None],
) -> None:
    """Marker present at run prefix → ``finalize()`` short-circuits, no stats are written.

    :param tmp_path: Pytest tmp dir; hosts the on-disk spec JSON + Hydra-style output_dir.
    :param fake_r2_remote: Local-typed rclone remote — asserted to still be
        free of any ``stats.npz`` after the no-op run.
    :param stub_finalize_setup: Used to flip the marker probe to "present".
    """
    stub_finalize_setup(0)
    spec = build_lance_smoke_spec(task_name="finalize-idempotent-lance")
    output_dir = tmp_path / "hydra_output"
    output_dir.mkdir()
    cfg = build_finalize_cfg(write_spec_to_root(spec, tmp_path), output_dir)

    finalize_dataset.finalize(cfg)

    stats_path = uri_to_local_path(fake_r2_remote, spec.r2.stats_uri())
    assert not stats_path.exists()


def test_finalize_raises_on_unsupported_output_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_finalize_setup: Callable[[int | None], None],  # noqa: ARG001 — installs stubs only
) -> None:
    """An ``output_format`` outside {lance} surfaces a clear ValueError.

    Pins the dispatcher's exhaustiveness contract — adding a second format
    without wiring its branch must trip this test rather than silently
    skip the artifact upload and write a misleading ``dataset.complete``.
    The fail-fast ``download_to_path`` / ``upload`` stubs make this a
    positive short-circuit check: the ValueError alone would still pass if
    the spec load moved *after* dispatch, but a download or upload firing
    before the raise proves dispatch ran and fails the test.

    :param tmp_path: Pytest tmp dir; hosts the Hydra-style output_dir.
    :param monkeypatch: Pytest fixture used to install a stub loader plus
        fail-fast download/upload stubs.
    :param stub_finalize_setup: Installs the auth + marker-probe stubs so the
        dispatcher (not the marker check) is the failure surface.
    """
    # model_copy bypasses validation, so an off-enum token reaches the dispatcher.
    bad_spec = build_lance_smoke_spec(task_name="finalize-bad-format").model_copy(
        update={"output_format": "parquet"}
    )
    monkeypatch.setattr(
        "synth_setter.cli.finalize_dataset.load_spec_from_root", lambda _uri: bad_spec
    )
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_to_path",
        lambda *a, **kw: pytest.fail("download_to_path should not be reached"),
    )
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.upload",
        lambda *a, **kw: pytest.fail("upload should not be reached"),
    )
    output_dir = tmp_path / "hydra_output"
    output_dir.mkdir()
    cfg = build_finalize_cfg("file:///unused", output_dir)

    with pytest.raises(ValueError, match="unsupported output_format"):
        finalize_dataset.finalize(cfg)


def test_finalize_dataset_main_resolves_hydra_logging_under_at_hydra_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_finalize_setup: Callable[[int | None], None],
) -> None:
    """Invoking ``main()`` under @hydra.main resolves every interpolation in the shared groups.

    The shared ``hydra/default.yaml`` interpolates ``${task_name}`` into both
    ``run.dir`` and ``job_logging.handlers.file.filename``, and the composed
    ``logger: wandb`` group interpolates ``${paths.output_dir}`` +
    ``${oc.env:WANDB_*}``. A missing override surfaces as a Hydra startup
    ``InterpolationKeyError`` *before* ``finalize()`` fires — a structure-only
    compose check (``return_hydra_config=True``) inspects unresolved templates
    and misses this. Drive the decorated ``main()`` for real with the
    marker-probe stub set to "present" so the body short-circuits at the
    idempotency check, isolating the test to Hydra-side resolution.
    ``WANDB_MODE=disabled`` makes the composed WandbLogger a no-op so no
    network or run dir is created.

    :param tmp_path: Hosts ``PROJECT_ROOT``, the on-disk spec JSON, and Hydra's run dir.
    :param monkeypatch: Pytest fixture used to point ``PROJECT_ROOT`` + ``sys.argv``.
    :param stub_finalize_setup: Used to flip the marker probe to "present" so the
        body skips the Lance dispatch.
    """
    stub_finalize_setup(0)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    spec = build_lance_smoke_spec(task_name="hydra-startup")
    dataset_root_uri = write_spec_to_root(spec, tmp_path)
    monkeypatch.setattr("sys.argv", ["finalize_dataset", f"dataset_root_uri={dataset_root_uri}"])

    finalize_dataset.main()

    # Hydra's run dir for this invocation lands under PROJECT_ROOT/logs/finalize_dataset/.
    # Existence proves @hydra.main resolved ${paths.log_dir}, ${now:…}, and
    # the ${task_name} interpolations the shared hydra group references.
    assert (tmp_path / "logs" / "finalize_dataset").is_dir()


def _build_finalize_cfg_with_offline_wandb(
    dataset_root_uri: str, output_dir: Path, save_dir: Path
) -> DictConfig:
    """Build a ``finalize()`` cfg carrying an offline ``WandbLogger`` group.

    Mirrors the production logger composition (``_target_`` + project) but
    pins ``offline=True`` and a tmp ``save_dir`` so ``finalize`` instantiates a
    real, hermetic wandb run rather than a no-op empty logger list.

    :param dataset_root_uri: Run-prefix URI passed through to ``load_spec_from_root``.
    :param output_dir: Finalize's scratch ``work_dir`` (must exist).
    :param save_dir: Where the offline run's ``wandb/`` dir is written.
    :returns: Mutable DictConfig with ``dataset_root_uri``, ``paths``, ``logger``.
    """
    return cast(
        DictConfig,
        OmegaConf.create(
            {
                "dataset_root_uri": dataset_root_uri,
                "paths": {"output_dir": str(output_dir)},
                "logger": {
                    "wandb": {
                        "_target_": "lightning.pytorch.loggers.wandb.WandbLogger",
                        "offline": True,
                        "save_dir": str(save_dir),
                        "id": None,
                        "job_type": "",
                        "project": "finalize-wandb-test-project",
                    }
                },
            }
        ),
    )


def test_finalize_logs_dataset_artifact_to_offline_wandb_run(
    tmp_path: Path,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real fragment finalization persists artifacts and offline W&B progress.

    :param tmp_path: Hosts staged data, the scratch work dir, and offline run dir.
    :param fake_r2_remote: Local-typed rclone remote where final artifacts land.
    :param monkeypatch: Pins a hermetic offline ``WANDB_*`` environment.
    """
    for key in [k for k in os.environ if k.startswith("WANDB_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "wandb-data"))
    wandb.teardown()

    spec = build_lance_smoke_spec(task_name="finalize-artifact-e2e")
    local_shard = tmp_path / "generated" / spec.shards[0].filename
    write_minimal_lance_shard(local_shard, spec)
    stage_lance_shard_attempt(
        spec,
        spec.shards[0],
        local_shard,
        worker_id="pod-a",
        attempt_uuid="u0000",
    )
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda: None)
    output_dir = tmp_path / "work"
    output_dir.mkdir()
    cfg = _build_finalize_cfg_with_offline_wandb(
        write_spec_to_root(spec, tmp_path), output_dir, tmp_path
    )

    finalize_dataset.finalize(cfg)
    assert wandb.run is None, "finalize() did not close the wandb run on return"

    offline_dirs = list((tmp_path / "wandb").glob(f"offline-run-*-{spec.run_id}"))
    assert len(offline_dirs) == 1, (
        f"expected one offline-run dir for {spec.run_id}, found {offline_dirs}"
    )
    binary_files = glob.glob(str(offline_dirs[0] / "run-*.wandb"))
    assert len(binary_files) == 1, (
        f"expected one .wandb binary in {offline_dirs[0]}, found {binary_files}"
    )

    artifact_name = f"data-{spec.task_name}"
    # The Lance run references the finalized split dataset + stats.npz; pin the
    # exact stats s3 URI rather than a bare `s3://` so the assertion can't pass
    # on an incidental reference. Bucket/prefix come straight off the spec.
    stats_ref = f"s3://{spec.r2.bucket}/{spec.r2.prefix}stats.npz"
    payload = read_run_binary(
        Path(binary_files[0]),
        until=lambda data: artifact_name.encode() in data and stats_ref.encode() in data,
    )
    assert artifact_name.encode() in payload, (
        f"dataset artifact {artifact_name!r} not recorded in offline run binary"
    )
    assert b"dataset" in payload, "artifact type 'dataset' not recorded"
    assert stats_ref.encode() in payload, (
        f"finalized stats reference {stats_ref!r} not recorded on the artifact"
    )
    # Metadata block round-trips through the real log → binary path.
    assert b"n_samples" in payload and b"git_sha" in payload, (
        "artifact metadata (n_samples / git_sha) not recorded in offline run binary"
    )

    rows = read_history_rows(
        Path(binary_files[0]),
        until=lambda scanned: (
            sum("finalize/shards_processed" in row for row in scanned) >= spec.num_shards
            and any("finalize/elapsed_seconds" in row for row in scanned)
        ),
    )
    shard_rows = [
        row
        for row in rows
        if "finalize/shards_processed" in row and "finalize/elapsed_seconds" not in row
    ]
    assert [json.loads(row["finalize/shards_processed"]) for row in shard_rows] == list(
        range(1, spec.num_shards + 1)
    )
    assert all(json.loads(row["finalize/shards_total"]) == spec.num_shards for row in shard_rows)
    artifact_rows = [
        row
        for row in rows
        if "finalize/artifacts_uploaded" in row and "finalize/elapsed_seconds" not in row
    ]
    assert [json.loads(row["finalize/artifacts_uploaded"]) for row in artifact_rows] == [
        1,
        2,
        3,
        4,
    ]
    summary_rows = [row for row in rows if "finalize/elapsed_seconds" in row]
    assert len(summary_rows) == 1
    summary = summary_rows[0]
    assert json.loads(summary["finalize/shards_processed"]) == spec.num_shards
    assert json.loads(summary["finalize/artifacts_uploaded"]) == 4
    assert json.loads(summary["finalize/elapsed_seconds"]) >= 0

    run_root = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    assert lance.dataset(str(run_root / "train.lance")).count_rows() == 4
    assert (run_root / "stats.npz").is_file()
    assert (run_root / "dataset.json").is_file()
    assert (run_root / "dataset.complete").is_file()
    assert not (run_root / "val.lance").exists()
    assert not (run_root / "test.lance").exists()


def test_finalize_keeps_r2_artifacts_when_live_metric_logging_fails(
    tmp_path: Path,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_finalize_setup: Callable[[int | None], None],  # noqa: ARG001 — installs stubs only
) -> None:
    """A W&B history failure does not interrupt standalone finalization.

    :param tmp_path: Hosts the spec JSON, scratch work dir, and offline run dir.
    :param fake_r2_remote: Local-typed rclone remote holding final artifacts.
    :param monkeypatch: Makes the live W&B metric sink fail for every progress row.
    :param stub_finalize_setup: Installs the auth and absent-marker stubs.
    """
    for key in [key for key in os.environ if key.startswith("WANDB_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "wandb-data"))
    wandb.teardown()
    metric_calls: list[Mapping[str, float]] = []

    def fail_log_metrics(
        self: WandbLogger,
        metrics: Mapping[str, float],
        *args: object,
        **kwargs: object,
    ) -> None:
        del self, args, kwargs
        metric_calls.append(metrics)
        raise RuntimeError("simulated W&B history outage")

    monkeypatch.setattr(WandbLogger, "log_metrics", fail_log_metrics)
    spec = build_lance_smoke_spec(task_name="finalize-metrics-swallow")
    stub_finalize_lance_io(monkeypatch)
    output_dir = tmp_path / "work"
    output_dir.mkdir()
    cfg = _build_finalize_cfg_with_offline_wandb(
        write_spec_to_root(spec, tmp_path), output_dir, tmp_path
    )

    finalize_dataset.finalize(cfg)

    assert len(metric_calls) == spec.num_shards + 5
    assert uri_to_local_path(fake_r2_remote, spec.r2.stats_uri()).is_file()
    assert uri_to_local_path(fake_r2_remote, spec.r2.dataset_complete_marker_uri()).is_file()


def test_finalize_failure_logs_partial_summary_then_closes_loggers_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_finalize_setup: Callable[[int | None], None],  # noqa: ARG001 — installs stubs only
) -> None:
    """A partial failure records its traceback and terminal W&B totals before closing.

    :param tmp_path: Hosts the spec JSON, scratch work_dir, and offline run dir.
    :param monkeypatch: Pins offline W&B, injects partial progress followed by failure, and records
        exception logging plus logger-close status.
    :param stub_finalize_setup: Installs the auth + marker-probe stubs.
    """
    for key in [k for k in os.environ if k.startswith("WANDB_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "wandb-data"))
    wandb.teardown()

    def boom(
        spec: DatasetSpec,
        work_dir: Path,
        progress_callback: finalize_dataset.FinalizeProgressCallback | None = None,
    ) -> NoReturn:
        del spec, work_dir
        assert progress_callback is not None
        progress_callback("shard_processed")
        progress_callback("artifact_uploaded")
        raise RuntimeError("simulated finalize_from_spec failure")

    monkeypatch.setattr("synth_setter.cli.finalize_dataset.finalize_from_spec", boom)

    exception_messages: list[str] = []

    class SpyLogger:
        def exception(self, message: str) -> None:
            exception_messages.append(message)

    monkeypatch.setattr(finalize_dataset, "logger", SpyLogger())

    real_close = finalize_dataset.close_loggers
    close_statuses: list[str] = []

    def spy_close(loggers: list[object], status: str) -> None:
        close_statuses.append(status)
        real_close(loggers, status)  # type: ignore[arg-type]

    monkeypatch.setattr("synth_setter.cli.finalize_dataset.close_loggers", spy_close)

    spec = build_lance_smoke_spec(task_name="finalize-failed-close")
    output_dir = tmp_path / "work"
    output_dir.mkdir()
    cfg = _build_finalize_cfg_with_offline_wandb(
        write_spec_to_root(spec, tmp_path), output_dir, tmp_path
    )

    with pytest.raises(RuntimeError, match="simulated finalize_from_spec failure"):
        finalize_dataset.finalize(cfg)

    assert exception_messages == [""]
    assert close_statuses == ["failed"], (
        f"finalize() must close loggers exactly once as failed, got {close_statuses}"
    )
    assert wandb.run is None, "finalize() left the wandb run open after a failed body"

    run_binary = next((tmp_path / "wandb").glob("offline-run-*/run-*.wandb"))
    summary_rows = [
        row for row in read_history_rows(run_binary) if "finalize/elapsed_seconds" in row
    ]
    assert len(summary_rows) == 1
    assert json.loads(summary_rows[0]["finalize/shards_processed"]) == 1
    assert json.loads(summary_rows[0]["finalize/artifacts_uploaded"]) == 1
    assert json.loads(summary_rows[0]["finalize/elapsed_seconds"]) >= 0


def test_finalize_swallows_artifact_log_failure_and_keeps_r2_artifacts(
    tmp_path: Path,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_finalize_setup: Callable[[int | None], None],  # noqa: ARG001 — installs stubs only
) -> None:
    """A wandb ``log_artifact`` failure is swallowed; ``finalize()`` still lands R2 artifacts.

    Pins ``_log_dataset_artifact``'s swallow contract: artifact logging runs
    *after* the R2 outputs and ``dataset.complete`` marker are already written,
    so a wandb failure must not abort the completed finalize. Injects the failure
    at ``build_dataset_artifact`` (called inside the ``try``) rather than spying
    ``log_artifact`` — patching the builder is the smallest seam that drives the
    ``except`` branch deterministically. A ``called`` flag asserts the builder
    actually ran, so a logger-type mismatch that skipped the artifact path
    entirely could not pass the test vacuously. State-based witness: the return
    is exception-free and both ``stats.npz`` and the marker exist on the fake
    remote.

    :param tmp_path: Hosts the spec JSON, scratch work_dir, and offline run dir.
    :param fake_r2_remote: Local-typed rclone remote where stats + marker land.
    :param monkeypatch: Pins a hermetic offline ``WANDB_*`` env, stubs the Lance
        direct-R2 I/O, and raises from ``build_dataset_artifact``.
    :param stub_finalize_setup: Installs the auth + marker-probe stubs.
    """
    for key in [k for k in os.environ if k.startswith("WANDB_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "wandb-data"))
    wandb.teardown()

    builder_calls: list[str] = []

    def boom(spec: DatasetSpec) -> NoReturn:
        builder_calls.append(spec.task_name)
        raise RuntimeError("simulated build_dataset_artifact failure")

    monkeypatch.setattr("synth_setter.cli.finalize_dataset.build_dataset_artifact", boom)

    spec = build_lance_smoke_spec(task_name="finalize-artifact-swallow")
    stub_finalize_lance_io(monkeypatch)
    output_dir = tmp_path / "work"
    output_dir.mkdir()
    cfg = _build_finalize_cfg_with_offline_wandb(
        write_spec_to_root(spec, tmp_path), output_dir, tmp_path
    )

    finalize_dataset.finalize(cfg)

    assert builder_calls == [spec.task_name], (
        f"build_dataset_artifact was not invoked (artifact path skipped?): {builder_calls}"
    )
    assert uri_to_local_path(fake_r2_remote, spec.r2.stats_uri()).is_file()
    assert uri_to_local_path(fake_r2_remote, spec.r2.dataset_complete_marker_uri()).is_file()
    assert wandb.run is None, "finalize() left the wandb run open after swallowing the failure"


def test_finalize_forces_wandb_resume_allow_when_wandb_cfg_present(
    tmp_path: Path,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_finalize_setup: Callable[[int | None], None],  # noqa: ARG001 — installs stubs only
) -> None:
    """``finalize()`` forces ``logger.wandb.resume="allow"`` before instantiating loggers.

    Pins finalize's resume-forcing: when ``logger.wandb`` is present in the cfg
    the run must attach to the pinned generation run rather than mint a new one.
    Captures the cfg ``instantiate_loggers`` actually receives (the cfg object is
    mutated in-place, so the captured reference reflects the forced value) and
    stops the body early via a ``finalize_from_spec`` raise — the resume mutation
    happens *before* dispatch, so no R2 work is needed and the offline run never
    opens.

    :param tmp_path: Hosts the spec JSON and scratch work_dir.
    :param fake_r2_remote: Local-typed rclone remote (unused for I/O; the body
        short-circuits) kept so the env-rooted cfg builder stays consistent.
    :param monkeypatch: Captures the ``instantiate_loggers`` argument and raises
        from ``finalize_from_spec`` to halt before any dispatch.
    :param stub_finalize_setup: Installs the auth + marker-probe stubs.
    """
    del fake_r2_remote
    captured_logger_cfg: dict[str, DictConfig] = {}

    def capture_instantiate(logger_cfg: DictConfig) -> list[object]:
        captured_logger_cfg["cfg"] = logger_cfg
        return []

    monkeypatch.setattr(
        "synth_setter.cli.finalize_dataset.instantiate_loggers", capture_instantiate
    )

    def boom(
        spec: DatasetSpec,
        work_dir: Path,
        progress_callback: finalize_dataset.FinalizeProgressCallback | None = None,
    ) -> NoReturn:
        del spec, work_dir, progress_callback
        raise RuntimeError("halt after logger setup")

    monkeypatch.setattr("synth_setter.cli.finalize_dataset.finalize_from_spec", boom)

    spec = build_lance_smoke_spec(task_name="finalize-resume-allow")
    output_dir = tmp_path / "work"
    output_dir.mkdir()
    cfg = _build_finalize_cfg_with_offline_wandb(
        write_spec_to_root(spec, tmp_path), output_dir, tmp_path
    )

    with pytest.raises(RuntimeError, match="halt after logger setup"):
        finalize_dataset.finalize(cfg)

    assert OmegaConf.select(cfg, "logger.wandb.resume") == "allow"
    assert OmegaConf.select(captured_logger_cfg["cfg"], "wandb.resume") == "allow"
