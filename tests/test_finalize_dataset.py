"""Tests for the ``synth-setter-finalize-dataset`` CLI entrypoint.

Covers tests that drive the in-process entrypoint surface — ``finalize(cfg)``
and ``main()`` — against the ``fake_r2_remote`` fixture (a local-typed rclone
remote rooted at ``tmp_path``; see ``tests/pipeline/conftest.py``). The spec is
written to disk as JSON and the cfg carries a ``file://`` URI pointing at it,
mirroring how production callers pass the R2 URI of ``input_spec.json``.

Two helpers stay stubbed because the local rclone backend can't simulate them
cleanly:

- ``ensure_r2_env_loaded`` — would require real ``RCLONE_CONFIG_R2_*`` secrets
  and a working ``rclone lsd r2:`` against real R2.
- ``object_size`` — ``rclone lsf`` against an absent key on the local backend
  exits 3 ("directory not found") instead of the empty-stdout semantics
  S3-compatible backends return; the marker probe in ``finalize()`` needs the
  "absent → None" branch, so the stub stays.

Keep this module to tests that drive ``finalize(cfg)`` / ``main()``. Branch-level
``finalize_wds`` / ``finalize_hdf5`` / ``finalize_from_spec`` tests — which
legitimately reference internals like ``reshard_dataset``, ``get_stats_hdf5``,
and ``stream_stats_wds`` — live in
``tests/pipeline/entrypoints/test_finalize_dataset_unit.py``; the
``build_dataset_artifact`` construction tests live in
``tests/pipeline/entrypoints/test_finalize_dataset_artifact.py``. Seeders and
spec builders shared across those lanes live in
``tests/helpers/finalize_shards.py``.
``tests/_meta/test_entrypoint_test_modules.py`` enforces that no private
``synth_setter.cli`` helper is imported here.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn, cast

import pytest
import wandb
from omegaconf import DictConfig, OmegaConf

from synth_setter.cli import finalize_dataset
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.schemas.spec import DatasetSpec
from tests.helpers.finalize_shards import (
    build_finalize_cfg,
    build_hdf5_smoke_spec,
    build_wds_smoke_spec,
    copy_shard_for_download,
    install_finalize_setup_stubs,
    seed_shard_files,
    seed_train_shards,
    stub_get_stats_hdf5,
    uri_to_local_path,
    write_spec_to_file,
)
from tests.helpers.wandb_offline import read_run_binary


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
    filesystems can tie two writes inside a single ``finalize`` call.

    :param tmp_path: Pytest tmp dir; hosts the on-disk spec JSON + Hydra-style output_dir.
    :param fake_r2_remote: Local-typed rclone remote; both artifacts land here.
    :param monkeypatch: Pytest fixture used to wrap ``synth_setter.pipeline.r2_io.upload``
        with an order-recording spy that still delegates to the real helper.
    :param stub_finalize_setup: Fixture-activation only — installs the
        ``ensure_r2_env_loaded`` / ``object_size`` stubs.
    """
    spec = build_wds_smoke_spec(task_name="finalize-marker-last-wds")
    seed_train_shards(fake_r2_remote, spec)
    output_dir = tmp_path / "hydra_output"
    output_dir.mkdir()
    cfg = build_finalize_cfg(write_spec_to_file(spec, tmp_path), output_dir)

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
    spec = build_wds_smoke_spec(task_name="finalize-idempotent-wds")
    output_dir = tmp_path / "hydra_output"
    output_dir.mkdir()
    cfg = build_finalize_cfg(write_spec_to_file(spec, tmp_path), output_dir)

    finalize_dataset.finalize(cfg)

    stats_path = uri_to_local_path(fake_r2_remote, spec.r2.stats_uri())
    assert not stats_path.exists()


def test_finalize_raises_on_unsupported_output_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_finalize_setup: Callable[[int | None], None],  # noqa: ARG001 — installs stubs only
) -> None:
    """An ``output_format`` outside {hdf5, wds} surfaces a clear ValueError.

    Pins the dispatcher's exhaustiveness contract — adding a third format
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
    bad_spec = build_wds_smoke_spec(task_name="finalize-bad-format").model_copy(
        update={"output_format": "parquet"}
    )
    monkeypatch.setattr(
        "synth_setter.cli.finalize_dataset.load_spec_from_uri", lambda _uri: bad_spec
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
        body skips the wds/hdf5 dispatch.
    """
    stub_finalize_setup(0)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    spec = build_wds_smoke_spec(task_name="hydra-startup")
    spec_uri = write_spec_to_file(spec, tmp_path)
    monkeypatch.setattr("sys.argv", ["finalize_dataset", f"dataset_spec_uri={spec_uri}"])

    finalize_dataset.main()

    # Hydra's run dir for this invocation lands under PROJECT_ROOT/logs/finalize_dataset/.
    # Existence proves @hydra.main resolved ${paths.log_dir}, ${now:…}, and
    # the ${task_name} interpolations the shared hydra group references.
    assert (tmp_path / "logs" / "finalize_dataset").is_dir()


def test_finalize_hdf5_marker_idempotency_short_circuits_before_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hdf5 dispatch: ``finalize()`` returns without any download when the marker exists.

    ``test_finalize_is_idempotent_when_marker_already_exists`` covers the
    wds branch; this test pins the hdf5-branch path so a regression that
    moved the marker check *inside* the format branch (after the dispatch
    table) would be caught. Positive assertion: ``object_size`` was probed
    exactly once against the marker URI (so a refactor that removed the
    probe and the dispatch entirely would fail rather than silently pass).

    :param tmp_path: Pytest tmp dir; hosts the on-disk spec JSON + output_dir.
    :param monkeypatch: Pytest fixture used to force ``object_size`` to
        return a present marker and to fail-fast on any download/upload.
    """
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_to_path",
        lambda *a, **kw: pytest.fail("download_to_path should not be reached"),
    )
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.upload",
        lambda *a, **kw: pytest.fail("upload should not be reached"),
    )
    monkeypatch.setattr("synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda *a, **k: None)
    probed_uris: list[str] = []

    def record_probe(uri: str) -> int:
        probed_uris.append(uri)
        return 0

    monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", record_probe)

    spec = build_hdf5_smoke_spec(task_name="finalize-hdf5-marker-present")
    output_dir = tmp_path / "hydra_output"
    output_dir.mkdir()
    cfg = build_finalize_cfg(write_spec_to_file(spec, tmp_path), output_dir)

    finalize_dataset.finalize(cfg)

    assert probed_uris == [spec.r2.dataset_complete_marker_uri()]


def test_finalize_hdf5_branch_uploads_marker_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hdf5 ``finalize(cfg)`` path writes ``dataset.complete`` strictly after every artifact.

    Pins the ``pipeline/CLAUDE.md`` ordering invariant for hdf5: an
    interrupted run must never leave a marker without the artifacts it
    advertises.

    :param tmp_path: Pytest tmp dir; hosts the fake R2 root + on-disk spec + output_dir.
    :param monkeypatch: Pytest fixture used to patch the full transport surface.
    """
    r2_stand_in = tmp_path / "r2"
    spec = build_hdf5_smoke_spec(task_name="finalize-hdf5-marker-last")
    seed_shard_files(r2_stand_in, spec)
    upload_order: list[str] = []

    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_to_path",
        lambda r2_uri, dst: copy_shard_for_download(r2_stand_in, r2_uri, dst),
    )

    def record_upload(src: str | Path, dst: str) -> None:
        del src
        upload_order.append(dst)

    monkeypatch.setattr("synth_setter.pipeline.r2_io.upload", record_upload)
    monkeypatch.setattr("synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda *a, **k: None)
    monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda _uri: None)
    stub_get_stats_hdf5(monkeypatch)

    output_dir = tmp_path / "hydra_output"
    output_dir.mkdir()
    cfg = build_finalize_cfg(write_spec_to_file(spec, tmp_path), output_dir)

    finalize_dataset.finalize(cfg)

    marker_uri = spec.r2.dataset_complete_marker_uri()
    marker_index = upload_order.index(marker_uri)
    # Marker strictly later than every artifact URI — the
    # ``pipeline/CLAUDE.md`` invariant ("never leave a marker without
    # artifacts") generalizes to splits-with-val-test, not just train.
    assert marker_index == len(upload_order) - 1
    for artifact_uri in (spec.r2.stats_uri(), spec.r2.split_h5_uri("train")):
        assert upload_order.index(artifact_uri) < marker_index


def _build_finalize_cfg_with_offline_wandb(
    spec_uri: str, output_dir: Path, save_dir: Path
) -> DictConfig:
    """Build a ``finalize()`` cfg carrying an offline ``WandbLogger`` group.

    Mirrors the production logger composition (``_target_`` + project) but
    pins ``offline=True`` and a tmp ``save_dir`` so ``finalize`` instantiates a
    real, hermetic wandb run rather than a no-op empty logger list.

    :param spec_uri: URI passed through to ``load_spec_from_uri``.
    :param output_dir: Finalize's scratch ``work_dir`` (must exist).
    :param save_dir: Where the offline run's ``wandb/`` dir is written.
    :returns: Mutable DictConfig with ``dataset_spec_uri``, ``paths``, ``logger``.
    """
    return cast(
        DictConfig,
        OmegaConf.create(
            {
                "dataset_spec_uri": spec_uri,
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
    stub_finalize_setup: Callable[[int | None], None],  # noqa: ARG001 — installs stubs only
) -> None:
    """``finalize(cfg)`` end-to-end logs a ``data-{id}`` ``dataset`` artifact with an R2 ref.

    Drives the real entrypoint against the local-typed remote (real rclone for
    the stats + marker writes) and a real ``WandbLogger(offline=True)``, then
    decodes the offline ``run-*.wandb`` binary to confirm the canonical dataset
    artifact landed — the producer node of the lineage DAG (#1471). No wandb
    internals are mocked; the artifact name, type, and ``s3://`` reference are
    read back from the bytes the live client wrote.

    :param tmp_path: Hosts the spec JSON, scratch work_dir, and offline run dir.
    :param fake_r2_remote: Local-typed rclone remote; seeded train shards land
        here so the wds stats pass has real tars to stream.
    :param monkeypatch: Pins a hermetic offline ``WANDB_*`` env.
    :param stub_finalize_setup: Installs the auth + marker-probe stubs.
    """
    for key in [k for k in os.environ if k.startswith("WANDB_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "wandb-data"))
    wandb.teardown()

    spec = build_wds_smoke_spec(task_name="finalize-artifact-e2e")
    seed_train_shards(fake_r2_remote, spec)
    output_dir = tmp_path / "work"
    output_dir.mkdir()
    cfg = _build_finalize_cfg_with_offline_wandb(
        write_spec_to_file(spec, tmp_path), output_dir, tmp_path
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
    # The wds run references the prefix dir + stats.npz; pin the exact stats
    # s3 URI rather than a bare `s3://` so the assertion can't pass on an
    # incidental reference. Bucket/prefix come straight off the spec.
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


def test_finalize_closes_loggers_failed_when_finalize_from_spec_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_finalize_setup: Callable[[int | None], None],  # noqa: ARG001 — installs stubs only
) -> None:
    """A ``finalize_from_spec`` failure propagates but still closes the wandb run as failed.

    Pins the ``finalize()`` ``try/finally`` contract (finalize_dataset.py:322-330):
    the body's exception must re-raise *and* the loggers must be closed with
    ``status="failed"`` so the data-generation run is not left dangling. A real
    offline ``WandbLogger`` is instantiated (via the offline-wandb cfg builder)
    so the close path runs ``wandb.finish()`` for real. ``close_loggers`` is
    wrapped with a spy that still delegates to the real helper: the spy captures
    the forwarded ``status`` (the state-based ``wandb.run is None`` witness alone
    can't distinguish ``"success"`` from ``"failed"``, nor a ``finally`` that
    skips the close entirely, because wandb teardown can null the run by other
    means). The failure is injected at ``finalize_from_spec`` so the ``except``
    sets ``status="failed"`` before re-raising.

    :param tmp_path: Hosts the spec JSON, scratch work_dir, and offline run dir.
    :param monkeypatch: Pins a hermetic offline ``WANDB_*`` env, raises from
        ``finalize_from_spec``, and wraps ``close_loggers`` with the status spy.
    :param stub_finalize_setup: Installs the auth + marker-probe stubs.
    """
    for key in [k for k in os.environ if k.startswith("WANDB_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "wandb-data"))
    wandb.teardown()

    def boom(spec: DatasetSpec, work_dir: Path) -> NoReturn:
        del spec, work_dir
        raise RuntimeError("simulated finalize_from_spec failure")

    monkeypatch.setattr("synth_setter.cli.finalize_dataset.finalize_from_spec", boom)

    real_close = finalize_dataset.close_loggers
    close_statuses: list[str] = []

    def spy_close(loggers: list[object], status: str) -> None:
        close_statuses.append(status)
        real_close(loggers, status)  # type: ignore[arg-type]

    monkeypatch.setattr("synth_setter.cli.finalize_dataset.close_loggers", spy_close)

    spec = build_wds_smoke_spec(task_name="finalize-failed-close")
    output_dir = tmp_path / "work"
    output_dir.mkdir()
    cfg = _build_finalize_cfg_with_offline_wandb(
        write_spec_to_file(spec, tmp_path), output_dir, tmp_path
    )

    with pytest.raises(RuntimeError, match="simulated finalize_from_spec failure"):
        finalize_dataset.finalize(cfg)

    assert close_statuses == ["failed"], (
        f"finalize() must close loggers exactly once as failed, got {close_statuses}"
    )
    assert wandb.run is None, "finalize() left the wandb run open after a failed body"


def test_finalize_swallows_artifact_log_failure_and_keeps_r2_artifacts(
    tmp_path: Path,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_finalize_setup: Callable[[int | None], None],  # noqa: ARG001 — installs stubs only
) -> None:
    """A wandb ``log_artifact`` failure is swallowed; ``finalize()`` still lands R2 artifacts.

    Pins ``_log_dataset_artifact``'s swallow contract (finalize_dataset.py:288-291):
    artifact logging runs *after* the R2 outputs and ``dataset.complete`` marker
    are already written, so a wandb failure must not abort the completed
    finalize. Injects the failure at ``build_dataset_artifact`` (called inside the
    ``try``) rather than spying ``log_artifact`` because the production wds path
    references the prefix dir — patching the builder is the smallest seam that
    drives the ``except`` branch deterministically. A ``called`` flag asserts the
    builder actually ran, so a logger-type mismatch that skipped the artifact
    path entirely could not pass the test vacuously. State-based witness: the
    return is exception-free and both ``stats.npz`` and the marker exist on the
    fake remote.

    :param tmp_path: Hosts the spec JSON, scratch work_dir, and offline run dir.
    :param fake_r2_remote: Local-typed rclone remote; seeded train shards land
        here so the wds stats pass has real tars and the outputs materialize.
    :param monkeypatch: Pins a hermetic offline ``WANDB_*`` env and raises from
        ``build_dataset_artifact``.
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

    spec = build_wds_smoke_spec(task_name="finalize-artifact-swallow")
    seed_train_shards(fake_r2_remote, spec)
    output_dir = tmp_path / "work"
    output_dir.mkdir()
    cfg = _build_finalize_cfg_with_offline_wandb(
        write_spec_to_file(spec, tmp_path), output_dir, tmp_path
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

    Pins finalize_dataset.py:319-320: when ``logger.wandb`` is present in the cfg
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

    def boom(spec: DatasetSpec, work_dir: Path) -> NoReturn:
        del spec, work_dir
        raise RuntimeError("halt after logger setup")

    monkeypatch.setattr("synth_setter.cli.finalize_dataset.finalize_from_spec", boom)

    spec = build_wds_smoke_spec(task_name="finalize-resume-allow")
    output_dir = tmp_path / "work"
    output_dir.mkdir()
    cfg = _build_finalize_cfg_with_offline_wandb(
        write_spec_to_file(spec, tmp_path), output_dir, tmp_path
    )

    with pytest.raises(RuntimeError, match="halt after logger setup"):
        finalize_dataset.finalize(cfg)

    assert OmegaConf.select(cfg, "logger.wandb.resume") == "allow"
    assert OmegaConf.select(captured_logger_cfg["cfg"], "wandb.resume") == "allow"
