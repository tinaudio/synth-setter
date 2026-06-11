"""Branch-level unit tests for ``synth_setter.cli.finalize_dataset``.

The entrypoint surface — ``finalize(cfg)`` / ``main()`` — is exercised in the
canonical ``tests/test_finalize_dataset.py`` module, which
``tests/_meta/test_entrypoint_test_modules.py`` guards to stay free of private
``synth_setter.cli`` references. This sibling module holds the per-branch tests
that drive ``finalize_wds`` / ``finalize_hdf5`` / ``finalize_from_spec`` directly
and legitimately reference module internals (``reshard_dataset``,
``get_stats_hdf5``, ``stream_stats_wds``) to pin the wds/hdf5 folds, the
reshard delegation, and the marker-last ordering at the branch altitude — so it
is deliberately NOT on the entrypoint-only rail.

Seeders and smoke-spec builders shared with the entrypoint and real-R2 lanes
live in ``tests/helpers/finalize_shards.py``.
"""

from __future__ import annotations

import inspect
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, cast
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest

from synth_setter.cli import finalize_dataset
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.stats import get_stats_hdf5 as real_get_stats_hdf5
from synth_setter.pipeline.data.stats import stream_stats_wds as real_stream_stats_wds
from tests.helpers.finalize_shards import (
    build_hdf5_smoke_spec,
    build_lance_smoke_spec,
    build_wds_smoke_spec,
    copy_shard_for_download,
    install_finalize_setup_stubs,
    seed_shard_files,
    seed_train_shards,
    stub_get_stats_hdf5,
    uri_to_local_path,
    write_minimal_lance_shard,
    write_minimal_wds_shard,
)


@pytest.fixture()
def stub_finalize_setup(monkeypatch: pytest.MonkeyPatch) -> Callable[[int | None], None]:
    """Install the auth + marker-probe stubs so ``finalize_from_spec`` proceeds.

    Thin wrapper over :func:`tests.helpers.finalize_shards.install_finalize_setup_stubs`;
    the branch-level ``finalize_from_spec`` tests need the same stub set as the
    entrypoint lane because that path also probes the ``dataset.complete`` marker.

    :param monkeypatch: Pytest fixture used to install the stubs.
    :returns: A setter overriding the marker-probe response (``None`` proceeds;
        an ``int`` short-circuits).
    """
    return install_finalize_setup_stubs(monkeypatch)


def _stage_for(uploads: dict[str, Path], destination_uri: str, tmp_path: Path) -> Path:
    """Allocate a unique local staging path under ``tmp_path`` for a fake upload.

    :param uploads: Mutable mapping that records ``destination_uri → local copy``.
    :param destination_uri: The would-be R2 URI of the upload.
    :param tmp_path: Test-scoped tmp dir to host the staged copy.
    :returns: A fresh path that ``shutil.copy`` can write to.
    """
    staged_root = tmp_path / "uploads"
    staged_root.mkdir(exist_ok=True)
    staged = staged_root / f"{len(uploads):03d}_{destination_uri.rsplit('/', 1)[-1]}"
    uploads[destination_uri] = staged
    return staged


def test_finalize_from_spec_uploads_stats_then_marker_at_canonical_uris(
    tmp_path: Path,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_finalize_setup: Callable[[int | None], None],  # noqa: ARG001 — installs stubs only
) -> None:
    """``finalize_from_spec`` honors the marker-last ordering without re-loading the spec.

    Mirrors ``test_finalize_uploads_stats_then_marker_at_canonical_uris`` but
    calls the in-memory entry point directly — no ``cfg`` synthesis, no
    ``load_spec_from_root`` round-trip — so the inline path
    (``generate_dataset.main`` will reuse) is pinned independently of the
    URI-driven entry point.

    :param tmp_path: Hosts the Hydra-style work_dir.
    :param fake_r2_remote: Local-typed rclone remote; both artifacts land here.
    :param monkeypatch: Pytest fixture used to wrap ``synth_setter.pipeline.r2_io.upload``
        with an order-recording spy that still delegates to the real helper.
    :param stub_finalize_setup: Fixture-activation only — installs the
        ``ensure_r2_env_loaded`` / ``object_size`` stubs.
    """
    spec = build_wds_smoke_spec(task_name="finalize-from-spec-marker-last")
    seed_train_shards(fake_r2_remote, spec)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    real_upload = r2_io.upload
    upload_order: list[str] = []

    def spy_upload(src: str | Path, dst: str) -> None:
        upload_order.append(dst)
        real_upload(src, dst)

    monkeypatch.setattr("synth_setter.pipeline.r2_io.upload", spy_upload)

    finalize_dataset.finalize_from_spec(spec, work_dir)

    stats_uri = spec.r2.stats_uri()
    marker_uri = spec.r2.dataset_complete_marker_uri()
    assert uri_to_local_path(fake_r2_remote, stats_uri).is_file()
    assert uri_to_local_path(fake_r2_remote, marker_uri).is_file()
    assert upload_order.count(stats_uri) == 1
    assert upload_order.count(marker_uri) == 1
    assert upload_order.index(marker_uri) == len(upload_order) - 1
    assert upload_order.index(stats_uri) < upload_order.index(marker_uri)


def test_finalize_from_spec_non_canonical_prefix_warns_and_proceeds(
    tmp_path: Path,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_finalize_setup: Callable[[int | None], None],  # noqa: ARG001 — installs stubs only
) -> None:
    """A custom (non-canonical) ``r2.prefix`` is finalized, not rejected.

    Specs may set ``r2.prefix`` independently of ``task_name``/``run_id`` — e.g.
    the oracle-eval e2e isolates its objects under ``test-runs/<test>/<uuid>/``.
    finalize reads the same prefix generate wrote to, so the spec is
    self-consistent; a prefix that diverges from ``make_r2_prefix`` is advisory
    (logged), never fatal. Pins both halves: finalize emits the warning and
    still lands its artifacts at the custom prefix.

    :param tmp_path: Hosts the scratch ``work_dir``.
    :param fake_r2_remote: Local-typed rclone remote; shards + outputs land here.
    :param monkeypatch: Patches ``finalize_dataset.logger`` with a recording
        mock (loguru output does not reach pytest ``caplog``).
    :param stub_finalize_setup: Fixture-activation only — installs the
        ``ensure_r2_env_loaded`` / ``object_size`` stubs.
    """
    spec = build_wds_smoke_spec(task_name="finalize-custom-prefix")
    custom_r2 = spec.r2.model_copy(
        update={"prefix": "test-runs/finalize-custom-prefix/abc123def456/"}
    )
    custom_spec = spec.model_copy(update={"r2": custom_r2})
    seed_train_shards(fake_r2_remote, custom_spec)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    recording_logger = MagicMock(wraps=finalize_dataset.logger)
    monkeypatch.setattr(finalize_dataset, "logger", recording_logger)

    finalize_dataset.finalize_from_spec(custom_spec, work_dir)

    assert uri_to_local_path(fake_r2_remote, custom_spec.r2.stats_uri()).is_file()
    assert uri_to_local_path(
        fake_r2_remote, custom_spec.r2.dataset_complete_marker_uri()
    ).is_file()
    assert any(
        "non-canonical r2 prefix" in str(call.args[0])
        for call in recording_logger.warning.call_args_list
    ), recording_logger.warning.call_args_list


def test_hdf5_finalize_produces_train_consumable_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``finalize_hdf5`` downloads shards, reshards, computes stats, uploads every artifact.

    Pins the train-consumable layout: each ``{train,val,test}.h5`` carries
    ``audio`` / ``mel_spec`` / ``param_array``; ``stats.npz`` carries
    ``mean`` / ``std``. Reshard runs for real against the seeded shards —
    only R2 transport (``_rclone_copy`` / ``r2_io.upload``) and the heavy
    Dask-driven ``get_stats_hdf5`` are stubbed.

    :param tmp_path: Pytest tmp dir; hosts the fake R2 root + staged uploads.
    :param monkeypatch: Pytest fixture used to install download/upload/stats stubs.
    """
    spec = build_hdf5_smoke_spec()
    r2_stand_in = tmp_path / "r2"
    seed_shard_files(r2_stand_in, spec)

    uploads: dict[str, Path] = {}
    downloaded_uris: list[str] = []

    def fake_download(r2_uri: str, dst: Path) -> None:
        downloaded_uris.append(r2_uri)
        copy_shard_for_download(r2_stand_in, r2_uri, dst)

    monkeypatch.setattr("synth_setter.pipeline.r2_io.download_to_path", fake_download)
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.upload",
        lambda src, dst: shutil.copy(src, _stage_for(uploads, dst, tmp_path)),
    )
    stub_get_stats_hdf5(monkeypatch)

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    finalize_dataset.finalize_hdf5(spec, work_dir)

    # Every shard (not just train) is downloaded — reshard needs them all to
    # produce val/test splits; a regression that narrowed to train would
    # silently drop val/test outputs.
    assert downloaded_uris == [spec.r2.shard_uri(shard) for shard in spec.shards]
    for split in ("train", "val", "test"):
        with h5py.File(uploads[spec.r2.split_h5_uri(split)], "r") as f:
            assert {"audio", "mel_spec", "param_array"} <= set(f.keys())
    with np.load(uploads[spec.r2.stats_uri()]) as st:
        assert set(st.files) == {"mean", "std"}


def test_finalize_hdf5_real_shards_end_to_end(
    fake_r2_remote: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: real rclone, real reshard, real upload — read-back after work_dir deleted.

    Exercises ``finalize_hdf5`` end-to-end against the ``fake_r2_remote``
    local-typed rclone remote (no subprocess mocks): real shard downloads,
    real ``reshard_dataset`` invocation, real ``r2_io.upload`` of every
    split + ``stats.npz`` via ``rclone copyto``. Only ``get_stats_hdf5`` is
    stubbed because its Dask client startup dominates runtime; the stub
    writes a real ``stats.npz`` so the upload step is the production one.

    After the call returns the work_dir is wiped; the uploaded ``train.h5``
    is read back from the fake R2 location with sibling shards available
    (the layout a downstream consumer sees). A row from ``audio`` is
    dereferenced to prove the VDS sources resolve relative to the file's
    directory — guards the absolute-path regression where ``h5py.VirtualSource``
    would embed the now-gone work_dir.

    :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir
        (see ``tests/pipeline/conftest.py``). Skips if rclone is missing.
    :param tmp_path: Pytest tmp dir; hosts the finalize scratch work_dir.
    :param monkeypatch: Pytest fixture used to stub the slow Dask stats compute.
    """
    spec = build_hdf5_smoke_spec(task_name="finalize-hdf5-e2e")

    # Seed shards into the fake R2 location where ``download_to_path`` will fetch them.
    shard_remote_dir = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    seed_shard_files(shard_remote_dir, spec)
    stub_get_stats_hdf5(monkeypatch)

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    finalize_dataset.finalize_hdf5(spec, work_dir)

    # Wipe the scratch dir before any read so the assertion proves the
    # uploaded artifacts stand on their own (VDS relative-path invariant).
    shutil.rmtree(work_dir)

    # Layout the consumer sees: every split + stats land flat under ``<prefix>``,
    # sibling to the source shards.
    landed_root = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    train_h5 = landed_root / "train.h5"
    val_h5 = landed_root / "val.h5"
    test_h5 = landed_root / "test.h5"
    stats_npz = landed_root / "stats.npz"
    assert train_h5.is_file()
    assert val_h5.is_file()
    assert test_h5.is_file()
    assert stats_npz.is_file()

    # The VDS resolves and a row dereferences — proves the embedded source paths
    # are relative (basename) and find sibling shards in ``landed_root``.
    with h5py.File(train_h5, "r") as f:
        assert {"audio", "mel_spec", "param_array"} <= set(f.keys())
        audio = cast(h5py.Dataset, f["audio"])
        assert audio.shape[0] == spec.train_val_test_sizes[0]
        # Dereferencing a row routes through ``h5py.VirtualSource`` — would raise
        # ``KeyError`` / return zeros if the embedded path didn't resolve.
        _ = audio[0]

    with np.load(stats_npz) as st:
        assert set(st.files) == {"mean", "std"}


def test_finalize_hdf5_only_uploads_splits_that_reshard_wrote(
    fake_r2_remote: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec with empty val+test splits: only ``train.h5`` + ``stats.npz`` land in R2.

    Reshard prunes ``val.h5``/``test.h5`` when their shard ranges are
    empty (``[lo, lo)``). Finalize's ``if split_h5.exists()`` guard must
    keep the val/test uploads from firing; a regression that removed the
    guard would either crash with FileNotFoundError on the missing local
    file or silently upload a stale artifact from a previous run.

    :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
    :param tmp_path: Pytest tmp dir; hosts the finalize scratch work_dir.
    :param monkeypatch: Pytest fixture used to stub the slow Dask stats compute.
    """
    spec = build_hdf5_smoke_spec(task_name="train-only-splits", train_val_test_sizes=(8, 0, 0))
    shard_remote_dir = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    seed_shard_files(shard_remote_dir, spec)
    stub_get_stats_hdf5(monkeypatch)

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    finalize_dataset.finalize_hdf5(spec, work_dir)

    landed_root = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    assert (landed_root / "train.h5").is_file()
    assert (landed_root / "stats.npz").is_file()
    # No val/test artifacts were ever uploaded — reshard never wrote them
    # and finalize's existence guard skipped the upload call.
    assert not (landed_root / "val.h5").exists()
    assert not (landed_root / "test.h5").exists()


def test_finalize_hdf5_propagates_split_upload_failure_before_stats_upload(
    fake_r2_remote: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mid-loop split-upload failure: neither ``stats.npz`` nor ``dataset.complete`` lands.

    The "never leave a marker without artifacts" invariant from
    ``pipeline/CLAUDE.md`` must hold for every failure stage, not just
    the stats stage. Wraps ``r2_io.upload`` to raise on the first
    ``.h5`` split upload so reshard ran (its train.h5 exists locally)
    but transport failed mid-flight.

    :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
    :param tmp_path: Pytest tmp dir; hosts the finalize scratch work_dir.
    :param monkeypatch: Pytest fixture used to wrap ``r2_io.upload`` with
        a failing wrapper. Interaction-based by necessity — failure
        injection at the transport layer has no state-based alternative.
    """
    spec = build_hdf5_smoke_spec(task_name="split-upload-fails")
    shard_remote_dir = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    seed_shard_files(shard_remote_dir, spec)
    stub_get_stats_hdf5(monkeypatch)

    def fail_on_split_upload(source: str | Path, destination_uri: str) -> None:
        del source
        if destination_uri.endswith(".h5"):
            raise RuntimeError(f"simulated split upload failure for {destination_uri}")
        pytest.fail(f"upload to {destination_uri} should not be reached after split failure")

    monkeypatch.setattr("synth_setter.pipeline.r2_io.upload", fail_on_split_upload)

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    with pytest.raises(RuntimeError, match="simulated split upload failure"):
        finalize_dataset.finalize_hdf5(spec, work_dir)

    landed_root = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    assert not (landed_root / "stats.npz").exists()
    assert not (landed_root / "dataset.complete").exists()


def test_finalize_hdf5_writes_input_spec_json_sibling_to_shards(
    fake_r2_remote: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``input_spec.json`` lands in ``work_dir`` before ``reshard_dataset`` is invoked.

    Reshard's default spec-discovery looks for ``<dataset_root>/input_spec.json``
    when no ``--spec`` override is passed; finalize relies on this default
    path. Pinning the write order here gives a finalize-side test instead
    of a low-signal ``FileNotFoundError`` surfacing from reshard.

    :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
    :param tmp_path: Pytest tmp dir; hosts the finalize scratch work_dir.
    :param monkeypatch: Pytest fixture used to wrap ``reshard_dataset`` and
        capture the work_dir contents at invocation time.
    """
    spec = build_hdf5_smoke_spec(task_name="input-spec-sibling")
    shard_remote_dir = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    seed_shard_files(shard_remote_dir, spec)
    stub_get_stats_hdf5(monkeypatch)

    captured_files: list[str] = []
    real_reshard = finalize_dataset.reshard_dataset

    def capturing_reshard(dataset_root: Path, *args: object, **kwargs: object) -> None:
        captured_files.extend(sorted(p.name for p in Path(dataset_root).iterdir()))
        real_reshard(dataset_root, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("synth_setter.cli.finalize_dataset.reshard_dataset", capturing_reshard)

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    finalize_dataset.finalize_hdf5(spec, work_dir)

    # ``input_spec.json`` was sibling to every downloaded shard before
    # reshard ran — reshard's default spec lookup will succeed without
    # any ``--spec`` flag.
    assert "input_spec.json" in captured_files
    for shard in spec.shards:
        assert shard.filename in captured_files


def test_finalize_hdf5_rejects_structurally_invalid_shard(
    fake_r2_remote: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed downloaded shard surfaces an ``OSError`` from reshard, no upload runs.

    Pins ``pipeline/CLAUDE.md``'s delegation contract: finalize hands
    structural validation to reshard, which opens every shard via
    ``h5py.File`` while staging splits. A garbage-payload shard makes
    that open raise ``OSError("file signature not found")`` — finalize
    propagates the raise instead of writing partial artifacts to R2.
    The h5py error itself does not embed the offending shard's name;
    enriching that message lives in a follow-up.

    :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
    :param tmp_path: Pytest tmp dir; hosts the finalize scratch work_dir.
    :param monkeypatch: Pytest fixture used to stub the slow Dask stats compute
        (defensively, in case reshard is ever changed to fail later).
    """
    spec = build_hdf5_smoke_spec(task_name="invalid-shard-reject")
    shard_remote_dir = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    seed_shard_files(shard_remote_dir, spec)
    # Garbage bytes at the first shard URI — reshard's h5py.File open refuses to read it.
    corrupted = spec.shards[0].filename
    (shard_remote_dir / corrupted).write_bytes(b"not an HDF5 file\n")
    stub_get_stats_hdf5(monkeypatch)

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    with pytest.raises(OSError, match="file signature not found"):
        finalize_dataset.finalize_hdf5(spec, work_dir)

    # No split or stats artifact landed at the remote — the per-shard
    # open ran before any upload, so the failure is total.
    landed_root = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    assert not (landed_root / "train.h5").exists()
    assert not (landed_root / "stats.npz").exists()


def test_finalize_hdf5_raises_on_empty_train_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty train split surfaces as a clear ValueError before any download work.

    Reshard prunes ``train.h5`` when the train range is empty, after which
    ``get_stats_hdf5`` would crash with a low-signal HDF5 error; the guard
    converts that into a contract violation the operator can fix.

    :param monkeypatch: Pytest fixture used to install transport stubs.
    :param tmp_path: Pytest tmp dir used as the in-process scratch work_dir.
    """
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_to_path",
        lambda *a, **kw: pytest.fail("download_to_path should not be reached"),
    )
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.upload",
        lambda *a, **kw: pytest.fail("upload should not be reached"),
    )

    spec = build_hdf5_smoke_spec(task_name="empty-train-hdf5", train_val_test_sizes=(0, 4, 4))
    with pytest.raises(ValueError, match="train split is empty"):
        finalize_dataset.finalize_hdf5(spec, tmp_path)


def test_finalize_hdf5_propagates_stats_failure_before_marker_upload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ``get_stats_hdf5`` failure surfaces and no split/stats/marker upload runs.

    ``get_stats_hdf5`` raises on degenerate bins (see
    ``synth_setter.pipeline.data.stats._check_degenerate_bins``); finalize
    must propagate the error rather than swallow it and proceed to upload
    ``dataset.complete``. Drives ``finalize_hdf5`` directly so the assertion
    is local to the stats step.

    :param monkeypatch: Pytest fixture used to install transport + stats stubs.
    :param tmp_path: Pytest tmp dir; hosts the fake R2 root + scratch work_dir.
    """
    spec = build_hdf5_smoke_spec(task_name="stats-raises-hdf5")
    r2_stand_in = tmp_path / "r2"
    seed_shard_files(r2_stand_in, spec)

    uploaded: list[str] = []
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_to_path",
        lambda r2_uri, dst: copy_shard_for_download(r2_stand_in, r2_uri, dst),
    )
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.upload",
        lambda src, dst: uploaded.append(dst),
    )

    def boom(train_h5_path: str, mask_degenerate: bool = False) -> NoReturn:
        del train_h5_path, mask_degenerate
        raise RuntimeError("degenerate bins")

    monkeypatch.setattr("synth_setter.cli.finalize_dataset.get_stats_hdf5", boom)

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    with pytest.raises(RuntimeError, match="degenerate bins"):
        finalize_dataset.finalize_hdf5(spec, work_dir)

    assert uploaded == []


def test_finalize_wds_downloads_every_train_shard_uri(
    fake_r2_remote: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multi-shard train split: every train shard's canonical URI is downloaded, in order.

    :param fake_r2_remote: Local-typed rclone remote where each train shard
        is seeded before finalize runs.
    :param tmp_path: Pytest tmp dir; ``work_dir`` is a subdir so the spy can
        distinguish finalize's transient downloads from the seeded sources.
    :param monkeypatch: Used to install the URI-recording spy that delegates
        to the real ``download_to_path``.
    """
    spec = build_wds_smoke_spec(task_name="multi-shard-train", train_val_test_sizes=(8, 0, 0))
    seed_train_shards(fake_r2_remote, spec)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    real_download = r2_io.download_to_path
    downloaded_uris: list[str] = []

    def spy_download(r2_uri: str, dest_path: Path) -> None:
        downloaded_uris.append(r2_uri)
        real_download(r2_uri, dest_path)

    monkeypatch.setattr("synth_setter.pipeline.r2_io.download_to_path", spy_download)

    finalize_dataset.finalize_wds(spec, work_dir)

    train_lo, train_hi = spec.split_shard_ranges["train"]
    expected_uris = [spec.r2.shard_uri(shard) for shard in spec.shards[train_lo:train_hi]]
    assert downloaded_uris == expected_uris
    assert uri_to_local_path(fake_r2_remote, spec.r2.stats_uri()).is_file()


def test_finalize_lance_writes_split_files_stats_and_marker_last(
    fake_r2_remote: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_finalize_setup: Callable[[int | None], None],  # noqa: ARG001
) -> None:
    """``finalize_from_spec`` handles Lance splits and uploads the marker last.

    :param fake_r2_remote: Local-typed rclone remote where train shards are seeded.
    :param tmp_path: Pytest tmp dir; hosts finalize scratch files.
    :param monkeypatch: Pytest fixture used to spy on upload order.
    :param stub_finalize_setup: Fixture-activation only.
    """
    spec = build_lance_smoke_spec(
        task_name="finalize-lance-marker-last",
        train_val_test_sizes=(4, 4, 0),
    )
    seed_train_shards(fake_r2_remote, spec)
    for shard in spec.shards[1:2]:
        write_minimal_lance_shard(
            uri_to_local_path(fake_r2_remote, spec.r2.shard_uri(shard)), spec
        )
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    real_upload = r2_io.upload
    upload_order: list[str] = []

    def spy_upload(src: str | Path, dst: str) -> None:
        upload_order.append(dst)
        real_upload(src, dst)

    monkeypatch.setattr("synth_setter.pipeline.r2_io.upload", spy_upload)

    finalize_dataset.finalize_from_spec(spec, work_dir)

    assert uri_to_local_path(fake_r2_remote, spec.r2.split_lance_uri("train")).is_file()
    assert uri_to_local_path(fake_r2_remote, spec.r2.split_lance_uri("val")).is_file()
    assert not uri_to_local_path(fake_r2_remote, spec.r2.split_lance_uri("test")).exists()
    assert uri_to_local_path(fake_r2_remote, spec.r2.stats_uri()).is_file()
    assert uri_to_local_path(fake_r2_remote, spec.r2.dataset_complete_marker_uri()).is_file()
    assert upload_order[-1] == spec.r2.dataset_complete_marker_uri()
    assert upload_order.index(spec.r2.stats_uri()) < upload_order.index(
        spec.r2.dataset_complete_marker_uri()
    )


def test_finalize_wds_raises_on_empty_train_split(fake_r2_remote: Path, tmp_path: Path) -> None:
    """An empty train split surfaces as a clear ValueError, not a misleading FileNotFoundError.

    :param fake_r2_remote: Local-typed rclone remote — asserted untouched because the empty-train
        guard short-circuits before any I/O.
    :param tmp_path: Pytest tmp dir used as finalize's local work_dir.
    """
    spec = build_wds_smoke_spec(task_name="empty-train", train_val_test_sizes=(0, 4, 0))

    with pytest.raises(ValueError, match="train split is empty"):
        finalize_dataset.finalize_wds(spec, tmp_path)

    assert [p for p in fake_r2_remote.rglob("*") if p.is_file()] == []


@pytest.mark.parametrize("flag", [True, False])
def test_finalize_wds_forwards_mask_degenerate_bins_to_stream_stats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flag: bool
) -> None:
    """``finalize_wds`` forwards ``spec.mask_degenerate_bins`` to ``stream_stats_wds`` verbatim.

    Pins the wire on both polarities so a regression that hard-wires the kwarg (True or False)
    fails the test rather than silently re-breaking smoke finalize.

    :param monkeypatch: Pytest fixture used to capture the forwarded kwarg.
    :param tmp_path: Pytest tmp dir; the download stub writes one minimal shard.
    :param flag: Parametrized polarity threaded through the wire via the spec field.
    """
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_to_path",
        lambda r2_uri, dest_path: write_minimal_wds_shard(dest_path),
    )
    monkeypatch.setattr("synth_setter.pipeline.r2_io.upload", lambda *a, **kw: None)

    captured: dict[str, bool] = {}

    def fake_stream_stats(shard_paths: object, mask_degenerate: bool = False) -> tuple[Any, Any]:
        # Drain the generator — keeps download_to_path + unlink firing the same
        # way the production Welford fold would.
        list(shard_paths)  # type: ignore[arg-type]
        captured["mask_degenerate"] = mask_degenerate
        return np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)

    monkeypatch.setattr("synth_setter.cli.finalize_dataset.stream_stats_wds", fake_stream_stats)

    spec = build_wds_smoke_spec(
        task_name=f"mask-forwards-{flag}",
        train_val_test_sizes=(4, 0, 0),
        mask_degenerate_bins=flag,
    )
    finalize_dataset.finalize_wds(spec, tmp_path)

    assert captured == {"mask_degenerate": flag}


@pytest.mark.parametrize("flag", [True, False])
def test_finalize_hdf5_forwards_mask_degenerate_bins_to_get_stats(
    fake_r2_remote: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: bool
) -> None:
    """``finalize_hdf5`` forwards ``spec.mask_degenerate_bins`` to ``get_stats_hdf5`` verbatim.

    Mirrors the wds wire test so a regression on the hdf5 branch surfaces the same way; the smoke-
    shard config opts in for the same reason and needs the same protection.

    :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
    :param tmp_path: Pytest tmp dir; hosts the finalize scratch work_dir.
    :param monkeypatch: Pytest fixture used to capture the forwarded kwarg.
    :param flag: Parametrized polarity threaded through the wire via the spec field.
    """
    captured: dict[str, bool] = {}

    def fake_get_stats(train_h5_path: str, mask_degenerate: bool = False) -> None:
        captured["mask_degenerate"] = mask_degenerate
        np.savez(
            Path(train_h5_path).parent / "stats.npz",
            mean=np.zeros((2, 8, 8), dtype=np.float32),
            std=np.ones((2, 8, 8), dtype=np.float32),
        )

    monkeypatch.setattr("synth_setter.cli.finalize_dataset.get_stats_hdf5", fake_get_stats)

    spec = build_hdf5_smoke_spec(
        task_name=f"mask-forwards-hdf5-{flag}",
        mask_degenerate_bins=flag,
    )
    shard_remote_dir = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    seed_shard_files(shard_remote_dir, spec)

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    finalize_dataset.finalize_hdf5(spec, work_dir)

    assert captured == {"mask_degenerate": flag}


def test_finalize_wds_unlinks_each_shard_after_folding(
    fake_r2_remote: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Peak-disk invariant: at any moment at most one shard sits in ``work_dir``.

    Wraps ``r2_io.download_to_path`` with a spy that counts concurrent shard
    files in ``work_dir`` right after each download lands. The wrapper
    delegates to the real helper so the rclone download still executes
    against ``fake_r2_remote``.

    :param fake_r2_remote: Local-typed rclone remote, seeded with two shards.
    :param tmp_path: Pytest tmp dir; ``work_dir`` is a subdir so the spy can
        count only finalize's transient shards.
    :param monkeypatch: Used to install the recording wrapper.
    """
    spec = build_wds_smoke_spec(task_name="peak-disk", train_val_test_sizes=(8, 0, 0))
    seed_train_shards(fake_r2_remote, spec)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    real_download = r2_io.download_to_path
    concurrent_shards_seen: list[int] = []

    def spy_download(r2_uri: str, dest_path: Path) -> None:
        real_download(r2_uri, dest_path)
        concurrent_shards_seen.append(len(list(work_dir.glob("shard-*.tar"))))

    monkeypatch.setattr("synth_setter.pipeline.r2_io.download_to_path", spy_download)

    finalize_dataset.finalize_wds(spec, work_dir)

    assert concurrent_shards_seen == [1, 1]


def test_stubbed_stats_signatures_match_production() -> None:
    """Stub stats signatures in the finalize test lanes match the real stats functions.

    Several tests stub ``finalize_dataset.get_stats_hdf5`` and
    ``finalize_dataset.stream_stats_wds`` with hand-written signatures
    (``(train_h5_path, mask_degenerate=False)`` and
    ``(shard_paths, mask_degenerate=False)``). If production renamed a kwarg or
    changed a default, those stubs would keep passing while masking a real
    break. Compare the production parameter names and defaults via
    ``inspect.signature`` — a state-based contract check, since there is no
    runtime behavior to observe here.
    """
    real_get_stats = inspect.signature(real_get_stats_hdf5)
    assert list(real_get_stats.parameters) == ["filename", "mask_degenerate"]
    assert real_get_stats.parameters["mask_degenerate"].default is False

    real_stream = inspect.signature(real_stream_stats_wds)
    assert list(real_stream.parameters) == ["shard_paths", "mask_degenerate"]
    assert real_stream.parameters["mask_degenerate"].default is False
