"""Tests for ``synth_setter.cli.finalize_dataset`` — finalize entrypoint.

End-to-end test invokes ``main()`` against the real ``smoke-shard-wds``
experiment so the Hydra compose + ``DatasetSpec`` construction stay
exercised; rclone (download / upload / auth ping) is stubbed.
"""

from __future__ import annotations

import io
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pytest

from synth_setter.cli import finalize_dataset
from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    MEL_SPEC_FIELD,
    PARAM_ARRAY_FIELD,
    audio_dataset_shape,
    mel_dataset_shape,
    param_array_dataset_shape,
)
from synth_setter.pipeline.schemas.spec import DatasetSpec

_FIXED_NOW = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)


def _write_minimal_wds_shard(dest: Path) -> None:
    """Write a tar at ``dest`` with one ``00000000.mel_spec.npy`` member.

    :param dest: Filesystem path where the tar is written.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    # 4 rows so Welford variance is non-degenerate.
    payload = np.arange(8, dtype=np.float32).reshape(4, 2)
    buf = io.BytesIO()
    np.save(buf, payload)
    member_bytes = buf.getvalue()
    with tarfile.open(dest, mode="w") as tar:
        info = tarfile.TarInfo(name="00000000.mel_spec.npy")
        info.size = len(member_bytes)
        tar.addfile(info, io.BytesIO(member_bytes))


def _build_wds_smoke_spec(
    task_name: str = "finalize-wds-unit",
    train_val_test_sizes: tuple[int, int, int] = (4, 0, 0),
) -> DatasetSpec:
    """Construct a wds ``DatasetSpec`` directly (no Hydra compose).

    :param task_name: Unique task name so each test gets a distinct r2.prefix.
    :param train_val_test_sizes: Three-tuple of sample counts; default is one
        4-sample shard.
    :returns: A frozen wds ``DatasetSpec`` whose shards are deterministic.
    """
    kwargs: dict[str, Any] = {
        "task_name": task_name,
        "output_format": "wds",
        "train_val_test_sizes": list(train_val_test_sizes),
        "base_seed": 42,
        "r2": {"bucket": "intermediate-data"},
        "render": {
            "plugin_path": "/fake/Plugin.vst3",
            "preset_path": "presets/surge-base.vstpreset",
            "param_spec_name": "surge_simple",
            "renderer_version": "1.0.0-test",
            "sample_rate": 16000,
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "samples_per_render_batch": 4,
            "samples_per_shard": 4,
            "gui_toggle_cadence": "never",
        },
    }
    return DatasetSpec(**kwargs)  # type: ignore[arg-type]


@pytest.fixture()
def patch_finalize_io(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], list[tuple[Path, str]]]:
    """Patch ``r2_io`` download/upload/auth so finalize never touches real R2.

    :param monkeypatch: Pytest fixture used to install the stubs.
    :returns:``(downloaded_uris, uploaded)`` — the two recording lists. Tests read these to assert
        behaviour.
    """
    downloaded_uris: list[str] = []
    uploaded: list[tuple[Path, str]] = []

    def fake_download(r2_uri: str, dest_path: Path) -> None:
        downloaded_uris.append(r2_uri)
        _write_minimal_wds_shard(dest_path)

    def record_upload(source: str | Path, destination_uri: str) -> None:
        uploaded.append((Path(source), destination_uri))

    monkeypatch.setattr("synth_setter.pipeline.r2_io.download_to_path", fake_download)
    monkeypatch.setattr("synth_setter.pipeline.r2_io.upload", record_upload)
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.ensure_r2_env_loaded",
        lambda *args, **kwargs: None,
    )
    # main() probes the marker URI for the idempotency gate; default is "absent".
    monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda uri: None)
    # Pin runtime fields so DatasetSpec.run_id is reproducible across the
    # main()-side compose and the assertion-side compose.
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._get_git_sha", lambda: "a" * 40)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._is_repo_dirty", lambda: False)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._utc_now", lambda: _FIXED_NOW)
    return downloaded_uris, uploaded


def test_main_uploads_stats_then_marker_at_canonical_uris(
    monkeypatch: pytest.MonkeyPatch,
    patch_finalize_io: tuple[list[str], list[tuple[Path, str]]],
) -> None:
    """``main()`` against smoke-shard-wds uploads exactly ``stats.npz`` then ``dataset.complete``.

    :param monkeypatch: Pytest fixture used to patch ``sys.argv``.
    :param patch_finalize_io: Recording stubs for download / upload / auth.
    """
    _downloaded, uploaded = patch_finalize_io
    monkeypatch.setattr(sys, "argv", ["finalize", "experiment=generate_dataset/smoke-shard-wds"])

    finalize_dataset.main()

    # Reconstruct the spec the same way main() does so we can assert exact URIs.
    spec = _compose_smoke_wds_spec()
    upload_destinations = [dst for _, dst in uploaded]
    assert upload_destinations == [
        spec.r2.stats_uri(),
        spec.r2.dataset_complete_marker_uri(),
    ]
    assert upload_destinations[-1] == spec.r2.dataset_complete_marker_uri()


def test_main_is_idempotent_when_marker_already_exists(
    monkeypatch: pytest.MonkeyPatch,
    patch_finalize_io: tuple[list[str], list[tuple[Path, str]]],
) -> None:
    """If ``dataset.complete`` already exists at the run prefix, ``main()`` does no work.

    :param monkeypatch: Pytest fixture used to patch ``sys.argv`` and ``object_size``.
    :param patch_finalize_io: Recording stubs; we override ``object_size`` after
        the fixture installs the default-absent stub.
    """
    downloaded, uploaded = patch_finalize_io
    monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda uri: 0)
    monkeypatch.setattr(sys, "argv", ["finalize", "experiment=generate_dataset/smoke-shard-wds"])

    finalize_dataset.main()

    assert downloaded == []
    assert uploaded == []


def _build_hdf5_smoke_spec(
    task_name: str = "finalize-hdf5-unit",
    train_val_test_sizes: tuple[int, int, int] = (8, 4, 4),
    samples_per_shard: int = 4,
) -> DatasetSpec:
    """Construct a small hdf5 ``DatasetSpec`` directly (no Hydra compose).

    :param task_name: Unique task name so each test gets a distinct r2.prefix.
    :param train_val_test_sizes: Three-tuple of sample counts; every entry must
        be a multiple of ``samples_per_shard``.
    :param samples_per_shard: Per-shard row count driving shard count derivation.
    :returns: A frozen hdf5 ``DatasetSpec`` whose shards are deterministic.
    """
    kwargs: dict[str, Any] = {
        "task_name": task_name,
        "output_format": "hdf5",
        "train_val_test_sizes": list(train_val_test_sizes),
        "base_seed": 42,
        "r2": {"bucket": "intermediate-data"},
        "render": {
            "plugin_path": "/fake/Plugin.vst3",
            "preset_path": "presets/surge-base.vstpreset",
            "param_spec_name": "surge_simple",
            "renderer_version": "1.0.0-test",
            "sample_rate": 16000,
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "samples_per_render_batch": samples_per_shard,
            "samples_per_shard": samples_per_shard,
            "gui_toggle_cadence": "never",
        },
    }
    return DatasetSpec(**kwargs)  # type: ignore[arg-type]


def _seed_shard_files(remote_root: Path, spec: DatasetSpec) -> None:
    """Write every ``spec.shards[i].filename`` as a structurally valid HDF5 shard.

    Shapes/dtypes match
    :func:`synth_setter.pipeline.ci.validate_shard.check_shard_contracts`.

    :param remote_root: Directory acting as the R2-side staging area.
    :param spec: Spec whose ``shards`` define the filenames to seed.
    """
    remote_root.mkdir(parents=True, exist_ok=True)
    render = spec.render
    audio_shape = audio_dataset_shape(
        render.samples_per_shard,
        render.channels,
        render.sample_rate,
        render.signal_duration_seconds,
    )
    mel_shape = mel_dataset_shape(
        render.samples_per_shard,
        render.channels,
        render.sample_rate,
        render.signal_duration_seconds,
    )
    param_shape = param_array_dataset_shape(render.samples_per_shard, spec.num_params)
    for shard in spec.shards:
        with h5py.File(remote_root / shard.filename, "w") as f:
            f.create_dataset(AUDIO_FIELD, shape=audio_shape, dtype=np.float32)
            f.create_dataset(MEL_SPEC_FIELD, shape=mel_shape, dtype=np.float32)
            f.create_dataset(PARAM_ARRAY_FIELD, shape=param_shape, dtype=np.float32)


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
    spec = _build_hdf5_smoke_spec()
    r2_stand_in = tmp_path / "r2"
    _seed_shard_files(r2_stand_in, spec)

    uploads: dict[str, Path] = {}
    downloaded_uris: list[str] = []

    def fake_download(r2_uri: str, dst: Path) -> None:
        # ``download_to_path`` is file→file (``rclone copyto``): ``dst`` is the
        # exact local path, not a directory. ``r2_uri`` carries the basename.
        downloaded_uris.append(r2_uri)
        shutil.copy(r2_stand_in / Path(r2_uri).name, Path(dst))

    monkeypatch.setattr("synth_setter.pipeline.r2_io.download_to_path", fake_download)
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.upload",
        lambda src, dst: shutil.copy(src, _stage_for(uploads, dst, tmp_path)),
    )
    # Skip the Dask-driven mean/std compute — orchestration is what this test pins.
    monkeypatch.setattr(
        "synth_setter.cli.finalize_dataset.get_stats_hdf5",
        lambda train_h5_path: np.savez(
            Path(train_h5_path).parent / "stats.npz",
            mean=np.zeros((2, 8, 8), dtype=np.float32),
            std=np.ones((2, 8, 8), dtype=np.float32),
        ),
    )

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
    spec = _build_hdf5_smoke_spec(task_name="finalize-hdf5-e2e")

    # Seed shards into the fake R2 location where ``download_to_path`` will fetch them.
    shard_remote_dir = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    _seed_shard_files(shard_remote_dir, spec)

    # Real Dask compute would dominate runtime; stub writes a sentinel ``stats.npz``
    # so the production ``r2_io.upload`` step still runs against a real file.
    monkeypatch.setattr(
        "synth_setter.cli.finalize_dataset.get_stats_hdf5",
        lambda train_h5_path: np.savez(
            Path(train_h5_path).parent / "stats.npz",
            mean=np.zeros((2, 8, 8), dtype=np.float32),
            std=np.ones((2, 8, 8), dtype=np.float32),
        ),
    )

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


def test_main_hdf5_branch_uploads_marker_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hdf5 ``main()`` path writes ``dataset.complete`` strictly after every artifact.

    Pins the ``pipeline/CLAUDE.md`` ordering invariant for hdf5: an
    interrupted run must never leave a marker without the artifacts it
    advertises.

    :param tmp_path: Pytest tmp dir; hosts the fake R2 root + staged uploads.
    :param monkeypatch: Pytest fixture used to patch the full transport surface.
    """
    r2_stand_in = tmp_path / "r2"
    upload_order: list[str] = []
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_to_path",
        lambda r2_uri, dst: shutil.copy(r2_stand_in / Path(r2_uri).name, Path(dst)),
    )

    def record_upload(src: str | Path, dst: str) -> None:
        upload_order.append(dst)

    monkeypatch.setattr("synth_setter.pipeline.r2_io.upload", record_upload)
    monkeypatch.setattr("synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda *a, **k: None)
    monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda uri: None)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._get_git_sha", lambda: "a" * 40)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._is_repo_dirty", lambda: False)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._utc_now", lambda: _FIXED_NOW)
    # Skip the Dask-driven mean/std compute — orchestration is what this test pins.
    monkeypatch.setattr(
        "synth_setter.cli.finalize_dataset.get_stats_hdf5",
        lambda train_h5_path: np.savez(
            Path(train_h5_path).parent / "stats.npz",
            mean=np.zeros((2, 8, 8), dtype=np.float32),
            std=np.ones((2, 8, 8), dtype=np.float32),
        ),
    )
    # Compose the smoke-shard hdf5 experiment so the entrypoint exercises real Hydra.
    monkeypatch.setattr(sys, "argv", ["finalize", "experiment=generate_dataset/smoke-shard"])

    # Re-compose the same spec ``main()`` will, then seed shards into the fake
    # R2 root so the patched ``download_to_path`` lands them under ``work_dir``.
    main_spec = _compose_smoke_hdf5_spec()
    _seed_shard_files(r2_stand_in, main_spec)

    finalize_dataset.main()

    marker_uri = main_spec.r2.dataset_complete_marker_uri()
    marker_index = upload_order.index(marker_uri)
    # Marker strictly later than every artifact URI — the
    # ``pipeline/CLAUDE.md`` invariant ("never leave a marker without
    # artifacts") generalizes to splits-with-val-test, not just train.
    assert marker_index == len(upload_order) - 1
    for artifact_uri in (main_spec.r2.stats_uri(), main_spec.r2.split_h5_uri("train")):
        assert upload_order.index(artifact_uri) < marker_index


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

    spec = _build_hdf5_smoke_spec(task_name="empty-train-hdf5", train_val_test_sizes=(0, 4, 4))
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
    spec = _build_hdf5_smoke_spec(task_name="stats-raises-hdf5")
    r2_stand_in = tmp_path / "r2"
    _seed_shard_files(r2_stand_in, spec)

    uploaded: list[str] = []
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_to_path",
        lambda r2_uri, dst: shutil.copy(r2_stand_in / Path(r2_uri).name, Path(dst)),
    )
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.upload",
        lambda src, dst: uploaded.append(dst),
    )

    def boom(train_h5_path: str) -> None:
        del train_h5_path
        raise RuntimeError("degenerate bins")

    monkeypatch.setattr("synth_setter.cli.finalize_dataset.get_stats_hdf5", boom)

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    with pytest.raises(RuntimeError, match="degenerate bins"):
        finalize_dataset.finalize_hdf5(spec, work_dir)

    assert uploaded == []


def _compose_smoke_hdf5_spec() -> DatasetSpec:
    """Re-compose the hdf5 smoke spec the same way ``main()`` does, for URI assertions.

    :returns: A ``DatasetSpec`` whose ``r2.prefix`` matches what ``main()``
        constructs from the ``smoke-shard`` experiment in the same process.
    """
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(finalize_dataset._CONFIG_DIR),  # noqa: SLF001 — test mirrors main()
    ):
        cfg = compose(
            config_name="dataset",
            overrides=["experiment=generate_dataset/smoke-shard"],
        )
    cfg.paths.root_dir = str(finalize_dataset._REPO_ROOT)  # noqa: SLF001
    cfg.paths.output_dir = str(finalize_dataset._REPO_ROOT)  # noqa: SLF001
    cfg.paths.work_dir = str(finalize_dataset._REPO_ROOT)  # noqa: SLF001
    return finalize_dataset.spec_from_cfg(cfg)


def test_finalize_wds_downloads_every_train_shard_uri(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Multi-shard train split: every ``spec.shards[train_lo:train_hi]`` URI is downloaded.

    :param monkeypatch: Pytest fixture used to install download/upload stubs.
    :param tmp_path: Pytest tmp dir used as the in-process scratch work_dir.
    """
    downloaded_uris: list[str] = []
    uploaded: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_to_path",
        lambda uri, dest: (downloaded_uris.append(uri), _write_minimal_wds_shard(dest)),
    )
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.upload",
        lambda source, destination_uri: uploaded.append((Path(source), destination_uri)),
    )

    spec = _build_wds_smoke_spec(task_name="multi-shard-train", train_val_test_sizes=(8, 0, 0))
    finalize_dataset.finalize_wds(spec, tmp_path)

    expected_uris = [spec.r2.shard_uri(shard) for shard in spec.shards]
    assert downloaded_uris == expected_uris
    assert len(uploaded) == 1
    assert uploaded[0][1] == spec.r2.stats_uri()


def test_finalize_wds_raises_on_empty_train_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty train split surfaces as a clear ValueError, not a misleading FileNotFoundError.

    :param monkeypatch: Pytest fixture used to install download/upload stubs.
    :param tmp_path: Pytest tmp dir used as the in-process scratch work_dir.
    """
    # download/upload must never be called — the empty-train guard short-circuits.
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_to_path",
        lambda *a, **kw: pytest.fail("download_to_path should not be reached"),
    )
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.upload",
        lambda *a, **kw: pytest.fail("upload should not be reached"),
    )

    spec = _build_wds_smoke_spec(task_name="empty-train", train_val_test_sizes=(0, 4, 0))
    with pytest.raises(ValueError, match="train split is empty"):
        finalize_dataset.finalize_wds(spec, tmp_path)


def test_finalize_wds_unlinks_each_shard_after_folding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Peak-disk invariant: at any moment at most one shard sits in ``work_dir``.

    :param monkeypatch: Pytest fixture used to install the download stub.
    :param tmp_path: Pytest tmp dir used as the in-process scratch work_dir.
    """
    concurrent_shards_seen: list[int] = []

    def fake_download(r2_uri: str, dest_path: Path) -> None:
        del r2_uri
        _write_minimal_wds_shard(dest_path)
        concurrent_shards_seen.append(len(list(tmp_path.glob("shard-*.tar"))))

    monkeypatch.setattr("synth_setter.pipeline.r2_io.download_to_path", fake_download)
    monkeypatch.setattr("synth_setter.pipeline.r2_io.upload", lambda src, dst: None)

    spec = _build_wds_smoke_spec(task_name="peak-disk", train_val_test_sizes=(8, 0, 0))
    finalize_dataset.finalize_wds(spec, tmp_path)

    assert concurrent_shards_seen == [1, 1]


def _compose_smoke_wds_spec() -> DatasetSpec:
    """Re-compose the wds smoke spec the same way ``main()`` does, for URI assertions.

    :returns: A ``DatasetSpec`` whose ``r2.prefix`` matches what ``main()``
        constructs from the ``smoke-shard-wds`` experiment in the same process.
    """
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(finalize_dataset._CONFIG_DIR),  # noqa: SLF001 — test mirrors main()
    ):
        cfg = compose(
            config_name="dataset",
            overrides=["experiment=generate_dataset/smoke-shard-wds"],
        )
    cfg.paths.root_dir = str(finalize_dataset._REPO_ROOT)  # noqa: SLF001
    cfg.paths.output_dir = str(finalize_dataset._REPO_ROOT)  # noqa: SLF001
    cfg.paths.work_dir = str(finalize_dataset._REPO_ROOT)  # noqa: SLF001
    return finalize_dataset.spec_from_cfg(cfg)
