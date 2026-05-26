"""Tests for ``synth_setter.cli.finalize_dataset`` — finalize entrypoint.

``run(cfg)`` tests seed train shards on the ``fake_r2_remote`` fixture
(a local-typed rclone remote rooted at ``tmp_path`` — see
``tests/pipeline/conftest.py``) and let ``finalize_dataset`` run the
real ``rclone copyto`` for download + upload against that remote. The
spec is written to disk as JSON and the run cfg carries a ``file://``
URI pointing at it, mirroring how production callers will pass the R2
URI of ``input_spec.json``.

Two helpers stay stubbed because the local rclone backend can't simulate
them cleanly:

- ``ensure_r2_env_loaded`` — would require real ``RCLONE_CONFIG_R2_*``
  secrets and a working ``rclone lsd r2:`` against real R2.
- ``object_size`` — ``rclone lsf`` against an absent key on the local
  backend exits 3 ("directory not found") instead of the empty-stdout
  semantics S3-compatible backends return; the marker probe in ``run()``
  needs the "absent → None" branch, so the stub stays.
"""

from __future__ import annotations

import io
import shutil
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from synth_setter.cli import finalize_dataset
from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    DATASET_FIELD_DTYPES,
    MEL_SPEC_FIELD,
    PARAM_ARRAY_FIELD,
    audio_dataset_shape,
    mel_dataset_shape,
    param_array_dataset_shape,
)
from synth_setter.pipeline.schemas.spec import DatasetSpec


def _write_minimal_wds_shard(dest: Path) -> None:
    """Write a tar at ``dest`` with one ``00000000.mel_spec.npy`` member.

    :param dest: Filesystem path where the tar is written. Parent dirs are created as needed.
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
    mask_degenerate_bins: bool = False,
) -> DatasetSpec:
    """Construct a wds ``DatasetSpec`` directly (no Hydra compose).

    :param task_name: Unique task name so each test gets a distinct r2.prefix.
    :param train_val_test_sizes: Three-tuple of sample counts; default is one
        4-sample shard.
    :param mask_degenerate_bins: Threaded onto the spec so wire tests can pin
        both polarities of the finalize stats-fold knob.
    :returns: A frozen wds ``DatasetSpec`` whose shards are deterministic.
    """
    kwargs: dict[str, Any] = {
        "task_name": task_name,
        "output_format": "wds",
        "train_val_test_sizes": list(train_val_test_sizes),
        "base_seed": 42,
        "mask_degenerate_bins": mask_degenerate_bins,
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


def _uri_to_local_path(fake_r2_remote: Path, r2_uri: str) -> Path:
    """Map an ``r2://bucket/key`` URI to its materialized path under the local-typed remote.

    The ``fake_r2_remote`` fixture sets ``RCLONE_CONFIG_R2_TYPE=local`` and
    chdirs into ``tmp_path``, so ``r2:bucket/key`` resolves to
    ``<tmp_path>/bucket/key`` — i.e. ``<fake_r2_remote>/bucket/key``.

    :param fake_r2_remote: Root of the local-typed remote (from the fixture).
    :param r2_uri: Canonical ``r2://bucket/key`` URI.
    :returns: The local filesystem path the URI materializes at.
    :raises ValueError: ``r2_uri`` does not start with the ``r2://`` scheme.
    """
    prefix = "r2://"
    if not r2_uri.startswith(prefix):
        raise ValueError(f"expected r2:// URI, got {r2_uri!r}")
    return fake_r2_remote / r2_uri[len(prefix) :]


def _seed_train_shards(fake_r2_remote: Path, spec: DatasetSpec) -> list[Path]:
    """Materialize each train shard under ``<fake_r2_remote>/<bucket>/<prefix>/<filename>``.

    Mirrors what ``generate_dataset.run`` would have uploaded earlier in the
    pipeline so finalize's ``r2_io.download_to_path`` finds a real tar under
    the local-typed remote.

    :param fake_r2_remote: Root of the local-typed remote.
    :param spec: Dataset spec whose ``shards`` and ``r2`` provide the layout.
    :returns: The seeded local shard paths, in train-range order.
    """
    train_lo, train_hi = spec.split_shard_ranges["train"]
    seeded: list[Path] = []
    for shard in spec.shards[train_lo:train_hi]:
        path = _uri_to_local_path(fake_r2_remote, spec.r2.shard_uri(shard))
        _write_minimal_wds_shard(path)
        seeded.append(path)
    return seeded


def _write_spec_to_file(spec: DatasetSpec, tmp_path: Path) -> str:
    """Persist ``spec`` as JSON under ``tmp_path`` and return its ``file://`` URI.

    Mirrors what generate-stage's ``upload_spec(spec)`` lands at the R2 input-spec URI
    so ``run()`` can re-hydrate the exact same ``DatasetSpec`` from local disk.

    :param spec: The frozen ``DatasetSpec`` to serialize.
    :param tmp_path: Test-scoped tmp dir; the JSON lands at ``<tmp_path>/spec/input_spec.json``.
    :returns: ``file://`` URI consumable by ``load_spec_from_uri``.
    """
    spec_dir = tmp_path / "spec"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_file = spec_dir / "input_spec.json"
    spec_file.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    return spec_file.as_uri()


def _build_run_cfg(spec_uri: str, output_dir: Path) -> DictConfig:
    """Synthesize a minimal ``run()`` cfg without invoking Hydra's @main decoration.

    ``run()`` only reads ``cfg.dataset_spec_uri`` and ``cfg.paths.output_dir``;
    a hand-built DictConfig is the lowest-friction equivalent of what
    @hydra.main would compose at the @main entry.

    :param spec_uri: URI passed through to ``load_spec_from_uri``.
    :param output_dir: Directory finalize uses as its scratch ``work_dir``.
        Must exist (``@hydra.main`` ordinarily creates it before ``main()`` runs).
    :returns: Mutable DictConfig with the two fields ``run()`` consumes.
    """
    return cast(
        DictConfig,
        OmegaConf.create({"dataset_spec_uri": spec_uri, "paths": {"output_dir": str(output_dir)}}),
    )


@pytest.fixture()
def stub_run_setup(monkeypatch: pytest.MonkeyPatch) -> Callable[[int | None], None]:
    """Stub ``ensure_r2_env_loaded`` + marker-probe; expose a marker-size setter.

    Leaves ``r2_io.download_to_path`` and ``r2_io.upload`` unstubbed — paired
    with ``fake_r2_remote`` they run real ``rclone copyto`` against the
    local-typed remote so tests can assert on materialized objects.

    :param monkeypatch: Pytest fixture used to install the stubs.
    :returns: A setter that overrides the marker-probe's "size in R2"
        response — ``None`` (default) makes ``run()`` proceed with finalize;
        an ``int`` triggers the idempotency short-circuit.
    """
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.ensure_r2_env_loaded",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda _uri: None)

    def _set_marker_size(size: int | None) -> None:
        monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda _uri: size)

    return _set_marker_size


def test_run_uploads_stats_then_marker_at_canonical_uris(
    tmp_path: Path,
    fake_r2_remote: Path,
    stub_run_setup: Callable[[int | None], None],  # noqa: ARG001 — installs stubs only
) -> None:
    """``run(cfg)`` against a wds spec lands ``stats.npz`` then ``dataset.complete``.

    Marker mtime is no earlier than stats mtime — the resumability contract
    pins marker-last.

    :param tmp_path: Pytest tmp dir; hosts the on-disk spec JSON + Hydra-style output_dir.
    :param fake_r2_remote: Local-typed rclone remote; both artifacts land here.
    :param stub_run_setup: Fixture-activation only — installs the
        ``ensure_r2_env_loaded`` / ``object_size`` stubs.
    """
    spec = _build_wds_smoke_spec(task_name="run-marker-last-wds")
    _seed_train_shards(fake_r2_remote, spec)
    output_dir = tmp_path / "hydra_output"
    output_dir.mkdir()
    cfg = _build_run_cfg(_write_spec_to_file(spec, tmp_path), output_dir)

    finalize_dataset.run(cfg)

    stats_path = _uri_to_local_path(fake_r2_remote, spec.r2.stats_uri())
    marker_path = _uri_to_local_path(fake_r2_remote, spec.r2.dataset_complete_marker_uri())
    assert stats_path.is_file()
    assert marker_path.is_file()
    assert stats_path.stat().st_mtime <= marker_path.stat().st_mtime


def test_run_is_idempotent_when_marker_already_exists(
    tmp_path: Path,
    fake_r2_remote: Path,
    stub_run_setup: Callable[[int | None], None],
) -> None:
    """Marker present at run prefix → ``run()`` short-circuits, no stats are written.

    :param tmp_path: Pytest tmp dir; hosts the on-disk spec JSON + Hydra-style output_dir.
    :param fake_r2_remote: Local-typed rclone remote — asserted to still be
        free of any ``stats.npz`` after the no-op run.
    :param stub_run_setup: Used to flip the marker probe to "present".
    """
    stub_run_setup(0)
    spec = _build_wds_smoke_spec(task_name="run-idempotent-wds")
    output_dir = tmp_path / "hydra_output"
    output_dir.mkdir()
    cfg = _build_run_cfg(_write_spec_to_file(spec, tmp_path), output_dir)

    finalize_dataset.run(cfg)

    stats_path = _uri_to_local_path(fake_r2_remote, spec.r2.stats_uri())
    assert not stats_path.exists()


def test_run_raises_on_unsupported_output_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_run_setup: Callable[[int | None], None],  # noqa: ARG001 — installs stubs only
) -> None:
    """An ``output_format`` outside {hdf5, wds} surfaces a clear ValueError.

    Pins the dispatcher's exhaustiveness contract — adding a third format
    without wiring its branch must trip this test rather than silently
    skip the artifact upload and write a misleading ``dataset.complete``.

    :param tmp_path: Pytest tmp dir; hosts the on-disk spec JSON.
    :param monkeypatch: Pytest fixture used to mutate the loaded spec's format.
    :param stub_run_setup: Installs the auth + marker-probe stubs so the
        dispatcher (not the marker check) is the failure surface.
    """

    def loader(_uri: str) -> DatasetSpec:
        spec = _build_wds_smoke_spec(task_name="run-bad-format")
        object.__setattr__(spec, "output_format", "parquet")
        return spec

    monkeypatch.setattr("synth_setter.cli.finalize_dataset.load_spec_from_uri", loader)
    output_dir = tmp_path / "hydra_output"
    output_dir.mkdir()
    cfg = _build_run_cfg("file:///unused", output_dir)

    with pytest.raises(ValueError, match="unsupported output_format"):
        finalize_dataset.run(cfg)


def test_finalize_dataset_main_resolves_hydra_logging_under_at_hydra_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_run_setup: Callable[[int | None], None],
) -> None:
    """Invoking ``main()`` under @hydra.main resolves every interpolation in the shared groups.

    The shared ``hydra/default.yaml`` interpolates ``${task_name}`` into both
    ``run.dir`` and ``job_logging.handlers.file.filename``. A missing override
    surfaces as a Hydra startup ``InterpolationKeyError`` *before* ``run()``
    fires — a structure-only compose check (``return_hydra_config=True``)
    inspects unresolved templates and misses this. Drive the decorated
    ``main()`` for real with the marker-probe stub set to "present" so the
    body short-circuits at the idempotency check, isolating the test to
    Hydra-side resolution.

    :param tmp_path: Hosts ``PROJECT_ROOT``, the on-disk spec JSON, and Hydra's run dir.
    :param monkeypatch: Pytest fixture used to point ``PROJECT_ROOT`` + ``sys.argv``.
    :param stub_run_setup: Used to flip the marker probe to "present" so the
        body skips the wds/hdf5 dispatch.
    """
    stub_run_setup(0)
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    spec = _build_wds_smoke_spec(task_name="hydra-startup")
    spec_uri = _write_spec_to_file(spec, tmp_path)
    monkeypatch.setattr("sys.argv", ["finalize_dataset", f"dataset_spec_uri={spec_uri}"])

    finalize_dataset.main()

    # Hydra's run dir for this invocation lands under PROJECT_ROOT/logs/finalize_dataset/.
    # Existence proves @hydra.main resolved ${paths.log_dir}, ${now:…}, and
    # the ${task_name} interpolations the shared hydra group references.
    assert (tmp_path / "logs" / "finalize_dataset").is_dir()


def _build_hdf5_smoke_spec(
    task_name: str = "finalize-hdf5-unit",
    train_val_test_sizes: tuple[int, int, int] = (8, 4, 4),
    samples_per_shard: int = 4,
    mask_degenerate_bins: bool = False,
) -> DatasetSpec:
    """Construct a small hdf5 ``DatasetSpec`` directly (no Hydra compose).

    :param task_name: Unique task name so each test gets a distinct r2.prefix.
    :param train_val_test_sizes: Three-tuple of sample counts; every entry must
        be a multiple of ``samples_per_shard``.
    :param samples_per_shard: Per-shard row count driving shard count derivation.
    :param mask_degenerate_bins: Threaded onto the spec so wire tests can pin
        both polarities of the finalize stats-fold knob.
    :returns: A frozen hdf5 ``DatasetSpec`` whose shards are deterministic.
    """
    kwargs: dict[str, Any] = {
        "task_name": task_name,
        "output_format": "hdf5",
        "train_val_test_sizes": list(train_val_test_sizes),
        "base_seed": 42,
        "mask_degenerate_bins": mask_degenerate_bins,
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
            f.create_dataset(
                AUDIO_FIELD, shape=audio_shape, dtype=DATASET_FIELD_DTYPES[AUDIO_FIELD]
            )
            f.create_dataset(
                MEL_SPEC_FIELD, shape=mel_shape, dtype=DATASET_FIELD_DTYPES[MEL_SPEC_FIELD]
            )
            f.create_dataset(
                PARAM_ARRAY_FIELD, shape=param_shape, dtype=DATASET_FIELD_DTYPES[PARAM_ARRAY_FIELD]
            )


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
        lambda train_h5_path, mask_degenerate=False: np.savez(
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
        lambda train_h5_path, mask_degenerate=False: np.savez(
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


def _stub_get_stats_hdf5(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``finalize_dataset.get_stats_hdf5`` to write a sentinel ``stats.npz``.

    Real Dask startup dominates runtime; tests that drive ``finalize_hdf5``
    end-to-end use this stub so the subsequent ``r2_io.upload`` step still
    runs against a real on-disk file produced sibling to ``train.h5``.

    :param monkeypatch: Pytest fixture used to install the stub.
    """
    monkeypatch.setattr(
        "synth_setter.cli.finalize_dataset.get_stats_hdf5",
        lambda train_h5_path, mask_degenerate=False: np.savez(
            Path(train_h5_path).parent / "stats.npz",
            mean=np.zeros((2, 8, 8), dtype=np.float32),
            std=np.ones((2, 8, 8), dtype=np.float32),
        ),
    )


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
    spec = _build_hdf5_smoke_spec(task_name="train-only-splits", train_val_test_sizes=(8, 0, 0))
    shard_remote_dir = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    _seed_shard_files(shard_remote_dir, spec)
    _stub_get_stats_hdf5(monkeypatch)

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
    spec = _build_hdf5_smoke_spec(task_name="split-upload-fails")
    shard_remote_dir = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    _seed_shard_files(shard_remote_dir, spec)
    _stub_get_stats_hdf5(monkeypatch)

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
    spec = _build_hdf5_smoke_spec(task_name="input-spec-sibling")
    shard_remote_dir = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    _seed_shard_files(shard_remote_dir, spec)
    _stub_get_stats_hdf5(monkeypatch)

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
    spec = _build_hdf5_smoke_spec(task_name="invalid-shard-reject")
    shard_remote_dir = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    _seed_shard_files(shard_remote_dir, spec)
    # Garbage bytes at the first shard URI — reshard's h5py.File open refuses to read it.
    corrupted = spec.shards[0].filename
    (shard_remote_dir / corrupted).write_bytes(b"not an HDF5 file\n")
    _stub_get_stats_hdf5(monkeypatch)

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    with pytest.raises(OSError, match="file signature not found"):
        finalize_dataset.finalize_hdf5(spec, work_dir)

    # No split or stats artifact landed at the remote — the per-shard
    # open ran before any upload, so the failure is total.
    landed_root = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    assert not (landed_root / "train.h5").exists()
    assert not (landed_root / "stats.npz").exists()


def test_run_hdf5_marker_idempotency_short_circuits_before_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hdf5 dispatch: ``run()`` returns without any download when the marker exists.

    ``test_run_is_idempotent_when_marker_already_exists`` covers the wds
    branch; this test pins the hdf5-branch path so a regression that
    moved the marker check *inside* the format branch (after the dispatch
    table) would be caught.

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
    monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda _uri: 0)

    spec = _build_hdf5_smoke_spec(task_name="run-hdf5-marker-present")
    output_dir = tmp_path / "hydra_output"
    output_dir.mkdir()
    cfg = _build_run_cfg(_write_spec_to_file(spec, tmp_path), output_dir)

    finalize_dataset.run(cfg)


def test_run_hdf5_branch_uploads_marker_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hdf5 ``run(cfg)`` path writes ``dataset.complete`` strictly after every artifact.

    Pins the ``pipeline/CLAUDE.md`` ordering invariant for hdf5: an
    interrupted run must never leave a marker without the artifacts it
    advertises.

    :param tmp_path: Pytest tmp dir; hosts the fake R2 root + on-disk spec + output_dir.
    :param monkeypatch: Pytest fixture used to patch the full transport surface.
    """
    r2_stand_in = tmp_path / "r2"
    spec = _build_hdf5_smoke_spec(task_name="run-hdf5-marker-last")
    _seed_shard_files(r2_stand_in, spec)
    upload_order: list[str] = []

    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_to_path",
        lambda r2_uri, dst: shutil.copy(r2_stand_in / Path(r2_uri).name, Path(dst)),
    )

    def record_upload(src: str | Path, dst: str) -> None:
        del src
        upload_order.append(dst)

    monkeypatch.setattr("synth_setter.pipeline.r2_io.upload", record_upload)
    monkeypatch.setattr("synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda *a, **k: None)
    monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda _uri: None)
    # Skip the Dask-driven mean/std compute — orchestration is what this test pins.
    monkeypatch.setattr(
        "synth_setter.cli.finalize_dataset.get_stats_hdf5",
        lambda train_h5_path, mask_degenerate=False: np.savez(
            Path(train_h5_path).parent / "stats.npz",
            mean=np.zeros((2, 8, 8), dtype=np.float32),
            std=np.ones((2, 8, 8), dtype=np.float32),
        ),
    )

    output_dir = tmp_path / "hydra_output"
    output_dir.mkdir()
    cfg = _build_run_cfg(_write_spec_to_file(spec, tmp_path), output_dir)

    finalize_dataset.run(cfg)

    marker_uri = spec.r2.dataset_complete_marker_uri()
    marker_index = upload_order.index(marker_uri)
    # Marker strictly later than every artifact URI — the
    # ``pipeline/CLAUDE.md`` invariant ("never leave a marker without
    # artifacts") generalizes to splits-with-val-test, not just train.
    assert marker_index == len(upload_order) - 1
    for artifact_uri in (spec.r2.stats_uri(), spec.r2.split_h5_uri("train")):
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

    def boom(train_h5_path: str, mask_degenerate: bool = False) -> None:
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
    from synth_setter.pipeline import r2_io

    spec = _build_wds_smoke_spec(task_name="multi-shard-train", train_val_test_sizes=(8, 0, 0))
    _seed_train_shards(fake_r2_remote, spec)
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
    assert _uri_to_local_path(fake_r2_remote, spec.r2.stats_uri()).is_file()


def test_finalize_wds_raises_on_empty_train_split(fake_r2_remote: Path, tmp_path: Path) -> None:
    """An empty train split surfaces as a clear ValueError, not a misleading FileNotFoundError.

    :param fake_r2_remote: Local-typed rclone remote — asserted untouched because the empty-train
        guard short-circuits before any I/O.
    :param tmp_path: Pytest tmp dir used as finalize's local work_dir.
    """
    spec = _build_wds_smoke_spec(task_name="empty-train", train_val_test_sizes=(0, 4, 0))

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
        lambda r2_uri, dest_path: _write_minimal_wds_shard(dest_path),
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

    spec = _build_wds_smoke_spec(
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

    spec = _build_hdf5_smoke_spec(
        task_name=f"mask-forwards-hdf5-{flag}",
        mask_degenerate_bins=flag,
    )
    shard_remote_dir = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    _seed_shard_files(shard_remote_dir, spec)

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
    from synth_setter.pipeline import r2_io

    spec = _build_wds_smoke_spec(task_name="peak-disk", train_val_test_sizes=(8, 0, 0))
    _seed_train_shards(fake_r2_remote, spec)
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
