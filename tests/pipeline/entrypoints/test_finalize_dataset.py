"""Tests for ``synth_setter.cli.finalize_dataset`` — finalize entrypoint.

``finalize(cfg)`` tests seed train shards on the ``fake_r2_remote``
fixture (a local-typed rclone remote rooted at ``tmp_path`` — see
``tests/pipeline/conftest.py``) and let ``finalize_dataset`` run the
real ``rclone copyto`` for download + upload against that remote. The
spec is written to disk as JSON and the cfg carries a ``file://`` URI
pointing at it, mirroring how production callers pass the R2 URI of
``input_spec.json``.

Two helpers stay stubbed because the local rclone backend can't simulate
them cleanly:

- ``ensure_r2_env_loaded`` — would require real ``RCLONE_CONFIG_R2_*``
  secrets and a working ``rclone lsd r2:`` against real R2.
- ``object_size`` — ``rclone lsf`` against an absent key on the local
  backend exits 3 ("directory not found") instead of the empty-stdout
  semantics S3-compatible backends return; the marker probe in
  ``finalize()`` needs the "absent → None" branch, so the stub stays.
"""

from __future__ import annotations

import glob
import inspect
import io
import os
import shutil
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, cast
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
import wandb
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
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.stats import get_stats_hdf5 as real_get_stats_hdf5
from synth_setter.pipeline.data.stats import stream_stats_wds as real_stream_stats_wds
from synth_setter.pipeline.schemas.spec import DatasetSpec
from tests.helpers.wandb_offline import read_run_binary


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
            "sample_rate": 44100,
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

    Mirrors what ``generate_dataset.generate`` would have uploaded earlier in the
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

    Mirrors generate-stage's ``upload_spec(spec)`` so ``finalize()``
    re-hydrates the same ``DatasetSpec`` via ``load_spec_from_uri``.

    :param spec: Frozen ``DatasetSpec`` to serialize.
    :param tmp_path: Test-scoped tmp dir.
    :returns: ``file://`` URI consumable by ``load_spec_from_uri``.
    """
    spec_dir = tmp_path / "spec"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_file = spec_dir / "input_spec.json"
    spec_file.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    return spec_file.as_uri()


def _build_finalize_cfg(spec_uri: str, output_dir: Path) -> DictConfig:
    """Synthesize a minimal ``finalize()`` cfg without invoking Hydra's @main decoration.

    :param spec_uri: URI passed through to ``load_spec_from_uri``.
    :param output_dir: Directory finalize uses as its scratch ``work_dir``.
        Must exist (``@hydra.main`` ordinarily creates it before ``main()`` runs).
    :returns: Mutable DictConfig with the two fields ``finalize()`` consumes.
    """
    return cast(
        DictConfig,
        OmegaConf.create({"dataset_spec_uri": spec_uri, "paths": {"output_dir": str(output_dir)}}),
    )


@pytest.fixture()
def stub_finalize_setup(monkeypatch: pytest.MonkeyPatch) -> Callable[[int | None], None]:
    """Stub ``ensure_r2_env_loaded`` + marker-probe; expose a marker-size setter.

    Leaves ``r2_io.download_to_path`` and ``r2_io.upload`` unstubbed — paired
    with ``fake_r2_remote`` they run real ``rclone copyto`` against the
    local-typed remote so tests can assert on materialized objects.

    :param monkeypatch: Pytest fixture used to install the stubs.
    :returns: A setter that overrides the marker-probe's "size in R2"
        response — ``None`` (default) makes ``finalize()`` proceed; an
        ``int`` triggers the idempotency short-circuit.
    """
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.ensure_r2_env_loaded",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda _uri: None)

    def _set_marker_size(size: int | None) -> None:
        monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda _uri: size)

    return _set_marker_size


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
    spec = _build_wds_smoke_spec(task_name="finalize-marker-last-wds")
    _seed_train_shards(fake_r2_remote, spec)
    output_dir = tmp_path / "hydra_output"
    output_dir.mkdir()
    cfg = _build_finalize_cfg(_write_spec_to_file(spec, tmp_path), output_dir)

    real_upload = r2_io.upload
    upload_order: list[str] = []

    def spy_upload(src: str | Path, dst: str) -> None:
        upload_order.append(dst)
        real_upload(src, dst)

    monkeypatch.setattr("synth_setter.pipeline.r2_io.upload", spy_upload)

    finalize_dataset.finalize(cfg)

    stats_uri = spec.r2.stats_uri()
    marker_uri = spec.r2.dataset_complete_marker_uri()
    assert _uri_to_local_path(fake_r2_remote, stats_uri).is_file()
    assert _uri_to_local_path(fake_r2_remote, marker_uri).is_file()
    assert upload_order.count(stats_uri) == 1
    assert upload_order.count(marker_uri) == 1
    assert upload_order.index(marker_uri) == len(upload_order) - 1
    assert upload_order.index(stats_uri) < upload_order.index(marker_uri)


def test_finalize_from_spec_uploads_stats_then_marker_at_canonical_uris(
    tmp_path: Path,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_finalize_setup: Callable[[int | None], None],  # noqa: ARG001 — installs stubs only
) -> None:
    """``finalize_from_spec`` honors the marker-last ordering without re-loading the spec.

    Mirrors ``test_finalize_uploads_stats_then_marker_at_canonical_uris`` but
    calls the in-memory entry point directly — no ``cfg`` synthesis, no
    ``load_spec_from_uri`` round-trip — so the inline path
    (``generate_dataset.main`` will reuse) is pinned independently of the
    URI-driven entry point.

    :param tmp_path: Hosts the Hydra-style work_dir.
    :param fake_r2_remote: Local-typed rclone remote; both artifacts land here.
    :param monkeypatch: Pytest fixture used to wrap ``synth_setter.pipeline.r2_io.upload``
        with an order-recording spy that still delegates to the real helper.
    :param stub_finalize_setup: Fixture-activation only — installs the
        ``ensure_r2_env_loaded`` / ``object_size`` stubs.
    """
    spec = _build_wds_smoke_spec(task_name="finalize-from-spec-marker-last")
    _seed_train_shards(fake_r2_remote, spec)
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
    assert _uri_to_local_path(fake_r2_remote, stats_uri).is_file()
    assert _uri_to_local_path(fake_r2_remote, marker_uri).is_file()
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
    :param stub_finalize_setup: Installs the auth + marker-probe stubs.
    """
    spec = _build_wds_smoke_spec(task_name="finalize-custom-prefix")
    custom_r2 = spec.r2.model_copy(
        update={"prefix": "test-runs/finalize-custom-prefix/abc123def456/"}
    )
    custom_spec = spec.model_copy(update={"r2": custom_r2})
    _seed_train_shards(fake_r2_remote, custom_spec)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    recording_logger = MagicMock(wraps=finalize_dataset.logger)
    monkeypatch.setattr(finalize_dataset, "logger", recording_logger)

    finalize_dataset.finalize_from_spec(custom_spec, work_dir)

    assert _uri_to_local_path(fake_r2_remote, custom_spec.r2.stats_uri()).is_file()
    assert _uri_to_local_path(
        fake_r2_remote, custom_spec.r2.dataset_complete_marker_uri()
    ).is_file()
    assert any(
        "non-canonical r2 prefix" in str(call.args[0])
        for call in recording_logger.warning.call_args_list
    ), recording_logger.warning.call_args_list


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
    spec = _build_wds_smoke_spec(task_name="finalize-idempotent-wds")
    output_dir = tmp_path / "hydra_output"
    output_dir.mkdir()
    cfg = _build_finalize_cfg(_write_spec_to_file(spec, tmp_path), output_dir)

    finalize_dataset.finalize(cfg)

    stats_path = _uri_to_local_path(fake_r2_remote, spec.r2.stats_uri())
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
    bad_spec = _build_wds_smoke_spec(task_name="finalize-bad-format").model_copy(
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
    cfg = _build_finalize_cfg("file:///unused", output_dir)

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
            "sample_rate": 44100,
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


def _stub_get_stats_hdf5(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``finalize_dataset.get_stats_hdf5`` to write a sentinel ``stats.npz``.

    Real Dask startup dominates runtime; tests that drive ``finalize_hdf5``
    end-to-end use this stub so the subsequent ``r2_io.upload`` step still
    runs against a real on-disk file produced sibling to ``train.h5``.

    :param monkeypatch: Pytest fixture used to install the stub.
    """

    def fake_stats(train_h5_path: str, mask_degenerate: bool = False) -> None:
        del mask_degenerate
        np.savez(
            Path(train_h5_path).parent / "stats.npz",
            mean=np.zeros((2, 8, 8), dtype=np.float32),
            std=np.ones((2, 8, 8), dtype=np.float32),
        )

    monkeypatch.setattr("synth_setter.cli.finalize_dataset.get_stats_hdf5", fake_stats)


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
    _stub_get_stats_hdf5(monkeypatch)

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
    _stub_get_stats_hdf5(monkeypatch)

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

    spec = _build_hdf5_smoke_spec(task_name="finalize-hdf5-marker-present")
    output_dir = tmp_path / "hydra_output"
    output_dir.mkdir()
    cfg = _build_finalize_cfg(_write_spec_to_file(spec, tmp_path), output_dir)

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
    spec = _build_hdf5_smoke_spec(task_name="finalize-hdf5-marker-last")
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
    _stub_get_stats_hdf5(monkeypatch)

    output_dir = tmp_path / "hydra_output"
    output_dir.mkdir()
    cfg = _build_finalize_cfg(_write_spec_to_file(spec, tmp_path), output_dir)

    finalize_dataset.finalize(cfg)

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

    spec = _build_wds_smoke_spec(task_name="finalize-artifact-e2e")
    _seed_train_shards(fake_r2_remote, spec)
    output_dir = tmp_path / "work"
    output_dir.mkdir()
    cfg = _build_finalize_cfg_with_offline_wandb(
        _write_spec_to_file(spec, tmp_path), output_dir, tmp_path
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

    spec = _build_wds_smoke_spec(task_name="finalize-failed-close")
    output_dir = tmp_path / "work"
    output_dir.mkdir()
    cfg = _build_finalize_cfg_with_offline_wandb(
        _write_spec_to_file(spec, tmp_path), output_dir, tmp_path
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
    drives the ``except`` branch deterministically. State-based witness: the
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

    def boom(spec: DatasetSpec) -> NoReturn:
        del spec
        raise RuntimeError("simulated build_dataset_artifact failure")

    monkeypatch.setattr("synth_setter.cli.finalize_dataset.build_dataset_artifact", boom)

    spec = _build_wds_smoke_spec(task_name="finalize-artifact-swallow")
    _seed_train_shards(fake_r2_remote, spec)
    output_dir = tmp_path / "work"
    output_dir.mkdir()
    cfg = _build_finalize_cfg_with_offline_wandb(
        _write_spec_to_file(spec, tmp_path), output_dir, tmp_path
    )

    finalize_dataset.finalize(cfg)

    assert _uri_to_local_path(fake_r2_remote, spec.r2.stats_uri()).is_file()
    assert _uri_to_local_path(fake_r2_remote, spec.r2.dataset_complete_marker_uri()).is_file()
    assert wandb.run is None, "finalize() left the wandb run open after swallowing the failure"


def test_finalize_forces_wandb_resume_allow_when_group_present(
    tmp_path: Path,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_finalize_setup: Callable[[int | None], None],  # noqa: ARG001 — installs stubs only
) -> None:
    """``finalize()`` forces ``logger.wandb.resume="allow"`` before instantiating loggers.

    Pins finalize_dataset.py:319-320: when a wandb group is present the run must
    attach to the pinned generation run rather than mint a new one. Captures the
    cfg ``instantiate_loggers`` actually receives (the cfg object is mutated
    in-place, so the captured reference reflects the forced value) and stops the
    body early via a ``finalize_from_spec`` raise — the resume mutation happens
    *before* dispatch, so no R2 work is needed and the offline run never opens.

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

    spec = _build_wds_smoke_spec(task_name="finalize-resume-allow")
    output_dir = tmp_path / "work"
    output_dir.mkdir()
    cfg = _build_finalize_cfg_with_offline_wandb(
        _write_spec_to_file(spec, tmp_path), output_dir, tmp_path
    )

    with pytest.raises(RuntimeError, match="halt after logger setup"):
        finalize_dataset.finalize(cfg)

    assert OmegaConf.select(cfg, "logger.wandb.resume") == "allow"
    assert OmegaConf.select(captured_logger_cfg["cfg"], "wandb.resume") == "allow"


def test_stubbed_stats_signatures_match_production() -> None:
    """Stub stats signatures in this file match the real stats functions.

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
