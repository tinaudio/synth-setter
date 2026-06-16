"""Shared shard/spec/cfg helpers for the finalize-dataset test lanes.

The finalize entrypoint test (``tests/test_finalize_dataset.py``), its branch
unit test (``tests/pipeline/entrypoints/test_finalize_dataset_unit.py``), and the
real-R2 integration test (``tests/integration/test_finalize_dataset_r2.py``) all
seed deterministic wds/hdf5 shards and build smoke ``DatasetSpec`` objects the
same way. Centralizing the writers + builders here keeps the lanes from
drifting: a change to a shard's on-disk layout updates every caller at once.
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

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    DATASET_FIELD_DTYPES,
    MEL_SPEC_FIELD,
    PARAM_ARRAY_FIELD,
    dataset_field_shapes,
)
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata
from synth_setter.pipeline.schemas.spec import DatasetSpec, RenderConfig


def write_minimal_wds_shard(dest: Path) -> None:
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


def build_wds_smoke_spec(
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


# sample_rate=100 keeps the mel front end at its minimum hop so shards stay tiny.
_LANCE_SMOKE_RENDER: dict[str, str | int | float] = {
    "plugin_path": "/fake/Plugin.vst3",
    "preset_path": "presets/surge-base.vstpreset",
    "param_spec_name": "surge_simple",
    "renderer_version": "1.0.0-test",
    "sample_rate": 100,
    "channels": 2,
    "velocity": 100,
    "signal_duration_seconds": 1.0,
    "min_loudness": -55.0,
    "samples_per_render_batch": 4,
    "samples_per_shard": 4,
    "gui_toggle_cadence": "never",
}


def build_lance_smoke_spec(
    task_name: str = "finalize-lance-unit",
    train_val_test_sizes: tuple[int, int, int] = (4, 0, 0),
    mask_degenerate_bins: bool = False,
    render: RenderConfig | None = None,
    base_seed: int = 42,
) -> DatasetSpec:
    """Construct a lance ``DatasetSpec`` directly (no Hydra compose).

    :param task_name: Unique task name so each test gets a distinct r2.prefix.
    :param train_val_test_sizes: Three-tuple of sample counts; default is one
        4-sample shard.
    :param mask_degenerate_bins: Threaded onto the spec for stats-fold tests.
    :param render: Optional render config replacing the smoke default — used by
        e2e tests that must wrap the exact config a writer rendered with.
    :param base_seed: Dataset seed used to derive shard seeds.
    :returns: A frozen lance ``DatasetSpec`` whose shards are deterministic.
    """
    kwargs: dict[str, Any] = {
        "task_name": task_name,
        "output_format": "lance",
        "train_val_test_sizes": list(train_val_test_sizes),
        "base_seed": base_seed,
        "mask_degenerate_bins": mask_degenerate_bins,
        "r2": {"bucket": "intermediate-data"},
        "render": render if render is not None else dict(_LANCE_SMOKE_RENDER),
    }
    return DatasetSpec(**kwargs)  # type: ignore[arg-type]


def build_hdf5_smoke_spec(
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


def smoke_shard_metadata(render: RenderConfig) -> ShardMetadata:
    """Project ``render`` onto the ``ShardMetadata`` fields the writers embed.

    Test-side twin of the writers' projection so shard seeders and validator
    tests build the sidecar payload one way.

    :param render: Render config supplying the sidecar field values.
    :returns: Strict ``ShardMetadata`` with every render-derived field filled.
    """
    return ShardMetadata(
        velocity=render.velocity,
        signal_duration_seconds=render.signal_duration_seconds,
        sample_rate=render.sample_rate,
        channels=render.channels,
        min_loudness=render.min_loudness,
        base_seed=render.base_seed,
        attempts_per_sample=render.attempts_per_sample,
    )


def write_minimal_lance_shard(dest: Path, spec: DatasetSpec) -> None:
    """Write a structurally valid Lance shard for ``spec`` at ``dest``.

    :param dest: Filesystem path where the Lance file is written.
    :param spec: Lance spec whose render shape/dtypes define the shard contract.
    """
    from synth_setter.pipeline.data.lance_shard import (
        lance_schema,
        record_batch_from_arrays,
        write_lance_dataset,
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    render = spec.render.model_copy(update={"base_seed": spec.shards[0].seed})
    shapes = dataset_field_shapes(render, spec.num_params)
    schema = lance_schema(shapes, smoke_shard_metadata(render))
    arrays = {
        AUDIO_FIELD: np.zeros(shapes[AUDIO_FIELD], dtype=DATASET_FIELD_DTYPES[AUDIO_FIELD]),
        MEL_SPEC_FIELD: np.arange(
            np.prod(shapes[MEL_SPEC_FIELD]),
            dtype=DATASET_FIELD_DTYPES[MEL_SPEC_FIELD],
        ).reshape(shapes[MEL_SPEC_FIELD]),
        PARAM_ARRAY_FIELD: np.zeros(
            shapes[PARAM_ARRAY_FIELD],
            dtype=DATASET_FIELD_DTYPES[PARAM_ARRAY_FIELD],
        ),
    }
    write_lance_dataset(dest, schema, [record_batch_from_arrays(arrays, schema)])


def uri_to_local_path(fake_r2_remote: Path, r2_uri: str) -> Path:
    """Map an ``r2://bucket/key`` URI to its path under the local-typed remote.

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


def seed_train_shards(fake_r2_remote: Path, spec: DatasetSpec) -> list[Path]:
    """Materialize each train shard under ``<remote>/<bucket>/<prefix>/<filename>``.

    Mirrors what ``generate_dataset.generate`` would have uploaded earlier in the
    pipeline so finalize's ``r2_io.download_to_path`` finds a real tar under
    the local-typed remote.

    :param fake_r2_remote: Root of the local-typed remote.
    :param spec: Dataset spec whose ``shards`` and ``r2`` provide the layout.
    :returns: The seeded local shard paths, in train-range order.
    :raises ValueError: ``spec.output_format`` has no minimal-shard writer here.
    """
    train_lo, train_hi = spec.split_shard_ranges["train"]
    seeded: list[Path] = []
    for shard in spec.shards[train_lo:train_hi]:
        path = uri_to_local_path(fake_r2_remote, spec.r2.shard_uri(shard))
        if spec.output_format == "lance":
            write_minimal_lance_shard(path, spec)
        elif spec.output_format == "wds":
            write_minimal_wds_shard(path)
        else:
            raise ValueError(f"no minimal shard writer for output_format {spec.output_format!r}")
        seeded.append(path)
    return seeded


def seed_shard_files(remote_root: Path, spec: DatasetSpec) -> None:
    """Write every ``spec.shards[i].filename`` as a structurally valid HDF5 shard.

    Shapes/dtypes match
    :func:`synth_setter.pipeline.ci.validate_shard.check_shard_contracts`.

    :param remote_root: Directory acting as the R2-side staging area.
    :param spec: Spec whose ``shards`` define the filenames to seed.
    """
    remote_root.mkdir(parents=True, exist_ok=True)
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    for shard in spec.shards:
        with h5py.File(remote_root / shard.filename, "w") as f:
            for field, shape in shapes.items():
                f.create_dataset(field, shape=shape, dtype=DATASET_FIELD_DTYPES[field])


def write_spec_to_root(spec: DatasetSpec, tmp_path: Path) -> str:
    """Persist ``spec`` as ``input_spec.json`` and return its dataset-root ``file://`` URI.

    Mirrors generate-stage's ``upload_spec(spec)`` so ``finalize()``
    re-hydrates the same ``DatasetSpec`` via ``load_spec_from_root`` — the
    returned URI points at the directory holding ``input_spec.json``, not the
    spec file itself.

    :param spec: Frozen ``DatasetSpec`` to serialize.
    :param tmp_path: Test-scoped tmp dir.
    :returns: ``file://`` URI of the dataset-root dir, consumable by ``load_spec_from_root``.
    """
    spec_dir = tmp_path / "spec"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_file = spec_dir / "input_spec.json"
    spec_file.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    return spec_dir.as_uri()


def build_finalize_cfg(dataset_root_uri: str, output_dir: Path) -> DictConfig:
    """Synthesize a minimal ``finalize()`` cfg without invoking Hydra's @main.

    :param dataset_root_uri: Run-prefix URI passed through to ``load_spec_from_root``.
    :param output_dir: Directory finalize uses as its scratch ``work_dir``.
        Must exist (``@hydra.main`` ordinarily creates it before ``main()`` runs).
    :returns: Mutable DictConfig with the two fields ``finalize()`` consumes.
    """
    return cast(
        DictConfig,
        OmegaConf.create(
            {"dataset_root_uri": dataset_root_uri, "paths": {"output_dir": str(output_dir)}}
        ),
    )


def stub_get_stats_hdf5(monkeypatch: pytest.MonkeyPatch) -> None:
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


def install_finalize_setup_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[int | None], None]:
    """Stub ``ensure_r2_env_loaded`` + the marker probe; return a marker-size setter.

    Leaves ``r2_io.download_to_path`` and ``r2_io.upload`` unstubbed — paired
    with ``fake_r2_remote`` they run real ``rclone copyto`` against the
    local-typed remote so callers can assert on materialized objects. The
    marker probe (``object_size``) defaults to ``None`` so ``finalize`` /
    ``finalize_from_spec`` proceed past the idempotency check.

    :param monkeypatch: Pytest fixture used to install the stubs.
    :returns: A setter that overrides the marker-probe's "size in R2"
        response — ``None`` makes the call proceed; an ``int`` triggers the
        idempotency short-circuit.
    """
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.ensure_r2_env_loaded",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda _uri: None)

    def _set_marker_size(size: int | None) -> None:
        monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda _uri: size)

    return _set_marker_size


def copy_shard_for_download(r2_stand_in: Path, r2_uri: str, dst: Path) -> None:
    """Copy the shard ``r2_uri`` names from ``r2_stand_in`` to ``dst``.

    Mirrors ``r2_io.download_to_path``'s file→file (``rclone copyto``)
    contract: ``dst`` is the exact local path, not a directory, and
    ``r2_uri`` carries the basename.

    :param r2_stand_in: Directory holding the pre-seeded shards by basename.
    :param r2_uri: Canonical shard URI whose basename selects the source file.
    :param dst: Exact local destination path for the copy.
    """
    shutil.copy(r2_stand_in / Path(r2_uri).name, Path(dst))
