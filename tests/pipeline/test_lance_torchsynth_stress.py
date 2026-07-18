"""Lance pipeline stress tests driven by the in-process torchsynth backend.

The torchsynth renderer needs no plugin host, so these tests exercise the real generation → shard
write → staging → finalize → dataloader path at scales and edge geometries the VST-backed suites
cannot afford. Everything here runs real code. Failure injection wraps the fragment writer only
after a real flush, then verifies that no manifest is committed.
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib.metadata import version as package_version
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
from omegaconf import DictConfig

from synth_setter.data.vst.shapes import MEL_SPEC_FIELD, PARAM_ARRAY_FIELD
from synth_setter.data.vst.torchsynth_param_spec import TORCHSYNTH_ADSR_PARAM_SPEC
from synth_setter.data.vst.writers import make_lance_dataset
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.schemas.spec import DatasetSpec, RenderConfig
from synth_setter.workspace import operator_workspace

_SAMPLE_RATE = 22_050
_DURATION_SECONDS = 0.5


def _torchsynth_render_cfg(**overrides: object) -> RenderConfig:
    """Build a small torchsynth render config for one stress scenario.

    :param \\*\\*overrides: ``RenderConfig`` field overrides for the exercised
        shard size, render-batch size, and base seed axes.
    :returns: Validated render config.
    """
    kwargs: dict[str, object] = {
        "plugin_path": "torchsynth",
        "plugin_state_path": "",
        "param_spec_name": "torchsynth_adsr",
        "renderer_version": package_version("torchsynth"),
        "renderer_backend": "torchsynth",
        "sample_rate": _SAMPLE_RATE,
        "channels": 2,
        "velocity": 100,
        "signal_duration_seconds": _DURATION_SECONDS,
        "min_loudness": -70.0,
        "samples_per_render_batch": 4,
        "samples_per_shard": 6,
        "base_seed": 1757,
        "plugin_reload_cadence": "once",
        "gui_toggle_cadence": "never",
    }
    kwargs.update(overrides)
    return RenderConfig(**kwargs)  # type: ignore[arg-type]


def _read_params(path: Path) -> np.ndarray:
    """Materialize the parameter column of a shard.

    :param path: ``.lance`` dataset directory.
    :returns: ``(rows, width)`` float32 array.
    """
    table = lance.dataset(str(path)).to_table(columns=[PARAM_ARRAY_FIELD])
    return table.column(PARAM_ARRAY_FIELD).combine_chunks().to_numpy_ndarray()


def _assert_batch_boundary_shard(path: Path, expected_rows: int) -> None:
    """Assert row, dtype, and uniqueness contracts for a boundary-sweep shard.

    :param path: Committed ``.lance`` dataset directory.
    :param expected_rows: Number of rows the shard must contain.
    """
    params = _read_params(path)
    assert params.dtype == np.float32
    assert params.shape == (expected_rows, len(TORCHSYNTH_ADSR_PARAM_SPEC))
    table = lance.dataset(str(path)).to_table(columns=["audio", MEL_SPEC_FIELD])
    audio = table.column("audio").combine_chunks().to_numpy_ndarray()
    mel = table.column(MEL_SPEC_FIELD).combine_chunks().to_numpy_ndarray()
    assert audio.dtype == np.float16
    assert mel.dtype == np.float32
    assert audio.shape[0] == expected_rows
    # Rounding gives each seeded continuous row a float32-stable identity.
    unique_rows = {tuple(np.round(row, 6)) for row in params}
    assert len(unique_rows) == expected_rows


@pytest.mark.parametrize(
    ("samples_per_shard", "samples_per_render_batch"),
    [
        pytest.param(1, 1, id="single-row"),
        pytest.param(6, 1, id="one-fragment-per-row"),
        pytest.param(6, 4, id="trailing-remainder"),
        pytest.param(6, 5, id="non-even-split"),
        pytest.param(6, 6, id="one-fragment"),
        pytest.param(6, 8, id="batch-larger-than-shard"),
    ],
)
def test_make_lance_dataset_batch_boundary_sweep(
    tmp_path: Path, samples_per_shard: int, samples_per_render_batch: int
) -> None:
    """Every batch/shard boundary combination commits exactly the requested rows.

    Fragment flushing happens per render batch plus once for the remainder, so off-by-ones would
    surface as missing or duplicated tail rows.

    :param tmp_path: Destination directory for the shard.
    :param samples_per_shard: Rows the shard must hold.
    :param samples_per_render_batch: Writer batch size under test.
    """
    shard = tmp_path / "shard.lance"

    make_lance_dataset(
        shard,
        _torchsynth_render_cfg(
            samples_per_shard=samples_per_shard,
            samples_per_render_batch=samples_per_render_batch,
        ),
    )

    _assert_batch_boundary_shard(shard, samples_per_shard)


def test_make_lance_dataset_rerun_overwrites_instead_of_appending(tmp_path: Path) -> None:
    """A rerun over an existing dataset directory replaces it — no stale rows survive.

    The writer's documented contract is non-resumable overwrite; appending stale fragments would
    silently double the dataset.

    :param tmp_path: Destination directory for the shard.
    """
    shard = tmp_path / "shard.lance"

    make_lance_dataset(shard, _torchsynth_render_cfg(base_seed=1757))
    first = _read_params(shard)
    make_lance_dataset(shard, _torchsynth_render_cfg(base_seed=1758))
    second = _read_params(shard)

    assert first.shape == second.shape == (6, len(TORCHSYNTH_ADSR_PARAM_SPEC))
    assert not np.array_equal(first, second)


def test_make_lance_dataset_rerun_removes_stale_files(tmp_path: Path) -> None:
    """A rerun drops stale files from the previous dataset directory.

    :param tmp_path: Destination directory for the shard.
    """
    shard = tmp_path / "shard.lance"

    make_lance_dataset(shard, _torchsynth_render_cfg(base_seed=1757))
    stale_file = shard / "data" / "stale.bin"
    stale_file.write_text("stale")

    make_lance_dataset(shard, _torchsynth_render_cfg(base_seed=1758))

    assert not stale_file.exists()


def test_make_lance_dataset_existing_file_path_is_replaced(tmp_path: Path) -> None:
    """A stale file at the target path is replaced by the rendered dataset.

    :param tmp_path: Destination directory for the shard.
    """
    shard = tmp_path / "shard.lance"
    shard.write_text("stale")

    make_lance_dataset(shard, _torchsynth_render_cfg())

    assert shard.is_dir()
    assert _read_params(shard).shape == (6, len(TORCHSYNTH_ADSR_PARAM_SPEC))


def test_make_lance_dataset_failure_after_fragment_commits_nothing_and_rerun_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure after a fragment flush leaves no dataset; a clean rerun succeeds.

    :param tmp_path: Destination directory for the shard.
    :param monkeypatch: Injects a failure after the first real fragment flush.
    """
    from synth_setter.pipeline.data import lance_shard

    shard = tmp_path / "shard.lance"
    real_lance_fragment = lance_shard.lance_fragment
    fragment_calls = 0

    def _fail_after_first_fragment(
        uri: Path | str,
        schema: pa.Schema,
        batch: pa.RecordBatch | Iterable[pa.RecordBatch],
        *,
        storage_options: dict[str, str] | None = None,
    ) -> lance.fragment.FragmentMetadata:
        nonlocal fragment_calls
        fragment_calls += 1
        if fragment_calls > 1:
            raise RuntimeError("injected fragment failure")
        return real_lance_fragment(uri, schema, batch, storage_options=storage_options)

    monkeypatch.setattr(lance_shard, "lance_fragment", _fail_after_first_fragment)
    with pytest.raises(RuntimeError, match="injected fragment failure"):
        make_lance_dataset(shard, _torchsynth_render_cfg())

    assert fragment_calls == 2
    assert not shard.exists()

    monkeypatch.setattr(lance_shard, "lance_fragment", real_lance_fragment)
    make_lance_dataset(shard, _torchsynth_render_cfg())
    assert _read_params(shard).shape == (6, len(TORCHSYNTH_ADSR_PARAM_SPEC))


def test_make_lance_dataset_failed_rerun_preserves_existing_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed rerun leaves the previously committed dataset intact.

    :param tmp_path: Destination directory for the shard.
    :param monkeypatch: Injects a failure after the first real fragment flush.
    """
    from synth_setter.pipeline.data import lance_shard

    shard = tmp_path / "shard.lance"
    make_lance_dataset(shard, _torchsynth_render_cfg(base_seed=1757))
    first = _read_params(shard)
    real_lance_fragment = lance_shard.lance_fragment
    fragment_calls = 0

    def _fail_after_first_fragment(
        uri: Path | str,
        schema: pa.Schema,
        batch: pa.RecordBatch | Iterable[pa.RecordBatch],
        *,
        storage_options: dict[str, str] | None = None,
    ) -> lance.fragment.FragmentMetadata:
        nonlocal fragment_calls
        fragment_calls += 1
        if fragment_calls > 1:
            raise RuntimeError("injected fragment failure")
        return real_lance_fragment(uri, schema, batch, storage_options=storage_options)

    monkeypatch.setattr(lance_shard, "lance_fragment", _fail_after_first_fragment)
    with pytest.raises(RuntimeError, match="injected fragment failure"):
        make_lance_dataset(shard, _torchsynth_render_cfg(base_seed=1758))

    assert fragment_calls == 2
    assert np.array_equal(_read_params(shard), first)


def test_shard_seeds_isolate_rows_across_shards(tmp_path: Path) -> None:
    """Distinct per-shard seeds draw fully distinct parameter rows.

    Mirrors the launcher's ``base_seed + shard_id`` scheme: shard-seed reuse
    (or an ignored seed) would duplicate rows across shards and silently
    shrink the dataset's effective size.

    :param tmp_path: Destination directory for the shards.
    """
    all_rows: list[np.ndarray] = []
    for shard_id in range(3):
        shard = tmp_path / f"shard-{shard_id:06d}.lance"
        make_lance_dataset(shard, _torchsynth_render_cfg(base_seed=1757 + shard_id))
        all_rows.append(_read_params(shard))

    stacked = np.concatenate(all_rows, axis=0)
    unique_rows = {tuple(np.round(row, 6)) for row in stacked}
    assert len(unique_rows) == stacked.shape[0]


def _compose_stress_cfg(tmp_path: Path) -> DictConfig:
    """Compose the torchsynth smoke experiment for a small sharded run.

    :param tmp_path: Root for Hydra output/work/log paths.
    :returns: Composed, path-pinned ``DictConfig`` with logging disabled.
    """
    from hydra import compose, initialize_config_module
    from omegaconf import open_dict

    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="dataset",
            overrides=[
                "experiment=generate_dataset/torchsynth-smoke",
                "train_val_test_sizes=[12,6,6]",
                "render.samples_per_shard=6",
                "render.samples_per_render_batch=4",
                f"render.sample_rate={_SAMPLE_RATE}",
                f"render.signal_duration_seconds={_DURATION_SECONDS}",
                "render.min_loudness=-70.0",
            ],
        )
        with open_dict(cfg):
            cfg.paths.root_dir = str(operator_workspace())
            cfg.paths.output_dir = str(tmp_path)
            cfg.paths.work_dir = str(tmp_path)
            cfg.paths.log_dir = str(tmp_path)
            cfg.r2.prefix = "fake-r2/torchsynth-stress/"
            cfg.logger = None
    return cfg


def _assert_finalized_split_rows(run_root: Path, expected_rows: dict[str, int]) -> None:
    """Assert every finalized split has its requested row count.

    :param run_root: Finalized dataset root.
    :param expected_rows: Expected row count keyed by split name.
    """
    for split, rows in expected_rows.items():
        split_ds = lance.dataset(str(run_root / f"{split}.lance"))
        assert split_ds.count_rows() == rows, split


def _assert_map_loader_round_trip(
    run_root: Path, param_spec_name: ParamSpecName, expected_train_rows: int
) -> None:
    """Assert the map loader preserves ragged tails and finite tensor widths.

    :param run_root: Finalized dataset root.
    :param param_spec_name: Registry name that determines encoded width.
    :param expected_train_rows: Rows the train loader must return.
    """
    from synth_setter.data.lance_datamodule import LanceVSTDataModule
    from synth_setter.data.vst.param_spec_registry import resolve_param_spec

    datamodule = LanceVSTDataModule(
        dataset_root=run_root,
        param_spec_name=param_spec_name,
        batch_size=4,
        num_workers=0,
    )
    datamodule.setup()
    encoded_width = len(resolve_param_spec(param_spec_name))
    seen = 0
    for batch in datamodule.train_dataloader():
        mel, params = batch["mel_spec"], batch["params"]
        assert params.shape[1] == encoded_width
        assert mel.shape[0] == params.shape[0]
        assert bool(mel.isfinite().all()) and bool(params.isfinite().all())
        seen += params.shape[0]
    assert seen == expected_train_rows


def _generate_and_finalize_stress_run(
    tmp_path: Path, fake_r2_remote: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, DatasetSpec]:
    """Run the real worker and finalizer from a non-repository directory.

    :param tmp_path: Scratch root for Hydra paths and finalizer work.
    :param fake_r2_remote: Local-filesystem root backing the ``r2:`` remote.
    :param monkeypatch: Pins worker state, finalizer auth, and the worker cwd.
    :returns: Finalized R2 run root and the specification used to create it.
    """
    from synth_setter.cli.finalize_dataset import finalize_lance
    from synth_setter.cli.generate_dataset import from_hydra, spec_from_cfg

    monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
    monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")
    cfg = _compose_stress_cfg(tmp_path)
    spec = spec_from_cfg(cfg)

    worker_cwd = fake_r2_remote / "worker-cwd"
    worker_cwd.mkdir()
    monkeypatch.chdir(worker_cwd)
    from_hydra(cfg)

    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda *args, **kwargs: None
    )
    work_dir = tmp_path / "finalize-work"
    work_dir.mkdir()
    finalize_lance(spec, work_dir)
    return worker_cwd / spec.r2.bucket / spec.r2.prefix, spec


@pytest.mark.slow
def test_from_hydra_torchsynth_multishard_finalize_dataloader_round_trip_returns_finite_rows(
    tmp_path: Path,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full pipeline returns finite rows for every finalized split.

    :param tmp_path: Scratch root for Hydra paths and the finalize work dir.
    :param fake_r2_remote: Activates the local-filesystem ``r2:`` remote.
    :param monkeypatch: Pins worker state, finalize auth, and a non-repository cwd.
    """
    run_root, spec = _generate_and_finalize_stress_run(tmp_path, fake_r2_remote, monkeypatch)

    assert spec.num_shards == 4
    assert (run_root / "stats.npz").is_file()
    expected_rows = {"train": 12, "val": 6, "test": 6}
    _assert_finalized_split_rows(run_root, expected_rows)
    # The map loader preserves ragged shard tails; the legacy batch-indexed path drops them.
    _assert_map_loader_round_trip(run_root, spec.render.param_spec_name, expected_rows["train"])
