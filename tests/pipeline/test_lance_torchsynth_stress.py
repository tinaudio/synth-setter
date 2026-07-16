"""Lance pipeline stress tests driven by the in-process torchsynth backend.

The torchsynth renderer needs no plugin host, so these tests exercise the real generation → shard
write → staging → finalize → dataloader path at scales and edge geometries the VST-backed suites
cannot afford. Everything here runs the real code — no renderer stubs, no fragment-commit stubs.
"""

from __future__ import annotations

from pathlib import Path

import lance
import numpy as np
import pytest
from omegaconf import DictConfig

from synth_setter.data.vst.shapes import PARAM_ARRAY_FIELD
from synth_setter.data.vst.torchsynth_param_spec import TORCHSYNTH_ADSR_PARAM_SPEC
from synth_setter.data.vst.writers import make_lance_dataset
from synth_setter.pipeline.schemas.spec import RenderConfig

_SAMPLE_RATE = 22_050
_DURATION_SECONDS = 0.5


def _torchsynth_render_cfg(**overrides: object) -> RenderConfig:
    """Build a small torchsynth render config for one stress scenario.

    :param \\*\\*overrides: ``RenderConfig`` field overrides — the stress axes are
        ``samples_per_shard``, ``samples_per_render_batch``, ``min_loudness``,
        ``attempts_per_sample``, and ``base_seed``.
    :returns: Validated render config.
    """
    kwargs: dict[str, object] = {
        "plugin_path": "torchsynth",
        "plugin_state_path": "",
        "param_spec_name": "torchsynth_adsr",
        "renderer_version": "1.0.2",
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


@pytest.mark.parametrize(
    ("samples_per_shard", "samples_per_render_batch"),
    [
        (1, 1),  # single-row shard, single-row batch
        (6, 1),  # one fragment per row
        (6, 4),  # trailing remainder fragment (4 + 2)
        (6, 5),  # prime-ish split (5 + 1)
        (6, 6),  # exactly one fragment
        (6, 8),  # batch larger than the shard
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

    params = _read_params(shard)
    assert params.shape == (samples_per_shard, len(TORCHSYNTH_ADSR_PARAM_SPEC))
    # Distinct rows prove no duplicated batch tails (row i is seeded by
    # (base_seed, i)); rounding gives a hashable float32-stable row identity.
    unique_rows = {tuple(np.round(row, 6)) for row in params}
    assert len(unique_rows) == samples_per_shard


def test_make_lance_dataset_rerun_overwrites_instead_of_appending(tmp_path: Path) -> None:
    """A rerun over an existing dataset directory replaces it — no stale rows survive.

    The writer's documented contract is non-resumable overwrite; appending stale fragments would
    silently double the dataset.

    :param tmp_path: Destination directory for the shard.
    """
    shard = tmp_path / "shard.lance"

    make_lance_dataset(shard, _torchsynth_render_cfg())
    first = _read_params(shard)
    make_lance_dataset(shard, _torchsynth_render_cfg())
    second = _read_params(shard)

    assert first.shape == second.shape == (6, len(TORCHSYNTH_ADSR_PARAM_SPEC))
    assert np.array_equal(first, second)


def test_make_lance_dataset_failed_render_commits_nothing_and_rerun_recovers(
    tmp_path: Path,
) -> None:
    """A mid-shard failure leaves no committed dataset; a clean rerun succeeds.

    ``min_loudness=0.0`` is unreachable, so the loudness gate exhausts its
    attempt budget and raises. Fragments written before the failure must stay
    uncommitted (no manifest references them), and a follow-up run over the
    same directory must produce a complete dataset.

    :param tmp_path: Destination directory for the shard.
    """
    shard = tmp_path / "shard.lance"

    with pytest.raises(RuntimeError, match="min_loudness"):
        make_lance_dataset(
            shard,
            _torchsynth_render_cfg(min_loudness=0.0, attempts_per_sample=2),
        )

    with pytest.raises(ValueError, match="was not found"):
        lance.dataset(str(shard))

    make_lance_dataset(shard, _torchsynth_render_cfg())
    assert _read_params(shard).shape == (6, len(TORCHSYNTH_ADSR_PARAM_SPEC))


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
    """Compose the torchsynth smoke experiment scaled to four tiny shards.

    :param tmp_path: Root for Hydra output/work/log paths.
    :returns: Composed, path-pinned ``DictConfig`` with logging disabled.
    """
    from hydra import compose, initialize_config_module
    from omegaconf import open_dict

    from tests.conftest import _set_workspace_root

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
            _set_workspace_root(cfg)
            cfg.paths.output_dir = str(tmp_path)
            cfg.paths.work_dir = str(tmp_path)
            cfg.paths.log_dir = str(tmp_path)
            cfg.r2.prefix = "fake-r2/torchsynth-stress/"
            cfg.logger = None
    return cfg


@pytest.mark.slow
def test_from_hydra_torchsynth_multishard_finalize_dataloader_round_trip(
    tmp_path: Path,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full pipeline holds with real renders: generate → stage → finalize → load.

    Four shards render in real worker subprocesses (in-process torchsynth, no
    plugin), stage to a fake ``r2:`` remote, finalize commits the split
    datasets + ``stats.npz``, and the production Lance datamodule then serves
    batches with the spec's parameter width. No renderer or commit stubs. The
    stages chain deliberately — the contract under test is the hand-off
    between them; per-stage assertions below localize a failure.

    :param tmp_path: Scratch root for Hydra paths and the finalize work dir.
    :param fake_r2_remote: Local-filesystem root backing the ``r2:`` remote.
    :param monkeypatch: Pins the worker rank/world contract and finalize auth.
    """
    from synth_setter.cli.finalize_dataset import finalize_lance
    from synth_setter.cli.generate_dataset import from_hydra, spec_from_cfg
    from synth_setter.data.lance_datamodule import LanceVSTDataModule
    from synth_setter.data.vst.param_spec_registry import resolve_param_spec

    monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
    monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")
    cfg = _compose_stress_cfg(tmp_path)

    spec = spec_from_cfg(cfg)
    assert spec.num_shards == 4

    from_hydra(cfg)

    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda *args, **kwargs: None
    )
    work_dir = tmp_path / "finalize-work"
    work_dir.mkdir()
    finalize_lance(spec, work_dir)

    run_root = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    assert (run_root / "stats.npz").is_file()
    expected_rows = {"train": 12, "val": 6, "test": 6}
    for split, rows in expected_rows.items():
        split_ds = lance.dataset(str(run_root / f"{split}.lance"))
        assert split_ds.count_rows() == rows, split

    # loader="map" preserves ragged shard tails (6-row shards, batch 4); the
    # legacy batch-indexed path documents dropping them.
    datamodule = LanceVSTDataModule(
        dataset_root=run_root,
        param_spec_name=spec.render.param_spec_name,
        batch_size=4,
        num_workers=0,
        loader="map",
    )
    datamodule.setup()
    encoded_width = len(resolve_param_spec(spec.render.param_spec_name))
    seen = 0
    for batch in datamodule.train_dataloader():
        mel, params = batch["mel_spec"], batch["params"]
        assert params.shape[1] == encoded_width
        assert mel.shape[0] == params.shape[0]
        assert bool(mel.isfinite().all()) and bool(params.isfinite().all())
        seen += params.shape[0]
    assert seen == expected_rows["train"]
