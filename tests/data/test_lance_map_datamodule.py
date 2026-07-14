"""Behavioral tests for the ``loader="map"`` path of ``LanceVSTDataModule``.

Covers the sample-indexed path behind the config switch (#1740):

* :class:`PrepareBatchCollate` — the picklable collate that turns
  ``LanceMapDataset.__getitems__``'s pre-collated tensor dict into
  :func:`prepare_batch` model batches.
* ``LanceVSTDataModule(loader="map")`` — Lightning ``setup`` / dataloader /
  ``teardown`` wiring over per-split :class:`LanceMapDataset` instances.

Every test drives real Lance datasets written through the pipeline writer —
no fakes or mocks of the storage layer anywhere in this module. Fixtures
remain small; coverage targets shapes, flags, values, and flow routing,
mirroring ``tests/data/test_lance_datamodule.py``.
"""

from __future__ import annotations

import contextlib
import pickle
from collections.abc import Iterator
from itertools import combinations, product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lightning import LightningModule, Trainer

from synth_setter.data.lance_datamodule import (
    LanceVSTDataModule,
    LanceVSTDataset,
    PrepareBatchCollate,
)
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.param_spec_name import ParamSpecName
from tests.helpers.lance_fixtures import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLES,
    MEL_SHAPE,
    NUM_PARAMS,
    make_shard_columns,
    write_mel_stats,
    write_seeded_lance_shard,
)


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Build a dataset root with ``train/val/test.lance`` + identity ``stats.npz``.

    :param tmp_path: Per-test tmpdir.
    :return: Path to the populated dataset root directory.
    """
    root = tmp_path / "data"
    root.mkdir()
    write_seeded_lance_shard(root / "train.lance", num_rows=16, seed=1)
    write_seeded_lance_shard(root / "val.lance", num_rows=6, seed=2)
    write_seeded_lance_shard(root / "test.lance", num_rows=6, seed=3)
    write_mel_stats(root)
    return root


@contextlib.contextmanager
def _set_up_map_module(**kwargs: object) -> Iterator[LanceVSTDataModule]:
    """Construct a ``loader="map"`` datamodule, ``setup``, yield, ``teardown``.

    :param \\*\\*kwargs: Forwarded to ``LanceVSTDataModule``; ``loader`` and
        cheap loader defaults (no workers, no pinning) are pre-set.
    :yields: The set-up datamodule for assertion work inside the ``with`` block.
    :ytype: LanceVSTDataModule
    """
    kwargs.setdefault("loader", "map")
    kwargs.setdefault("num_workers", 0)
    kwargs.setdefault("pin_memory", False)
    kwargs.setdefault("param_spec_name", ParamSpecName("surge_xt"))
    module = LanceVSTDataModule(**kwargs)  # type: ignore[arg-type]
    module.setup()
    try:
        yield module
    finally:
        module.teardown()


def _unwrap(maybe_tensor: torch.Tensor | None) -> torch.Tensor:
    """Assert ``maybe_tensor`` is non-None and return it narrowed.

    :param maybe_tensor: The batch value to narrow.
    :return: The same tensor, now typed as non-None.
    """
    assert maybe_tensor is not None
    return maybe_tensor


def _params_in_order(loader: torch.utils.data.DataLoader) -> np.ndarray:
    """Concatenate ``params`` across all yielded batches in iteration order.

    :param loader: Loader whose epoch is materialized.
    :return: ``(total_rows, num_params)`` array.
    """
    return torch.cat([_unwrap(batch["params"]) for batch in loader]).numpy()


class _DDPIndexRecorder(LightningModule):
    """Record source-row indices observed by one spawned Lightning rank."""

    def __init__(self, source_rows: np.ndarray, output_dir: Path) -> None:
        """Store the exact source rows used to recover batch indices.

        :param source_rows: Model-ready parameter rows in source order.
        :param output_dir: Directory receiving one index file per rank.
        """
        super().__init__()
        self.source_rows: torch.Tensor
        self.register_buffer("source_rows", torch.from_numpy(source_rows))
        self.output_dir = output_dir
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.seen_indices: list[int] = []

    def training_step(
        self, batch: dict[str, torch.Tensor | None], batch_idx: int
    ) -> torch.Tensor:
        """Recover and retain the source index of every row in this rank's batch.

        :param batch: Model-ready batch from the map dataloader.
        :param batch_idx: Batch index assigned by Lightning; unused.
        :return: A zero-valued differentiable loss.
        :raises AssertionError: If a batch row does not identify one source row.
        """
        del batch_idx
        params = _unwrap(batch["params"])
        matches = torch.all(params[:, None, :] == self.source_rows[None, :, :], dim=2)
        if not torch.all(matches.sum(dim=1) == 1):
            raise AssertionError("each DDP batch row must match exactly one source row")
        self.seen_indices.extend(matches.to(torch.int64).argmax(dim=1).cpu().tolist())
        return self.scale * 0

    def on_train_epoch_end(self) -> None:
        """Persist this rank's observed source indices for the parent assertion."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            self.output_dir / f"rank-{self.global_rank}.npy",
            np.asarray(self.seen_indices, dtype=np.int64),
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return the minimal optimizer required by the Trainer loop.

        :return: SGD over the dummy differentiable scale.
        """
        return torch.optim.SGD(self.parameters(), lr=0.0)


class TestPrepareBatchCollate:
    """The map path's per-batch semantics owner: a picklable bridge into ``prepare_batch``."""

    def _raw_batch(self, num_rows: int = 4) -> dict[str, torch.Tensor]:
        """Build the pre-collated tensor dict ``LanceMapDataset.__getitems__`` yields.

        :param num_rows: Batch size of the synthetic batch.
        :return: Column-name-to-tensor mapping matching the shard schema.
        """
        columns = make_shard_columns(num_rows, seed=7)
        return {name: torch.from_numpy(values) for name, values in columns.items()}

    def test_collate_returns_float32_model_batch_with_noise(self) -> None:
        """The collate emits the ``prepare_batch`` contract: float32 tensors + noise."""
        collate = PrepareBatchCollate(mean=None, std=None, rescale_params=False, ot=False)
        batch = collate(self._raw_batch(num_rows=4))
        assert _unwrap(batch["params"]).shape == (4, NUM_PARAMS)
        assert _unwrap(batch["noise"]).shape == (4, NUM_PARAMS)
        for key in ("mel_spec", "m2l", "params", "noise", "audio"):
            assert _unwrap(batch[key]).dtype == torch.float32, key

    def test_collate_missing_optional_columns_map_to_none(self) -> None:
        """A projected-out modality is absent from the raw dict and lands as ``None``."""
        collate = PrepareBatchCollate(mean=None, std=None, rescale_params=False, ot=False)
        raw = self._raw_batch()
        del raw["audio"], raw["music2latent"]
        batch = collate(raw)
        assert batch["audio"] is None
        assert batch["m2l"] is None
        assert _unwrap(batch["mel_spec"]).shape == (4, *MEL_SHAPE)

    def test_collate_normalizes_mel_with_mean_and_std(self) -> None:
        """``(mel - mean) / std`` is applied when stats are provided."""
        collate = PrepareBatchCollate(
            mean=np.full(MEL_SHAPE, 1.0, dtype=np.float32),
            std=np.full(MEL_SHAPE, 2.0, dtype=np.float32),
            rescale_params=False,
            ot=False,
        )
        raw = self._raw_batch()
        raw["mel_spec"] = torch.full((4, *MEL_SHAPE), 3.0)
        mel = _unwrap(collate(raw)["mel_spec"])
        assert torch.allclose(mel, torch.full_like(mel, 1.0))

    def test_collate_rescale_params_centers_to_minus_one_one(self) -> None:
        """``rescale_params=True`` applies ``p * 2 - 1`` element-wise."""
        raw = self._raw_batch()
        plain = PrepareBatchCollate(mean=None, std=None, rescale_params=False, ot=False)(dict(raw))
        rescaled = PrepareBatchCollate(mean=None, std=None, rescale_params=True, ot=False)(
            dict(raw)
        )
        assert torch.allclose(_unwrap(rescaled["params"]), _unwrap(plain["params"]) * 2 - 1)

    def test_collate_ot_true_permutes_rows_bijectively(self) -> None:
        """``ot=True`` may only reorder rows — sorted rows match the ``ot=False`` batch."""
        raw = self._raw_batch()
        plain = PrepareBatchCollate(mean=None, std=None, rescale_params=False, ot=False)(
            dict(raw)
        )["params"]
        matched = PrepareBatchCollate(mean=None, std=None, rescale_params=False, ot=True)(
            dict(raw)
        )["params"]
        plain_sorted = _unwrap(plain)[torch.argsort(_unwrap(plain)[:, 0])]
        matched_sorted = _unwrap(matched)[torch.argsort(_unwrap(matched)[:, 0])]
        assert torch.allclose(matched_sorted, plain_sorted)

    def test_collate_round_trips_through_pickle_for_spawn_workers(self) -> None:
        """Spawn workers pickle the collate; it must survive even after first use."""
        collate = PrepareBatchCollate(mean=None, std=None, rescale_params=False, ot=False)
        collate(self._raw_batch())  # materialize the lazy generator before pickling
        # S301: round-trips a trusted in-process object to pin spawn-worker
        # pickling; no untrusted data is deserialized.
        clone = pickle.loads(pickle.dumps(collate))  # noqa: S301
        batch = clone(self._raw_batch())
        assert _unwrap(batch["noise"]).shape == (4, NUM_PARAMS)

    def test_collate_noise_advances_across_calls_and_varies_by_row(self) -> None:
        """One instance's cached generator advances: fresh noise per batch, distinct rows."""
        collate = PrepareBatchCollate(mean=None, std=None, rescale_params=False, ot=False)
        first = _unwrap(collate(self._raw_batch())["noise"])
        second = _unwrap(collate(self._raw_batch())["noise"])
        assert not torch.equal(first, second)
        assert not torch.equal(first[0], first[1])

    def test_collate_same_global_seed_reproduces_noise(self) -> None:
        """Construction draws its seed from the global RNG, so ``seed_everything`` governs."""

        def noise_after_seed() -> torch.Tensor:
            torch.manual_seed(1234)
            collate = PrepareBatchCollate(mean=None, std=None, rescale_params=False, ot=False)
            return _unwrap(collate(self._raw_batch())["noise"])

        assert torch.equal(noise_after_seed(), noise_after_seed())

    def test_collate_ddp_ranks_receive_distinct_reproducible_noise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In-process DDP ranks derive distinct reproducible noise.

        :param monkeypatch: Fixture controlling the distributed rank reported to the collate.
        """
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

        def noise_for_rank(rank: int) -> torch.Tensor:
            monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
            torch.manual_seed(1234)
            collate = PrepareBatchCollate(
                mean=None, std=None, rescale_params=False, ot=False
            )
            return _unwrap(collate(self._raw_batch())["noise"])

        assert not torch.equal(noise_for_rank(0), noise_for_rank(1))
        assert torch.equal(noise_for_rank(1), noise_for_rank(1))

    def test_collate_ddp_worker_grid_receives_distinct_reproducible_noise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every distributed rank-worker pair receives its own noise stream.

        :param monkeypatch: Fixture controlling distributed rank and worker identity.
        """
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        base_seed = 91
        num_workers = 2

        def noise_for(rank: int, worker_id: int) -> torch.Tensor:
            monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
            monkeypatch.setattr(
                torch.utils.data,
                "get_worker_info",
                lambda: SimpleNamespace(
                    seed=base_seed + worker_id, num_workers=num_workers
                ),
            )
            collate = PrepareBatchCollate(
                mean=None, std=None, rescale_params=False, ot=False
            )
            return _unwrap(collate(self._raw_batch())["noise"])

        rank_worker_pairs = list(product(range(2), range(num_workers)))
        first_pass = [noise_for(*pair) for pair in rank_worker_pairs]
        second_pass = [noise_for(*pair) for pair in rank_worker_pairs]

        assert all(
            torch.equal(first, second)
            for first, second in zip(first_pass, second_pass, strict=True)
        )
        assert all(
            not torch.equal(left, right)
            for left, right in combinations(first_pass, 2)
        )


class TestLanceMapDataModuleSetup:
    """``loader`` routing at construction and ``setup`` time."""

    def test_default_loader_stays_legacy(self, dataset_root: Path) -> None:
        """No ``loader`` argument keeps the batch-indexed ``LanceVSTDataset`` path.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        module = LanceVSTDataModule(
            dataset_root=dataset_root,
            batch_size=2,
            ot=False,
            param_spec_name=ParamSpecName("surge_xt"),
        )
        module.setup()
        try:
            assert isinstance(module.train_dataset, LanceVSTDataset)
        finally:
            module.teardown()

    def test_map_loaders_are_sample_indexed(self, dataset_root: Path) -> None:
        """``loader="map"`` loaders carry sample semantics: row-count datasets, real batch size.

        The legacy path is batch-indexed (``batch_size=None``, dataset length in
        batches); this verifies that Lightning receives sample-indexed semantics.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_map_module(dataset_root=dataset_root, batch_size=2, ot=False) as module:
            loaders = {
                "train": (module.train_dataloader(), 16),
                "val": (module.val_dataloader(), 6),
                "test": (module.test_dataloader(), 6),
                "predict": (module.predict_dataloader(), 6),
            }
            assert module.train_dataset is loaders["train"][0].dataset
            assert module.val_dataset is loaders["val"][0].dataset
            assert module.test_dataset is loaders["test"][0].dataset
            assert module.predict_dataset is loaders["predict"][0].dataset
            for name, (loader, num_rows) in loaders.items():
                assert loader.batch_size == 2, name
                assert len(loader.dataset) == num_rows, name  # type: ignore[arg-type]

    def test_unknown_loader_raises_value_error(self, dataset_root: Path) -> None:
        """An unregistered ``loader`` value fails at construction, not mid-training.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with pytest.raises(ValueError, match="loader"):
            # Hydra configs pass arbitrary strings at runtime, so the guard is
            # the trust boundary the Literal annotation cannot enforce.
            LanceVSTDataModule(
                dataset_root=dataset_root,
                batch_size=2,
                loader="iterable",  # type: ignore[arg-type]
                param_spec_name=ParamSpecName("surge_xt"),
            )

    def test_persistent_workers_without_workers_raises_value_error(
        self, dataset_root: Path
    ) -> None:
        """Worker persistence fails at construction when no workers exist.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with pytest.raises(ValueError, match="persistent_workers requires num_workers > 0"):
            LanceVSTDataModule(
                dataset_root=dataset_root,
                batch_size=2,
                loader="map",
                num_workers=0,
                persistent_workers=True,
                param_spec_name=ParamSpecName("surge_xt"),
            )

    def test_map_missing_stats_raises_file_not_found(self, tmp_path: Path) -> None:
        """``use_saved_mean_and_variance=True`` with no ``stats.npz`` errors at setup.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        root = tmp_path / "data"
        root.mkdir()
        for split in ("train", "val", "test"):
            write_seeded_lance_shard(root / f"{split}.lance", num_rows=4)
        module = LanceVSTDataModule(
            dataset_root=root,
            batch_size=2,
            loader="map",
            param_spec_name=ParamSpecName("surge_xt"),
        )
        with pytest.raises(FileNotFoundError, match="stats.npz"):
            module.setup()

    def test_map_unregistered_param_spec_raises_at_setup(self, dataset_root: Path) -> None:
        """Legacy parity: an unregistered ``param_spec_name`` fails fast at setup.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        module = LanceVSTDataModule(
            dataset_root=dataset_root,
            batch_size=2,
            loader="map",
            param_spec_name=ParamSpecName("no-such-spec"),
        )
        with pytest.raises(KeyError, match="no-such-spec"):
            module.setup()

    def test_map_teardown_then_setup_serves_batches_again(self, dataset_root: Path) -> None:
        """A teardown/setup cycle rebuilds working splits (Lightning reuses modules).

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        module = LanceVSTDataModule(
            dataset_root=dataset_root,
            batch_size=2,
            loader="map",
            num_workers=0,
            pin_memory=False,
            param_spec_name=ParamSpecName("surge_xt"),
        )
        module.setup()
        module.teardown()
        module.setup()
        try:
            batch = next(iter(module.val_dataloader()))
            assert _unwrap(batch["params"]).shape == (2, NUM_PARAMS)
        finally:
            module.teardown()

    def test_train_loader_exposes_standard_sampler_before_lightning(
        self, dataset_root: Path
    ) -> None:
        """The map loader stays on Lightning's supported sampler-replacement path.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_map_module(dataset_root=dataset_root, batch_size=4) as module:
            loader = module.train_dataloader()

        assert isinstance(loader.sampler, torch.utils.data.RandomSampler)
        assert loader.batch_size == 4


class TestLanceMapDataModuleFlows:
    """Dataloader semantics per Lightning flow: train / val / test / predict."""

    def test_val_loader_yields_source_rows_in_order_with_ragged_tail(
        self, dataset_root: Path
    ) -> None:
        """Sequential val iteration returns exactly the written rows, tail included.

        The map path keeps the ragged final batch instead of legacy's floor-divide drop.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        source = make_shard_columns(6, seed=2)["param_array"] * 2 - 1
        with _set_up_map_module(dataset_root=dataset_root, batch_size=4, ot=False) as module:
            batches = list(module.val_dataloader())
            assert [len(_unwrap(b["params"])) for b in batches] == [4, 2]
            np.testing.assert_array_equal(
                torch.cat([_unwrap(b["params"]) for b in batches]).numpy(), source
            )

    def test_train_loader_shuffles_and_covers_every_row(self, dataset_root: Path) -> None:
        """One train epoch is a row permutation: full coverage, non-sequential order.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        torch.manual_seed(0)
        source = make_shard_columns(16, seed=1)["param_array"] * 2 - 1
        with _set_up_map_module(dataset_root=dataset_root, batch_size=4, ot=False) as module:
            epoch = _params_in_order(module.train_dataloader())
        assert epoch.shape == source.shape
        assert not np.array_equal(epoch, source), "train epoch came back unshuffled"
        order = np.lexsort(epoch.T[::-1])
        source_order = np.lexsort(source.T[::-1])
        np.testing.assert_array_equal(epoch[order], source[source_order])

    def test_train_loader_drops_ragged_tail(self, dataset_root: Path) -> None:
        """Train keeps legacy floor-divide semantics: no short trailing batch.

        A trailing batch as small as one row would break batch-statistics
        layers mid-training; eval loaders keep the tail instead.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_map_module(dataset_root=dataset_root, batch_size=5, ot=False) as module:
            batches = list(module.train_dataloader())
        assert [len(_unwrap(b["params"])) for b in batches] == [5, 5, 5]

    def test_val_and_test_loaders_do_not_shuffle_or_ot(self, dataset_root: Path) -> None:
        """Eval splits keep source row order even when the module trains with OT.

        In-order equality is only possible if neither shuffling nor the OT row permutation touched
        the eval loaders.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_map_module(dataset_root=dataset_root, batch_size=3, ot=True) as module:
            np.testing.assert_array_equal(
                _params_in_order(module.val_dataloader()),
                make_shard_columns(6, seed=2)["param_array"] * 2 - 1,
            )
            np.testing.assert_array_equal(
                _params_in_order(module.test_dataloader()),
                make_shard_columns(6, seed=3)["param_array"] * 2 - 1,
            )

    def test_predict_loader_reads_audio_eval_loaders_do_not(self, dataset_root: Path) -> None:
        """Only the predict flow pays for the audio column.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_map_module(dataset_root=dataset_root, batch_size=2, ot=False) as module:
            val_batch = next(iter(module.val_dataloader()))
            predict_batch = next(iter(module.predict_dataloader()))
        assert val_batch["audio"] is None
        assert _unwrap(predict_batch["audio"]).shape == (2, AUDIO_CHANNELS, AUDIO_SAMPLES)
        assert _unwrap(predict_batch["audio"]).dtype == torch.float32

    def test_m2l_conditioning_swaps_mel_for_music2latent(self, dataset_root: Path) -> None:
        """``conditioning="m2l"`` projects ``music2latent`` and drops ``mel_spec``.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_map_module(
            dataset_root=dataset_root, batch_size=2, ot=False, conditioning="m2l"
        ) as module:
            batch = next(iter(module.val_dataloader()))
        assert batch["mel_spec"] is None
        np.testing.assert_array_equal(
            _unwrap(batch["m2l"]).numpy(), make_shard_columns(6, seed=2)["music2latent"][:2]
        )

    def test_mel_normalized_with_saved_stats(self, tmp_path: Path) -> None:
        """Loaded ``stats.npz`` mean/std are applied as ``(mel - mean) / std``.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        root = tmp_path / "data"
        root.mkdir()
        for split in ("train", "val", "test"):
            write_seeded_lance_shard(root / f"{split}.lance", num_rows=4, mel_fill=3.0)
        write_mel_stats(root, mean=1.0, std=2.0)
        with _set_up_map_module(dataset_root=root, batch_size=2, ot=False) as module:
            mel = _unwrap(next(iter(module.val_dataloader()))["mel_spec"])
        assert torch.allclose(mel, torch.full_like(mel, 1.0))

    def test_predict_file_outside_root_uses_its_own_stats(self, tmp_path: Path) -> None:
        """A ``predict_file`` in another directory normalizes with that directory's stats.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        root = tmp_path / "data"
        root.mkdir()
        for split in ("train", "val", "test"):
            write_seeded_lance_shard(root / f"{split}.lance", num_rows=4)
        write_mel_stats(root, mean=0.0, std=1.0)
        predict_dir = tmp_path / "capture"
        predict_dir.mkdir()
        write_seeded_lance_shard(predict_dir / "predict.lance", num_rows=4, mel_fill=3.0)
        write_mel_stats(predict_dir, mean=1.0, std=2.0)
        with _set_up_map_module(
            dataset_root=root,
            batch_size=2,
            ot=False,
            predict_file=predict_dir / "predict.lance",
        ) as module:
            mel = _unwrap(next(iter(module.predict_dataloader()))["mel_spec"])
        assert torch.allclose(mel, torch.full_like(mel, 1.0))

    def test_batches_are_float32_contiguous_and_writable(self, dataset_root: Path) -> None:
        """Model-batch tensors own writable memory out of the Arrow decode path.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_map_module(dataset_root=dataset_root, batch_size=2, ot=False) as module:
            batch = next(iter(module.predict_dataloader()))
        for key in ("mel_spec", "params", "noise", "audio"):
            tensor = _unwrap(batch[key])
            assert tensor.dtype == torch.float32, key
            assert tensor.is_contiguous(), key
            assert tensor.numpy().flags.writeable, key


class TestLanceMapDataModuleModes:
    """Fake mode, ``repeat_first_batch``, and worker-process parity."""

    def test_fake_mode_synthesizes_batches_without_files(self, tmp_path: Path) -> None:
        """``fake=True`` under ``loader="map"`` never touches Lance — an empty root works.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        with _set_up_map_module(
            dataset_root=tmp_path,
            batch_size=2,
            ot=False,
            fake=True,
            use_saved_mean_and_variance=False,
        ) as module:
            batch = next(iter(module.val_dataloader()))
        assert _unwrap(batch["params"]).shape == (2, len(param_specs["surge_xt"]))

    def test_repeat_first_batch_every_train_batch_is_the_first(self, dataset_root: Path) -> None:
        """``repeat_first_batch=True`` yields rows ``[0, batch_size)`` for every batch.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        first_rows = make_shard_columns(16, seed=1)["param_array"][:4] * 2 - 1
        with _set_up_map_module(
            dataset_root=dataset_root, batch_size=4, ot=False, repeat_first_batch=True
        ) as module:
            batches = list(module.train_dataloader())
        assert len(batches) == 4
        for batch in batches:
            np.testing.assert_array_equal(_unwrap(batch["params"]).numpy(), first_rows)

    def test_repeat_first_batch_drops_ragged_tail(self, dataset_root: Path) -> None:
        """A row count not divisible by ``batch_size`` never yields a truncated repeat.

        Legacy floor-divides the row count, so every repeated batch contains
        the full first ``batch_size`` rows.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        first_rows = make_shard_columns(16, seed=1)["param_array"][:5] * 2 - 1
        with _set_up_map_module(
            dataset_root=dataset_root, batch_size=5, ot=False, repeat_first_batch=True
        ) as module:
            batches = list(module.train_dataloader())
        assert [len(_unwrap(b["params"])) for b in batches] == [5, 5, 5]
        for batch in batches:
            np.testing.assert_array_equal(_unwrap(batch["params"]).numpy(), first_rows)

    def test_repeat_first_batch_smaller_dataset_than_batch_raises(
        self, dataset_root: Path
    ) -> None:
        """A split with less than one full batch fails fast instead of yielding nothing.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with (
            pytest.raises(ValueError, match="full batch"),
            _set_up_map_module(
                dataset_root=dataset_root, batch_size=64, ot=False, repeat_first_batch=True
            ) as module,
        ):
            next(iter(module.train_dataloader()))

    def test_stats_off_leaves_mel_unnormalized(self, tmp_path: Path) -> None:
        """``use_saved_mean_and_variance=False`` skips stats even when a file exists.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        root = tmp_path / "data"
        root.mkdir()
        for split in ("train", "val", "test"):
            write_seeded_lance_shard(root / f"{split}.lance", num_rows=4, mel_fill=3.0)
        write_mel_stats(root, mean=1.0, std=2.0)
        with _set_up_map_module(
            dataset_root=root, batch_size=2, ot=False, use_saved_mean_and_variance=False
        ) as module:
            mel = _unwrap(next(iter(module.val_dataloader()))["mel_spec"])
        assert torch.allclose(mel, torch.full_like(mel, 3.0))

    def test_repeat_first_batch_folds_val_but_never_predict(self, dataset_root: Path) -> None:
        """Eval splits repeat the first batch like legacy; predict stays unfolded.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        val_first = make_shard_columns(6, seed=2)["param_array"][:3] * 2 - 1
        test_source = make_shard_columns(6, seed=3)["param_array"] * 2 - 1
        with _set_up_map_module(
            dataset_root=dataset_root, batch_size=3, ot=False, repeat_first_batch=True
        ) as module:
            for batch in module.val_dataloader():
                np.testing.assert_array_equal(_unwrap(batch["params"]).numpy(), val_first)
            # predict defaults to the test split; unfolded means source order.
            np.testing.assert_array_equal(
                _params_in_order(module.predict_dataloader()), test_source
            )

    @pytest.mark.slow
    def test_val_loader_spawn_workers_match_in_process(self, dataset_root: Path) -> None:
        """Spawn workers read the same rows as in-process loading.

        The map path pickles the dataset and collate into spawned workers (Lance is not fork-safe)
        — parity proves both survive the round-trip.

        :param dataset_root: Fixture-provided dataset-root directory.
        """

        def collect(num_workers: int) -> np.ndarray:
            with _set_up_map_module(
                dataset_root=dataset_root, batch_size=2, ot=False, num_workers=num_workers
            ) as module:
                return _params_in_order(module.val_dataloader())

        np.testing.assert_array_equal(collect(num_workers=2), collect(num_workers=0))

    @pytest.mark.slow
    def test_val_loader_spawn_worker_noise_is_distinct_and_reproducible(
        self, dataset_root: Path
    ) -> None:
        """Spawn workers receive reproducible seeds without sharing a noise stream.

        :param dataset_root: Fixture-provided dataset-root directory.
        """

        def collect() -> torch.Tensor:
            torch.manual_seed(47)
            with _set_up_map_module(
                dataset_root=dataset_root, batch_size=2, ot=False, num_workers=2
            ) as module:
                batches = [_unwrap(batch["noise"]) for batch in module.val_dataloader()]
            return torch.stack(batches)

        first = collect()
        second = collect()

        assert torch.equal(first, second)
        assert not torch.equal(first[0], first[1])

    @pytest.mark.slow
    @pytest.mark.parametrize("num_rows", [15, 16])
    def test_ddp_sim_default_sampler_covers_epoch(
        self, tmp_path: Path, num_rows: int
    ) -> None:
        """Lightning's default distributed sampler covers divisible and padded epochs.

        Uses the same CPU, two-device, ``ddp_spawn`` geometry as
        ``configs/trainer/ddp_sim.yaml``. The map loader deliberately supplies
        no custom sampler, leaving Lightning's default replacement path active.
        Non-divisible epochs repeat one row so both ranks receive equal work;
        divisible epochs remain disjoint.

        :param tmp_path: Directory receiving the spawned ranks' row-index files.
        :param num_rows: Training rows; odd values exercise sampler padding.
        """
        dataset_root = tmp_path / "data"
        dataset_root.mkdir()
        for seed, split in enumerate(("train", "val", "test"), start=1):
            write_seeded_lance_shard(
                dataset_root / f"{split}.lance", num_rows=num_rows, seed=seed
            )
        write_mel_stats(dataset_root)
        source_rows = make_shard_columns(num_rows, seed=1)["param_array"] * 2 - 1
        output_dir = tmp_path / "rank-indices"
        module = LanceVSTDataModule(
            dataset_root=dataset_root,
            batch_size=4,
            loader="map",
            num_workers=0,
            ot=False,
            pin_memory=False,
            param_spec_name=ParamSpecName("surge_xt"),
        )
        trainer = Trainer(
            accelerator="cpu",
            devices=2,
            strategy="ddp_spawn",
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
        )

        trainer.fit(_DDPIndexRecorder(source_rows, output_dir), datamodule=module)

        rank_sets = [set(np.load(output_dir / f"rank-{rank}.npy").tolist()) for rank in range(2)]
        assert rank_sets[0]
        assert rank_sets[1]
        assert rank_sets[0] | rank_sets[1] == set(range(len(source_rows)))
        overlap = rank_sets[0] & rank_sets[1]
        assert len(overlap) == num_rows % 2


class _FlowProbe(LightningModule):
    """Minimal Lightning module recording the batches each flow received.

    The datamodule is the system under test; this probe only pins the batch contract (param width,
    conditioning present, matched noise) at every Lightning entrypoint and gives the trainer a real
    optimizable parameter.
    """

    def __init__(self, num_params: int) -> None:
        """Size the probe for one param spec.

        :param num_params: Expected ``params`` width per batch.
        """
        super().__init__()
        self.num_params = num_params
        self.head = torch.nn.Linear(num_params, 1)
        self.flows_seen: set[str] = set()

    def _check_batch(self, batch: dict[str, torch.Tensor | None], flow: str) -> torch.Tensor:
        """Assert one batch carries the model contract and return a loss.

        :param batch: Model batch as produced by ``prepare_batch``.
        :param flow: Lightning flow name being recorded.
        :returns: Scalar loss over the params head.
        """
        params = batch["params"]
        noise = batch["noise"]
        mel = batch["mel_spec"]
        assert params is not None and params.shape[1] == self.num_params
        assert noise is not None and noise.shape == params.shape
        assert mel is not None and mel.shape[0] == params.shape[0]
        self.flows_seen.add(flow)
        return self.head(params).mean()

    def training_step(self, batch: dict[str, torch.Tensor | None], batch_idx: int) -> torch.Tensor:
        """Record the train flow.

        :param batch: Model batch as produced by ``prepare_batch``.
        :param batch_idx: Lightning batch index; unused.
        :returns: Scalar loss.
        """
        return self._check_batch(batch, "fit")

    def validation_step(self, batch: dict[str, torch.Tensor | None], batch_idx: int) -> None:
        """Record the validation flow.

        :param batch: Model batch as produced by ``prepare_batch``.
        :param batch_idx: Lightning batch index; unused.
        """
        self._check_batch(batch, "validate")

    def test_step(self, batch: dict[str, torch.Tensor | None], batch_idx: int) -> None:
        """Record the test flow.

        :param batch: Model batch as produced by ``prepare_batch``.
        :param batch_idx: Lightning batch index; unused.
        """
        self._check_batch(batch, "test")

    def predict_step(self, batch: dict[str, torch.Tensor | None], batch_idx: int) -> torch.Tensor:
        """Record the predict flow; predict batches must also carry audio.

        :param batch: Model batch as produced by ``prepare_batch``.
        :param batch_idx: Lightning batch index; unused.
        :returns: The params tensor, echoing the oracle predict contract.
        """
        assert batch["audio"] is not None
        self._check_batch(batch, "predict")
        params = batch["params"]
        assert params is not None
        return params

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Give the trainer a real optimizer over the probe head.

        :returns: SGD over the head parameters.
        """
        return torch.optim.SGD(self.parameters(), lr=1e-3)


class TestTrainerFlowsAcrossParamSpecs:
    """Fit/validate/test/predict through a real Trainer, one run per registered param spec."""

    @pytest.mark.parametrize("param_spec_name", sorted(param_specs))
    def test_all_trainer_flows_run_on_map_loader(
        self, tmp_path: Path, param_spec_name: str
    ) -> None:
        """Every Lightning flow round-trips map-loader batches at the spec's width.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param param_spec_name: Registry key of the param spec under test.
        """
        num_params = len(param_specs[param_spec_name])
        root = tmp_path / "data"
        root.mkdir()
        for seed, split in enumerate(("train", "val", "test")):
            write_seeded_lance_shard(
                root / f"{split}.lance", num_rows=4, num_params=num_params, seed=seed
            )
        write_mel_stats(root)

        with _set_up_map_module(
            dataset_root=root,
            batch_size=2,
            ot=True,
            param_spec_name=ParamSpecName(param_spec_name),
        ) as module:
            probe = _FlowProbe(num_params)
            trainer = Trainer(
                accelerator="cpu",
                max_epochs=1,
                limit_train_batches=1,
                limit_val_batches=1,
                limit_test_batches=1,
                limit_predict_batches=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            trainer.fit(probe, datamodule=module)
            trainer.test(probe, datamodule=module, verbose=False)
            trainer.predict(probe, datamodule=module)

        assert probe.flows_seen == {"fit", "validate", "test", "predict"}


class TestLegacyMapParity:
    """Cross-loader parity: the map path must reproduce legacy batch values."""

    @pytest.mark.parametrize("split", ["val", "test", "predict"])
    def test_map_loader_matches_legacy_batches_on_eval_splits(
        self, dataset_root: Path, split: str
    ) -> None:
        """Same root and config: legacy and map eval epochs match batch-for-batch.

        Controls isolate loader mechanics: ``ot=False`` (OT permutes rows), eval
        splits (no shuffle), and ``batch_size`` dividing each split evenly
        (legacy floor-divides ragged tails away; map keeps them). Noise generators
        are seeded independently per path by design, so noise is compared on shape only.

        :param dataset_root: Fixture-provided dataset-root directory.
        :param split: Eval split whose dataloader epoch is compared.
        """
        epochs: dict[str, list[dict[str, torch.Tensor | None]]] = {}
        for loader in ("legacy", "map"):
            with _set_up_map_module(
                dataset_root=dataset_root, batch_size=3, ot=False, loader=loader
            ) as module:
                epochs[loader] = list(getattr(module, f"{split}_dataloader")())

        for legacy_batch, map_batch in zip(epochs["legacy"], epochs["map"], strict=True):
            assert legacy_batch.keys() == map_batch.keys()
            for key in legacy_batch:
                legacy_value, map_value = legacy_batch[key], map_batch[key]
                if key == "noise":
                    assert _unwrap(legacy_value).shape == _unwrap(map_value).shape
                elif legacy_value is None or map_value is None:
                    assert legacy_value is None and map_value is None, key
                else:
                    torch.testing.assert_close(legacy_value, map_value, msg=key)
