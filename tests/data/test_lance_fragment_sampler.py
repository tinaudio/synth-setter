"""Behavioral tests for the opt-in Lance fragment-sampler train path (#2251)."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

import lance
import numpy as np
import pytest
import torch

from synth_setter.data.lance_datamodule import LanceVSTDataModule
from synth_setter.data.lance_torch import LanceMapDataset
from synth_setter.param_spec_name import ParamSpecName
from tests.helpers.lance_fixtures import (
    NUM_PARAMS,
    make_shard_columns,
    shard_record_batch,
    write_mel_stats,
    write_seeded_lance_shard,
)

_TRAIN_ROWS = 24
_ROWS_PER_FRAGMENT = 4


def _write_fragmented_shard(
    path: Path,
    num_rows: int,
    *,
    seed: int,
    mel_fill: float | None = None,
) -> dict[str, np.ndarray]:
    """Write a multi-fragment Lance shard by capping rows per data file.

    :param path: Output ``.lance`` dataset directory.
    :param num_rows: Total rows across all fragments.
    :param seed: Seed for the per-row arrays.
    :param mel_fill: When set, fill ``mel_spec`` with this constant so
        normalization is exactly checkable.
    :returns: The column arrays that were written.
    """
    columns = make_shard_columns(num_rows, seed=seed)
    if mel_fill is not None:
        columns["mel_spec"] = np.full_like(columns["mel_spec"], mel_fill)
    batch = shard_record_batch(columns)
    lance.write_dataset(
        batch,
        str(path),
        schema=batch.schema,
        max_rows_per_file=_ROWS_PER_FRAGMENT,
        max_rows_per_group=_ROWS_PER_FRAGMENT,
    )
    return columns


def _write_eval_splits(root: Path, *, mel_fill: float | None = None) -> None:
    """Write the ``val``/``test`` splits every datamodule setup requires.

    :param root: Dataset root directory.
    :param mel_fill: Forwarded to the shard writer for constant-mel roots.
    """
    write_seeded_lance_shard(root / "val.lance", num_rows=6, seed=2, mel_fill=mel_fill)
    write_seeded_lance_shard(root / "test.lance", num_rows=6, seed=3, mel_fill=mel_fill)


@pytest.fixture
def fragmented_root(tmp_path: Path) -> Path:
    """Build a dataset root whose train split spans multiple Lance fragments.

    :param tmp_path: Per-test tmpdir.
    :returns: Path to the populated dataset root directory.
    """
    root = tmp_path / "data"
    root.mkdir()
    _write_fragmented_shard(root / "train.lance", _TRAIN_ROWS, seed=1)
    _write_eval_splits(root)
    write_mel_stats(root)
    return root


@contextlib.contextmanager
def _set_up_module(**kwargs: object) -> Iterator[LanceVSTDataModule]:
    """Construct, set up, yield, and tear down a fragment-sampler datamodule.

    :param \\*\\*kwargs: Forwarded to ``LanceVSTDataModule``; cheap loader
        defaults and the fragment-sampler flag are pre-set.
    :yields: The set-up datamodule for assertion work inside the ``with`` block.
    :ytype: LanceVSTDataModule
    """
    kwargs.setdefault("use_fragment_sampler", True)
    kwargs.setdefault("num_workers", 0)
    kwargs.setdefault("pin_memory", False)
    kwargs.setdefault("batch_size", 4)
    kwargs.setdefault("ot", False)
    kwargs.setdefault("param_spec_name", ParamSpecName("surge_xt"))
    module = LanceVSTDataModule(**kwargs)  # type: ignore[arg-type]
    module.setup()
    try:
        yield module
    finally:
        module.teardown()


def _params_epoch(loader: torch.utils.data.DataLoader) -> np.ndarray:
    """Concatenate ``params`` across one full epoch in iteration order.

    :param loader: Loader whose epoch is materialized.
    :returns: ``(total_rows, num_params)`` array.
    """
    parts = [batch["params"] for batch in loader]
    assert all(part is not None for part in parts)
    return torch.cat(parts).numpy()


def _sorted_rows(rows: np.ndarray) -> np.ndarray:
    """Return ``rows`` sorted lexicographically for order-free comparison.

    :param rows: ``(num_rows, num_params)`` array.
    :returns: The same rows in canonical order.
    """
    return rows[np.lexsort(rows.T[::-1])]


class TestFragmentSamplerTrainPath:
    """Flag-on train loader semantics against the map-path contract."""

    def test_fragment_sampler_on_epoch_covers_source_rows_rescaled(
        self, fragmented_root: Path
    ) -> None:
        """One flag-on epoch yields exactly the stored rows mapped to ``[-1, 1]``.

        :param fragmented_root: Fixture root with a multi-fragment train split.
        """
        source = make_shard_columns(_TRAIN_ROWS, seed=1)["param_array"] * 2 - 1
        with _set_up_module(dataset_root=fragmented_root) as module:
            epoch = _params_epoch(module.train_dataloader())
        assert epoch.shape == (_TRAIN_ROWS, NUM_PARAMS)
        np.testing.assert_allclose(_sorted_rows(epoch), _sorted_rows(source), rtol=1e-6)

    def test_fragment_sampler_workers_partition_fragments_without_duplicate_rows(
        self, fragmented_root: Path
    ) -> None:
        """Spawn workers divide fragment reads while preserving full epoch coverage.

        :param fragmented_root: Fixture root with a multi-fragment train split.
        """
        source = make_shard_columns(_TRAIN_ROWS, seed=1)["param_array"] * 2 - 1
        with _set_up_module(
            dataset_root=fragmented_root,
            num_workers=2,
            ot=True,
            prefetch_factor=3,
        ) as module:
            loader = module.train_dataloader()
            assert loader.num_workers == 2
            assert loader.prefetch_factor == 3
            epoch = _params_epoch(loader)
        assert epoch.shape == (_TRAIN_ROWS, NUM_PARAMS)
        np.testing.assert_allclose(_sorted_rows(epoch), _sorted_rows(source), rtol=1e-6)

    def test_fragment_sampler_on_batch_matches_map_path_contract(
        self, tmp_path: Path
    ) -> None:
        """Flag-on batches carry the map path's keys, dtypes, and normalization.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        root = tmp_path / "data"
        root.mkdir()
        _write_fragmented_shard(root / "train.lance", 8, seed=1, mel_fill=3.0)
        _write_eval_splits(root, mel_fill=3.0)
        write_mel_stats(root, mean=1.0, std=2.0)
        with _set_up_module(dataset_root=root) as module:
            fragment_batch = next(iter(module.train_dataloader()))
        with _set_up_module(dataset_root=root, use_fragment_sampler=False) as module:
            map_batch = next(iter(module.train_dataloader()))
        assert set(fragment_batch) == set(map_batch)
        assert fragment_batch["audio"] is None
        assert fragment_batch["m2l"] is None
        for key in ("mel_spec", "params", "noise"):
            tensor = fragment_batch[key]
            assert tensor is not None and tensor.dtype == torch.float32, key
        mel = fragment_batch["mel_spec"]
        assert mel is not None
        assert torch.allclose(mel, torch.full_like(mel, 1.0))

    def test_fragment_sampler_on_ot_true_permutes_rows_bijectively(
        self, fragmented_root: Path
    ) -> None:
        """OT on the fragment path only reorders rows within each epoch.

        :param fragmented_root: Fixture root with a multi-fragment train split.
        """
        source = make_shard_columns(_TRAIN_ROWS, seed=1)["param_array"] * 2 - 1
        with _set_up_module(dataset_root=fragmented_root, ot=True) as module:
            epoch = _params_epoch(module.train_dataloader())
        np.testing.assert_allclose(_sorted_rows(epoch), _sorted_rows(source), rtol=1e-6)


class TestFragmentSamplerEpochShuffle:
    """Deterministic per-epoch fragment-order randomization."""

    def _two_epochs(self, root: Path) -> tuple[np.ndarray, np.ndarray]:
        """Materialize two consecutive epochs from one train loader.

        :param root: Dataset root with a multi-fragment train split.
        :returns: Params for epoch 0 and epoch 1 in iteration order.
        """
        torch.manual_seed(1234)
        with _set_up_module(dataset_root=root) as module:
            loader = module.train_dataloader()
            return _params_epoch(loader), _params_epoch(loader)

    def test_fragment_sampler_consecutive_epochs_yield_different_order(
        self, fragmented_root: Path
    ) -> None:
        """Re-iterating the train loader reshuffles the fragment order.

        :param fragmented_root: Fixture root with a multi-fragment train split.
        """
        first, second = self._two_epochs(fragmented_root)
        assert not np.array_equal(first, second)
        np.testing.assert_allclose(_sorted_rows(first), _sorted_rows(second), rtol=1e-6)

    @pytest.mark.parametrize("persistent_workers", [False, True])
    def test_fragment_sampler_workers_reshuffle_next_epoch(
        self, fragmented_root: Path, *, persistent_workers: bool
    ) -> None:
        """Worker lifecycles receive a new shared fragment order each epoch.

        :param fragmented_root: Fixture root with a multi-fragment train split.
        :param persistent_workers: Whether workers survive between epoch iterators.
        """
        with _set_up_module(
            dataset_root=fragmented_root,
            num_workers=2,
            persistent_workers=persistent_workers,
        ) as module:
            loader = module.train_dataloader()
            first = _params_epoch(loader)
            second = _params_epoch(loader)
        assert not np.array_equal(first, second)
        np.testing.assert_allclose(_sorted_rows(first), _sorted_rows(second), rtol=1e-6)

    def test_fragment_sampler_same_global_seed_reproduces_epoch_orders(
        self, fragmented_root: Path
    ) -> None:
        """The same global seed reproduces every epoch's exact row order.

        :param fragmented_root: Fixture root with a multi-fragment train split.
        """
        first_run = self._two_epochs(fragmented_root)
        second_run = self._two_epochs(fragmented_root)
        np.testing.assert_array_equal(first_run[0], second_run[0])
        np.testing.assert_array_equal(first_run[1], second_run[1])


class TestFragmentSamplerRouting:
    """Flag routing: default path untouched, incompatible modes rejected."""

    def test_fragment_sampler_off_keeps_map_train_loader(self, fragmented_root: Path) -> None:
        """The default flag leaves the sample-indexed map train loader in place.

        :param fragmented_root: Fixture root with a multi-fragment train split.
        """
        with _set_up_module(
            dataset_root=fragmented_root, use_fragment_sampler=False
        ) as module:
            loader = module.train_dataloader()
        assert isinstance(loader.dataset, LanceMapDataset)
        assert isinstance(loader.sampler, torch.utils.data.RandomSampler)
        assert loader.batch_size == 4

    def test_fragment_sampler_on_leaves_val_loader_map_style(
        self, fragmented_root: Path
    ) -> None:
        """Flag-on modules still serve val through the ordered map path.

        :param fragmented_root: Fixture root with a multi-fragment train split.
        """
        source = make_shard_columns(6, seed=2)["param_array"] * 2 - 1
        with _set_up_module(dataset_root=fragmented_root) as module:
            loader = module.val_dataloader()
            assert isinstance(loader.dataset, LanceMapDataset)
            np.testing.assert_allclose(_params_epoch(loader), source, rtol=1e-6)

    def test_fragment_sampler_with_fake_mode_raises_value_error(self) -> None:
        """Fake mode cannot silently bypass the fragment-sampler request."""
        with pytest.raises(ValueError, match="use_fragment_sampler"):
            LanceVSTDataModule(
                dataset_root="unused",
                use_fragment_sampler=True,
                fake=True,
                param_spec_name=ParamSpecName("surge_xt"),
            )

    def test_fragment_sampler_with_repeat_first_batch_raises_value_error(self) -> None:
        """Repeat-first-batch mode cannot silently bypass the fragment-sampler request."""
        with pytest.raises(ValueError, match="use_fragment_sampler"):
            LanceVSTDataModule(
                dataset_root="unused",
                use_fragment_sampler=True,
                repeat_first_batch=True,
                param_spec_name=ParamSpecName("surge_xt"),
            )
