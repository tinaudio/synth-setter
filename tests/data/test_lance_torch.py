"""End-to-end tests for the native ``lance.torch`` dataloaders.

Every test drives a real Lance dataset written through the pipeline writer
(:func:`synth_setter.pipeline.data.lance_shard.write_lance_dataset`) — no
fakes or mocks anywhere in this module.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn
from torch.utils.data import DataLoader

from synth_setter.data.lance_torch import (
    LanceMapDataset,
    _batch_to_shaped_tensors,
    lance_iterable_dataloader,
    lance_map_dataloader,
    map_dataloader_over,
)
from synth_setter.pipeline.data.lance_shard import write_lance_dataset
from tests.helpers.lance_fixtures import write_lance_shard
from tests.helpers.lance_torch_datasets import (
    NUM_PARAMS,
    ROWS,
    write_random_lance_dataset,
)

BATCH_SIZE = 8


class _TakeRecorder:
    """Record projected reads while delegating to a real Lance dataset."""

    def __init__(self, dataset: lance.LanceDataset) -> None:
        """Store the dataset that serves recorded reads.

        :param dataset: Real local Lance dataset.
        """
        self.dataset = dataset
        self.calls: list[tuple[list[int], list[str] | None]] = []

    def take(self, indices: list[int], *, columns: list[str] | None) -> pa.Table:
        """Record and delegate one projected ``take``.

        :param indices: Requested row indices.
        :param columns: Requested projection.
        :return: Lance result table.
        """
        self.calls.append((indices, columns))
        return self.dataset.take(indices, columns=columns)


@pytest.fixture(scope="module")
def lance_dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, np.ndarray]]:
    """Write one shared read-only Lance dataset for the module's loaders.

    :param tmp_path_factory: Pytest factory providing the module-scoped dir.
    :returns: ``(dataset_dir, source_arrays)`` pair.
    """
    dest = tmp_path_factory.mktemp("lance_torch") / "train.lance"
    arrays = write_random_lance_dataset(dest)
    return dest, arrays


def _concat_batches(batches: list[dict[str, torch.Tensor]], column: str) -> np.ndarray:
    """Concatenate one column across loader batches into a single ndarray.

    :param batches: Batches as yielded by either dataloader.
    :param column: Column to extract.
    :returns: ``(total_rows, *inner_shape)`` array in iteration order.
    """
    return torch.cat([batch[column] for batch in batches]).numpy()


def _assert_first_batch_matches_schema(first: dict[str, torch.Tensor]) -> None:
    """Assert one batch carries the written schema's shapes and dtypes.

    :param first: First batch yielded by either dataloader (batch size 8).
    """
    assert first["audio"].shape == (8, 2, 10)
    assert first["audio"].dtype == torch.float16
    assert first["mel_spec"].shape == (8, 2, 128, 3)
    assert first["mel_spec"].dtype == torch.float32
    assert first["param_array"].shape == (8, NUM_PARAMS)
    assert first["param_array"].dtype == torch.float32


def _sort_rows(rows: np.ndarray) -> np.ndarray:
    """Sort 2-D array rows lexicographically, keeping each row intact.

    ``np.sort(..., axis=0)`` would sort each column independently and miss
    cross-row scrambling; whole-row ordering makes multiset-of-rows
    comparisons exact.

    :param rows: ``(N, width)`` array.
    :returns: The same rows in lexicographic order.
    """
    return rows[np.lexsort(rows.T[::-1])]


def _assert_ranks_partition_param_rows(
    per_rank: list[np.ndarray], arrays: dict[str, np.ndarray]
) -> None:
    """Assert two rank shards are equal-sized, disjoint, and cover every row.

    :param per_rank: ``param_array`` rows collected by each of the two ranks.
    :param arrays: Source arrays the dataset was written from.
    """
    assert per_rank[0].shape == per_rank[1].shape == (16, NUM_PARAMS)
    union = np.concatenate(per_rank)
    np.testing.assert_array_equal(_sort_rows(union), _sort_rows(arrays["param_array"]))


def _assert_first_batch_owns_writable_memory(loader: DataLoader) -> None:
    """Assert a batch is safely writable — no tensor aliases Arrow's read-only buffers.

    ``torch.from_numpy`` over a non-writable array emits a ``UserWarning`` (and
    writing to the result is undefined behavior); escalating warnings to errors
    while the batch is built pins the defensive copy in the Arrow decode.

    :param loader: Loader whose first batch is checked.
    """
    with warnings.catch_warnings():
        # torch.from_numpy's exact wording for a read-only source array.
        warnings.filterwarnings("error", message=".*not writable.*", category=UserWarning)
        batch = next(iter(loader))
    batch["mel_spec"] += 1.0


def _assert_short_final_batch(loader_factory: Callable[[Path], DataLoader], dest: Path) -> None:
    """Assert a 10-row dataset yields a full batch then a ragged 2-row batch.

    :param loader_factory: Builds the loader under test for a dataset directory.
    :param dest: Directory to write the 10-row single-column dataset into.
    """
    values = np.arange(10 * NUM_PARAMS, dtype=np.float32).reshape(10, NUM_PARAMS)
    write_lance_shard(dest, {"param_array": values})

    batches = list(loader_factory(dest))

    assert [len(batch["param_array"]) for batch in batches] == [8, 2]
    np.testing.assert_array_equal(_concat_batches(batches, "param_array"), values)


class TestMapDataloader:
    """Behavior of ``lance_map_dataloader`` over a real local dataset."""

    def test_batches_unshuffled_preserve_shapes_dtypes_and_values(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """Un-shuffled iteration returns exactly the written tensors, shaped per schema.

        :param lance_dataset: Module-shared dataset; source arrays are the ground truth.
        """
        dest, arrays = lance_dataset
        loader = lance_map_dataloader(dest, batch_size=BATCH_SIZE, shuffle=False)

        batches = list(loader)

        assert len(batches) == 4
        _assert_first_batch_matches_schema(batches[0])
        for field, source in arrays.items():
            np.testing.assert_array_equal(_concat_batches(batches, field), source)

    def test_len_of_loader_dataset_matches_row_count(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """The map-style dataset reports the dataset's row count.

        :param lance_dataset: Module-shared dataset; source arrays are the ground truth.
        """
        dest, _ = lance_dataset
        loader = lance_map_dataloader(dest, batch_size=BATCH_SIZE)

        assert len(loader.dataset) == ROWS  # type: ignore[arg-type]

    def test_getitems_preserves_projected_unsorted_duplicate_rows_in_one_take(
        self,
        lance_dataset: tuple[Path, dict[str, np.ndarray]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One projected read preserves duplicate, out-of-order source values.

        :param lance_dataset: Module-shared dataset; source arrays are the ground truth.
        :param monkeypatch: Fixture installing a recorder over the real dataset.
        """
        dest, arrays = lance_dataset
        columns = list(arrays)
        indices = [3, 0, 7, 7, 1]
        dataset = LanceMapDataset(dest, columns=columns)
        recorder = _TakeRecorder(lance.dataset(dest))
        monkeypatch.setattr(dataset, "_ds", recorder)
        monkeypatch.setattr(dataset, "_opening_pid", os.getpid())
        batch = dataset.__getitems__(indices)
        for column in columns:
            expected = arrays[column][indices]
            np.testing.assert_array_equal(batch[column].numpy(), expected)
            assert batch[column].numpy().dtype == expected.dtype
            assert batch[column].shape == expected.shape

        assert recorder.calls == [(indices, columns)]

    def test_getitems_reopens_handle_after_process_change(
        self,
        lance_dataset: tuple[Path, dict[str, np.ndarray]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A forked worker never reuses the parent process's Lance handle.

        :param lance_dataset: Module-shared dataset; source arrays are the ground truth.
        :param monkeypatch: Fixture simulating a worker PID and recording the reopen.
        """
        dest, _ = lance_dataset
        dataset = LanceMapDataset(dest, columns=["param_array"])
        inherited = _TakeRecorder(lance.dataset(dest))
        reopened = _TakeRecorder(lance.dataset(dest))
        monkeypatch.setattr(dataset, "_ds", inherited)
        monkeypatch.setattr(dataset, "_opening_pid", 100)
        monkeypatch.setattr("synth_setter.data.lance_torch.os.getpid", lambda: 101)
        monkeypatch.setattr("synth_setter.data.lance_torch.lance.dataset", lambda *args, **kwargs: reopened)

        dataset.__getitems__([0, 1])

        assert inherited.calls == []
        assert reopened.calls == [([0, 1], ["param_array"])]
        assert dataset._opening_pid == 101

    def test_columns_projection_returns_only_requested_columns(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """Column projection restricts batches to the requested columns.

        :param lance_dataset: Module-shared dataset; source arrays are the ground truth.
        """
        dest, _ = lance_dataset
        loader = lance_map_dataloader(dest, batch_size=BATCH_SIZE, columns=["param_array"])

        batch = next(iter(loader))

        assert set(batch.keys()) == {"param_array"}

    def test_shuffle_covers_all_rows_exactly_once(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """Shuffled iteration is a permutation: every written row appears once.

        :param lance_dataset: Module-shared dataset; source arrays are the ground truth.
        """
        dest, arrays = lance_dataset
        loader = lance_map_dataloader(dest, batch_size=BATCH_SIZE, shuffle=True)

        rows = _concat_batches(list(loader), "param_array")

        assert rows.shape == arrays["param_array"].shape
        np.testing.assert_array_equal(_sort_rows(rows), _sort_rows(arrays["param_array"]))

    def test_batch_tensors_are_writable(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """Batches own their memory: in-place writes must not hit Arrow's buffers.

        :param lance_dataset: Module-shared dataset; source arrays are the ground truth.
        """
        dest, _ = lance_dataset

        _assert_first_batch_owns_writable_memory(lance_map_dataloader(dest, batch_size=BATCH_SIZE))

    def test_single_item_indexing_returns_row_tensors(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """``dataset[i]`` returns one row as per-column tensors matching the source.

        :param lance_dataset: Module-shared dataset; source arrays are the ground truth.
        """
        dest, arrays = lance_dataset

        item = LanceMapDataset(dest)[3]

        assert item["mel_spec"].shape == (2, 128, 3)
        np.testing.assert_array_equal(item["param_array"].numpy(), arrays["param_array"][3])

    def test_persistent_workers_without_workers_is_effectively_disabled(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """In-process loaders safely disable configured worker persistence.

        :param lance_dataset: Module-shared dataset used to construct the loader.
        """
        dest, _ = lance_dataset

        loader = map_dataloader_over(
            LanceMapDataset(dest),
            batch_size=BATCH_SIZE,
            num_workers=0,
            persistent_workers=True,
        )

        assert loader.persistent_workers is False
        assert next(iter(loader))["param_array"].shape == (BATCH_SIZE, NUM_PARAMS)

    def test_prefetch_factor_with_workers_reaches_dataloader(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """A configured prefetch depth reaches loaders that own worker processes.

        :param lance_dataset: Module-shared dataset used to construct the loader.
        """
        dest, _ = lance_dataset

        loader = map_dataloader_over(
            LanceMapDataset(dest),
            batch_size=BATCH_SIZE,
            num_workers=2,
            prefetch_factor=4,
        )

        assert loader.prefetch_factor == 4

    def test_prefetch_factor_default_none_keeps_pytorch_default(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """Leaving the prefetch depth unset preserves PyTorch's own default.

        :param lance_dataset: Module-shared dataset used to construct the loader.
        """
        dest, _ = lance_dataset

        loader = map_dataloader_over(
            LanceMapDataset(dest), batch_size=BATCH_SIZE, num_workers=2
        )

        plain_default = DataLoader(
            LanceMapDataset(dest), batch_size=BATCH_SIZE, num_workers=2
        )
        assert loader.prefetch_factor == plain_default.prefetch_factor

    def test_prefetch_factor_without_workers_is_effectively_disabled(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """In-process loaders drop the configured prefetch depth (PyTorch forbids it).

        :param lance_dataset: Module-shared dataset used to construct the loader.
        """
        dest, _ = lance_dataset

        loader = map_dataloader_over(
            LanceMapDataset(dest),
            batch_size=BATCH_SIZE,
            num_workers=0,
            prefetch_factor=4,
        )

        assert loader.prefetch_factor is None
        assert next(iter(loader))["param_array"].shape == (BATCH_SIZE, NUM_PARAMS)

    def test_lance_map_dataloader_forwards_prefetch_factor(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """The public factory forwards a configured prefetch depth to its loader.

        :param lance_dataset: Module-shared dataset used to construct the loader.
        """
        dest, _ = lance_dataset

        loader = lance_map_dataloader(
            dest, batch_size=BATCH_SIZE, num_workers=2, prefetch_factor=4
        )

        assert loader.prefetch_factor == 4

    @pytest.mark.dataloader_multiprocess
    @pytest.mark.xdist_group(name="dataloader-multiprocess")
    @pytest.mark.slow
    def test_prefetch_factor_with_spawn_workers_delivers_batches(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """Spawn workers deliver every row when a non-default prefetch depth is set.

        :param lance_dataset: Module-shared dataset; source arrays are the ground truth.
        """
        dest, arrays = lance_dataset
        loader = map_dataloader_over(
            LanceMapDataset(dest, columns=["param_array"]),
            batch_size=BATCH_SIZE,
            num_workers=2,
            shuffle=False,
            prefetch_factor=4,
        )

        rows = _concat_batches(list(loader), "param_array")

        np.testing.assert_array_equal(rows, arrays["param_array"])

    def test_short_final_batch_preserves_all_rows(self, tmp_path: Path) -> None:
        """A row count not divisible by ``batch_size`` yields a ragged final batch.

        :param tmp_path: Scratch dir for the 10-row dataset.
        """
        _assert_short_final_batch(
            lambda dest: lance_map_dataloader(dest, batch_size=BATCH_SIZE, shuffle=False),
            tmp_path / "short.lance",
        )

    def test_sampler_overrides_shuffle(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """An explicit sampler controls order even when shuffle is requested.

        :param lance_dataset: Module-shared dataset; source arrays are the ground truth.
        """
        dest, arrays = lance_dataset
        dataset = LanceMapDataset(dest, columns=["param_array"])
        loader = map_dataloader_over(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=torch.utils.data.SequentialSampler(dataset),
            shuffle=True,
        )

        rows = _concat_batches(list(loader), "param_array")

        np.testing.assert_array_equal(rows, arrays["param_array"])

    @pytest.mark.dataloader_multiprocess
    @pytest.mark.xdist_group(name="dataloader-multiprocess")
    @pytest.mark.slow
    def test_spawn_workers_cover_all_rows(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """Multiprocessing workers (spawn) read the full dataset without fork hangs.

        :param lance_dataset: Module-shared dataset; source arrays are the ground truth.
        """
        dest, arrays = lance_dataset
        loader = lance_map_dataloader(
            dest,
            batch_size=BATCH_SIZE,
            num_workers=2,
            columns=["param_array"],
            shuffle=False,
            persistent_workers=True,
        )
        assert loader.persistent_workers

        rows = _concat_batches(list(loader), "param_array")

        np.testing.assert_array_equal(rows, arrays["param_array"])


class TestIterableDataloader:
    """Behavior of ``lance_iterable_dataloader`` over a real local dataset."""

    def test_batches_preserve_shapes_dtypes_and_values(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """Scan order round-trips the written tensors with schema inner shapes intact.

        :param lance_dataset: Module-shared dataset; source arrays are the ground truth.
        """
        dest, arrays = lance_dataset
        loader = lance_iterable_dataloader(dest, batch_size=BATCH_SIZE)

        batches = list(loader)

        assert len(batches) == 4
        _assert_first_batch_matches_schema(batches[0])
        for field, source in arrays.items():
            np.testing.assert_array_equal(_concat_batches(batches, field), source)

    def test_columns_projection_returns_only_requested_columns(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """Column projection restricts batches to the requested columns.

        :param lance_dataset: Module-shared dataset; source arrays are the ground truth.
        """
        dest, _ = lance_dataset
        loader = lance_iterable_dataloader(dest, batch_size=BATCH_SIZE, columns=["mel_spec"])

        batch = next(iter(loader))

        assert set(batch.keys()) == {"mel_spec"}

    def test_batch_tensors_are_writable(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """Batches own their memory: in-place writes must not hit Arrow's buffers.

        :param lance_dataset: Module-shared dataset; source arrays are the ground truth.
        """
        dest, _ = lance_dataset

        _assert_first_batch_owns_writable_memory(
            lance_iterable_dataloader(dest, batch_size=BATCH_SIZE)
        )

    def test_short_final_batch_preserves_all_rows(self, tmp_path: Path) -> None:
        """A row count not divisible by ``batch_size`` yields a ragged final batch.

        :param tmp_path: Scratch dir for the 10-row dataset.
        """
        _assert_short_final_batch(
            lambda dest: lance_iterable_dataloader(dest, batch_size=BATCH_SIZE),
            tmp_path / "short.lance",
        )

    @pytest.mark.parametrize(
        ("rank", "world_size", "match"),
        [
            (0, None, "rank and world_size"),
            (None, 2, "rank and world_size"),
            (2, 2, r"rank must be in \[0, world_size\)"),
            (-1, 2, r"rank must be in \[0, world_size\)"),
        ],
    )
    def test_invalid_rank_world_size_raises_value_error(
        self,
        lance_dataset: tuple[Path, dict[str, np.ndarray]],
        rank: int | None,
        world_size: int | None,
        match: str,
    ) -> None:
        """Asymmetric or out-of-range ``rank``/``world_size`` is rejected up front.

        :param lance_dataset: Module-shared dataset; source arrays are the ground truth.
        :param rank: Shard index to pass, or ``None`` to omit it.
        :param world_size: Shard count to pass, or ``None`` to omit it.
        :param match: Expected ``ValueError`` message fragment.
        """
        dest, _ = lance_dataset

        with pytest.raises(ValueError, match=match):
            lance_iterable_dataloader(
                dest, batch_size=BATCH_SIZE, rank=rank, world_size=world_size
            )

    def test_unsupported_column_type_raises_type_error(self, tmp_path: Path) -> None:
        """A column with no tensor representation (string) fails loudly, not silently.

        :param tmp_path: Scratch dir for the single-column dataset.
        """
        schema = pa.schema([pa.field("label", pa.string(), nullable=False)])
        batch = pa.record_batch([pa.array(["a", "b", "c"])], schema=schema)
        dest = tmp_path / "labels.lance"
        write_lance_dataset(dest, schema, [batch])

        loader = lance_iterable_dataloader(dest, batch_size=2)

        with pytest.raises(TypeError, match="no tensor representation"):
            next(iter(loader))

    def test_fixed_size_list_column_reads_as_2d_tensor(self, tmp_path: Path) -> None:
        """Embedding-style fixed-size-list columns (e.g. ``clap``) load as ``(rows, dim)``.

        :param tmp_path: Scratch dir for the single-column dataset.
        """
        embedding_dim = 4
        values = np.arange(ROWS * embedding_dim, dtype=np.float32).reshape(ROWS, embedding_dim)
        schema = pa.schema(
            [pa.field("clap", pa.list_(pa.float32(), embedding_dim), nullable=False)]
        )
        batch = pa.record_batch(
            [pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), embedding_dim)],
            schema=schema,
        )
        dest = tmp_path / "clap.lance"
        write_lance_dataset(dest, schema, [batch])

        loader = lance_iterable_dataloader(dest, batch_size=BATCH_SIZE)

        read = _concat_batches(list(loader), "clap")
        np.testing.assert_array_equal(read, values)

    def test_rank_shards_are_disjoint_and_complete(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """Explicit ``rank``/``world_size`` split the dataset without overlap or loss.

        :param lance_dataset: Module-shared dataset; source arrays are the ground truth.
        """
        dest, arrays = lance_dataset
        per_rank = [
            _concat_batches(
                list(
                    lance_iterable_dataloader(
                        dest,
                        batch_size=BATCH_SIZE,
                        columns=["param_array"],
                        rank=rank,
                        world_size=2,
                    )
                ),
                "param_array",
            )
            for rank in (0, 1)
        ]

        _assert_ranks_partition_param_rows(per_rank, arrays)


def test_zero_row_dataset_yields_no_batches(tmp_path: Path) -> None:
    """An empty split produces empty loaders, not errors, from both dataloaders.

    :param tmp_path: Scratch dir for the zero-row dataset.
    """
    schema = pa.schema(
        [
            pa.field(
                "param_array", pa.fixed_shape_tensor(pa.float32(), (NUM_PARAMS,)), nullable=False
            )
        ]
    )
    dest = tmp_path / "empty.lance"
    write_lance_dataset(dest, schema, [])

    map_loader = lance_map_dataloader(dest, batch_size=BATCH_SIZE)
    iterable_loader = lance_iterable_dataloader(dest, batch_size=BATCH_SIZE)

    assert len(map_loader.dataset) == 0  # type: ignore[arg-type]
    assert list(map_loader) == []
    assert list(iterable_loader) == []


def test_batch_to_shaped_tensors_preserves_shapes_on_handbuilt_batch() -> None:
    """The conversion keeps per-row tensor shapes and dtypes on a hand-built batch."""
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    batch = pa.record_batch({"mel": pa.FixedShapeTensorArray.from_numpy_ndarray(values)})

    tensors = _batch_to_shaped_tensors(batch)

    assert tensors["mel"].shape == (2, 3, 4)
    assert tensors["mel"].dtype == torch.float32
    np.testing.assert_array_equal(tensors["mel"].numpy(), values)


def test_column_with_nulls_raises_value_error() -> None:
    """A null row in a fixed-size-list column is rejected by name, not misread."""
    column = pa.array([[1.0, 2.0], None], type=pa.list_(pa.float32(), 2))
    batch = pa.record_batch({"clap": column})

    with pytest.raises(ValueError, match="clap"):
        _batch_to_shaped_tensors(batch)


def test_blob_projected_dict_batch_raises_type_error() -> None:
    """Lance hands blob-projected batches to ``to_tensor_fn`` as dicts; reject them loudly.

    Direct call against the ``to_tensor_fn`` contract's dict shape — the only
    way Lance produces one is a blob column projection, which the loaders do
    not support.
    """
    with pytest.raises(TypeError, match="blob columns"):
        _batch_to_shaped_tensors({"audio_mp3": [b"\x00"]})


def _collect_ddp_rank_rows(rank: int, world_size: int, dataset_dir: str, out_dir: str) -> None:
    """Read one DDP rank's shard inside a real ``torch.distributed`` process group.

    Spawned as a subprocess entrypoint by the DDP test below; must stay at
    module top level so the spawn context can pickle it.

    :param rank: This process's rank within the gloo group.
    :param world_size: Total processes in the gloo group.
    :param dataset_dir: Lance dataset directory to read.
    :param out_dir: Directory receiving one ``rank<i>.npy`` result per rank.
    """

    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{out_dir}/ddp_init",
    )
    try:
        loader = lance_iterable_dataloader(
            dataset_dir, batch_size=BATCH_SIZE, columns=["param_array"]
        )
        rows = _concat_batches(list(loader), "param_array")
        np.save(Path(out_dir) / f"rank{rank}.npy", rows)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_iterable_dataloader_autodetects_real_torch_distributed_shards(
    lance_dataset: tuple[Path, dict[str, np.ndarray]], tmp_path: Path
) -> None:
    """Under a real 2-process gloo group each rank reads a disjoint half of the rows.

    :param lance_dataset: Module-shared dataset; source arrays are the ground truth.
    :param tmp_path: Scratch dir for the gloo rendezvous file and rank outputs.
    """
    dest, arrays = lance_dataset
    world_size = 2

    spawn(
        _collect_ddp_rank_rows,
        args=(world_size, str(dest), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    per_rank = [np.load(tmp_path / f"rank{rank}.npy") for rank in range(world_size)]
    _assert_ranks_partition_param_rows(per_rank, arrays)
