"""End-to-end tests for the native ``lance.torch`` dataloaders.

Every test drives a real Lance dataset written through the pipeline writer
(:func:`synth_setter.pipeline.data.lance_shard.write_lance_dataset`) — no
fakes or mocks anywhere in this module.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

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
)
from synth_setter.pipeline.data.lance_shard import write_lance_dataset
from tests.helpers.lance_fixtures import write_lance_shard
from tests.helpers.lance_torch_datasets import (
    NUM_PARAMS,
    ROWS,
    write_random_lance_dataset,
)


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


def _assert_ranks_partition_param_rows(
    per_rank: list[np.ndarray], arrays: dict[str, np.ndarray]
) -> None:
    """Assert two rank shards are equal-sized, disjoint, and cover every row.

    :param per_rank: ``param_array`` rows collected by each of the two ranks.
    :param arrays: Source arrays the dataset was written from.
    """
    assert per_rank[0].shape == per_rank[1].shape == (16, NUM_PARAMS)
    union = np.concatenate(per_rank)
    np.testing.assert_array_equal(np.sort(union, axis=0), np.sort(arrays["param_array"], axis=0))


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
    """Assert a 10-row dataset yields full-then-ragged batches at ``batch_size=8``.

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

        :param lance_dataset: Module-shared dataset; the source arrays are the round-trip ground
            truth.
        """
        dest, arrays = lance_dataset
        loader = lance_map_dataloader(dest, batch_size=8, shuffle=False)

        batches = list(loader)

        assert len(batches) == 4
        _assert_first_batch_matches_schema(batches[0])
        for field, source in arrays.items():
            np.testing.assert_array_equal(_concat_batches(batches, field), source)

    def test_len_of_loader_dataset_matches_row_count(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """The map-style dataset reports the dataset's row count.

        :param lance_dataset: Module-shared dataset; the source arrays are the round-trip ground
            truth.
        """
        dest, _ = lance_dataset
        loader = lance_map_dataloader(dest, batch_size=8)

        assert len(loader.dataset) == ROWS  # type: ignore[arg-type]

    def test_columns_projection_returns_only_requested_columns(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """Column projection restricts batches to the requested columns.

        :param lance_dataset: Module-shared dataset; the source arrays are the round-trip ground
            truth.
        """
        dest, _ = lance_dataset
        loader = lance_map_dataloader(dest, batch_size=8, columns=["param_array"])

        batch = next(iter(loader))

        assert set(batch.keys()) == {"param_array"}

    def test_shuffle_covers_all_rows_exactly_once(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """Shuffled iteration is a permutation: every written row appears once.

        :param lance_dataset: Module-shared dataset; the source arrays are the round-trip ground truth.
        """
        dest, arrays = lance_dataset
        loader = lance_map_dataloader(dest, batch_size=8, shuffle=True)

        rows = _concat_batches(list(loader), "param_array")

        assert rows.shape == arrays["param_array"].shape
        np.testing.assert_array_equal(
            np.sort(rows, axis=0), np.sort(arrays["param_array"], axis=0)
        )

    def test_batch_tensors_are_writable(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """Batches own their memory: in-place writes must not hit Arrow's buffers.

        :param lance_dataset: Module-shared dataset; the source arrays are the round-trip ground truth.
        """
        dest, _ = lance_dataset

        _assert_first_batch_owns_writable_memory(lance_map_dataloader(dest, batch_size=8))

    def test_single_item_indexing_returns_row_tensors(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """``dataset[i]`` returns one row as per-column tensors matching the source.

        :param lance_dataset: Module-shared dataset; the source arrays are the round-trip ground
            truth.
        """
        dest, arrays = lance_dataset

        item = LanceMapDataset(dest)[3]

        assert item["mel_spec"].shape == (2, 128, 3)
        np.testing.assert_array_equal(item["param_array"].numpy(), arrays["param_array"][3])

    def test_short_final_batch_preserves_all_rows(self, tmp_path: Path) -> None:
        """A row count not divisible by ``batch_size`` yields a ragged final batch.

        :param tmp_path: Scratch dir for the 10-row dataset.
        """
        _assert_short_final_batch(
            lambda dest: lance_map_dataloader(dest, batch_size=8, shuffle=False),
            tmp_path / "short.lance",
        )

    @pytest.mark.slow
    def test_spawn_workers_cover_all_rows(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """Multiprocessing workers (spawn) read the full dataset without fork hangs.

        :param lance_dataset: Module-shared dataset; the source arrays are the round-trip ground
            truth.
        """
        dest, arrays = lance_dataset
        loader = lance_map_dataloader(
            dest, batch_size=8, num_workers=2, columns=["param_array"], shuffle=False
        )

        rows = _concat_batches(list(loader), "param_array")

        np.testing.assert_array_equal(rows, arrays["param_array"])


class TestIterableDataloader:
    """Behavior of ``lance_iterable_dataloader`` over a real local dataset."""

    def test_batches_preserve_shapes_dtypes_and_values(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """Scan order round-trips the written tensors with schema inner shapes intact.

        :param lance_dataset: Module-shared dataset; the source arrays are the round-trip ground
            truth.
        """
        dest, arrays = lance_dataset
        loader = lance_iterable_dataloader(dest, batch_size=8)

        batches = list(loader)

        assert len(batches) == 4
        _assert_first_batch_matches_schema(batches[0])
        for field, source in arrays.items():
            np.testing.assert_array_equal(_concat_batches(batches, field), source)

    def test_columns_projection_returns_only_requested_columns(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """Column projection restricts batches to the requested columns.

        :param lance_dataset: Module-shared dataset; the source arrays are the round-trip ground
            truth.
        """
        dest, _ = lance_dataset
        loader = lance_iterable_dataloader(dest, batch_size=8, columns=["mel_spec"])

        batch = next(iter(loader))

        assert set(batch.keys()) == {"mel_spec"}

    def test_batch_tensors_are_writable(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """Batches own their memory: in-place writes must not hit Arrow's buffers.

        :param lance_dataset: Module-shared dataset; the source arrays are the round-trip ground truth.
        """
        dest, _ = lance_dataset

        _assert_first_batch_owns_writable_memory(lance_iterable_dataloader(dest, batch_size=8))

    def test_short_final_batch_preserves_all_rows(self, tmp_path: Path) -> None:
        """A row count not divisible by ``batch_size`` yields a ragged final batch.

        :param tmp_path: Scratch dir for the 10-row dataset.
        """
        _assert_short_final_batch(
            lambda dest: lance_iterable_dataloader(dest, batch_size=8),
            tmp_path / "short.lance",
        )

    def test_rank_without_world_size_raises_value_error(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """Passing only one of ``rank``/``world_size`` is rejected up front.

        :param lance_dataset: Module-shared dataset; the source arrays are the round-trip ground
            truth.
        """
        dest, _ = lance_dataset

        with pytest.raises(ValueError, match="rank and world_size"):
            lance_iterable_dataloader(dest, batch_size=8, rank=0)

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

        loader = lance_iterable_dataloader(dest, batch_size=8)

        read = _concat_batches(list(loader), "clap")
        np.testing.assert_array_equal(read, values)

    def test_rank_shards_are_disjoint_and_complete(
        self, lance_dataset: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        """Explicit ``rank``/``world_size`` split the dataset without overlap or loss.

        :param lance_dataset: Module-shared dataset; the source arrays are the round-trip ground
            truth.
        """
        dest, arrays = lance_dataset
        per_rank = [
            _concat_batches(
                list(
                    lance_iterable_dataloader(
                        dest, batch_size=8, columns=["param_array"], rank=rank, world_size=2
                    )
                ),
                "param_array",
            )
            for rank in (0, 1)
        ]

        _assert_ranks_partition_param_rows(per_rank, arrays)


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
        loader = lance_iterable_dataloader(dataset_dir, batch_size=8, columns=["param_array"])
        rows = _concat_batches(list(loader), "param_array")
        np.save(Path(out_dir) / f"rank{rank}.npy", rows)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_iterable_dataloader_autodetects_real_torch_distributed_shards(
    lance_dataset: tuple[Path, dict[str, np.ndarray]], tmp_path: Path
) -> None:
    """Under a real 2-process gloo group each rank reads a disjoint half of the rows.

    :param lance_dataset: Module-shared dataset; the source arrays are the round-trip ground truth.
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
