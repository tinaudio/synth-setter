"""Native ``lance.torch`` dataloaders over Lance shard/split datasets.

Thin factories over Lance's own PyTorch integration (``LanceDataset``,
``SafeLanceDataset``, ``ShardedBatchSampler``) rather than the batch-indexed
adapter in :mod:`synth_setter.data.lance_datamodule`. Both loaders stream
object storage natively: pass ``storage_options`` (see
:func:`synth_setter.pipeline.r2_io.r2_storage_options`) with an ``s3://`` URI.

Typical usage::

    loader = lance_map_dataloader("data/train.lance", batch_size=128, shuffle=True)
    for batch in loader:  # {"mel_spec": (128, C, 128, F) tensor, ...}
        ...
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import lance
import pyarrow as pa
import torch
from lance.sampler import ShardedBatchSampler
from lance.torch.data import LanceDataset, SafeLanceDataset, get_safe_loader
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _column_to_tensor(array: pa.Array | pa.ChunkedArray, name: str) -> torch.Tensor:
    """Convert one Arrow column to a writable tensor, keeping tensor inner shapes.

    Lance's default conversion flattens fixed-shape tensor columns to
    ``(rows, prod(shape))``; this keeps the schema's per-row shape.

    :param array: Column values for one batch.
    :param name: Column name, for error messages.
    :returns: ``(rows, *inner_shape)`` tensor owning its memory.
    :raises TypeError: The column type has no tensor representation (e.g. blob).
    :raises ValueError: The column contains nulls (pipeline columns are ``nullable=False``).
    """
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if array.null_count:
        raise ValueError(f"column {name!r} contains nulls; expected nullable=False data")
    if isinstance(array.type, pa.FixedShapeTensorType):
        values = array.to_numpy_ndarray()
    elif pa.types.is_fixed_size_list(array.type):
        values = array.flatten().to_numpy().reshape(len(array), array.type.list_size)
    else:
        raise TypeError(f"column {name!r} type {array.type} has no tensor representation")
    # Arrow buffers are read-only; torch.from_numpy over a read-only view is
    # undefined behavior on write, so copy unless numpy already owns the memory.
    return torch.from_numpy(values if values.flags.writeable else values.copy())


def _batch_to_shaped_tensors(
    batch: pa.RecordBatch | dict[str, Any],
    *,
    hf_converter: dict[str, Any] | None = None,
    use_blob_api: bool = False,
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Convert a record batch to named tensors (``to_tensor_fn`` contract).

    :param batch: One scanner batch.
    :param hf_converter: Accepted per the ``to_tensor_fn`` contract; unused.
    :param use_blob_api: Accepted per the ``to_tensor_fn`` contract; unused.
    :param \\*\\*kwargs: Extra keywords Lance's iterator may pass; unused.
    :returns: One writable tensor per column, schema inner shapes preserved.
    :raises TypeError: A blob column was projected (Lance then passes a dict);
        blob columns have no tensor representation here.
    """
    del hf_converter, use_blob_api, kwargs
    if isinstance(batch, dict):
        raise TypeError("blob columns are not supported by the lance_torch dataloaders")
    return {name: _column_to_tensor(batch[name], name) for name in batch.column_names}


def _dataset_options(storage_options: dict[str, str] | None) -> dict[str, dict[str, str]] | None:
    """Wrap ``storage_options`` in the ``lance.dataset`` keyword mapping.

    :param storage_options: Object-store config for a cloud URI; ``None`` local.
    :returns: Keywords for ``lance.dataset``, or ``None`` when local.
    """
    return {"storage_options": storage_options} if storage_options else None


class LanceMapDataset(SafeLanceDataset):
    """Map-style dataset yielding dict-of-tensor items from a Lance dataset.

    Inherits ``SafeLanceDataset``'s worker-safe lazy open (each spawned worker
    reopens its own handle) and swaps the ``to_pylist`` decode for a direct
    Arrow-to-tensor conversion that keeps schema tensor shapes and dtypes.
    """

    def __init__(
        self,
        uri: str | Path,
        *,
        columns: Sequence[str] | None = None,
        storage_options: dict[str, str] | None = None,
    ):
        """Open the dataset lazily for map-style access.

        :param uri: Dataset directory (local path or ``s3://`` URI).
        :param columns: Columns each item carries; ``None`` reads all.
        :param storage_options: Object-store config for a cloud ``uri`` (see
            :func:`synth_setter.pipeline.r2_io.r2_storage_options`); ``None`` local.
        """
        super().__init__(str(uri), dataset_options=_dataset_options(storage_options))
        self._columns = list(columns) if columns is not None else None
        self._opening_pid: int | None = None

    def __getitems__(self, indices: Sequence[int]) -> dict[str, torch.Tensor]:
        """Fetch a batch of rows in one ``take`` call, already collated.

        Returns the whole batch as one column dict rather than a list of
        per-row samples: splitting rows out only for ``default_collate`` to
        re-stack them would copy every column. Pair with
        :func:`_prebatched_collate` (the factory's default ``collate_fn``).

        :param indices: Row indices, in the order the batch should carry them.
        :returns: One ``(len(indices), *inner_shape)`` tensor per column.
        """
        current_pid = os.getpid()
        if self._ds is None or self._opening_pid != current_pid:
            # Forked workers must reopen rather than reuse an inherited Lance handle.
            self._ds = lance.dataset(self.uri, **self.dataset_options)
            self._opening_pid = current_pid
        table = self._ds.take(list(indices), columns=self._columns)
        return {name: _column_to_tensor(table[name], name) for name in table.column_names}

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Fetch one row as a dict of per-row tensors.

        :param idx: Row index.
        :returns: One ``(*inner_shape,)`` tensor per column.
        """
        return {name: rows[0] for name, rows in self.__getitems__([idx]).items()}


def _prebatched_collate(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Pass through a batch ``LanceMapDataset.__getitems__`` already collated.

    :param batch: Column dict built by the dataset's batched fetch.
    :returns: The batch as-is — module-level (not a lambda) so spawn workers can pickle it.
    """
    return batch


def map_dataloader_over(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool | None = None,
    sampler: torch.utils.data.Sampler[int] | Iterable[int] | None = None,
    collate_fn: Callable[[dict[str, torch.Tensor]], object] | None = None,
    pin_memory: bool = False,
    drop_last: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    """Wrap an existing map-style dataset in a (spawn-safe) DataLoader.

    Worker processes use Lance's ``get_safe_loader`` (spawn context — Lance
    datasets are not fork-safe), so ``dataset`` and ``collate_fn`` must be
    picklable when ``num_workers > 0``.

    :param dataset: Map-style dataset whose ``__getitems__`` pre-collates a
        batch (:class:`LanceMapDataset` or a wrapper over one).
    :param batch_size: Rows per yielded batch.
    :param num_workers: DataLoader worker processes; ``0`` loads in-process.
    :param shuffle: Whether to randomize sample order; ignored when ``sampler`` is set.
    :param sampler: Optional source of sample indices.
    :param collate_fn: Optional batch transformation after the projected Lance read.
    :param pin_memory: Whether DataLoader pins tensors before returning them.
    :param drop_last: Whether to discard a shorter final batch.
    :param persistent_workers: Whether worker processes survive across iterator resets.
    :returns: DataLoader over ``dataset``.
    :raises ValueError: If worker persistence is enabled without worker processes.
    """
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    effective_collate = collate_fn or _prebatched_collate
    effective_shuffle = None if sampler is not None else shuffle
    if num_workers == 0:
        # get_safe_loader requires workers; plain DataLoader supports in-process loading.
        # Cast bridges __getitems__' column dict with DataLoader's list-oriented stub.
        typed_collate = cast(Callable[[list[object]], object], effective_collate)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=effective_shuffle,
            sampler=sampler,
            collate_fn=typed_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
        )
    return get_safe_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=effective_shuffle,
        sampler=sampler,
        collate_fn=effective_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )


def lance_map_dataloader(
    uri: str | Path,
    *,
    batch_size: int,
    num_workers: int = 0,
    columns: Sequence[str] | None = None,
    storage_options: dict[str, str] | None = None,
    shuffle: bool | None = None,
    sampler: torch.utils.data.Sampler[int] | Iterable[int] | None = None,
    collate_fn: Callable[[dict[str, torch.Tensor]], object] | None = None,
    pin_memory: bool = False,
    drop_last: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    """Build a map-style DataLoader (random access, shuffling, DDP-samplable).

    :param uri: Dataset directory (local path or ``s3://`` URI).
    :param batch_size: Rows per yielded batch.
    :param num_workers: DataLoader worker processes; ``0`` loads in-process.
    :param columns: Columns each batch carries; ``None`` reads all.
    :param storage_options: Object-store config for a cloud ``uri`` (see
        :func:`synth_setter.pipeline.r2_io.r2_storage_options`); ``None`` local.
    :param shuffle: Whether to randomize sample order; ignored when ``sampler`` is set.
    :param sampler: Optional source of sample indices.
    :param collate_fn: Optional batch transformation after the projected Lance read.
    :param pin_memory: Whether DataLoader pins tensors before returning them.
    :param drop_last: Whether to discard a shorter final batch.
    :param persistent_workers: Whether worker processes survive across iterator resets.
    :returns: DataLoader yielding ``{column: (<=batch_size, *inner_shape) tensor}`` —
        the final batch is shorter when the row count is not divisible by ``batch_size``.
    :raises ValueError: If worker persistence is enabled without worker processes.
    """
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    dataset = LanceMapDataset(uri, columns=columns, storage_options=storage_options)
    logger.info(
        "lance map dataloader: uri=%s rows=%d columns=%s batch_size=%d num_workers=%d",
        uri,
        len(dataset),
        columns,
        batch_size,
        num_workers,
    )
    return map_dataloader_over(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )


def lance_iterable_dataloader(
    uri: str | Path,
    *,
    batch_size: int,
    columns: Sequence[str] | None = None,
    storage_options: dict[str, str] | None = None,
    rank: int | None = None,
    world_size: int | None = None,
) -> DataLoader:
    """Build an iterable DataLoader (sequential scan, native object-store streaming).

    Batches form inside Lance's Rust scanner, so the DataLoader wraps them
    with ``batch_size=None`` and no workers. Distributed sharding is
    batch-granular (``ShardedBatchSampler``) — fragment-granular sharding
    starves ranks on the pipeline's few-fragment splits. With no explicit
    ``rank``/``world_size``, an initialized ``torch.distributed`` process
    group is detected automatically; otherwise the loader scans everything.

    :param uri: Dataset directory (local path or ``s3://`` URI).
    :param batch_size: Rows per yielded batch.
    :param columns: Columns each batch carries; ``None`` reads all.
    :param storage_options: Object-store config for a cloud ``uri`` (see
        :func:`synth_setter.pipeline.r2_io.r2_storage_options`); ``None`` local.
    :param rank: Explicit shard index; requires ``world_size``.
    :param world_size: Explicit shard count; requires ``rank``.
    :returns: DataLoader yielding ``{column: (<=batch_size, *inner_shape) tensor}`` —
        the final batch is shorter when the row count is not divisible by ``batch_size``.
    :raises ValueError: Exactly one of ``rank`` / ``world_size`` was provided, or
        ``rank`` is outside ``[0, world_size)``.
    """
    if (rank is None) != (world_size is None):
        raise ValueError("rank and world_size must be provided together")
    sampler = None
    if rank is not None and world_size is not None:
        if not 0 <= rank < world_size:
            raise ValueError(
                f"rank must be in [0, world_size); got rank={rank}, world_size={world_size}"
            )
        sampler = ShardedBatchSampler(rank, world_size)
    logger.info(
        "lance iterable dataloader: uri=%s columns=%s batch_size=%d rank=%s world_size=%s",
        uri,
        columns,
        batch_size,
        rank,
        world_size,
    )
    dataset = LanceDataset(
        str(uri),
        batch_size,
        dataset_options=_dataset_options(storage_options),
        columns=list(columns) if columns is not None else None,
        shard_granularity="batch",
        to_tensor_fn=_batch_to_shaped_tensors,
        sampler=sampler,
    )
    return DataLoader(dataset, batch_size=None, num_workers=0)
