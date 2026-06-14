"""Lance-backed dataloading for VST datasets.

``LanceShardFile`` adapts a single-file Lance shard — the format the data
pipeline's writer and finalize steps emit via
:func:`synth_setter.pipeline.data.lance_shard.write_lance_file` — to the
minimal h5py-``File``-like read surface ``VSTDataset`` consumes, so the Lance
subclasses inherit every batching / normalization / OT behavior unchanged.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, cast

import numpy as np
import pyarrow as pa
from lance.file import LanceFileReader

from synth_setter.data.vst_datamodule import VSTDataModule, VSTDataset


class LanceColumn:
    """H5py-``Dataset``-like read view over one fixed-shape tensor column."""

    def __init__(self, shard: LanceShardFile, name: str, inner_shape: tuple[int, ...]):
        """Wrap one column of an open Lance shard.

        :param shard: Open shard the reads go through.
        :param name: Column name within the schema.
        :param inner_shape: Per-row tensor shape from the schema.
        """
        self._shard = shard
        self._name = name
        # Backs the h5py-like ``shape`` property; decode reads the shape from
        # Arrow's tensor type, so reads don't consult this.
        self._inner_shape = inner_shape

    @property
    def shape(self) -> tuple[int, ...]:
        """``(num_rows, *tensor_shape)``, mirroring ``h5py.Dataset.shape``.

        :return: Row count followed by the per-row tensor shape.
        """
        return (self._shard.num_rows, *self._inner_shape)

    def __getitem__(self, idx: slice | Sequence[int] | np.ndarray) -> np.ndarray:
        """Materialize the selected rows as one numpy array.

        :param idx: Slice (positive step), or ascending per-row integer
            indices — the same sorted-index contract h5py enforces; the
            samplers sort.
        :return: ``(len(idx), *tensor_shape)`` array of the column's dtype.
        :raises ValueError: If the slice step is negative or fancy indices are
            not in ascending order.
        """
        reader = self._shard.column_reader(self._name)
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._shard.num_rows)
            if step < 1:
                # Mirrors h5py, and keeps reads on the ascending-order contract.
                raise ValueError(f"slice step must be >= 1, got {step}")
            if step == 1:
                results = reader.read_range(start, max(stop - start, 0))
            else:
                results = reader.take_rows(list(range(start, stop, step)))
        else:
            indices = [int(i) for i in idx]
            # Explicit check so the contract doesn't ride on take_rows' message,
            # which may change across pylance versions.
            if any(b < a for a, b in zip(indices, indices[1:])):
                raise ValueError(f"fancy indices must be in ascending order, got {indices}")
            results = reader.take_rows(indices)
        chunk = results.to_table().column(self._name).combine_chunks()
        array = chunk.to_numpy_ndarray()
        # Copy out of Arrow's read-only buffer: h5py reads return writable arrays,
        # and torch.from_numpy over a read-only view is undefined behavior on write.
        return array if array.flags.writeable else array.copy()


class LanceShardFile:
    """Read-only adapter exposing a single-file Lance shard via the h5py-``File`` read surface."""

    def __init__(self, path: str | Path):
        """Open the ``.lance`` shard file read-only.

        :param path: Path to the ``.lance`` shard file.
        :raises ValueError: If ``path`` is missing or is a directory (the Lance
            *dataset* layout, not the single-file shard format the pipeline
            writes) — a stable contract independent of which exception
            ``LanceFileReader`` raises.
        """
        path = Path(path)
        self._path = str(path)
        if path.is_dir():
            raise ValueError(
                f"expected a single-file Lance shard, got a directory "
                f"(legacy Lance dataset layout?): {self._path}"
            )
        if not path.is_file():
            raise ValueError(f"Lance shard file was not found: {self._path}")
        metadata = LanceFileReader(self._path).metadata()
        # The shard is immutable, so row count and per-column tensor shapes are
        # cached once instead of re-querying file metadata on every batch read.
        self.num_rows = metadata.num_rows
        self._inner_shapes = {
            field.name: tuple(cast(pa.FixedShapeTensorType, field.type).shape)
            for field in metadata.schema
        }
        self._readers: dict[str, LanceFileReader] | None = {}
        self._pid = os.getpid()

    def column_reader(self, name: str) -> LanceFileReader:
        """Return the projected reader for ``name``, reopening after a fork.

        Lance readers are not fork-safe, so a forked DataLoader worker must not reuse the readers
        it inherited from the parent — each worker opens its own on first read.

        :param name: Column name to project.
        :return: Single-column reader owned by the current process.
        :raises ValueError: If the shard has been closed.
        """
        if self._readers is None:
            raise ValueError("I/O on closed LanceShardFile")
        if os.getpid() != self._pid:
            self._readers = {}
            self._pid = os.getpid()
        reader = self._readers.get(name)
        if reader is None:
            reader = LanceFileReader(self._path, columns=[name])
            self._readers[name] = reader
        return reader

    def __getitem__(self, name: str) -> LanceColumn:
        """Return the named column view.

        :param name: Column name in the Lance schema (e.g. ``"mel_spec"``).
        :return: Lazy read view over that column.
        :raises KeyError: If the schema has no such column — at lookup, like h5py, not on first
            read.
        :raises ValueError: If the shard has been closed.
        """
        if self._readers is None:
            raise ValueError("I/O on closed LanceShardFile")
        if name not in self._inner_shapes:
            raise KeyError(name)
        return LanceColumn(self, name, self._inner_shapes[name])

    def __bool__(self) -> bool:
        """Mirror ``h5py.File`` truthiness: True while open, False after ``close``.

        :return: Whether the shard is still open.
        """
        return self._readers is not None

    def close(self) -> None:
        """Release the underlying Lance readers (idempotent)."""
        self._readers = None


class LanceVSTDataset(VSTDataset):
    """``VSTDataset`` reading a single-file ``.lance`` shard instead of HDF5."""

    def _open(self, dataset_file: str | Path) -> LanceShardFile:
        """Open the ``.lance`` shard file read-only.

        :param dataset_file: Path to the ``.lance`` shard file.
        :return: Adapter handle the base-class readers consume.
        """
        return LanceShardFile(dataset_file)


class LanceVSTDataModule(VSTDataModule):
    """``VSTDataModule`` over ``train/val/test.lance`` splits.

    .. attribute :: dataset_cls

        Dataset class each split opens (``LanceVSTDataset``).

    .. attribute :: shard_suffix

        Shard filename suffix selecting ``*.lance`` splits.
    """

    dataset_cls: ClassVar[type[VSTDataset]] = LanceVSTDataset
    shard_suffix: ClassVar[str] = ".lance"
