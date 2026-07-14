"""Lance-backed dataloading for VST datasets.

``LanceShardFile`` adapts a Lance dataset directory — the format the data
pipeline's writer and finalize steps emit via
:func:`synth_setter.pipeline.data.lance_shard.write_lance_dataset` — to the
minimal h5py-``File``-like read surface ``VSTDataset`` consumes, so the Lance
subclasses inherit every batching / normalization / OT behavior unchanged.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, cast

import lance
import numpy as np
import pyarrow as pa
import torch
from lance.torch.data import get_safe_loader
from torch.utils.data import DataLoader

from synth_setter.data.vst_datamodule import ShiftedBatchSampler, VSTDataModule, VSTDataset


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

        :param idx: Slice (positive step) or per-row integer indices in any
            order — ``LanceDataset.take`` preserves the requested order, so
            unlike the single-file reader this needs no ascending contract.
        :return: ``(len(idx), *tensor_shape)`` array of the column's dtype.
        :raises ValueError: If the slice step is negative.
        """
        dataset = self._shard.live_dataset()
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._shard.num_rows)
            if step < 1:
                raise ValueError(f"slice step must be >= 1, got {step}")
            if step == 1:
                table = dataset.scanner(
                    columns=[self._name], offset=start, limit=max(stop - start, 0)
                ).to_table()
            else:
                table = dataset.take(list(range(start, stop, step)), columns=[self._name])
        else:
            table = dataset.take([int(i) for i in idx], columns=[self._name])
        chunk = table.column(self._name).combine_chunks()
        array = chunk.to_numpy_ndarray()
        # Copy out of Arrow's read-only buffer: h5py reads return writable arrays,
        # and torch.from_numpy over a read-only view is undefined behavior on write.
        return array if array.flags.writeable else array.copy()


class LanceShardFile:
    """Read-only adapter exposing a Lance dataset directory via the h5py-``File`` read surface."""

    def __init__(self, path: str | Path):
        """Open the ``.lance`` shard dataset read-only.

        :param path: Path to the ``.lance`` dataset directory.
        :raises ValueError: If ``path`` is missing or is a file rather than a
            Lance dataset directory — a stable contract independent of which
            exception ``lance.dataset`` would raise.
        """
        path = Path(path)
        self._path = str(path)
        if path.is_file():
            raise ValueError(
                f"expected a Lance dataset directory, got a file "
                f"(legacy single-file Lance shard?): {self._path}"
            )
        if not path.is_dir():
            raise ValueError(f"Lance shard dataset was not found: {self._path}")
        dataset = lance.dataset(self._path)
        # count_rows()/schema each traverse the version manifest, so cache them
        # once: the shard is immutable, and reads happen per training batch.
        self.num_rows = dataset.count_rows()
        self._inner_shapes = {
            field.name: tuple(cast(pa.FixedShapeTensorType, field.type).shape)
            for field in dataset.schema
        }
        self._dataset: lance.LanceDataset | None = dataset
        self._pid = os.getpid()
        self._closed = False

    def __getstate__(self) -> dict[str, object]:
        """Drop the native handle before sending this shard to a spawned worker.

        :returns: Pickle state that recreates the handle in the receiving process.
        """
        state = self.__dict__.copy()
        state["_dataset"] = None
        state["_pid"] = None
        return state

    def live_dataset(self) -> lance.LanceDataset:
        """Return the dataset handle, reopening after a process boundary.

        Lance datasets are process-local, so each DataLoader worker opens its own handle on first
        read.

        :return: Dataset handle owned by the current process.
        :raises ValueError: If the shard has been closed.
        """
        if self._closed:
            raise ValueError("I/O on closed LanceShardFile")
        if self._dataset is None or os.getpid() != self._pid:
            self._dataset = lance.dataset(self._path)
            self._pid = os.getpid()
        return self._dataset

    def __getitem__(self, name: str) -> LanceColumn:
        """Return the named column view.

        :param name: Column name in the Lance schema (e.g. ``"mel_spec"``).
        :return: Lazy read view over that column.
        :raises KeyError: If the schema has no such column — at lookup, like h5py, not on first
            read.
        :raises ValueError: If the shard has been closed.
        """
        if self._closed:
            raise ValueError("I/O on closed LanceShardFile")
        if name not in self._inner_shapes:
            raise KeyError(name)
        return LanceColumn(self, name, self._inner_shapes[name])

    def __bool__(self) -> bool:
        """Mirror ``h5py.File`` truthiness: True while open, False after ``close``.

        :return: Whether the shard is still open.
        """
        return not self._closed

    def close(self) -> None:
        """Release the underlying Lance dataset handle (idempotent)."""
        self._dataset = None
        self._closed = True


class LanceVSTDataset(VSTDataset):
    """``VSTDataset`` reading a ``.lance`` dataset directory instead of HDF5."""

    def __getstate__(self) -> dict[str, object]:
        """Remove the generator state that cannot cross a spawned worker boundary.

        :returns: Pickle state with a generator recreated by ``__setstate__``.
        """
        state = self.__dict__.copy()
        state["generator"] = None
        state["_worker_reseed_done"] = False
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore a spawned worker's dataset state with a local generator.

        :param state: Pickle state supplied by ``__getstate__``.
        """
        self.__dict__.update(state)
        self.generator = torch.Generator()

    def _open(self, dataset_file: str | Path) -> LanceShardFile:
        """Open the ``.lance`` shard dataset read-only.

        :param dataset_file: Path to the ``.lance`` dataset directory.
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

    def _dataloader(
        self,
        dataset: VSTDataset,
        *,
        sampler: ShiftedBatchSampler | None = None,
    ) -> DataLoader:
        """Build a loader that isolates Lance handles in spawned workers.

        :param dataset: Split dataset the loader reads.
        :param sampler: Optional training sampler; ``None`` keeps sequential order.
        :returns: Dataloader over ``dataset``.
        """
        loader_kwargs = {
            "batch_size": None,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "sampler": sampler,
            "shuffle": False,
        }
        if self.num_workers == 0:
            return DataLoader(dataset, **loader_kwargs)
        return get_safe_loader(dataset, **loader_kwargs)

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader with Lance's safe worker context.

        :returns: Dataloader over the training dataset.
        """
        return self._dataloader(
            self.train_dataset,
            sampler=ShiftedBatchSampler(self.batch_size, len(self.train_dataset)),
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader with Lance's safe worker context.

        :returns: Dataloader over the validation dataset.
        """
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader with Lance's safe worker context.

        :returns: Dataloader over the test dataset.
        """
        return self._dataloader(self.test_dataset)

    def predict_dataloader(self) -> DataLoader:
        """Return the prediction dataloader with Lance's safe worker context.

        :returns: Dataloader over the prediction dataset.
        """
        return self._dataloader(self.predict_dataset)
