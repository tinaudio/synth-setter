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

from synth_setter.data.surge_datamodule import VSTDataModule, VSTDataset
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.constants import STATS_NPZ_FILENAME
from synth_setter.pipeline.data.lance_shard import tensor_chunk_to_numpy


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
        array = tensor_chunk_to_numpy(chunk, self._inner_shape)
        # Copy out of Arrow's read-only buffer: h5py reads return writable arrays,
        # and torch.from_numpy over a read-only view is undefined behavior on write.
        return array if array.flags.writeable else array.copy()


def _is_remote_uri(path: str) -> bool:
    """Return whether ``path`` is an ``s3://`` / ``r2://`` cloud URI lance streams over.

    :param path: Candidate shard path or URI.
    :returns: ``True`` for an ``s3://`` / ``r2://`` URI; ``False`` keeps the local-fs guards.
    """
    return path.startswith(("s3://", "r2://"))


class LanceShardFile:
    """Read-only adapter exposing a single-file Lance shard via the h5py-``File`` read surface."""

    def __init__(self, path: str | Path, storage_options: dict[str, str] | None = None):
        """Open the ``.lance`` shard read-only from local disk or an ``s3://`` URI.

        :param path: Path to the ``.lance`` shard file, or an ``s3://`` URI when
            ``storage_options`` is set (native R2 streaming).
        :param storage_options: ``object_store`` kwargs forwarded to every
            ``LanceFileReader`` when ``path`` is a cloud URI; ``None`` reads local.
        :raises ValueError: For a local ``path`` that is missing or is a directory
            (the Lance *dataset* layout, not the single-file shard format the
            pipeline writes) — a stable contract independent of which exception
            ``LanceFileReader`` raises. Remote URIs skip these filesystem guards.
        """
        self._path = str(path)
        self._storage_options = storage_options
        if not _is_remote_uri(self._path):
            local = Path(path)
            if local.is_dir():
                raise ValueError(
                    f"expected a single-file Lance shard, got a directory "
                    f"(legacy Lance dataset layout?): {self._path}"
                )
            if not local.is_file():
                raise ValueError(f"Lance shard file was not found: {self._path}")
        metadata = LanceFileReader(self._path, storage_options=storage_options).metadata()
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
            reader = LanceFileReader(
                self._path, storage_options=self._storage_options, columns=[name]
            )
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
        """Open the ``.lance`` shard read-only, streaming from R2 when configured.

        :param dataset_file: Path to the ``.lance`` shard file, or an ``s3://``
            URI when ``self.storage_options`` is set.
        :return: Adapter handle the base-class readers consume.
        """
        return LanceShardFile(dataset_file, storage_options=self.storage_options)


class LanceVSTDataModule(VSTDataModule):
    """``VSTDataModule`` over ``train/val/test.lance`` splits.

    With ``stream_from_r2`` the splits are read natively over R2's S3 API
    instead of being downloaded — ``prepare_data`` fetches only the small
    ``stats.npz`` and ``setup`` opens each split from its ``s3://`` URI.

    .. attribute :: dataset_cls

        Dataset class each split opens (``LanceVSTDataset``).

    .. attribute :: shard_suffix

        Shard filename suffix selecting ``*.lance`` splits.

    .. attribute :: supports_streaming

        ``True`` — Lance splits can be read natively from R2 via ``stream_from_r2``.
    """

    dataset_cls: ClassVar[type[VSTDataset]] = LanceVSTDataset
    shard_suffix: ClassVar[str] = ".lance"
    supports_streaming: ClassVar[bool] = True

    def _dataset_prefix(self) -> str:
        """Return ``download_dataset_root_uri`` with any trailing slash stripped.

        :returns: The ``r2://`` dataset prefix used to build split/stats URIs.
        :raises ValueError: ``download_dataset_root_uri`` is unset (streaming
            paths only reach here after the constructor's guard, so this is a
            defensive narrow for the type checker).
        """
        if self.download_dataset_root_uri is None:
            raise ValueError("streaming requires download_dataset_root_uri")
        return self.download_dataset_root_uri.rstrip("/")

    def _remote_split_uri(self, basename: str) -> str:
        """Return the ``s3://`` URI for ``basename`` under the dataset prefix.

        :param basename: Split filename, e.g. ``"train.lance"``.
        :returns: ``s3://<bucket>/<prefix>/<basename>`` for native lance reads.
        """
        return r2_io.to_s3_uri(f"{self._dataset_prefix()}/{basename}")

    def _hydrate_dataset_root(self) -> None:
        """Fetch only ``stats.npz`` when streaming; otherwise mirror the whole prefix."""
        if not self.stream_from_r2:
            super()._hydrate_dataset_root()
            return
        if not self.use_saved_mean_and_variance:
            return
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        r2_io.download_to_path(
            f"{self._dataset_prefix()}/{STATS_NPZ_FILENAME}",
            self.dataset_root / STATS_NPZ_FILENAME,
        )

    def _split_target(self, basename: str) -> str | Path:
        """Return the ``s3://`` split URI when streaming, else the local path.

        :param basename: Split filename, e.g. ``"train.lance"``.
        :returns: Remote URI (streaming) or ``dataset_root / basename`` (local).
        """
        if self.stream_from_r2:
            return self._remote_split_uri(basename)
        return super()._split_target(basename)

    def _dataset_extra_kwargs(self) -> dict[str, object]:
        """Supply ``storage_options`` + the local ``stats.npz`` path when streaming.

        ``stats_file`` is injected only when ``use_saved_mean_and_variance`` is set — the same
        condition :meth:`_hydrate_dataset_root` fetches it under — so it can't point at a missing file.

        :returns: Streaming extras for each split, or ``{}`` for the local path.
        """
        if not self.stream_from_r2:
            return {}
        extra: dict[str, object] = {"storage_options": r2_io.r2_storage_options()}
        if self.use_saved_mean_and_variance:
            extra["stats_file"] = self.dataset_root / STATS_NPZ_FILENAME
        return extra
