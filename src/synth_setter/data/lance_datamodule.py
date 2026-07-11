"""Lance-backed dataloading for VST datasets.

``LanceShardFile`` adapts one Lance dataset directory — the format the data
pipeline's writer emits via
:func:`synth_setter.pipeline.data.lance_shard.write_lance_dataset` — to the
minimal h5py-``File``-like read surface ``VSTDataset`` consumes, so the Lance
subclasses inherit every batching / normalization / OT behavior unchanged.
``ShardedLanceFile`` layers global row indexing over an ordered list of such
directories, letting ``ShardedLanceVSTDataModule`` train straight off a
dataset run directory (``shard-*.lance`` + ``input_spec.json`` +
``stats.npz``) without any merged per-split datasets.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

import lance
import numpy as np
import pyarrow as pa

from synth_setter.data.vst_datamodule import ShardFile, VSTDataModule, VSTDataset
from synth_setter.pipeline.spec_io import load_spec_from_root

if TYPE_CHECKING:
    from synth_setter.pipeline.schemas.spec import DatasetSpec, Split


@lru_cache(maxsize=8)
def _cached_run_spec(run_root: str) -> DatasetSpec:
    """Load and cache the run directory's ``DatasetSpec``.

    ``setup()`` opens four splits against the same run root, and each spec
    validation is O(num_shards) (the shard list is materialized by a
    whole-model validator) — caching parses the immutable per-run spec once
    per process instead of once per split. Keyed on ``run_root`` alone: this
    assumes a given path names one run for the process lifetime, which holds
    under one-run-per-process training; a sweep that remounts the same path to
    different content would need an explicit ``cache_clear()``.

    :param run_root: Run directory (or URI) holding ``input_spec.json``.
    :returns: The parsed spec.
    """
    return load_spec_from_root(run_root)


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

    def live_dataset(self) -> lance.LanceDataset:
        """Return the dataset handle, reopening after a fork.

        Lance datasets are not fork-safe, so a forked DataLoader worker must not reuse the handle
        it inherited from the parent — each worker reopens its own on first read.

        :return: Dataset handle owned by the current process.
        :raises ValueError: If the shard has been closed.
        """
        if self._dataset is None:
            raise ValueError("I/O on closed LanceShardFile")
        if os.getpid() != self._pid:
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
        if self._dataset is None:
            raise ValueError("I/O on closed LanceShardFile")
        if name not in self._inner_shapes:
            raise KeyError(name)
        return LanceColumn(self, name, self._inner_shapes[name])

    def __bool__(self) -> bool:
        """Mirror ``h5py.File`` truthiness: True while open, False after ``close``.

        :return: Whether the shard is still open.
        """
        return self._dataset is not None

    def close(self) -> None:
        """Release the underlying Lance dataset handle (idempotent)."""
        self._dataset = None


class ShardedLanceColumn:
    """H5py-``Dataset``-like read view over one column spanning ordered shards."""

    def __init__(self, file: ShardedLanceFile, name: str, inner_shape: tuple[int, ...]):
        """Wrap one column of an open sharded Lance dataset.

        :param file: Open sharded handle the reads go through.
        :param name: Column name within the shared shard schema.
        :param inner_shape: Per-row tensor shape from the schema.
        """
        self._file = file
        self._name = name
        self._inner_shape = inner_shape

    @property
    def shape(self) -> tuple[int, ...]:
        """``(total_rows, *tensor_shape)`` across every shard, mirroring h5py.

        :return: Global row count followed by the per-row tensor shape.
        """
        return (self._file.num_rows, *self._inner_shape)

    def __getitem__(self, idx: slice | Sequence[int] | np.ndarray) -> np.ndarray:
        """Materialize the selected global rows as one numpy array.

        :param idx: Slice (positive step) or per-row integer indices in any
            order; order is preserved in the result.
        :return: ``(len(idx), *tensor_shape)`` array of the column's dtype.
        :raises ValueError: If the slice step is negative or the selection is empty.
        """
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._file.num_rows)
            if step < 1:
                raise ValueError(f"slice step must be >= 1, got {step}")
            if step == 1:
                return self._read_contiguous(start, stop)
            indices = np.arange(start, stop, step, dtype=np.int64)
        else:
            indices = np.asarray(idx, dtype=np.int64)
        return self._read_rows(indices)

    def _read_contiguous(self, start: int, stop: int) -> np.ndarray:
        """Read global rows ``[start, stop)`` shard by shard and stitch in order.

        :param start: Inclusive global start row.
        :param stop: Exclusive global stop row.
        :return: ``(stop - start, *tensor_shape)`` array.
        :raises ValueError: If the range is empty.
        """
        if start >= stop:
            raise ValueError(f"empty selection: rows [{start}, {stop})")
        rows = self._file.rows_per_shard
        pieces = []
        pos = start
        while pos < stop:
            shard_index = pos // rows
            local_lo = pos - shard_index * rows
            local_hi = min(stop - shard_index * rows, rows)
            pieces.append(self._file.shard(shard_index)[self._name][local_lo:local_hi])
            pos = shard_index * rows + local_hi
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)

    def _read_rows(self, indices: np.ndarray) -> np.ndarray:
        """Gather arbitrary global rows, preserving the requested order.

        :param indices: Global row indices (any order, duplicates allowed).
        :return: ``(len(indices), *tensor_shape)`` array.
        :raises ValueError: If the selection is empty.
        :raises IndexError: If any index falls outside ``[0, num_rows)``.
        """
        if indices.size == 0:
            raise ValueError("empty selection: no row indices given")
        out_of_range = (indices < 0) | (indices >= self._file.num_rows)
        if out_of_range.any():
            raise IndexError(
                f"row indices out of range [0, {self._file.num_rows}): "
                f"{indices[out_of_range][:5].tolist()}"
            )
        rows = self._file.rows_per_shard
        shard_ids = indices // rows

        def read_shard(shard_id: int) -> tuple[np.ndarray, np.ndarray]:
            mask = shard_ids == shard_id
            return mask, self._file.shard(shard_id)[self._name][indices[mask] - shard_id * rows]

        unique_ids = np.unique(shard_ids)
        first_mask, first_piece = read_shard(int(unique_ids[0]))
        out = np.empty((indices.shape[0], *first_piece.shape[1:]), dtype=first_piece.dtype)
        out[first_mask] = first_piece
        for shard_id in unique_ids[1:]:
            mask, piece = read_shard(int(shard_id))
            out[mask] = piece
        return out


class ShardedLanceFile:
    """Read-only h5py-``File``-like surface over ordered ``shard-*.lance`` directories.

    Presents N equal-row shards as one logical dataset. Row geometry comes
    from the caller (the pipeline spec's ``render.samples_per_shard``, which
    workers validate before upload), so construction opens only the first
    shard (for the schema); the rest open lazily on first read and are
    verified against the declared geometry at that point. ``num_rows`` holds
    the total logical rows (``rows_per_shard * len(shard_paths)``) and
    ``rows_per_shard`` the per-shard row count every shard must carry.
    """

    def __init__(self, shard_paths: Sequence[str | Path], rows_per_shard: int):
        """Open the sharded dataset read-only.

        :param shard_paths: Ordered ``.lance`` dataset directories, one per shard.
        :param rows_per_shard: Rows each shard carries (equal-size shards).
        :raises ValueError: If ``shard_paths`` is empty or ``rows_per_shard < 1``.
        """
        if not shard_paths:
            raise ValueError("ShardedLanceFile requires at least one shard path")
        if rows_per_shard < 1:
            raise ValueError(f"rows_per_shard must be >= 1, got {rows_per_shard}")
        self._shard_paths = [Path(path) for path in shard_paths]
        self.rows_per_shard = rows_per_shard
        self.num_rows = rows_per_shard * len(self._shard_paths)
        self._shards: list[LanceShardFile | None] = [None] * len(self._shard_paths)
        self._closed = False
        # All shards share the writer schema, so shard 0 answers column lookups
        # (and gets its row count verified) up front.
        self.shard(0)

    def shard(self, index: int) -> LanceShardFile:
        """Return the open handle for shard ``index``, opening it on first use.

        :param index: Zero-based shard position within ``shard_paths``.
        :return: Open single-shard handle.
        :raises ValueError: If the file is closed, or the shard's actual row
            count differs from ``rows_per_shard``.
        """
        if self._closed:
            raise ValueError("I/O on closed ShardedLanceFile")
        opened = self._shards[index]
        if opened is None:
            opened = LanceShardFile(self._shard_paths[index])
            if opened.num_rows != self.rows_per_shard:
                raise ValueError(
                    f"shard {self._shard_paths[index]} has {opened.num_rows} rows, "
                    f"expected {self.rows_per_shard} (from the dataset spec's "
                    f"render.samples_per_shard)"
                )
            self._shards[index] = opened
        return opened

    def __getitem__(self, name: str) -> ShardedLanceColumn:  # noqa: DOC502 — raises propagate from shard(0) lookup
        """Return the named column view spanning every shard.

        :param name: Column name in the Lance schema (e.g. ``"mel_spec"``).
        :return: Lazy read view over that column.
        :raises KeyError: If the schema has no such column — at lookup, like h5py.
        :raises ValueError: If the file has been closed.
        """
        first_column = self.shard(0)[name]
        return ShardedLanceColumn(self, name, first_column.shape[1:])

    def __bool__(self) -> bool:
        """Mirror ``h5py.File`` truthiness: True while open, False after ``close``.

        :return: Whether the sharded file is still open.
        """
        return not self._closed

    def close(self) -> None:
        """Release every opened shard handle (idempotent)."""
        self._closed = True
        for opened in self._shards:
            if opened is not None:
                opened.close()


class LanceVSTDataset(VSTDataset):
    """``VSTDataset`` reading a ``.lance`` dataset directory instead of HDF5."""

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


class ShardedLanceVSTDataset(VSTDataset):
    """``VSTDataset`` reading a whole split from the run's per-shard ``.lance`` datasets.

    ``dataset_file`` is a *virtual* split path ``<run_root>/<split>.lance``:
    when no dataset exists at that literal path, the split's shard list is
    resolved from the sibling ``input_spec.json`` — the layout generate plus
    the stats-only finalize leave under the R2 run prefix, exposed locally by
    an ``rclone mount`` (or a full download). A path to an existing ``.lance``
    dataset directory is opened as-is (the ``predict_file`` escape hatch).
    """

    def _open(self, dataset_file: str | Path) -> ShardFile:  # noqa: DOC503 — FileNotFoundError propagates from load_spec_from_root
        """Resolve and open the split's shards (or a literal dataset directory).

        :param dataset_file: ``<run_root>/<split>.lance`` virtual path, or a
            real ``.lance`` dataset directory.
        :return: Read handle spanning every shard of the split.
        :raises FileNotFoundError: No dataset directory at ``dataset_file`` and
            no sibling ``input_spec.json`` to resolve shards from.
        :raises ValueError: The virtual filename is not a split name, or the
            split's shard range is empty.
        """
        path = Path(dataset_file)
        if path.is_dir():
            return LanceShardFile(path)
        spec = _cached_run_spec(str(path.parent))
        split = path.name.removesuffix(".lance")
        if split not in spec.split_shard_ranges:
            raise ValueError(
                f"{path.name!r} does not name a split; expected one of "
                f"{[f'{name}.lance' for name in spec.split_shard_ranges]}"
            )
        lo, hi = spec.split_shard_ranges[cast("Split", split)]
        if lo >= hi:
            raise ValueError(
                f"{split} split is empty (split_shard_ranges[{split!r}]={(lo, hi)!r}); "
                f"nothing to read"
            )
        shard_paths = [path.parent / shard.filename for shard in spec.shards[lo:hi]]
        return ShardedLanceFile(shard_paths, spec.render.samples_per_shard)


class ShardedLanceVSTDataModule(VSTDataModule):
    """``VSTDataModule`` over a sharded Lance dataset run directory.

    ``dataset_root`` points at the run prefix directory itself (typically an
    ``rclone mount`` of ``r2://<bucket>/<prefix>``) holding ``shard-*.lance``,
    ``input_spec.json``, and ``stats.npz``; each split resolves its shard
    subset via the spec's ``split_shard_ranges``.

    .. attribute :: dataset_cls

        Dataset class each split opens (``ShardedLanceVSTDataset``).

    .. attribute :: shard_suffix

        Virtual split filename suffix (``<split>.lance``).
    """

    dataset_cls: ClassVar[type[VSTDataset]] = ShardedLanceVSTDataset
    shard_suffix: ClassVar[str] = ".lance"
