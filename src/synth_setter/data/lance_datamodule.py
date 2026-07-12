"""Lance-backed dataloading for VST datasets.

``LanceShardFile`` adapts a Lance dataset directory — the format the data
pipeline's writer and finalize steps emit via
:func:`synth_setter.pipeline.data.lance_shard.write_lance_dataset` — to the
minimal h5py-``File``-like read surface ``VSTDataset`` consumes, so the Lance
subclasses inherit every batching / normalization / OT behavior unchanged.

``LanceVSTDataModule(loader="map")`` instead wires the native sample-indexed
:class:`synth_setter.data.lance_torch.LanceMapDataset` into standard
``DataLoader`` semantics, with :class:`PrepareBatchCollate` bridging batches
into :func:`synth_setter.data.vst_datamodule.prepare_batch`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import lance
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import DataLoader

from synth_setter.data.lance_torch import LanceMapDataset, map_dataloader_over
from synth_setter.data.vst_datamodule import (
    RawBatch,
    VSTDataModule,
    VSTDataset,
    load_dataset_statistics,
    prepare_batch,
)


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


class LanceVSTDataset(VSTDataset):
    """``VSTDataset`` reading a ``.lance`` dataset directory instead of HDF5."""

    def _open(self, dataset_file: str | Path) -> LanceShardFile:
        """Open the ``.lance`` shard dataset read-only.

        :param dataset_file: Path to the ``.lance`` dataset directory.
        :return: Adapter handle the base-class readers consume.
        """
        return LanceShardFile(dataset_file)


class PrepareBatchCollate:
    """Picklable ``collate_fn`` bridging ``LanceMapDataset`` batches into ``prepare_batch``.

    ``LanceMapDataset.__getitems__`` already returns one pre-collated column
    dict per batch, so this receives the whole batch (never a sample list) and
    must not re-stack it. The noise RNG is created lazily per process: the
    construction-time seed draw comes from the global RNG (so Lightning's
    ``seed_everything`` governs it), and inside a dataloader worker the
    DataLoader-assigned worker seed is used instead so workers don't share a
    stream. ``torch.Generator`` cannot be pickled, so it is dropped on
    pickling and re-created lazily — spawn workers each re-derive their own.
    """

    def __init__(
        self,
        *,
        mean: np.ndarray | None,
        std: np.ndarray | None,
        rescale_params: bool,
        ot: bool,
    ) -> None:
        """Fix the per-batch semantics this collate applies.

        :param mean: Mel mean to subtract, or ``None`` to skip normalization.
        :param std: Mel std to divide by, or ``None`` to skip normalization.
        :param rescale_params: Whether to map params from ``[0, 1]`` to ``[-1, 1]``.
        :param ot: Whether to Hungarian-match noise to params per batch.
        """
        self.mean = mean
        self.std = std
        self.rescale_params = rescale_params
        self.ot = ot
        self._seed = int(torch.randint(2**63 - 1, (1,)).item())
        self._generator: torch.Generator | None = None

    def __getstate__(self) -> dict[str, object]:
        """Drop the unpicklable generator so spawn workers can receive the collate.

        :returns: ``__dict__`` copy with ``_generator`` reset to ``None``.
        """
        state = self.__dict__.copy()
        state["_generator"] = None
        return state

    def _live_generator(self) -> torch.Generator:
        """Return this process's noise RNG, creating and seeding it on first use.

        :returns: Generator seeded from the worker seed (inside a worker) or the construction-time
            global-RNG draw (in-process loading).
        """
        generator = self._generator
        if generator is None:
            generator = torch.Generator()
            worker_info = torch.utils.data.get_worker_info()
            generator.manual_seed(worker_info.seed if worker_info else self._seed)
            self._generator = generator
        return generator

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | None]:
        """Turn one pre-collated column batch into model-ready tensors.

        :param batch: Column-name-to-tensor mapping from ``__getitems__``;
            column names match the :class:`RawBatch` keys.
        :returns: ``prepare_batch`` output — ``{"mel_spec", "m2l", "params",
            "noise", "audio"}`` float32 tensors, ``None`` for unread modalities.
        """
        raw = cast("RawBatch", {name: tensor.numpy() for name, tensor in batch.items()})
        return prepare_batch(
            raw,
            mean=self.mean,
            std=self.std,
            rescale_params=self.rescale_params,
            ot=self.ot,
            generator=self._live_generator(),
        )


class _RepeatFirstRows(torch.utils.data.Dataset):
    """Map every row index onto the first ``batch_size`` rows of the wrapped dataset.

    Sample-indexed re-expression of ``repeat_first_batch``: with a sequential
    sampler, every yielded batch is exactly rows ``[0, batch_size)`` in order,
    preserving the legacy overfit-one-batch debugging intent.
    """

    def __init__(self, dataset: LanceMapDataset, batch_size: int) -> None:
        """Wrap ``dataset`` so reads cycle within its first ``batch_size`` rows.

        :param dataset: Sample-indexed dataset to wrap.
        :param batch_size: Row modulus every index is folded into.
        """
        self._dataset = dataset
        self._batch_size = batch_size

    def __len__(self) -> int:
        """Return the wrapped dataset's full row count (epoch length is unchanged).

        :returns: Row count of the wrapped dataset.
        """
        return len(self._dataset)

    def __getitems__(self, indices: Sequence[int]) -> dict[str, torch.Tensor]:
        """Fetch the batch with every index folded into the first-batch rows.

        :param indices: Row indices as drawn by the sampler.
        :returns: One ``(len(indices), *inner_shape)`` tensor per column.
        """
        return self._dataset.__getitems__([i % self._batch_size for i in indices])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Fetch one row, folded into the first-batch rows.

        :param idx: Row index as drawn by the sampler.
        :returns: One ``(*inner_shape,)`` tensor per column.
        """
        return self._dataset[idx % self._batch_size]


# DOC601/DOC603: pydoclint can't read sphinx ``:ivar:`` docs, so dataclass
# fields are documented in the docstring body instead.
@dataclass
class _MapSplit:  # noqa: DOC601, DOC603
    """One split's map-path pieces: the sample-indexed ``dataset`` and its ``collate``."""

    dataset: LanceMapDataset | _RepeatFirstRows
    collate: PrepareBatchCollate


class LanceVSTDataModule(VSTDataModule):
    """``VSTDataModule`` over ``train/val/test.lance`` splits.

    ``loader`` selects the read path: ``"legacy"`` keeps the batch-indexed
    ``LanceVSTDataset`` adapter; ``"map"`` builds a sample-indexed
    :class:`LanceMapDataset` per split behind standard ``DataLoader``
    semantics (``shuffle`` on train, ragged final batch kept, DDP via
    Lightning's ``DistributedSampler`` injection). Fake mode always uses the
    legacy path — it synthesizes batches in memory and never touches Lance.

    .. attribute :: dataset_cls

        Dataset class each split opens on the legacy path (``LanceVSTDataset``).

    .. attribute :: shard_suffix

        Shard filename suffix selecting ``*.lance`` splits.
    """

    dataset_cls: ClassVar[type[VSTDataset]] = LanceVSTDataset
    shard_suffix: ClassVar[str] = ".lance"

    def __init__(
        self, *args: Any, loader: Literal["legacy", "map"] = "legacy", **kwargs: Any
    ) -> None:
        """Store the loader selection on top of the base datamodule config.

        :param \\*args: Positional ``VSTDataModule`` arguments.
        :param loader: Read path per split: ``"legacy"`` (batch-indexed
            adapter) or ``"map"`` (sample-indexed ``LanceMapDataset``).
        :param \\*\\*kwargs: Keyword ``VSTDataModule`` arguments.
        :raises ValueError: If ``loader`` names an unknown read path.
        """
        super().__init__(*args, **kwargs)
        if loader not in ("legacy", "map"):
            raise ValueError(f"loader must be 'legacy' or 'map', got {loader!r}")
        self.loader = loader
        self._map_splits: dict[str, _MapSplit] = {}

    @property
    def _map_mode(self) -> bool:
        """Whether dataloading goes through the sample-indexed map path.

        :returns: True for ``loader="map"`` outside fake mode; fake batches
            are synthesized in memory, so they stay on the legacy path.
        """
        return self.loader == "map" and not self.fake

    def _build_map_split(
        self, shard_path: Path, *, ot: bool, read_audio: bool, repeat_first_batch: bool
    ) -> _MapSplit:
        """Build one split's map dataset and collate.

        :param shard_path: ``.lance`` dataset directory of the split.
        :param ot: Whether this split Hungarian-matches noise to params.
        :param read_audio: Whether to project the ``audio`` column.
        :param repeat_first_batch: Whether reads cycle within the first batch.
        :returns: The split's dataset/collate pair.
        """
        columns = ["param_array"]
        columns.append("mel_spec" if self.conditioning == "mel" else "music2latent")
        if read_audio:
            columns.append("audio")
        dataset: LanceMapDataset | _RepeatFirstRows = LanceMapDataset(shard_path, columns=columns)
        if repeat_first_batch:
            dataset = _RepeatFirstRows(dataset, self.batch_size)
        mean = std = None
        if self.use_saved_mean_and_variance:
            mean, std = load_dataset_statistics(shard_path)
        collate = PrepareBatchCollate(mean=mean, std=std, rescale_params=True, ot=ot)
        return _MapSplit(dataset=dataset, collate=collate)

    def setup(self, stage: str | None = None) -> None:
        """Build per-split datasets for the selected loader path.

        :param stage: Lightning stage hint; unused because every split is constructed eagerly
            (matching the legacy path).
        """
        if not self._map_mode:
            super().setup(stage)
            return
        self._map_splits = {
            "train": self._build_map_split(
                self.dataset_root / f"train{self.shard_suffix}",
                ot=self.ot,
                read_audio=False,
                repeat_first_batch=self.repeat_first_batch,
            ),
            "val": self._build_map_split(
                self.dataset_root / f"val{self.shard_suffix}",
                ot=False,
                read_audio=False,
                repeat_first_batch=self.repeat_first_batch,
            ),
            "test": self._build_map_split(
                self.dataset_root / f"test{self.shard_suffix}",
                ot=False,
                read_audio=False,
                repeat_first_batch=self.repeat_first_batch,
            ),
            # Predict deliberately skips repeat_first_batch, matching legacy setup.
            "predict": self._build_map_split(
                self.predict_file, ot=False, read_audio=True, repeat_first_batch=False
            ),
        }

    def _map_dataloader(self, split: str, *, shuffle: bool) -> DataLoader:
        """Build the sample-indexed loader for one set-up split.

        :param split: Split key in ``_map_splits``.
        :param shuffle: Whether the sampler permutes row order.
        :returns: DataLoader yielding ``prepare_batch`` model batches.
        """
        pieces = self._map_splits[split]
        return map_dataloader_over(
            pieces.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            collate_fn=pieces.collate,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader for the selected loader path.

        :returns: Dataloader over the training split.
        """
        if not self._map_mode:
            return super().train_dataloader()
        # repeat_first_batch needs sequential order so every batch is exactly
        # rows [0, batch_size) — shuffling would draw random multisets of them.
        return self._map_dataloader("train", shuffle=not self.repeat_first_batch)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader for the selected loader path.

        :returns: Dataloader over the validation split.
        """
        if not self._map_mode:
            return super().val_dataloader()
        return self._map_dataloader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader for the selected loader path.

        :returns: Dataloader over the test split.
        """
        if not self._map_mode:
            return super().test_dataloader()
        return self._map_dataloader("test", shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        """Return the prediction dataloader for the selected loader path.

        :returns: Dataloader over the prediction split.
        """
        if not self._map_mode:
            return super().predict_dataloader()
        return self._map_dataloader("predict", shuffle=False)

    def teardown(self, stage: str | None = None) -> None:
        """Release split resources for the selected loader path.

        :param stage: Lightning stage hint; unused because all splits are released together.
        """
        if not self._map_mode:
            super().teardown(stage)
            return
        # LanceMapDataset has no close API: handles open lazily per process
        # and are released with the references.
        self._map_splits = {}
