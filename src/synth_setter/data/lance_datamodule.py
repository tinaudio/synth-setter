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

import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, cast

import lance
import numpy as np
import pyarrow as pa
import torch
from lance.torch.data import get_safe_loader
from torch.utils.data import DataLoader

from synth_setter.conditioning import ConditioningMode
from synth_setter.data.lance_torch import LanceMapDataset, map_dataloader_over
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.data.vst_datamodule import (
    RawBatch,
    ShiftedBatchSampler,
    VSTDataModule,
    VSTDataset,
    draw_generator_seed,
    load_dataset_statistics,
    prepare_batch,
)
from synth_setter.param_spec_name import ParamSpecName

logger = logging.getLogger(__name__)


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


class PrepareBatchCollate:
    """Adapt pre-collated map batches through ``prepare_batch`` with a process-local RNG.

    The RNG is dropped on pickle and lazily seeded from the worker seed or global RNG draw.
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
        self._seed = draw_generator_seed()
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
        raw = cast(RawBatch, {name: tensor.numpy() for name, tensor in batch.items()})
        return prepare_batch(
            raw,
            mean=self.mean,
            std=self.std,
            rescale_params=self.rescale_params,
            ot=self.ot,
            generator=self._live_generator(),
        )


class _RepeatFirstBatchSampler(torch.utils.data.Sampler):
    """Repeat the first full batch for each complete batch in an epoch.

    Indices fold modulo ``batch_size``; debug-only and not DDP-aware.
    """

    def __init__(self, num_rows: int, batch_size: int) -> None:
        """Fix the epoch geometry the sampler folds.

        :param num_rows: Dataset row count; floored to full batches.
        :param batch_size: Row modulus every index is folded into.
        :raises ValueError: If the dataset holds less than one full batch — flooring would
            otherwise yield a silently empty epoch.
        """
        if num_rows < batch_size:
            raise ValueError(
                f"repeat_first_batch needs at least one full batch: "
                f"{num_rows} rows < batch_size {batch_size}"
            )
        self._num_indices = num_rows - num_rows % batch_size
        self._batch_size = batch_size

    def __len__(self) -> int:
        """Return the number of indices per epoch.

        :returns: Row count floored to a multiple of the batch size.
        """
        return self._num_indices

    def __iter__(self) -> Iterator[int]:
        """Yield the folded index sequence for one epoch.

        :yields int: The next row index, always within ``[0, batch_size)``.
        """
        for i in range(self._num_indices):
            yield i % self._batch_size


@dataclass(frozen=True)
class _MapSplit:
    """One split's map-path pieces.

    .. attribute :: dataset

        Sample-indexed dataset the split's loader reads.

    .. attribute :: collate

        Collate carrying the split's batch semantics (stats, rescale, OT).
    """

    dataset: LanceMapDataset
    collate: PrepareBatchCollate


class LanceVSTDataModule(VSTDataModule):
    """Read Lance splits through either the batch-indexed legacy or sample-indexed map path.

    Fake mode remains in-memory on the legacy path; map evaluation preserves ragged tails.

    .. attribute :: dataset_cls

        Dataset class each split opens on the legacy path (``LanceVSTDataset``).

    .. attribute :: shard_suffix

        Shard filename suffix selecting ``*.lance`` splits.
    """

    dataset_cls: ClassVar[type[VSTDataset]] = LanceVSTDataset
    shard_suffix: ClassVar[str] = ".lance"

    def __init__(
        self,
        dataset_root: str | Path,
        download_dataset_root_uri: str | None = None,
        use_saved_mean_and_variance: bool = True,
        batch_size: int = 1024,
        ot: bool = True,
        num_workers: int = 0,
        fake: bool = False,
        repeat_first_batch: bool = False,
        predict_file: str | Path | None = None,
        conditioning: ConditioningMode = "mel",
        pin_memory: bool = True,
        *,
        param_spec_name: ParamSpecName,
        loader: Literal["legacy", "map"] = "legacy",
    ) -> None:
        """Store the loader selection on top of the base datamodule config.

        Mirrors the base signature explicitly (rather than ``*args``
        pass-through) so call sites keep full type checking.

        :param dataset_root: Local directory holding the per-split ``.lance`` datasets.
        :param download_dataset_root_uri: R2 URI to hydrate ``dataset_root`` from, or
            ``None`` to use the local directory as-is.
        :param use_saved_mean_and_variance: Whether to load and apply saved mel stats.
        :param batch_size: Number of samples per batch.
        :param ot: Whether to optimal-transport match noise to params per batch.
        :param num_workers: Number of dataloader worker processes.
        :param fake: Whether datasets synthesise random batches instead of reading shards.
        :param repeat_first_batch: Whether every batch repeats the first one.
        :param predict_file: Dataset used for prediction; defaults to the test split.
        :param conditioning: Conditioning feature, either ``"mel"`` or ``"m2l"``.
        :param pin_memory: Whether dataloaders pin memory for faster host-to-device copies.
        :param param_spec_name: Registry key selecting the param spec width.
        :param loader: Read path per split: ``"legacy"`` (batch-indexed
            adapter) or ``"map"`` (sample-indexed ``LanceMapDataset``).
        :raises ValueError: If ``loader`` names an unknown read path.
        """
        if loader not in ("legacy", "map"):
            raise ValueError(f"loader must be 'legacy' or 'map', got {loader!r}")
        super().__init__(
            dataset_root=dataset_root,
            download_dataset_root_uri=download_dataset_root_uri,
            use_saved_mean_and_variance=use_saved_mean_and_variance,
            batch_size=batch_size,
            ot=ot,
            num_workers=num_workers,
            fake=fake,
            repeat_first_batch=repeat_first_batch,
            predict_file=predict_file,
            conditioning=conditioning,
            pin_memory=pin_memory,
            param_spec_name=param_spec_name,
        )
        self.loader = loader
        self._map_splits: dict[str, _MapSplit] = {}

    def _legacy_dataloader(
        self,
        dataset: VSTDataset,
        *,
        sampler: ShiftedBatchSampler | None = None,
    ) -> DataLoader:
        """Build a legacy loader that isolates Lance handles in spawned workers.

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

    @property
    def _map_mode(self) -> bool:
        """Whether dataloading goes through the sample-indexed map path.

        :returns: True for ``loader="map"`` outside fake mode; fake batches
            are synthesized in memory, so they stay on the legacy path.
        """
        return self.loader == "map" and not self.fake

    def _build_map_split(
        self,
        shard_path: Path,
        *,
        ot: bool,
        read_audio: bool,
        stats: tuple[np.ndarray, np.ndarray] | None,
    ) -> _MapSplit:
        """Build one split's map dataset and collate.

        :param shard_path: ``.lance`` dataset directory of the split.
        :param ot: Whether this split Hungarian-matches noise to params.
        :param read_audio: Whether to project the ``audio`` column.
        :param stats: Mel ``(mean, std)`` to normalize with, or ``None`` to skip.
        :returns: The split's dataset/collate pair.
        """
        columns = ["param_array"]
        columns.append("mel_spec" if self.conditioning == "mel" else "music2latent")
        if read_audio:
            columns.append("audio")
        dataset = LanceMapDataset(shard_path, columns=columns)
        mean, std = stats if stats is not None else (None, None)
        collate = PrepareBatchCollate(mean=mean, std=std, rescale_params=True, ot=ot)
        return _MapSplit(dataset=dataset, collate=collate)

    def setup(self, stage: str | None = None) -> None:
        """Build per-split datasets for the selected loader path.

        :param stage: Lightning stage hint; unused because every split is constructed eagerly
            (matching the legacy path).
        """
        if not self._map_mode:
            if self.loader == "map" and self.fake:
                logger.info(
                    "fake mode ignores loader='map': batches use the legacy in-memory path"
                )
            super().setup(stage)
            return
        resolve_param_spec(self.param_spec_name)
        train_shard = self.dataset_root / f"train{self.shard_suffix}"
        split_stats = predict_stats = None
        if self.use_saved_mean_and_variance:
            # train/val/test share the root's stats.npz; only a predict_file
            # outside the root carries its own.
            split_stats = load_dataset_statistics(train_shard)
            predict_stats = (
                split_stats
                if self.predict_file.parent == self.dataset_root
                else load_dataset_statistics(self.predict_file)
            )
        self._map_splits = {
            "train": self._build_map_split(
                train_shard, ot=self.ot, read_audio=False, stats=split_stats
            ),
            "val": self._build_map_split(
                self.dataset_root / f"val{self.shard_suffix}",
                ot=False,
                read_audio=False,
                stats=split_stats,
            ),
            "test": self._build_map_split(
                self.dataset_root / f"test{self.shard_suffix}",
                ot=False,
                read_audio=False,
                stats=split_stats,
            ),
            "predict": self._build_map_split(
                self.predict_file, ot=False, read_audio=True, stats=predict_stats
            ),
        }

    def _map_dataloader(self, split: str, *, shuffle: bool, drop_last: bool) -> DataLoader:
        """Build the sample-indexed loader for one set-up split.

        :param split: Split key in ``_map_splits``.
        :param shuffle: Whether the sampler permutes row order.
        :param drop_last: Whether a ragged final batch is dropped.
        :returns: DataLoader yielding ``prepare_batch`` model batches.
        """
        pieces = self._map_splits[split]
        # Legacy parity: train/val/test repeat the first batch when configured;
        # predict never does. The sampler replaces (and overrides) shuffling.
        sampler = None
        if self.repeat_first_batch and split != "predict":
            sampler = _RepeatFirstBatchSampler(len(pieces.dataset), self.batch_size)
        return map_dataloader_over(
            pieces.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=pieces.collate,
            pin_memory=self.pin_memory,
            sampler=sampler,
            shuffle=shuffle if sampler is None else None,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader for the selected loader path.

        :returns: Dataloader over the training split.
        """
        if not self._map_mode:
            return self._legacy_dataloader(
                self.train_dataset,
                sampler=ShiftedBatchSampler(self.batch_size, len(self.train_dataset)),
            )
        # drop_last matches legacy's floor-divide: a trailing short batch (down
        # to size 1) would break batch-statistics layers during training.
        return self._map_dataloader("train", shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader for the selected loader path.

        :returns: Dataloader over the validation split.
        """
        if not self._map_mode:
            return self._legacy_dataloader(self.val_dataset)
        return self._map_dataloader("val", shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader for the selected loader path.

        :returns: Dataloader over the test split.
        """
        if not self._map_mode:
            return self._legacy_dataloader(self.test_dataset)
        return self._map_dataloader("test", shuffle=False, drop_last=False)

    def predict_dataloader(self) -> DataLoader:
        """Return the prediction dataloader for the selected loader path.

        :returns: Dataloader over the prediction split.
        """
        if not self._map_mode:
            return self._legacy_dataloader(self.predict_dataset)
        return self._map_dataloader("predict", shuffle=False, drop_last=False)

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
