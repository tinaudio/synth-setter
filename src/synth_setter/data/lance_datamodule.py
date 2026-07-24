"""Map-style Lance dataloading for VST training and evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import lance
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import DataLoader

from synth_setter.conditioning import (
    Conditioning,
    EmbeddingConditioningSpec,
    resolve_embedding_conditioning,
)
from synth_setter.data.lance_torch import (
    LanceMapDataset,
    batch_to_shaped_tensors,
    map_dataloader_over,
)
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.data.vst_datamodule import (
    RawBatch,
    VSTDataModule,
    draw_generator_seed,
    load_dataset_statistics,
    prepare_batch,
    ranked_generator_seed,
)
from synth_setter.param_spec_name import ParamSpecName

_FAKE_BATCHES_PER_EPOCH = 10_000
_FAKE_AUDIO_SHAPE = (2, 44100 * 4)
_FAKE_MEL_SHAPE = (2, 128, 401)

type ModelBatch = dict[str, torch.Tensor | None]


def _fixed_embedding_shape(field: pa.Field) -> tuple[int, ...]:
    """Return a supported embedding field's fixed per-row shape.

    :param field: Arrow field selected by an embedding conditioning spec.
    :returns: Fixed per-row shape.
    :raises TypeError: If storage is variable-length, non-tensor, or non-floating.
    """
    field_type = field.type
    if pa.types.is_list(field_type) or pa.types.is_large_list(field_type):
        raise TypeError(
            f"conditioning column {field.name!r} uses variable-length type {field_type}; "
            "expected fixed-size list or Lance fixed-shape tensor"
        )
    if isinstance(field_type, pa.FixedShapeTensorType):
        shape = tuple(field_type.shape)
        value_type = field_type.value_type
    elif pa.types.is_fixed_size_list(field_type):
        shape = (field_type.list_size,)
        value_type = field_type.value_type
    else:
        raise TypeError(
            f"conditioning column {field.name!r} has unsupported type {field_type}; "
            "expected fixed-size list or Lance fixed-shape tensor"
        )
    if not pa.types.is_floating(value_type):
        raise TypeError(
            f"conditioning column {field.name!r} must contain floating-point values, "
            f"got {value_type}"
        )
    return shape


def _validate_embedding_column(
    shard_path: Path, spec: EmbeddingConditioningSpec
) -> None:
    """Validate one Lance split against a fixed-shape embedding specification.

    :param shard_path: Lance dataset selected for a Lightning split.
    :param spec: Expected column and per-row shape.
    :raises KeyError: If the configured column is absent.
    :raises ValueError: If shape differs, the split is empty, or its sample is non-finite.
    """
    dataset = lance.dataset(str(shard_path))
    column_index = dataset.schema.get_field_index(spec.column)
    if column_index < 0:
        raise KeyError(
            f"conditioning column {spec.column!r} is absent from {shard_path}"
        )
    shape = _fixed_embedding_shape(dataset.schema.field(column_index))
    if shape != spec.input_shape:
        raise ValueError(
            f"conditioning column {spec.column!r} has shape {shape}, "
            f"expected {spec.input_shape}"
        )
    if dataset.count_rows() == 0:
        raise ValueError(
            f"conditioning column {spec.column!r} cannot be sampled from empty {shard_path}"
        )
    sample = dataset.take([0], columns=[spec.column]).combine_chunks()
    record_batch = sample.to_batches()[0]
    values = batch_to_shaped_tensors(record_batch)[spec.column]
    if not torch.isfinite(values).all():
        raise ValueError(
            f"conditioning column {spec.column!r} sample contains non-finite values"
        )


class PrepareBatchCollate:
    """Transform pre-collated Lance columns with a process-local noise RNG."""

    def __init__(
        self,
        *,
        mean: np.ndarray | None,
        std: np.ndarray | None,
        rescale_params: bool,
        ot: bool,
        conditioning_column: str | None = None,
        preserve_legacy_m2l: bool = False,
    ) -> None:
        """Configure model-batch transformation semantics.

        :param mean: Mel mean, or ``None`` to skip normalization.
        :param std: Mel standard deviation, or ``None`` to skip normalization.
        :param rescale_params: Whether to map parameters to ``[-1, 1]``.
        :param ot: Whether to Hungarian-match noise to parameters.
        :param conditioning_column: Generic embedding column to expose as ``conditioning``.
        :param preserve_legacy_m2l: Whether ``music2latent`` also populates ``m2l``.
        """
        self.mean = mean
        self.std = std
        self.rescale_params = rescale_params
        self.ot = ot
        self.conditioning_column = conditioning_column
        self.preserve_legacy_m2l = preserve_legacy_m2l
        self._rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        self._seed = draw_generator_seed()
        self._generator: torch.Generator | None = None

    def __getstate__(self) -> dict[str, object]:
        """Drop the process-local generator before worker serialization.

        :returns: Pickle state with a lazily recreated generator.
        """
        state = self.__dict__.copy()
        state["_generator"] = None
        return state

    def _live_generator(self) -> torch.Generator:
        """Return this process's lazily seeded noise generator.

        :returns: Generator namespaced by worker and distributed rank.
        """
        generator = self._generator
        if generator is None:
            generator = torch.Generator()
            worker_info = torch.utils.data.get_worker_info()
            seed = (
                ranked_generator_seed(worker_info.seed, self._rank, worker_info.num_workers)
                if worker_info
                else ranked_generator_seed(self._seed, self._rank)
            )
            generator.manual_seed(seed)
            self._generator = generator
        return generator

    def __call__(self, batch: object) -> ModelBatch:
        """Convert stored Lance columns to the model batch contract.

        :param batch: Pre-collated stored columns from :class:`LanceMapDataset`.
        :returns: Float32 model batch with generated noise.
        """
        columns = cast(dict[str, torch.Tensor], batch)
        raw_values = {name: tensor.numpy() for name, tensor in columns.items()}
        if self.conditioning_column is not None:
            conditioning = raw_values[self.conditioning_column]
            raw_values["conditioning"] = conditioning
            if self.conditioning_column == "music2latent" and not self.preserve_legacy_m2l:
                del raw_values["music2latent"]
        raw = cast(RawBatch, raw_values)
        return prepare_batch(
            raw,
            mean=self.mean,
            std=self.std,
            rescale_params=self.rescale_params,
            ot=self.ot,
            generator=self._live_generator(),
        )


class _FakeMapDataset(torch.utils.data.Dataset[ModelBatch]):
    """Sample-indexed synthetic dataset retaining the historical fake batch contract."""

    def __init__(
        self,
        *,
        batch_size: int,
        num_params: int,
        read_audio: bool,
        conditioning: Conditioning,
    ) -> None:
        """Configure synthetic sample shapes and epoch length.

        :param batch_size: Samples per batch, used to retain 10,000 batches per epoch.
        :param num_params: Width of parameter and noise tensors.
        :param read_audio: Whether generated samples include prediction audio.
        :param conditioning: Synthetic conditioning modality to populate.
        """
        self._num_rows = batch_size * _FAKE_BATCHES_PER_EPOCH
        self._num_params = num_params
        self._read_audio = read_audio
        self._preserve_legacy_m2l = (
            isinstance(conditioning, str) and conditioning == "m2l"
        )
        self._embedding_conditioning = resolve_embedding_conditioning(conditioning)

    def __len__(self) -> int:
        """Return the sample count corresponding to 10,000 full batches.

        :returns: Effective samples per epoch.
        """
        return self._num_rows

    def _batch(self, num_rows: int) -> ModelBatch:
        """Draw one synthetic model batch from the worker's global RNG.

        :param num_rows: Number of requested sample indices.
        :returns: Model-ready random tensors with the configured shapes.
        """
        audio = torch.randn(num_rows, *_FAKE_AUDIO_SHAPE) if self._read_audio else None
        mel_spec = (
            torch.randn(num_rows, *_FAKE_MEL_SHAPE)
            if self._embedding_conditioning is None
            else None
        )
        conditioning = (
            torch.randn(num_rows, *self._embedding_conditioning.input_shape)
            if self._embedding_conditioning is not None
            else None
        )
        m2l = conditioning if self._preserve_legacy_m2l else None
        params = torch.rand(num_rows, self._num_params) * 2 - 1
        noise = torch.randn_like(params)
        return {
            "mel_spec": mel_spec,
            "m2l": m2l,
            "conditioning": conditioning,
            "params": params,
            "noise": noise,
            "audio": audio,
        }

    def __getitems__(self, indices: Sequence[int]) -> ModelBatch:
        """Draw one pre-collated batch for requested sample indices.

        :param indices: Sample indices selected by the dataloader.
        :returns: Synthetic batch with ``len(indices)`` rows.
        """
        return self._batch(len(indices))

    def __getitem__(self, index: int) -> ModelBatch:
        """Draw one synthetic sample.

        :param index: Sample index; values are synthetic and index-independent.
        :returns: Model batch fields without a leading batch dimension.
        """
        del index
        batch = self._batch(1)
        return {name: value[0] if value is not None else None for name, value in batch.items()}


def _model_batch_passthrough(batch: object) -> ModelBatch:
    """Return a pre-collated synthetic batch unchanged.

    :param batch: Synthetic model batch.
    :returns: The same batch.
    """
    return cast(ModelBatch, batch)


class _RepeatFirstBatchDataset(torch.utils.data.Dataset[ModelBatch]):
    """Fold every requested sample index into the first full batch."""

    def __init__(
        self, dataset: LanceMapDataset | _FakeMapDataset, batch_size: int
    ) -> None:
        """Wrap a map dataset with first-batch index folding.

        :param dataset: Sample-indexed real or synthetic dataset.
        :param batch_size: Row modulus used for index folding.
        :raises ValueError: If the dataset has less than one full batch.
        """
        num_rows = len(dataset)
        if num_rows < batch_size:
            raise ValueError(
                f"repeat_first_batch needs at least one full batch: "
                f"{num_rows} rows < batch_size {batch_size}"
            )
        self._dataset = dataset
        self._num_rows = num_rows - num_rows % batch_size
        self._batch_size = batch_size
        self._frozen_batch = (
            dataset.__getitems__(range(batch_size))
            if isinstance(dataset, _FakeMapDataset)
            else None
        )

    def __len__(self) -> int:
        """Return the source row count floored to complete batches.

        :returns: Effective sample count.
        """
        return self._num_rows

    def __getitems__(self, indices: Sequence[int]) -> ModelBatch:
        """Read requested rows after folding indices into the first batch.

        :param indices: Sample indices selected by the dataloader.
        :returns: Pre-collated model or stored columns.
        """
        folded = [index % self._batch_size for index in indices]
        if self._frozen_batch is not None:
            return {
                name: value[folded] if value is not None else None
                for name, value in self._frozen_batch.items()
            }
        return cast(ModelBatch, self._dataset.__getitems__(folded))

    def __getitem__(self, index: int) -> ModelBatch:
        """Read one row after folding its index into the first batch.

        :param index: Sample index selected by the dataloader.
        :returns: One row of model or stored columns.
        """
        batch = self.__getitems__([index])
        return {name: value[0] if value is not None else None for name, value in batch.items()}


type _SplitDataset = LanceMapDataset | _FakeMapDataset


@dataclass(frozen=True)
class _MapSplit:
    """Dataset and collate operation for one Lightning split.

    .. attribute :: dataset

       Sample-indexed dataset for this split.

    .. attribute :: collate

       Batch transformation applied after sample retrieval.
    """

    dataset: _SplitDataset
    collate: Callable[[object], ModelBatch]


class LanceVSTDataModule(VSTDataModule):
    """Read VST splits through stage-aware, sample-indexed map semantics."""

    _ALL_SPLITS = ("train", "val", "test", "predict")
    _STAGE_SPLITS = {
        "fit": ("train", "val"),
        "validate": ("val",),
        "test": ("test",),
        "predict": ("predict",),
    }

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        download_dataset_root_uri: str | None = None,
        use_saved_mean_and_variance: bool = True,
        batch_size: int = 1024,
        ot: bool = True,
        num_workers: int = 0,
        fake: bool = False,
        repeat_first_batch: bool = False,
        predict_file: str | Path | None = None,
        conditioning: Conditioning = "mel",
        pin_memory: bool = True,
        param_spec_name: ParamSpecName,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        download_dataset_txids: dict[str, str] | None = None,
        download_dataset_row_limit: int | None = None,
    ) -> None:
        """Store map-style Lance loader configuration.

        :param dataset_root: Local directory holding per-split Lance datasets.
        :param download_dataset_root_uri: R2 or file URI used to hydrate ``dataset_root``.
        :param use_saved_mean_and_variance: Whether to apply saved mel statistics.
        :param batch_size: Samples per model batch.
        :param ot: Whether training batches use optimal-transport matching.
        :param num_workers: Worker processes per dataloader.
        :param fake: Whether to synthesize samples instead of reading Lance.
        :param repeat_first_batch: Whether non-predict loaders repeat their first batch.
        :param predict_file: Prediction split; defaults to ``test.lance``.
        :param conditioning: Legacy mel/m2l mode or a fixed-shape embedding spec.
        :param pin_memory: Whether dataloaders pin returned tensors.
        :param param_spec_name: Registry key selecting parameter width.
        :param persistent_workers: Whether positive worker counts persist between iterators.
        :param prefetch_factor: Batches prefetched per worker; ``None`` keeps
            PyTorch's default, and in-process loading ignores it.
        :param download_dataset_txids: Per-split transaction uuids pinning the
            source snapshots; present selects the materialize path.
        :param download_dataset_row_limit: First-N rows per split at materialization
            time.
        """
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
            download_dataset_txids=download_dataset_txids,
            download_dataset_row_limit=download_dataset_row_limit,
        )
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self._splits: dict[str, _MapSplit] = {}
        self._setup_stage: str | None = None

    def _dataset_for(self, split: str) -> _SplitDataset:
        """Return one built split through the public dataset attributes.

        :param split: Split key created by :meth:`setup`.
        :returns: Sample-indexed dataset for the split.
        :raises AttributeError: If the current stage did not build the split.
        """
        try:
            return self._splits[split].dataset
        except KeyError as exc:
            raise AttributeError(f"{split}_dataset is unavailable") from exc

    @property
    def train_dataset(self) -> _SplitDataset:
        """Return the training dataset built for the current stage."""
        return self._dataset_for("train")

    @property
    def val_dataset(self) -> _SplitDataset:
        """Return the validation dataset built for the current stage."""
        return self._dataset_for("val")

    @property
    def test_dataset(self) -> _SplitDataset:
        """Return the test dataset built for the current stage."""
        return self._dataset_for("test")

    @property
    def predict_dataset(self) -> _SplitDataset:
        """Return the prediction dataset built for the current stage."""
        return self._dataset_for("predict")

    def _build_lance_split(
        self,
        shard_path: Path,
        *,
        ot: bool,
        read_audio: bool,
        stats: tuple[np.ndarray, np.ndarray] | None,
    ) -> _MapSplit:
        """Build one real Lance split and its batch transformer.

        :param shard_path: Lance dataset directory.
        :param ot: Whether to match batch noise to parameters.
        :param read_audio: Whether to project prediction audio.
        :param stats: Mel ``(mean, std)``, or ``None`` to skip normalization.
        :returns: Sample-indexed dataset and collate operation.
        """
        spec = self.embedding_conditioning
        if spec is not None:
            _validate_embedding_column(shard_path, spec)
        columns = self._loader_columns(read_audio=read_audio)
        mean, std = stats if stats is not None else (None, None)
        return _MapSplit(
            dataset=LanceMapDataset(shard_path, columns=columns),
            collate=PrepareBatchCollate(
                mean=mean,
                std=std,
                rescale_params=True,
                ot=ot,
                conditioning_column=spec.column if spec is not None else None,
                preserve_legacy_m2l=(
                    isinstance(self.conditioning, str) and self.conditioning == "m2l"
                ),
            ),
        )

    def _build_fake_split(self, *, num_params: int, read_audio: bool) -> _MapSplit:
        """Build one sample-indexed in-memory split.

        :param num_params: Selected parameter-spec width.
        :param read_audio: Whether prediction audio is generated.
        :returns: Synthetic dataset and pass-through collate.
        """
        return _MapSplit(
            dataset=_FakeMapDataset(
                batch_size=self.batch_size,
                num_params=num_params,
                read_audio=read_audio,
                conditioning=self.conditioning,
            ),
            collate=_model_batch_passthrough,
        )

    def _build_real_splits(self, split_names: Sequence[str]) -> dict[str, _MapSplit]:
        """Build the requested on-disk Lance splits.

        :param split_names: Split names required by the current stage.
        :returns: Requested split datasets and collate operations.
        """
        train_shard = self.dataset_root / f"train{self.shard_suffix}"
        split_stats = predict_stats = None
        if self.use_saved_mean_and_variance and self.embedding_conditioning is None:
            if any(name != "predict" for name in split_names):
                split_stats = load_dataset_statistics(train_shard)
            if "predict" in split_names:
                predict_stats = (
                    split_stats
                    if split_stats is not None
                    and self.predict_file.parent == self.dataset_root
                    else load_dataset_statistics(self.predict_file)
                )
        shard_paths = {
            "train": train_shard,
            "val": self.dataset_root / f"val{self.shard_suffix}",
            "test": self.dataset_root / f"test{self.shard_suffix}",
            "predict": self.predict_file,
        }
        return {
            name: self._build_lance_split(
                shard_paths[name],
                ot=self.ot if name == "train" else False,
                read_audio=name == "predict",
                stats=predict_stats if name == "predict" else split_stats,
            )
            for name in split_names
        }

    def setup(self, stage: str | None = None) -> None:
        """Build the sample-indexed splits required by a Lightning stage.

        :param stage: Lightning stage hint; ``None`` retains eager all-split setup.
        """
        split_names = (
            self._ALL_SPLITS
            if stage is None
            else self._STAGE_SPLITS.get(stage, self._ALL_SPLITS)
        )
        num_params = resolve_param_spec(self.param_spec_name).encoded_width
        if self.fake:
            self._splits = {
                name: self._build_fake_split(
                    num_params=num_params, read_audio=name == "predict"
                )
                for name in split_names
            }
        else:
            self._splits = self._build_real_splits(split_names)
        self._setup_stage = stage

    def _dataloader(self, split: str, *, shuffle: bool, drop_last: bool) -> DataLoader:
        """Build one standard map-style dataloader.

        :param split: Split key created by :meth:`setup`.
        :param shuffle: Whether to randomize sample order.
        :param drop_last: Whether to discard a ragged final batch.
        :returns: Dataloader yielding model-ready batches.
        :raises RuntimeError: If :meth:`setup` did not build the requested split.
        """
        try:
            pieces = self._splits[split]
        except KeyError as exc:
            raise RuntimeError(
                f"{split} split was not built by setup(stage={self._setup_stage!r})"
            ) from exc
        repeats_first_batch = self.repeat_first_batch and split != "predict"
        dataset = pieces.dataset
        if repeats_first_batch:
            dataset = _RepeatFirstBatchDataset(pieces.dataset, self.batch_size)
        return map_dataloader_over(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=pieces.collate,
            pin_memory=self.pin_memory,
            shuffle=False if repeats_first_batch else shuffle,
            drop_last=drop_last,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the shuffled training loader with ragged tails dropped.

        :returns: Sample-indexed training dataloader.
        """
        return self._dataloader("train", shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """Return the ordered validation loader, retaining a ragged tail.

        :returns: Sample-indexed validation dataloader.
        """
        return self._dataloader("val", shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        """Return the ordered test loader, retaining a ragged tail.

        :returns: Sample-indexed test dataloader.
        """
        return self._dataloader("test", shuffle=False, drop_last=False)

    def predict_dataloader(self) -> DataLoader:
        """Return the ordered prediction loader including source audio.

        :returns: Sample-indexed prediction dataloader.
        """
        return self._dataloader("predict", shuffle=False, drop_last=False)

    def teardown(self, stage: str | None = None) -> None:
        """Release references to process-local Lance datasets.

        :param stage: Lightning stage hint; all splits are released together.
        """
        del stage
        self._splits = {}


SurgeXTDataset = LanceMapDataset
SurgeDataModule = LanceVSTDataModule
