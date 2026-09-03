"""Map-style Lance dataloading for VST training and evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import cast

import lance
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import DataLoader

from synth_setter.conditioning import (
    NUM_SKETCH_CONTROLS,
    NUM_SKETCH_TRACK_ROWS,
    SKETCH_CENTROID_CHILD,
    SKETCH_CENTROID_ROW,
    SKETCH_CTRL_FIELD,
    SKETCH_LOUDNESS_CHILD,
    SKETCH_LOUDNESS_ROW,
    SKETCH_PITCH_BINS,
    SKETCH_PITCH_CHILD,
    Conditioning,
    EmbeddingConditioningSpec,
    SketchControls,
    SketchControlSpec,
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
    _validate_param_jitter_amount,
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
    field = dataset.schema.field(column_index)
    shape = _fixed_embedding_shape(field)
    flattened_shape = (prod(spec.input_shape),)
    flattened_fixed_list = pa.types.is_fixed_size_list(field.type) and shape == flattened_shape
    if shape != spec.input_shape and not flattened_fixed_list:
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


def _sketch_child_shapes(num_frames: int) -> dict[str, tuple[int, ...]]:
    """Return the per-row shape each stored sketch struct child must have.

    :param num_frames: Mel-grid frames per stored control row.
    :returns: Expected shapes keyed by struct child name.
    """
    return {
        SKETCH_LOUDNESS_CHILD: (num_frames,),
        SKETCH_CENTROID_CHILD: (num_frames,),
        SKETCH_PITCH_CHILD: (SKETCH_PITCH_BINS, num_frames),
    }


def _stack_sketch_children(
    loudness: np.ndarray, centroid: np.ndarray, pitch: np.ndarray
) -> np.ndarray:
    """Reassemble stored struct children into the flat model control stack.

    Inverts the write-time split bit-for-bit: loudness and centroid land on
    their ``SKETCH_*_ROW`` rows, pitch fills the remaining block.

    :param loudness: ``(B, F)`` loudness rows.
    :param centroid: ``(B, F)`` centroid rows.
    :param pitch: ``(B, SKETCH_PITCH_BINS, F)`` pitch activations.
    :returns: ``(B, NUM_SKETCH_CONTROLS, F)`` stacked controls.
    """
    tracks = np.empty(
        (len(loudness), NUM_SKETCH_TRACK_ROWS, loudness.shape[-1]), dtype=loudness.dtype
    )
    tracks[:, SKETCH_LOUDNESS_ROW] = loudness
    tracks[:, SKETCH_CENTROID_ROW] = centroid
    return np.concatenate([tracks, pitch], axis=1)


def _validate_sketch_column(shard_path: Path, sketch: SketchControlSpec) -> None:
    """Validate one Lance split against the nested sketch storage layout (#2707).

    :param shard_path: Lance dataset selected for a Lightning split.
    :param sketch: Configured sketch-control spec.
    :raises KeyError: If the configured struct column is absent.
    :raises ValueError: If the column is not the nested struct (e.g. a legacy flat dataset), a
        child is missing or mis-shaped, the split is empty, or its sample is non-finite.
    """
    dataset = lance.dataset(str(shard_path))
    column_index = dataset.schema.get_field_index(sketch.column)
    if column_index < 0:
        raise KeyError(f"sketch column {sketch.column!r} is absent from {shard_path}")
    field = dataset.schema.field(column_index)
    if not pa.types.is_struct(field.type):
        raise ValueError(
            f"sketch column {sketch.column!r} in {shard_path} has non-struct type "
            f"{field.type}; this dataset stores the pre-#2707 flat layout — "
            "re-run the sketch add-embeddings backfill to rewrite it as a struct"
        )
    for child, expected_shape in _sketch_child_shapes(sketch.num_frames).items():
        child_index = field.type.get_field_index(child)
        if child_index < 0:
            raise ValueError(
                f"sketch column {sketch.column!r} in {shard_path} is missing "
                f"struct child {child!r}"
            )
        shape = _fixed_embedding_shape(field.type.field(child_index))
        if shape != expected_shape:
            raise ValueError(
                f"sketch child {child!r} in {shard_path} has shape {shape}, "
                f"expected {expected_shape}"
            )
    if dataset.count_rows() == 0:
        raise ValueError(
            f"sketch column {sketch.column!r} cannot be sampled from empty {shard_path}"
        )
    sample = dataset.take([0], columns=[sketch.column]).combine_chunks()
    tensors = batch_to_shaped_tensors(sample.to_batches()[0])
    for child in _sketch_child_shapes(sketch.num_frames):
        if not torch.isfinite(tensors[f"{sketch.column}.{child}"]).all():
            raise ValueError(
                f"sketch child {child!r} sample in {shard_path} contains non-finite values"
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
        param_jitter_amount: float = 0.0,
        conditioning_column: str | None = None,
        conditioning_shape: tuple[int, ...] | None = None,
        sketch_column: str | None = None,
        sketch_pitch_zero_threshold: float | None = None,
        preserve_legacy_m2l: bool = False,
    ) -> None:
        """Configure model-batch transformation semantics.

        :param mean: Mel mean, or ``None`` to skip normalization.
        :param std: Mel standard deviation, or ``None`` to skip normalization.
        :param rescale_params: Whether to map parameters to ``[-1, 1]``.
        :param ot: Whether to Hungarian-match noise to parameters.
        :param param_jitter_amount: Maximum absolute uniform offset in the encoded
            ``[0, 1]`` domain; zero disables jitter.
        :param conditioning_column: Generic embedding column to expose as ``conditioning``.
        :param conditioning_shape: Per-row model shape restored from flattened storage.
        :param sketch_column: Stored sketch struct column whose expanded
            children are reassembled into ``sketch_ctrl``.
        :param sketch_pitch_zero_threshold: Pitch zero-bin threshold (#2614),
            or ``None`` to pass activations through unbinned.
        :param preserve_legacy_m2l: Whether ``music2latent`` also populates ``m2l``.
        """
        self.mean = mean
        self.std = std
        _validate_param_jitter_amount(param_jitter_amount)
        self.rescale_params = rescale_params
        self.ot = ot
        self.param_jitter_amount = param_jitter_amount
        self.conditioning_column = conditioning_column
        self.conditioning_shape = conditioning_shape
        self.sketch_column = sketch_column
        self.sketch_pitch_zero_threshold = sketch_pitch_zero_threshold
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
            if self.conditioning_shape is not None:
                conditioning = conditioning.reshape(len(conditioning), *self.conditioning_shape)
            raw_values["conditioning"] = conditioning
            if self.conditioning_column == "music2latent" and not self.preserve_legacy_m2l:
                del raw_values["music2latent"]
        if self.sketch_column is not None:
            prefix = f"{self.sketch_column}."
            loudness = raw_values.pop(f"{prefix}{SKETCH_LOUDNESS_CHILD}")
            centroid = raw_values.pop(f"{prefix}{SKETCH_CENTROID_CHILD}")
            pitch = raw_values.pop(f"{prefix}{SKETCH_PITCH_CHILD}")
            # Unread companions (e.g. the vec child) never reach prepare_batch.
            for key in [key for key in raw_values if key.startswith(prefix)]:
                del raw_values[key]
            raw_values[SKETCH_CTRL_FIELD] = _stack_sketch_children(loudness, centroid, pitch)
        raw = cast(RawBatch, raw_values)
        return prepare_batch(
            raw,
            mean=self.mean,
            std=self.std,
            rescale_params=self.rescale_params,
            ot=self.ot,
            generator=self._live_generator(),
            sketch_pitch_zero_threshold=self.sketch_pitch_zero_threshold,
            param_jitter_amount=self.param_jitter_amount,
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
        sketch: SketchControlSpec | None = None,
    ) -> None:
        """Configure synthetic sample shapes and epoch length.

        :param batch_size: Samples per batch, used to retain 10,000 batches per epoch.
        :param num_params: Width of parameter and noise tensors.
        :param read_audio: Whether generated samples include prediction audio.
        :param conditioning: Synthetic conditioning modality to populate.
        :param sketch: Optional sketch spec adding synthetic control matrices.
        """
        self._num_rows = batch_size * _FAKE_BATCHES_PER_EPOCH
        self._num_params = num_params
        self._read_audio = read_audio or conditioning == "audio"
        self._read_mel = conditioning == "mel"
        self._preserve_legacy_m2l = (
            isinstance(conditioning, str) and conditioning == "m2l"
        )
        self._embedding_conditioning = resolve_embedding_conditioning(conditioning)
        self._sketch = sketch

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
        mel = torch.randn(num_rows, *_FAKE_MEL_SHAPE) if self._read_mel else None
        conditioning = (
            torch.randn(num_rows, *self._embedding_conditioning.input_shape)
            if self._embedding_conditioning is not None
            else None
        )
        m2l = conditioning if self._preserve_legacy_m2l else None
        if self._sketch is not None:
            # Stored layout: signed-unit loudness/centroid, unit-interval pitch.
            sketch = torch.rand(num_rows, NUM_SKETCH_CONTROLS, self._sketch.num_frames)
            sketch[:, :NUM_SKETCH_TRACK_ROWS] = sketch[:, :NUM_SKETCH_TRACK_ROWS] * 2 - 1
        else:
            sketch = None
        params = torch.rand(num_rows, self._num_params) * 2 - 1
        noise = torch.randn_like(params)
        return {
            "mel": mel,
            "m2l": m2l,
            "conditioning": conditioning,
            "sketch_ctrl": sketch,
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


@dataclass(frozen=True)
class _FakeBatchCollate:
    """Apply training augmentation after fake sample selection and repetition.

    .. attribute :: param_jitter_amount

        Maximum normalized-domain offset applied to targets.
    """

    param_jitter_amount: float

    def __call__(self, batch: object) -> ModelBatch:
        """Return a fake batch with fresh bounded parameter jitter when enabled.

        :param batch: Pre-collated synthetic model batch.
        :returns: Model batch with independently jittered parameter targets.
        """
        model_batch = cast(ModelBatch, batch)
        if self.param_jitter_amount == 0:
            return model_batch

        params = cast(torch.Tensor, model_batch["params"])
        signed_jitter_amount = 2 * self.param_jitter_amount
        jitter = torch.empty_like(params).uniform_(
            -signed_jitter_amount, signed_jitter_amount
        )
        return {
            **model_batch,
            "params": (params + jitter).clamp(-1.0, 1.0),
        }


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
        param_jitter_amount: float = 0.0,
        num_workers: int = 0,
        val_num_workers: int = 0,
        fake: bool = False,
        repeat_first_batch: bool = False,
        predict_file: str | Path | None = None,
        conditioning: Conditioning = "mel",
        sketch: SketchControls = None,
        pin_memory: bool = True,
        param_spec_name: ParamSpecName,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        download_dataset_txids: dict[str, str] | None = None,
        download_dataset_row_limit: int | None = None,
        high_memory_materialization: bool = False,
    ) -> None:
        """Store map-style Lance loader configuration.

        :param dataset_root: Local directory holding per-split Lance datasets.
        :param download_dataset_root_uri: R2 or file URI used to hydrate ``dataset_root``.
        :param use_saved_mean_and_variance: Whether to apply saved mel statistics.
        :param batch_size: Samples per model batch.
        :param ot: Whether training batches use optimal-transport matching.
        :param param_jitter_amount: Training-only maximum absolute uniform offset in
            the normalized parameter domain; zero disables jitter.
        :param num_workers: Worker processes for training, test, and prediction loaders.
        :param val_num_workers: Worker processes for the validation loader.
        :param fake: Whether to synthesize samples instead of reading Lance.
        :param repeat_first_batch: Whether non-predict loaders repeat their first batch.
        :param predict_file: Prediction split; defaults to ``test.lance``.
        :param conditioning: Legacy mel/m2l mode or a fixed-shape embedding spec.
        :param sketch: Optional sketch-control spec adding its stored column to
            every split's read set (#2612).
        :param pin_memory: Whether dataloaders pin returned tensors.
        :param param_spec_name: Registry key selecting parameter width.
        :param persistent_workers: Whether positive worker counts persist between iterators.
        :param prefetch_factor: Batches prefetched per worker; ``None`` keeps
            PyTorch's default, and in-process loading ignores it.
        :param download_dataset_txids: Per-split transaction uuids pinning the
            source snapshots.
        :param download_dataset_row_limit: First-N rows per split at materialization
            time. Without txids, disposable runs use the latest source snapshots.
        :param high_memory_materialization: Whether to use high-memory Lance tuning.
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
            sketch=sketch,
            pin_memory=pin_memory,
            param_spec_name=param_spec_name,
            download_dataset_txids=download_dataset_txids,
            download_dataset_row_limit=download_dataset_row_limit,
            high_memory_materialization=high_memory_materialization,
        )
        _validate_param_jitter_amount(param_jitter_amount)
        self.param_jitter_amount = param_jitter_amount
        self.val_num_workers = val_num_workers
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
        param_jitter_amount: float,
        read_audio: bool,
        stats: tuple[np.ndarray, np.ndarray] | None,
    ) -> _MapSplit:
        """Build one real Lance split and its batch transformer.

        :param shard_path: Lance dataset directory.
        :param ot: Whether to match batch noise to parameters.
        :param param_jitter_amount: Maximum absolute uniform parameter offset.
        :param read_audio: Whether to project prediction audio.
        :param stats: Mel ``(mean, std)``, or ``None`` to skip normalization.
        :returns: Sample-indexed dataset and collate operation.
        """
        spec = self.embedding_conditioning
        if spec is not None:
            _validate_embedding_column(shard_path, spec)
        sketch = self.sketch_controls
        if sketch is not None:
            _validate_sketch_column(shard_path, sketch)
        columns = self._loader_columns(read_audio=read_audio)
        mean, std = stats if stats is not None else (None, None)
        return _MapSplit(
            dataset=LanceMapDataset(shard_path, columns=columns),
            collate=PrepareBatchCollate(
                mean=mean,
                std=std,
                rescale_params=True,
                ot=ot,
                param_jitter_amount=param_jitter_amount,
                conditioning_column=spec.column if spec is not None else None,
                conditioning_shape=spec.input_shape if spec is not None else None,
                sketch_column=sketch.column if sketch is not None else None,
                sketch_pitch_zero_threshold=(
                    sketch.pitch_zero_threshold if sketch is not None else None
                ),
                preserve_legacy_m2l=(
                    isinstance(self.conditioning, str) and self.conditioning == "m2l"
                ),
            ),
        )

    def _build_fake_split(
        self, *, num_params: int, read_audio: bool, param_jitter_amount: float
    ) -> _MapSplit:
        """Build one sample-indexed in-memory split.

        :param num_params: Selected parameter-spec width.
        :param read_audio: Whether prediction audio is generated.
        :param param_jitter_amount: Maximum normalized-domain offset applied to targets.
        :returns: Synthetic dataset and pass-through collate.
        """
        return _MapSplit(
            dataset=_FakeMapDataset(
                batch_size=self.batch_size,
                num_params=num_params,
                read_audio=read_audio,
                conditioning=self.conditioning,
                sketch=self.sketch_controls,
            ),
            collate=_FakeBatchCollate(param_jitter_amount),
        )

    def _build_real_splits(self, split_names: Sequence[str]) -> dict[str, _MapSplit]:
        """Build the requested on-disk Lance splits.

        :param split_names: Split names required by the current stage.
        :returns: Requested split datasets and collate operations.
        """
        train_shard = self.dataset_root / f"train{self.shard_suffix}"
        split_stats = predict_stats = None
        if self.use_saved_mean_and_variance and self._conditioning_column() == "mel_spec":
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
                param_jitter_amount=(
                    self.param_jitter_amount if name == "train" else 0.0
                ),
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
                    num_params=num_params,
                    read_audio=name == "predict",
                    param_jitter_amount=(
                        self.param_jitter_amount if name == "train" else 0.0
                    ),
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
        num_workers = self.val_num_workers if split == "val" else self.num_workers
        return map_dataloader_over(
            dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
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
