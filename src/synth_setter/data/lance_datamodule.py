"""Map-style Lance dataloading for VST training and evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from synth_setter.conditioning import ConditioningMode
from synth_setter.data.lance_torch import LanceMapDataset, map_dataloader_over
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
_FAKE_M2L_SHAPE = (128, 42)
_FAKE_MEL_SHAPE = (2, 128, 401)

type ModelBatch = dict[str, torch.Tensor | None]


class PrepareBatchCollate:
    """Transform pre-collated Lance columns with a process-local noise RNG."""

    def __init__(
        self,
        *,
        mean: np.ndarray | None,
        std: np.ndarray | None,
        rescale_params: bool,
        ot: bool,
    ) -> None:
        """Configure model-batch transformation semantics.

        :param mean: Mel mean, or ``None`` to skip normalization.
        :param std: Mel standard deviation, or ``None`` to skip normalization.
        :param rescale_params: Whether to map parameters to ``[-1, 1]``.
        :param ot: Whether to Hungarian-match noise to parameters.
        """
        self.mean = mean
        self.std = std
        self.rescale_params = rescale_params
        self.ot = ot
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
        raw = cast(RawBatch, {name: tensor.numpy() for name, tensor in columns.items()})
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
        conditioning: ConditioningMode,
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
        self._conditioning = conditioning

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
            if self._conditioning == "mel"
            else None
        )
        m2l = (
            torch.randn(num_rows, *_FAKE_M2L_SHAPE)
            if self._conditioning == "m2l"
            else None
        )
        params = torch.rand(num_rows, self._num_params) * 2 - 1
        noise = torch.randn_like(params)
        return {
            "mel_spec": mel_spec,
            "m2l": m2l,
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
    """Read every VST split through sample-indexed map semantics.

    .. attribute :: train_dataset

       Sample-indexed training dataset.

    .. attribute :: val_dataset

       Sample-indexed validation dataset.

    .. attribute :: test_dataset

       Sample-indexed test dataset.

    .. attribute :: predict_dataset

       Sample-indexed prediction dataset.
    """

    train_dataset: _SplitDataset
    val_dataset: _SplitDataset
    test_dataset: _SplitDataset
    predict_dataset: _SplitDataset

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
        conditioning: ConditioningMode = "mel",
        pin_memory: bool = True,
        param_spec_name: ParamSpecName,
        persistent_workers: bool = False,
    ) -> None:
        """Store map-style Lance loader configuration.

        :param dataset_root: Local directory holding per-split Lance datasets.
        :param download_dataset_root_uri: R2 URI used to hydrate ``dataset_root``.
        :param use_saved_mean_and_variance: Whether to apply saved mel statistics.
        :param batch_size: Samples per model batch.
        :param ot: Whether training batches use optimal-transport matching.
        :param num_workers: Worker processes per dataloader.
        :param fake: Whether to synthesize samples instead of reading Lance.
        :param repeat_first_batch: Whether non-predict loaders repeat their first batch.
        :param predict_file: Prediction split; defaults to ``test.lance``.
        :param conditioning: Conditioning feature, ``"mel"`` or ``"m2l"``.
        :param pin_memory: Whether dataloaders pin returned tensors.
        :param param_spec_name: Registry key selecting parameter width.
        :param persistent_workers: Whether positive worker counts persist between iterators.
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
        )
        self.persistent_workers = persistent_workers
        self._splits: dict[str, _MapSplit] = {}

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
        columns = ["param_array"]
        columns.append("mel_spec" if self.conditioning == "mel" else "music2latent")
        if read_audio:
            columns.append("audio")
        mean, std = stats if stats is not None else (None, None)
        return _MapSplit(
            dataset=LanceMapDataset(shard_path, columns=columns),
            collate=PrepareBatchCollate(
                mean=mean,
                std=std,
                rescale_params=True,
                ot=ot,
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

    def setup(self, stage: str | None = None) -> None:
        """Build all sample-indexed train, validation, test, and prediction splits.

        :param stage: Lightning stage hint; all splits are built eagerly.
        """
        del stage
        num_params = resolve_param_spec(self.param_spec_name).encoded_width
        if self.fake:
            self._splits = {
                "train": self._build_fake_split(num_params=num_params, read_audio=False),
                "val": self._build_fake_split(num_params=num_params, read_audio=False),
                "test": self._build_fake_split(num_params=num_params, read_audio=False),
                "predict": self._build_fake_split(num_params=num_params, read_audio=True),
            }
        else:
            train_shard = self.dataset_root / f"train{self.shard_suffix}"
            split_stats = predict_stats = None
            if self.use_saved_mean_and_variance:
                split_stats = load_dataset_statistics(train_shard)
                predict_stats = (
                    split_stats
                    if self.predict_file.parent == self.dataset_root
                    else load_dataset_statistics(self.predict_file)
                )
            self._splits = {
                "train": self._build_lance_split(
                    train_shard, ot=self.ot, read_audio=False, stats=split_stats
                ),
                "val": self._build_lance_split(
                    self.dataset_root / f"val{self.shard_suffix}",
                    ot=False,
                    read_audio=False,
                    stats=split_stats,
                ),
                "test": self._build_lance_split(
                    self.dataset_root / f"test{self.shard_suffix}",
                    ot=False,
                    read_audio=False,
                    stats=split_stats,
                ),
                "predict": self._build_lance_split(
                    self.predict_file,
                    ot=False,
                    read_audio=True,
                    stats=predict_stats,
                ),
            }
        self.train_dataset = self._splits["train"].dataset
        self.val_dataset = self._splits["val"].dataset
        self.test_dataset = self._splits["test"].dataset
        self.predict_dataset = self._splits["predict"].dataset

    def _dataloader(self, split: str, *, shuffle: bool, drop_last: bool) -> DataLoader:
        """Build one standard map-style dataloader.

        :param split: Split key created by :meth:`setup`.
        :param shuffle: Whether to randomize sample order.
        :param drop_last: Whether to discard a ragged final batch.
        :returns: Dataloader yielding model-ready batches.
        """
        pieces = self._splits[split]
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
        for name in ("train_dataset", "val_dataset", "test_dataset", "predict_dataset"):
            if hasattr(self, name):
                delattr(self, name)


SurgeXTDataset = LanceMapDataset
SurgeDataModule = LanceVSTDataModule
