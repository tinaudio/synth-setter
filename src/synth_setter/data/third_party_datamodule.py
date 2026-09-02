"""Predict-only datamodule serving published third-party audio corpora (#2886).

The corpora under ``r2:experiments/third_party`` store source WAV bytes as
``lance.blob.v2`` columns and are never rewritten. Decode, resample, up-mix,
length-pinning, amplitude scaling, and the mel front-end happen per batch.
Compose ``datamodule=third_party/nsynth_test`` with
``evaluation.no_params=true evaluation.rerender_target=false`` for prediction.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import lance
import numpy as np
import pyarrow as pa
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from lightning import LightningDataModule
from pedalboard.io import AudioFile
from torch.utils.data import DataLoader, Dataset

from synth_setter.data.vst.shapes import AUDIO_FIELD, make_spectrogram
from synth_setter.data.vst_datamodule import load_mel_statistics
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.lance_materialize import _retry_lance_read

log = logging.getLogger(__name__)

_PREDICT_STAGES = frozenset({"predict", None})
_MEL_STATS_CACHE_DIR = ".mel-stats"
_BLOB_EXTENSION_NAME = "lance.blob.v2"


def _is_blob_encoded(field: pa.Field) -> bool:
    """Return whether a column is readable through the blob API.

    :param field: Schema field for the configured audio column.
    :returns: True when either supported blob encoding is present.
    """
    if getattr(field.type, "extension_name", None) == _BLOB_EXTENSION_NAME:
        return True
    return (field.metadata or {}).get(b"lance-encoding:blob") == b"true"


def _validate_config(
    conditioning: str,
    *,
    use_saved_mean_and_variance: bool,
    mel_stats_uri: str | None,
    row_limit: int | None,
) -> None:
    """Reject a configuration that cannot be served correctly.

    :param conditioning: Conditioning mode; this layer accepts only ``mel``.
    :param use_saved_mean_and_variance: Whether mel standardization is enabled.
    :param mel_stats_uri: Configured statistics source, if any.
    :param row_limit: Configured row cap, if any.
    :raises ValueError: The conditioning mode or normalization configuration is invalid.
    """
    if conditioning != "mel":
        raise ValueError(
            f"ThirdPartyAudioDataModule accepts mel conditioning only, got {conditioning!r}"
        )
    if not isinstance(use_saved_mean_and_variance, bool):
        raise ValueError(
            "use_saved_mean_and_variance must be a boolean, got "
            f'{use_saved_mean_and_variance!r}; a quoted "false" would otherwise enable '
            "normalization"
        )
    if row_limit is not None and (not isinstance(row_limit, int) or isinstance(row_limit, bool)):
        raise ValueError(f"row_limit must be an integer, got {row_limit!r}")
    if row_limit is not None and row_limit < 1:
        raise ValueError(
            f"row_limit must be at least 1, got {row_limit}; an empty sweep writes no "
            "predictions and fails downstream instead of here"
        )
    if use_saved_mean_and_variance and mel_stats_uri is None:
        raise ValueError(
            "mel conditioning with use_saved_mean_and_variance=true requires "
            "mel_stats_uri — point it at the statistics the checkpoint trained "
            "with, not at this corpus"
        )
    if not use_saved_mean_and_variance and mel_stats_uri is not None:
        raise ValueError(
            f"mel_stats_uri={mel_stats_uri!r} is set with use_saved_mean_and_variance=false, "
            "so the statistics would be dropped and the checkpoint fed raw mel"
        )


def _validate_numeric_config(
    *,
    sample_rate: int,
    channels: int,
    signal_duration_seconds: float,
    amplitude_scale: float,
    dataset_version: int,
    batch_size: int,
    num_workers: int,
) -> int:
    """Validate the numeric render contract and return its sample count.

    :param sample_rate: Target sample rate in Hz.
    :param channels: Target channel count.
    :param signal_duration_seconds: Target clip duration.
    :param amplitude_scale: Gain applied to decoded audio.
    :param dataset_version: Immutable Lance snapshot to serve.
    :param batch_size: Rows per predict batch.
    :param num_workers: Dataloader workers decoding rows.
    :returns: Target samples per clip.
    :raises ValueError: A value has the wrong type or lies outside its valid domain.
    """
    for name, value in (
        ("sample_rate", sample_rate),
        ("channels", channels),
        ("dataset_version", dataset_version),
        ("batch_size", batch_size),
        ("num_workers", num_workers),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name}={value!r} must be an integer")
    if isinstance(signal_duration_seconds, bool) or not isinstance(
        signal_duration_seconds, (int, float)
    ):
        raise ValueError(f"signal_duration_seconds={signal_duration_seconds!r} must be a number")
    if isinstance(amplitude_scale, bool) or not isinstance(amplitude_scale, (int, float)):
        raise ValueError(f"amplitude_scale={amplitude_scale!r} must be a number")
    if dataset_version <= 0:
        raise ValueError(f"dataset_version={dataset_version} must be positive")
    if batch_size <= 0:
        raise ValueError(f"batch_size={batch_size} must be positive")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate={sample_rate} must be positive")
    if not math.isfinite(signal_duration_seconds) or signal_duration_seconds <= 0:
        raise ValueError(
            f"signal_duration_seconds={signal_duration_seconds} must be positive and finite"
        )
    if not math.isfinite(amplitude_scale) or amplitude_scale <= 0:
        raise ValueError(f"amplitude_scale={amplitude_scale} must be positive and finite")
    if num_workers < 0:
        raise ValueError(f"num_workers={num_workers} must not be negative")
    num_samples = int(sample_rate * signal_duration_seconds)
    if num_samples <= 0:
        raise ValueError(
            f"sample_rate={sample_rate} x signal_duration_seconds="
            f"{signal_duration_seconds} yields {num_samples} samples per clip; "
            "the render contract needs a positive sample count"
        )
    if channels <= 0:
        raise ValueError(f"channels={channels} must be positive")
    return num_samples


def decode_clip(
    data: bytes,
    *,
    sample_rate: int,
    channels: int,
    num_samples: int,
    amplitude_scale: float,
) -> np.ndarray:
    """Decode one source clip onto the render contract's audio grid.

    :param data: Source container bytes in any format pedalboard reads.
    :param sample_rate: Target sample rate in Hz.
    :param channels: Target channel count; a mono source is duplicated.
    :param num_samples: Target sample count; shorter clips pad, longer ones truncate.
    :param amplitude_scale: Gain applied after length-pinning.
    :returns: ``(channels, num_samples)`` float32 audio.
    :raises ValueError: Samples are non-finite, out of range, or have an unsupported channel count.
    """
    with AudioFile(io.BytesIO(data)).resampled_to(sample_rate) as handle:
        audio = handle.read(handle.frames)
    if audio.shape[0] == 1 < channels:
        audio = np.repeat(audio, channels, axis=0)
    elif audio.shape[0] != channels:
        raise ValueError(f"source has {audio.shape[0]} channels; render contract wants {channels}")
    if audio.shape[1] < num_samples:
        audio = np.pad(audio, [(0, 0), (0, num_samples - audio.shape[1])])
    clip = np.ascontiguousarray(audio[:, :num_samples] * amplitude_scale, dtype=np.float32)
    if not np.isfinite(clip).all():
        raise ValueError("decoded audio contains non-finite samples")
    if np.abs(clip).max(initial=0.0) > 1.0:
        raise ValueError(
            f"decoded audio leaves [-1, 1] after amplitude_scale={amplitude_scale}; "
            "the mel front-end and model contract assume normalized audio"
        )
    return clip


class _BlobAudioDataset(Dataset[dict[str, torch.Tensor]]):
    """Row-indexed corpus view decoding blob audio and mel on the worker."""

    def __init__(
        self,
        uri: str,
        *,
        storage_options: Mapping[str, str] | None,
        version: int,
        audio_column: str,
        sample_rate: int,
        channels: int,
        num_samples: int,
        amplitude_scale: float,
        rows: int,
    ) -> None:
        """Configure the per-row decode.

        :param uri: Lance dataset location.
        :param storage_options: Object-store options for a remote ``uri``.
        :param version: Lance dataset version every reader pins to.
        :param audio_column: Blob column holding source container bytes.
        :param sample_rate: Target sample rate in Hz.
        :param channels: Target channel count.
        :param num_samples: Target sample count per clip.
        :param amplitude_scale: Gain applied to decoded audio.
        :param rows: Number of rows served.
        """
        self.uri = uri
        self.storage_options = dict(storage_options) if storage_options else None
        self.version = version
        self.audio_column = audio_column
        self.sample_rate = sample_rate
        self.channels = channels
        self.num_samples = num_samples
        self.amplitude_scale = amplitude_scale
        self.rows = rows
        self._dataset: lance.LanceDataset | None = None

    def __getstate__(self) -> dict[str, object]:
        """Drop the open dataset so every worker opens its own handle.

        :returns: Pickle state with no live Lance handle.
        """
        return {**self.__dict__, "_dataset": None}

    def __len__(self) -> int:
        """Expose the configured served-row count.

        :returns: Rows this dataset serves.
        """
        return self.rows

    def _open(self) -> lance.LanceDataset:
        """Return this process's dataset handle, opening it on first use.

        :returns: Open Lance dataset.
        """
        if self._dataset is None:
            self._dataset = _retry_lance_read(
                "third_party_worker_open",
                lambda: lance.dataset(
                    self.uri, version=self.version, storage_options=self.storage_options
                ),
            )
        return self._dataset

    def _decode(self, data: bytes) -> dict[str, torch.Tensor]:
        """Decode one stored container into model audio and mel tensors.

        :param data: Source audio container bytes.
        :returns: ``audio`` and ``mel`` for one clip.
        """
        audio = decode_clip(
            data,
            sample_rate=self.sample_rate,
            channels=self.channels,
            num_samples=self.num_samples,
            amplitude_scale=self.amplitude_scale,
        )
        mel = make_spectrogram(audio, self.sample_rate).astype(np.float32)
        return {"audio": torch.from_numpy(audio), "mel": torch.from_numpy(mel)}

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, torch.Tensor]]:
        """Decode one ordered index batch through Lance's native blob scheduler.

        :param indices: Row indices, including any requested duplicates.
        :returns: One decoded sample per index in the requested order.
        """
        selected = list(indices)
        blobs = _retry_lance_read(
            "third_party_blob_read",
            lambda: self._open().read_blobs(
                self.audio_column,
                indices=selected,
                preserve_order=True,
            ),
        )
        return [self._decode(data) for _, data in blobs]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Decode one row onto the render contract.

        :param index: Row index.
        :returns: ``audio`` and ``mel`` for one clip.
        """
        return self.__getitems__((index,))[0]


class ThirdPartyAudioDataModule(LightningDataModule):
    """Serve an immutable third-party audio corpus to ``trainer.predict``.

    Emits no ``params``: these corpora carry no ground-truth patch, so an eval
    run against one must also disable target re-rendering.
    """

    def __init__(
        self,
        dataset_uri: str,
        *,
        sample_rate: int,
        channels: int,
        signal_duration_seconds: float,
        dataset_version: int,
        audio_column: str = AUDIO_FIELD,
        amplitude_scale: float = 1.0,
        batch_size: int = 32,
        num_workers: int = 0,
        row_limit: int | None = None,
        conditioning: str = "mel",
        use_saved_mean_and_variance: bool = False,
        mel_stats_uri: str | None = None,
        stats_cache_dir: str | None = None,
    ) -> None:
        """Configure the corpus and render contract it maps onto.

        :param dataset_uri: Corpus Lance dataset; local path or ``r2://`` URI.
        :param sample_rate: Target sample rate in Hz.
        :param channels: Target channel count.
        :param signal_duration_seconds: Target clip duration.
        :param dataset_version: Immutable Lance snapshot to serve.
        :param audio_column: Blob column holding source container bytes.
        :param amplitude_scale: Gain applied to decoded audio before the mel front-end.
        :param batch_size: Rows per predict batch.
        :param num_workers: Dataloader workers decoding rows.
        :param row_limit: Serve only the first N rows; ``None`` serves the whole corpus.
        :param conditioning: Conditioning mode; only ``mel`` is accepted.
        :param use_saved_mean_and_variance: Whether to standardize mel with saved statistics.
        :param mel_stats_uri: Training mel statistics, local or ``r2://``.
        :param stats_cache_dir: Directory for fetched statistics.
        """
        super().__init__()
        _validate_config(
            conditioning,
            use_saved_mean_and_variance=use_saved_mean_and_variance,
            mel_stats_uri=mel_stats_uri,
            row_limit=row_limit,
        )
        num_samples = _validate_numeric_config(
            sample_rate=sample_rate,
            channels=channels,
            signal_duration_seconds=signal_duration_seconds,
            amplitude_scale=amplitude_scale,
            dataset_version=dataset_version,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.dataset_uri = dataset_uri
        self.dataset_version = dataset_version
        self.sample_rate = sample_rate
        self.channels = channels
        self.num_samples = num_samples
        self.audio_column = audio_column
        self.amplitude_scale = amplitude_scale
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.row_limit = row_limit
        self.conditioning = conditioning
        self.mel_stats_uri = mel_stats_uri
        self.stats_cache_dir = Path(stats_cache_dir or Path.cwd() / _MEL_STATS_CACHE_DIR)
        self._statistics: tuple[torch.Tensor, torch.Tensor] | None = None
        self._predict_dataset: _BlobAudioDataset | None = None
        self._resolved_dataset_version: int | None = None

    @property
    def served_row_count(self) -> int:
        """Return the number of rows available through explicit prediction reads.

        :returns: Served rows after applying ``row_limit``.
        :raises RuntimeError: ``setup('predict')`` has not run.
        """
        if self._predict_dataset is None:
            raise RuntimeError("predict split is not built; call setup('predict') first")
        return len(self._predict_dataset)

    @property
    def resolved_dataset_version(self) -> int:
        """Return the immutable Lance version opened during prediction setup.

        :returns: Resolved Lance snapshot version.
        :raises RuntimeError: ``setup('predict')`` has not run.
        """
        if self._resolved_dataset_version is None:
            raise RuntimeError("predict split is not built; call setup('predict') first")
        return self._resolved_dataset_version

    @jaxtyped(typechecker=beartype)
    def audio_rows(
        self, indices: Sequence[int]
    ) -> Float[torch.Tensor, "rows channels samples"]:
        """Decode explicit corpus rows onto this datamodule's model audio grid.

        :param indices: Row indices in the requested output order; duplicates are allowed.
        :returns: Float32 audio with shape ``(rows, channels, samples)``.
        :raises RuntimeError: ``setup('predict')`` has not run.
        :raises IndexError: An index is not an integer in the served row range.
        """
        if self._predict_dataset is None:
            raise RuntimeError("predict split is not built; call setup('predict') first")
        selected = list(indices)
        row_count = len(self._predict_dataset)
        for index in selected:
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < row_count:
                raise IndexError(
                    f"row index {index!r} is outside served range [0, {row_count})"
                )
        if not selected:
            return torch.empty(
                (0, self.channels, self.num_samples),
                dtype=torch.float32,
            )
        samples = self._predict_dataset.__getitems__(selected)
        return torch.stack([sample["audio"] for sample in samples])

    def cached_stats_path(self) -> Path:
        """Return the local path the configured statistics object resolves to.

        :returns: Local ``.npz`` path, unique per ``r2://`` URI.
        """
        uri = cast(str, self.mel_stats_uri)
        if not r2_io.is_r2_uri(uri):
            return Path(uri)
        digest = hashlib.sha256(uri.encode()).hexdigest()[:16]
        return self.stats_cache_dir / f"{digest}-{Path(uri).name}"

    def _local_stats_file(self) -> Path:
        """Return a readable local path for the configured statistics object.

        :returns: Local ``.npz`` path, downloaded once per distinct ``r2://`` URI.
        """
        destination = self.cached_stats_path()
        if not r2_io.is_r2_uri(cast(str, self.mel_stats_uri)):
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            staged = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
            r2_io.download_to_path(cast(str, self.mel_stats_uri), staged)
            staged.replace(destination)
        return destination

    def _open_corpus(self) -> tuple[_BlobAudioDataset, int]:
        """Open the corpus, validate its audio column, and build the predict split.

        :returns: The predict dataset and pinned Lance version.
        :raises KeyError: The corpus has no configured audio column.
        :raises ValueError: The audio column is not blob-encoded or the corpus is empty.
        """
        uri, storage_options = (
            r2_io.lance_target(self.dataset_uri)
            if r2_io.is_r2_uri(self.dataset_uri)
            else (self.dataset_uri, None)
        )
        dataset = _retry_lance_read(
            "third_party_corpus_open",
            lambda: lance.dataset(
                uri,
                version=self.dataset_version,
                storage_options=storage_options,
            ),
        )
        if self.audio_column not in dataset.schema.names:
            raise KeyError(
                f"corpus {self.dataset_uri} has no {self.audio_column!r} column; "
                f"columns: {', '.join(dataset.schema.names)}"
            )
        if not _is_blob_encoded(dataset.schema.field(self.audio_column)):
            raise ValueError(
                f"column {self.audio_column!r} in {self.dataset_uri} is not blob-encoded, "
                "so its source containers cannot be read through the blob API"
            )
        rows = _retry_lance_read("third_party_row_count", dataset.count_rows)
        if rows == 0:
            raise ValueError(
                f"corpus {self.dataset_uri} has no rows; an empty sweep writes no "
                "predictions and fails downstream instead of here"
            )
        log.info(
            "third-party corpus %s pinned at version %s", self.dataset_uri, dataset.version
        )
        return (
            _BlobAudioDataset(
                uri,
                storage_options=storage_options,
                version=dataset.version,
                audio_column=self.audio_column,
                sample_rate=self.sample_rate,
                channels=self.channels,
                num_samples=self.num_samples,
                amplitude_scale=self.amplitude_scale,
                rows=rows if self.row_limit is None else min(rows, self.row_limit),
            ),
            dataset.version,
        )

    def setup(self, stage: str | None = None) -> None:
        """Open the corpus and load mel statistics for prediction.

        :param stage: Lightning stage hint; only prediction is served.
        :raises ValueError: The stage is unsupported or statistics are invalid in float32.
        """
        if stage not in _PREDICT_STAGES:
            raise ValueError(f"{type(self).__name__} serves prediction only, got stage {stage!r}")
        self._predict_dataset, self._resolved_dataset_version = self._open_corpus()
        if self.mel_stats_uri is not None and self._statistics is None:
            mean, std = load_mel_statistics(self._local_stats_file())
            mean_f32 = torch.as_tensor(mean, dtype=torch.float32)
            std_f32 = torch.as_tensor(std, dtype=torch.float32)
            if not bool(torch.isfinite(mean_f32).all() and torch.isfinite(std_f32).all()):
                raise ValueError(
                    f"mel statistics from {self.mel_stats_uri} are not representable in float32"
                )
            if not bool((std_f32 > 0).all()):
                raise ValueError(
                    f"mel statistics from {self.mel_stats_uri} contain standard deviations "
                    "that underflow to zero in float32"
                )
            self._statistics = (mean_f32, std_f32)

    def predict_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        """Return the corpus loader in stored row order.

        :returns: Un-shuffled predict dataloader.
        :raises RuntimeError: ``setup`` has not run.
        """
        if self._predict_dataset is None:
            raise RuntimeError("predict split is not built; call setup('predict') first")
        return DataLoader(
            self._predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def on_before_batch_transfer(
        self, batch: Mapping[str, torch.Tensor], dataloader_idx: int
    ) -> dict[str, torch.Tensor]:
        """Normalize mel conditioning before model transfer.

        :param batch: Decoded ``audio`` and ``mel`` batch.
        :param dataloader_idx: Unused; the datamodule serves one loader.
        :returns: Model batch with normalized mel when configured.
        :raises ValueError: Mel normalization produced non-finite values.
        """
        del dataloader_idx
        model_batch = dict(batch)
        if self._statistics is None:
            return model_batch
        mean, std = self._statistics
        normalized = (model_batch["mel"] - mean) / std
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError(
                f"mel normalization with statistics from {self.mel_stats_uri} produced "
                "non-finite values"
            )
        model_batch["mel"] = normalized
        return model_batch
