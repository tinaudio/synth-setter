"""Predict-only datamodule serving published third-party audio corpora (#2886).

The corpora under ``r2:experiments/third_party`` store source WAV bytes as
``lance.blob.v2`` columns and are never rewritten: decode, resample, up-mix,
length-pinning, the mel front-end, and conditioning all happen per batch here,
so an eval run is a function of the immutable corpus plus this code.

Conditioning is computed through the very ``EMBEDDING_REGISTRY`` entries that
write the stored columns, so a checkpoint trained on a stored column loads
unmodified — the model batch key and per-row shape are identical.

Typical usage selects a corpus and the checkpoint's conditioning through Hydra::

    python -m synth_setter.cli.eval experiment=surge/flow_simple_440k \
      datamodule=third_party/nsynth_test synth=surge_simple render=vst \
      conditioning=clap mode=predict callbacks=eval_vst \
      evaluation.no_params=true evaluation.rerender_target=false \
      ckpt_path=<checkpoint>
"""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import lance
import numpy as np
import pyarrow as pa
import torch
from lightning import LightningDataModule
from pedalboard.io import AudioFile
from torch.utils.data import DataLoader, Dataset

from synth_setter.conditioning import (
    SKETCH_CENTROID_CHILD,
    SKETCH_CTRL_FIELD,
    SKETCH_LOUDNESS_CHILD,
    SKETCH_PITCH_CHILD,
    SKETCH_PITCH_SLICE,
    Conditioning,
    EmbeddingConditioningSpec,
    SketchControls,
    SketchControlSpec,
    resolve_embedding_conditioning,
    resolve_sketch_controls,
)
from synth_setter.data.lance_datamodule import stack_sketch_children
from synth_setter.data.lance_torch import batch_to_shaped_tensors
from synth_setter.data.vst.shapes import AUDIO_FIELD, make_spectrogram
from synth_setter.data.vst_datamodule import load_mel_statistics
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.add_embeddings import EMBEDDING_REGISTRY, EmbeddingSpec
from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig

if TYPE_CHECKING:
    from synth_setter.pipeline.data.add_embeddings import Encoder

_PREDICT_STAGES = frozenset({"predict", None})
_MEL_STATS_CACHE_DIR = ".mel-stats"


def _is_audio_sourced(spec: EmbeddingSpec) -> bool:
    """Return whether a policy can be computed from stored audio alone.

    :param spec: Registry entry.
    :returns: True when audio is its only input and it re-renders nothing.
    """
    return spec.input_fields == (AUDIO_FIELD,) and not spec.rerenders


def registry_spec(column: str) -> EmbeddingSpec:
    """Look up an embedding policy servable from a corpus's audio.

    :param column: Stored column name, which doubles as the registry key.
    :returns: Registry entry for ``column``.
    :raises KeyError: ``column`` is not a registry key.
    :raises ValueError: The policy needs inputs a third-party corpus cannot supply,
        such as stored parameters or a re-render.
    """
    try:
        spec = EMBEDDING_REGISTRY[column]
    except KeyError:
        servable = ", ".join(sorted(name for name, entry in EMBEDDING_REGISTRY.items() if _is_audio_sourced(entry)))
        raise KeyError(
            f"{column!r} is not an embedding registry column; servable columns: {servable}"
        ) from None
    if not _is_audio_sourced(spec):
        raise ValueError(
            f"{column!r} is derived from {', '.join(spec.input_fields)} rather than audio "
            "alone, so a third-party corpus cannot supply its input"
        )
    return spec


class LiveEmbedding:
    """Compute one registry embedding for a batch of audio without storing it."""

    def __init__(self, spec: EmbeddingSpec, encoder: Encoder, *, sample_rate: int) -> None:
        """Bind a registry policy to a loaded encoder.

        :param spec: Registry entry whose ``encode_column`` shapes and validates the output.
        :param encoder: Loaded encoder matching ``spec``'s call signature.
        :param sample_rate: Sample rate of the audio handed to the encoder.
        """
        self.spec = spec
        self.encoder = encoder
        self.sample_rate = sample_rate

    def __call__(self, audio: np.ndarray) -> dict[str, torch.Tensor]:
        """Encode one ``(B, C, T)`` audio batch into stored-layout tensors.

        :param audio: Channel-leading audio batch.
        :returns: One tensor per leaf column; struct children land under dotted keys.
        """
        column = self.spec.encode_column({AUDIO_FIELD: audio}, self.sample_rate, self.encoder)
        return batch_to_shaped_tensors(
            pa.RecordBatch.from_arrays([column], names=[self.spec.column])
        )

    @classmethod
    def from_registry(
        cls,
        column: str,
        *,
        sample_rate: int,
        lance_uri: str,
        device: str | None = None,
        checkpoint: str | None = None,
    ) -> LiveEmbedding:
        """Load the registry encoder writing ``column`` for live use.

        :param column: ``EMBEDDING_REGISTRY`` key, which is also the stored column name.
        :param sample_rate: Sample rate of the audio the encoder will be handed.
        :param lance_uri: Corpus the encoder serves, recorded on the run config.
        :param device: Torch device for the encoder; ``None`` auto-selects.
        :param checkpoint: Checkpoint override; ``None`` uses the registry default.
        :returns: Encoder bound to its registry policy.
        """
        spec = registry_spec(column)
        config = AddEmbeddingsConfig(
            lance_uri=lance_uri, embeddings=(column,), device=device, build_index=False
        )
        return cls(
            spec,
            spec.load_encoder(checkpoint or spec.default_checkpoint, config),
            sample_rate=sample_rate,
        )


def _validate_conditioning_config(
    conditioning: Conditioning,
    *,
    use_saved_mean_and_variance: bool,
    mel_stats_uri: str | None,
    row_limit: int | None,
) -> None:
    """Reject a configuration that cannot be served correctly.

    :param conditioning: Configured conditioning mode or embedding spec.
    :param use_saved_mean_and_variance: Whether mel standardization is enabled.
    :param mel_stats_uri: Configured statistics source, if any.
    :param row_limit: Configured row cap, if any.
    :raises ValueError: Mel is standardized without a statistics source, or the row cap is
        negative.
    """
    if row_limit is not None and row_limit < 0:
        raise ValueError(f"row_limit must be non-negative, got {row_limit}")
    if use_saved_mean_and_variance and mel_stats_uri is None and conditioning == "mel":
        raise ValueError(
            "mel conditioning with use_saved_mean_and_variance=true requires "
            "mel_stats_uri — point it at the statistics the checkpoint trained "
            "with, not at this corpus"
        )


def decode_clip(
    data: bytes,
    *,
    sample_rate: int,
    channels: int,
    num_samples: int,
    amplitude_scale: float,
) -> np.ndarray:
    """Decode one source clip onto the render contract's audio grid.

    The container's own header supplies the source rate, so corpora recorded at
    different rates need no per-corpus configuration.

    :param data: Source container bytes in any format pedalboard reads.
    :param sample_rate: Target sample rate in Hz.
    :param channels: Target channel count; a mono source is duplicated.
    :param num_samples: Target sample count; shorter clips pad, longer ones truncate.
    :param amplitude_scale: Gain applied after length-pinning.
    :returns: ``(channels, num_samples)`` float32 audio.
    :raises ValueError: The source disagrees with the target channel count, or the
        scaled samples leave the ``[-1, 1]`` storage range the model contract assumes.
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
    """Row-indexed corpus view decoding blob audio, and its mel, on the worker."""

    def __init__(
        self,
        uri: str,
        *,
        storage_options: dict[str, str] | None,
        audio_column: str,
        sample_rate: int,
        channels: int,
        num_samples: int,
        amplitude_scale: float,
        with_mel: bool,
        rows: int,
    ) -> None:
        """Configure the per-row decode.

        :param uri: Lance dataset location.
        :param storage_options: Object-store options for a remote ``uri``.
        :param audio_column: Blob column holding source container bytes.
        :param sample_rate: Target sample rate in Hz.
        :param channels: Target channel count.
        :param num_samples: Target sample count per clip.
        :param amplitude_scale: Gain applied to decoded audio.
        :param with_mel: Whether to compute the training mel front-end.
        :param rows: Number of rows served.
        """
        self.uri = uri
        self.storage_options = storage_options
        self.audio_column = audio_column
        self.sample_rate = sample_rate
        self.channels = channels
        self.num_samples = num_samples
        self.amplitude_scale = amplitude_scale
        self.with_mel = with_mel
        self.rows = rows
        self._dataset: lance.LanceDataset | None = None

    def __getstate__(self) -> dict[str, object]:
        """Drop the open dataset so every worker opens its own handle.

        :returns: Pickle state with no live Lance handle.
        """
        return {**self.__dict__, "_dataset": None}

    def __len__(self) -> int:
        """Return the number of served rows.

        :returns: Row count.
        """
        return self.rows

    def _open(self) -> lance.LanceDataset:
        """Return this process's dataset handle, opening it on first use.

        :returns: Open Lance dataset.
        """
        if self._dataset is None:
            self._dataset = lance.dataset(self.uri, storage_options=self.storage_options)
        return self._dataset

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Decode one row onto the render contract.

        :param index: Row index.
        :returns: ``audio`` and, when configured, ``mel`` for one clip.
        """
        blob = self._open().take_blobs(self.audio_column, indices=[index])[0]
        try:
            data = blob.read()
        finally:
            blob.close()
        audio = decode_clip(
            data,
            sample_rate=self.sample_rate,
            channels=self.channels,
            num_samples=self.num_samples,
            amplitude_scale=self.amplitude_scale,
        )
        row = {"audio": torch.from_numpy(audio)}
        if self.with_mel:
            row["mel"] = torch.from_numpy(
                make_spectrogram(audio, self.sample_rate).astype(np.float32)
            )
        return row


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
        audio_column: str = AUDIO_FIELD,
        amplitude_scale: float = 1.0,
        batch_size: int = 32,
        num_workers: int = 0,
        row_limit: int | None = None,
        conditioning: Conditioning = "mel",
        sketch: SketchControls = None,
        embedding_device: str | None = None,
        embedding_checkpoint: str | None = None,
        embedding_encoder: Encoder | None = None,
        use_saved_mean_and_variance: bool = False,
        mel_stats_uri: str | None = None,
        stats_cache_dir: str | None = None,
    ) -> None:
        """Configure the corpus, the render contract it maps onto, and conditioning.

        :param dataset_uri: Corpus Lance dataset; local path or ``r2://`` URI.
        :param sample_rate: Target sample rate in Hz, matching the checkpoint's render config.
        :param channels: Target channel count.
        :param signal_duration_seconds: Target clip duration.
        :param audio_column: Blob column holding source container bytes.
        :param amplitude_scale: Gain applied to decoded audio before the mel front-end.
        :param batch_size: Rows per predict batch.
        :param num_workers: Dataloader workers decoding rows.
        :param row_limit: Serve only the first N rows; ``None`` serves the whole corpus.
        :param conditioning: The checkpoint's conditioning mode or embedding spec.
        :param sketch: Sketch-control spec when the checkpoint takes control tokens.
        :param embedding_device: Torch device for live encoders; ``None`` auto-selects.
        :param embedding_checkpoint: Encoder checkpoint override.
        :param embedding_encoder: Pre-loaded encoder, bypassing registry loading.
        :param use_saved_mean_and_variance: Whether to standardize mel with saved statistics.
        :param mel_stats_uri: ``.npz`` of the training run's mel statistics; local or ``r2://``.
        :param stats_cache_dir: Directory a fetched statistics object is cached in;
            ``None`` uses the working directory.
        """
        super().__init__()
        _validate_conditioning_config(
            conditioning,
            use_saved_mean_and_variance=use_saved_mean_and_variance,
            mel_stats_uri=mel_stats_uri,
            row_limit=row_limit,
        )
        self.dataset_uri = dataset_uri
        self.sample_rate = sample_rate
        self.channels = channels
        self.num_samples = int(sample_rate * signal_duration_seconds)
        self.audio_column = audio_column
        self.amplitude_scale = amplitude_scale
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.row_limit = row_limit
        self.conditioning = conditioning
        self.embedding = resolve_embedding_conditioning(conditioning)
        self.sketch = resolve_sketch_controls(sketch)
        self.mel_stats_uri = mel_stats_uri if use_saved_mean_and_variance else None
        self.stats_cache_dir = Path(stats_cache_dir or Path.cwd() / _MEL_STATS_CACHE_DIR)
        self._embedding_device = embedding_device
        self._embedding_checkpoint = embedding_checkpoint
        self._embedding_encoder = embedding_encoder
        # Reject an unservable column at construction rather than mid-sweep.
        for spec in (self.embedding, self.sketch):
            if spec is not None:
                registry_spec(spec.column)
        self._live: dict[str, LiveEmbedding] = {}
        self._statistics: tuple[torch.Tensor, torch.Tensor] | None = None
        self._predict_dataset: _BlobAudioDataset | None = None

    @property
    def _with_mel(self) -> bool:
        """Return whether the configured conditioning consumes the mel front-end.

        :returns: True when the model reads the ``mel`` batch entry.
        """
        return self.conditioning == "mel"

    def cached_stats_path(self) -> Path:
        """Return the local path the configured statistics object resolves to.

        Every training split names its statistics ``stats.npz``, so the cache
        slot is keyed by a digest of the full URI: two checkpoints trained on
        different corpora must not share one slot.

        :returns: Local ``.npz`` path; unique per ``r2://`` URI.
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
            # Rename is atomic within the directory, so a concurrent eval either
            # sees no file or the complete one — never a partial download.
            staged = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
            r2_io.download_to_path(cast(str, self.mel_stats_uri), staged)
            staged.replace(destination)
        return destination

    def setup(self, stage: str | None = None) -> None:
        """Open the corpus and load the conditioning encoders for a predict run.

        :param stage: Lightning stage hint; only prediction is served.
        :raises ValueError: ``stage`` is not a prediction stage.
        :raises KeyError: The corpus has no column named ``audio_column``.
        """
        if stage not in _PREDICT_STAGES:
            raise ValueError(f"{type(self).__name__} serves prediction only, got stage {stage!r}")
        uri, storage_options = (
            r2_io.lance_target(self.dataset_uri)
            if r2_io.is_r2_uri(self.dataset_uri)
            else (self.dataset_uri, None)
        )
        dataset = lance.dataset(uri, storage_options=storage_options)
        if self.audio_column not in dataset.schema.names:
            raise KeyError(
                f"corpus {self.dataset_uri} has no {self.audio_column!r} column; "
                f"columns: {', '.join(dataset.schema.names)}"
            )
        rows = dataset.count_rows()
        self._predict_dataset = _BlobAudioDataset(
            uri,
            storage_options=storage_options,
            audio_column=self.audio_column,
            sample_rate=self.sample_rate,
            channels=self.channels,
            num_samples=self.num_samples,
            amplitude_scale=self.amplitude_scale,
            with_mel=self._with_mel,
            rows=rows if self.row_limit is None else min(rows, self.row_limit),
        )
        if self.mel_stats_uri is not None and self._statistics is None:
            mean, std = load_mel_statistics(self._local_stats_file())
            self._statistics = (
                torch.as_tensor(mean, dtype=torch.float32),
                torch.as_tensor(std, dtype=torch.float32),
            )
        self._live = self._load_encoders()

    def _load_encoders(self) -> dict[str, LiveEmbedding]:
        """Load one live encoder per conditioning stream the checkpoint consumes.

        :returns: Live encoders keyed by stored column name.
        """
        columns = [spec.column for spec in (self.embedding, self.sketch) if spec is not None]
        if self._embedding_encoder is not None:
            return {
                column: LiveEmbedding(
                    registry_spec(column), self._embedding_encoder, sample_rate=self.sample_rate
                )
                for column in columns
            }
        return {
            column: LiveEmbedding.from_registry(
                column,
                sample_rate=self.sample_rate,
                lance_uri=self.dataset_uri,
                device=self._embedding_device,
                checkpoint=self._embedding_checkpoint,
            )
            for column in columns
        }

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
        """Add live conditioning and mel normalization to a decoded batch.

        Runs in the main process so one encoder — not one per dataloader worker —
        holds the frozen weights.

        :param batch: Decoded ``audio``, and ``mel`` when configured, for one batch.
        :param dataloader_idx: Unused; the datamodule serves one loader.
        :returns: The model batch a predict step consumes.
        """
        del dataloader_idx
        model_batch = dict(batch)
        if self._statistics is not None and "mel" in model_batch:
            mean, std = self._statistics
            model_batch["mel"] = (model_batch["mel"] - mean) / std
        if not self._live:
            return model_batch
        audio = model_batch["audio"].numpy()
        if self.embedding is not None:
            model_batch["conditioning"] = self._encode_conditioning(audio, self.embedding)
        if self.sketch is not None:
            model_batch[SKETCH_CTRL_FIELD] = self._encode_sketch(audio, self.sketch)
        return model_batch

    def _encode_conditioning(
        self, audio: np.ndarray, embedding: EmbeddingConditioningSpec
    ) -> torch.Tensor:
        """Encode content conditioning and pin it to the checkpoint's per-row shape.

        :param audio: ``(B, C, T)`` audio batch.
        :param embedding: Configured column and per-row shape.
        :returns: ``(B, *input_shape)`` conditioning tensor.
        :raises ValueError: The live shape contradicts the checkpoint's spec.
        """
        tensor = self._live[embedding.column](audio)[embedding.column]
        if tuple(tensor.shape[1:]) != embedding.input_shape:
            raise ValueError(
                f"live {embedding.column!r} rows are {tuple(tensor.shape[1:])}, but the "
                f"checkpoint's conditioning input_shape is {embedding.input_shape}"
            )
        return tensor.to(torch.float32)

    def _encode_sketch(self, audio: np.ndarray, sketch: SketchControlSpec) -> torch.Tensor:
        """Extract sketch controls and reassemble the model's control stack.

        :param audio: ``(B, C, T)`` audio batch.
        :param sketch: Resolved sketch-control spec naming the column and zero-bin threshold.
        :returns: ``(B, NUM_SKETCH_CONTROLS, frames)`` controls with pitch zero-binned.
        :raises ValueError: The live frame axis contradicts the checkpoint's ``num_frames``.
        """
        children = self._live[sketch.column](audio)
        prefix = f"{sketch.column}."
        controls = torch.from_numpy(
            stack_sketch_children(
                children[f"{prefix}{SKETCH_LOUDNESS_CHILD}"].numpy(),
                children[f"{prefix}{SKETCH_CENTROID_CHILD}"].numpy(),
                children[f"{prefix}{SKETCH_PITCH_CHILD}"].numpy(),
            )
        ).to(torch.float32)
        if controls.shape[-1] != sketch.num_frames:
            raise ValueError(
                f"live sketch controls span {controls.shape[-1]} frames, but the "
                f"checkpoint's num_frames is {sketch.num_frames}; the corpus duration "
                "or sample rate does not match the render contract it trained on"
            )
        pitch = controls[:, SKETCH_PITCH_SLICE]
        controls[:, SKETCH_PITCH_SLICE] = pitch.where(pitch >= sketch.pitch_zero_threshold, 0.0)
        return controls
