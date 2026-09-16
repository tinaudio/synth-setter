"""Materialized, snapshot-pinned input audio for effect dataset generation.

Foreign snapshots rarely match the renderer grid, so every row is pinned to it on
read: resample to the renderer rate, mean downmix to mono, then first-N truncate
with zero-pad tail. The operations mirror ``third_party_datamodule.decode`` (which
cannot be reused directly — it consumes encoded containers, while pool rows are
already-decoded tensors), and the versioned ``INPUT_AUDIO_ADAPTATION_POLICY``
token enters the render-contract digest while ``snapshot_txid`` transitively pins
the source geometry behind it.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lance
import librosa
import numpy as np

from synth_setter.data.vst.seeding import seed_for_input_audio
from synth_setter.data.vst.shapes import AUDIO_FIELD
from synth_setter.model_cache import synth_setter_cache_dir
from synth_setter.pipeline.data.lance_materialize import materialize_lance_subset
from synth_setter.pipeline.data.lance_shard import read_shard_metadata
from synth_setter.pipeline.schemas.spec import InputAudioSource
from synth_setter.renderer_backend import INPUT_AUDIO_ADAPTATION_POLICY

if TYPE_CHECKING:
    from lance import LanceDataset

_PROJECTION = (AUDIO_FIELD,)


def _split_uri(source: InputAudioSource) -> str:
    return f"{source.dataset_uri.rstrip('/')}/{source.split}.lance"


def _cache_key(source: InputAudioSource) -> str:
    identity = {
        "dataset_uri": source.dataset_uri,
        "projection": _PROJECTION,
        "snapshot_txid": source.snapshot_txid,
        "split": source.split,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _pin_to_grid(
    row: np.ndarray, *, source_sample_rate: int, sample_rate: int, frames: int
) -> np.ndarray:
    """Pin one source row to the renderer grid.

    :param row: Source audio shaped ``(channels, source_frames)``.
    :param source_sample_rate: Source snapshot rate in Hz.
    :param sample_rate: Renderer rate in Hz.
    :param frames: Renderer frame count.
    :returns: Contiguous finite float32 mono waveform shaped ``(frames,)``.
    """
    if source_sample_rate != sample_rate:
        row = librosa.resample(row, orig_sr=source_sample_rate, target_sr=sample_rate)
    mono = row.mean(axis=0) if row.shape[0] > 1 else row[0]
    if mono.shape[0] < frames:
        mono = np.pad(mono, (0, frames - mono.shape[0]))
    return np.ascontiguousarray(mono[:frames], dtype=np.float32)


class InputAudioPool:
    """Expose mono rows from one locally materialized Lance snapshot.

    .. attribute :: materialized_path

        Local projected Lance dataset used for row reads.

    .. attribute :: row_count

        Number of selectable source rows.

    .. attribute :: source_sample_rate

        Snapshot shard rate in Hz, before adaptation.

    .. attribute :: source_channels

        Snapshot audio channel count, before the mono downmix.

    .. attribute :: source_frames

        Snapshot frames per row, before length pinning.

    .. attribute :: adapted

        Whether any row needs resampling, downmix, or length pinning.
    """

    materialized_path: Path
    row_count: int
    source_sample_rate: int
    source_channels: int
    source_frames: int
    adapted: bool

    def __init__(
        self,
        source: InputAudioSource,
        *,
        sample_rate: int,
        frames: int,
    ) -> None:
        """Materialize and validate a source pool for one renderer geometry.

        :param source: Pinned dataset split identity.
        :param sample_rate: Renderer sample rate in Hz; foreign rates resample to it.
        :param frames: Renderer frame count; longer rows truncate, shorter ones pad.
        :raises ValueError: The source is empty or its audio column is not 2-D.
        """
        cache_dir = synth_setter_cache_dir() / "input-audio" / _cache_key(source)
        self.materialized_path = cache_dir / f"{source.split}.lance"
        lock_path = cache_dir / ".materialize.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                materialize_lance_subset(
                    _split_uri(source),
                    self.materialized_path,
                    txid=source.snapshot_txid,
                    columns=_PROJECTION,
                )
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

        self._sampling_seed = source.sampling_seed
        self._sample_rate = sample_rate
        self._frames = frames
        self._dataset: LanceDataset = lance.dataset(str(self.materialized_path))
        self.row_count = self._dataset.count_rows()
        if self.row_count < 1:
            raise ValueError("input audio pool must be non-empty")

        metadata = read_shard_metadata(self._dataset.schema)
        self.source_sample_rate = metadata.sample_rate
        shape: tuple[int, ...] = tuple(getattr(self._dataset.schema.field(AUDIO_FIELD).type, "shape", ()))
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError(f"input audio row shape {shape} is not (channels, frames)")
        self.source_channels, self.source_frames = shape
        self.adapted = (
            self.source_sample_rate != sample_rate
            or (self.source_channels, self.source_frames) != (1, frames)
        )

    def adaptation_provenance(self) -> dict[str, Any]:
        """Describe the grid pinning applied to every row.

        :returns: Policy token with source and target geometry; the digest covers the
            policy while ``snapshot_txid`` pins the source side behind it.
        """
        return {
            "policy": INPUT_AUDIO_ADAPTATION_POLICY,
            "source_sample_rate": self.source_sample_rate,
            "source_channels": self.source_channels,
            "source_frames": self.source_frames,
            "target_sample_rate": self._sample_rate,
            "target_frames": self._frames,
            "adapted": self.adapted,
        }

    def row_index(self, base_seed: int, sample_idx: int, attempt: int) -> int:
        """Select one row deterministically for an output attempt.

        :param base_seed: Dataset or split master seed.
        :param sample_idx: Stable logical output row index.
        :param attempt: Loudness-gate retry attempt for the row.
        :returns: A valid row index in this pool.
        """
        return (
            seed_for_input_audio(base_seed, self._sampling_seed, sample_idx, attempt)
            % self.row_count
        )

    def take(self, row_index: int) -> np.ndarray:
        """Read one source row as a finite contiguous mono waveform.

        :param row_index: Zero-based row index in the materialized snapshot.
        :returns: Float32 mono waveform shaped ``(frames,)`` pinned to the grid.
        :raises IndexError: The row index is outside the pool.
        :raises ValueError: The selected row contains NaN or infinity.
        """
        if not 0 <= row_index < self.row_count:
            raise IndexError(f"input audio row index {row_index} outside [0, {self.row_count})")
        scalar = self._dataset.take([row_index], columns=[AUDIO_FIELD])[AUDIO_FIELD][0]
        row = np.asarray(scalar.as_py(), dtype=np.float32).reshape(
            self.source_channels, self.source_frames
        )
        if not np.isfinite(row).all():
            raise ValueError(f"input audio row {row_index} must contain only finite samples")
        if not self.adapted:
            return np.ascontiguousarray(row[0])
        pinned = _pin_to_grid(
            row,
            source_sample_rate=self.source_sample_rate,
            sample_rate=self._sample_rate,
            frames=self._frames,
        )
        if not np.isfinite(pinned).all():
            raise ValueError(f"input audio row {row_index} must contain only finite samples")
        return pinned
