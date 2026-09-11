"""Materialized, snapshot-pinned input audio for effect dataset generation."""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import lance
import numpy as np

from synth_setter.data.vst.seeding import seed_for_input_audio
from synth_setter.data.vst.shapes import AUDIO_FIELD
from synth_setter.model_cache import synth_setter_cache_dir
from synth_setter.pipeline.data.lance_materialize import materialize_lance_subset
from synth_setter.pipeline.data.lance_shard import read_shard_metadata
from synth_setter.pipeline.schemas.spec import InputAudioSource

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


class InputAudioPool:
    """Expose mono rows from one locally materialized Lance snapshot.

    .. attribute :: materialized_path

        Local projected Lance dataset used for row reads.

    .. attribute :: row_count

        Number of selectable source rows.
    """

    materialized_path: Path
    row_count: int

    def __init__(
        self,
        source: InputAudioSource,
        *,
        sample_rate: int,
        frames: int,
    ) -> None:
        """Materialize and validate a source pool for one renderer geometry.

        :param source: Pinned dataset split identity.
        :param sample_rate: Required source sample rate in Hz.
        :param frames: Required frame count per mono source row.
        :raises ValueError: The source is empty or its metadata/geometry is incompatible.
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
        self._dataset: LanceDataset = lance.dataset(str(self.materialized_path))
        self.row_count = self._dataset.count_rows()
        if self.row_count < 1:
            raise ValueError("input audio pool must be non-empty")

        metadata = read_shard_metadata(self._dataset.schema)
        if metadata.sample_rate != sample_rate:
            raise ValueError(
                f"input audio sample rate {metadata.sample_rate} != renderer sample rate {sample_rate}"
            )
        field = self._dataset.schema.field(AUDIO_FIELD)
        shape = tuple(getattr(field.type, "shape", ()))
        expected_shape = (1, frames)
        if shape != expected_shape:
            raise ValueError(f"input audio row shape {shape} != expected {expected_shape}")

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
        :returns: Float32 mono waveform shaped ``(frames,)``.
        :raises IndexError: The row index is outside the pool.
        :raises ValueError: The selected row contains NaN or infinity.
        """
        if not 0 <= row_index < self.row_count:
            raise IndexError(f"input audio row index {row_index} outside [0, {self.row_count})")
        scalar = self._dataset.take([row_index], columns=[AUDIO_FIELD])[AUDIO_FIELD][0]
        row = np.asarray(scalar.as_py(), dtype=np.float32).reshape(-1)
        if not np.isfinite(row).all():
            raise ValueError(f"input audio row {row_index} must contain only finite samples")
        return np.ascontiguousarray(row)
