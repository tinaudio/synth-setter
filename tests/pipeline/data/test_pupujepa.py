"""Behavior tests for PupuJEPA Tiny embedding augmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import synth_setter.pipeline.data.pupujepa as pupujepa_module
from synth_setter.data.vst.shapes import AUDIO_FIELD, PUPUJEPA_TINY_FIELD
from synth_setter.models.components.pupujepa_encoder import PupuJepaAudioEncoder
from synth_setter.pipeline.data.add_embeddings import EMBEDDING_REGISTRY, IndexSpec
from synth_setter.pipeline.data.pupujepa import (
    PUPUJEPA_ENCODE_MAX_BATCH,
    encode_pupujepa_column,
    pupujepa_encoder_input,
)
from synth_setter.pupujepa import (
    DEFAULT_PUPUJEPA_TINY_CHECKPOINT,
    PUPUJEPA_EMBEDDING_DIM,
    PUPUJEPA_CHECKPOINT_REVISION,
    PUPUJEPA_SAMPLE_RATE,
    PUPUJEPA_TINY_ARGS_FILE,
    PUPUJEPA_TINY_WEIGHTS_FILE,
    pupujepa_num_time_patches,
    resolve_pupujepa_checkpoint,
)


def test_default_checkpoint_with_tampered_artifact_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pinned revision cannot label modified cached bytes as canonical.

    :param tmp_path: Scratch Hugging Face snapshot.
    :param monkeypatch: Fixture replacing the download boundary.
    """
    snapshot = tmp_path / PUPUJEPA_CHECKPOINT_REVISION
    args_path = snapshot / PUPUJEPA_TINY_ARGS_FILE
    weights_path = snapshot / PUPUJEPA_TINY_WEIGHTS_FILE
    args_path.parent.mkdir(parents=True)
    weights_path.parent.mkdir(parents=True)
    args_path.write_text("{}")
    weights_path.write_bytes(b"tampered")
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **_kwargs: str(snapshot))

    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        resolve_pupujepa_checkpoint()


def test_pupujepa_registry_spec_pins_solo_mean_pooled_sequence() -> None:
    """The registry exposes the immutable PupuJEPA Tiny sequence policy."""
    spec = EMBEDDING_REGISTRY["pupujepa_tiny"]

    assert spec.column == PUPUJEPA_TINY_FIELD
    assert spec.default_checkpoint == DEFAULT_PUPUJEPA_TINY_CHECKPOINT
    assert spec.co_resident is False
    assert spec.index == IndexSpec(
        pool="mean",
        vector_column=f"{PUPUJEPA_TINY_FIELD}_vec",
        vector_dim=PUPUJEPA_EMBEDDING_DIM,
    )


@pytest.mark.parametrize(
    ("num_samples", "sample_rate", "expected_patches"),
    [(960, 24_000, 1), (24_000, 24_000, 25), (96_000, 24_000, 100), (176_400, 44_100, 100)],
)
def test_pupujepa_num_time_patches_supported_lengths_matches_contract(
    num_samples: int, sample_rate: int, expected_patches: int
) -> None:
    """Source lengths map to the 25 Hz PupuJEPA patch grid.

    :param num_samples: Source clip length.
    :param sample_rate: Source sample rate in Hz.
    :param expected_patches: Expected temporal patch count.
    """
    assert pupujepa_num_time_patches(num_samples, sample_rate) == expected_patches


def test_pupujepa_encoder_input_duplicated_stereo_matches_mono() -> None:
    """A duplicated stereo render and its mono source downmix identically."""
    mono = np.linspace(-1.0, 1.0, 960, dtype=np.float32)[None, None, :]
    stereo = np.repeat(mono, 2, axis=1)

    np.testing.assert_array_equal(
        pupujepa_encoder_input(stereo, PUPUJEPA_SAMPLE_RATE),
        pupujepa_encoder_input(mono, PUPUJEPA_SAMPLE_RATE),
    )


@pytest.mark.parametrize(
    ("audio", "sample_rate", "message"),
    [
        (np.zeros((2, 960), dtype=np.float32), 24_000, r"expected a \(B, C, T\) batch"),
        (np.zeros((0, 1, 960), dtype=np.float32), 24_000, "non-empty batch"),
        (np.zeros((1, 3, 960), dtype=np.float32), 24_000, "1 or 2 channels"),
        (np.zeros((1, 1, 960), dtype=np.float32), 0, "positive sample_rate"),
        (np.full((1, 1, 960), np.nan, dtype=np.float32), 24_000, "non-finite"),
        (np.zeros((1, 1, 959), dtype=np.float32), 24_000, "one complete time patch"),
    ],
)
def test_pupujepa_encoder_input_invalid_audio_raises(
    audio: np.ndarray, sample_rate: int, message: str
) -> None:
    """Malformed source audio fails before teacher inference.

    :param audio: Candidate audio batch.
    :param sample_rate: Candidate source sample rate.
    :param message: Expected failure detail.
    """
    with pytest.raises(ValueError, match=message):
        pupujepa_encoder_input(audio, sample_rate)


def test_encode_pupujepa_column_valid_sequence_preserves_orientation() -> None:
    """Channel-major teacher sequences persist without transposition."""
    audio = np.zeros((2, 1, 96_000), dtype=np.float32)
    output = np.ones((2, PUPUJEPA_EMBEDDING_DIM, 100), dtype=np.float32)

    encoded = encode_pupujepa_column(
        {AUDIO_FIELD: audio},
        PUPUJEPA_SAMPLE_RATE,
        lambda _audio, _sample_rate: output,
    )

    assert encoded.to_numpy_ndarray().shape == (2, PUPUJEPA_EMBEDDING_DIM, 100)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (np.ones((2, 100, PUPUJEPA_EMBEDDING_DIM), dtype=np.float32), "produced shape"),
        (np.full((2, PUPUJEPA_EMBEDDING_DIM, 100), np.inf, dtype=np.float32), "non-finite"),
    ],
)
def test_encode_pupujepa_column_invalid_output_raises(
    output: np.ndarray, message: str
) -> None:
    """Wrong orientation and non-finite teacher output cannot reach Lance.

    :param output: Candidate encoder output.
    :param message: Expected failure detail.
    """
    audio = np.zeros((2, 1, 96_000), dtype=np.float32)

    with pytest.raises(ValueError, match=message):
        encode_pupujepa_column(
            {AUDIO_FIELD: audio},
            PUPUJEPA_SAMPLE_RATE,
            lambda _audio, _sample_rate: output,
        )


class _BatchRecordingEncoder(torch.nn.Module):
    """Return row-identifying sequences while recording bounded batches."""

    def __init__(self) -> None:
        """Initialize an empty batch-size ledger."""
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, audio: torch.Tensor, sample_rate: int | None = None) -> torch.Tensor:
        """Expand each row's first sample over the production embedding shape.

        :param audio: Mono waveform batch.
        :param sample_rate: Source sample rate.
        :returns: Row-identifying teacher-shaped sequence.
        """
        assert sample_rate is not None
        self.batch_sizes.append(len(audio))
        frames = pupujepa_num_time_patches(audio.shape[-1], sample_rate)
        return audio[:, :1, None].expand(-1, PUPUJEPA_EMBEDDING_DIM, frames)


def test_load_pupujepa_audio_encoder_bounds_chunks_and_preserves_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline inference caps resident rows without reordering them.

    :param monkeypatch: Fixture replacing only the pretrained materialization boundary.
    """
    model = _BatchRecordingEncoder()
    monkeypatch.setattr(
        PupuJepaAudioEncoder,
        "from_pretrained",
        lambda **_kwargs: model,
    )
    rows = PUPUJEPA_ENCODE_MAX_BATCH + 1
    audio = np.stack(
        [np.full((1, 960), row / rows, dtype=np.float32) for row in range(rows)]
    )

    encode = pupujepa_module.load_pupujepa_audio_encoder(device="cpu")
    embeddings = encode(audio, PUPUJEPA_SAMPLE_RATE)

    assert model.batch_sizes == [PUPUJEPA_ENCODE_MAX_BATCH, 1]
    assert embeddings.shape == (rows, PUPUJEPA_EMBEDDING_DIM, 1)
    np.testing.assert_allclose(embeddings[:, 0, 0], np.arange(rows) / rows)
