"""Behavior tests for the frozen differentiable SAME encoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from synth_setter.models.components.same_encoder import SameAudioEncoder
from synth_setter.same import SAME_SAMPLE_RATE
from tests.helpers.same_reference import (
    SAME_HF_CHECKPOINTS,
    TINY_SAME_DOWNSAMPLING_RATIO,
    TINY_SAME_LATENT_DIM,
    same_reference_audio,
)

_ROWS = 2
_LENGTH = 4096


def _encoder(checkpoint: Path, sample_rate: int = SAME_SAMPLE_RATE) -> SameAudioEncoder:
    """Load the tiny checkpoint through the production entry point.

    :param checkpoint: Local SAME checkpoint directory.
    :param sample_rate: Rate of the waveforms the encoder will be handed.
    :returns: Frozen differentiable encoder.
    """
    return SameAudioEncoder.from_pretrained(sample_rate=sample_rate, checkpoint=str(checkpoint))


def test_encoder_returns_one_latent_sequence_per_waveform(tiny_same_checkpoint: Path) -> None:
    """The distance reduces over latent axes, so rows must stay separable.

    :param tiny_same_checkpoint: Loadable SAME checkpoint.
    """
    torch.manual_seed(0)
    audio = torch.randn(_ROWS, _LENGTH).clamp(-1.0, 1.0)

    latents = _encoder(tiny_same_checkpoint)(audio)

    assert latents.shape == (
        _ROWS,
        TINY_SAME_LATENT_DIM,
        2 * _LENGTH // (2 * TINY_SAME_DOWNSAMPLING_RATIO),
    )
    assert torch.isfinite(latents).all()
    assert not torch.equal(latents[0], latents[1])


def test_gradient_reaches_the_waveform(tiny_same_checkpoint: Path) -> None:
    """SAME's patched pretransform encodes under ``no_grad`` unless grad is enabled.

    :param tiny_same_checkpoint: Loadable SAME checkpoint.
    """
    torch.manual_seed(0)
    audio = torch.randn(_ROWS, _LENGTH).clamp(-1.0, 1.0).requires_grad_(True)

    _encoder(tiny_same_checkpoint)(audio).pow(2).mean().backward()

    assert audio.grad is not None
    assert torch.isfinite(audio.grad).all()
    assert torch.count_nonzero(audio.grad) > 0


def test_encoding_is_deterministic_across_calls(tiny_same_checkpoint: Path) -> None:
    """A moving metric space would make the audio term chase its own noise.

    :param tiny_same_checkpoint: Loadable SAME checkpoint.
    """
    torch.manual_seed(0)
    audio = torch.randn(_ROWS, _LENGTH).clamp(-1.0, 1.0)
    encoder = _encoder(tiny_same_checkpoint)

    assert torch.equal(encoder(audio), encoder(audio))


def test_half_rate_input_yields_the_same_frame_count(tiny_same_checkpoint: Path) -> None:
    """Latent width tracks clip duration, not the caller's sample rate.

    :param tiny_same_checkpoint: Loadable SAME checkpoint.
    """
    torch.manual_seed(0)
    native = torch.randn(_ROWS, _LENGTH).clamp(-1.0, 1.0)
    half_rate = native[:, ::2].contiguous()

    native_frames = _encoder(tiny_same_checkpoint)(native).shape[-1]
    half_rate_frames = _encoder(tiny_same_checkpoint, SAME_SAMPLE_RATE // 2)(half_rate).shape[-1]

    assert half_rate_frames == native_frames


def test_encoder_stays_in_eval_mode_when_the_module_trains(tiny_same_checkpoint: Path) -> None:
    """Training mode would let SAME's normalization statistics drift.

    :param tiny_same_checkpoint: Loadable SAME checkpoint.
    """
    encoder = _encoder(tiny_same_checkpoint)

    encoder.train(True)

    assert not encoder.autoencoder.training


def test_trainable_autoencoder_is_rejected(tiny_same_checkpoint: Path) -> None:
    """A trainable backbone could move the space instead of matching the target.

    :param tiny_same_checkpoint: Loadable SAME checkpoint.
    """
    trainable = _encoder(tiny_same_checkpoint).autoencoder.requires_grad_(True)

    with pytest.raises(ValueError, match="frozen"):
        SameAudioEncoder(sample_rate=SAME_SAMPLE_RATE, autoencoder=trainable)


def test_mono_waveform_encodes_as_duplicated_stereo(tiny_same_checkpoint: Path) -> None:
    """SAME is a stereo model, and the pipeline column duplicates mono the same way.

    :param tiny_same_checkpoint: Loadable SAME checkpoint.
    """
    from synth_setter.pipeline.data.add_embeddings import load_same_audio_encoder

    torch.manual_seed(0)
    mono = torch.randn(_ROWS, _LENGTH).clamp(-1.0, 1.0)
    stereo = np.repeat(mono.numpy()[:, None, :], 2, axis=1)

    online = _encoder(tiny_same_checkpoint)(mono).detach().numpy()
    offline = load_same_audio_encoder(str(tiny_same_checkpoint), "cpu")(stereo)

    np.testing.assert_allclose(online, offline, rtol=1e-5, atol=1e-6)


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.same_e2e
def test_real_same_s_encoder_matches_the_pipeline_column_geometry() -> None:
    """Online scoring is only comparable to stored conditioning if both agree exactly."""
    from huggingface_hub import snapshot_download

    from synth_setter.pipeline.data.add_embeddings import load_same_audio_encoder
    from synth_setter.same import same_s_num_latent_frames

    repo_id, revision = SAME_HF_CHECKPOINTS["same_s"]
    checkpoint = snapshot_download(repo_id, revision=revision)
    torch.manual_seed(0)
    mono = torch.from_numpy(same_reference_audio(SAME_SAMPLE_RATE)[:, 0, :])
    stereo = np.repeat(mono.numpy()[:, None, :], 2, axis=1)

    online = _encoder(Path(checkpoint))(mono).detach().numpy()
    offline = load_same_audio_encoder(checkpoint, "cpu")(stereo)

    assert online.shape[-1] == same_s_num_latent_frames(mono.shape[-1], SAME_SAMPLE_RATE)
    np.testing.assert_allclose(online, offline, rtol=1e-5, atol=1e-6)
