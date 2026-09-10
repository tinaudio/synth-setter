"""Tests for online tonal-interval-vector extraction."""

import pytest
import torch

from synth_setter.conditioning import TIVBackend
from synth_setter.features.tiv import chroma_to_tiv, extract_tiv_batch


def test_extract_tiv_cpu_a4_tone_matches_reference_hpcp_coordinates() -> None:
    """Reference HPCP for A4 preserves the C-rooted DFT phase and audio weights."""
    pytest.importorskip("essentia")
    samples = torch.arange(4096, dtype=torch.float32)
    audio = torch.sin(2.0 * torch.pi * 440.0 * samples / 44_100)[None, None]

    tiv = extract_tiv_batch(audio, sample_rate=44_100, backend="essentia")

    expected = torch.tensor([0.0, 3.0, -8.0, 0.0, 0.0, -11.5, 15.0, 0.0, 0.0, 14.5, -7.5, 0.0])
    torch.testing.assert_close(tiv[0, :, 2], expected)
    assert tiv.shape == (1, 12, 9)


def test_extract_tiv_cpu_silence_returns_finite_zeros() -> None:
    """Silent CPU frames have no peaks and therefore no tonal coordinates."""
    pytest.importorskip("essentia")

    tiv = extract_tiv_batch(torch.zeros(2, 2, 4096), 44_100, backend="essentia")

    assert torch.count_nonzero(tiv) == 0
    assert torch.isfinite(tiv).all()


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["torch", "essentia"])
def test_extract_tiv_cuda_batch_returns_controls_on_input_device(backend: TIVBackend) -> None:
    """Both frontends return CUDA controls usable by the GPU sketch tokenizer.

    :param backend: Device-local STFT or CPU HPCP implementation.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if backend == "essentia":
        pytest.importorskip("essentia")
    samples = torch.arange(4096, dtype=torch.float32)
    audio = torch.sin(2.0 * torch.pi * 440.0 * samples / 44_100)[None, None]

    cpu_controls = extract_tiv_batch(audio, 44_100, backend=backend)
    cuda_controls = extract_tiv_batch(audio.cuda(), 44_100, backend=backend)

    assert cuda_controls.is_cuda
    torch.testing.assert_close(cuda_controls.cpu(), cpu_controls, atol=2e-5, rtol=2e-5)


def test_extract_tiv_cpu_wrong_sample_rate_raises() -> None:
    """Reference-style HPCP rejects audio outside its 44.1 kHz contract."""
    with pytest.raises(ValueError, match="44100 Hz"):
        extract_tiv_batch(torch.zeros(1, 1, 4096), 16_000, backend="essentia")


def test_chroma_to_tiv_c_pitch_matches_tivlib_audio_weights() -> None:
    """A unit C profile has real coefficients equal to TIVlib's audio weights."""
    chroma = torch.zeros(1, 12, 1)
    chroma[:, 0] = 1.0

    tiv = chroma_to_tiv(chroma)

    expected = torch.tensor(
        [3.0, 0.0, 8.0, 0.0, 11.5, 0.0, 15.0, 0.0, 14.5, 0.0, 7.5, 0.0]
    ).reshape(1, 12, 1)
    torch.testing.assert_close(tiv, expected)


def test_chroma_to_tiv_amplitude_scaled_profile_is_unchanged() -> None:
    """L1 chroma normalization makes TIV coordinates amplitude-invariant."""
    chroma = torch.tensor([[[1.0], [2.0], [1.0]] + [[0.0]] * 9])

    torch.testing.assert_close(chroma_to_tiv(chroma), chroma_to_tiv(chroma * 17.0))


def test_chroma_to_tiv_subnormal_energy_remains_l1_normalized() -> None:
    """Any nonzero finite chroma uses its L1 norm rather than an epsilon floor."""
    chroma = torch.zeros(1, 12, 1)
    chroma[:, 0] = 1e-40

    tiv = chroma_to_tiv(chroma)

    assert tiv[0, 0, 0] == 3.0


def test_chroma_to_tiv_silence_returns_finite_zeros() -> None:
    """A zero-energy chroma frame produces no NaN or tonal coordinates."""
    tiv = chroma_to_tiv(torch.zeros(2, 12, 3))

    assert torch.isfinite(tiv).all()
    assert torch.count_nonzero(tiv) == 0


def test_extract_tiv_batch_audio_returns_temporal_float32_controls() -> None:
    """The audio front end emits one 12-channel TIV sequence per waveform."""
    samples = torch.arange(4096, dtype=torch.float32)
    audio = torch.sin(2.0 * torch.pi * 440.0 * samples / 44_100.0)
    stereo = audio[None, None].expand(2, 2, -1).contiguous()

    tiv = extract_tiv_batch(stereo, sample_rate=44_100)

    assert tiv.shape[:2] == (2, 12)
    assert tiv.shape[2] > 0
    assert tiv.dtype == torch.float32
    assert torch.isfinite(tiv).all()
