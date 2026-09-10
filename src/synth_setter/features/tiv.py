"""TIVlib audio-weighted vectors from device-local STFT chroma or CPU HPCP.

The PyTorch frontend maps STFT bins to 12-TET pitch classes on the batch device; Essentia follows
TIVlib's example HPCP settings but retains temporal frames.
"""

from importlib import import_module

import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped

from synth_setter.conditioning import TIVBackend

_TIV_AUDIO_WEIGHTS = (3.0, 8.0, 11.5, 15.0, 14.5, 7.5)
_CHROMA_N_FFT = 2048
_CHROMA_HOP_LENGTH = 512
_MIDI_A4 = 69.0
_A4_HZ = 440.0


@jaxtyped(typechecker=beartype)
def chroma_to_tiv(
    chroma: Float[torch.Tensor, "batch 12 frames"],
) -> Float[torch.Tensor, "batch 12 frames"]:
    """Convert non-negative chroma to weighted real-imaginary TIV channels.

    :param chroma: Pitch-class energy with C at row zero.
    :returns: Interleaved real and imaginary channels for DFT bins 1 through 6.
    """
    energy = chroma.sum(dim=1, keepdim=True)
    denominator = energy.where(energy > 0, 1.0)
    normalized = chroma / denominator
    coefficients = torch.fft.rfft(normalized, n=12, dim=1)[:, 1:7]
    weights = coefficients.new_tensor(_TIV_AUDIO_WEIGHTS)[None, :, None]
    weighted = coefficients * weights
    return torch.stack((weighted.real, weighted.imag), dim=2).flatten(1, 2).to(torch.float32)


@jaxtyped(typechecker=beartype)
def _audio_chroma(
    audio: Float[torch.Tensor, "batch channels samples"], sample_rate: int
) -> Float[torch.Tensor, "batch 12 frames"]:
    """Compute nearest-bin 12-TET chroma from a waveform batch.

    :param audio: Waveforms shaped ``(batch, channels, samples)``.
    :param sample_rate: Waveform sample rate in Hz.
    :returns: Non-negative STFT-magnitude chroma with C at row zero.
    :raises ValueError: The sample rate is not positive.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    mono = audio.to(torch.float32).mean(dim=1)
    if mono.shape[-1] < _CHROMA_N_FFT:
        mono = F.pad(mono, (0, _CHROMA_N_FFT - mono.shape[-1]))
    spectrum = torch.stft(
        mono,
        n_fft=_CHROMA_N_FFT,
        hop_length=_CHROMA_HOP_LENGTH,
        window=torch.blackman_window(_CHROMA_N_FFT, device=mono.device),
        center=False,
        return_complex=True,
    ).abs()
    frequencies = torch.fft.rfftfreq(_CHROMA_N_FFT, d=1.0 / sample_rate, device=mono.device)
    positive = frequencies > 0
    midi = _MIDI_A4 + 12.0 * torch.log2(frequencies[positive] / _A4_HZ)
    pitch_classes = torch.round(midi).to(torch.int64).remainder(12)
    chroma = spectrum.new_zeros(spectrum.shape[0], 12, spectrum.shape[2])
    indices = pitch_classes[None, :, None].expand(spectrum.shape[0], -1, spectrum.shape[2])
    chroma.scatter_add_(1, indices, spectrum[:, positive])
    return chroma


@jaxtyped(typechecker=beartype)
def _audio_hpcp(
    audio: Float[torch.Tensor, "batch channels samples"], sample_rate: int
) -> Float[torch.Tensor, "batch 12 frames"]:
    """Compute temporal CPU HPCP using the TIVlib example's peak settings.

    :param audio: Waveform batch, copied to CPU for Essentia extraction.
    :param sample_rate: Waveform sample rate in Hz; no resampling is performed.
    :returns: C-rooted, unnormalized HPCP frames on CPU.
    :raises ImportError: The optional Essentia dependency is unavailable.
    :raises ValueError: Audio is empty or not sampled at 44.1 kHz.
    """
    if sample_rate != 44_100:
        raise ValueError("Essentia TIV requires 44100 Hz audio; resample the dataset first")
    if audio.shape[0] == 0 or audio.shape[1] == 0 or audio.shape[2] == 0:
        raise ValueError("Essentia TIV requires nonempty waveform batches")
    try:
        standard = import_module("essentia.standard")
    except ImportError as exc:
        raise ImportError("CPU TIV requires Essentia: uv sync --extra tiv-cpu") from exc

    mono = audio.detach().to(device="cpu", dtype=torch.float32).mean(dim=1).numpy()
    windowing = standard.Windowing(type="blackmanharris62")
    spectrum = standard.Spectrum()
    peaks = standard.SpectralPeaks(
        sampleRate=sample_rate,
        orderBy="magnitude",
        magnitudeThreshold=0.001,
        maxPeaks=5,
        minFrequency=20,
        maxFrequency=8000,
    )
    hpcp = standard.HPCP(sampleRate=sample_rate, maxFrequency=8000, normalized="none")
    profiles = [
        np.stack(
            [
                np.roll(hpcp(*peaks(spectrum(windowing(frame)))), -3)
                for frame in standard.FrameGenerator(waveform, frameSize=1024, hopSize=512)
            ],
            axis=-1,
        )
        for waveform in mono
    ]
    return torch.from_numpy(np.stack(profiles))


@jaxtyped(typechecker=beartype)
def extract_tiv_batch(
    audio: Float[torch.Tensor, "batch channels samples"],
    sample_rate: int,
    *,
    backend: TIVBackend = "torch",
) -> Float[torch.Tensor, "batch 12 frames"]:
    """Extract temporal TIV controls from a waveform batch.

    :param audio: Waveforms shaped ``(batch, channels, samples)``.
    :param sample_rate: Waveform sample rate in Hz.
    :param backend: Device-local STFT chroma or CPU Essentia HPCP extraction.
    :returns: Float32 weighted TIV coordinates on the input audio device.
    """
    chroma = (
        _audio_hpcp(audio, sample_rate)
        if backend == "essentia"
        else _audio_chroma(audio, sample_rate)
    )
    return chroma_to_tiv(chroma).to(audio.device)
