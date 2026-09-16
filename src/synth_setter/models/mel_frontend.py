"""Exportable mel front end reproducing the sketch CLI's librosa features in-graph.

Typical usage::

    frontend = NormalizedMelFrontend(sample_rate=44_100, mean=stats["mean"], std=stats["std"])
    export_frontend_onnx(frontend.eval(), Path("bundle/frontend.onnx"))
"""

from pathlib import Path

import librosa
import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from scipy.signal import get_window
from torch import nn
from torch.nn import functional as F  # noqa: N812 — torch convention

from synth_setter.data.vst.shapes import mel_hop_length, mel_n_fft

_OPSET_VERSION = 18
_N_MELS = 128
_SPEC_SHAPE = (1, _N_MELS, 401)
_DURATION_SECONDS = 4.0
# librosa.power_to_db defaults used by ``make_spectrogram``.
_AMIN = 1e-10
_TOP_DB = 80.0


class NormalizedMelFrontend(nn.Module):
    """Waveform to statistics-normalized log-mel on the fixed ``(1, 128, 401)`` grid.

    The STFT is a strided convolution against window-scaled DFT kernels so the graph exports
    without an STFT operator and runs on ONNX Runtime Web.

    .. attribute :: kernels

        Window-scaled cosine then negative-sine DFT rows shaped ``(2 * bins, 1, n_fft)``.

    .. attribute :: mel_basis

        Slaney-normalized librosa mel filterbank shaped ``(128, bins)``.

    .. attribute :: mean

        Training mel mean on the ``(1, 128, 401)`` grid.

    .. attribute :: std

        Training mel standard deviation on the ``(1, 128, 401)`` grid.
    """

    kernels: torch.Tensor
    mel_basis: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

    @jaxtyped(typechecker=beartype)
    def __init__(self, *, sample_rate: int, mean: np.ndarray, std: np.ndarray) -> None:
        """Bind the analysis kernels and training statistics as buffers.

        :param sample_rate: Waveform rate in Hz; fixes n_fft, hop, and the mel basis.
        :param mean: Training mel mean shaped ``(1, 128, 401)``.
        :param std: Training mel standard deviation shaped ``(1, 128, 401)``, strictly positive.
        :raises ValueError: Statistics are off-grid, non-finite, or not strictly positive.
        """
        super().__init__()
        mean_array = np.asarray(mean, dtype=np.float32)
        std_array = np.asarray(std, dtype=np.float32)
        if mean_array.shape != _SPEC_SHAPE or std_array.shape != _SPEC_SHAPE:
            raise ValueError(f"mel statistics must be shaped {_SPEC_SHAPE}")
        if not np.isfinite(mean_array).all() or not np.isfinite(std_array).all():
            raise ValueError("mel statistics must be finite")
        if np.any(std_array <= 0.0):
            raise ValueError("mel standard deviations must be strictly positive")
        n_fft = mel_n_fft(sample_rate)
        self.hop = mel_hop_length(sample_rate)
        self.pad = n_fft // 2
        self.frames = int(sample_rate * _DURATION_SECONDS)
        window = get_window("hamming", n_fft, fftbins=True)
        bins = np.arange(n_fft // 2 + 1)
        phase = 2.0 * np.pi * np.outer(bins, np.arange(n_fft)) / n_fft
        kernels = np.concatenate((np.cos(phase), -np.sin(phase))) * window
        self.register_buffer("kernels", torch.from_numpy(kernels[:, None, :].astype(np.float32)))
        basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=_N_MELS)
        self.register_buffer("mel_basis", torch.from_numpy(np.asarray(basis, dtype=np.float32)))
        self.register_buffer("mean", torch.from_numpy(mean_array))
        self.register_buffer("std", torch.from_numpy(std_array))

    @jaxtyped(typechecker=beartype)
    def forward(
        self, waveform: Float[torch.Tensor, "batch samples"]
    ) -> Float[torch.Tensor, "batch 1 bins frames"]:
        """Return the normalized log-mel for a centered, zero-padded STFT.

        :param waveform: Mono float32 waveform shaped ``(batch, samples)``.
        :returns: Normalized spectrogram shaped ``(batch, 1, 128, 401)``.
        """
        padded = F.pad(waveform.unsqueeze(1), (self.pad, self.pad))
        spectrum = F.conv1d(padded, self.kernels, stride=self.hop)
        real, imag = spectrum.chunk(2, dim=1)
        power = real * real + imag * imag
        mel = torch.matmul(self.mel_basis, power)
        amin = torch.full_like(mel, _AMIN)
        log_mel = 10.0 * torch.log10(torch.maximum(mel, amin))
        reference = torch.amax(mel, dim=(1, 2), keepdim=True)
        log_mel = log_mel - 10.0 * torch.log10(torch.maximum(reference, amin))
        floor = torch.amax(log_mel, dim=(1, 2), keepdim=True) - _TOP_DB
        log_mel = torch.maximum(log_mel, floor)
        return ((log_mel - self.mean) / self.std).unsqueeze(1)


@jaxtyped(typechecker=beartype)
def export_frontend_onnx(frontend: NormalizedMelFrontend, output: Path) -> None:
    """Write the fixed-shape single-clip front-end graph.

    :param frontend: CPU front end in evaluation mode.
    :param output: Absent ``.onnx`` destination.
    :raises FileExistsError: The destination already exists.
    """
    if output.exists():
        raise FileExistsError(output)
    waveform = torch.zeros(1, frontend.frames)
    with torch.no_grad():
        torch.onnx.export(
            frontend,
            (waveform,),
            output,
            input_names=["waveform"],
            output_names=["mel"],
            opset_version=_OPSET_VERSION,
            dynamo=True,
            external_data=False,
        )
