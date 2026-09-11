"""Exportable stereo mel and music-sketch front ends for browser inference."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, cast

import librosa
import numpy as np
import pesto
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import nn
from torch.nn import functional as F  # noqa: N812 — torch convention
from torchaudio.transforms import Resample

from synth_setter.data.vst.shapes import DEFAULT_PESTO_CHECKPOINT, mel_hop_length, mel_n_fft
from synth_setter.models.mel_frontend import NormalizedMelFrontend

_OPSET_VERSION = 18
_SAMPLE_FRAMES = 176_400
_MEL_FRAMES = 401
_NUM_CONTROLS = 386
_PITCH_BINS = 384
_PESTO_BINS = 251
_PESTO_CROP = 16
_PESTO_SHIFT_BINS = 20
_LOUDNESS_SAMPLE_RATE = 16_000
_LOUDNESS_N_FFT = 1024
_LOUDNESS_MIN_DB = -100.0
_LOUDNESS_MAX_DB = 80.0
_LOUDNESS_PEAK_RANGE_DB = 80.0
_LOG_EPSILON = torch.finfo(torch.float32).eps
_MIN_HZ = 440.0 * 2.0 ** (-69.0 / 12.0)


@jaxtyped(typechecker=beartype)
def _dft_kernels(n_fft: int, window: np.ndarray) -> Float[torch.Tensor, "twice_bins 1 n_fft"]:
    bins = np.arange(n_fft // 2 + 1)
    phase = 2.0 * np.pi * np.outer(bins, np.arange(n_fft)) / n_fft
    kernels = np.concatenate((np.cos(phase), -np.sin(phase))) * window
    return torch.from_numpy(kernels[:, None, :].astype(np.float32))


class StereoMelFrontend(nn.Module):
    """Apply the native mono mel front end independently to both stereo channels."""

    @jaxtyped(typechecker=beartype)
    def __init__(self, *, sample_rate: int, mean: np.ndarray, std: np.ndarray) -> None:
        """Bind channel-specific statistics to two fixed-grid mel front ends.

        :param sample_rate: Waveform rate in Hz.
        :param mean: Training means shaped ``(2, 128, 401)``.
        :param std: Positive training standard deviations with the same shape.
        :raises ValueError: Statistics do not contain exactly two channels.
        """
        super().__init__()
        if np.shape(mean) != (2, 128, _MEL_FRAMES) or np.shape(std) != (2, 128, _MEL_FRAMES):
            raise ValueError("stereo mel statistics must be shaped (2, 128, 401)")
        self.left = NormalizedMelFrontend(
            sample_rate=sample_rate, mean=np.asarray(mean)[0:1], std=np.asarray(std)[0:1]
        )
        self.right = NormalizedMelFrontend(
            sample_rate=sample_rate, mean=np.asarray(mean)[1:2], std=np.asarray(std)[1:2]
        )
        self.frames = self.left.frames

    @jaxtyped(typechecker=beartype)
    def forward(
        self, waveform: Float[torch.Tensor, "batch 2 samples"]
    ) -> Float[torch.Tensor, "batch 2 128 401"]:
        """Return channel-specific normalized log-mel features.

        :param waveform: Stereo float32 clips on the fixed four-second grid.
        :returns: Normalized mel shaped ``(batch, 2, 128, 401)``.
        """
        left = self.left(waveform[:, 0])
        right = self.right(waveform[:, 1])
        return torch.cat((left, right), dim=1)


class MusicSketchFrontend(nn.Module):
    """Reproduce native loudness, centroid, PESTO activation, pooling, and thresholding.

    .. attribute :: a_weighting

        Fixed loudness-bin A-weighting.

    .. attribute :: centroid_frequencies

        Centroid DFT-bin frequencies in Hz.

    .. attribute :: centroid_kernels

        Windowed centroid DFT kernels.

    .. attribute :: cqt_kernels

        Trained PESTO CQT kernels.

    .. attribute :: cqt_lengths

        PESTO librosa-normalization factors.

    .. attribute :: encoder

        Trained activation-only PESTO encoder.

    .. attribute :: loudness_kernels

        Windowed loudness DFT kernels.

    .. attribute :: resample_kernel

        Frozen torchaudio 44.1-to-16 kHz sinc kernel.
    """

    a_weighting: Float[torch.Tensor, "1 bins 1"]
    centroid_frequencies: Float[torch.Tensor, "1 bins 1"]
    centroid_kernels: Float[torch.Tensor, "twice_bins 1 n_fft"]
    cqt_kernels: Float[torch.Tensor, "twice_bins 1 n_fft"]
    cqt_lengths: Float[torch.Tensor, "bins 1"]
    encoder: nn.Module
    loudness_kernels: Float[torch.Tensor, "twice_bins 1 n_fft"]
    resample_kernel: Float[torch.Tensor, "channels 1 width"]

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        *,
        sample_rate: int,
        output_frames: int,
        pitch_zero_threshold: float,
        pesto_checkpoint: str = DEFAULT_PESTO_CHECKPOINT,
    ) -> None:
        """Freeze production analysis kernels and the cached trained PESTO encoder.

        :param sample_rate: Input waveform rate in Hz; only 44.1 kHz is supported.
        :param output_frames: Temporal size expected by the checkpoint; must be 32.
        :param pitch_zero_threshold: Activations below this value become zero after pooling.
        :param pesto_checkpoint: Cached PESTO checkpoint name.
        :raises ValueError: The requested grid or PESTO model differs from production.
        """
        super().__init__()
        if sample_rate != 44_100 or output_frames != 32:
            raise ValueError("music sketch export requires 44.1 kHz and 32 output frames")
        if not 0.0 <= pitch_zero_threshold <= 1.0:
            raise ValueError("pitch zero threshold must be within [0, 1]")
        model = cast(Any, pesto.load_model(pesto_checkpoint, step_size=10.0).eval())
        model.preprocessor.hcqt(torch.zeros(1, _SAMPLE_FRAMES), sr=sample_rate)
        cqt = model.preprocessor.hcqt_kernels.cqt_kernels[0]
        shift_bins = round(model.shift.item() * model.bins_per_semitone)
        if (
            cqt.kernel_width != 8192
            or model.encoder.hparams["output_dim"] != _PITCH_BINS
            or shift_bins != _PESTO_SHIFT_BINS
        ):
            raise ValueError("PESTO checkpoint does not match the mir-1k_g7 browser graph")
        self.encoder = copy.deepcopy(model.encoder).eval()
        self.register_buffer("cqt_kernels", cqt.conv.weight.detach().clone())
        self.register_buffer("cqt_lengths", cqt.sqrt_lengths.detach().clone())

        resampler = cast(Any, Resample)(sample_rate, _LOUDNESS_SAMPLE_RATE)
        self.register_buffer("resample_kernel", resampler.kernel.detach().clone())
        self.resample_width = resampler.width
        self.resample_stride = sample_rate // math.gcd(sample_rate, _LOUDNESS_SAMPLE_RATE)
        self.resample_channels = _LOUDNESS_SAMPLE_RATE // math.gcd(
            sample_rate, _LOUDNESS_SAMPLE_RATE
        )
        hann = np.asarray(torch.hann_window(_LOUDNESS_N_FFT).numpy())
        self.register_buffer("loudness_kernels", _dft_kernels(_LOUDNESS_N_FFT, hann))
        frequencies = librosa.fft_frequencies(sr=_LOUDNESS_SAMPLE_RATE, n_fft=_LOUDNESS_N_FFT)
        a_weighting = librosa.A_weighting(frequencies, min_db=None) - 20.0
        self.register_buffer(
            "a_weighting", torch.tensor(a_weighting, dtype=torch.float32)[None, :, None]
        )

        centroid_n_fft = mel_n_fft(sample_rate)
        hamming = np.asarray(torch.hamming_window(centroid_n_fft).numpy())
        self.register_buffer("centroid_kernels", _dft_kernels(centroid_n_fft, hamming))
        self.register_buffer(
            "centroid_frequencies",
            torch.linspace(0, sample_rate // 2, centroid_n_fft // 2 + 1)[None, :, None],
        )
        self.sample_rate = sample_rate
        self.output_frames = output_frames
        self.pitch_zero_threshold = pitch_zero_threshold
        self.frames = _SAMPLE_FRAMES

    @jaxtyped(typechecker=beartype)
    def _spectrum(
        self,
        waveform: Float[torch.Tensor, "batch samples"],
        kernels: Float[torch.Tensor, "twice_bins 1 n_fft"],
        hop: int,
        pad_mode: str,
    ) -> tuple[Float[torch.Tensor, "batch bins frames"], Float[torch.Tensor, "batch bins frames"]]:
        """Return real and imaginary DFT components using exportable convolutions.

        :param waveform: Mono waveform batch.
        :param kernels: Window-scaled real then imaginary DFT rows.
        :param hop: Frame stride in samples.
        :param pad_mode: Center-padding mode used by the native transform.
        :returns: Real and imaginary spectra.
        """
        pad = kernels.shape[-1] // 2
        padded = F.pad(waveform[:, None], (pad, pad), mode=pad_mode)
        spectrum = F.conv1d(padded, kernels, stride=hop)
        real, imag = spectrum.chunk(2, dim=1)
        return real, imag

    @jaxtyped(typechecker=beartype)
    def _resample(
        self, waveform: Float[torch.Tensor, "batch samples"]
    ) -> Float[torch.Tensor, "batch resampled"]:
        """Apply torchaudio's frozen 44.1-to-16 kHz sinc kernel.

        :param waveform: Mono 44.1 kHz clips.
        :returns: Clips on the 16 kHz loudness grid.
        """
        right = self.resample_width + self.resample_stride
        padded = F.pad(waveform, (self.resample_width, right))
        convolved = F.conv1d(padded[:, None], self.resample_kernel, stride=self.resample_stride)
        return convolved.transpose(1, 2).reshape(waveform.shape[0], -1)[:, :64_000]

    @jaxtyped(typechecker=beartype)
    def _loudness(
        self, waveform: Float[torch.Tensor, "batch samples"]
    ) -> Float[torch.Tensor, "batch 1 401"]:
        """Return native A-weighted signed-unit loudness.

        :param waveform: Mono input clips.
        :returns: Loudness on the mel frame grid.
        """
        resampled = self._resample(waveform)
        real, imag = self._spectrum(resampled, self.loudness_kernels, 160, "constant")
        power = real.square() + imag.square()
        db = 10.0 * torch.log10(power.clamp_min(1e-10))
        floor = db.amax(dim=(1, 2), keepdim=True) - _LOUDNESS_PEAK_RANGE_DB
        weighted = (torch.maximum(db, floor) + self.a_weighting).clamp_min(_LOUDNESS_MIN_DB)
        unit = (weighted.mean(dim=1, keepdim=True) - _LOUDNESS_MIN_DB) / (
            _LOUDNESS_MAX_DB - _LOUDNESS_MIN_DB
        )
        return unit.clamp(0.0, 1.0) * 2.0 - 1.0

    @jaxtyped(typechecker=beartype)
    def _centroid(
        self, waveform: Float[torch.Tensor, "batch samples"]
    ) -> Float[torch.Tensor, "batch 1 401"]:
        """Return native signed-unit MIDI spectral centroid.

        :param waveform: Mono input clips.
        :returns: Centroid on the mel frame grid.
        """
        real, imag = self._spectrum(
            waveform, self.centroid_kernels, mel_hop_length(self.sample_rate), "reflect"
        )
        scale = torch.maximum(real.abs(), imag.abs())
        safe_scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
        magnitude = safe_scale * torch.sqrt(
            (real / safe_scale).square() + (imag / safe_scale).square()
        )
        magnitude = torch.where(scale == 0.0, torch.zeros_like(magnitude), magnitude)
        centroid = (magnitude * self.centroid_frequencies).sum(1) / magnitude.sum(1)
        centroid = torch.where(torch.isnan(centroid), torch.zeros_like(centroid), centroid)
        midi = 69.0 + 12.0 * torch.log2(centroid.clamp_min(_MIN_HZ) / 440.0)
        return (midi.clamp(0.0, 127.0) / 127.0 * 2.0 - 1.0)[:, None]

    @jaxtyped(typechecker=beartype)
    def _pitch(
        self, waveform: Float[torch.Tensor, "batch samples"]
    ) -> Float[torch.Tensor, "batch 384 401"]:
        """Return raw trained PESTO activations without confidence or frequency paths.

        :param waveform: Mono input clips.
        :returns: Shifted activation matrix on the mel frame grid.
        """
        padded = F.pad(waveform[:, None], (4096, 4096), mode="reflect")
        cqt = F.conv1d(padded, self.cqt_kernels, stride=441)
        cqt = cqt.reshape(waveform.shape[0], 2, _PESTO_BINS, _MEL_FRAMES)
        magnitude = torch.sqrt((cqt.square()).sum(dim=1)) * self.cqt_lengths[None]
        log_magnitude = 20.0 * torch.log10(magnitude.clamp_min(_LOG_EPSILON))
        # PESTO mutates this tensor while deriving energy before encoder cropping.
        encoder_input = log_magnitude * (math.log(10.0) / 10.0)
        cropped = encoder_input[:, _PESTO_CROP:-_PESTO_CROP].permute(0, 2, 1)
        activations = self.encoder(cropped.reshape(-1, 1, 219))
        activations = activations.reshape(waveform.shape[0], _MEL_FRAMES, _PITCH_BINS)
        return activations.roll(-_PESTO_SHIFT_BINS, dims=-1).transpose(1, 2)

    @jaxtyped(typechecker=beartype)
    def _pool(
        self, controls: Float[torch.Tensor, "batch 386 401"]
    ) -> Float[torch.Tensor, "batch 386 output_frames"]:
        """Apply the native adaptive mean/max windows without an ONNX adaptive-pool op.

        :param controls: Unpooled scalar and pitch tracks.
        :returns: Controls on the checkpoint temporal grid.
        """
        tracks, pitch = controls[:, :2], controls[:, 2:]
        pooled_tracks = []
        pooled_pitch = []
        for index in range(self.output_frames):
            start = index * _MEL_FRAMES // self.output_frames
            end = math.ceil((index + 1) * _MEL_FRAMES / self.output_frames)
            pooled_tracks.append(tracks[:, :, start:end].mean(dim=-1))
            pooled_pitch.append(pitch[:, :, start:end].amax(dim=-1))
        return torch.cat(
            (torch.stack(pooled_tracks, dim=-1), torch.stack(pooled_pitch, dim=-1)), dim=1
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, waveform: Float[torch.Tensor, "batch 2 samples"]
    ) -> Float[torch.Tensor, "batch 386 output_frames"]:
        """Return model-ready pooled controls from arbitrary stereo content.

        :param waveform: Stereo float32 clips on the fixed four-second grid.
        :returns: Loudness, centroid, and thresholded PESTO activations.
        """
        mono = waveform.mean(dim=1)
        controls = self._pool(
            torch.cat((self._loudness(mono), self._centroid(mono), self._pitch(mono)), dim=1)
        )
        tracks, pitch = controls[:, :2], controls[:, 2:]
        return torch.cat(
            (tracks, torch.where(pitch >= self.pitch_zero_threshold, pitch, 0.0)), dim=1
        )


@jaxtyped(typechecker=beartype)
def export_stereo_mel_onnx(frontend: StereoMelFrontend, output: Path) -> None:
    """Export the fixed-shape stereo mel graph.

    :param frontend: CPU evaluation front end.
    :param output: Absent graph path.
    """
    _export_graph(frontend, output, "mel")


@jaxtyped(typechecker=beartype)
def export_music_sketch_onnx(frontend: MusicSketchFrontend, output: Path) -> None:
    """Export the fixed-shape music-sketch graph.

    :param frontend: CPU evaluation front end.
    :param output: Absent graph path.
    """
    _export_graph(frontend, output, "sketch_ctrl")


@jaxtyped(typechecker=beartype)
def _export_graph(frontend: nn.Module, output: Path, output_name: str) -> None:
    """Write one fixed browser graph.

    :param frontend: Exportable CPU module.
    :param output: Absent graph path.
    :param output_name: Public ONNX output name.
    :raises FileExistsError: The destination exists.
    """
    if output.exists():
        raise FileExistsError(output)
    with torch.no_grad():
        torch.onnx.export(
            frontend,
            (torch.zeros(1, 2, _SAMPLE_FRAMES),),
            output,
            input_names=["waveform"],
            output_names=[output_name],
            opset_version=_OPSET_VERSION,
            dynamo=True,
            external_data=False,
        )
