"""Sketch-control extraction: loudness, spectral centroid, and pitch contours.

Implements the control set of Sketch2Sound (arXiv:2412.08550) / FlashFoley on the
repo's mel frame grid (100 fps). Loudness and centroid use fixed affine
normalization to ``[-1, 1]`` so a sketch means the same thing for every
checkpoint; pitch is PESTO's raw activation matrix in ``[0, 1]`` (rows
``SKETCH_PITCH_SLICE``). Storage averages the scalar tracks and maximum-pools
pitch to the canonical storage grid without thresholding, so the zero-bin stays
tunable without a re-backfill. Tracking issue: #2612.

Typical usage::

    controls = extract_sketch_controls(audio, sample_rate)  # (386, frames)
    loudness = controls[SKETCH_LOUDNESS_ROW]
    pitch_activations = controls[SKETCH_PITCH_SLICE]
"""

import librosa
import torch
import torchaudio
from beartype import beartype
from jaxtyping import Float, jaxtyped

from synth_setter.data.vst.shapes import (
    DEFAULT_PESTO_CHECKPOINT,
    MEL_FRAMES_PER_SECOND,
    NUM_SKETCH_CONTROLS,
    SKETCH_CENTROID_ROW,
    SKETCH_LOUDNESS_ROW,
    SKETCH_PITCH_BINS,
    SKETCH_PITCH_SLICE,
    mel_hop_length,
    mel_n_fft,
    mel_n_frames_from_samples,
)

__all__ = [
    "DEFAULT_PESTO_CHECKPOINT",
    "NUM_SKETCH_CONTROLS",
    "SKETCH_CENTROID_ROW",
    "SKETCH_LOUDNESS_ROW",
    "SKETCH_PITCH_BINS",
    "SKETCH_PITCH_SLICE",
    "extract_sketch_controls",
    "extract_sketch_controls_batch",
    "load_pesto_model",
    "loudness_track",
    "pitch_track",
    "sketch_num_frames",
    "spectral_centroid_track",
]

_MIDI_A4 = 69.0
_MIDI_MAX = 127.0
_A4_HZ = 440.0
# Guards log2(0) -> -inf and negative inputs -> NaN ahead of the MIDI clamp.
_MIN_HZ = _A4_HZ * 2.0 ** (-_MIDI_A4 / 12.0)
# PESTO step in ms matching the 100 fps mel grid for every sample rate.
_PESTO_STEP_MS = 1000.0 / MEL_FRAMES_PER_SECOND

# Match FlashFoley's A-weighted loudness front end; constants below define it.
_LOUDNESS_SAMPLE_RATE = 16000
_LOUDNESS_N_FFT = 1024
_LOUDNESS_REF_DB = 20.0
_LOUDNESS_MIN_DB = -100.0
_LOUDNESS_MAX_DB = 80.0
# Treat frames outside the clip-relative peak range as that clip's floor.
_LOUDNESS_PEAK_RANGE_DB = 80.0

_pesto_model = None
_pesto_checkpoint: str | None = None
_pesto_device: torch.device | None = None
_a_weights: torch.Tensor | None = None


def load_pesto_model(
    checkpoint: str | None = None, device: str | torch.device | None = None
) -> torch.nn.Module:
    """Load and cache the PESTO model on the 10 ms sketch frame grid.

    Called eagerly by the pipeline's registry loader so the batch transform
    stays free of model-file I/O; the extraction functions fall back to it
    lazily for single-clip use.

    :param checkpoint: PESTO checkpoint name; ``None`` reuses the cached model,
        loading ``DEFAULT_PESTO_CHECKPOINT`` when none is cached yet.
    :param device: Torch device to hold the weights; ``None`` reuses the cached
        device, defaulting to CPU.
    :returns: The cached process-wide model.
    """
    global _pesto_model, _pesto_checkpoint, _pesto_device
    target = checkpoint or _pesto_checkpoint or DEFAULT_PESTO_CHECKPOINT
    target_device = torch.device(device) if device is not None else _pesto_device
    if target_device is None:
        target_device = torch.device("cpu")
    if _pesto_model is None or _pesto_checkpoint != target:
        import pesto

        _pesto_model = pesto.load_model(target, step_size=_PESTO_STEP_MS)
        _pesto_checkpoint = target
        # A freshly loaded model sits on CPU; clear the flag so the move below runs.
        _pesto_device = None
    if _pesto_device != target_device:
        _pesto_model = _pesto_model.to(target_device)
        _pesto_device = target_device
    return _pesto_model


def sketch_num_frames(num_samples: int, sample_rate: int) -> int:
    """Frame count of every sketch track on the centered mel grid.

    :param num_samples: Waveform length in samples.
    :param sample_rate: Audio sample rate in Hz.
    :returns: ``num_samples // hop + 1`` frames.
    """
    return mel_n_frames_from_samples(num_samples, sample_rate)


@jaxtyped(typechecker=beartype)
def _to_mono_batch(audio: torch.Tensor) -> Float[torch.Tensor, "batch time"]:
    """Downmix ``(C, T)`` or ``(B, C, T)`` audio to a ``(B, T)`` float32 batch.

    :param audio: Multichannel waveform(s).
    :returns: Mono batch.
    """
    batched = audio if audio.ndim == 3 else audio[None]
    return batched.to(torch.float32).mean(dim=1)


@jaxtyped(typechecker=beartype)
def _signed_unit(zero_one: Float[torch.Tensor, "*shape"]) -> Float[torch.Tensor, "*shape"]:
    """Map a ``[0, 1]`` track to the model's ``[-1, 1]`` convention.

    :param zero_one: Track normalized to the unit interval.
    :returns: Signed-unit track.
    """
    return zero_one * 2.0 - 1.0


@jaxtyped(typechecker=beartype)
def _midi_from_hz(hz: Float[torch.Tensor, "*shape"]) -> Float[torch.Tensor, "*shape"]:
    """Convert Hz to the papers' MIDI-like ``[-1, 1]`` scale.

    :param hz: Frequencies; non-positive values clamp to the MIDI floor.
    :returns: Signed-unit MIDI-scale values.
    """
    midi = _MIDI_A4 + 12.0 * torch.log2(hz.clamp_min(_MIN_HZ) / _A4_HZ)
    return _signed_unit(midi.clamp(0.0, _MIDI_MAX) / _MIDI_MAX)


@jaxtyped(typechecker=beartype)
def _fit_frames(
    track: Float[torch.Tensor, "batch channel time"], num_frames: int
) -> Float[torch.Tensor, "batch channel frames_out"]:
    """Trim or edge-pad the last axis to the shared frame grid.

    :param track: Per-frame track.
    :param num_frames: Target frame count.
    :returns: Track on the target grid.
    """
    if track.shape[-1] >= num_frames:
        return track[..., :num_frames]
    pad = track[..., -1:].expand(*track.shape[:-1], num_frames - track.shape[-1])
    return torch.cat((track, pad), dim=-1)


def _a_weighting_db(device: torch.device) -> torch.Tensor:
    """A-weighting curve in dB for the loudness STFT bins, cached per process.

    :param device: Device the weights are moved to.
    :returns: ``(bins, 1)`` weights relative to ``_LOUDNESS_REF_DB``.
    """
    global _a_weights
    if _a_weights is None:
        freqs = librosa.fft_frequencies(sr=_LOUDNESS_SAMPLE_RATE, n_fft=_LOUDNESS_N_FFT)
        curve = librosa.A_weighting(freqs, min_db=None) - _LOUDNESS_REF_DB
        _a_weights = torch.tensor(curve, dtype=torch.float32)[:, None]
    return _a_weights.to(device)


@jaxtyped(typechecker=beartype)
def _loudness_batch(
    mono: Float[torch.Tensor, "batch time"], sample_rate: int
) -> Float[torch.Tensor, "batch 1 frames"]:
    """A-weighted per-frame loudness in signed-unit dB scale for a mono batch.

    Follows the FlashFoley recipe: 16 kHz STFT power in dB with a per-clip
    ``peak - 80`` floor, A-weighting added, mean across bins, then the fixed
    ``[-100, 80]`` dB affine to ``[-1, 1]``.

    :param mono: Waveforms at ``sample_rate``.
    :param sample_rate: Audio sample rate in Hz.
    :returns: Loudness tracks on the 10 ms grid.
    """
    hop = mel_hop_length(sample_rate)
    resampled = torchaudio.functional.resample(mono, sample_rate, _LOUDNESS_SAMPLE_RATE)
    hop = int(hop * _LOUDNESS_SAMPLE_RATE / sample_rate)
    window = torch.hann_window(_LOUDNESS_N_FFT, device=mono.device)
    stft = torch.stft(
        resampled,
        n_fft=_LOUDNESS_N_FFT,
        hop_length=hop,
        win_length=_LOUDNESS_N_FFT,
        window=window,
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    power = stft.abs().pow(2)
    db = 10.0 * torch.log10(power.clamp_min(1e-10))
    peak_floor = db.amax(dim=(-2, -1), keepdim=True) - _LOUDNESS_PEAK_RANGE_DB
    db = torch.maximum(db, peak_floor)
    weighted = (db + _a_weighting_db(mono.device)).clamp_min(_LOUDNESS_MIN_DB)
    loudness_db = weighted.mean(dim=1, keepdim=True)
    unit = (loudness_db - _LOUDNESS_MIN_DB) / (_LOUDNESS_MAX_DB - _LOUDNESS_MIN_DB)
    return _signed_unit(unit.clamp(0.0, 1.0))


@jaxtyped(typechecker=beartype)
def _centroid_batch(
    mono: Float[torch.Tensor, "batch time"], sample_rate: int
) -> Float[torch.Tensor, "batch 1 frames"]:
    """Per-frame spectral-centroid track on the MIDI-like scale for a mono batch.

    :param mono: Waveforms at ``sample_rate``.
    :param sample_rate: Audio sample rate in Hz.
    :returns: Centroid tracks on the mel frame grid.
    """
    win = mel_n_fft(sample_rate)
    centroid_hz = torchaudio.functional.spectral_centroid(
        mono,
        sample_rate,
        pad=0,
        # Matches shapes.MEL_WINDOW so the centroid and mel front ends agree.
        window=torch.hamming_window(win, device=mono.device),
        n_fft=win,
        hop_length=mel_hop_length(sample_rate),
        win_length=win,
    )
    # Silent frames divide 0/0; treat them as the MIDI floor.
    centroid_hz = torch.nan_to_num(centroid_hz, nan=0.0)
    return _midi_from_hz(centroid_hz)[:, None, :]


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def _pitch_batch(
    mono: Float[torch.Tensor, "batch time"], sample_rate: int
) -> Float[torch.Tensor, f"batch {SKETCH_PITCH_BINS} frames"]:
    """Raw PESTO pitch-activation matrices for a mono batch.

    Rows are pitch-bin probabilities in ``[0, 1]`` (three bins per semitone;
    a frame's bins sum to ~1 when voiced and stay near zero on silence).
    Stored unthresholded — zero-binning below Sketch2Sound's 0.1 happens at
    consumption time (#2614).

    :param mono: Waveforms at ``sample_rate``.
    :param sample_rate: Audio sample rate in Hz.
    :returns: Activation tracks at the PESTO step size (10 ms).
    """
    _preds, _confidence, _, activations = load_pesto_model(device=mono.device)(mono, sample_rate)
    return activations.detach().to(torch.float32).transpose(1, 2)


@jaxtyped(typechecker=beartype)
def loudness_track(audio: torch.Tensor, sample_rate: int) -> Float[torch.Tensor, " frames"]:
    """A-weighted per-frame loudness of one clip in ``[-1, 1]``.

    Fixed dB affine: -100 dB maps to -1, 80 dB to +1; silence lands at the
    clip-relative floor.

    :param audio: ``(C, T)`` waveform.
    :param sample_rate: Audio sample rate in Hz.
    :returns: Loudness track on the mel frame grid.
    """
    num_frames = sketch_num_frames(audio.shape[-1], sample_rate)
    return _fit_frames(_loudness_batch(_to_mono_batch(audio), sample_rate), num_frames)[0, 0]


@jaxtyped(typechecker=beartype)
def spectral_centroid_track(
    audio: torch.Tensor, sample_rate: int
) -> Float[torch.Tensor, " frames"]:
    """Per-frame spectral centroid of one clip on the MIDI-like ``[-1, 1]`` scale.

    :param audio: ``(C, T)`` waveform.
    :param sample_rate: Audio sample rate in Hz.
    :returns: Centroid track on the mel frame grid.
    """
    num_frames = sketch_num_frames(audio.shape[-1], sample_rate)
    return _fit_frames(_centroid_batch(_to_mono_batch(audio), sample_rate), num_frames)[0, 0]


@jaxtyped(typechecker=beartype)
def pitch_track(
    audio: torch.Tensor, sample_rate: int
) -> Float[torch.Tensor, f"{SKETCH_PITCH_BINS} frames"]:
    """Raw PESTO pitch-activation matrix of one clip in ``[0, 1]``.

    :param audio: ``(C, T)`` waveform.
    :param sample_rate: Audio sample rate in Hz.
    :returns: Activation matrix on the mel frame grid.
    """
    num_frames = sketch_num_frames(audio.shape[-1], sample_rate)
    return _fit_frames(_pitch_batch(_to_mono_batch(audio), sample_rate), num_frames)[0]


@jaxtyped(typechecker=beartype)
def extract_sketch_controls_batch(
    audio: torch.Tensor, sample_rate: int, device: str | torch.device | None = None
) -> Float[torch.Tensor, f"batch {NUM_SKETCH_CONTROLS} frames"]:
    """Extract all sketch controls for an audio batch.

    Every track follows the batch, so moving it here puts the whole extractor —
    PESTO included — on one device.

    :param audio: ``(B, C, T)`` waveforms.
    :param sample_rate: Audio sample rate in Hz.
    :param device: Torch device to extract on; ``None`` keeps ``audio``'s.
    :returns: Float32 controls; rows are ``SKETCH_LOUDNESS_ROW``,
        ``SKETCH_CENTROID_ROW``, then ``SKETCH_PITCH_SLICE``.
    """
    mono = _to_mono_batch(audio)
    if device is not None:
        mono = mono.to(device)
    num_frames = sketch_num_frames(mono.shape[-1], sample_rate)
    tracks = (
        _loudness_batch(mono, sample_rate),
        _centroid_batch(mono, sample_rate),
        _pitch_batch(mono, sample_rate),
    )
    return torch.cat([_fit_frames(track, num_frames) for track in tracks], dim=1).to(torch.float32)


@jaxtyped(typechecker=beartype)
def extract_sketch_controls(
    audio: torch.Tensor, sample_rate: int
) -> Float[torch.Tensor, f"{NUM_SKETCH_CONTROLS} frames"]:
    """Extract all sketch controls for one clip.

    :param audio: ``(C, T)`` waveform.
    :param sample_rate: Audio sample rate in Hz.
    :returns: Float32 controls; rows are ``SKETCH_LOUDNESS_ROW``,
        ``SKETCH_CENTROID_ROW``, then ``SKETCH_PITCH_SLICE``.
    """
    return extract_sketch_controls_batch(audio[None], sample_rate)[0]
