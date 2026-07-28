"""Sketch2Sound-style time-varying control extraction on the mel frame grid.

All tracks use fixed absolute scalings (dB floor, MIDI/127) rather than dataset
statistics: sketches are user input in absolute units and must stay
checkpoint-portable across datasets.
"""

import functools

import torch
import torchaudio

from synth_setter.data.vst.shapes import mel_hop_length, mel_n_fft

_LOUDNESS_DB_FLOOR = -80.0
_EPS = 1e-10


def _mono(audio: torch.Tensor) -> torch.Tensor:
    """Downmix ``(channels, samples)`` audio to a mono ``(samples,)`` signal.

    :param audio: Audio of shape ``(channels, samples)`` or ``(samples,)``.
    :returns: Mono signal.
    """
    return audio.mean(dim=0) if audio.ndim == 2 else audio


def _match_length(track: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Trim or edge-pad a 1-D track to exactly ``n_frames``.

    :param track: Per-frame values of shape ``(frames,)``.
    :param n_frames: Target frame count.
    :returns: Track of shape ``(n_frames,)``.
    """
    if track.shape[-1] >= n_frames:
        return track[..., :n_frames]
    return torch.nn.functional.pad(track, (0, n_frames - track.shape[-1]), mode="replicate")


def _n_frames(audio: torch.Tensor, sample_rate: int) -> int:
    """Return the mel-grid frame count for ``audio`` (librosa ``center=True``).

    :param audio: Audio of shape ``(channels, samples)`` or ``(samples,)``.
    :param sample_rate: Sample rate in Hz.
    :returns: Frame count ``1 + samples // hop``.
    """
    return 1 + audio.shape[-1] // mel_hop_length(sample_rate)


def _midi_from_hz(frequency_hz: torch.Tensor) -> torch.Tensor:
    """Convert frequencies in Hz to MIDI note numbers.

    :param frequency_hz: Frequencies in Hz; non-positive values map far below 0 and are expected to
        be clamped by the caller.
    :returns: MIDI note numbers.
    """
    return 69.0 + 12.0 * torch.log2(frequency_hz.clamp_min(_EPS) / 440.0)


def loudness_track(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Per-frame RMS loudness on the mel grid, mapped to [0, 1] via (dB + 80) / 80.

    :param audio: Audio of shape ``(channels, samples)`` or ``(samples,)``.
    :param sample_rate: Sample rate in Hz.
    :returns: Loudness of shape ``(n_frames,)`` in [0, 1].
    """
    mono = _mono(audio)
    n_fft = mel_n_fft(sample_rate)
    hop = mel_hop_length(sample_rate)
    padded = torch.nn.functional.pad(mono[None], (n_fft // 2, n_fft // 2), mode="reflect")[0]
    frames = padded.unfold(-1, n_fft, hop)
    rms = frames.pow(2).mean(-1).sqrt()
    db = 20.0 * torch.log10(rms.clamp_min(10.0 ** (_LOUDNESS_DB_FLOOR / 20.0)))
    track = ((db - _LOUDNESS_DB_FLOOR) / -_LOUDNESS_DB_FLOOR).clamp(0.0, 1.0)
    return _match_length(track, _n_frames(audio, sample_rate))


def spectral_centroid_track(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Per-frame spectral centroid on the mel grid, as MIDI / 127 clamped to [0, 1].

    :param audio: Audio of shape ``(channels, samples)`` or ``(samples,)``.
    :param sample_rate: Sample rate in Hz.
    :returns: Centroid of shape ``(n_frames,)`` in [0, 1]; silent frames map to 0.
    """
    mono = _mono(audio)
    n_fft = mel_n_fft(sample_rate)
    centroid_hz = torchaudio.functional.spectral_centroid(
        mono[None],
        sample_rate,
        pad=0,
        window=torch.hann_window(n_fft, device=mono.device),
        n_fft=n_fft,
        hop_length=mel_hop_length(sample_rate),
        win_length=n_fft,
    )[0]
    # Silent frames divide 0/0 into NaN; a 0 Hz centroid clamps to the 0 end.
    centroid_hz = torch.nan_to_num(centroid_hz, nan=0.0)
    track = (_midi_from_hz(centroid_hz) / 127.0).clamp(0.0, 1.0)
    return _match_length(track, _n_frames(audio, sample_rate))


@functools.cache
def _pesto_model(step_size_ms: float) -> torch.nn.Module:
    """Load and cache the PESTO pitch estimator.

    :param step_size_ms: Analysis hop in milliseconds.
    :returns: Loaded PESTO model.
    """
    import pesto

    return pesto.load_model("mir-1k_g7", step_size=step_size_ms)


@torch.no_grad()
def pitch_track(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Per-frame PESTO f0 on the mel grid, as MIDI / 127 clamped to [0, 1].

    :param audio: Audio of shape ``(channels, samples)`` or ``(samples,)``.
    :param sample_rate: Sample rate in Hz.
    :returns: Pitch of shape ``(n_frames,)`` in [0, 1].
    """
    mono = _mono(audio)
    step_size_ms = 1000.0 * mel_hop_length(sample_rate) / sample_rate
    model = _pesto_model(step_size_ms)
    # TODO(follow-up): pesto confidence zero-binning
    midi, _confidence, _amplitude, _activations = model(mono[None], sample_rate)
    track = (midi[0] / 127.0).clamp(0.0, 1.0)
    return _match_length(track, _n_frames(audio, sample_rate))


def extract_sketch_controls(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Stack loudness, spectral centroid, and pitch tracks mapped to [-1, 1].

    :param audio: Audio of shape ``(channels, samples)`` or ``(samples,)``.
    :param sample_rate: Sample rate in Hz.
    :returns: Control matrix of shape ``(3, n_frames)`` float32 in [-1, 1].
    :raises ValueError: If any extracted track contains non-finite values.
    """
    tracks = torch.stack(
        [
            loudness_track(audio, sample_rate),
            spectral_centroid_track(audio, sample_rate),
            pitch_track(audio, sample_rate),
        ]
    ).to(dtype=torch.float32)
    if not torch.isfinite(tracks).all():
        raise ValueError("sketch control extraction produced non-finite values")
    return tracks * 2.0 - 1.0
