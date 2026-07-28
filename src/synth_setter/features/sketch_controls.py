"""Sketch-control extraction: loudness, spectral centroid, and pitch contours.

Implements the control set of Sketch2Sound (arXiv:2412.08550) / FlashFoley on the
repo's mel frame grid (100 fps), normalized with fixed affine constants so a
sketch means the same thing for every checkpoint. Tracking issue: #2612.
"""

import torch
import torchaudio

from synth_setter.data.vst.shapes import (
    MEL_FRAMES_PER_SECOND,
    NUM_SKETCH_CONTROLS,
    mel_hop_length,
    mel_n_fft,
    mel_n_frames_from_samples,
)

__all__ = [
    "NUM_SKETCH_CONTROLS",
    "extract_sketch_controls",
    "extract_sketch_controls_batch",
    "loudness_track",
    "pitch_track",
    "sketch_num_frames",
    "spectral_centroid_track",
]
_LOUDNESS_FLOOR_DB = -80.0
_MIDI_A4 = 69.0
_MIDI_MAX = 127.0
_A4_HZ = 440.0
# Guards log2(0) -> -inf and negative inputs -> NaN ahead of the MIDI clamp.
_MIN_HZ = _A4_HZ * 2.0 ** (-_MIDI_A4 / 12.0)
# PESTO step in ms matching the 100 fps mel grid for every sample rate.
_PESTO_STEP_MS = 1000.0 / MEL_FRAMES_PER_SECOND

_pesto_model = None


def sketch_num_frames(num_samples: int, sample_rate: int) -> int:
    """Frame count of every sketch track on the centered mel grid.

    :param num_samples: Waveform length in samples.
    :param sample_rate: Audio sample rate in Hz.
    :returns: ``num_samples // hop + 1`` frames.
    """
    return mel_n_frames_from_samples(num_samples, sample_rate)


def _to_mono_batch(audio: torch.Tensor) -> torch.Tensor:
    """Downmix ``(C, T)`` or ``(B, C, T)`` audio to a ``(B, T)`` float32 batch.

    :param audio: Multichannel waveform(s).
    :returns: Mono batch.
    """
    batched = audio if audio.ndim == 3 else audio[None]
    return batched.to(torch.float32).mean(dim=1)


def _signed_unit(zero_one: torch.Tensor) -> torch.Tensor:
    """Map a ``[0, 1]`` track to the model's ``[-1, 1]`` convention.

    :param zero_one: Track normalized to the unit interval.
    :returns: Signed-unit track.
    """
    return zero_one * 2.0 - 1.0


def _midi_from_hz(hz: torch.Tensor) -> torch.Tensor:
    """Convert Hz to the papers' MIDI-like ``[-1, 1]`` scale.

    :param hz: Frequencies; non-positive values clamp to the MIDI floor.
    :returns: Signed-unit MIDI-scale values.
    """
    midi = _MIDI_A4 + 12.0 * torch.log2(hz.clamp_min(_MIN_HZ) / _A4_HZ)
    return _signed_midi(midi)


def _signed_midi(midi: torch.Tensor) -> torch.Tensor:
    """Clamp MIDI values to ``[0, 127]`` and map to ``[-1, 1]``.

    :param midi: Fractional MIDI note numbers.
    :returns: Signed-unit MIDI-scale values.
    """
    return _signed_unit(midi.clamp(0.0, _MIDI_MAX) / _MIDI_MAX)


def _fit_frames(track: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Trim or edge-pad the last axis to the shared frame grid.

    :param track: ``(..., F)`` per-frame track.
    :param num_frames: Target frame count.
    :returns: ``(..., num_frames)`` track.
    """
    if track.shape[-1] >= num_frames:
        return track[..., :num_frames]
    pad = track[..., -1:].expand(*track.shape[:-1], num_frames - track.shape[-1])
    return torch.cat((track, pad), dim=-1)


def _loudness_batch(mono: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Per-frame RMS loudness in signed-unit dB scale for a mono batch.

    :param mono: ``(B, T)`` waveforms.
    :param sample_rate: Audio sample rate in Hz.
    :returns: ``(B, F)`` loudness tracks.
    """
    # TODO(#2615): A-weighted spectral RMS (Sketch2Sound's variant).
    # Zero-pads edges while the centroid's stft reflects; interior frames agree.
    win = mel_n_fft(sample_rate)
    hop = mel_hop_length(sample_rate)
    padded = torch.nn.functional.pad(mono, (win // 2, win // 2))
    frames = padded.unfold(-1, win, hop)
    rms = frames.pow(2).mean(-1).sqrt()
    db = 20.0 * torch.log10(rms.clamp_min(10.0 ** (_LOUDNESS_FLOOR_DB / 20.0)))
    return _signed_unit(
        (db.clamp(_LOUDNESS_FLOOR_DB, 0.0) - _LOUDNESS_FLOOR_DB) / -_LOUDNESS_FLOOR_DB
    )


def _centroid_batch(mono: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Per-frame spectral-centroid track on the MIDI-like scale for a mono batch.

    :param mono: ``(B, T)`` waveforms.
    :param sample_rate: Audio sample rate in Hz.
    :returns: ``(B, F)`` centroid tracks.
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
    return _midi_from_hz(centroid_hz)


@torch.no_grad()
def _pitch_batch(mono: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Per-frame PESTO f0 track on the MIDI-like scale for a mono batch.

    :param mono: ``(B, T)`` waveforms.
    :param sample_rate: Audio sample rate in Hz.
    :returns: ``(B, F)`` pitch tracks at the PESTO step size (10 ms).
    """
    # TODO(#2614): zero out low-confidence frames (Sketch2Sound threshold 0.1).
    import pesto

    global _pesto_model
    if _pesto_model is None:
        _pesto_model = pesto.load_model("mir-1k_g7", step_size=_PESTO_STEP_MS)
    # PESTO predicts fractional semitones (convert_to_freq=False default).
    preds, _confidence, _, _ = _pesto_model(mono, sample_rate)
    return _signed_midi(preds)


def loudness_track(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Per-frame RMS loudness of one clip in ``[-1, 1]``.

    dB floor is -80 (silence maps to -1); 0 dBFS maps to +1.

    :param audio: ``(C, T)`` waveform.
    :param sample_rate: Audio sample rate in Hz.
    :returns: ``(F,)`` loudness track on the mel frame grid.
    """
    num_frames = sketch_num_frames(audio.shape[-1], sample_rate)
    return _fit_frames(_loudness_batch(_to_mono_batch(audio), sample_rate), num_frames)[0]


def spectral_centroid_track(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Per-frame spectral centroid of one clip on the MIDI-like ``[-1, 1]`` scale.

    :param audio: ``(C, T)`` waveform.
    :param sample_rate: Audio sample rate in Hz.
    :returns: ``(F,)`` centroid track on the mel frame grid.
    """
    num_frames = sketch_num_frames(audio.shape[-1], sample_rate)
    return _fit_frames(_centroid_batch(_to_mono_batch(audio), sample_rate), num_frames)[0]


def pitch_track(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Per-frame PESTO f0 of one clip on the MIDI-like ``[-1, 1]`` scale.

    :param audio: ``(C, T)`` waveform.
    :param sample_rate: Audio sample rate in Hz.
    :returns: ``(F,)`` pitch track on the mel frame grid.
    """
    num_frames = sketch_num_frames(audio.shape[-1], sample_rate)
    return _fit_frames(_pitch_batch(_to_mono_batch(audio), sample_rate), num_frames)[0]


def extract_sketch_controls_batch(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Extract all sketch controls for an audio batch.

    :param audio: ``(B, C, T)`` waveforms.
    :param sample_rate: Audio sample rate in Hz.
    :returns: ``(B, NUM_SKETCH_CONTROLS, F)`` float32 controls, rows ordered
        loudness, centroid, pitch.
    """
    mono = _to_mono_batch(audio)
    num_frames = sketch_num_frames(mono.shape[-1], sample_rate)
    tracks = (
        _loudness_batch(mono, sample_rate),
        _centroid_batch(mono, sample_rate),
        _pitch_batch(mono, sample_rate),
    )
    return torch.stack([_fit_frames(track, num_frames) for track in tracks], dim=1).to(
        torch.float32
    )


def extract_sketch_controls(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Extract all sketch controls for one clip.

    :param audio: ``(C, T)`` waveform.
    :param sample_rate: Audio sample rate in Hz.
    :returns: ``(NUM_SKETCH_CONTROLS, F)`` float32 controls, rows ordered
        loudness, centroid, pitch.
    """
    return extract_sketch_controls_batch(audio[None], sample_rate)[0]
