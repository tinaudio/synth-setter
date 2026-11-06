"""Offline sketch-control extraction keyed on a checkpoint's control profile."""

import numpy as np
import torch
from jaxtyping import Float

from synth_setter.conditioning import SketchControlSpec
from synth_setter.features.pyfdn_controls import extract_reverb_sketch
from synth_setter.features.sketch_controls import extract_sketch_controls_batch
from synth_setter.features.tiv import extract_tiv_batch
from synth_setter.sketch import prepare_music_sketch_controls, resample_sketch_controls


@torch.no_grad()
def extract_music_sketch_controls(
    audio: torch.Tensor, sample_rate: int, spec: SketchControlSpec
) -> Float[torch.Tensor, "batch controls num_frames"]:
    """Extract music controls for a waveform batch on the checkpoint frame grid.

    The single raw-to-checkpoint path shared by offline extraction and both online
    datamodule hooks, so threshold, grid, and autograd policy cannot drift between them.

    :param audio: ``(batch, channels, samples)`` waveforms on the extraction device.
    :param sample_rate: Waveform sample rate in Hz.
    :param spec: Music sketch contract selecting frame count, resampling, and threshold.
    :returns: Float32 controls shaped ``(batch, controls, num_frames)`` on ``audio``'s device;
        a non-music spec or a clip shorter than its grid is rejected by the preparation step.
    """
    music = prepare_music_sketch_controls(extract_sketch_controls_batch(audio, sample_rate), spec)
    if spec.profile == "music_tiv":
        tiv = extract_tiv_batch(audio, sample_rate, backend=spec.tiv_backend)
        tiv = resample_sketch_controls(tiv, spec.num_frames, spec.temporal_resampling)
        return torch.cat((music, tiv), dim=1)
    return music


def extract_profile_controls(
    audio: np.ndarray, sample_rate: int, spec: SketchControlSpec
) -> Float[torch.Tensor, "1 controls frames"]:
    """Extract one clip's sketch controls on the checkpoint's control grid.

    :param audio: Channel-first float32 waveform.
    :param sample_rate: Waveform sample rate in Hz.
    :param spec: Checkpoint sketch-control contract selecting the profile.
    :returns: Float32 controls shaped ``(1, controls, num_frames)``.
    :raises ValueError: The profile has no offline extractor or the audio violates it.
    """
    if spec.profile == "pyfdn_reverb":
        if audio.ndim != 2 or audio.shape[0] != 1:
            raise ValueError("pyfdn_reverb sketch controls require a mono (1, samples) waveform")
        sketch = extract_reverb_sketch(audio[0], sample_rate)
        return torch.from_numpy(sketch).unsqueeze(0)
    if spec.profile not in ("music", "music_tiv"):
        raise ValueError(f"no offline sketch extractor for profile {spec.profile!r}")
    return extract_music_sketch_controls(torch.from_numpy(audio)[None], sample_rate, spec)
