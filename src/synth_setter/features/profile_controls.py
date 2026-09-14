"""Offline sketch-control extraction keyed on a checkpoint's control profile."""

import numpy as np
import torch
from jaxtyping import Float

from synth_setter.conditioning import SketchControlSpec
from synth_setter.features.pyfdn_controls import extract_reverb_sketch
from synth_setter.features.sketch_controls import extract_sketch_controls
from synth_setter.sketch import pool_sketch_controls


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
    if spec.profile != "music":
        raise ValueError(f"no offline sketch extractor for profile {spec.profile!r}")
    controls = extract_sketch_controls(torch.from_numpy(audio), sample_rate).unsqueeze(0)
    controls = pool_sketch_controls(controls, spec.num_frames).to(dtype=torch.float32).clone()
    pitch = controls[:, 2:]
    controls[:, 2:] = pitch.where(pitch >= spec.pitch_zero_threshold, 0.0)
    return controls
