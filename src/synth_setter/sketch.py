"""Shared sketch-control pooling contract.

Typical usage::

    pooled = pool_sketch_controls(controls)
"""

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped

from synth_setter.conditioning import (
    NUM_SKETCH_CONTROLS,
    SKETCH_PITCH_SLICE,
    SKETCH_STORAGE_FRAMES,
)


@jaxtyped(typechecker=beartype)
def pool_sketch_controls(
    controls: Float[torch.Tensor, f"batch {NUM_SKETCH_CONTROLS} frames"],
    output_frames: int = SKETCH_STORAGE_FRAMES,
) -> Float[torch.Tensor, f"batch {NUM_SKETCH_CONTROLS} output_frames"]:
    """Pool controls with track means and pitch maxima.

    :param controls: Loudness, centroid, and pitch controls on any temporal grid.
    :param output_frames: Frames retained along the pooled time axis.
    :returns: Controls with averaged tracks and maximum-pooled pitch activations.
    """
    tracks = F.adaptive_avg_pool1d(controls[:, : SKETCH_PITCH_SLICE.start], output_frames)
    pitch = F.adaptive_max_pool1d(controls[:, SKETCH_PITCH_SLICE], output_frames)
    return torch.cat((tracks, pitch), dim=1)
