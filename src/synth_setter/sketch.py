"""Shared sketch-control pooling contract.

Typical usage::

    pooled = pool_sketch_controls(controls)
"""

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped

from synth_setter.conditioning import NUM_SKETCH_CONTROLS, SKETCH_STORAGE_FRAMES


@jaxtyped(typechecker=beartype)
def pool_sketch_controls(
    controls: Float[torch.Tensor, f"batch {NUM_SKETCH_CONTROLS} frames"],
    output_frames: int = SKETCH_STORAGE_FRAMES,
) -> Float[torch.Tensor, f"batch {NUM_SKETCH_CONTROLS} output_frames"]:
    """Pool controls with track and pitch means.

    :param controls: Loudness, centroid, and pitch controls on any temporal grid.
    :param output_frames: Frames retained along the pooled time axis.
    :returns: Controls with averaged tracks and averaged pitch activations.
    :raises ValueError: The source grid is empty or output frames are nonpositive.
    """
    if controls.shape[-1] == 0 or output_frames <= 0:
        raise ValueError("Sketch pooling requires nonempty input and positive output frames")
    if controls.shape[-1] == output_frames:
        return controls.clone()
    return F.adaptive_avg_pool1d(controls, output_frames)
