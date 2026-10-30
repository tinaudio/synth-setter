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
def median_filter_sketch_controls(
    controls: Float[torch.Tensor, f"batch {NUM_SKETCH_CONTROLS} frames"],
    kernel_size: int,
) -> Float[torch.Tensor, f"batch {NUM_SKETCH_CONTROLS} frames"]:
    """Apply FlashFoley's zero-padded temporal median filter.

    Even kernels use the lower median and discard the extra trailing frame, matching FlashFoley's
    released training implementation.

    :param controls: Loudness, centroid, and pitch controls on one temporal grid.
    :param kernel_size: Positive temporal window width in control frames.
    :returns: Filtered controls with input shape, dtype, and device preserved.
    :raises ValueError: The kernel size is not positive.
    """
    if kernel_size <= 0:
        raise ValueError("Sketch median filtering requires a positive kernel size")
    padding = kernel_size // 2
    padded = F.pad(controls, (padding, padding), mode="constant", value=0.0)
    windows = padded.unfold(dimension=-1, size=kernel_size, step=1)
    filtered = windows.median(dim=-1).values
    return filtered[..., : controls.shape[-1]]


@jaxtyped(typechecker=beartype)
def random_median_filter_sketch_controls(
    controls: Float[torch.Tensor, f"batch {NUM_SKETCH_CONTROLS} frames"],
    max_kernel_size: int,
) -> Float[torch.Tensor, f"batch {NUM_SKETCH_CONTROLS} frames"]:
    """Sample and apply one median-filter width per batch row.

    The sampled width is shared by loudness, centroid, and pitch, preserving
    their temporal alignment while varying sketch specificity per example.

    :param controls: Music sketch controls on the model's temporal grid.
    :param max_kernel_size: Inclusive upper bound sampled uniformly from ``[1, max]``.
    :returns: Randomly filtered controls with input shape, dtype, and device preserved.
    :raises ValueError: The maximum kernel size is not positive.
    """
    if max_kernel_size <= 0:
        raise ValueError("Sketch median filtering requires a positive maximum kernel size")
    if max_kernel_size == 1:
        return controls.clone()

    kernel_sizes = torch.randint(
        1,
        max_kernel_size + 1,
        (controls.shape[0],),
        device=controls.device,
    )
    filtered = controls.clone()
    for kernel_size in range(2, max_kernel_size + 1):
        candidate = median_filter_sketch_controls(controls, kernel_size)
        selected_rows = kernel_sizes == kernel_size
        filtered = torch.where(selected_rows[:, None, None], candidate, filtered)
    return filtered


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
