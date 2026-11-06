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
    SketchControlSpec,
    SketchTemporalResampling,
)


@jaxtyped(typechecker=beartype)
def median_filter_sketch_controls(
    controls: Float[torch.Tensor, "batch channels frames"],
    kernel_size: int,
) -> Float[torch.Tensor, "batch channels frames"]:
    """Apply FlashFoley's zero-padded temporal median filter.

    Even kernels use the lower median and discard the extra trailing frame, matching FlashFoley's
    released training implementation.

    :param controls: Control channels on one temporal grid.
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
    controls: Float[torch.Tensor, "batch channels frames"],
    max_kernel_size: int,
) -> Float[torch.Tensor, "batch channels frames"]:
    """Sample and apply one median-filter width per batch row.

    The sampled width is shared by all channels, preserving temporal alignment.

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
def resample_sketch_controls(
    controls: Float[torch.Tensor, "batch channels frames"],
    output_frames: int,
    mode: SketchTemporalResampling,
) -> Float[torch.Tensor, "batch channels output_frames"]:
    """Resample controls onto a positive temporal grid.

    :param controls: Control channels on any nonempty temporal grid.
    :param output_frames: Positive output-grid length.
    :param mode: ``average`` for adaptive average pooling or ``linear`` interpolation.
    :returns: Resampled controls with input dtype and device preserved.
    :raises ValueError: The source or output grid is empty.
    """
    if controls.shape[-1] == 0 or output_frames <= 0:
        raise ValueError("Sketch resampling requires nonempty input and positive output frames")
    if controls.shape[-1] == output_frames:
        return controls.clone()
    if mode == "average":
        return F.adaptive_avg_pool1d(controls, output_frames)
    if controls.shape[-1] < output_frames:
        return F.interpolate(controls, size=output_frames, mode="linear", align_corners=False)
    # Plain linear reduction samples two neighbours per output frame and skips the rest;
    # anti-aliasing (bilinear-only in torch, hence the unit height axis) low-passes first.
    return F.interpolate(
        controls.unsqueeze(2),
        size=(1, output_frames),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(2)


@jaxtyped(typechecker=beartype)
def prepare_music_sketch_controls(
    controls: Float[torch.Tensor, f"batch {NUM_SKETCH_CONTROLS} frames"],
    spec: SketchControlSpec,
) -> Float[torch.Tensor, f"batch {NUM_SKETCH_CONTROLS} num_frames"]:
    """Prepare extracted music controls on the checkpoint frame grid.

    :param controls: Raw extracted controls on a nonempty temporal grid.
    :param spec: Music sketch contract selecting frame count, resampling, and pitch threshold.
    :returns: Independent float32 controls on ``spec.num_frames`` preserving the input device.
    :raises ValueError: The spec does not select a music profile, or the extracted grid is
        sparser than the frame grid (upsampling would fabricate frames).
    """
    if spec.profile not in ("music", "music_tiv"):
        raise ValueError("music-control preparation requires a music profile")
    if controls.shape[-1] < spec.num_frames:
        raise ValueError(
            f"music-control preparation requires at least {spec.num_frames} frames, "
            f"got {controls.shape[-1]}"
        )
    # resample_sketch_controls always returns fresh storage, so no clone is needed here.
    prepared = resample_sketch_controls(
        controls,
        output_frames=spec.num_frames,
        mode=spec.temporal_resampling,
    ).to(dtype=torch.float32)
    pitch = prepared[:, SKETCH_PITCH_SLICE]
    prepared[:, SKETCH_PITCH_SLICE] = pitch.where(
        pitch >= spec.pitch_zero_threshold,
        0.0,
    )
    return prepared


@jaxtyped(typechecker=beartype)
def pool_sketch_controls(
    controls: Float[torch.Tensor, f"batch {NUM_SKETCH_CONTROLS} frames"],
    output_frames: int = SKETCH_STORAGE_FRAMES,
) -> Float[torch.Tensor, f"batch {NUM_SKETCH_CONTROLS} output_frames"]:
    """Pool controls with track and pitch means.

    :param controls: Loudness, centroid, and pitch controls on any temporal grid.
    :param output_frames: Frames retained along the pooled time axis.
    :returns: Controls with averaged tracks and averaged pitch activations; empty or
        nonpositive grids are rejected by :func:`resample_sketch_controls`.
    """
    return resample_sketch_controls(controls, output_frames, "average")
