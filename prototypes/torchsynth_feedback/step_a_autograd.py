"""Step A of the simulator-feedback spike (#2553): TorchSynth autograd proof.

Renders a random parameter batch with gradients enabled, backprops a
log-spectral cost against a target render, and reports which of the 76
inferable parameters receive zero/NaN gradients, plus render throughput
(forward and forward+backward) on CPU and GPU.

Run: ``uv run python -m prototypes.torchsynth_feedback.step_a_autograd``
"""

from __future__ import annotations

import logging
import time

import torch

from prototypes.torchsynth_feedback.grad_render import (
    log_spectral_distance,
    render_torchsynth_grad,
)
from synth_setter.data.vst.torchsynth_param_spec import INFERABLE_SPEC

SAMPLE_RATE = 44_100
SIGNAL_LENGTH = 4_410
MIDI_PITCH = 60
BATCH_SIZE = 64
_LOG = logging.getLogger(__name__)


def gradient_audit(device: str, batch_size: int = BATCH_SIZE, seed: int = 0) -> None:
    """Backprop a spectral cost through the render and report per-param gradients.

    :param device: Torch device string.
    :param batch_size: Fixed render batch size (renderer cache is keyed on it).
    :param seed: RNG seed for the parameter draws.
    """
    generator = torch.Generator().manual_seed(seed)
    target_params = torch.rand((batch_size, len(INFERABLE_SPEC)), generator=generator).to(device)
    with torch.no_grad():
        target_audio = render_torchsynth_grad(
            target_params,
            sample_rate=SAMPLE_RATE,
            signal_length=SIGNAL_LENGTH,
            midi_pitch=MIDI_PITCH,
        )
    params = (
        torch.rand((batch_size, len(INFERABLE_SPEC)), generator=generator)
        .to(device)
        .requires_grad_(True)
    )
    audio = render_torchsynth_grad(
        params,
        sample_rate=SAMPLE_RATE,
        signal_length=SIGNAL_LENGTH,
        midi_pitch=MIDI_PITCH,
    )
    cost = log_spectral_distance(audio, target_audio).sum()
    cost.backward()

    grad = params.grad
    assert grad is not None, "params.grad is None: autograd did not reach the parameters"
    _LOG.info(f"\n=== gradient audit ({device}, batch={batch_size}) ===")
    _LOG.info(f"cost: {cost.item():.4f}  grad finite: {torch.isfinite(grad).all().item()}")
    _LOG.info(f"{'param':<32} {'mean|grad|':>12} {'zero-frac':>10} {'nan-frac':>9}")
    zero_params, nan_params = [], []
    for index, spec in enumerate(INFERABLE_SPEC):
        column = grad[:, index]
        mean_abs = column.abs().mean().item()
        zero_frac = (column == 0).float().mean().item()
        nan_frac = torch.isnan(column).float().mean().item()
        _LOG.info(f"{spec.key:<32} {mean_abs:>12.3e} {zero_frac:>10.2f} {nan_frac:>9.2f}")
        if zero_frac == 1.0:
            zero_params.append(spec.key)
        if nan_frac > 0.0:
            nan_params.append(spec.key)
    _LOG.info(f"\nall-zero-grad params ({len(zero_params)}): {zero_params}")
    _LOG.info(f"params with NaN grads ({len(nan_params)}): {nan_params}")


def benchmark(device: str, batch_size: int = BATCH_SIZE, iters: int = 10) -> None:
    """Measure render throughput forward-only and forward+backward.

    :param device: Torch device string.
    :param batch_size: Fixed render batch size.
    :param iters: Timed iterations (after one warmup).
    """
    params = torch.rand((batch_size, len(INFERABLE_SPEC)), device=device)
    target = torch.rand((batch_size, len(INFERABLE_SPEC)), device=device)
    with torch.no_grad():
        target_audio = render_torchsynth_grad(
            target, sample_rate=SAMPLE_RATE, signal_length=SIGNAL_LENGTH, midi_pitch=MIDI_PITCH
        )

    def run(with_backward: bool) -> float:
        for _ in range(2):  # warmup (includes renderer construction on first call)
            _step(with_backward)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            _step(with_backward)
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - start) / iters

    def _step(with_backward: bool) -> None:
        p = params.clone().requires_grad_(with_backward)
        audio = render_torchsynth_grad(
            p, sample_rate=SAMPLE_RATE, signal_length=SIGNAL_LENGTH, midi_pitch=MIDI_PITCH
        )
        if with_backward:
            log_spectral_distance(audio, target_audio).sum().backward()

    fwd = run(False)
    fwd_bwd = run(True)
    _LOG.info(f"\n=== throughput ({device}, batch={batch_size}) ===")
    _LOG.info(f"forward:          {fwd * 1000:8.1f} ms/batch  {batch_size / fwd:10.1f} samples/s")
    _LOG.info(
        f"forward+backward: {fwd_bwd * 1000:8.1f} ms/batch  {batch_size / fwd_bwd:10.1f} samples/s"
    )


def main() -> None:
    """Run the audit and benchmark on CPU and, when available, GPU."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(0)
    gradient_audit("cpu")
    benchmark("cpu")
    if torch.cuda.is_available():
        gradient_audit("cuda")
        benchmark("cuda")


if __name__ == "__main__":
    main()
