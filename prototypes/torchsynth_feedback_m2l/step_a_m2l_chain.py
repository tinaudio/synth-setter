"""Step A of the m2l simulator-feedback spike (#2557): full grad-chain proof.

Backprops a pooled-L2 cost in music2latent latent space through the chain
``render_torchsynth_grad -> (optional resample) -> M2LGradEncoder`` and reports:

- audio-input gradients of the m2l encoder alone (differentiability verdict),
- per-parameter gradient statistics for the full chain (zero/NaN audit),
- gradient survival through ``torchaudio.functional.resample``,
- forward and forward+backward throughput plus peak VRAM per batch size.

Run: ``uv run python -m prototypes.torchsynth_feedback_m2l.step_a_m2l_chain``
"""

from __future__ import annotations

import logging
import time

import torch
import torchaudio.functional as audio_fn

from prototypes.torchsynth_feedback_m2l.grad_render import render_torchsynth_grad
from prototypes.torchsynth_feedback_m2l.m2l_grad import (
    M2L_SAMPLE_RATE,
    M2LGradEncoder,
    m2l_pooled_l2,
)
from synth_setter.data.vst.torchsynth_param_spec import INFERABLE_SPEC

SIGNAL_LENGTH = 22_050
MIDI_PITCH = 60
AUDIT_BATCH = 32
BENCH_BATCHES = (16, 32, 64)
_LOG = logging.getLogger(__name__)


def frontend_parity_check(device: str) -> None:
    """Compare the NaN-safe STFT frontend against music2latent's original.

    :param device: Torch device string.
    """
    from music2latent.audio import to_representation_encoder

    from prototypes.torchsynth_feedback_m2l.m2l_grad import _safe_representation_encoder

    audio = 0.3 * torch.randn(2, SIGNAL_LENGTH, device=device)
    original = to_representation_encoder(audio)
    safe = _safe_representation_encoder(audio)
    difference = (original - safe).abs()
    _LOG.info(
        "=== frontend parity (%s) === max|diff|=%.3e mean|orig|=%.3e",
        device,
        difference.max().item(),
        original.abs().mean().item(),
    )


def audio_input_grad_check(m2l: M2LGradEncoder, device: str) -> None:
    """Verify the m2l encoder alone passes finite non-zero grads to its audio input.

    :param m2l: Grad-enabled encoder.
    :param device: Torch device string.
    """
    audio = (0.1 * torch.randn(2, SIGNAL_LENGTH, device=device)).requires_grad_(True)
    target = m2l(0.1 * torch.randn(2, SIGNAL_LENGTH, device=device)).detach()
    m2l_pooled_l2(m2l(audio), target).sum().backward()
    grad = audio.grad
    assert grad is not None, "no gradient reached the m2l audio input"
    finite = torch.isfinite(grad).all().item()
    nonzero_frac = (grad != 0).float().mean().item()
    _LOG.info(
        "=== m2l audio-input grad (%s) === finite=%s nonzero_frac=%.3f mean|g|=%.3e",
        device,
        finite,
        nonzero_frac,
        grad.abs().mean().item(),
    )


def full_chain_audit(
    m2l: M2LGradEncoder, device: str, batch_size: int = AUDIT_BATCH, seed: int = 0
) -> None:
    """Backprop the m2l cost through the render and report per-param gradients.

    :param m2l: Grad-enabled encoder.
    :param device: Torch device string.
    :param batch_size: Fixed render batch size (renderer cache is keyed on it).
    :param seed: RNG seed for the parameter draws.
    """
    generator = torch.Generator().manual_seed(seed)
    target_params = torch.rand((batch_size, len(INFERABLE_SPEC)), generator=generator).to(device)
    with torch.no_grad():
        target_audio = render_torchsynth_grad(
            target_params,
            sample_rate=M2L_SAMPLE_RATE,
            signal_length=SIGNAL_LENGTH,
            midi_pitch=MIDI_PITCH,
        )
        target_latents = m2l(target_audio)
    params = (
        torch.rand((batch_size, len(INFERABLE_SPEC)), generator=generator)
        .to(device)
        .requires_grad_(True)
    )
    audio = render_torchsynth_grad(
        params,
        sample_rate=M2L_SAMPLE_RATE,
        signal_length=SIGNAL_LENGTH,
        midi_pitch=MIDI_PITCH,
    )
    cost = m2l_pooled_l2(m2l(audio), target_latents)
    cost.sum().backward()

    grad = params.grad
    assert grad is not None, "params.grad is None: autograd did not reach the parameters"
    _LOG.info("\n=== full-chain gradient audit (%s, batch=%d) ===", device, batch_size)
    _LOG.info(
        "cost mean: %.4f  grad finite: %s", cost.mean().item(), torch.isfinite(grad).all().item()
    )
    _LOG.info("%-32s %12s %10s %9s", "param", "mean|grad|", "zero-frac", "nan-frac")
    zero_params, nan_params = [], []
    for index, spec in enumerate(INFERABLE_SPEC):
        column = grad[:, index]
        mean_abs = column.abs().mean().item()
        zero_frac = (column == 0).float().mean().item()
        nan_frac = torch.isnan(column).float().mean().item()
        _LOG.info("%-32s %12.3e %10.2f %9.2f", spec.key, mean_abs, zero_frac, nan_frac)
        if zero_frac == 1.0:
            zero_params.append(spec.key)
        if nan_frac > 0.0:
            nan_params.append(spec.key)
    _LOG.info("\nall-zero-grad params (%d): %s", len(zero_params), zero_params)
    _LOG.info("params with NaN grads (%d): %s", len(nan_params), nan_params)


def resample_grad_check(m2l: M2LGradEncoder, device: str, batch_size: int = 8) -> None:
    """Prove gradients survive a differentiable 22.05->44.1 kHz resample stage.

    :param m2l: Grad-enabled encoder.
    :param device: Torch device string.
    :param batch_size: Fixed render batch size for the low-rate renderer.
    """
    low_rate = M2L_SAMPLE_RATE // 2
    params = torch.rand((batch_size, len(INFERABLE_SPEC)), device=device).requires_grad_(True)
    audio_low = render_torchsynth_grad(
        params,
        sample_rate=low_rate,
        signal_length=SIGNAL_LENGTH // 2,
        midi_pitch=MIDI_PITCH,
    )
    audio = audio_fn.resample(audio_low, orig_freq=low_rate, new_freq=M2L_SAMPLE_RATE)
    target = m2l(torch.zeros_like(audio)).detach()
    m2l_pooled_l2(m2l(audio), target).sum().backward()
    grad = params.grad
    assert grad is not None
    _LOG.info(
        "\n=== resample grad check (%s) === finite=%s nonzero-param-frac=%.3f",
        device,
        torch.isfinite(grad).all().item(),
        (grad.abs().sum(dim=0) > 0).float().mean().item(),
    )


def benchmark(m2l: M2LGradEncoder, device: str, batch_size: int, iters: int = 5) -> None:
    """Measure full-chain throughput and peak VRAM, forward-only and forward+backward.

    :param m2l: Grad-enabled encoder.
    :param device: Torch device string.
    :param batch_size: Fixed render batch size.
    :param iters: Timed iterations (after warmup).
    """
    params = torch.rand((batch_size, len(INFERABLE_SPEC)), device=device)
    with torch.no_grad():
        target_audio = render_torchsynth_grad(
            params,
            sample_rate=M2L_SAMPLE_RATE,
            signal_length=SIGNAL_LENGTH,
            midi_pitch=MIDI_PITCH,
        )
        target_latents = m2l(target_audio)

    def _step(with_backward: bool) -> None:
        p = params.clone().requires_grad_(with_backward)
        context = torch.enable_grad() if with_backward else torch.no_grad()
        with context:
            audio = render_torchsynth_grad(
                p,
                sample_rate=M2L_SAMPLE_RATE,
                signal_length=SIGNAL_LENGTH,
                midi_pitch=MIDI_PITCH,
            )
            cost = m2l_pooled_l2(m2l(audio), target_latents).sum()
        if with_backward:
            cost.backward()

    def run(with_backward: bool) -> tuple[float, float]:
        for _ in range(2):
            _step(with_backward)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(iters):
            _step(with_backward)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iters
        peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else float("nan")
        return elapsed, peak

    fwd, fwd_peak = run(False)
    fwd_bwd, fwd_bwd_peak = run(True)
    _LOG.info(
        "\n=== throughput (%s, batch=%d, %.2fs audio) ===",
        device,
        batch_size,
        SIGNAL_LENGTH / M2L_SAMPLE_RATE,
    )
    _LOG.info(
        "forward:          %8.1f ms/batch %10.1f samples/s  peak %.2f GiB",
        fwd * 1000,
        batch_size / fwd,
        fwd_peak,
    )
    _LOG.info(
        "forward+backward: %8.1f ms/batch %10.1f samples/s  peak %.2f GiB",
        fwd_bwd * 1000,
        batch_size / fwd_bwd,
        fwd_bwd_peak,
    )


def main() -> None:
    """Run the audits and benchmarks on GPU (falling back to a small CPU run)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        device = "cuda"
        m2l = M2LGradEncoder(device)
        frontend_parity_check(device)
        audio_input_grad_check(m2l, device)
        full_chain_audit(m2l, device)
        resample_grad_check(m2l, device)
        for batch_size in BENCH_BATCHES:
            benchmark(m2l, device, batch_size)
    else:
        m2l = M2LGradEncoder("cpu")
        audio_input_grad_check(m2l, "cpu")
        full_chain_audit(m2l, "cpu", batch_size=4)


if __name__ == "__main__":
    main()
