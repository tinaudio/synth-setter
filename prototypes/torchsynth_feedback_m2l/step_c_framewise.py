"""Step C follow-up for #2557: feedback arm with an unpooled framewise m2l cost.

The pooled-L2 feedback arm barely beat the ablation while the mel cell's
spectral cost gained -0.55 dB LSD (#2553); this arm tests whether pooling the
latent time axis is what starves the control signal. Trains one feedback
control field whose cost is the framewise m2l MSE (cost scale calibrated on a
random-pair probe) and evaluates it on the same held-out stream as
``step_c_finetune``.

Run: ``uv run python -m prototypes.torchsynth_feedback_m2l.step_c_framewise``
"""

from __future__ import annotations

import logging
import time

import torch

from prototypes.torchsynth_feedback_m2l.flow_m2l import (
    CONTROL_T_MIN,
    ControlField,
    MultiScaleLogMel,
    SampleConfig,
    build_base_flow,
    control_signal,
    sample_batch,
)
from prototypes.torchsynth_feedback_m2l.m2l_grad import M2LGradEncoder, m2l_framewise_mse
from prototypes.torchsynth_feedback_m2l.step_b_pretrain import ARTIFACTS_DIR
from prototypes.torchsynth_feedback_m2l.step_c_finetune import (
    BATCH_SIZE,
    FINETUNE_STEPS,
    LEARNING_RATE,
    evaluate_arm,
)

_LOG = logging.getLogger(__name__)


def calibrate_cost_scale(m2l: M2LGradEncoder, device: str) -> float:
    """Measure the random-pair framewise cost and return a scale making it O(1).

    :param m2l: Grad-enabled m2l encoder.
    :param device: Torch device string.
    :returns: Reciprocal of the mean random-pair framewise MSE.
    """
    generator = torch.Generator().manual_seed(555)
    _, audio_a = sample_batch(BATCH_SIZE, device, generator)
    _, audio_b = sample_batch(BATCH_SIZE, device, generator)
    with torch.no_grad():
        cost = m2l_framewise_mse(m2l(audio_a), m2l(audio_b)).mean().item()
    _LOG.info("random-pair framewise m2l MSE: %.3f -> cost_scale %.4f", cost, 1.0 / cost)
    return 1.0 / cost


def main() -> None:
    """Train and evaluate the framewise-cost feedback arm."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    encoder, vector_field = build_base_flow(device)
    checkpoint = torch.load(ARTIFACTS_DIR / "base_flow.pt", weights_only=True)
    encoder.load_state_dict(checkpoint["encoder"])
    vector_field.load_state_dict(checkpoint["vector_field"])
    encoder.requires_grad_(False).eval()
    vector_field.requires_grad_(False).eval()
    m2l = M2LGradEncoder(device)
    mel_metric = MultiScaleLogMel().to(device)
    cost_scale = calibrate_cost_scale(m2l, device)

    control = ControlField().to(device)
    optimizer = torch.optim.AdamW(control.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(7)
    start = time.perf_counter()
    for step in range(FINETUNE_STEPS):
        params, audio = sample_batch(BATCH_SIZE, device, generator)
        with torch.no_grad():
            conditioning = encoder(audio)
            t = CONTROL_T_MIN + (1 - CONTROL_T_MIN) * torch.rand(params.shape[0], 1, device=device)
            x0 = torch.randn_like(params)
            x_t = x0 * (1 - t) + params * t
            target = params - x0
            base_velocity = vector_field(x_t, t, conditioning)
            target_latents = m2l(audio)
        signal = control_signal(
            x_t, t, base_velocity, target_latents, m2l, framewise=True, cost_scale=cost_scale
        )
        effective = base_velocity + control(t, base_velocity, signal)
        loss = (effective - target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 250 == 0 or step == FINETUNE_STEPS - 1:
            _LOG.info(
                "[framewise] step %5d loss %.4f (%.1f s)",
                step,
                loss.item(),
                time.perf_counter() - start,
            )

    metrics = evaluate_arm(
        "feedback_framewise",
        encoder,
        vector_field,
        control,
        True,
        m2l,
        mel_metric,
        device,
        config=SampleConfig(steps=40, feedback=True, framewise=True, cost_scale=cost_scale),
    )
    torch.save(control.state_dict(), ARTIFACTS_DIR / "control_feedback_framewise.pt")
    _LOG.info(
        "framewise arm: param_mse=%.4f mel=%.4f lsd=%.3f m2l=%.3f",
        metrics["param_mse"],
        metrics["mel"],
        metrics["lsd"],
        metrics["m2l"],
    )


if __name__ == "__main__":
    main()
