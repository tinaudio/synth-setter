"""Step C of the m2l simulator-feedback spike (#2557): control-field finetuning.

Loads the frozen Step B base flow and compares three arms on held-out data:

- ``base``: frozen base flow only.
- ``no_feedback``: control field trained with ``c = 0`` (capacity-matched
  ablation).
- ``feedback``: control field trained with the m2l-space gradient control
  signal ``c = [C; grad C]`` from an in-loop differentiable torchsynth render
  encoded by music2latent.

Both control arms train only the control field with the ordinary CFM loss on
``t ~ U(0.8, 1)``; the base flow and encoder stay frozen. CFG dropout is off
during finetuning (the control signal presumes conditioning is present).

Run: ``uv run python -m prototypes.torchsynth_feedback_m2l.step_c_finetune``
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
    SpectrumEncoder,
    build_base_flow,
    control_signal,
    eval_metrics,
    sample_batch,
    sample_ode,
)
from prototypes.torchsynth_feedback_m2l.m2l_grad import M2LGradEncoder
from prototypes.torchsynth_feedback_m2l.step_b_pretrain import ARTIFACTS_DIR
from synth_setter.data.vst.torchsynth_param_spec import NUM_PARAMS
from synth_setter.models.components.vector_field import VectorField

_LOG = logging.getLogger(__name__)

BATCH_SIZE = 64
FINETUNE_STEPS = 2_000
LEARNING_RATE = 3e-4
EVAL_BATCHES = 4
EVAL_SEED = 999


def finetune_control(
    encoder: SpectrumEncoder,
    vector_field: VectorField,
    m2l: M2LGradEncoder,
    device: str,
    *,
    feedback: bool,
    seed: int = 7,
) -> ControlField:
    """Train a control field on t in [0.8, 1] with the CFM loss.

    :param encoder: Frozen conditioning encoder.
    :param vector_field: Frozen base flow.
    :param m2l: Grad-enabled m2l encoder for the control signal.
    :param device: Torch device string.
    :param feedback: Whether the control signal carries the simulator gradient (else zeros — the
        ablation arm).
    :param seed: RNG seed for the online data stream.
    :returns: Trained control field.
    """
    control = ControlField().to(device)
    _LOG.info(
        "control field params: %d (feedback=%s)",
        sum(p.numel() for p in control.parameters()),
        feedback,
    )
    optimizer = torch.optim.AdamW(control.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(seed)
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
        if feedback:
            with torch.no_grad():
                target_latents = m2l(audio)
            signal = control_signal(x_t, t, base_velocity, target_latents, m2l)
        else:
            signal = torch.zeros((params.shape[0], NUM_PARAMS + 1), device=device)
        effective = base_velocity + control(t, base_velocity, signal)
        loss = (effective - target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 250 == 0 or step == FINETUNE_STEPS - 1:
            _LOG.info(
                "[%s] step %5d loss %.4f (%.1f s)",
                "feedback" if feedback else "no_feedback",
                step,
                loss.item(),
                time.perf_counter() - start,
            )
    return control


def evaluate_arm(
    label: str,
    encoder: SpectrumEncoder,
    vector_field: VectorField,
    control: ControlField | None,
    feedback: bool,
    m2l: M2LGradEncoder,
    mel_metric: MultiScaleLogMel,
    device: str,
    config: SampleConfig | None = None,
) -> dict[str, float]:
    """Evaluate one arm on the shared held-out stream (common protocol).

    :param label: Arm name for the log line.
    :param encoder: Frozen conditioning encoder.
    :param vector_field: Frozen base flow.
    :param control: Control field, or ``None`` for the base-only arm.
    :param feedback: Whether sampling feeds the simulator signal to the control.
    :param m2l: Grad-enabled m2l encoder.
    :param mel_metric: Shared multi-scale log-mel module.
    :param device: Torch device string.
    :param config: Sampling config override; defaults to 40 steps with ``feedback``.
    :returns: Mean ``param_mse``, ``mel``, ``lsd``, ``m2l`` over held-out batches.
    """
    if config is None:
        config = SampleConfig(steps=40, feedback=feedback)
    generator = torch.Generator().manual_seed(EVAL_SEED)
    noise_generator = torch.Generator(device="cpu").manual_seed(EVAL_SEED + 1)
    totals: dict[str, float] = {}
    for _ in range(EVAL_BATCHES):
        params, audio = sample_batch(BATCH_SIZE, device, generator)
        noise = torch.randn(params.shape, generator=noise_generator).to(device)
        preds = sample_ode(
            encoder,
            vector_field,
            audio,
            noise,
            control_field=control,
            m2l=m2l,
            config=config,
        )
        metrics = eval_metrics(preds, params, audio, mel_metric, m2l)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value / EVAL_BATCHES
    _LOG.info(
        "[arm=%s] param_mse=%.4f mel=%.4f lsd=%.3f m2l=%.3f",
        label,
        totals["param_mse"],
        totals["mel"],
        totals["lsd"],
        totals["m2l"],
    )
    return totals


def main() -> None:
    """Finetune both control arms and print the three-way comparison."""
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

    results = {
        "base": evaluate_arm("base", encoder, vector_field, None, False, m2l, mel_metric, device)
    }
    ablation = finetune_control(encoder, vector_field, m2l, device, feedback=False)
    results["no_feedback"] = evaluate_arm(
        "no_feedback", encoder, vector_field, ablation, False, m2l, mel_metric, device
    )
    torch.save(ablation.state_dict(), ARTIFACTS_DIR / "control_no_feedback.pt")
    control = finetune_control(encoder, vector_field, m2l, device, feedback=True)
    results["feedback"] = evaluate_arm(
        "feedback", encoder, vector_field, control, True, m2l, mel_metric, device
    )
    torch.save(control.state_dict(), ARTIFACTS_DIR / "control_feedback.pt")

    _LOG.info("\n=== summary (param_mse in [-1,1] space / mel / lsd dB / m2l) ===")
    for label, metrics in results.items():
        _LOG.info(
            "%-12s param_mse=%.4f mel=%.4f lsd=%.3f m2l=%.3f",
            label,
            metrics["param_mse"],
            metrics["mel"],
            metrics["lsd"],
            metrics["m2l"],
        )


if __name__ == "__main__":
    main()
