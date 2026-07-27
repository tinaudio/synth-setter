"""Step C of the simulator-feedback spike (#2553): control-field finetuning.

Loads the frozen Step B base flow and compares three arms on held-out data:

- ``base``: frozen base flow only.
- ``no_feedback``: control field trained with ``c = 0`` (capacity-matched
  ablation).
- ``feedback``: control field trained with the gradient control signal
  ``c = [C; grad C]`` from an in-loop differentiable torchsynth render.

Both control arms train only the control field with the ordinary CFM loss on
``t ~ U(0.8, 1)``; the base flow and encoder stay frozen. CFG dropout is off
during finetuning (the control signal presumes conditioning is present).

Run: ``uv run python -m prototypes.torchsynth_feedback.step_c_finetune``
"""

from __future__ import annotations

import logging
import time

import torch

from prototypes.torchsynth_feedback.flow import (
    CONTROL_T_MIN,
    MIDI_PITCH,
    SAMPLE_RATE,
    SIGNAL_LENGTH,
    ControlField,
    SampleConfig,
    SpectrumEncoder,
    build_base_flow,
    control_signal,
    sample_batch,
    sample_ode,
)
from prototypes.torchsynth_feedback.grad_render import (
    log_spectral_distance,
    multi_scale_log_mel_distance,
)
from prototypes.torchsynth_feedback.step_b_pretrain import ARTIFACTS_DIR
from synth_setter.data.torchsynth_datamodule import render_torchsynth
from synth_setter.data.vst.torchsynth_param_spec import NUM_PARAMS
from synth_setter.models.components.vector_field import VectorField

_LOG = logging.getLogger(__name__)

BATCH_SIZE = 128
FINETUNE_STEPS = 6_000
LEARNING_RATE = 3e-4
EVAL_BATCHES = 4
EVAL_SEED = 999


def finetune_control(
    encoder: SpectrumEncoder,
    vector_field: VectorField,
    device: str,
    *,
    feedback: bool,
    seed: int = 7,
) -> ControlField:
    """Train a control field on t in [0.8, 1] with the CFM loss.

    :param encoder: Frozen conditioning encoder.
    :param vector_field: Frozen base flow.
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
            signal = control_signal(x_t, t, base_velocity, audio)
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
    device: str,
) -> dict[str, float]:
    """Evaluate one arm on the shared held-out stream.

    :param label: Arm name for the log line.
    :param encoder: Frozen conditioning encoder.
    :param vector_field: Frozen base flow.
    :param control: Control field, or ``None`` for the base-only arm.
    :param feedback: Whether sampling feeds the simulator signal to the control.
    :param device: Torch device string.
    :returns: Mean ``param_mse`` (model space) and ``lsd`` over held-out batches.
    """
    generator = torch.Generator().manual_seed(EVAL_SEED)
    noise_generator = torch.Generator(device="cpu").manual_seed(EVAL_SEED + 1)
    mses, lsds, mslms = [], [], []
    for _ in range(EVAL_BATCHES):
        params, audio = sample_batch(BATCH_SIZE, device, generator)
        noise = torch.randn(params.shape, generator=noise_generator).to(device)
        preds = sample_ode(
            encoder,
            vector_field,
            audio,
            noise,
            control_field=control,
            config=SampleConfig(steps=50, feedback=feedback),
        )
        mses.append((preds - params).square().mean().item())
        with torch.no_grad():
            pred_audio = render_torchsynth(
                ((preds + 1) / 2).clamp(0, 1),
                sample_rate=SAMPLE_RATE,
                signal_length=SIGNAL_LENGTH,
                midi_pitch=MIDI_PITCH,
            )
            lsds.append(log_spectral_distance(pred_audio, audio).mean().item())
            mslms.append(
                multi_scale_log_mel_distance(pred_audio, audio, SAMPLE_RATE).mean().item()
            )
    result = {
        "param_mse": sum(mses) / len(mses),
        "lsd": sum(lsds) / len(lsds),
        "mslm": sum(mslms) / len(mslms),
    }
    _LOG.info(
        "[arm=%s] param_mse=%.4f lsd=%.3f mslm=%.4f",
        label,
        result["param_mse"],
        result["lsd"],
        result["mslm"],
    )
    return result


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

    results = {"base": evaluate_arm("base", encoder, vector_field, None, False, device)}
    ablation = finetune_control(encoder, vector_field, device, feedback=False)
    results["no_feedback"] = evaluate_arm(
        "no_feedback", encoder, vector_field, ablation, False, device
    )
    torch.save(ablation.state_dict(), ARTIFACTS_DIR / "control_no_feedback.pt")
    control = finetune_control(encoder, vector_field, device, feedback=True)
    results["feedback"] = evaluate_arm("feedback", encoder, vector_field, control, True, device)
    torch.save(control.state_dict(), ARTIFACTS_DIR / "control_feedback.pt")

    _LOG.info("\n=== summary (param_mse in [-1,1] space / LSD dB / multi-scale log-mel) ===")
    for label, metrics in results.items():
        _LOG.info(
            "%-12s param_mse=%.4f lsd=%.3f mslm=%.4f",
            label,
            metrics["param_mse"],
            metrics["lsd"],
            metrics["mslm"],
        )


if __name__ == "__main__":
    main()
