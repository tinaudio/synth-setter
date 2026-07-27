"""Step B of the m2l simulator-feedback spike (#2557): pretrain a small base flow.

Trains the encoder + vector field with the conditional flow-matching loss on
online torchsynth data (fresh random params rendered per step), then evaluates
held-out metrics via Euler sampling. Recipe reused from the #2553 mel cell;
geometry is 0.5 s (m2l minimum) and batch 64 (VRAM cap from Step A), with
proportionally fewer steps. Overtraining a toy is fine — the goal is a
non-trivial frozen base for Step C.

Run: ``uv run python -m prototypes.torchsynth_feedback_m2l.step_b_pretrain``
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from prototypes.torchsynth_feedback_m2l.flow_m2l import (
    MultiScaleLogMel,
    SampleConfig,
    SpectrumEncoder,
    build_base_flow,
    eval_metrics,
    sample_batch,
    sample_ode,
)
from prototypes.torchsynth_feedback_m2l.m2l_grad import M2LGradEncoder
from synth_setter.models.components.vector_field import VectorField

_LOG = logging.getLogger(__name__)

BATCH_SIZE = 64
TRAIN_STEPS = 40_000
LEARNING_RATE = 1e-3
CFG_DROPOUT_RATE = 0.1
EVAL_BATCHES = 4
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def evaluate(
    encoder: SpectrumEncoder,
    vector_field: VectorField,
    device: str,
    seed: int = 999,
    label: str = "eval",
) -> dict[str, float]:
    """Report held-out metrics (common protocol) via Euler sampling.

    :param encoder: Audio conditioning encoder.
    :param vector_field: Base flow to sample from.
    :param device: Torch device string.
    :param seed: Held-out draw seed (disjoint from the training stream).
    :param label: Tag for the log line.
    :returns: Mean ``param_mse``, ``mel``, ``lsd``, ``m2l`` over held-out batches.
    """
    m2l = M2LGradEncoder(device)
    mel_metric = MultiScaleLogMel().to(device)
    generator = torch.Generator().manual_seed(seed)
    totals: dict[str, float] = {}
    for _ in range(EVAL_BATCHES):
        params, audio = sample_batch(BATCH_SIZE, device, generator)
        noise = torch.randn_like(params)
        preds = sample_ode(encoder, vector_field, audio, noise, config=SampleConfig(steps=40))
        metrics = eval_metrics(preds, params, audio, mel_metric, m2l)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value / EVAL_BATCHES
    _LOG.info(
        "[%s] param_mse=%.4f mel=%.4f lsd=%.3f m2l=%.3f",
        label,
        totals["param_mse"],
        totals["mel"],
        totals["lsd"],
        totals["m2l"],
    )
    return totals


def main() -> None:
    """Pretrain the base flow and save it under ``artifacts/``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    encoder, vector_field = build_base_flow(device)
    parameters = list(encoder.parameters()) + list(vector_field.parameters())
    total = sum(p.numel() for p in parameters)
    _LOG.info("base flow params: %d (device=%s)", total, device)
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_STEPS)

    generator = torch.Generator().manual_seed(123)
    start = time.perf_counter()
    for step in range(TRAIN_STEPS):
        params, audio = sample_batch(BATCH_SIZE, device, generator)
        conditioning = encoder(audio)
        z = vector_field.apply_dropout(conditioning, CFG_DROPOUT_RATE)
        with torch.no_grad():
            t = torch.rand(params.shape[0], 1, device=device)
            x0 = torch.randn_like(params)
            x_t = x0 * (1 - t) + params * t
            target = params - x0
        prediction = vector_field(x_t, t, z)
        loss = (prediction - target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 500 == 0 or step == TRAIN_STEPS - 1:
            _LOG.info(
                "step %5d loss %.4f (%.1f s)", step, loss.item(), time.perf_counter() - start
            )

    evaluate(encoder, vector_field, device, label="base-flow")
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    torch.save(
        {"encoder": encoder.state_dict(), "vector_field": vector_field.state_dict()},
        ARTIFACTS_DIR / "base_flow.pt",
    )
    _LOG.info("saved %s", ARTIFACTS_DIR / "base_flow.pt")


if __name__ == "__main__":
    main()
