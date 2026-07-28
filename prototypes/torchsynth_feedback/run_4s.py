"""Four-second audio-loss run: stage A pretrain, then one arm per audio-loss setting.

Uses the repo's :class:`LogMelEncoder` for conditioning — it computes log-mel from
raw waveforms inside the model, so the same differentiable front-end serves the
conditioning path and the latent-loss path with no librosa parity surface.

Run: ``python -m prototypes.torchsynth_feedback.run_4s``
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import torch
import wandb

from prototypes.torchsynth_feedback.audio_loss import (
    AudioDistance,
    FinetuneConfig,
    finetune_audio_loss,
)
from prototypes.torchsynth_feedback.flow import SAMPLE_RATE
from prototypes.torchsynth_feedback.step_b_pretrain import evaluate, pretrain_flow
from synth_setter.data.vst.torchsynth_param_spec import NUM_PARAMS
from synth_setter.models.components.residual_mlp import LogMelEncoder
from synth_setter.models.components.vector_field import VectorField

_LOG = logging.getLogger(__name__)

SIGNAL_LENGTH_4S = 176_400
CONDITIONING_DIM = 256
PRETRAIN_STEPS = 12_000
FINETUNE_STEPS = 2_000
BATCH_SIZE = 32
EVAL_KWARGS = {"batch_size": 128, "eval_batches": 1, "signal_length": SIGNAL_LENGTH_4S}
ARTIFACTS = Path(__file__).parent / "artifacts_4s"


def build_4s_flow(device: str) -> tuple[LogMelEncoder, VectorField]:
    """Build the log-mel conditioning encoder and vector field for 4 s audio.

    Mel settings mirror ``experiment/torchsynth/ffn.yaml`` so the front-end matches
    the established torchsynth recipe.

    :param device: Torch device string.
    :returns: Encoder and vector field on ``device``.
    """
    encoder = LogMelEncoder(
        in_dim=SIGNAL_LENGTH_4S,
        hidden_dim=16,
        out_dim=CONDITIONING_DIM,
        sample_rate=SAMPLE_RATE,
        center=True,
        f_min=0.0,
        f_max=None,
        n_fft=None,
        hop_length=None,
        n_mels=128,
        pad_mode="constant",
        power=2.0,
        mel_norm="slaney",
        mel_scale="slaney",
        window="hamming",
        amin=1.0e-10,
        top_db=80.0,
        num_blocks=4,
        kernel_size=7,
        norm="bn",
    ).to(device)
    vector_field = VectorField(
        field_dim=NUM_PARAMS, hidden_dim=256, conditioning_dim=CONDITIONING_DIM, num_blocks=4
    ).to(device)
    return encoder, vector_field


def main() -> None:
    """Pretrain at 4 s, then finetune one arm per audio-loss setting."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    ARTIFACTS.mkdir(exist_ok=True)

    run = wandb.init(
        project="synth-setter",
        group="torchsynth-audio-loss-4s",
        name="stage-a-4s",
        job_type="pretrain",
        config={
            "signal_length": SIGNAL_LENGTH_4S,
            "batch_size": BATCH_SIZE,
            "pretrain_steps": PRETRAIN_STEPS,
            "encoder": "LogMelEncoder",
        },
    )
    _LOG.info("W&B run: %s", run.url)

    encoder, vector_field = build_4s_flow(device)
    pretrain_flow(
        encoder,
        vector_field,
        device,
        steps=PRETRAIN_STEPS,
        batch_size=BATCH_SIZE,
        signal_length=SIGNAL_LENGTH_4S,
        on_step=lambda step, loss: run.log({"stage_a/loss": loss, "stage_a/step": step}),
    )
    base = evaluate(encoder, vector_field, device, label="stage_a", **EVAL_KWARGS)
    run.summary.update({f"stage_a/{k}": v for k, v in base.items()})
    torch.save(
        {"encoder": encoder.state_dict(), "vector_field": vector_field.state_dict()},
        ARTIFACTS / "base_flow_4s.pt",
    )
    run.finish()

    results = {"stage_a": base}
    arms = (
        ("lambda_0", 0.0, AudioDistance.MSLM),
        ("mslm_lambda_0p3", 0.3, AudioDistance.MSLM),
        ("latent_lambda_0p3", 0.3, AudioDistance.LATENT),
    )
    for label, lambda_audio, distance in arms:
        arm_run = wandb.init(
            project="synth-setter",
            group="torchsynth-audio-loss-4s",
            name=label,
            job_type="finetune",
            config={
                "lambda_audio": lambda_audio,
                "distance": str(distance),
                "signal_length": SIGNAL_LENGTH_4S,
                "steps": FINETUNE_STEPS,
            },
            reinit=True,
        )
        arm_encoder = copy.deepcopy(encoder)
        arm_field = copy.deepcopy(vector_field)
        history = finetune_audio_loss(
            arm_encoder,
            arm_field,
            device,
            FinetuneConfig(
                steps=FINETUNE_STEPS,
                batch_size=BATCH_SIZE,
                lambda_audio=lambda_audio,
                distance=distance,
                signal_length=SIGNAL_LENGTH_4S,
            ),
            on_step=lambda step, metrics: arm_run.log(
                {f"stage_b/{k}": v for k, v in metrics.items()} | {"stage_b/step": step}
            ),
        )
        metrics = evaluate(arm_encoder, arm_field, device, label=label, **EVAL_KWARGS)
        metrics["final_loss"] = history[-1]["loss"]
        arm_run.summary.update(metrics)
        arm_run.finish()
        results[label] = metrics

    (ARTIFACTS / "results_4s.json").write_text(json.dumps(results, indent=2))
    _LOG.info("DONE\n%s", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
