"""Stage B of the audio-loss flow spike: finetune the flow through the renderer.

Loads the Step B base flow, finetunes it with the flow-matching loss plus a
differentiable-render spectral term, and reports the common held-out protocol
before and after so the run is comparable to the #2553 control-field arms.

Run: ``uv run python -m prototypes.torchsynth_feedback.step_d_audio_loss``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from prototypes.torchsynth_feedback.audio_loss import FinetuneConfig, finetune_audio_loss
from prototypes.torchsynth_feedback.flow import build_base_flow
from prototypes.torchsynth_feedback.step_b_pretrain import ARTIFACTS_DIR, evaluate

_LOG = logging.getLogger(__name__)

BASE_FLOW_NAME = "base_flow.pt"
FINETUNED_NAME = "audio_loss_flow.pt"
METRICS_NAME = "audio_loss_metrics.json"


def run_stage_b(
    config: FinetuneConfig,
    artifacts_dir: Path,
    device: str = "cpu",
    *,
    eval_batch_size: int = 128,
    eval_batches: int = 4,
    eval_sample_steps: int = 50,
) -> dict[str, dict[str, float]]:
    """Finetune the saved base flow with the audio loss and score it either side.

    :param config: Optimization settings for the finetune.
    :param artifacts_dir: Directory holding ``base_flow.pt``; receives the outputs.
    :param device: Torch device string.
    :param eval_batch_size: Rows per held-out evaluation batch.
    :param eval_batches: Held-out batches to average over.
    :param eval_sample_steps: RK4 steps used to draw held-out predictions.
    :returns: Held-out metrics under the ``before`` and ``after`` keys.
    :raises FileNotFoundError: No base flow saved in ``artifacts_dir``.
    """
    checkpoint_path = artifacts_dir / BASE_FLOW_NAME
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"run step_b_pretrain first: {checkpoint_path} is missing")
    encoder, vector_field = build_base_flow(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    encoder.load_state_dict(checkpoint["encoder"])
    vector_field.load_state_dict(checkpoint["vector_field"])

    eval_kwargs = {
        "batch_size": eval_batch_size,
        "eval_batches": eval_batches,
        "sample_steps": eval_sample_steps,
    }
    before = evaluate(encoder, vector_field, device, label="before", **eval_kwargs)
    history = finetune_audio_loss(encoder, vector_field, device, config)
    after = evaluate(encoder, vector_field, device, label="after", **eval_kwargs)

    torch.save(
        {"encoder": encoder.state_dict(), "vector_field": vector_field.state_dict()},
        artifacts_dir / FINETUNED_NAME,
    )
    metrics = {"before": before, "after": after}
    (artifacts_dir / METRICS_NAME).write_text(
        json.dumps({**metrics, "final_train_loss": history[-1]}, indent=2)
    )
    _LOG.info("param_mse %.4f -> %.4f", before["param_mse"], after["param_mse"])
    _LOG.info("mslm %.4f -> %.4f", before["mslm"], after["mslm"])
    return metrics


def main() -> None:
    """Run stage B on the default artifacts directory."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    run_stage_b(FinetuneConfig(), ARTIFACTS_DIR, device)


if __name__ == "__main__":
    main()
