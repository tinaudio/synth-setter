"""Paired held-out eval for the decisive simulator-feedback finetune (#2554 follow-up).

Samples the frozen base flow and the finetuned corrected flow on the same rows
with identical seeded noise, renders both predictions plus the target through
surgepy, and writes per-row param MSE + canonical audio metrics (mss, wmfcc,
sot, rms) to CSV. Deterministic noise + ``shuffle=False`` val rows make rows
pairable across separately-launched arms.

Run inside the training job's Hydra run dir so the hydrated dataset is reused::

    python -m synth_setter.evaluation.eval_feedback_finetune \
        --experiment surge/flow_simple_ft_fb \
        --ckpt train-run/checkpoints/last.ckpt \
        --work-dir train-run --out train-run/eval_feedback.csv
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch

from synth_setter.data.vst.param_spec import decode_model_output
from synth_setter.data.vst.renderers import AudioRenderer
from synth_setter.data.vst_datamodule import load_dataset_statistics
from synth_setter.evaluation.compute_audio_metrics import (
    compute_mss,
    compute_rms,
    compute_sot,
    compute_wmfcc,
)
from synth_setter.models.vst_flow_finetune_module import VSTFlowMatchingFinetuneModule
from synth_setter.renderer_factory import make_audio_renderer
from synth_setter.utils import register_resolvers

log = logging.getLogger(__name__)

_AUDIO_METRICS = {
    "mss": compute_mss,
    "wmfcc": compute_wmfcc,
    "sot": compute_sot,
    "rms": compute_rms,
}


def paired_delta(arm: np.ndarray, base: np.ndarray) -> tuple[float, float]:
    """Return the mean and standard error of the per-row difference ``arm - base``.

    :param arm: Per-row metric values for the treatment arm.
    :param base: Per-row metric values for the comparison arm, same length.
    :returns: ``(mean_delta, sem)`` where SEM uses the sample (ddof=1) std.
    :raises ValueError: If the two arrays cannot be paired row-for-row.
    """
    if arm.shape != base.shape:
        raise ValueError(f"arrays must be paired row-for-row; got {arm.shape} vs {base.shape}")
    delta = arm - base
    sem = float(delta.std(ddof=1) / np.sqrt(delta.size)) if delta.size > 1 else 0.0
    return float(delta.mean()), sem


def _compose_cfg(experiment: str, work_dir: Path):
    """Compose the train config for ``experiment`` with paths pinned to ``work_dir``.

    :param experiment: Hydra experiment selector (e.g. ``surge/flow_simple_ft_fb``).
    :param work_dir: Directory whose ``data/`` subdir holds the hydrated dataset.
    :returns: The composed ``DictConfig``.
    """
    from hydra import compose, initialize_config_module

    register_resolvers()
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        return compose(
            config_name="train.yaml",
            overrides=[
                f"experiment={experiment}",
                f"paths.output_dir={work_dir}",
                f"paths.work_dir={work_dir}",
                "logger=csv",
            ],
        )


def _render_row(
    module: VSTFlowMatchingFinetuneModule, renderer: AudioRenderer, row: np.ndarray
) -> np.ndarray:
    """Decode one model-output row and render it through the shared renderer.

    :param module: Finetune module supplying the param spec and note sanitizer.
    :param renderer: Audio renderer shared across all rows.
    :param row: Encoded parameter row in ``[-1, 1]``.
    :returns: Rendered audio shaped ``(channels, samples)``.
    """
    synth_params, note_params = decode_model_output(row, module._param_spec)
    start, end = module._sanitized_note_window(*note_params["note_start_and_end"])
    audio = renderer.render(
        synth_params, int(note_params["pitch"]), module.hparams["velocity"], (start, end)
    )
    return np.asarray(audio, dtype=np.float32)


@click.command()
@click.option("--experiment", required=True, help="Hydra experiment selector for the arm.")
@click.option("--ckpt", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--work-dir", required=True, type=click.Path(path_type=Path))
@click.option("--out", required=True, type=click.Path(path_type=Path))
@click.option("--num-rows", default=512, show_default=True)
@click.option("--steps", default=100, show_default=True, help="Euler ODE steps per sample.")
@click.option("--noise-seed", default=20260728, show_default=True)
def main(
    experiment: str,
    ckpt: Path,
    work_dir: Path,
    out: Path,
    num_rows: int,
    steps: int,
    noise_seed: int,
) -> None:
    """Run the paired base-vs-arm eval and write per-row metrics to ``--out``.

    :param experiment: Hydra experiment selector for the arm under eval.
    :param ckpt: Finetuned Lightning checkpoint to load.
    :param work_dir: Training run dir whose ``data/`` holds the hydrated dataset.
    :param out: Destination CSV path for the per-row metrics.
    :param num_rows: Held-out val rows to evaluate.
    :param steps: Euler ODE steps per sample.
    :param noise_seed: Seed for the shared per-row initial noise.
    """
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("synth_setter.evaluation.compute_audio_metrics").setLevel(logging.WARNING)
    cfg = _compose_cfg(experiment, work_dir)

    from hydra.utils import instantiate

    datamodule = instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup("fit")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = VSTFlowMatchingFinetuneModule.load_from_checkpoint(
        str(ckpt), map_location=device, weights_only=False
    )
    module.eval().to(device)
    # Wire the render/rep runtime explicitly: there is no trainer here, and the
    # ablation arm never builds a renderer on its own.
    renderer = make_audio_renderer(module._render_config())
    module._renderer = renderer
    if datamodule.use_saved_mean_and_variance:
        module._mel_stats = load_dataset_statistics(Path(datamodule.dataset_root) / "train.lance")
    module._runtime_ready = True

    num_params = int(module.base.hparams["num_params"])
    generator = torch.Generator().manual_seed(noise_seed)
    noise_all = torch.randn(num_rows, num_params, generator=generator)

    records: list[dict[str, float]] = []
    started = time.perf_counter()
    row_idx = 0
    for batch in datamodule.val_dataloader():
        if row_idx >= num_rows:
            break
        mel = batch["mel_spec"].to(device)
        params = batch["params"]
        take = min(mel.shape[0], num_rows - row_idx)
        noise = noise_all[row_idx : row_idx + take].to(device)
        with torch.no_grad():
            pred_base = module.sample_with_feedback(mel[:take], noise, steps, apply_control=False)
            pred_arm = module.sample_with_feedback(mel[:take], noise, steps)
        for i in range(take):
            target_audio = _render_row(module, renderer, params[i].numpy())
            base_audio = _render_row(module, renderer, pred_base[i].cpu().numpy())
            arm_audio = _render_row(module, renderer, pred_arm[i].cpu().numpy())
            record: dict[str, float] = {
                "row": float(row_idx + i),
                "param_mse_base": float((pred_base[i].cpu() - params[i]).square().mean()),
                "param_mse_arm": float((pred_arm[i].cpu() - params[i]).square().mean()),
            }
            for name, fn in _AUDIO_METRICS.items():
                record[f"{name}_base"] = float(fn(target_audio, base_audio))
                record[f"{name}_arm"] = float(fn(target_audio, arm_audio))
            records.append(record)
        row_idx += take
        log.info(
            "evaluated %d/%d rows (%.1fs elapsed)",
            row_idx,
            num_rows,
            time.perf_counter() - started,
        )

    frame = pd.DataFrame.from_records(records)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    log.info("wrote %d rows to %s", len(frame), out)

    for name in ("param_mse", *_AUDIO_METRICS):
        arm = frame[f"{name}_arm"].to_numpy()
        base = frame[f"{name}_base"].to_numpy()
        mean, sem = paired_delta(arm, base)
        log.info(
            "%s: base %.4f | arm %.4f | arm-base %+.4f +/- %.4f",
            name,
            base.mean(),
            arm.mean(),
            mean,
            sem,
        )


if __name__ == "__main__":
    main()
