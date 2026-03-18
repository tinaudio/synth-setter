import os
from pathlib import Path
from typing import Literal

import click
import hydra
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.vst import param_specs
from src.models.components.transformer import LearntProjection
from src.utils import register_resolvers


def wandb_dir_to_ckpt_and_hparams(
    wandb_dir: Path, ckpt_type: Literal["best", "last"]
) -> tuple[Path, Path]:
    log_dir = wandb_dir.parent.parent
    ckpt_dir = log_dir / "checkpoints"

    csv_dir = log_dir / "csv"
    hparam_files = csv_dir.glob("*/hparams.yaml")
    hparam_file = max(hparam_files, key=lambda x: x.stat().st_mtime)

    if ckpt_type == "last":
        logger.info(f"Using last checkpoint for {log_dir}")
        ckpt_file = ckpt_dir / "last.ckpt"
    elif ckpt_type == "best":
        logger.info(f"Using best checkpoint for {log_dir}")
        ckpt_files = ckpt_dir.glob("epoch*.ckpt")

        # most recent file
        ckpt_files = sorted(ckpt_files, key=lambda x: x.stat().st_mtime, reverse=True)
        ckpt_file = ckpt_files[0]

    return ckpt_file, hparam_file


def get_state_dict(ckpt_file: Path, map_location: str = "cuda") -> dict:
    logger.info(f"Loading checkpoint from {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location=map_location, weights_only=False)
    state_dict = ckpt["state_dict"]
    return state_dict


def instantiate_model(
    model_cfg: DictConfig, ckpt_file: Path, map_location: str = "cuda"
) -> torch.nn.Module:

    logger.info(f"Instantiating model from {ckpt_file} with config:")
    logger.info(OmegaConf.to_yaml(model_cfg))
    model = hydra.utils.instantiate(model_cfg)

    logger.info("Model instantiated")
    model.to(device=map_location)

    state_dict = get_state_dict(ckpt_file, map_location=map_location)

    logger.info("Mapping state dict to params")
    model.setup(None)
    model.load_state_dict(state_dict, strict=False)

    return model


def instantiate_datamodule(data_cfg: DictConfig):
    logger.info("Instantiating datamodule with config:")
    logger.info(OmegaConf.to_yaml(data_cfg))
    dm = hydra.utils.instantiate(data_cfg)
    dm.setup("fit")

    return dm


def sort_assignment(assignment: np.ndarray):
    assignment = np.abs(assignment)
    idxs = np.argsort(assignment, axis=-1)
    idxs = np.lexsort(idxs.T)
    assignment = assignment[idxs]
    return assignment


def longest_matching_initial_substring(a: str, b: str) -> str:
    longest = ""
    for i in range(min(len(a), len(b))):
        if a[i] == b[i]:
            longest = longest + a[i]
        else:
            break

    return longest


def strip_scene_id(param_name: str) -> str:
    if param_name.startswith("a_"):
        return param_name[2:]

    return param_name


PREFIXES = (
    "amp_eg",
    "filter_eg",
    "filter_1",
    "filter_2",
    "waveshaper",
    "osc_1",
    "osc_2",
    "osc_3",
    "lfo_1",
    "lfo_2",
    "lfo_3",
    "lfo_4",
    "lfo_5",
    "lfo_6",
    "noise",
    "ring_modulation_1x2",
    "ring_modulation_2x3",
    "fx_a1",
    "fx_a2",
    "fx_a3",
    "fm",
)

RENAMES = {
    "amp_eg": "Amp. EG",
    "filter_eg": "Filt. EG",
    "feedback": "Feedback",
    "filter_balance": "Filt. Balance",
    "filter_configuration": "Filt. Routing",
    "highpass": "HPF",
    "filter_1": "Filter 1",
    "filter_2": "Filter 2",
    "waveshaper": "Waveshaper",
    "osc_1": "Osc. 1",
    "osc_2": "Osc. 2",
    "osc_3": "Osc. 3",
    "osc_drift": "Osc. Drift",
    "fm": "Freq. Mod.",
    "lfo_1": "LFO 1",
    "lfo_2": "LFO 2",
    "lfo_3": "LFO 3",
    "lfo_4": "LFO 4",
    "lfo_5": "LFO 5",
    "lfo_6": "Pitch EG",
    "noise": "Noise",
    "pan": "Pan",
    "ring_modulation_1x2": "Ring Mod. 1x2",
    "ring_modulation_2x3": "Ring Mod. 2x3",
    "vca_gain": "VCA Gain",
    "width": "Width",
    "fx_a1": "FX: Chorus",
    "fx_a2": "FX: Delay",
    "fx_a3": "FX: Reverb",
    "pitch": "Note Pitch",
    "note_start_and_end": "Note On/Off",
}


def kosc_intervals(spec: str):
    _, k = spec.rsplit("_", 1)
    k = int(k)

    return [("Frequency", k), ("Amplitude", k), ("Waveform", k)]


def get_labels(spec: str):
    if spec.startswith("k_"):
        return kosc_intervals(spec)

    param_spec = param_specs[spec]

    synth_intervals = [(p.name, len(p)) for p in param_spec.synth_params]
    note_intervals = [(p.name, len(p)) for p in param_spec.note_params]
    intervals = synth_intervals + note_intervals

    intervals = [(strip_scene_id(n), l) for n, l in intervals]
    true_intervals = []

    current_prefix = None
    current_prefix_length = 0

    for cur_name, cur_len in intervals:
        should_continue = False
        for prefix in PREFIXES:
            if cur_name.startswith(prefix):
                if prefix != current_prefix and current_prefix is not None:
                    true_intervals.append((current_prefix, current_prefix_length))

                    current_prefix = prefix
                    current_prefix_length = cur_len

                    should_continue = True
                    break
                if prefix == current_prefix:
                    current_prefix_length += cur_len

                    should_continue = True
                    break
                if current_prefix is None:
                    current_prefix = prefix
                    current_prefix_length = cur_len

                    should_continue = True
                    break

        if should_continue:
            continue

        if current_prefix is not None:
            true_intervals.append((current_prefix, current_prefix_length))

        current_prefix = None
        current_prefix_length = 0

        true_intervals.append((cur_name, cur_len))

    true_intervals = [(RENAMES.get(name, name), length) for name, length in true_intervals]

    return true_intervals


def add_labels(fig: plt.Figure, ax: plt.Axes, spec: str, axis: Literal["x", "y"] = "x"):
    if axis == "x":
        get_ticks = ax.get_xticks
        set_ticks = ax.set_xticks
        get_ticklabels = ax.get_xticklabels
        set_ticklabels = ax.set_xticklabels
        add_line = ax.axvline
    else:
        get_ticks = ax.get_yticks
        set_ticks = ax.set_yticks
        get_ticklabels = ax.get_yticklabels
        set_ticklabels = ax.set_yticklabels
        add_line = ax.axhline

    intervals = get_labels(spec)
    labels = [label for label, _ in intervals]

    centers = []
    start = 0
    for label, length in intervals:
        center = start + (length - 1) / 2
        centers.append(center)
        start += length

        add_line(start - 0.5, color="k", alpha=0.5)

    set_ticks(centers)
    set_ticklabels(labels)
    if axis == "x":
        plt.setp(get_ticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    else:
        plt.setp(get_ticklabels(), rotation=-45, ha="right", rotation_mode="anchor")

    # fig.canvas.draw()
    # renderer = fig.canvas.get_renderer()
    # text_objs = get_ticklabels()
    # bboxes = [txt.get_window_extent(renderer=renderer) for txt in text_objs]
    # current_shift = 0
    # last_end = -1e9 if axis == "x" else 1e9  # track right edge of the last label
    #
    # denom = math.cos(math.pi / 4)
    # min_perp_dist = 15
    #
    # for txt, bbox in zip(text_objs, bboxes):
    #     # if this bbox starts before the last one ends, we have an overlap
    #     if axis == "x":
    #         perp_dist = (bbox.x1 - last_end - current_shift) * denom
    #     else:
    #         perp_dist = ((last_end + current_shift) - bbox.y1) * denom
    #
    #     if perp_dist < min_perp_dist:
    #         shift = (min_perp_dist - perp_dist) / denom
    #         current_shift = shift
    #     else:
    #         # reset shift if no overlap
    #         current_shift = 0
    #     # move the text by modifying its 'y' position in data coordinates
    #     # You can also do this in axes or figure fraction coordinates if you prefer.
    #     x0, y0 = txt.get_position()
    #     x0, y0 = ax.transData.transform((x0, y0))
    #
    #     if axis == "x":
    #         y0 = y0 + current_shift / 100
    #     else:
    #         x0 = x0 - current_shift / 100
    #
    #     x0, y0 = ax.transData.inverted().transform((x0, y0))
    #
    #     txt.set_position((x0, y0))  # 72 points per inch
    #
    #     # fig.canvas.draw()
    #     # bbox = txt.get_window_extent(renderer=renderer)
    #
    #     last_end = bbox.x1 if axis == "x" else bbox.y1


def plot_assignment(proj: LearntProjection, spec: str):
    assignment = proj.assignment.detach().cpu().numpy()
    assignment = sort_assignment(assignment)

    plt.rcParams.update({"font.size": 14})

    size = 14
    ratio = assignment.shape[1] / assignment.shape[0]

    print(assignment.shape)
    if assignment.shape[1] > assignment.shape[0]:
        figsize = size * (ratio - 0.2), size
    else:
        figsize = size, size / (ratio + 0.1)
    print(figsize)

    fig = plt.figure(figsize=figsize, dpi=120)
    ax = fig.add_axes([0.1, 0.1, 0.75, 0.8])
    # fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=120)

    maxval = np.abs(assignment).max().item()
    img = ax.imshow(
        assignment,
        aspect="equal",
        vmin=-maxval,
        vmax=maxval,
        cmap="RdBu",
    )
    fig.colorbar(img, ax=ax, fraction=0.05, pad=0.05)

    # ax.set_title("Assignment")

    add_labels(fig, ax, spec)

    ax.set_ylabel("Tokens")
    fig.tight_layout()
    # fig.suptitle("Learnt Assignment")
    fig.tight_layout()

    return fig


def cosine_self_sim(x: np.ndarray) -> np.ndarray:
    dot_prod = np.einsum("ik,jk->ij", x, x)
    norm = np.einsum("ik,ik->i", x, x)
    return dot_prod / norm


def plot_embeds(proj: LearntProjection, spec: str):
    in_embed = proj.in_projection.detach().cpu().numpy()
    out_embed = proj.out_projection.detach().cpu().numpy()

    in_sim = cosine_self_sim(in_embed)
    out_sim = cosine_self_sim(out_embed.T)

    # cosine similarities

    fig = plt.figure(figsize=(32, 12), dpi=120)
    # fig, ax = plt.subplots(1, 2, figsize=(12, 8), dpi=120)
    ax = [
        fig.add_axes([0.15, 0.1, 0.33, 0.8]),
        fig.add_axes([0.63, 0.1, 0.33, 0.8]),
    ]

    in_img = ax[0].imshow(
        in_sim,
        aspect="equal",
        vmin=-1,
        vmax=1,
        cmap="RdBu",
    )
    out_img = ax[1].imshow(
        out_sim,
        aspect="equal",
        vmin=-1,
        vmax=1,
        cmap="RdBu",
    )

    add_labels(fig, ax[0], spec, "x")
    add_labels(fig, ax[0], spec, "y")
    add_labels(fig, ax[1], spec, "x")
    add_labels(fig, ax[1], spec, "y")

    fig.colorbar(out_img, ax=ax, location="right", fraction=0.05, pad=0.02)

    ax[0].set_title("In Embedding")
    ax[1].set_title("Out Embedding")

    fig.tight_layout()

    return fig


def plot_param2tok(proj: LearntProjection, out_dir: str, spec: str):
    logger.info("Plotting assignment")
    assignment_fig = plot_assignment(proj, spec)
    logger.info("Plotting done")
    logger.info("Plotting embeds")
    embed_fig = plot_embeds(proj, spec)
    logger.info("Plotting done")
    logger.info(f"Saving to {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    assignment_fig.savefig(f"{out_dir}/assignment.svg")
    embed_fig.savefig(f"{out_dir}/embeds.svg")
    logger.info("Saved")


@click.command()
@click.argument("wandb_id", type=str)
@click.argument("out_dir", type=str)
@click.option("--spec", "-s", type=str, default="surge_xt")
@click.option("--log-dir", "-l", type=str, default="logs")
@click.option("--ckpt_type", "-c", type=str, default="last")
@click.option("--device", "-d", type=str, default="cuda")
def main(
    wandb_id: str,
    out_dir: str,
    spec: str,
    log_dir: str = "logs",
    ckpt_type: Literal["best", "last"] = "last",
    device: str = "cuda",
):

    register_resolvers()
    log_dir = Path(log_dir)
    possible_wandb_dirs = list(log_dir.glob(f"**/*{wandb_id}/"))
    logger.info(f"Found {len(possible_wandb_dirs)} log dirs matching wandb id")

    ckpts_and_hparams = list(
        map(lambda x: wandb_dir_to_ckpt_and_hparams(x, ckpt_type), possible_wandb_dirs)
    )

    if len(ckpts_and_hparams) > 1:
        # take the one with the most recently updated hparam file
        ckpt_file, hparam_file = max(ckpts_and_hparams, key=lambda x: x[1].stat().st_mtime)
    elif len(ckpts_and_hparams) == 1:
        ckpt_file, hparam_file = ckpts_and_hparams[0]
    else:
        raise RuntimeError("Could not find wandb id in any of the log directories.")

    cfg = OmegaConf.load(hparam_file)

    model = instantiate_model(cfg.model, ckpt_file, device)
    torch.set_grad_enabled(False)

    plot_param2tok(model.vector_field.projection, out_dir, spec)


if __name__ == "__main__":
    main()
