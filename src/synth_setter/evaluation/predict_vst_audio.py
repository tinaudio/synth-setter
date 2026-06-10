"""Render predicted-parameter and target audio from a trained model for offline evaluation."""

import os
from pathlib import Path

import click
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pedalboard.io import AudioFile
from tqdm import tqdm, trange

from synth_setter.data.vst import param_specs
from synth_setter.data.vst.core import render_params
from synth_setter.data.vst.param_spec import NoteParams, ParamSpec
from synth_setter.data.vst.param_spec_registry import default_plugin_path, preset_paths


def make_spectrogram(audio: np.ndarray, sample_rate: float) -> np.ndarray:
    channels = audio.shape[0]

    specs = []
    for channel in range(channels):
        spec = librosa.feature.melspectrogram(
            y=audio[channel],
            sr=sample_rate,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            window="hamming",
        )
        spec_db = librosa.power_to_db(spec, ref=np.max)
        specs.append(spec_db)

    return specs


def write_spectrograms(
    pred_audio: np.ndarray,
    target_audio: np.ndarray,
    sample_rate: float,
    save_path: str,
) -> np.ndarray:
    pred_specs = make_spectrogram(pred_audio, sample_rate)
    target_specs = make_spectrogram(target_audio, sample_rate)

    channels = len(pred_specs) + len(target_specs)

    fig, axs = plt.subplots(channels, 1, figsize=(8, 3 * channels))

    for i, spec in enumerate(pred_specs):
        spec = librosa.amplitude_to_db(spec, ref=np.max)
        librosa.display.specshow(
            spec,
            sr=sample_rate,
            hop_length=512,
            x_axis="time",
            y_axis="mel",
            ax=axs[i],
            cmap="magma",
        )
        axs[i].set_title(f"Pred (Chan {i + 1})")

    for i, spec in enumerate(target_specs):
        spec = librosa.amplitude_to_db(spec, ref=np.max)
        librosa.display.specshow(
            spec,
            sr=sample_rate,
            hop_length=512,
            x_axis="time",
            y_axis="mel",
            ax=axs[i + len(pred_specs)],
            cmap="magma",
        )
        axs[i + len(pred_specs)].set_title(f"Target (Chan {i + 1})")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def params_to_csv(
    target_synth_params: dict[str, float] | None,
    target_note_params: NoteParams | None,
    pred_synth_params: dict[str, float],
    pred_note_params: NoteParams,
    save_path: str,
    param_spec: ParamSpec,
) -> None:
    """Write the target and predicted parameters to a CSV file."""
    row_names = list(pred_synth_params.keys()) + list(pred_note_params.keys())

    synth_df = pd.DataFrame({"pred": pred_synth_params, "target": target_synth_params})
    note_df = pd.DataFrame({"pred": pred_note_params, "target": target_note_params})
    df = pd.concat([synth_df, note_df])

    df.to_csv(save_path)


def resolve_preset_path(preset_path: str | None, param_spec: str) -> str:
    """Return ``preset_path`` when given, else the registry's default preset for ``param_spec``.

    ``None`` with an unregistered ``param_spec`` propagates the registry ``KeyError``.

    :param preset_path: Explicit preset path; ``None`` selects the registry default.
    :param param_spec: Registry key naming the spec whose default preset to use.
    :returns: Resolved preset path.
    """
    return preset_paths[param_spec] if preset_path is None else preset_path


@click.command()
@click.argument("pred_dir", type=str)
@click.argument("output_dir", type=str)
@click.option("--plugin_path", "-p", type=str, default=default_plugin_path)
@click.option("--preset_path", "-r", type=str, default=None)
@click.option("--sample_rate", "-s", type=float, default=44100.0)
@click.option("--channels", "-c", type=int, default=2)
@click.option("--velocity", "-v", type=int, default=100)
@click.option("--signal_duration_seconds", "-d", type=float, default=4.0)
@click.option("--param_spec", type=str, default="surge_xt")
@click.option("--rerender_target", "-t", is_flag=True, default=False)
@click.option("--no-params", "-X", is_flag=True, default=False)
@click.option("--skip-spectrogram", "-S", is_flag=True, default=False)
def main(
    pred_dir: str,
    output_dir: str,
    plugin_path: str,
    preset_path: str | None = None,
    sample_rate: float = 44100.0,
    channels: int = 2,
    velocity: int = 100,
    signal_duration_seconds: float = 4.0,
    param_spec: str = "surge_xt",
    rerender_target: bool = False,
    no_params: bool = False,
    skip_spectrogram: bool = False,
):
    preset_path = resolve_preset_path(preset_path, param_spec)
    spec = param_specs[param_spec]
    os.makedirs(output_dir, exist_ok=True)

    # render_params loads the plugin (and applies preset_path) on every call,
    # so no upfront load_plugin is needed here.

    # list the .pt files with accompanying indices (each file has name
    # pred-{index}.pt, and we want to sort by index)
    pred_path = Path(pred_dir)
    pred_files = [f for f in pred_path.glob("pred-*.pt") if f.is_file()]
    indices = [int(f.stem.split("-")[1]) for f in pred_files]
    target_audio_files = [pred_path / f"target-audio-{i}.pt" for i in indices]

    if no_params:
        target_param_files = [None] * len(pred_files)
    else:
        target_param_files = [pred_path / f"target-params-{i}.pt" for i in indices]

    # 4. foreach .pt file
    current_offset = 0
    for i, (pred_file, target_param_file, target_audio_file) in tqdm(
        enumerate(zip(pred_files, target_param_files, target_audio_files))
    ):
        pred_params = torch.load(pred_file, map_location="cpu")
        target_audio = torch.load(target_audio_file, map_location="cpu").numpy()

        if target_param_file is None:
            target_params = None
        else:
            target_params = torch.load(target_param_file, map_location="cpu")

        # 5. iterate over its internal rows and render the audio
        for j in trange(pred_params.shape[0]):
            file_idx = current_offset + j
            sample_dir = os.path.join(output_dir, f"sample_{file_idx}")
            os.makedirs(sample_dir, exist_ok=True)

            row_params = pred_params[j].float().numpy()
            row_params_scaled = (row_params + 1) / 2
            row_params_scaled = np.clip(row_params_scaled, 0, 1)
            synth_params, note_params = spec.decode(row_params_scaled)

            pred_audio = render_params(
                plugin_path,
                synth_params,
                int(note_params["pitch"]),
                velocity,
                note_params["note_start_and_end"],
                signal_duration_seconds,
                sample_rate,
                channels,
                preset_path=preset_path,
            )

            target_synth_params: dict[str, float] | None = None
            target_note_params: NoteParams | None = None

            out_target = os.path.join(sample_dir, "target.wav")
            if rerender_target and target_params is not None:
                target_params_ = target_params[j].numpy()
                target_params_ = (target_params_ + 1) / 2
                target_params_ = np.clip(target_params_, 0, 1)
                target_synth_params, target_note_params = spec.decode(target_params_)

                new_target = render_params(
                    plugin_path,
                    target_synth_params,
                    int(target_note_params["pitch"]),
                    velocity,
                    target_note_params["note_start_and_end"],
                    signal_duration_seconds,
                    sample_rate,
                    channels,
                    preset_path=preset_path,
                )
                with AudioFile(out_target, "w", sample_rate, channels) as f:
                    f.write(new_target.T)

            else:
                with AudioFile(out_target, "w", sample_rate, channels) as f:
                    f.write(target_audio[j].T)

            out_pred = os.path.join(sample_dir, "pred.wav")
            with AudioFile(out_pred, "w", sample_rate, channels) as f:
                f.write(pred_audio.T)

            if not skip_spectrogram:
                write_spectrograms(
                    pred_audio,
                    target_audio[j],
                    sample_rate,
                    os.path.join(sample_dir, "spec.png"),
                )

            params_to_csv(
                target_synth_params if target_params is not None else None,
                target_note_params if target_params is not None else None,
                synth_params,
                note_params,
                os.path.join(sample_dir, "params.csv"),
                spec,
            )

        current_offset += pred_params.shape[0]


if __name__ == "__main__":
    main()
