"""Render predicted-parameter and target audio from a trained model for offline evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import click
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils
import torch
from tqdm import tqdm, trange

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from synth_setter.data.vst import param_specs
from synth_setter.data.vst.core import render_params, write_wav
from synth_setter.data.vst.param_spec import ParamSpec

_MEL_HOP_LENGTH = 512
_MEL_N_FFT = 2048
_MEL_N_MELS = 128


def make_spectrogram(audio: np.ndarray, sample_rate: float) -> list[np.ndarray]:
    """Return one log-mel spectrogram per channel of ``audio``.

    :param audio: ``(channels, samples)`` waveform.
    :param sample_rate: Audio sample rate in Hz.
    :returns: One ``(n_mels, frames)`` log-mel spectrogram per channel.
    :rtype: list[np.ndarray]
    """
    specs = []
    for channel_audio in audio:
        power = librosa.feature.melspectrogram(
            y=channel_audio,
            sr=sample_rate,
            n_mels=_MEL_N_MELS,
            n_fft=_MEL_N_FFT,
            hop_length=_MEL_HOP_LENGTH,
            window="hamming",
        )
        specs.append(librosa.power_to_db(power, ref=np.max))
    return specs


def write_spectrograms(
    pred_audio: np.ndarray,
    target_audio: np.ndarray,
    sample_rate: float,
    save_path: str,
) -> None:
    """Plot stacked pred + target mel spectrograms to ``save_path``.

    :param pred_audio: ``(channels, samples)`` predicted-parameter render.
    :param target_audio: ``(channels, samples)`` ground-truth render.
    :param sample_rate: Audio sample rate in Hz.
    :param save_path: Path to write the PNG figure to.
    """
    pred_specs = make_spectrogram(pred_audio, sample_rate)
    target_specs = make_spectrogram(target_audio, sample_rate)

    panels = [(spec, f"Pred (Chan {i + 1})") for i, spec in enumerate(pred_specs)]
    panels += [(spec, f"Target (Chan {i + 1})") for i, spec in enumerate(target_specs)]

    fig, axs = plt.subplots(len(panels), 1, figsize=(8, 3 * len(panels)))
    # subplots(1, 1) returns a single Axes, not an array — normalize so the zip below works.
    axs = np.atleast_1d(axs)

    for ax, (spec, title) in zip(axs, panels):
        # spec is already log-mel dB from make_spectrogram's power_to_db.
        # Passing through amplitude_to_db a second time double-converts.
        librosa.display.specshow(
            spec,
            sr=sample_rate,
            hop_length=_MEL_HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
            ax=ax,
            cmap="magma",
            vmax=0,
        )
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def params_to_csv(
    target_synth_params: dict[str, float] | None,
    target_note_params: dict[str, Any] | None,
    pred_synth_params: dict[str, float],
    pred_note_params: dict[str, Any],
    save_path: str,
) -> None:
    """Write the target and predicted parameters to a CSV file.

    :param target_synth_params: Target synth params, or ``None`` if unavailable.
    :param target_note_params: Target note params (e.g. ``pitch: int``,
        ``note_start_and_end: tuple[float, float]``) or ``None`` if unavailable.
    :param pred_synth_params: Predicted synth params.
    :param pred_note_params: Predicted note params (mixed-value types — see
        ``target_note_params``).
    :param save_path: Path to write the CSV to.
    """
    synth_df = pd.DataFrame({"pred": pred_synth_params, "target": target_synth_params})
    note_df = pd.DataFrame({"pred": pred_note_params, "target": target_note_params})
    df = pd.concat([synth_df, note_df])
    df.to_csv(save_path)


def _decode_row(
    row_params: np.ndarray, param_spec: ParamSpec
) -> tuple[dict[str, float], dict[str, Any]]:
    """Map a model-output row in ``[-1, 1]`` to ``(synth_params, note_params)``.

    :param row_params: One row of model output, values in ``[-1, 1]``.
    :param param_spec: Spec used to decode the row into named param dicts.
    :returns: ``(synth_params, note_params)`` dicts as returned by ``param_spec.decode``.
    :rtype: tuple[dict[str, float], dict[str, Any]]
    """
    scaled = np.clip((row_params + 1) / 2, 0, 1)
    return param_spec.decode(scaled)


def _render(
    plugin_path: str,
    preset_path: str,
    synth_params: dict[str, float],
    note_params: dict[str, Any],
    velocity: int,
    signal_duration_seconds: float,
    sample_rate: float,
    channels: int,
) -> np.ndarray:
    """Render a single VST sample given decoded synth + note params.

    :param plugin_path: Path to the VST3 plugin.
    :param preset_path: Path to the VST preset to load before rendering.
    :param synth_params: Decoded synth-side parameter dict.
    :param note_params: Decoded note-side dict (must contain ``pitch`` and
        ``note_start_and_end``).
    :param velocity: MIDI note-on velocity (0-127).
    :param signal_duration_seconds: Total render duration in seconds.
    :param sample_rate: Audio sample rate in Hz.
    :param channels: Number of output channels.
    :returns: Rendered ``(channels, samples)`` audio.
    :rtype: np.ndarray
    """
    return render_params(
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


def _list_pred_files(pred_dir: Path) -> tuple[list[Path], list[int]]:
    """Return ``(pred_files, indices)`` sorted by the index suffix in ``pred-{i}.pt``.

    :param pred_dir: Directory containing ``pred-{i}.pt`` files.
    :returns: ``(pred_files, indices)`` — both ordered by ascending index.
    :rtype: tuple[list[Path], list[int]]
    """
    indexed = sorted(
        ((int(f.stem.split("-")[1]), f) for f in pred_dir.glob("pred-*.pt") if f.is_file()),
        key=lambda pair: pair[0],
    )
    indices = [i for i, _ in indexed]
    pred_files = [f for _, f in indexed]
    return pred_files, indices


@click.command()
@click.argument("pred_dir", type=str)
@click.argument("output_dir", type=str)
@click.option("--plugin_path", "-p", type=str, default="plugins/Surge XT.vst3")
@click.option("--preset_path", "-r", type=str, default="presets/surge-base.vstpreset")
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
    plugin_path: str = "plugins/Surge XT.vst3",
    preset_path: str = "presets/surge-base.vstpreset",
    sample_rate: float = 44100.0,
    channels: int = 2,
    velocity: int = 100,
    signal_duration_seconds: float = 4.0,
    param_spec: str = "surge_xt",
    rerender_target: bool = False,
    no_params: bool = False,
    skip_spectrogram: bool = False,
) -> None:
    spec = param_specs[param_spec]
    os.makedirs(output_dir, exist_ok=True)

    pred_path = Path(pred_dir)
    pred_files, indices = _list_pred_files(pred_path)
    target_audio_files = [pred_path / f"target-audio-{i}.pt" for i in indices]
    target_param_files: list[Path | None] = (
        [None] * len(pred_files)
        if no_params
        else [pred_path / f"target-params-{i}.pt" for i in indices]
    )

    current_offset = 0
    for pred_file, target_param_file, target_audio_file in tqdm(
        zip(pred_files, target_param_files, target_audio_files),
        total=len(pred_files),
    ):
        pred_params = torch.load(pred_file, map_location="cpu")
        target_audio = torch.load(target_audio_file, map_location="cpu").numpy()
        target_params = (
            None
            if target_param_file is None
            else torch.load(target_param_file, map_location="cpu")
        )

        for j in trange(pred_params.shape[0]):
            sample_dir = os.path.join(output_dir, f"sample_{current_offset + j}")
            os.makedirs(sample_dir, exist_ok=True)

            pred_synth, pred_note = _decode_row(pred_params[j].float().numpy(), spec)
            pred_audio = _render(
                plugin_path,
                preset_path,
                pred_synth,
                pred_note,
                velocity,
                signal_duration_seconds,
                sample_rate,
                channels,
            )

            if target_params is None:
                target_synth, target_note = None, None
            else:
                target_synth, target_note = _decode_row(target_params[j].numpy(), spec)

            out_target = os.path.join(sample_dir, "target.wav")
            if rerender_target and target_synth is not None:
                rendered_target = _render(
                    plugin_path,
                    preset_path,
                    target_synth,
                    target_note,
                    velocity,
                    signal_duration_seconds,
                    sample_rate,
                    channels,
                )
                write_wav(rendered_target, out_target, sample_rate, channels)
            else:
                write_wav(target_audio[j], out_target, sample_rate, channels)

            write_wav(pred_audio, os.path.join(sample_dir, "pred.wav"), sample_rate, channels)

            if not skip_spectrogram:
                write_spectrograms(
                    pred_audio,
                    target_audio[j],
                    sample_rate,
                    os.path.join(sample_dir, "spec.png"),
                )

            params_to_csv(
                target_synth,
                target_note,
                pred_synth,
                pred_note,
                os.path.join(sample_dir, "params.csv"),
            )

        current_offset += pred_params.shape[0]


if __name__ == "__main__":
    main()
