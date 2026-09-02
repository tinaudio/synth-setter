"""Render predicted-parameter and target audio from a trained model for offline evaluation."""

import os
from collections.abc import Callable
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pedalboard.io import AudioFile
from pydantic_settings import BaseSettings, CliApp, CliPositionalArg, SettingsConfigDict
from tqdm import tqdm, trange

from synth_setter.data.vst import param_specs
from synth_setter.data.vst.core import run_with_editor_held_open
from synth_setter.data.vst.param_spec import (
    NoteParams,
    ParamSpec,
    decode_model_output,
    require_note_params,
    require_scalar_synth_params,
)
from synth_setter.data.vst.renderers import AudioRenderer, PedalboardRenderer
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.renderer_factory import make_audio_renderer

RenderFn = Callable[[dict[str, float], int, tuple[float, float]], np.ndarray]


class _PredictAudioCliArgs(RenderConfig, BaseSettings):
    """Render configuration plus prediction-artifact CLI inputs.

    .. attribute :: model_config

        Pydantic settings and CLI parsing policy.

    .. attribute :: pred_dir

        Directory containing prediction tensors.

    .. attribute :: output_dir

        Destination for rendered artifacts.

    .. attribute :: rerender_target

        Whether to render target parameters instead of using staged audio.

    .. attribute :: no_params

        Whether staged target parameters are absent.

    .. attribute :: skip_spectrogram

        Whether to omit spectrogram artifacts.
    """

    model_config = SettingsConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        cli_kebab_case=True,
    )

    pred_dir: CliPositionalArg[Path]
    output_dir: CliPositionalArg[Path]
    rerender_target: bool = False
    no_params: bool = False
    skip_spectrogram: bool = False


def make_spectrogram(audio: np.ndarray, sample_rate: float) -> list[np.ndarray]:
    """Compute one dB-scaled mel spectrogram per channel.

    :param audio: Waveform of shape ``(channels, samples)``.
    :param sample_rate: Sample rate in Hz.
    :returns: One ``(n_mels, frames)`` dB-scaled array per channel.
    """
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
    *,
    pred_effective_note_window: tuple[float, float],
) -> None:
    """Write raw, target, and effective rendered parameters to a CSV file.

    :param target_synth_params: Target synth values, or ``None`` when absent.
    :param target_note_params: Target note values, or ``None`` when absent.
    :param pred_synth_params: Raw decoded prediction synth values.
    :param pred_note_params: Raw decoded prediction note values.
    :param save_path: Destination CSV path.
    :param param_spec: Parameter ordering contract for the rendered synth.
    :param pred_effective_note_window: Note window used to render ``pred.wav``.
    """
    row_names = list(pred_synth_params.keys()) + list(pred_note_params.keys())

    synth_df = pd.DataFrame({"pred": pred_synth_params, "target": target_synth_params})
    note_df = pd.DataFrame({"pred": pred_note_params, "target": target_note_params})
    df = pd.concat([synth_df, note_df])
    df["pred_effective"] = df["pred"]
    df.at["note_start_and_end", "pred_effective"] = pred_effective_note_window

    df.to_csv(save_path)


def _canonicalize_prediction_note_window(
    note_window: tuple[float, float],
    *,
    signal_duration_seconds: float,
    sample_rate: int,
) -> tuple[float, float]:
    """Return a finite chronological prediction window accepted by renderers.

    :param note_window: Model-predicted note endpoints in seconds.
    :param signal_duration_seconds: Maximum renderable endpoint in seconds.
    :param sample_rate: Render sample rate in Hz.
    :returns: Clipped chronological endpoints separated by at least one available sample.
    :raises ValueError: Either predicted endpoint is non-finite.
    """
    start, end = sorted(float(value) for value in note_window)
    if not np.isfinite([start, end]).all():
        raise ValueError(f"predicted note window must be finite, got {note_window!r}")

    start = min(max(start, 0.0), signal_duration_seconds)
    end = min(max(end, 0.0), signal_duration_seconds)
    minimum_duration = min(1.0 / sample_rate, signal_duration_seconds)
    if end - start >= minimum_duration:
        return start, end
    if start + minimum_duration <= signal_duration_seconds:
        return start, start + minimum_duration
    return signal_duration_seconds - minimum_duration, signal_duration_seconds


def _make_render_fn(args: _PredictAudioCliArgs, renderer: AudioRenderer) -> RenderFn:
    """Apply capture-time GUI warm-up cadence to one renderer session.

    :param args: Validated renderer lifecycle configuration.
    :param renderer: Renderer session used for every prediction and target row.
    :returns: Row renderer honoring ``gui_toggle_cadence``.
    """
    warmup_pending = args.gui_toggle_cadence == "once"

    def render(
        synth_params: dict[str, float],
        pitch: int,
        note_start_and_end: tuple[float, float],
    ) -> np.ndarray:
        nonlocal warmup_pending
        warmup = args.gui_toggle_cadence == "render" or warmup_pending
        audio = renderer.render(
            synth_params,
            pitch,
            args.velocity,
            note_start_and_end,
            warmup=warmup,
        )
        warmup_pending = False
        return audio

    return render


def _render_prediction_artifacts(
    args: _PredictAudioCliArgs,
    spec: ParamSpec,
    render: RenderFn,
) -> None:
    """Write prediction and target artifacts for every staged tensor row.

    :param args: Validated artifact paths and output options.
    :param spec: Parameter decoder for each prediction row.
    :param render: Renderer call carrying the configured GUI cadence.
    :raises ValueError: A sample has neither staged nor re-renderable target audio.
    """
    sample_rate = args.sample_rate
    channels = args.channels

    pred_dir = str(args.pred_dir)
    output_dir = str(args.output_dir)
    rerender_target = args.rerender_target
    no_params = args.no_params
    skip_spectrogram = args.skip_spectrogram

    # Glob order defines output numbering; numeric batch ordering is tracked in #2446.
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

        # Absent with rerender_target is a supported layout: ValAudioProbe stages
        # only pred + target-params, because training val batches carry no raw audio.
        if target_audio_file.is_file():
            target_audio = torch.load(target_audio_file, map_location="cpu").numpy()
        else:
            target_audio = None

        if target_param_file is None:
            target_params = None
        else:
            target_params = torch.load(target_param_file, map_location="cpu")

        if target_audio is None and not (rerender_target and target_params is not None):
            raise ValueError(
                f"{target_audio_file} is missing and --rerender-target is off (or "
                "target params are absent): there is no target audio source. Stage "
                "target-audio tensors or pass --rerender-target with target params."
            )

        # 5. iterate over its internal rows and render the audio
        for j in trange(pred_params.shape[0]):
            file_idx = current_offset + j
            sample_dir = os.path.join(output_dir, f"sample_{file_idx}")
            os.makedirs(sample_dir, exist_ok=True)

            row_params = pred_params[j].float().numpy()
            synth_values, note_values = decode_model_output(row_params, spec)
            synth_params = require_scalar_synth_params(synth_values)
            note_params = require_note_params(note_values)
            note_params["note_start_and_end"] = tuple(
                float(value) for value in note_params["note_start_and_end"]
            )
            render_note_window = _canonicalize_prediction_note_window(
                note_params["note_start_and_end"],
                signal_duration_seconds=args.signal_duration_seconds,
                sample_rate=args.sample_rate,
            )
            pred_audio = render(
                synth_params,
                int(note_params["pitch"]),
                render_note_window,
            )

            target_synth_params: dict[str, float] | None = None
            target_note_params: NoteParams | None = None
            # Dataset audio when staged; the rerender branch fills it only when absent,
            # so a staged tensor keeps the spectrogram on dataset audio.
            target_for_spec = target_audio[j] if target_audio is not None else None

            out_target = os.path.join(sample_dir, "target.wav")
            if rerender_target and target_params is not None:
                # .float() aligns the target path with the pred path's float32 contract.
                target_params_ = target_params[j].float().numpy()
                target_synth_values, target_note_values = decode_model_output(target_params_, spec)
                target_synth_params = require_scalar_synth_params(target_synth_values)
                target_note_params = require_note_params(target_note_values)

                new_target = render(
                    target_synth_params,
                    int(target_note_params["pitch"]),
                    target_note_params["note_start_and_end"],
                )
                with AudioFile(out_target, "w", sample_rate, channels) as f:
                    f.write(new_target.T)
                if target_for_spec is None:
                    target_for_spec = new_target

            else:
                with AudioFile(out_target, "w", sample_rate, channels) as f:
                    f.write(target_audio[j].T)

            out_pred = os.path.join(sample_dir, "pred.wav")
            with AudioFile(out_pred, "w", sample_rate, channels) as f:
                f.write(pred_audio.T)

            if not skip_spectrogram:
                write_spectrograms(
                    pred_audio,
                    target_for_spec,
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
                pred_effective_note_window=render_note_window,
            )

        current_offset += pred_params.shape[0]


def render_prediction_audio(args: _PredictAudioCliArgs) -> None:
    """Render prediction artifacts through the configured production backend.

    :param args: Validated render configuration and artifact paths.
    :raises RuntimeError: An always-on GUI config lacks a cached Pedalboard plugin.
    """
    spec = param_specs[args.param_spec_name]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    renderer = make_audio_renderer(args)
    render = _make_render_fn(args, renderer)

    if args.gui_toggle_cadence != "always_on":
        _render_prediction_artifacts(args, spec, render)
        return
    if not isinstance(renderer, PedalboardRenderer) or renderer.plugin is None:
        raise RuntimeError("always-on GUI rendering requires a cached Pedalboard plugin")
    run_with_editor_held_open(
        renderer.plugin,
        lambda: _render_prediction_artifacts(args, spec, render),
    )


def main(cli_args: list[str] | None = None) -> None:
    """Parse the process request and render prediction artifacts.

    :param cli_args: Explicit arguments for tests; ``None`` reads ``sys.argv``.
    """
    render_prediction_audio(CliApp.run(_PredictAudioCliArgs, cli_args=cli_args))


if __name__ == "__main__":
    main()
