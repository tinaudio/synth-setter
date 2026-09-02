"""Render Surge patches from paired content and vocal-sketch audio."""

from __future__ import annotations

import csv
import hashlib
import math
import subprocess
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Literal

import click
import numpy as np
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator

from synth_setter.cli.clap_render import (
    _render_wav,
    _resolve_device,
    _workspace_render_config,
    resolve_inverse_checkpoint,
)
from synth_setter.conditioning import conditioning_batch_key, resolve_sketch_controls
from synth_setter.data.third_party_datamodule import decode_clip
from synth_setter.data.vst.core import write_wav
from synth_setter.data.vst.param_spec import (
    decode_model_output,
    require_note_params,
    require_scalar_synth_params,
)
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.data.vst.shapes import make_spectrogram
from synth_setter.evaluation.compute_audio_metrics import compute_metrics_on_dir
from synth_setter.evaluation.predict_vst_audio import params_to_csv
from synth_setter.features.sketch_controls import extract_sketch_controls
from synth_setter.model_cache import synth_setter_cache_dir
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.sketch import pool_sketch_controls
from synth_setter.workspace import operator_workspace

_DeviceSetting = Literal["auto", "cpu", "cuda", "mps"]
_CONFIG_NAME = "sketch_render"
_EXPECTED_CONDITIONING = "mel"
_METRIC_FIELDS = (
    "content_cfg",
    "sketch_cfg",
    "seed",
    "mss",
    "wmfcc",
    "sot",
    "rms",
    "r2_uri",
)


class _SketchRenderSettings(BaseModel):
    """Validated defaults for sketch-conditioned rendering.

    .. attribute :: model_config

        Strict immutable Pydantic validation.

    .. attribute :: checkpoint

        Local path or R2 URI for the inverse model.

    .. attribute :: stats

        Local path or R2 URI for mel statistics.

    .. attribute :: output_dir

        Default local artifact directory.

    .. attribute :: upload_prefix

        Default R2 artifact prefix.

    .. attribute :: device

        Requested inference device.

    .. attribute :: seed

        Initial flow-noise seed.

    .. attribute :: sample_steps

        Flow integration steps.

    .. attribute :: render

        Surge renderer identity and audio grid.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    checkpoint: str
    stats: str
    output_dir: Path
    upload_prefix: str
    device: _DeviceSetting
    seed: int
    sample_steps: int
    render: RenderConfig

    @field_validator("output_dir", mode="before")
    @classmethod
    def _parse_output_dir(cls, value: object) -> Path:
        """Parse the configured local output path.

        :param value: Candidate path value.
        :returns: Filesystem path.
        :raises TypeError: The value is not text or a path.
        """
        if not isinstance(value, (str, Path)):
            raise TypeError("output_dir must be a filesystem path")
        return Path(value)

    @field_validator("upload_prefix")
    @classmethod
    def _validate_upload_prefix(cls, value: str) -> str:
        """Require an R2 upload prefix.

        :param value: Candidate upload prefix.
        :returns: Prefix without a trailing slash.
        :raises ValueError: The value is not an R2 URI.
        """
        if not r2_io.is_r2_uri(value):
            raise ValueError("upload_prefix must use r2://")
        return value.rstrip("/")


def cfg_grid(
    content_strengths: Sequence[float], sketch_strengths: Sequence[float]
) -> tuple[tuple[float, float], ...]:
    """Expand independent guidance values into a content-major grid.

    :param content_strengths: Requested content guidance values.
    :param sketch_strengths: Requested sketch guidance values.
    :returns: Cartesian product preserving argument order.
    :raises ValueError: Either axis is empty, non-finite, or negative.
    """
    if not content_strengths or not sketch_strengths:
        raise ValueError("content and sketch CFG lists must be non-empty")
    for strength in (*content_strengths, *sketch_strengths):
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("CFG strengths must be finite and non-negative")
    return tuple(product(content_strengths, sketch_strengths))


def _load_settings() -> _SketchRenderSettings:
    """Compose sketch rendering defaults with the Surge identity.

    :returns: Strict settings ready for one invocation.
    :raises TypeError: Hydra nodes do not resolve to mappings.
    """
    with initialize_config_module(config_module="synth_setter.configs", version_base="1.3"):
        cfg = compose(config_name=_CONFIG_NAME)
    values = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("sketch_render config must resolve to a mapping")
    render_values = values.pop("render")
    synth_values = values.pop("synth")
    if not isinstance(render_values, dict) or not isinstance(synth_values, dict):
        raise TypeError("sketch_render render and synth nodes must resolve to mappings")
    render_values["synth"] = synth_values
    values["render"] = RenderConfig.model_validate(render_values)
    return _SketchRenderSettings.model_validate(values)


def _resolve_stats(source: str) -> Path:
    """Resolve local or R2 mel statistics to a reusable file.

    :param source: Local path or exact R2 object URI.
    :returns: Existing local path or cached R2 object.
    :raises FileNotFoundError: A local source does not exist.
    """
    if not r2_io.is_r2_uri(source):
        local = Path(source).expanduser()
        if not local.is_file():
            raise FileNotFoundError(f"mel statistics do not exist: {local}")
        return local.resolve()
    cache_key = hashlib.sha256(source.encode()).hexdigest()
    cached = synth_setter_cache_dir() / "models" / "mel-stats" / cache_key / "stats.npz"
    if cached.is_file() and cached.stat().st_size > 0:
        return cached
    cached.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cached.parent, suffix=".npz", delete=False) as stream:
        staging = Path(stream.name)
    try:
        r2_io.download_to_path(source, staging)
        staging.replace(cached)
    finally:
        staging.unlink(missing_ok=True)
    return cached


def load_audio_file(
    path: Path, *, sample_rate: int, channels: int, num_samples: int
) -> np.ndarray:
    """Decode audio onto a fixed grid, including codecs FFmpeg must transcode.

    :param path: Source audio file.
    :param sample_rate: Target sample rate in Hz.
    :param channels: Target channel count.
    :param num_samples: Target samples per channel.
    :returns: Channel-first float32 waveform.
    :raises ValueError: Neither decoder produces finite normalized audio.
    """
    try:
        return decode_clip(
            path.read_bytes(),
            sample_rate=sample_rate,
            channels=channels,
            num_samples=num_samples,
            amplitude_scale=1.0,
        )
    except ValueError as decode_error:
        result = subprocess.run(  # noqa: S603 — arguments are passed without a shell
            [  # noqa: S607 — FFmpeg resolves from the operator environment
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ac",
                str(channels),
                "-ar",
                str(sample_rate),
                "pipe:1",
            ],
            capture_output=True,
            check=False,
            timeout=60,
        )
        if result.returncode != 0:
            detail = result.stderr.decode(errors="replace").strip()
            raise ValueError(f"FFmpeg could not decode {path}: {detail}") from decode_error
        samples = np.frombuffer(result.stdout, dtype="<f4")
        if samples.size % channels:
            raise ValueError(f"FFmpeg returned an incomplete audio frame for {path}")
        audio = samples.reshape(-1, channels).T
        if audio.shape[1] < num_samples:
            audio = np.pad(audio, ((0, 0), (0, num_samples - audio.shape[1])))
        clip = np.ascontiguousarray(audio[:, :num_samples], dtype=np.float32)
        if not np.isfinite(clip).all():
            raise ValueError(f"FFmpeg returned non-finite audio for {path}")
        # Band-limited resampling can overshoot PCM bounds; restore the model's input contract.
        return np.clip(clip, -1.0, 1.0)


def _load_audio(path: Path, render: RenderConfig) -> np.ndarray:
    """Decode one audio file onto the render grid.

    :param path: Source audio file.
    :param render: Target sample rate, channels, and duration.
    :returns: Channel-first float32 waveform.
    """
    return load_audio_file(
        path,
        sample_rate=render.sample_rate,
        channels=render.channels,
        num_samples=int(render.sample_rate * render.signal_duration_seconds),
    )


def _prepare_inputs(
    sketch_audio: np.ndarray,
    content_audio: np.ndarray,
    stats_path: Path,
    model: VSTFlowMatchingModule,
    render: RenderConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Prepare normalized content mel and pooled sketch controls.

    :param sketch_audio: Channel-first sketch waveform.
    :param content_audio: Channel-first content waveform.
    :param stats_path: Training mel statistics archive.
    :param model: Loaded sketch-conditioned model.
    :param render: Active audio grid.
    :param device: Inference device.
    :returns: Model batch with one content/sketch pair.
    :raises ValueError: Statistics or checkpoint sketch controls are invalid.
    """
    with np.load(stats_path) as stats:
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("mel statistics must be finite with positive standard deviations")
    mel = make_spectrogram(content_audio, render.sample_rate)
    normalized = np.asarray((mel - mean) / std, dtype=np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized content mel contains non-finite values")

    sketch_spec = resolve_sketch_controls(model.hparams["sketch_controls"])
    if sketch_spec is None:
        raise ValueError("checkpoint must configure sketch_controls")
    controls = extract_sketch_controls(
        torch.from_numpy(sketch_audio), render.sample_rate
    ).unsqueeze(0)
    controls = pool_sketch_controls(controls, sketch_spec.num_frames).to(dtype=torch.float32)
    controls = controls.clone()
    pitch = controls[:, 2:]
    controls[:, 2:] = pitch.where(pitch >= sketch_spec.pitch_zero_threshold, 0.0)
    return {
        _EXPECTED_CONDITIONING: torch.from_numpy(normalized).unsqueeze(0).to(device),
        "sketch_ctrl": controls.to(device),
    }


def _load_model(
    checkpoint: Path, render: RenderConfig, device: torch.device
) -> VSTFlowMatchingModule:
    """Load and validate one mel-plus-sketch checkpoint.

    :param checkpoint: Materialized Lightning checkpoint.
    :param render: Active Surge parameter specification.
    :param device: Inference device.
    :returns: Evaluation model on ``device``.
    :raises ValueError: Conditioning, sketch controls, or output width is incompatible.
    """
    model = VSTFlowMatchingModule.load_from_checkpoint(
        checkpoint,
        map_location=device,
        weights_only=False,
    )
    if conditioning_batch_key(model.hparams["conditioning"]) != _EXPECTED_CONDITIONING:
        raise ValueError("checkpoint must use mel content conditioning")
    if resolve_sketch_controls(model.hparams["sketch_controls"]) is None:
        raise ValueError("checkpoint must configure sketch_controls")
    expected_width = len(param_specs[render.param_spec_name])
    if model.hparams["num_params"] != expected_width:
        raise ValueError(
            f"checkpoint output width {model.hparams['num_params']} does not match "
            f"{render.param_spec_name} width {expected_width}"
        )
    return model.to(device).eval()


def cfg_arm_name(content_cfg: float, sketch_cfg: float) -> str:
    """Build a stable filesystem name for one guidance arm.

    :param content_cfg: Content guidance strength.
    :param sketch_cfg: Sketch guidance strength.
    :returns: Arm identifier.
    """
    return f"cfg-c{content_cfg:g}-s{sketch_cfg:g}"


def _write_metrics(path: Path, row: dict[str, str | int | float]) -> None:
    """Write one arm's guidance and audio metrics.

    :param path: CSV destination.
    :param row: Values for every metric field.
    """
    fieldnames: list[str] = list(_METRIC_FIELDS)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def _run_id(sketch_path: Path, content_path: Path) -> str:
    """Build a unique identifier carrying pair identity.

    :param sketch_path: Sketch source path.
    :param content_path: Content source path.
    :returns: Timestamped identifier.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    identity = hashlib.sha256(
        f"{sketch_path.resolve()}\0{content_path.resolve()}".encode()
    ).hexdigest()
    return f"sketch-{timestamp}-{identity[:8]}"


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("sketch_wav", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("content_wav", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--checkpoint", envvar="SYNTH_SETTER_SKETCH_CHECKPOINT")
@click.option("--stats", "stats_source", envvar="SYNTH_SETTER_SKETCH_MEL_STATS")
@click.option("--content-cfg", type=float, multiple=True, default=(2.0,), show_default=True)
@click.option("--sketch-cfg", type=float, multiple=True, default=(2.0,), show_default=True)
@click.option("--sample-steps", type=int)
@click.option("--seed", type=int)
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--upload-prefix", help="Exact r2:// directory receiving this pair's arms.")
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    default=None,
)
@click.option("--upload/--no-upload", default=True, show_default=True)
def main(
    sketch_wav: Path,
    content_wav: Path,
    checkpoint: str | None,
    stats_source: str | None,
    content_cfg: tuple[float, ...],
    sketch_cfg: tuple[float, ...],
    sample_steps: int | None,
    seed: int | None,
    output_dir: Path | None,
    upload_prefix: str | None,
    device: _DeviceSetting | None,
    upload: bool,
) -> None:
    """Render SKETCH_WAV timing and pitch with CONTENT_WAV timbre.

    Repeat ``--content-cfg`` and ``--sketch-cfg`` to render their Cartesian grid.

    :param sketch_wav: Vocal sketch source audio.
    :param content_wav: Content/timbre reference audio.
    :param checkpoint: Optional sketch-conditioned checkpoint override.
    :param stats_source: Optional mel-statistics override.
    :param content_cfg: Content guidance strengths.
    :param sketch_cfg: Sketch guidance strengths.
    :param sample_steps: Optional integration-step override.
    :param seed: Initial-noise seed shared across every CFG arm.
    :param output_dir: Local pair output directory.
    :param upload_prefix: Exact R2 pair destination.
    :param device: Optional inference device override.
    :param upload: Whether to upload each completed arm.
    :raises click.ClickException: CLI arguments are inconsistent.
    """
    settings = _load_settings()
    grid = cfg_grid(content_cfg, sketch_cfg)
    selected_steps = settings.sample_steps if sample_steps is None else sample_steps
    if selected_steps <= 0:
        raise click.ClickException("--sample-steps must be positive")
    if upload_prefix is not None and not r2_io.is_r2_uri(upload_prefix):
        raise click.ClickException("--upload-prefix must use r2://")
    if upload_prefix is not None and not upload:
        raise click.ClickException("--upload-prefix cannot be combined with --no-upload")

    checkpoint_source = checkpoint or settings.checkpoint
    stats = stats_source or settings.stats
    selected_device = _resolve_device(device or settings.device)
    selected_seed = settings.seed if seed is None else seed
    run_id = _run_id(sketch_wav, content_wav)
    pair_output = output_dir or settings.output_dir / run_id
    if not pair_output.is_absolute():
        pair_output = operator_workspace() / pair_output
    pair_output = pair_output.resolve()
    pair_upload = (upload_prefix or f"{settings.upload_prefix}/{run_id}").rstrip("/")

    if upload or r2_io.is_r2_uri(checkpoint_source) or r2_io.is_r2_uri(stats):
        r2_io.ensure_r2_env_loaded()
    render = _workspace_render_config(settings.render)
    model = _load_model(
        resolve_inverse_checkpoint(checkpoint_source),
        render,
        selected_device,
    )
    sketch_audio = _load_audio(sketch_wav, render)
    content_audio = _load_audio(content_wav, render)
    batch = _prepare_inputs(
        sketch_audio,
        content_audio,
        _resolve_stats(stats),
        model,
        render,
        selected_device,
    )
    generator = torch.Generator(device=selected_device).manual_seed(selected_seed)
    noise = torch.randn(
        (1, model.hparams["num_params"]),
        generator=generator,
        device=selected_device,
        dtype=torch.float32,
    )

    spec = param_specs[render.param_spec_name]
    target_params_path = content_wav.with_suffix(".params.npy")
    target_params = np.load(target_params_path) if target_params_path.is_file() else None
    for content_strength, sketch_strength in grid:
        arm = cfg_arm_name(content_strength, sketch_strength)
        arm_dir = pair_output / "arms" / arm
        if arm_dir.exists():
            raise click.ClickException(f"refusing to overwrite existing arm: {arm_dir}")
        arm_dir.mkdir(parents=True)
        prediction = (
            model.sample_batch(
                batch,
                noise=noise,
                content_cfg_strength=content_strength,
                sketch_cfg_strength=sketch_strength,
                sample_steps=selected_steps,
            )
            .detach()
            .cpu()
        )
        write_wav(sketch_audio, str(arm_dir / "sketch.wav"), render.sample_rate, render.channels)
        write_wav(content_audio, str(arm_dir / "target.wav"), render.sample_rate, render.channels)
        _render_wav(prediction, render, arm_dir / "pred.wav")

        pred_synth, pred_note = decode_model_output(prediction[0].float().numpy(), spec)
        pred_note_params = require_note_params(pred_note)
        pred_note_start, pred_note_end = sorted(pred_note_params["note_start_and_end"])
        target_synth = target_note = None
        if target_params is not None:
            target_synth, target_note = decode_model_output(np.asarray(target_params), spec)
        params_to_csv(
            require_scalar_synth_params(target_synth) if target_synth is not None else None,
            require_note_params(target_note) if target_note is not None else None,
            require_scalar_synth_params(pred_synth),
            pred_note_params,
            str(arm_dir / "params.csv"),
            spec,
            pred_effective_note_window=(pred_note_start, pred_note_end),
        )
        metrics = compute_metrics_on_dir(arm_dir)
        arm_uri = f"{pair_upload}/arms/{arm}" if upload else ""
        _write_metrics(
            arm_dir / "metrics.csv",
            {
                "content_cfg": content_strength,
                "sketch_cfg": sketch_strength,
                "seed": selected_seed,
                **metrics,
                "r2_uri": arm_uri,
            },
        )
        if upload:
            r2_io.upload_dir(arm_dir, arm_uri)
        click.echo(f"{arm}: {arm_dir}")
    if upload:
        click.echo(f"R2: {pair_upload}")


if __name__ == "__main__":
    main()
