"""Render a Surge patch from reference timbre and guide sketch controls.

Run ``synth-setter-clap --guide_audio guide.wav --ref_audio reference.wav``.
Checkpoint rotation requires updating its pinned digest and state-shape contract together.
"""

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import click
import numpy as np
import torch
from pedalboard.io import AudioFile

from synth_setter.data.vst.core import extract_renderer_version
from synth_setter.data.vst.generate_vst_dataset import make_spectrogram
from synth_setter.data.vst.param_spec import NoteParams, decode_model_output
from synth_setter.data.vst.param_spec_registry import default_plugin_path, param_specs
from synth_setter.evaluation.predict_vst_audio import params_to_csv
from synth_setter.features.sketch_controls import SKETCH_PITCH_SLICE, extract_sketch_controls
from synth_setter.model_cache import cache_r2_file
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.renderer_factory import make_audio_renderer
from synth_setter.resources import as_file, surge_simple_preset, vst_headless_wrapper
from synth_setter.run_id import make_wandb_run_id
from synth_setter.synth_spec import SYNTHS, SynthName
from synth_setter.utils.logging_utils import resolve_git_sha

DEFAULT_CHECKPOINT_URI = (
    "r2://intermediate-data/checkpoints/flow_sketch_prelim/"
    "flow_sketch_prelim-20260729T175159901Z-b60d52e9bbc04054ba7dcf4ccabfcb4b/last.ckpt"
)
DEFAULT_STATS_URI = (
    "r2://experiments/data/surge-simple-lance-1k-2k-2k/"
    "surge-simple-lance-1k-2k-2k-20260716T163226347Z/stats.npz"
)
_CHECKPOINT_SHA256 = "e6e3d1c702b092718c3d81d634bba5b015d031e211fd6bd005e35a54dea2f89a"
_STATS_SHA256 = "397415e5f900056256ac6b6047d3601c6c478b9d760c2c45c1599b1e45e31acb"
_CACHE_NAMESPACE = "surge-sketch-cli"
_HEADLESS_ENV = "SYNTH_SETTER_CLAP_HEADLESS"
_OUTPUT_ROOT = Path("outputs/synth-setter-clap")
_R2_OUTPUT_ROOT = "r2://intermediate-data/eval/synth-setter-clap"
_SAMPLE_RATE = 44100
_CHANNELS = 2
_DURATION_SECONDS = 4.0
_SERVING_SEED = 0
_HEADLESS_TIMEOUT_SECONDS = 1200
_SURGE_PARAM_SPEC_NAME = "surge_simple"
_EXPECTED_PARAM_WIDTH = param_specs[_SURGE_PARAM_SPEC_NAME].encoded_width
_PITCH_ZERO_THRESHOLD = 0.1
_MIN_LOUDNESS_DB = -55.0
_EXPECTED_STATE_SHAPES = {
    "encoder.patch_embed.projection.weight": (512, 2, 16, 16),
    "sketch_tokens.positional_encoding": (1, 32, 512),
    "sketch_tokens.projections.centroid.weight": (512, 1),
    "sketch_tokens.projections.loudness.weight": (512, 1),
    "sketch_tokens.projections.pitch.weight": (512, 384),
    "vector_field.projection._assignment": (128, _EXPECTED_PARAM_WIDTH),
}
_EXPECTED_SKETCH_CONFIG = {
    "column": "sketch",
    "num_frames": 401,
    "num_control_tokens": 32,
    "pitch_zero_threshold": _PITCH_ZERO_THRESHOLD,
}


@dataclass(frozen=True)
class PreparedAudioInputs:
    """Normalized waveforms and model features for one render request.

    .. attribute :: guide_audio

        Normalized stereo guide waveform.

    .. attribute :: ref_audio

        Normalized stereo reference waveform.

    .. attribute :: ref_mel

        Stats-normalized reference mel spectrogram.

    .. attribute :: sketch_controls

        Guide controls on the model frame grid.
    """

    guide_audio: torch.Tensor
    ref_audio: torch.Tensor
    ref_mel: torch.Tensor
    sketch_controls: torch.Tensor


@dataclass(frozen=True)
class RenderedPatch:
    """Rendered waveform and the decoded patch that produced it.

    .. attribute :: audio

        Rendered stereo waveform.

    .. attribute :: synth_params

        Renderer-native Surge parameters.

    .. attribute :: note_params

        Decoded MIDI pitch and note interval.
    """

    audio: np.ndarray
    synth_params: dict[str, float]
    note_params: NoteParams


def _fit_audio_to_model_grid(audio: np.ndarray) -> torch.Tensor:
    """Fit channel-first audio to the checkpoint's stereo sample grid.

    :param audio: Resampled channel-first waveform.
    :returns: Float32 waveform shaped ``(2, 176400)``.
    :raises ValueError: The source has more than two channels.
    """
    if audio.shape[0] == 1:
        audio = np.repeat(audio, _CHANNELS, axis=0)
    elif audio.shape[0] != _CHANNELS:
        raise ValueError(f"audio must have one or two channels, found {audio.shape[0]}")
    target_samples = int(_SAMPLE_RATE * _DURATION_SECONDS)
    if audio.shape[1] < target_samples:
        audio = np.pad(audio, ((0, 0), (0, target_samples - audio.shape[1])))
    fitted_audio = audio[:, :target_samples]
    if not np.isfinite(fitted_audio).all() or np.any(np.abs(fitted_audio) > 1.0):
        raise ValueError("input audio must be finite and within [-1, 1]")
    return torch.from_numpy(fitted_audio).to(dtype=torch.float32)


def _load_model_audio(path: Path) -> torch.Tensor:
    """Load an unshifted clip onto the checkpoint's sample grid.

    :param path: Source audio accepted by Pedalboard.
    :returns: Float32 waveform shaped ``(2, 176400)``.
    """
    target_samples = int(_SAMPLE_RATE * _DURATION_SECONDS)
    with AudioFile(str(path), "r").resampled_to(_SAMPLE_RATE) as audio_file:
        return _fit_audio_to_model_grid(audio_file.read(target_samples))


def _normalize_reference_mel(ref_audio: torch.Tensor, stats_file: Path) -> torch.Tensor:
    """Apply the checkpoint's saved statistics to its training mel frontend.

    :param ref_audio: Float32 waveform shaped ``(2, 176400)``.
    :param stats_file: Training mel mean and standard deviation.
    :returns: Float32 normalized mel shaped ``(2, 128, 401)``.
    :raises ValueError: Normalization produces a non-finite value.
    """
    with np.load(stats_file) as stats:
        mel = (make_spectrogram(ref_audio.numpy(), _SAMPLE_RATE) - stats["mean"]) / stats["std"]
    if not np.isfinite(mel).all():
        raise ValueError("reference mel normalization produced non-finite values")
    return torch.from_numpy(mel).to(dtype=torch.float32)


def prepare_audio_inputs(
    guide_audio: Path, ref_audio: Path, stats_file: Path
) -> PreparedAudioInputs:
    """Prepare unshifted guide controls and reference mel on the training grid.

    :param guide_audio: Audio supplying sketch controls.
    :param ref_audio: Audio supplying mel/timbre conditioning.
    :param stats_file: Training mel mean and standard deviation.
    :returns: Four-second stereo waveforms and features on the 401-frame grid.
    """
    guide_waveform = _load_model_audio(guide_audio)
    ref_waveform = _load_model_audio(ref_audio)
    sketch_controls = extract_sketch_controls(guide_waveform, _SAMPLE_RATE)
    pitch = sketch_controls[SKETCH_PITCH_SLICE]
    sketch_controls[SKETCH_PITCH_SLICE] = pitch.where(pitch >= _PITCH_ZERO_THRESHOLD, 0.0)
    return PreparedAudioInputs(
        guide_audio=guide_waveform,
        ref_audio=ref_waveform,
        ref_mel=_normalize_reference_mel(ref_waveform, stats_file),
        sketch_controls=sketch_controls,
    )


def _write_wav(path: Path, audio: np.ndarray | torch.Tensor) -> None:
    """Write finite normalized channel-first audio at the model sample rate.

    :param path: Destination WAV path.
    :param audio: Waveform shaped ``(2, 176400)`` with values in ``[-1, 1]``.
    :raises ValueError: The waveform is non-finite or outside ``[-1, 1]``.
    """
    values = audio.detach().cpu().numpy() if isinstance(audio, torch.Tensor) else audio
    if not np.isfinite(values).all() or np.any(np.abs(values) > 1.0):
        raise ValueError("output audio must be finite and within [-1, 1]")
    with AudioFile(str(path), "w", _SAMPLE_RATE, _CHANNELS) as audio_file:
        audio_file.write(values)


def write_output_artifacts(
    output_dir: Path, prepared: PreparedAudioInputs, patch: RenderedPatch
) -> None:
    """Persist normalized inputs, rendered audio, and decoded patch parameters.

    :param output_dir: Retained local output directory.
    :param prepared: Normalized guide/reference audio and model features.
    :param patch: Rendered prediction and decoded parameters.
    """
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_wav(output_dir / "guide.wav", prepared.guide_audio)
    _write_wav(output_dir / "ref.wav", prepared.ref_audio)
    _write_wav(output_dir / "pred.wav", patch.audio)
    params_to_csv(
        None,
        None,
        patch.synth_params,
        patch.note_params,
        str(output_dir / "params.csv"),
        param_specs[_SURGE_PARAM_SPEC_NAME],
    )


def write_run_manifest(output_dir: Path, destination_uri: str) -> None:
    """Record immutable model, feature, render, and destination provenance.

    :param output_dir: Retained local output directory.
    :param destination_uri: R2 prefix receiving the run artifacts.
    """
    synth_identity = SYNTHS[SynthName(_SURGE_PARAM_SPEC_NAME)]
    manifest = {
        "schema_version": 1,
        "run_id": output_dir.name,
        "r2_uri": destination_uri,
        "code_version": version("synth-setter"),
        "git_sha": resolve_git_sha(),
        "checkpoint": {"uri": DEFAULT_CHECKPOINT_URI, "sha256": _CHECKPOINT_SHA256},
        "stats": {"uri": DEFAULT_STATS_URI, "sha256": _STATS_SHA256},
        "render": {
            "param_spec": _SURGE_PARAM_SPEC_NAME,
            "synth_version": synth_identity.synth_version,
            "sample_rate": _SAMPLE_RATE,
            "channels": _CHANNELS,
            "duration_seconds": _DURATION_SECONDS,
            "seed": _SERVING_SEED,
        },
    }
    temporary_path = output_dir / ".manifest.json.tmp"
    temporary_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, output_dir / "manifest.json")


def upload_output_artifacts(output_dir: Path, destination_uri: str) -> None:
    """Upload the output artifact directory through the shared rclone path.

    :param output_dir: Directory containing only public output artifacts.
    :param destination_uri: Unique R2 prefix receiving the directory contents.
    """
    r2_io.upload_dir(output_dir, destination_uri)


def validate_checkpoint_compatibility(checkpoint: Mapping[str, object]) -> dict[str, object]:
    """Validate the immutable Surge sketch checkpoint before model construction.

    :param checkpoint: Deserialized Lightning checkpoint mapping.
    :returns: Current-form sketch configuration for ``load_from_checkpoint``.
    :raises ValueError: The synth identity, conditioning, or tensor shapes differ.
    """
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, Mapping):
        raise ValueError("checkpoint has no hyper_parameters mapping")

    if hyper_parameters.get("num_params") != _EXPECTED_PARAM_WIDTH:
        raise ValueError(
            f"checkpoint num_params must be {_EXPECTED_PARAM_WIDTH} for {_SURGE_PARAM_SPEC_NAME}"
        )
    param_spec = hyper_parameters.get("param_spec")
    if param_spec not in (None, _SURGE_PARAM_SPEC_NAME):
        raise ValueError(
            f"checkpoint param_spec must be {_SURGE_PARAM_SPEC_NAME!r}, got {param_spec!r}"
        )
    if hyper_parameters.get("conditioning") != "mel":
        raise ValueError("checkpoint conditioning must be 'mel'")

    sketch = hyper_parameters.get("sketch_controls")
    if not isinstance(sketch, Mapping):
        raise ValueError("checkpoint must enable sketch controls")
    normalized_sketch = dict(sketch)
    legacy_token_count = normalized_sketch.pop("num_ctrl_tokens", None)
    current_token_count = normalized_sketch.get("num_control_tokens")
    if (
        legacy_token_count is not None
        and current_token_count is not None
        and legacy_token_count != current_token_count
    ):
        raise ValueError("checkpoint sketch token-count fields conflict")
    if current_token_count is None:
        normalized_sketch["num_control_tokens"] = legacy_token_count
    if normalized_sketch != _EXPECTED_SKETCH_CONFIG:
        raise ValueError(
            f"checkpoint sketch config must be {_EXPECTED_SKETCH_CONFIG}, got {normalized_sketch}"
        )

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint has no state_dict mapping")
    for name, expected_shape in _EXPECTED_STATE_SHAPES.items():
        tensor = state_dict.get(name)
        actual_shape = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else None
        if actual_shape != expected_shape:
            raise ValueError(
                f"checkpoint {name} shape must be {expected_shape}, got {actual_shape}"
            )
    return normalized_sketch


def _validate_stats(stats_file: Path) -> None:
    """Require the mel-stat arrays expected by the pinned training frontend.

    :param stats_file: Candidate ``stats.npz`` artifact.
    :raises ValueError: Mean/std shapes or values differ from the model input contract.
    """
    with np.load(stats_file) as stats:
        if set(stats.files) != {"mean", "std"}:
            raise ValueError("stats.npz must contain exactly mean and std arrays")
        mean = stats["mean"]
        std = stats["std"]
    expected_shape = (_CHANNELS, 128, 401)
    if mean.shape != expected_shape or std.shape != expected_shape:
        raise ValueError(f"stats mean/std must both have shape {expected_shape}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("stats mean/std must be finite and std must be positive")


def _load_model(checkpoint_path: Path, device: torch.device) -> VSTFlowMatchingModule:
    """Validate and load the pinned flow checkpoint on ``device``.

    :param checkpoint_path: Digest-pinned Lightning checkpoint.
    :param device: Inference device.
    :returns: Compatible model in eval mode.
    :raises ValueError: The checkpoint root or compatibility contract is invalid.
    """
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    sketch_config = validate_checkpoint_compatibility(payload)
    del payload
    model = VSTFlowMatchingModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        weights_only=False,
        sketch_controls=sketch_config,
    )
    model.to(device)
    model.eval()
    return model


def _predict_patch(
    prepared: PreparedAudioInputs, model: VSTFlowMatchingModule
) -> tuple[dict[str, float], NoteParams]:
    """Infer and decode one Surge patch through the model's predict step.

    :param prepared: Reference mel and guide controls.
    :param model: Compatible flow model in eval mode.
    :returns: Renderer-native synth and note parameters.
    :raises ValueError: The prediction is non-finite or has the wrong shape.
    """
    batch = {
        "mel": prepared.ref_mel.unsqueeze(0).to(model.device),
        "sketch_ctrl": prepared.sketch_controls.unsqueeze(0).to(model.device),
    }
    torch.manual_seed(_SERVING_SEED)
    if model.device.type == "cuda":
        torch.cuda.manual_seed_all(_SERVING_SEED)
    with torch.no_grad():
        prediction, _ = model.predict_step(batch, 0)
    expected_shape = (1, _EXPECTED_PARAM_WIDTH)
    if tuple(prediction.shape) != expected_shape or not torch.isfinite(prediction).all():
        raise ValueError(
            f"model prediction must be finite with shape {expected_shape}, "
            f"got {tuple(prediction.shape)}"
        )
    return decode_model_output(
        prediction[0].detach().cpu().float().numpy(), param_specs[_SURGE_PARAM_SPEC_NAME]
    )


def _render_patch(synth_params: dict[str, float], note_params: NoteParams) -> RenderedPatch:
    """Render decoded parameters through the production Surge VST backend.

    :param synth_params: Renderer-native Surge parameter values.
    :param note_params: Decoded MIDI pitch and note interval.
    :returns: Rendered stereo patch and its parameters.
    :raises ValueError: The installed Surge version differs from the pinned contract.
    """
    plugin_path = str(Path(default_plugin_path()).expanduser().resolve())
    synth_identity = SYNTHS[SynthName(_SURGE_PARAM_SPEC_NAME)]
    actual_version = extract_renderer_version(Path(plugin_path))
    if actual_version != synth_identity.synth_version:
        raise ValueError(
            f"Surge version must be {synth_identity.synth_version!r}, got "
            f"{actual_version!r} from {plugin_path}"
        )
    with as_file(surge_simple_preset()) as preset_path:
        synth = synth_identity.model_copy(
            update={
                "plugin_path": plugin_path,
                "plugin_state_path": str(preset_path),
            }
        )
        render_config = RenderConfig(
            synth=synth,
            renderer_backend="pedalboard",
            sample_rate=_SAMPLE_RATE,
            channels=_CHANNELS,
            velocity=100,
            signal_duration_seconds=_DURATION_SECONDS,
            min_loudness=_MIN_LOUDNESS_DB,
            samples_per_shard=1,
            samples_per_render_batch=1,
            plugin_reload_cadence="once",
            gui_toggle_cadence="once",
        )
        renderer = make_audio_renderer(render_config)
        note_start, note_end = sorted(note_params["note_start_and_end"])
        audio = renderer.render(
            synth_params,
            int(note_params["pitch"]),
            render_config.velocity,
            (note_start, note_end),
            warmup=True,
        )
    return RenderedPatch(audio=audio, synth_params=synth_params, note_params=note_params)


def _run_request(guide_audio: Path, ref_audio: Path) -> tuple[Path, str]:
    """Run one complete local render and R2 upload.

    :param guide_audio: Audio supplying sketch controls.
    :param ref_audio: Audio supplying mel conditioning.
    :returns: Retained local output path and uploaded R2 prefix.
    """
    r2_io.ensure_r2_env_loaded()
    checkpoint_path = cache_r2_file(DEFAULT_CHECKPOINT_URI, _CACHE_NAMESPACE, _CHECKPOINT_SHA256)
    stats_path = cache_r2_file(DEFAULT_STATS_URI, _CACHE_NAMESPACE, _STATS_SHA256)
    _validate_stats(stats_path)

    prepared = prepare_audio_inputs(guide_audio, ref_audio, stats_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint_path, device)
    synth_params, note_params = _predict_patch(prepared, model)
    patch = _render_patch(synth_params, note_params)

    run_id = f"{make_wandb_run_id('synth-setter-clap')}-{uuid4().hex[:8]}"
    output_dir = _OUTPUT_ROOT / run_id
    destination_uri = f"{_R2_OUTPUT_ROOT}/{run_id}"
    write_output_artifacts(output_dir, prepared, patch)
    write_run_manifest(output_dir, destination_uri)
    resolved_output = output_dir.resolve()
    click.echo(f"Local output: {resolved_output}", err=True)
    upload_output_artifacts(output_dir, destination_uri)
    return resolved_output, destination_uri


def _run_under_headless_wrapper(guide_audio: Path, ref_audio: Path) -> None:
    """Re-enter the public module under the packaged Linux X11 wrapper.

    :param guide_audio: Audio supplying sketch controls.
    :param ref_audio: Audio supplying mel conditioning.
    """
    with as_file(vst_headless_wrapper()) as wrapper:
        subprocess.run(  # noqa: S603 — fixed package entrypoint and validated paths
            [
                str(wrapper),
                sys.executable,
                "-m",
                "synth_setter.cli.clap",
                "--guide_audio",
                str(guide_audio.resolve()),
                "--ref_audio",
                str(ref_audio.resolve()),
            ],
            check=True,
            env={**os.environ, _HEADLESS_ENV: "1"},
            timeout=_HEADLESS_TIMEOUT_SECONDS,
        )


@click.command(help="Infer, render, retain, and upload one Surge patch.")
@click.option(
    "--guide_audio",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Guide audio supplying sketch controls.",
)
@click.option(
    "--ref_audio",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Reference audio supplying mel/timbre conditioning.",
)
def main(guide_audio: Path, ref_audio: Path) -> None:
    """Infer, render, retain, and upload one Surge patch.

    :param guide_audio: Audio supplying sketch controls.
    :param ref_audio: Audio supplying mel/timbre conditioning.
    """
    if sys.platform == "linux" and os.environ.get(_HEADLESS_ENV) != "1":
        _run_under_headless_wrapper(guide_audio, ref_audio)
        return

    _, destination_uri = _run_request(guide_audio, ref_audio)
    click.echo(destination_uri)


if __name__ == "__main__":
    main()
