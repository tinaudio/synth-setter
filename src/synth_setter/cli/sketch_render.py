"""Render a Surge patch from reference timbre and guide sketch controls.

Run ``synth-setter-sketch-render --guide-audio guide.wav --reference-audio reference.wav``.
Checkpoint rotation requires updating its pinned digest and state-shape contract together.
"""

import json
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import Literal
from uuid import uuid4

import click
import numpy as np
import torch
from click.core import ParameterSource
from pedalboard.io import AudioFile
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from synth_setter.cli._cfg_strength import (
    CfgStrengths,
    temporary_cfg_strength_overrides,
    validate_cfg_strength,
)
from synth_setter.data.vst.core import extract_renderer_version
from synth_setter.data.vst.param_spec import (
    NoteParams,
    decode_model_output,
    require_note_params,
    require_scalar_synth_params,
)
from synth_setter.data.vst.param_spec_registry import default_plugin_path, param_specs
from synth_setter.data.vst.shapes import make_spectrogram
from synth_setter.data.vst_datamodule import load_mel_statistics
from synth_setter.evaluation.predict_vst_audio import (
    _canonicalize_prediction_note_window,
    params_to_csv,
)
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
_CACHE_NAMESPACE = "surge-sketch-render"
_HEADLESS_ENV = "SYNTH_SETTER_SKETCH_RENDER_HEADLESS"
_OUTPUT_ROOT = Path("outputs/synth-setter-sketch-render")
_R2_OUTPUT_ROOT = "r2://intermediate-data/eval/synth-setter-sketch-render"
_OUTPUT_ROOT_ENV = "SYNTH_SETTER_SKETCH_OUTPUT_ROOT"
_R2_OUTPUT_ROOT_ENV = "SYNTH_SETTER_SKETCH_UPLOAD_PREFIX"
_SAMPLE_RATE = 44100
_CHANNELS = 2
_DURATION_SECONDS = 4.0
_EXPECTED_AUDIO_SHAPE = (_CHANNELS, int(_SAMPLE_RATE * _DURATION_SECONDS))
_SERVING_SEED = 0
_HEADLESS_TIMEOUT_SECONDS = 1200
_SURGE_PARAM_SPEC_NAME = "surge_simple"
_EXPECTED_PARAM_WIDTH = param_specs[_SURGE_PARAM_SPEC_NAME].encoded_width
_PITCH_ZERO_THRESHOLD = 0.1
_MIN_LOUDNESS_DB = -55.0
_EXPECTED_STATE_SHAPES = MappingProxyType(
    {
        "encoder.patch_embed.projection.weight": (512, 2, 16, 16),
        "sketch_tokens.positional_encoding": (1, 32, 512),
        "sketch_tokens.projections.centroid.weight": (512, 1),
        "sketch_tokens.projections.loudness.weight": (512, 1),
        "sketch_tokens.projections.pitch.weight": (512, 384),
        "vector_field.projection._assignment": (128, _EXPECTED_PARAM_WIDTH),
    }
)
_EXPECTED_SKETCH_CONFIG = MappingProxyType(
    {
        "column": "sketch",
        "num_frames": 401,
        "num_control_tokens": 32,
        "pitch_zero_threshold": _PITCH_ZERO_THRESHOLD,
    }
)
_RETAINED_ARTIFACT_FILENAMES = (
    "guide.wav",
    "manifest.json",
    "params.csv",
    "pred.wav",
    "ref.wav",
)


class _ManifestArtifact(BaseModel):
    """Immutable R2 artifact identity at the retry-upload trust boundary.

    .. attribute :: model_config

        Strict, frozen parsing configuration.

    .. attribute :: uri

        Immutable R2 artifact URI.

    .. attribute :: sha256

        Required artifact SHA-256 digest.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    uri: str
    sha256: str

    @field_validator("uri")
    @classmethod
    def _uri_is_r2(cls, value: str) -> str:
        if not r2_io.is_r2_uri(value):
            raise ValueError("manifest artifact URI must use r2://")
        return value

    @field_validator("sha256")
    @classmethod
    def _sha256_is_valid(cls, value: str) -> str:
        if len(value) != 64:
            raise ValueError("manifest artifact SHA-256 must contain 64 hex characters")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError("manifest artifact SHA-256 must be hexadecimal") from exc
        return value


class _ManifestRender(BaseModel):
    """Immutable renderer identity and settings for one retained run.

    .. attribute :: model_config

        Strict, frozen parsing configuration.

    .. attribute :: param_spec

        Parameter specification name.

    .. attribute :: synth_version

        Required synthesizer version.

    .. attribute :: sample_rate

        Render sample rate in hertz.

    .. attribute :: channels

        Render channel count.

    .. attribute :: duration_seconds

        Render duration in seconds.

    .. attribute :: seed

        Inference seed.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    param_spec: str = Field(min_length=1)
    synth_version: str = Field(min_length=1)
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    seed: int


class _RetainedRunManifest(BaseModel):
    """Complete versioned provenance required for retry upload.

    .. attribute :: model_config

        Strict, frozen parsing configuration.

    .. attribute :: schema_version

        Manifest contract version.

    .. attribute :: run_id

        Unique render-run identifier.

    .. attribute :: r2_uri

        Destination R2 prefix.

    .. attribute :: code_version

        Installed package version.

    .. attribute :: git_sha

        Source revision identifier.

    .. attribute :: content_cfg_strength

        Effective reference guidance strength.

    .. attribute :: sketch_cfg_strength

        Effective sketch guidance strength.

    .. attribute :: checkpoint

        Inverse checkpoint identity.

    .. attribute :: stats

        Normalization-statistics identity.

    .. attribute :: render

        Renderer identity and settings.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    schema_version: Literal[1]
    run_id: str = Field(min_length=1)
    r2_uri: str
    code_version: str = Field(min_length=1)
    git_sha: str = Field(min_length=1)
    content_cfg_strength: float = Field(ge=0, allow_inf_nan=False)
    sketch_cfg_strength: float = Field(ge=0, allow_inf_nan=False)
    checkpoint: _ManifestArtifact
    stats: _ManifestArtifact
    render: _ManifestRender

    @field_validator("r2_uri")
    @classmethod
    def _r2_uri_is_valid(cls, value: str) -> str:
        if not r2_io.is_r2_uri(value):
            raise ValueError("manifest r2_uri must use r2://")
        return value


@dataclass(frozen=True)
class PreparedAudioInputs:
    """Normalized waveforms and model features for one render request.

    .. attribute :: guide_audio

        Normalized stereo guide waveform.

    .. attribute :: reference_audio

        Normalized stereo reference waveform.

    .. attribute :: ref_mel

        Stats-normalized reference mel spectrogram.

    .. attribute :: sketch_controls

        Guide controls on the model frame grid.
    """

    guide_audio: torch.Tensor
    reference_audio: torch.Tensor
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

        Raw decoded MIDI pitch and note interval.

    .. attribute :: effective_note_window

        Canonical note interval used by the renderer.
    """

    audio: np.ndarray
    synth_params: dict[str, float]
    note_params: NoteParams
    effective_note_window: tuple[float, float]


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
    target_samples = _EXPECTED_AUDIO_SHAPE[1]
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


def _normalize_reference_mel(reference_audio: torch.Tensor, stats_file: Path) -> torch.Tensor:
    """Apply the checkpoint's saved statistics to its training mel frontend.

    :param reference_audio: Float32 waveform shaped ``(2, 176400)``.
    :param stats_file: Training mel mean and standard deviation.
    :returns: Float32 normalized mel shaped ``(2, 128, 401)``.
    :raises ValueError: Normalization produces a non-finite value.
    """
    mean, std = load_mel_statistics(stats_file)
    mel = (make_spectrogram(reference_audio.numpy(), _SAMPLE_RATE) - mean) / std
    if not np.isfinite(mel).all():
        raise ValueError("reference mel normalization produced non-finite values")
    return torch.from_numpy(mel).to(dtype=torch.float32)


def prepare_audio_inputs(
    guide_audio: Path, reference_audio: Path, stats_file: Path
) -> PreparedAudioInputs:
    """Prepare unshifted guide controls and reference mel on the training grid.

    :param guide_audio: Audio supplying sketch controls.
    :param reference_audio: Audio supplying mel/timbre conditioning.
    :param stats_file: Training mel mean and standard deviation.
    :returns: Four-second stereo waveforms and features on the 401-frame grid.
    """
    guide_waveform = _load_model_audio(guide_audio)
    ref_waveform = _load_model_audio(reference_audio)
    sketch_controls = extract_sketch_controls(guide_waveform, _SAMPLE_RATE)
    pitch = sketch_controls[SKETCH_PITCH_SLICE]
    sketch_controls[SKETCH_PITCH_SLICE] = pitch.where(pitch >= _PITCH_ZERO_THRESHOLD, 0.0)
    return PreparedAudioInputs(
        guide_audio=guide_waveform,
        reference_audio=ref_waveform,
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
    if values.shape != _EXPECTED_AUDIO_SHAPE:
        raise ValueError(f"output audio shape must be {_EXPECTED_AUDIO_SHAPE}, got {values.shape}")
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
    _write_wav(output_dir / "ref.wav", prepared.reference_audio)
    _write_wav(output_dir / "pred.wav", patch.audio)
    params_to_csv(
        None,
        None,
        patch.synth_params,
        patch.note_params,
        str(output_dir / "params.csv"),
        param_specs[_SURGE_PARAM_SPEC_NAME],
        pred_effective_note_window=patch.effective_note_window,
    )


def write_run_manifest(
    output_dir: Path,
    destination_uri: str,
    cfg_strengths: CfgStrengths[float],
) -> None:
    """Record immutable model, feature, render, and destination provenance.

    :param output_dir: Retained local output directory.
    :param destination_uri: R2 prefix receiving the run artifacts.
    :param cfg_strengths: Effective content and sketch guidance strengths.
    """
    synth_identity = SYNTHS[SynthName(_SURGE_PARAM_SPEC_NAME)]
    manifest = {
        "schema_version": 1,
        "run_id": output_dir.name,
        "r2_uri": destination_uri,
        "code_version": version("synth-setter"),
        "git_sha": resolve_git_sha(),
        "content_cfg_strength": cfg_strengths.content,
        "sketch_cfg_strength": cfg_strengths.sketch,
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


def retry_output_upload(output_dir: Path) -> str:
    """Validate and upload one retained sketch-render directory.

    :param output_dir: Existing directory from a failed upload attempt.
    :returns: Recorded R2 destination.
    :raises click.ClickException: The retained manifest or artifact set is invalid.
    """
    for filename in _RETAINED_ARTIFACT_FILENAMES:
        artifact = output_dir / filename
        if not artifact.is_file() or artifact.is_symlink():
            raise click.ClickException(f"missing retained artifact: {filename}")
    unexpected = sorted(
        path.name for path in output_dir.iterdir() if path.name not in _RETAINED_ARTIFACT_FILENAMES
    )
    if unexpected:
        raise click.ClickException(f"unexpected retained artifact: {unexpected[0]}")

    manifest_path = output_dir / "manifest.json"
    try:
        manifest = _RetainedRunManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError) as exc:
        raise click.ClickException("could not read manifest.json") from exc
    except ValidationError as exc:
        detail = str(exc.errors()[0]["msg"]).removeprefix("Value error, ")
        raise click.ClickException(f"invalid manifest.json: {detail}") from exc
    if manifest.run_id != output_dir.name:
        raise click.ClickException("manifest run_id must match output directory name")

    r2_io.ensure_r2_env_loaded()
    upload_output_artifacts(output_dir, manifest.r2_uri)
    return manifest.r2_uri


def _normalize_sketch_config(hyper_parameters: Mapping[str, object]) -> dict[str, object]:
    """Normalize legacy sketch metadata and enforce the serving contract.

    :param hyper_parameters: Checkpoint hyperparameter mapping.
    :returns: Current-form sketch configuration.
    :raises ValueError: Sketch controls are absent, ambiguous, or incompatible.
    """
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
    return normalized_sketch


def _validate_state_shapes(checkpoint: Mapping[str, object]) -> None:
    """Require state tensors that pin the serving architecture.

    :param checkpoint: Deserialized Lightning checkpoint mapping.
    :raises ValueError: The state mapping or a required tensor shape differs.
    """
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

    normalized_sketch = _normalize_sketch_config(hyper_parameters)
    _validate_state_shapes(checkpoint)
    return normalized_sketch


def _validate_stats(stats_file: Path) -> None:
    """Require the mel-stat arrays expected by the pinned training frontend.

    :param stats_file: Candidate ``stats.npz`` artifact.
    :raises ValueError: Mean/std shapes or values differ from the model input contract.
    """
    mean, std = load_mel_statistics(stats_file)
    expected_shape = (_CHANNELS, 128, 401)
    if mean.shape != expected_shape or std.shape != expected_shape:
        raise ValueError(f"stats mean/std must both have shape {expected_shape}")


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
    prepared: PreparedAudioInputs,
    model: VSTFlowMatchingModule,
    requested_strengths: CfgStrengths[float | None],
) -> tuple[dict[str, float], NoteParams, CfgStrengths[float]]:
    """Infer and decode one Surge patch through the model's predict step.

    :param prepared: Reference mel and guide controls.
    :param model: Compatible flow model in eval mode.
    :param requested_strengths: Optional content and sketch guidance overrides.
    :returns: Renderer parameters and effective guidance strengths.
    :raises ValueError: The prediction is non-finite or has the wrong shape.
    """
    batch = {
        "mel": prepared.ref_mel.unsqueeze(0).to(model.device),
        "sketch_ctrl": prepared.sketch_controls.unsqueeze(0).to(model.device),
    }
    torch.manual_seed(_SERVING_SEED)
    if model.device.type == "cuda":
        torch.cuda.manual_seed_all(_SERVING_SEED)
    with temporary_cfg_strength_overrides(
        model.hparams, requested_strengths
    ) as effective_strengths:
        with torch.no_grad():
            prediction, _ = model.predict_step(batch, 0)
    expected_shape = (1, _EXPECTED_PARAM_WIDTH)
    if tuple(prediction.shape) != expected_shape or not torch.isfinite(prediction).all():
        raise ValueError(
            f"model prediction must be finite with shape {expected_shape}, "
            f"got {tuple(prediction.shape)}"
        )
    synth_values, note_values = decode_model_output(
        prediction[0].detach().cpu().float().numpy(), param_specs[_SURGE_PARAM_SPEC_NAME]
    )
    synth_params = require_scalar_synth_params(synth_values)
    note_params = require_note_params(note_values)
    return synth_params, note_params, effective_strengths


def _canonical_note_params(note_params: NoteParams) -> tuple[NoteParams, tuple[float, float]]:
    """Preserve decoded notes while deriving the renderer-safe interval.

    :param note_params: Decoded MIDI pitch and note interval.
    :returns: Raw note parameters and canonical renderer interval.
    """
    decoded_note_window = note_params["note_start_and_end"]
    raw_note_window = (float(decoded_note_window[0]), float(decoded_note_window[1]))
    raw_note_params = note_params.copy()
    raw_note_params["note_start_and_end"] = raw_note_window
    effective_note_window = _canonicalize_prediction_note_window(
        raw_note_window,
        signal_duration_seconds=_DURATION_SECONDS,
        sample_rate=_SAMPLE_RATE,
    )
    return raw_note_params, effective_note_window


@contextmanager
def _surge_render_config() -> Iterator[RenderConfig]:
    """Validate Surge identity and yield its preset-backed render configuration.

    :yields RenderConfig: Preset-backed render configuration.
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
        yield RenderConfig(
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


def _render_patch(synth_params: dict[str, float], note_params: NoteParams) -> RenderedPatch:
    """Render decoded parameters through the production Surge VST backend.

    :param synth_params: Renderer-native Surge parameter values.
    :param note_params: Decoded MIDI pitch and note interval.
    :returns: Rendered stereo patch and its parameters.
    """
    raw_note_params, effective_note_window = _canonical_note_params(note_params)
    with _surge_render_config() as render_config:
        renderer = make_audio_renderer(render_config)
        audio = renderer.render(
            synth_params,
            int(raw_note_params["pitch"]),
            render_config.velocity,
            effective_note_window,
            warmup=True,
        )
    return RenderedPatch(
        audio=audio,
        synth_params=synth_params,
        note_params=raw_note_params,
        effective_note_window=effective_note_window,
    )


def _artifact_roots() -> tuple[Path, str]:
    """Resolve operator-overridable local and R2 artifact roots.

    :returns: Local output root and validated R2 upload prefix.
    :raises ValueError: The configured upload prefix is not an R2 URI.
    """
    output_root = Path(os.environ.get(_OUTPUT_ROOT_ENV, str(_OUTPUT_ROOT))).expanduser()
    upload_root = os.environ.get(_R2_OUTPUT_ROOT_ENV, _R2_OUTPUT_ROOT).rstrip("/")
    if not r2_io.is_r2_uri(upload_root):
        raise ValueError(f"{_R2_OUTPUT_ROOT_ENV} must use r2://")
    return output_root, upload_root


def _run_request(
    guide_audio: Path,
    reference_audio: Path,
    requested_strengths: CfgStrengths[float | None],
) -> tuple[Path, str]:
    """Run one complete local render and R2 upload.

    :param guide_audio: Audio supplying sketch controls.
    :param reference_audio: Audio supplying mel conditioning.
    :param requested_strengths: Optional content and sketch guidance overrides.
    :returns: Retained local output path and uploaded R2 prefix.
    """
    r2_io.ensure_r2_env_loaded()
    checkpoint_path = cache_r2_file(DEFAULT_CHECKPOINT_URI, _CACHE_NAMESPACE, _CHECKPOINT_SHA256)
    stats_path = cache_r2_file(DEFAULT_STATS_URI, _CACHE_NAMESPACE, _STATS_SHA256)
    _validate_stats(stats_path)

    prepared = prepare_audio_inputs(guide_audio, reference_audio, stats_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint_path, device)
    synth_params, note_params, effective_strengths = _predict_patch(
        prepared, model, requested_strengths
    )
    patch = _render_patch(synth_params, note_params)

    run_id = f"{make_wandb_run_id('synth-setter-sketch-render')}-{uuid4().hex[:8]}"
    output_root, upload_root = _artifact_roots()
    output_dir = output_root / run_id
    destination_uri = f"{upload_root}/{run_id}"
    write_output_artifacts(output_dir, prepared, patch)
    write_run_manifest(output_dir, destination_uri, effective_strengths)
    resolved_output = output_dir.resolve()
    click.echo(f"Local output: {resolved_output}", err=True)
    upload_output_artifacts(output_dir, destination_uri)
    return resolved_output, destination_uri


def _run_under_headless_wrapper(
    guide_audio: Path,
    reference_audio: Path,
    requested_strengths: CfgStrengths[float | None],
) -> None:
    """Re-enter the public module under the packaged Linux X11 wrapper.

    :param guide_audio: Audio supplying sketch controls.
    :param reference_audio: Audio supplying mel conditioning.
    :param requested_strengths: Optional content and sketch guidance overrides.
    """
    with as_file(vst_headless_wrapper()) as wrapper:
        command = [
            "/bin/bash",
            str(wrapper),
            sys.executable,
            "-m",
            "synth_setter.cli.sketch_render",
            "--guide-audio",
            str(guide_audio.resolve()),
            "--reference-audio",
            str(reference_audio.resolve()),
        ]
        if requested_strengths.content is not None:
            command.extend(["--content-cfg-strength", str(requested_strengths.content)])
        if requested_strengths.sketch is not None:
            command.extend(["--sketch-cfg-strength", str(requested_strengths.sketch)])
        subprocess.run(  # noqa: S603 — fixed package entrypoint and validated paths
            command,
            check=True,
            env={**os.environ, _HEADLESS_ENV: "1"},
            timeout=_HEADLESS_TIMEOUT_SECONDS,
        )


@click.command(help="Infer, render, retain, and upload one Surge patch.")
@click.option(
    "--guide-audio",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Guide audio supplying sketch controls.",
)
@click.option(
    "--reference-audio",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Reference audio supplying mel/timbre conditioning.",
)
@click.option(
    "--content-cfg-strength",
    type=float,
    callback=validate_cfg_strength,
    help="Reference-mel guidance override; omitted uses the checkpoint value.",
)
@click.option(
    "--sketch-cfg-strength",
    type=float,
    callback=validate_cfg_strength,
    help="Sketch-control guidance override; omitted uses the checkpoint value.",
)
@click.option(
    "--retry-upload",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Upload a retained sketch-render directory without rerunning inference.",
)
@click.pass_context
def main(
    context: click.Context,
    guide_audio: Path | None,
    reference_audio: Path | None,
    content_cfg_strength: float | None,
    sketch_cfg_strength: float | None,
    retry_upload: Path | None,
) -> None:
    """Infer, render, retain, and upload one Surge patch.

    :param context: Active Click invocation context.
    :param guide_audio: Optional audio supplying sketch controls.
    :param reference_audio: Optional audio supplying mel/timbre conditioning.
    :param content_cfg_strength: Optional reference-mel guidance override.
    :param sketch_cfg_strength: Optional sketch-control guidance override.
    :param retry_upload: Optional retained sketch-render directory to upload.
    :raises click.ClickException: Retry and render inputs conflict or are incomplete.
    """
    if retry_upload is not None:
        retry_conflicts = (
            ("guide_audio", "--guide-audio"),
            ("reference_audio", "--reference-audio"),
            ("content_cfg_strength", "--content-cfg-strength"),
            ("sketch_cfg_strength", "--sketch-cfg-strength"),
        )
        for parameter, option in retry_conflicts:
            if context.get_parameter_source(parameter) is ParameterSource.COMMANDLINE:
                raise click.ClickException(f"{option} cannot be combined with --retry-upload")
        click.echo(retry_output_upload(retry_upload))
        return

    if guide_audio is None or reference_audio is None:
        raise click.ClickException("--guide-audio and --reference-audio must be provided together")

    requested_strengths = CfgStrengths(
        content=content_cfg_strength,
        sketch=sketch_cfg_strength,
    )
    if sys.platform == "linux" and os.environ.get(_HEADLESS_ENV) != "1":
        _run_under_headless_wrapper(guide_audio, reference_audio, requested_strengths)
        return

    _, destination_uri = _run_request(guide_audio, reference_audio, requested_strengths)
    click.echo(destination_uri)


if __name__ == "__main__":
    main()
