"""Render a Surge patch from content timbre and sketch controls.

Run ``synth-setter-sketch-render --sketch-corpus nsynth_test --content-corpus esc50``.
Checkpoint rotation requires updating the pinned digest and serving identity together.
"""

import hashlib
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Literal, cast
from uuid import uuid4

import click
import numpy as np
import torch
from click.core import ParameterSource
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from synth_setter.cli._cfg_strength import validate_cfg_strength
from synth_setter.conditioning import SketchControlSpec
from synth_setter.data.audio_datamodule import load_audio_file_to_grid
from synth_setter.data.third_party_datamodule import ThirdPartyAudioDataModule
from synth_setter.data.vst.core import extract_renderer_version, write_wav
from synth_setter.data.vst.param_spec_registry import default_plugin_path, param_specs
from synth_setter.data.vst_datamodule import (
    PreparedAudioInputs,
    load_mel_statistics,
    prepare_paired_audio_inputs,
)
from synth_setter.evaluation.predict_vst_audio import (
    RenderedPrediction,
    make_prediction_render_fn,
    params_to_csv,
    render_prediction_row,
)
from synth_setter.model_cache import cache_r2_file
from synth_setter.models.cfg import CfgStrengths
from synth_setter.models.vst_flow_matching_module import (
    InferenceRequirements,
    VSTFlowMatchingModule,
)
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
_DEFAULT_SEED = 0
_HEADLESS_TIMEOUT_SECONDS = 1200
_SURGE_PARAM_SPEC_NAME = "surge_simple"
_EXPECTED_PARAM_WIDTH = param_specs[_SURGE_PARAM_SPEC_NAME].encoded_width
_MIN_LOUDNESS_DB = -55.0
_EXPECTED_SKETCH_SPEC = SketchControlSpec(num_frames=401)
_RETAINED_ARTIFACT_FILENAMES = (
    "guide.wav",
    "manifest.json",
    "params.csv",
    "pred.wav",
    "ref.wav",
)
CorpusName = Literal["esc50", "nsynth_test"]
StreamName = Literal["sketch", "content"]
_CORPUS_NAMES: tuple[CorpusName, ...] = ("esc50", "nsynth_test")


type _InputSource = Path | CorpusName


@dataclass(frozen=True, kw_only=True)
class _RenderRequest:
    sketch: _InputSource
    content: _InputSource
    strengths: CfgStrengths[float | None]
    seed: int


def _compose_corpus_datamodule(config_name: str) -> ThirdPartyAudioDataModule:
    """Compose one checked-in corpus onto the CLI's fixed audio grid.

    :param config_name: Public corpus config name without its Hydra group prefix.
    :returns: Configured, un-setup third-party datamodule.
    :raises ValueError: The config name or composed target is unsupported.
    """
    if config_name not in _CORPUS_NAMES:
        raise ValueError(f"unknown corpus config {config_name!r}")
    overrides = [
        f"+datamodule=third_party/{config_name}",
        f"datamodule.sample_rate={_SAMPLE_RATE}",
        f"datamodule.channels={_CHANNELS}",
        f"datamodule.signal_duration_seconds={_DURATION_SECONDS}",
        "datamodule.use_saved_mean_and_variance=false",
        "datamodule.mel_stats_uri=null",
        "datamodule.stats_cache_dir=null",
        "datamodule.num_workers=0",
    ]
    with initialize_config_module(config_module="synth_setter.configs", version_base="1.3"):
        cfg = compose(overrides=overrides)
    datamodule = instantiate(cfg.datamodule)
    if not isinstance(datamodule, ThirdPartyAudioDataModule):
        raise ValueError(f"corpus config {config_name!r} did not compose a third-party datamodule")
    return datamodule


def _selection_rng(seed: int, domain: str) -> np.random.Generator:
    """Build a domain-separated local generator for one selection decision.

    :param seed: Public render seed.
    :param domain: Stable stream and corpus identity.
    :returns: Local generator that does not mutate process RNG state.
    """
    digest = hashlib.sha256(f"{seed}:{domain}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], byteorder="big"))


def _select_corpus_rows(
    request: _RenderRequest, row_counts: Mapping[str, int]
) -> dict[StreamName, int]:
    """Select deterministic explicit rows without touching process RNG state.

    :param request: Validated source selectors and effective seed.
    :param row_counts: Served row count keyed by selected corpus name.
    :returns: Selected row keyed by input stream.
    :raises ValueError: A selected corpus is empty or shared with fewer than two rows.
    """
    sources: dict[StreamName, _InputSource] = {
        "sketch": request.sketch,
        "content": request.content,
    }
    if isinstance(request.sketch, str) and request.sketch == request.content:
        row_count = row_counts[request.sketch]
        if row_count < 2:
            raise ValueError(
                f"corpus {request.sketch!r} needs at least two rows when used for both inputs"
            )
        rows = _selection_rng(request.seed, f"shared:{request.sketch}").choice(
            row_count, size=2, replace=False
        )
        return {"sketch": int(rows[0]), "content": int(rows[1])}

    selected: dict[StreamName, int] = {}
    for stream, source in sources.items():
        if isinstance(source, Path):
            continue
        row_count = row_counts[source]
        if row_count < 1:
            raise ValueError(f"corpus {source!r} has no served rows")
        selected[stream] = int(
            _selection_rng(request.seed, f"{stream}:{source}").integers(row_count)
        )
    return selected


class _FileInputProvenance(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    kind: Literal["file"] = "file"
    path: str = Field(min_length=1)
    sha256: str

    @field_validator("path")
    @classmethod
    def _path_is_absolute(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("file provenance path must be absolute")
        return value

    @field_validator("sha256")
    @classmethod
    def _sha256_is_valid(cls, value: str) -> str:
        if len(value) != 64:
            raise ValueError("file provenance SHA-256 must contain 64 hex characters")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError("file provenance SHA-256 must be hexadecimal") from exc
        return value


class _CorpusInputProvenance(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    kind: Literal["corpus"] = "corpus"
    config_name: CorpusName
    dataset_uri: str = Field(min_length=1)
    dataset_version: int = Field(gt=0)
    row_index: int = Field(ge=0)


_InputProvenance = Annotated[
    _FileInputProvenance | _CorpusInputProvenance,
    Field(discriminator="kind"),
]


type _ResolvedInputs = dict[
    StreamName,
    tuple[Float[torch.Tensor, "channels samples"], _InputProvenance],
]


def _resolve_request_inputs(request: _RenderRequest) -> _ResolvedInputs:
    """Decode selected files or corpus rows onto the fixed model grid.

    :param request: Validated source selectors and selection seed.
    :returns: Audio and strict provenance keyed by input stream.
    """
    r2_io.ensure_r2_env_loaded()
    sources: dict[StreamName, _InputSource] = {
        "sketch": request.sketch,
        "content": request.content,
    }
    corpus_names: list[CorpusName] = sorted(
        {cast(CorpusName, source) for source in sources.values() if isinstance(source, str)}
    )
    datamodules: dict[CorpusName, ThirdPartyAudioDataModule] = {}
    for config_name in corpus_names:
        datamodule = _compose_corpus_datamodule(config_name)
        datamodule.setup("predict")
        datamodules[config_name] = datamodule
    selected = _select_corpus_rows(
        request,
        {name: datamodule.served_row_count for name, datamodule in datamodules.items()},
    )

    resolved: _ResolvedInputs = {}
    for stream_name, source in sources.items():
        if isinstance(source, Path):
            resolved_path = source.resolve()
            with resolved_path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            audio = load_audio_file_to_grid(
                resolved_path,
                segment_length_seconds=_DURATION_SECONDS,
                leading_padding_seconds=0.0,
                amp_scale=1.0,
                sample_rate=_SAMPLE_RATE,
            )
            provenance: _InputProvenance = _FileInputProvenance(
                path=str(resolved_path), sha256=digest
            )
        else:
            config_name = cast(CorpusName, source)
            row_index = selected[stream_name]
            datamodule = datamodules[config_name]
            audio = datamodule.audio_rows([row_index])[0]
            provenance = _CorpusInputProvenance(
                config_name=config_name,
                dataset_uri=datamodule.dataset_uri,
                dataset_version=datamodule.resolved_dataset_version,
                row_index=row_index,
            )
        resolved[stream_name] = (audio, provenance)
    return resolved


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

        Effective content guidance strength.

    .. attribute :: sketch_cfg_strength

        Effective sketch guidance strength.

    .. attribute :: checkpoint

        Inverse checkpoint identity.

    .. attribute :: stats

        Normalization-statistics identity.

    .. attribute :: sketch_input

        Sketch source identity.

    .. attribute :: content_input

        Content source identity.

    .. attribute :: render

        Renderer identity and settings.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    schema_version: Literal[2]
    run_id: str = Field(min_length=1)
    r2_uri: str
    code_version: str = Field(min_length=1)
    git_sha: str = Field(min_length=1)
    content_cfg_strength: float = Field(ge=0, allow_inf_nan=False)
    sketch_cfg_strength: float = Field(ge=0, allow_inf_nan=False)
    checkpoint: _ManifestArtifact
    stats: _ManifestArtifact
    sketch_input: _InputProvenance
    content_input: _InputProvenance
    render: _ManifestRender

    @field_validator("r2_uri")
    @classmethod
    def _r2_uri_is_valid(cls, value: str) -> str:
        if not r2_io.is_r2_uri(value):
            raise ValueError("manifest r2_uri must use r2://")
        return value


def write_output_artifacts(
    output_dir: Path, prepared: PreparedAudioInputs, patch: RenderedPrediction
) -> None:
    """Persist normalized inputs, rendered audio, and decoded patch parameters.

    :param output_dir: Retained local output directory.
    :param prepared: Normalized sketch/content audio and model features.
    :param patch: Rendered prediction and decoded parameters.
    :raises ValueError: An artifact does not match the fixed model audio shape.
    """
    output_dir.mkdir(parents=True, exist_ok=False)
    audio_artifacts = (
        ("guide.wav", prepared.guide_audio.detach().cpu().numpy()),
        ("ref.wav", prepared.reference_audio.detach().cpu().numpy()),
        ("pred.wav", patch.audio),
    )
    for filename, audio in audio_artifacts:
        if audio.shape != _EXPECTED_AUDIO_SHAPE:
            raise ValueError(
                f"output audio shape must be {_EXPECTED_AUDIO_SHAPE}, got {audio.shape}"
            )
        write_wav(
            audio,
            output_dir / filename,
            sample_rate=_SAMPLE_RATE,
            channels=_CHANNELS,
        )
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
    *,
    request: _RenderRequest,
    provenance: Mapping[StreamName, _InputProvenance],
) -> None:
    """Record immutable input, model, render, and destination provenance.

    :param output_dir: Retained local output directory.
    :param destination_uri: R2 prefix receiving the run artifacts.
    :param request: Source selectors and effective inference policy.
    :param provenance: Strict source identities keyed by input stream.
    :raises ValueError: Effective guidance strengths are unavailable.
    """
    content_strength = request.strengths.content
    sketch_strength = request.strengths.sketch
    if content_strength is None or sketch_strength is None:
        raise ValueError("manifest requires effective content and sketch guidance strengths")

    synth_identity = SYNTHS[SynthName(_SURGE_PARAM_SPEC_NAME)]
    manifest = _RetainedRunManifest(
        schema_version=2,
        run_id=output_dir.name,
        r2_uri=destination_uri,
        code_version=version("synth-setter"),
        git_sha=resolve_git_sha(),
        content_cfg_strength=content_strength,
        sketch_cfg_strength=sketch_strength,
        checkpoint=_ManifestArtifact(uri=DEFAULT_CHECKPOINT_URI, sha256=_CHECKPOINT_SHA256),
        stats=_ManifestArtifact(uri=DEFAULT_STATS_URI, sha256=_STATS_SHA256),
        sketch_input=provenance["sketch"],
        content_input=provenance["content"],
        render=_ManifestRender(
            param_spec=_SURGE_PARAM_SPEC_NAME,
            synth_version=synth_identity.synth_version,
            sample_rate=_SAMPLE_RATE,
            channels=_CHANNELS,
            duration_seconds=_DURATION_SECONDS,
            seed=request.seed,
        ),
    )
    temporary_path = output_dir / ".manifest.json.tmp"
    temporary_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
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


def _validate_stats(stats_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load mel statistics matching the pinned training frontend.

    :param stats_file: Candidate ``stats.npz`` artifact.
    :returns: Validated mean and standard-deviation arrays.
    :raises ValueError: Mean/std shapes or values differ from the model input contract.
    """
    mean, std = load_mel_statistics(stats_file)
    expected_shape = (_CHANNELS, 128, 401)
    if mean.shape != expected_shape or std.shape != expected_shape:
        raise ValueError(f"stats mean/std must both have shape {expected_shape}")
    return mean, std


def _load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[VSTFlowMatchingModule, SketchControlSpec]:
    """Validate and load the pinned flow checkpoint on ``device``.

    :param checkpoint_path: Digest-pinned Lightning checkpoint.
    :param device: Inference device.
    :returns: Compatible model in eval mode and its resolved sketch spec.
    :raises ValueError: The checkpoint root or compatibility contract is invalid.
    """
    model = VSTFlowMatchingModule.load_for_inference(checkpoint_path, map_location=device)
    model.require_inference_capabilities(
        InferenceRequirements(
            conditioning="mel",
            sketch_controls=_EXPECTED_SKETCH_SPEC,
            param_spec=_SURGE_PARAM_SPEC_NAME,
            num_params=_EXPECTED_PARAM_WIDTH,
        )
    )
    sketch_spec = model.sketch_control_spec
    if sketch_spec is None:
        raise ValueError("checkpoint must support sketch controls")
    model.to(device)
    model.eval()
    return model, sketch_spec


def _predict_patch(
    prepared: PreparedAudioInputs,
    model: VSTFlowMatchingModule,
    requested_strengths: CfgStrengths[float | None],
    *,
    seed: int,
) -> tuple[np.ndarray, CfgStrengths[float]]:
    """Infer one model-space Surge prediction row.

    :param prepared: Content mel and sketch controls.
    :param model: Compatible flow model in eval mode.
    :param requested_strengths: Optional content and sketch guidance overrides.
    :param seed: Local model-sampling seed.
    :returns: CPU model-space row and effective guidance strengths.
    :raises ValueError: The prediction is non-finite or has the wrong shape.
    """
    batch = {
        "mel": prepared.ref_mel.unsqueeze(0).to(model.device),
        "sketch_ctrl": prepared.sketch_controls.unsqueeze(0).to(model.device),
    }
    generator = torch.Generator(device=model.device).manual_seed(seed)
    with torch.no_grad():
        sampled = model.sample_batch(
            batch,
            generator=generator,
            strengths=requested_strengths,
        )
    prediction = sampled.predictions
    effective_strengths = sampled.strengths
    expected_shape = (1, _EXPECTED_PARAM_WIDTH)
    if prediction.dtype is not torch.float32:
        raise ValueError(f"model prediction dtype must be torch.float32, got {prediction.dtype}")
    if tuple(prediction.shape) != expected_shape or not torch.isfinite(prediction).all():
        raise ValueError(
            f"model prediction must be finite with shape {expected_shape}, "
            f"got {tuple(prediction.shape)}"
        )
    return prediction[0].detach().cpu().float().numpy(), effective_strengths


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


def _run_request(request: _RenderRequest) -> tuple[Path, str]:
    """Run one complete local render and R2 upload.

    :param request: Validated source selectors, guidance strengths, and seed.
    :returns: Retained local output path and uploaded R2 prefix.
    """
    resolved_inputs = _resolve_request_inputs(request)
    sketch_audio, sketch_provenance = resolved_inputs["sketch"]
    content_audio, content_provenance = resolved_inputs["content"]
    checkpoint_path = cache_r2_file(DEFAULT_CHECKPOINT_URI, _CACHE_NAMESPACE, _CHECKPOINT_SHA256)
    stats_path = cache_r2_file(DEFAULT_STATS_URI, _CACHE_NAMESPACE, _STATS_SHA256)
    mean, std = _validate_stats(stats_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sketch_spec = _load_model(checkpoint_path, device)
    prepared = prepare_paired_audio_inputs(
        sketch_audio,
        content_audio,
        sample_rate=_SAMPLE_RATE,
        mean=mean,
        std=std,
        sketch_spec=sketch_spec,
    )
    prediction_row, effective_strengths = _predict_patch(
        prepared,
        model,
        request.strengths,
        seed=request.seed,
    )
    with _surge_render_config() as render_config:
        renderer = make_audio_renderer(render_config)
        patch = render_prediction_row(
            prediction_row,
            param_specs[_SURGE_PARAM_SPEC_NAME],
            make_prediction_render_fn(render_config, renderer),
            signal_duration_seconds=render_config.signal_duration_seconds,
            sample_rate=render_config.sample_rate,
        )

    run_id = f"{make_wandb_run_id('synth-setter-sketch-render')}-{uuid4().hex[:8]}"
    output_root, upload_root = _artifact_roots()
    output_dir = output_root / run_id
    destination_uri = f"{upload_root}/{run_id}"
    write_output_artifacts(output_dir, prepared, patch)
    write_run_manifest(
        output_dir,
        destination_uri,
        request=replace(request, strengths=effective_strengths),
        provenance={
            "sketch": sketch_provenance,
            "content": content_provenance,
        },
    )
    resolved_output = output_dir.resolve()
    click.echo(f"Local output: {resolved_output}", err=True)
    upload_output_artifacts(output_dir, destination_uri)
    return resolved_output, destination_uri


def _run_under_headless_wrapper(request: _RenderRequest) -> None:
    """Re-enter the public module under the packaged Linux X11 wrapper.

    :param request: Validated source selectors, guidance strengths, and seed.
    """
    with as_file(vst_headless_wrapper()) as wrapper:
        command = [
            "/bin/bash",
            str(wrapper),
            sys.executable,
            "-m",
            "synth_setter.cli.sketch_render",
        ]
        for name, source in (("sketch", request.sketch), ("content", request.content)):
            if isinstance(source, Path):
                command.extend([f"--{name}-audio", str(source.resolve())])
            else:
                command.extend([f"--{name}-corpus", source])
        command.extend(["--seed", str(request.seed)])
        if request.strengths.content is not None:
            command.extend(["--content-cfg-strength", str(request.strengths.content)])
        if request.strengths.sketch is not None:
            command.extend(["--sketch-cfg-strength", str(request.strengths.sketch)])
        subprocess.run(  # noqa: S603 — fixed package entrypoint and validated paths
            command,
            check=True,
            env={**os.environ, _HEADLESS_ENV: "1"},
            timeout=_HEADLESS_TIMEOUT_SECONDS,
        )


def _require_input_source(
    stream: StreamName,
    audio: Path | None,
    corpus: CorpusName | None,
) -> _InputSource:
    """Require exactly one file or corpus source for an input stream.

    :param stream: Input stream used in the CLI error.
    :param audio: Optional local audio source.
    :param corpus: Optional checked-in corpus source.
    :returns: The selected source.
    :raises click.ClickException: Neither or both source forms were supplied.
    """
    if (audio is None) == (corpus is None):
        raise click.ClickException(
            f"exactly one of --{stream}-audio or --{stream}-corpus is required"
        )
    return audio if audio is not None else cast(CorpusName, corpus)


@click.command(help="Infer, render, retain, and upload one Surge patch.")
@click.option(
    "--sketch-audio",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="File supplying sketch controls.",
)
@click.option(
    "--sketch-corpus",
    type=click.Choice(_CORPUS_NAMES),
    help="Checked-in corpus supplying sketch controls.",
)
@click.option(
    "--content-audio",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="File supplying mel/timbre conditioning.",
)
@click.option(
    "--content-corpus",
    type=click.Choice(_CORPUS_NAMES),
    help="Checked-in corpus supplying mel/timbre conditioning.",
)
@click.option(
    "--seed",
    type=int,
    default=_DEFAULT_SEED,
    show_default=True,
    help="Corpus selection and model sampling seed.",
)
@click.option(
    "--content-cfg-strength",
    type=float,
    callback=validate_cfg_strength,
    help="Content-mel guidance override; omitted uses the checkpoint value.",
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
    sketch_audio: Path | None,
    sketch_corpus: CorpusName | None,
    content_audio: Path | None,
    content_corpus: CorpusName | None,
    seed: int,
    content_cfg_strength: float | None,
    sketch_cfg_strength: float | None,
    retry_upload: Path | None,
) -> None:
    """Infer, render, retain, and upload one Surge patch.

    :param context: Active Click invocation context.
    :param sketch_audio: Optional file supplying sketch controls.
    :param sketch_corpus: Optional corpus supplying sketch controls.
    :param content_audio: Optional file supplying mel/timbre conditioning.
    :param content_corpus: Optional corpus supplying mel/timbre conditioning.
    :param seed: Corpus selection and model sampling seed.
    :param content_cfg_strength: Optional content-mel guidance override.
    :param sketch_cfg_strength: Optional sketch-control guidance override.
    :param retry_upload: Optional retained sketch-render directory to upload.
    :raises click.ClickException: Retry and render inputs conflict or are incomplete.
    """
    if retry_upload is not None:
        retry_conflicts = (
            ("content_audio", "--content-audio"),
            ("content_corpus", "--content-corpus"),
            ("content_cfg_strength", "--content-cfg-strength"),
            ("seed", "--seed"),
            ("sketch_audio", "--sketch-audio"),
            ("sketch_corpus", "--sketch-corpus"),
            ("sketch_cfg_strength", "--sketch-cfg-strength"),
        )
        for parameter, option in retry_conflicts:
            if context.get_parameter_source(parameter) is ParameterSource.COMMANDLINE:
                raise click.ClickException(f"{option} cannot be combined with --retry-upload")
        click.echo(retry_output_upload(retry_upload))
        return

    request = _RenderRequest(
        sketch=_require_input_source("sketch", sketch_audio, sketch_corpus),
        content=_require_input_source("content", content_audio, content_corpus),
        strengths=CfgStrengths(
            content=content_cfg_strength,
            sketch=sketch_cfg_strength,
        ),
        seed=seed,
    )
    try:
        if sys.platform == "linux" and os.environ.get(_HEADLESS_ENV) != "1":
            _run_under_headless_wrapper(request)
            return

        _, destination_uri = _run_request(request)
    except click.ClickException:
        raise
    except Exception as exc:
        detail = str(exc).strip() or type(exc).__name__
        raise click.ClickException(detail) from exc
    click.echo(destination_uri)


if __name__ == "__main__":
    main()
