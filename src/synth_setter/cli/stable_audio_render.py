"""Render a Surge patch from a Stable Audio-generated SAME latent."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Protocol, cast

import click
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator

from synth_setter.cli.clap_render import (
    _predict_patch as predict_patch,
)
from synth_setter.cli.clap_render import (
    _render_wav as render_wav,
)
from synth_setter.cli.clap_render import (
    _resolve_device as resolve_device,
)
from synth_setter.cli.clap_render import (
    _workspace_render_config as workspace_render_config,
)
from synth_setter.cli.clap_render import (
    resolve_inverse_checkpoint,
)
from synth_setter.conditioning import (
    Conditioning,
    EmbeddingConditioningSpec,
    resolve_embedding_conditioning,
)
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.same import (
    SAME_EMBEDDING_DIM,
    SAME_SAMPLE_RATE,
    same_l_num_latent_frames,
)
from synth_setter.workspace import operator_workspace

_DeviceSetting = Literal["auto", "cpu", "cuda", "mps"]
_ModelSelector = Literal["small", "medium"]
_DEFAULT_CONFIG_NAME = "stable_audio_render"
_RENDER_DURATION_SECONDS = 4.0
_EXPECTED_LATENT_FRAMES = same_l_num_latent_frames(
    int(_RENDER_DURATION_SECONDS * SAME_SAMPLE_RATE), SAME_SAMPLE_RATE
)
_EXPECTED_LATENT_SHAPE = (1, SAME_EMBEDDING_DIM, _EXPECTED_LATENT_FRAMES)
_GENERATION_STEPS = 8
_CFG_SCALE = 1.0
_PROVENANCE_FIELDS: tuple[str, ...] = (
    "prompt",
    "model",
    "stable_audio_model",
    "model_repo",
    "model_revision",
    "conditioning",
    "latent_shape",
    "latent_norm",
    "inverse_checkpoint",
    "inverse_checkpoint_sha256",
    "seed",
    "wav_r2_uri",
    "csv_r2_uri",
)


class StableAudioProfile(BaseModel):
    """Bind one text model to its compatible SAME-conditioned inverse.

    .. attribute :: model_config

        Strict immutable Pydantic settings validation.

    .. attribute :: model_name

        Stable Audio package model selector.

    .. attribute :: repo_id

        Hugging Face repository containing the model snapshot.

    .. attribute :: revision

        Immutable Hugging Face commit revision.

    .. attribute :: conditioning

        SAME column and latent shape required by the inverse checkpoint.

    .. attribute :: inverse_checkpoint

        Default local path or R2 URI for inverse inference.

    .. attribute :: inverse_checkpoint_sha256

        Required digest for the default inverse checkpoint.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    model_name: str = Field(min_length=1)
    repo_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    conditioning: EmbeddingConditioningSpec
    inverse_checkpoint: str = Field(min_length=1)
    inverse_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("conditioning", mode="before")
    @classmethod
    def _parse_conditioning(cls, value: object) -> EmbeddingConditioningSpec:
        """Convert Hydra's shape list into the immutable embedding contract.

        :param value: Candidate conditioning mapping.
        :returns: Validated embedding conditioning.
        :raises ValueError: The value names raw rather than embedding conditioning.
        """
        conditioning = resolve_embedding_conditioning(cast(Conditioning, value))
        if conditioning is None:
            raise ValueError("Stable Audio profiles require embedding conditioning")
        return conditioning


class _StableAudioRenderSettings(BaseModel):
    """Validate model profiles and common render defaults.

    .. attribute :: model_config

        Strict immutable Pydantic settings validation.

    .. attribute :: profiles

        User-facing model selectors and their complete inference identities.

    .. attribute :: output_dir

        Default directory for local WAV and CSV artifacts.

    .. attribute :: upload_prefix

        R2 prefix receiving generated artifacts.

    .. attribute :: device

        Requested inference device or automatic selection.

    .. attribute :: seed

        Shared Stable Audio and inverse-model sampling seed.

    .. attribute :: render

        Surge renderer identity and audio settings.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    profiles: dict[str, StableAudioProfile]
    output_dir: Path
    upload_prefix: str
    device: _DeviceSetting
    seed: int
    render: RenderConfig

    @field_validator("output_dir", mode="before")
    @classmethod
    def _parse_output_dir(cls, value: object) -> Path:
        """Parse the YAML path while retaining strict validation elsewhere.

        :param value: Candidate path from Hydra composition.
        :returns: Filesystem path used for local output.
        :raises TypeError: The value is neither text nor a path.
        """
        if not isinstance(value, (str, Path)):
            raise TypeError("output_dir must be a filesystem path")
        return Path(value)

    @field_validator("upload_prefix")
    @classmethod
    def _upload_prefix_is_r2(cls, value: str) -> str:
        """Normalize a validated R2 upload prefix.

        :param value: Candidate upload prefix.
        :returns: R2 prefix without a trailing slash.
        :raises ValueError: The prefix does not use the R2 URI scheme.
        """
        if not r2_io.is_r2_uri(value):
            raise ValueError("upload_prefix must use r2://")
        return value.rstrip("/")


class _StableAudioLatentModel(Protocol):
    """Expose the Stable Audio surface required by the CLI."""

    @property
    def model_config(self) -> Mapping[str, object]:
        """Return the source model configuration."""
        ...

    def generate(
        self,
        *,
        prompt: str,
        duration: float,
        steps: int,
        cfg_scale: float,
        batch_size: int,
        sample_size: int,
        duration_padding_sec: float,
        return_latents: bool,
        seed: int,
    ) -> torch.Tensor:
        """Sample one latent batch in the selected autoencoder space.

        :param prompt: Natural-language audio description.
        :param duration: Requested content duration in seconds.
        :param steps: Diffusion sampler step count.
        :param cfg_scale: Classifier-free guidance scale.
        :param batch_size: Number of latents to sample.
        :param sample_size: Source model's maximum audio sample budget.
        :param duration_padding_sec: Extra duration used during latent sizing.
        :param return_latents: Whether to skip waveform decoding.
        :param seed: Diffusion noise seed.
        :returns: Generated latent batch.
        """
        ...


def _load_settings() -> _StableAudioRenderSettings:
    """Compose strict Stable Audio and Surge render settings.

    :returns: Validated model profiles and render defaults.
    :raises TypeError: The composed root, render, or synth node is not a mapping.
    """
    with initialize_config_module(config_module="synth_setter.configs", version_base="1.3"):
        cfg = compose(config_name=_DEFAULT_CONFIG_NAME)
    values = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("stable_audio_render config must resolve to a mapping")
    render_values = values.pop("render")
    synth_values = values.pop("synth")
    if not isinstance(render_values, dict) or not isinstance(synth_values, dict):
        raise TypeError("stable_audio_render render and synth nodes must resolve to mappings")
    render_values["synth"] = synth_values
    values["render"] = RenderConfig.model_validate(render_values)
    return _StableAudioRenderSettings.model_validate(values)


def load_profile(selector: _ModelSelector) -> StableAudioProfile:
    """Return the immutable text-model and inverse-checkpoint pairing.

    :param selector: User-facing Small or Medium profile.
    :returns: Stable Audio and SAME-conditioned inverse identity.
    """
    return _load_settings().profiles[selector]


def _localize_prompt_conditioner(model_config: dict[str, object], snapshot: Path) -> None:
    """Pin the nested T5Gemma loader to the same immutable local snapshot.

    :param model_config: Parsed Stable Audio model configuration.
    :param snapshot: Immutable local Hugging Face snapshot.
    :raises ValueError: The config has no unique prompt conditioner.
    """
    model = model_config.get("model")
    if not isinstance(model, dict):
        raise ValueError("Stable Audio config has no model mapping")
    conditioning = model.get("conditioning")
    if not isinstance(conditioning, dict):
        raise ValueError("Stable Audio config has no conditioning mapping")
    configs = conditioning.get("configs")
    if not isinstance(configs, list):
        raise ValueError("Stable Audio config has no conditioning config list")
    prompts = [
        entry for entry in configs if isinstance(entry, dict) and entry.get("id") == "prompt"
    ]
    if len(prompts) != 1:
        raise ValueError("Stable Audio config must contain one prompt conditioner")
    prompt_config = prompts[0].get("config")
    if not isinstance(prompt_config, dict):
        raise ValueError("Stable Audio prompt conditioner has no config mapping")
    prompt_config["repo_id"] = str(snapshot)


def _checkpoint_target_key(source_key: str, target_keys: set[str]) -> str | None:
    """Resolve the key remapping accepted by Stable Audio's upstream loader.

    :param source_key: Checkpoint tensor name.
    :param target_keys: Names in the constructed model state.
    :returns: Matching model key, or ``None`` when no key matches.
    """
    if source_key in target_keys:
        return source_key
    parts = source_key.split(".")
    for index in range(1, len(parts)):
        candidate = ".".join(parts[:index]) + "." + ".".join(parts[index + 1 :])
        if candidate in target_keys:
            return candidate
    return None


def _load_cuda_diffusion_streaming(
    model_config: dict[str, object],
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load fp16 weights one tensor at a time to bound Medium's host-memory peak.

    :param model_config: Parsed Stable Audio model configuration.
    :param checkpoint_path: Safetensors checkpoint in the immutable snapshot.
    :param device: CUDA device receiving model weights.
    :returns: Frozen conditional diffusion model.
    :raises RuntimeError: A checkpoint key or tensor shape does not match the model.
    """
    from safetensors import safe_open
    from stable_audio_3.factory import create_diffusion_cond_from_config

    prior_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        model = create_diffusion_cond_from_config(model_config)
    finally:
        torch.set_default_dtype(prior_dtype)

    model.to(device)
    state = model.state_dict()
    target_keys = set(state)
    loaded: set[str] = set()
    with torch.no_grad(), safe_open(checkpoint_path, framework="pt", device=str(device)) as source:
        for source_key in source.keys():
            target_key = _checkpoint_target_key(source_key, target_keys)
            if target_key is None:
                raise RuntimeError(
                    f"Stable Audio checkpoint key has no model target: {source_key}"
                )
            source_tensor = source.get_tensor(source_key)
            target_tensor = state[target_key]
            if source_tensor.shape != target_tensor.shape:
                raise RuntimeError(
                    f"Stable Audio checkpoint shape mismatch for {source_key}: "
                    f"{tuple(source_tensor.shape)} != {tuple(target_tensor.shape)}"
                )
            target_tensor.copy_(source_tensor.to(dtype=target_tensor.dtype))
            loaded.add(target_key)
    missing = sorted(target_keys - loaded)
    if missing:
        raise RuntimeError(f"Stable Audio checkpoint is missing model key: {missing[0]}")
    return model.eval().requires_grad_(False)


def load_stable_audio_model(
    profile: StableAudioProfile,
    device: torch.device,
) -> _StableAudioLatentModel:
    """Load one immutable Stable Audio snapshot with its local text conditioner.

    :param profile: Model repository, revision, and latent contract.
    :param device: Inference device.
    :returns: Stable Audio latent generator.
    :raises ValueError: The pinned model configuration is not a mapping.
    """
    from huggingface_hub import snapshot_download
    from stable_audio_3 import StableAudioModel
    from stable_audio_3.loading_utils import load_diffusion_cond

    snapshot = Path(snapshot_download(profile.repo_id, revision=profile.revision))
    config_path = snapshot / "model_config.json"
    checkpoint_path = snapshot / "model.safetensors"
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(model_config, dict):
        raise ValueError("Stable Audio model_config.json must contain a mapping")
    _localize_prompt_conditioner(model_config, snapshot)
    use_half = device.type == "cuda"
    if use_half:
        diffusion = _load_cuda_diffusion_streaming(model_config, checkpoint_path, device)
    else:
        diffusion = load_diffusion_cond(
            model_config,
            str(checkpoint_path),
            device=str(device),
            model_half=False,
        )
    setattr(diffusion, "use_lora", False)
    setattr(diffusion, "lora_names", [])
    return cast(
        _StableAudioLatentModel,
        StableAudioModel(diffusion, model_config, str(device), use_half),
    )


def validate_same_latent(latent: torch.Tensor) -> torch.Tensor:
    """Return a finite CPU float32 latent matching the trained inverse geometry.

    :param latent: Generated Stable Audio latent batch.
    :returns: Detached latent in the inverse model's canonical dtype and device.
    :raises ValueError: Shape or values cannot satisfy the inverse-model contract.
    """
    if tuple(latent.shape) != _EXPECTED_LATENT_SHAPE:
        raise ValueError(
            f"Stable Audio latent shape {tuple(latent.shape)} does not match "
            f"expected {_EXPECTED_LATENT_SHAPE}"
        )
    latent = latent.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(latent).all():
        raise ValueError("Stable Audio latent must contain only finite values")
    return latent


def generate_same_latent(
    model: _StableAudioLatentModel,
    prompt: str,
    seed: int,
) -> torch.Tensor:
    """Generate the fixed four-second SAME latent consumed by Surge inversion.

    :param model: Loaded Stable Audio latent generator.
    :param prompt: Natural-language audio description.
    :param seed: Diffusion noise seed.
    :returns: Finite ``(1, 256, 44)`` CPU float32 latent.
    :raises ValueError: Model geometry or generated values violate the inverse contract.
    """
    sample_size = model.model_config.get("sample_size")
    if not isinstance(sample_size, int):
        raise ValueError("Stable Audio model config has no integer sample_size")
    latent = model.generate(
        prompt=prompt,
        duration=_RENDER_DURATION_SECONDS,
        steps=_GENERATION_STEPS,
        cfg_scale=_CFG_SCALE,
        batch_size=1,
        sample_size=sample_size,
        duration_padding_sec=0.0,
        return_latents=True,
        seed=seed,
    )
    return validate_same_latent(latent)


def _run_id(prompt: str, selector: _ModelSelector) -> str:
    """Build a prompt- and profile-addressed artifact identifier.

    :param prompt: Normalized text prompt.
    :param selector: Small or Medium profile name.
    :returns: Unique filesystem- and R2-safe identifier.
    """
    from datetime import UTC, datetime

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    prompt_id = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    return f"sao-{selector}-{timestamp}-{prompt_id}"


def _resolve_output(
    output: Path | None,
    settings: _StableAudioRenderSettings,
    run_id: str,
) -> Path:
    """Resolve a new absolute WAV path without overwriting either artifact.

    :param output: Explicit WAV destination, or ``None`` for the configured directory.
    :param settings: Validated render defaults.
    :param run_id: Unique filename stem.
    :returns: Absolute WAV destination.
    :raises click.ClickException: The WAV or adjacent CSV already exists.
    """
    if output is None:
        output = settings.output_dir / f"{run_id}.wav"
    if not output.is_absolute():
        output = operator_workspace() / output
    output = output.resolve()
    collisions = [path for path in (output, output.with_suffix(".csv")) if path.exists()]
    if collisions:
        raise click.ClickException(f"refusing to overwrite existing output: {collisions[0]}")
    return output


def _csv_uri_for_wav(wav_uri: str) -> str:
    """Return the provenance URI adjacent to a rendered WAV URI.

    :param wav_uri: R2 destination for rendered audio.
    :returns: Same object path with a ``.csv`` suffix.
    """
    if wav_uri.casefold().endswith(".wav"):
        return wav_uri[:-4] + ".csv"
    return f"{wav_uri}.csv"


def _write_provenance(path: Path, row: Mapping[str, str | int | float]) -> None:
    """Persist one Stable Audio-to-Surge inference identity.

    :param path: CSV destination.
    :param row: Values for every provenance field.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_PROVENANCE_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def _clear_device_cache(device: torch.device) -> None:
    """Release unreferenced source-model memory before inverse inference.

    :param device: Source model's former inference device.
    """
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("text_prompt")
@click.option(
    "--model",
    "selector",
    type=click.Choice(["small", "medium"]),
    default="small",
    show_default=True,
    help="Stable Audio profile; small uses SAME-S and medium uses SAME-L.",
)
@click.option(
    "--checkpoint",
    envvar="SYNTH_SETTER_SAO_INVERSE_CHECKPOINT",
    help="SAME-conditioned inverse checkpoint path or R2 URI.",
)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), help="Local WAV path.")
@click.option("--upload-uri", help="Exact r2:// destination for the WAV.")
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    default=None,
    help="Inference device [default: auto].",
)
@click.option(
    "--seed", type=int, default=None, help="Shared diffusion and inverse seed [default: 0]."
)
@click.option("--upload/--no-upload", default=True, show_default=True)
def main(
    text_prompt: str,
    selector: _ModelSelector,
    checkpoint: str | None,
    output: Path | None,
    upload_uri: str | None,
    device: _DeviceSetting | None,
    seed: int | None,
    upload: bool,
) -> None:
    """Render TEXT_PROMPT through Stable Audio and a SAME-conditioned Surge inverse.

    Example: synth-setter-sao "warm analog pad" --model small

    :param text_prompt: Natural-language sound description.
    :param selector: Stable Audio Small-Music or Medium profile.
    :param checkpoint: Optional inverse-checkpoint override.
    :param output: Optional local WAV destination.
    :param upload_uri: Optional exact R2 object destination.
    :param device: Optional torch-device override.
    :param seed: Shared Stable Audio and inverse sampling seed.
    :param upload: Whether to upload the WAV and provenance CSV.
    :raises click.ClickException: CLI arguments are invalid.
    """
    prompt = text_prompt.strip()
    if not prompt:
        raise click.ClickException("prompt must contain text")
    if upload_uri is not None and not upload:
        raise click.ClickException("--upload-uri cannot be combined with --no-upload")
    if upload_uri is not None and not r2_io.is_r2_uri(upload_uri):
        raise click.ClickException("--upload-uri must use r2://")

    settings = _load_settings()
    profile = settings.profiles[selector]
    selected_device = resolve_device(device or settings.device)
    selected_seed = settings.seed if seed is None else seed
    inverse_source = checkpoint or profile.inverse_checkpoint
    run_id = _run_id(prompt, selector)
    output_path = _resolve_output(output, settings, run_id)
    provenance_path = output_path.with_suffix(".csv")
    default_wav_destination = f"{settings.upload_prefix}/{run_id}.wav"
    wav_destination = (upload_uri or default_wav_destination) if upload else ""
    csv_destination = _csv_uri_for_wav(wav_destination) if upload else ""

    if upload or r2_io.is_r2_uri(inverse_source):
        click.echo("Checking R2 access...", err=True)
        r2_io.ensure_r2_env_loaded()

    click.echo(f"Generating {profile.conditioning.column} with {profile.model_name}...", err=True)
    source_model = load_stable_audio_model(profile, selected_device)
    latent = generate_same_latent(source_model, prompt, selected_seed)
    del source_model
    _clear_device_cache(selected_device)

    click.echo("Loading inverse checkpoint...", err=True)
    expected_digest = profile.inverse_checkpoint_sha256 if checkpoint is None else None
    inverse_checkpoint = resolve_inverse_checkpoint(inverse_source, expected_digest)
    render = workspace_render_config(settings.render)
    prediction = predict_patch(
        latent.to(selected_device),
        inverse_checkpoint,
        render,
        selected_device,
        selected_seed,
        profile.conditioning,
    )
    click.echo("Rendering Surge patch...", err=True)
    render_wav(prediction, render, output_path)

    _write_provenance(
        provenance_path,
        {
            "prompt": prompt,
            "model": selector,
            "stable_audio_model": profile.model_name,
            "model_repo": profile.repo_id,
            "model_revision": profile.revision,
            "conditioning": profile.conditioning.column,
            "latent_shape": "x".join(str(size) for size in latent.shape),
            "latent_norm": float(torch.linalg.vector_norm(latent)),
            "inverse_checkpoint": inverse_source,
            "inverse_checkpoint_sha256": expected_digest or "",
            "seed": selected_seed,
            "wav_r2_uri": wav_destination,
            "csv_r2_uri": csv_destination,
        },
    )

    click.echo(f"Local WAV: {output_path}")
    click.echo(f"Local CSV: {provenance_path}")
    if upload:
        click.echo(f"Uploading {wav_destination}...", err=True)
        r2_io.upload_to_uri(output_path, wav_destination)
        r2_io.upload_to_uri(provenance_path, csv_destination)
        click.echo(f"R2 WAV: {wav_destination}")
        click.echo(f"R2 CSV: {csv_destination}")


if __name__ == "__main__":
    main()
