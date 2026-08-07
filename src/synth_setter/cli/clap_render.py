"""Render a Surge patch from text through CLAP-conditioned inverse inference."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import click
import numpy as np
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator

from synth_setter.clap import (
    DEFAULT_CLAP_TRAINING_CHECKPOINT_SHA256,
    clap_checkpoint_sha256,
    resolve_clap_checkpoint,
)
from synth_setter.conditioning import resolve_embedding_conditioning
from synth_setter.data.vst.core import write_wav
from synth_setter.data.vst.param_spec import decode_model_output
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.model_cache import synth_setter_cache_dir
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.renderer_factory import make_audio_renderer
from synth_setter.synth_spec import SynthSpec
from synth_setter.workspace import operator_workspace

_DeviceSetting = Literal["auto", "cpu", "cuda", "mps"]
_EXPECTED_CONDITIONING_COLUMN = "clap"
_EXPECTED_CONDITIONING_SHAPE = (512,)
_DEFAULT_CONFIG_NAME = "clap_render"
_HASH_CHUNK_BYTES = 1024 * 1024
_COMPARISON_CSV_FIELDS: tuple[str, ...] = (
    "prompt",
    "wav_r2_uri",
    "csv_r2_uri",
    "seed",
    "text_embedding_norm",
    "audio_embedding_norm",
    "cosine_similarity",
    "cosine_distance",
)


@dataclass(frozen=True)
class EmbeddingComparison:
    """Cosine comparison between one prompt and rendered-audio embedding.

    .. attribute :: text_embedding_norm

        L2 norm of the prompt embedding.

    .. attribute :: audio_embedding_norm

        L2 norm of the rendered-audio embedding.

    .. attribute :: cosine_similarity

        Normalized dot product in ``[-1, 1]``.

    .. attribute :: cosine_distance

        ``1 - cosine_similarity`` in ``[0, 2]``.
    """

    text_embedding_norm: float
    audio_embedding_norm: float
    cosine_similarity: float
    cosine_distance: float


def compare_embeddings(text: np.ndarray, audio: np.ndarray) -> EmbeddingComparison:
    """Compare one text/audio CLAP pair with explicit norm diagnostics.

    :param text: Prompt embedding shaped ``(1, embedding_dim)``.
    :param audio: Rendered-audio embedding with the same shape.
    :returns: Cosine similarity, distance, and source-vector norms.
    :raises ValueError: Shapes differ, values are non-finite, or either norm is zero.
    """
    if text.shape != audio.shape or text.ndim != 2 or text.shape[0] != 1:
        raise ValueError(
            f"embedding shapes must match as (1, dim), got {text.shape} and {audio.shape}"
        )
    if not np.isfinite(text).all() or not np.isfinite(audio).all():
        raise ValueError("embeddings must contain only finite values")
    text_norm = float(np.linalg.norm(text[0]))
    audio_norm = float(np.linalg.norm(audio[0]))
    if text_norm == 0.0 or audio_norm == 0.0:
        raise ValueError("embeddings must have non-zero norms")
    similarity = float(np.dot(text[0], audio[0]) / (text_norm * audio_norm))
    similarity = float(np.clip(similarity, -1.0, 1.0))
    return EmbeddingComparison(
        text_embedding_norm=text_norm,
        audio_embedding_norm=audio_norm,
        cosine_similarity=similarity,
        cosine_distance=1.0 - similarity,
    )


def summarize_cosine_distances(distances: Sequence[float]) -> dict[str, float | int]:
    """Compute population statistics for a non-empty distance collection.

    :param distances: Finite cosine distances.
    :returns: Count, center, spread, extrema, and quartiles.
    :raises ValueError: No distances are supplied or any value is non-finite.
    """
    values = np.asarray(distances, dtype=np.float64)
    if values.size == 0:
        raise ValueError("at least one cosine distance is required")
    if not np.isfinite(values).all():
        raise ValueError("cosine distances must contain only finite values")
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std_population": float(values.std(ddof=0)),
        "min": float(values.min()),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "max": float(values.max()),
    }


def write_comparison_csv(path: Path, row: Mapping[str, str | int | float]) -> None:
    """Write one prompt/audio comparison with a stable column order.

    :param path: CSV destination.
    :param row: Values for every comparison field.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_COMPARISON_CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def write_summary_csv(path: Path, summary: Mapping[str, float | int]) -> None:
    """Write aggregate statistics as one row keyed by statistic name.

    :param path: CSV destination.
    :param summary: Ordered statistic names and values.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


class _ClapRenderSettings(BaseModel):
    """Validated zero-argument defaults composed from ``clap_render.yaml``.

    .. attribute :: model_config

        Strict immutable Pydantic settings validation.

    .. attribute :: inverse_checkpoint

        Local path or R2 URI for the CLAP-conditioned inverse model.

    .. attribute :: inverse_checkpoint_sha256

        Required digest for the default inverse checkpoint.

    .. attribute :: clap_checkpoint

        Local directory, R2 prefix, or Hugging Face identifier for CLAP.

    .. attribute :: output_dir

        Default directory for local WAV files.

    .. attribute :: upload_prefix

        R2 prefix receiving generated WAV files.

    .. attribute :: device

        Requested inference device or automatic selection.

    .. attribute :: seed

        Flow sampling seed.

    .. attribute :: render

        Surge renderer identity and audio settings.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    inverse_checkpoint: str
    inverse_checkpoint_sha256: str
    clap_checkpoint: str
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

    @field_validator("inverse_checkpoint_sha256")
    @classmethod
    def _inverse_digest_is_sha256(cls, value: str) -> str:
        """Require a complete hexadecimal SHA-256 digest.

        :param value: Configured digest.
        :returns: Validated digest unchanged.
        :raises ValueError: The digest has the wrong length or non-hexadecimal characters.
        """
        if len(value) != 64:
            raise ValueError("inverse_checkpoint_sha256 must contain 64 hex characters")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError("inverse_checkpoint_sha256 must be hexadecimal") from exc
        return value

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


def _load_settings() -> _ClapRenderSettings:
    """Compose the CLI defaults and join the render and synth identities.

    :returns: Strict settings ready for one CLI invocation.
    :raises TypeError: The composed root, render, or synth node is not a mapping.
    """
    with initialize_config_module(config_module="synth_setter.configs", version_base="1.3"):
        cfg = compose(config_name=_DEFAULT_CONFIG_NAME)
    values = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("clap_render config must resolve to a mapping")
    render_values = values.pop("render")
    synth_values = values.pop("synth")
    if not isinstance(render_values, dict) or not isinstance(synth_values, dict):
        raise TypeError("clap_render render and synth nodes must resolve to mappings")
    render_values["synth"] = synth_values
    values["render"] = RenderConfig.model_validate(render_values)
    return _ClapRenderSettings.model_validate(values)


@contextmanager
def _checkpoint_lock(path: Path) -> Iterator[None]:
    """Serialize publication of one cached inverse checkpoint.

    :param path: Cache destination whose sibling lock is acquired. :yields: Control while this
        process owns the file lock.
    """
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _file_sha256(path: Path) -> str:
    """Hash a checkpoint without loading it into memory.

    :param path: Checkpoint file.
    :returns: Lowercase SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_matches(path: Path, expected_sha256: str | None) -> bool:
    """Check file presence, non-empty content, and an optional digest.

    :param path: Candidate checkpoint.
    :param expected_sha256: Required digest, or ``None`` to accept any non-empty file.
    :returns: Whether the checkpoint satisfies the requested identity.
    """
    return (
        path.is_file()
        and path.stat().st_size > 0
        and (expected_sha256 is None or _file_sha256(path) == expected_sha256)
    )


def resolve_inverse_checkpoint(checkpoint: str, expected_sha256: str | None = None) -> Path:
    """Resolve a local or R2 inverse-model checkpoint to a reusable local file.

    :param checkpoint: Local checkpoint path or exact R2 object URI.
    :param expected_sha256: Optional required file digest.
    :returns: Existing local path or the cached R2 object.
    :raises FileNotFoundError: A local checkpoint does not exist.
    :raises RuntimeError: A checkpoint is empty or fails digest validation.
    """
    local = Path(checkpoint).expanduser()
    if local.is_file():
        if not _checkpoint_matches(local, expected_sha256):
            raise RuntimeError(f"inverse checkpoint SHA-256 mismatch: {local}")
        return local.resolve()
    if not r2_io.is_r2_uri(checkpoint):
        raise FileNotFoundError(f"inverse checkpoint does not exist: {local}")

    source_key = hashlib.sha256(checkpoint.encode()).hexdigest()
    cached = synth_setter_cache_dir() / "models" / "inverse" / source_key / "model.ckpt"
    with _checkpoint_lock(cached):
        if _checkpoint_matches(cached, expected_sha256):
            return cached
        cached.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=".model-", suffix=".ckpt", dir=cached.parent, delete=False
        ) as stream:
            staging = Path(stream.name)
        try:
            r2_io.download_to_path(checkpoint, staging)
            if staging.stat().st_size == 0:
                raise RuntimeError(f"downloaded inverse checkpoint is empty: {checkpoint}")
            if not _checkpoint_matches(staging, expected_sha256):
                raise RuntimeError(f"inverse checkpoint SHA-256 mismatch: {checkpoint}")
            staging.replace(cached)
        finally:
            staging.unlink(missing_ok=True)
    return cached


def _resolve_device(requested: _DeviceSetting) -> torch.device:
    """Resolve an available torch device, preferring CUDA then MPS.

    :param requested: Explicit backend or automatic selection.
    :returns: Available torch device.
    :raises click.ClickException: An explicitly requested accelerator is unavailable.
    """
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise click.ClickException("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise click.ClickException("MPS was requested but is unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _encode_text(prompt: str, checkpoint_dir: str, device: torch.device) -> torch.Tensor:
    """Encode one prompt in CLAP's shared text-audio space.

    :param prompt: Non-blank text condition.
    :param checkpoint_dir: Materialized Transformers CLAP directory.
    :param device: Torch device hosting inference.
    :returns: Normalized conditioning tensor shaped ``(1, 512)``.
    :raises RuntimeError: CLAP returns a non-tensor or incompatible shape.
    """
    from transformers import ClapModel, ClapProcessor

    processor = ClapProcessor.from_pretrained(checkpoint_dir)
    model = ClapModel.from_pretrained(checkpoint_dir).to(device).eval()  # pyright: ignore
    processor_kwargs = {
        "text": [prompt],
        "padding": True,
        "truncation": True,
        "return_tensors": "pt",
    }
    inputs = processor(**processor_kwargs)
    device_inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        features = model.get_text_features(**device_inputs)
        embedding = features.pooler_output  # pyright: ignore
    if not isinstance(embedding, torch.Tensor):
        raise RuntimeError("CLAP text encoder did not return a tensor")
    if tuple(embedding.shape) != (1, *_EXPECTED_CONDITIONING_SHAPE):
        raise RuntimeError(
            f"CLAP text embedding shape {tuple(embedding.shape)} does not match "
            f"{(1, *_EXPECTED_CONDITIONING_SHAPE)}"
        )
    return embedding


def _encode_audio(
    audio: np.ndarray,
    sample_rate: int,
    checkpoint_dir: str,
    device: torch.device,
) -> np.ndarray:
    """Encode a stereo Surge render through the training CLAP audio path.

    :param audio: Channel-first waveform shaped ``(channels, samples)``.
    :param sample_rate: Render sample rate in Hz.
    :param checkpoint_dir: Materialized CLAP checkpoint directory.
    :param device: Torch inference device.
    :returns: Audio embedding shaped ``(1, 512)``.
    :raises RuntimeError: The production encoder returns an incompatible shape.
    """
    from synth_setter.pipeline.data.add_embeddings import load_clap_audio_encoder

    mono: np.ndarray = np.asarray(
        np.mean(audio, axis=0, dtype=np.float32, keepdims=True),
        dtype=np.float32,
    )
    encoder = load_clap_audio_encoder(checkpoint_dir, str(device))
    embedding = encoder(mono, sample_rate)
    if embedding.shape != (1, *_EXPECTED_CONDITIONING_SHAPE):
        raise RuntimeError(
            f"CLAP audio embedding shape {embedding.shape} does not match "
            f"{(1, *_EXPECTED_CONDITIONING_SHAPE)}"
        )
    return embedding


def _validate_inverse_model(model: VSTFlowMatchingModule, render: RenderConfig) -> None:
    """Require a text-compatible CLAP checkpoint for the selected Surge spec.

    :param model: Loaded inverse model.
    :param render: Active Surge renderer identity.
    :raises ValueError: Conditioning, sketch controls, or output width is incompatible.
    """
    conditioning = resolve_embedding_conditioning(model.hparams["conditioning"])
    if (
        conditioning is None
        or conditioning.column != _EXPECTED_CONDITIONING_COLUMN
        or conditioning.input_shape != _EXPECTED_CONDITIONING_SHAPE
    ):
        raise ValueError(
            "inverse checkpoint must use cached CLAP conditioning with input shape [512]"
        )
    if model.hparams["sketch_controls"] is not None:
        raise ValueError("text-only rendering does not support sketch-conditioned checkpoints")
    expected_width = len(param_specs[render.param_spec_name])
    checkpoint_width = model.hparams["num_params"]
    if checkpoint_width != expected_width:
        raise ValueError(
            f"checkpoint output width {checkpoint_width} does not match "
            f"{render.param_spec_name} width {expected_width}"
        )


def _predict_patch(
    embedding: torch.Tensor,
    checkpoint: Path,
    render: RenderConfig,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Sample one model-space Surge parameter row reproducibly.

    :param embedding: CLAP condition shaped ``(1, 512)``.
    :param checkpoint: Trusted Lightning checkpoint.
    :param render: Renderer identity used for compatibility validation.
    :param device: Torch inference device.
    :param seed: Flow noise seed.
    :returns: CPU prediction shaped ``(1, len(param_spec))``.
    """
    model = VSTFlowMatchingModule.load_from_checkpoint(
        checkpoint,
        map_location=device,
        weights_only=False,
    )
    _validate_inverse_model(model, render)
    model.to(device).eval()
    torch.manual_seed(seed)
    with torch.inference_mode():
        prediction, _ = model.predict_step({"conditioning": embedding}, 0)
    return prediction.detach().cpu()


def _workspace_render_config(render: RenderConfig) -> RenderConfig:
    """Anchor a relative preset path to the operator workspace.

    :param render: Composed render configuration.
    :returns: Configuration with a concrete preset path.
    """
    preset = Path(render.plugin_state_path).expanduser()
    if preset.is_absolute():
        return render
    synth_values = render.synth.model_dump()
    synth_values["plugin_state_path"] = str(operator_workspace() / preset)
    return render.model_copy(update={"synth": SynthSpec.model_validate(synth_values)})


def _render_wav(prediction: torch.Tensor, render: RenderConfig, output: Path) -> np.ndarray:
    """Decode one prediction and persist its production Surge render.

    :param prediction: Model-space parameter row shaped ``(1, len(param_spec))``.
    :param render: Surge renderer and audio configuration.
    :param output: New WAV destination.
    :returns: Channel-first rendered waveform written to ``output``.
    """
    spec = param_specs[render.param_spec_name]
    synth_params, note_params = decode_model_output(prediction[0].float().numpy(), spec)
    note_start, note_end = sorted(note_params["note_start_and_end"])
    renderer = make_audio_renderer(render)
    audio = renderer.render(
        synth_params,
        int(note_params["pitch"]),
        render.velocity,
        (note_start, note_end),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_wav(audio, str(output), render.sample_rate, render.channels)
    return audio


def _csv_uri_for_wav(wav_uri: str) -> str:
    """Derive the adjacent comparison CSV URI from a WAV object URI.

    :param wav_uri: R2 destination for rendered audio.
    :returns: Same object path with a ``.csv`` suffix.
    """
    if wav_uri.casefold().endswith(".wav"):
        return wav_uri[:-4] + ".csv"
    return f"{wav_uri}.csv"


def _run_id(prompt: str) -> str:
    """Build a unique timestamped identifier carrying prompt identity.

    :param prompt: Normalized prompt text.
    :returns: Filesystem- and R2-safe run identifier.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    prompt_id = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    return f"clap-{timestamp}-{prompt_id}"


def _resolve_output(output: Path | None, settings: _ClapRenderSettings, run_id: str) -> Path:
    """Resolve a new absolute WAV path without overwriting an artifact.

    :param output: Explicit destination, or ``None`` for the configured directory.
    :param settings: Validated CLI defaults.
    :param run_id: Unique filename stem.
    :returns: Absolute WAV destination.
    :raises click.ClickException: The destination already exists.
    """
    if output is None:
        output = settings.output_dir / f"{run_id}.wav"
    if not output.is_absolute():
        output = operator_workspace() / output
    output = output.resolve()
    if output.exists():
        raise click.ClickException(f"refusing to overwrite existing output: {output}")
    return output


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("text_prompt", required=False)
@click.option(
    "--guide_audio",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Guide audio supplying sketch controls.",
)
@click.option(
    "--ref_audio",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Reference audio supplying mel/timbre conditioning.",
)
@click.option(
    "--checkpoint",
    envvar="SYNTH_SETTER_CLAP_INVERSE_CHECKPOINT",
    help="CLAP-conditioned inverse checkpoint path or R2 URI.",
)
@click.option(
    "--clap-checkpoint",
    envvar="SYNTH_SETTER_CLAP_ENCODER_CHECKPOINT",
    help="LAION-CLAP checkpoint directory, Hugging Face id, or R2 prefix.",
)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), help="Local WAV path.")
@click.option("--upload-uri", help="Exact r2:// destination for the WAV.")
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    default=None,
    help="Inference device [default: auto].",
)
@click.option("--seed", type=int, default=None, help="Flow sampling seed [default: 0].")
@click.option("--upload/--no-upload", default=True, show_default=True)
def main(
    text_prompt: str | None,
    guide_audio: Path | None,
    ref_audio: Path | None,
    checkpoint: str | None,
    clap_checkpoint: str | None,
    output: Path | None,
    upload_uri: str | None,
    device: _DeviceSetting | None,
    seed: int | None,
    upload: bool,
) -> None:
    """Render a text prompt or guide/reference audio as a Surge WAV.

    Use ``synth-setter-clap "frog croak"`` for text or pass both audio options.

    :param text_prompt: Optional natural-language sound description.
    :param guide_audio: Optional audio supplying sketch controls.
    :param ref_audio: Optional audio supplying mel/timbre conditioning.
    :param checkpoint: Optional inverse-checkpoint override.
    :param clap_checkpoint: Optional CLAP encoder override.
    :param output: Optional local WAV destination.
    :param upload_uri: Optional exact R2 object destination.
    :param device: Optional torch-device override.
    :param seed: Optional flow sampling seed.
    :param upload: Whether to upload the rendered WAV.
    :raises click.ClickException: CLI arguments are invalid.
    :raises RuntimeError: The default CLAP checkpoint fails identity validation.
    """
    audio_mode = guide_audio is not None or ref_audio is not None
    if audio_mode:
        if guide_audio is None or ref_audio is None:
            raise click.ClickException("--guide_audio and --ref_audio must be provided together")
        if text_prompt is not None:
            raise click.ClickException("TEXT_PROMPT cannot be combined with guide/reference audio")
        from synth_setter.cli.clap import main as guide_audio_main

        guide_audio_main.main(
            args=["--guide_audio", str(guide_audio), "--ref_audio", str(ref_audio)],
            prog_name="synth-setter-clap",
            standalone_mode=False,
        )
        return

    if text_prompt is None:
        raise click.ClickException("provide TEXT_PROMPT or guide/reference audio")
    prompt = text_prompt.strip()
    if not prompt:
        raise click.ClickException("prompt must contain text")
    if upload_uri is not None and not upload:
        raise click.ClickException("--upload-uri cannot be combined with --no-upload")
    if upload_uri is not None and not r2_io.is_r2_uri(upload_uri):
        raise click.ClickException("--upload-uri must use r2://")

    settings = _load_settings()
    inverse_source = checkpoint or settings.inverse_checkpoint
    clap_source = clap_checkpoint or settings.clap_checkpoint
    selected_device = _resolve_device(device or settings.device)
    selected_seed = settings.seed if seed is None else seed
    run_id = _run_id(prompt)
    output_path = _resolve_output(output, settings, run_id)
    metrics_path = output_path.with_suffix(".csv")
    default_wav_destination = f"{settings.upload_prefix}/{run_id}.wav"
    wav_destination = (upload_uri or default_wav_destination) if upload else ""
    csv_destination = _csv_uri_for_wav(wav_destination) if upload else ""

    if upload or r2_io.is_r2_uri(inverse_source) or r2_io.is_r2_uri(clap_source):
        click.echo("Checking R2 access...", err=True)
        r2_io.ensure_r2_env_loaded()

    click.echo(f"Encoding text with CLAP on {selected_device}...", err=True)
    clap_checkpoint_dir = resolve_clap_checkpoint(clap_source)
    if (
        clap_checkpoint is None
        and clap_checkpoint_sha256(Path(clap_checkpoint_dir))
        != DEFAULT_CLAP_TRAINING_CHECKPOINT_SHA256
    ):
        raise RuntimeError("default CLAP checkpoint SHA-256 mismatch")
    embedding = _encode_text(prompt, clap_checkpoint_dir, selected_device)
    click.echo("Loading inverse checkpoint...", err=True)
    expected_inverse_sha256 = settings.inverse_checkpoint_sha256 if checkpoint is None else None
    inverse_checkpoint = resolve_inverse_checkpoint(inverse_source, expected_inverse_sha256)
    render = _workspace_render_config(settings.render)
    prediction = _predict_patch(
        embedding,
        inverse_checkpoint,
        render,
        selected_device,
        selected_seed,
    )
    click.echo("Rendering Surge patch...", err=True)
    audio = _render_wav(prediction, render, output_path)
    click.echo("Encoding rendered audio with CLAP...", err=True)
    audio_embedding = _encode_audio(
        audio,
        render.sample_rate,
        clap_checkpoint_dir,
        selected_device,
    )
    comparison = compare_embeddings(embedding.detach().cpu().float().numpy(), audio_embedding)
    write_comparison_csv(
        metrics_path,
        {
            "prompt": prompt,
            "wav_r2_uri": wav_destination,
            "csv_r2_uri": csv_destination,
            "seed": selected_seed,
            "text_embedding_norm": comparison.text_embedding_norm,
            "audio_embedding_norm": comparison.audio_embedding_norm,
            "cosine_similarity": comparison.cosine_similarity,
            "cosine_distance": comparison.cosine_distance,
        },
    )

    click.echo(f"Cosine distance: {comparison.cosine_distance:.9g}")
    click.echo(f"Local WAV: {output_path}")
    click.echo(f"Local CSV: {metrics_path}")
    if upload:
        click.echo(f"Uploading {wav_destination}...", err=True)
        r2_io.upload_to_uri(output_path, wav_destination)
        r2_io.upload_to_uri(metrics_path, csv_destination)
        click.echo(f"R2 WAV: {wav_destination}")
        click.echo(f"R2 CSV: {csv_destination}")


if __name__ == "__main__":
    main()
