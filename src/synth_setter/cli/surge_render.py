"""Share inverse-checkpoint inference and Surge artifact rendering across text CLIs."""

from __future__ import annotations

import fcntl
import hashlib
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import click
import numpy as np
import torch

from synth_setter.conditioning import EmbeddingConditioningSpec, resolve_embedding_conditioning
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

DeviceSetting = Literal["auto", "cpu", "cuda", "mps"]
_HASH_CHUNK_BYTES = 1024 * 1024


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


def checkpoint_sha256(path: Path) -> str:
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
        and (expected_sha256 is None or checkpoint_sha256(path) == expected_sha256)
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


def resolve_device(requested: DeviceSetting) -> torch.device:
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


def validate_inverse_model(
    model: VSTFlowMatchingModule,
    render: RenderConfig,
    expected_conditioning: EmbeddingConditioningSpec,
) -> None:
    """Require the selected embedding contract and Surge parameter width.

    :param model: Loaded inverse model.
    :param render: Active Surge renderer identity.
    :param expected_conditioning: Embedding column and per-row tensor shape.
    :raises ValueError: Conditioning, sketch controls, or output width is incompatible.
    """
    conditioning = resolve_embedding_conditioning(model.hparams["conditioning"])
    if conditioning != expected_conditioning:
        shape = ", ".join(str(size) for size in expected_conditioning.input_shape)
        raise ValueError(
            f"inverse checkpoint must use cached {expected_conditioning.column} "
            f"conditioning with input shape [{shape}]"
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


def predict_patch(
    embedding: torch.Tensor,
    checkpoint: Path,
    render: RenderConfig,
    device: torch.device,
    seed: int,
    expected_conditioning: EmbeddingConditioningSpec,
) -> torch.Tensor:
    """Sample one model-space Surge parameter row reproducibly.

    :param embedding: Conditioning batch matching ``expected_conditioning``.
    :param checkpoint: Trusted Lightning checkpoint.
    :param render: Renderer identity used for compatibility validation.
    :param device: Torch inference device.
    :param seed: Flow noise seed.
    :param expected_conditioning: Required checkpoint embedding contract.
    :returns: CPU prediction shaped ``(1, len(param_spec))``.
    """
    model = VSTFlowMatchingModule.load_from_checkpoint(
        checkpoint,
        map_location=device,
        weights_only=False,
    )
    validate_inverse_model(model, render, expected_conditioning)
    model.to(device).eval()
    torch.manual_seed(seed)
    with torch.inference_mode():
        prediction, _ = model.predict_step({"conditioning": embedding}, 0)
    return prediction.detach().cpu()


def workspace_render_config(render: RenderConfig) -> RenderConfig:
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


def validate_rendered_audio(audio: np.ndarray) -> None:
    """Reject non-finite or out-of-domain audio before artifact publication.

    :param audio: Rendered channel-first waveform.
    :raises ValueError: Audio is not finite or bounded to ``[-1, 1]``.
    """
    if not np.isfinite(audio).all():
        raise ValueError("rendered audio must contain only finite values")
    if np.any(audio < -1.0) or np.any(audio > 1.0):
        raise ValueError("rendered audio must remain in [-1, 1]")


def render_wav(prediction: torch.Tensor, render: RenderConfig, output: Path) -> np.ndarray:
    """Decode, validate, and persist one production Surge render.

    :param prediction: Model-space parameter row shaped ``(1, len(param_spec))``.
    :param render: Surge renderer and audio configuration.
    :param output: New WAV destination.
    :returns: Validated channel-first waveform written to ``output``.
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
    validate_rendered_audio(audio)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_wav(audio, str(output), render.sample_rate, render.channels)
    return audio
