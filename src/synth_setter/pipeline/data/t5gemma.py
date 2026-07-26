"""Stable Audio 3 text conditioning: frozen T5Gemma with learned padding substitution.

Embeddings are numerically equivalent to SA3's own prompt conditioning, so the
conditioner class, tokenizer, truncation, and padding all come from
``stable_audio_3`` rather than being reimplemented here. The learned padding
embedding is always read from a checkpoint — SA3's DiT cross-attends over every
context position unmasked, so pad content is semantically live and zeros are not
a valid substitute.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import structlog
import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from pydantic import AliasPath, BaseModel, ConfigDict, Field
from torch import Tensor

from synth_setter.model_cache import embedding_model_dir, synth_setter_cache_dir
from synth_setter.pipeline import r2_io

logger = structlog.get_logger(__name__)

DEFAULT_T5GEMMA_CHECKPOINT: str = "r2://intermediate-data/models/sa3-small-music"
T5GEMMA_EMBEDDING_DIM: int = 768
T5GEMMA_MAX_LENGTH: int = 256
# Only model name SA3's conditioner accepts; the checkpoint supplies the weights.
_T5GEMMA_MODEL_NAME: str = "google/t5gemma-b-b-ul2"
_PROMPT_CONDITIONER_ID: str = "prompt"
_PADDING_EMBEDDING_KEY: str = "conditioner.conditioners.prompt.padding_embedding"
_LEARNED_PADDING_MODE: str = "learned"
_DEFAULT_T5GEMMA_CACHE_NAMES: dict[str, str] = {
    DEFAULT_T5GEMMA_CHECKPOINT: "sa3-small-music",
}

type TextEncodeFn = Callable[[list[str]], Float[np.ndarray, "batch dim seq"]]


@dataclass(frozen=True)
class _PromptConditioner(BaseModel):
    """Validate the prompt entry and flatten its nested settings.

    .. attribute :: id

        Canonical prompt conditioner identifier.

    .. attribute :: type

        Canonical T5Gemma conditioner type.

    .. attribute :: max_length

        Token budget read from the nested config.

    .. attribute :: padding_mode

        Padding strategy read from the nested config.

    .. attribute :: subfolder

        Checkpoint-relative encoder directory.

    .. attribute :: model_config

        Strict Pydantic validation settings.
    """

    id: Literal["prompt"]
    type: Literal["t5gemma"]
    max_length: int = Field(validation_alias=AliasPath("config", "max_length"))
    padding_mode: str = Field(validation_alias=AliasPath("config", "padding_mode"))
    subfolder: str = Field(validation_alias=AliasPath("config", "subfolder"))

    model_config = ConfigDict(strict=True)


class _ConditioningConfig(BaseModel):
    """Validate the conditioning width and preserve entries for prompt selection.

    .. attribute :: configs

        Raw conditioner entries; the selected prompt receives stricter validation.

    .. attribute :: cond_dim

        Shared conditioner output width.

    .. attribute :: model_config

        Strict Pydantic validation settings.
    """

    configs: list[dict[str, object]]
    cond_dim: int

    model_config = ConfigDict(strict=True)


@dataclass(frozen=True)
class T5GemmaConditionerConfig:
    """Prompt-conditioner settings read from an SA3 ``model_config.json``.

    .. attribute :: max_length

        Token budget the tokenizer truncates and pads to.

    .. attribute :: padding_mode

        How pad positions are filled; only ``learned`` reproduces SA3.

    .. attribute :: subfolder

        Checkpoint-relative directory holding the T5Gemma encoder.

    .. attribute :: cond_dim

        Conditioner output width.
    """

    max_length: int
    padding_mode: str
    subfolder: str
    cond_dim: int


def _resolve_t5gemma_checkpoint_dir(checkpoint: str) -> Path:
    """Resolve a local, R2, or HuggingFace SA3 checkpoint directory.

    :param checkpoint: Checkpoint directory, R2 prefix, or HuggingFace repo id.
    :returns: Local directory containing the SA3 model files.
    """
    if r2_io.is_r2_uri(checkpoint):
        cache_name = _DEFAULT_T5GEMMA_CACHE_NAMES.get(checkpoint)
        if cache_name is None:
            cache_key = checkpoint.removeprefix("r2://").strip("/")
            cache_dir = synth_setter_cache_dir() / "models" / cache_key
        else:
            cache_dir = embedding_model_dir(cache_name)
        r2_io.ensure_r2_env_loaded()
        r2_io.download_dir_no_overwrite(checkpoint, cache_dir)
        return cache_dir
    local = Path(checkpoint)
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(checkpoint))


def _read_conditioner_config(checkpoint_dir: Path) -> T5GemmaConditionerConfig:
    """Read the prompt conditioner's settings from a checkpoint.

    ``max_length`` and ``padding_mode`` are authoritative in the checkpoint: the
    conditioner class defaults to legacy ``"zero"`` padding, which would silently
    produce embeddings that are not SA3's.

    :param checkpoint_dir: Directory containing ``model_config.json``.
    :returns: Settings for the prompt conditioner.
    :raises ValueError: The checkpoint lacks a canonical learned-padding T5Gemma conditioner.
    """
    raw_checkpoint = json.loads((checkpoint_dir / "model_config.json").read_text())
    try:
        raw_conditioning = raw_checkpoint["model"]["conditioning"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{checkpoint_dir} has no model conditioning config") from exc
    conditioning = _ConditioningConfig.model_validate(raw_conditioning)
    prompts = [c for c in conditioning.configs if c.get("id") == _PROMPT_CONDITIONER_ID]
    if not prompts:
        raise ValueError(
            f"{checkpoint_dir} has no {_PROMPT_CONDITIONER_ID!r} conditioner; "
            "it cannot produce text embeddings"
        )
    prompt = _PromptConditioner.model_validate(prompts[0])
    max_length = prompt.max_length
    if max_length != T5GEMMA_MAX_LENGTH:
        raise ValueError(
            f"{checkpoint_dir} declares max_length={max_length}, expected {T5GEMMA_MAX_LENGTH}"
        )
    cond_dim = conditioning.cond_dim
    if cond_dim != T5GEMMA_EMBEDDING_DIM:
        raise ValueError(
            f"{checkpoint_dir} declares cond_dim={cond_dim}, expected {T5GEMMA_EMBEDDING_DIM}"
        )
    padding_mode = prompt.padding_mode
    if padding_mode != _LEARNED_PADDING_MODE:
        raise ValueError(
            f"{checkpoint_dir} declares padding_mode={padding_mode!r}, expected "
            f"{_LEARNED_PADDING_MODE!r}; other modes do not reproduce SA3 conditioning"
        )
    return T5GemmaConditionerConfig(
        max_length=max_length,
        padding_mode=padding_mode,
        subfolder=prompt.subfolder,
        cond_dim=cond_dim,
    )


@jaxtyped(typechecker=beartype)
def _load_padding_embedding(checkpoint_dir: Path, device: str) -> Float[Tensor, " dim"]:
    """Read the learned padding embedding out of a checkpoint's safetensors.

    :param checkpoint_dir: Directory containing ``model.safetensors``.
    :param device: Torch device receiving the tensor.
    :returns: The learned padding embedding.
    :raises ValueError: The checkpoint carries no learned padding embedding.
    """
    from safetensors import safe_open

    with safe_open(checkpoint_dir / "model.safetensors", framework="pt", device=device) as weights:
        if _PADDING_EMBEDDING_KEY not in weights.keys():
            raise ValueError(
                f"{checkpoint_dir} has no {_PADDING_EMBEDDING_KEY!r}; the learned padding "
                "embedding cannot be substituted"
            )
        return weights.get_tensor(_PADDING_EMBEDDING_KEY)


def load_t5gemma_text_encoder(checkpoint: str, device: str) -> TextEncodeFn:
    """Load SA3's prompt conditioner and return an encoder over prompt batches.

    :param checkpoint: Local directory, R2 mirror, or HuggingFace repo id.
    :param device: Resolved Torch device.
    :returns: Encoder producing ``(B, T5GEMMA_EMBEDDING_DIM, max_length)`` embeddings.
    :raises ImportError: The optional ``sa3`` extra is unavailable.
    """
    try:
        from stable_audio_3.models.conditioners import T5GemmaConditioner
    except ImportError as exc:
        raise ImportError(
            "loading T5Gemma text encoders requires the optional `sa3` extra — "
            "install it with `uv sync --extra sa3`"
        ) from exc

    checkpoint_dir = _resolve_t5gemma_checkpoint_dir(checkpoint)
    config = _read_conditioner_config(checkpoint_dir)
    logger.info(
        "loading_t5gemma_checkpoint",
        checkpoint=checkpoint,
        device=device,
        max_length=config.max_length,
        padding_mode=config.padding_mode,
    )
    conditioner = T5GemmaConditioner(
        output_dim=config.cond_dim,
        model_name=_T5GEMMA_MODEL_NAME,
        # Upstream annotates max_length as str while defaulting it to int 128.
        max_length=config.max_length,  # pyright: ignore[reportArgumentType]
        padding_mode=config.padding_mode,
        model_path=str(checkpoint_dir),
        subfolder=config.subfolder,
    )
    conditioner.padding_embedding.data.copy_(_load_padding_embedding(checkpoint_dir, device))
    # The checkpoint declares bfloat16, whose SDPA kernels differ enough between torch
    # releases to shift these embeddings by up to 11.5 (std 1.75). float32 is bitwise
    # identical across torch 2.7.1 and 2.12.0, so a stored column stays reproducible.
    # SA3 stashes the frozen encoder in __dict__, so it reads as the class, not an instance.
    cast("torch.nn.Module", conditioner.model).to(torch.float32)
    conditioner = conditioner.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def _encode_prompt(prompt: str) -> Float[np.ndarray, "one seq dim"]:
        embeddings, _ = conditioner([prompt], device)
        return embeddings.float().cpu().numpy()

    @jaxtyped(typechecker=beartype)
    def encode(prompts: list[str]) -> Float[np.ndarray, "batch dim seq"]:
        if not prompts:
            return np.empty(
                (0, T5GEMMA_EMBEDDING_DIM, T5GEMMA_MAX_LENGTH), dtype=np.float32
            )
        unique_embeddings = {
            prompt: _encode_prompt(prompt)[0] for prompt in dict.fromkeys(prompts)
        }
        batch = np.stack([unique_embeddings[prompt] for prompt in prompts])
        return rearrange(batch, "b seq dim -> b dim seq")

    return encode
