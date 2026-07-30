"""Shared SAME checkpoint identity, materialization, and latent-frame geometry.

Lives outside the pipeline package so the training path can load SAME without importing the add-
embeddings CLI's Lance, Hydra, and librosa dependencies (plugin dlopen hazard, #2549).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from synth_setter.model_cache import embedding_model_dir, synth_setter_cache_dir
from synth_setter.pipeline import r2_io

if TYPE_CHECKING:
    from torch import nn

DEFAULT_SAME_S_CHECKPOINT: str = "r2://intermediate-data/models/same-s"
DEFAULT_SAME_L_CHECKPOINT: str = "r2://intermediate-data/models/same-l"
# Shared cache names predating per-URI namespacing; other sources hash their own prefix.
_DEFAULT_SAME_CACHE_NAMES: dict[str, str] = {
    DEFAULT_SAME_L_CHECKPOINT: "same-l",
    DEFAULT_SAME_S_CHECKPOINT: "same-s",
}

SAME_EMBEDDING_DIM: int = 256
SAME_SAMPLE_RATE: int = 44100
SAME_DOWNSAMPLING_RATIO: int = 4096
# SAME-S shifts chunk midpoints, so frames land in pairs spanning two hops.
SAME_S_PAD_BLOCK_SAMPLES: int = 2 * SAME_DOWNSAMPLING_RATIO


def resolve_same_checkpoint(checkpoint: str) -> Path:
    """Resolve a local, R2, or HuggingFace SAME checkpoint directory.

    :param checkpoint: Checkpoint directory, R2 prefix, or HuggingFace repo id.
    :returns: Local directory containing SAME model files.
    """
    if r2_io.is_r2_uri(checkpoint):
        cache_name = _DEFAULT_SAME_CACHE_NAMES.get(checkpoint)
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


def load_same_autoencoder(checkpoint_dir: Path) -> nn.Module:
    """Build a frozen SAME autoencoder from a materialized checkpoint directory.

    :param checkpoint_dir: Directory holding ``model_config.json`` and ``model.safetensors``.
    :returns: SAME autoencoder in eval mode with every parameter frozen.
    """
    import json

    from safetensors.torch import load_file
    from stable_audio_3.factory import create_autoencoder_from_config

    model_config = json.loads((checkpoint_dir / "model_config.json").read_text())
    model = create_autoencoder_from_config(model_config["model"], model_config["sample_rate"])
    model.load_state_dict(load_file(checkpoint_dir / "model.safetensors"), strict=True)
    return model.eval().requires_grad_(False)


def _same_resampled_samples(num_samples: int, sample_rate: int) -> int:
    """Return the ceiling sample count after resampling to SAME's 44.1 kHz input rate.

    :param num_samples: Positive source clip length in samples.
    :param sample_rate: Positive source sample rate in Hz.
    :returns: Resampled clip length in samples.
    :raises ValueError: Either input is non-positive.
    """
    if num_samples < 1 or sample_rate < 1:
        raise ValueError(f"need positive num_samples/sample_rate, got {num_samples}/{sample_rate}")
    return math.ceil(num_samples * SAME_SAMPLE_RATE / sample_rate)


def same_s_num_latent_frames(num_samples: int, sample_rate: int) -> int:
    """Return SAME-S's even frame count after resampling and two-hop padding.

    :param num_samples: Positive source clip length in samples.
    :param sample_rate: Positive source sample rate in Hz.
    :returns: Two frames per complete or partial 8192-sample block.
    """
    resampled = _same_resampled_samples(num_samples, sample_rate)
    return 2 * math.ceil(resampled / SAME_S_PAD_BLOCK_SAMPLES)


def same_l_num_latent_frames(num_samples: int, sample_rate: int) -> int:
    """Return SAME-L's frame count after resampling to its 4096-sample hop.

    :param num_samples: Positive source clip length in samples.
    :param sample_rate: Positive source sample rate in Hz.
    :returns: One frame per complete or partial 4096-sample block.
    """
    resampled = _same_resampled_samples(num_samples, sample_rate)
    return math.ceil(resampled / SAME_DOWNSAMPLING_RATIO)
