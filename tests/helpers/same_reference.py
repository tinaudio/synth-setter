"""Inputs and provenance shared by SAME parity tests and regeneration."""

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np

SAME_REFERENCE_DIR = Path(__file__).parents[1] / "fixtures" / "same"
SAME_REFERENCE_RANDOM_SEED = 0
SAME_REFERENCE_ROWS = 2

SAME_HF_CHECKPOINTS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "same_s": (
            "stabilityai/SAME-S",
            "fbeb3dcf53a326e5682f38e22e7f740202d44232",
        ),
        "same_l": (
            "stabilityai/SAME-L",
            "41acf79dd242877d6499a1108ca5dba5d5eecfc5",
        ),
    }
)


TINY_SAME_LATENT_DIM = 8
TINY_SAME_DOWNSAMPLING_RATIO = 512
# Chunk and stride are production values: SAME's chunked attention only divides evenly at
# those sizes, so only the widths and depths shrink.
_TINY_SAME_BLOCK: dict[str, object] = {
    "channels": 16,
    "c_mults": [1],
    "strides": [16],
    "latent_dim": TINY_SAME_LATENT_DIM,
    "transformer_depths": [1],
    "checkpointing": False,
    "differential": True,
    "dyt": True,
    "dim_heads": 8,
    "variable_stride": True,
    "chunk_size": 32,
    "chunk_midpoint_shift": True,
    "mask_noise": 0.0,
}
_TINY_SAME_PATCH_SIZE = 32


def tiny_same_model_config() -> dict[str, object]:
    """Build a SAME autoencoder config small enough for CPU unit tests.

    :returns: ``model`` block accepted by ``create_autoencoder_from_config``.
    """
    return {
        "pretransform": {
            "type": "patched",
            "config": {"patch_size": _TINY_SAME_PATCH_SIZE, "channels": 2},
        },
        "encoder": {
            "type": "same",
            "config": {"in_channels": 2 * _TINY_SAME_PATCH_SIZE, **_TINY_SAME_BLOCK},
        },
        "decoder": {
            "type": "same",
            "config": {
                "out_channels": 2 * _TINY_SAME_PATCH_SIZE,
                "sinusoidal_blocks": [0],
                "conv_mapping": True,
                **_TINY_SAME_BLOCK,
            },
        },
        "bottleneck": {
            "type": "softnorm",
            "config": {
                "dim": TINY_SAME_LATENT_DIM,
                "noise_augment_dim": 0,
                "noise_regularize": True,
                "auto_scale": True,
                "freeze": True,
            },
        },
        "latent_dim": TINY_SAME_LATENT_DIM,
        "downsampling_ratio": TINY_SAME_DOWNSAMPLING_RATIO,
        "io_channels": 2,
    }


def write_tiny_same_checkpoint(destination: Path, sample_rate: int) -> Path:
    """Materialize a loadable SAME checkpoint with random, non-degenerate weights.

    SAME zero-initializes its residual output projections, which would make every input gradient
    exactly zero; the perturbation gives untrained weights a real gradient path.

    :param destination: Directory receiving the config and weights.
    :param sample_rate: Rate recorded in the checkpoint config.
    :returns: The materialized checkpoint directory.
    """
    import torch
    from safetensors.torch import save_file
    from stable_audio_3.factory import create_autoencoder_from_config

    model_config = tiny_same_model_config()
    torch.manual_seed(SAME_REFERENCE_RANDOM_SEED)
    model = create_autoencoder_from_config(model_config, sample_rate)
    with torch.no_grad():
        for parameter in model.parameters():
            if not parameter.any():
                parameter.normal_(std=0.05)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "model_config.json").write_text(
        json.dumps({"sample_rate": sample_rate, "model": model_config})
    )
    save_file(model.state_dict(), destination / "model.safetensors")
    return destination


def same_reference_audio(sample_rate: int) -> np.ndarray:
    """Build two deterministic stereo chirps at the requested sample rate.

    :param sample_rate: Samples per second.
    :returns: ``(2, 2, sample_rate)`` float32 audio.
    """
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    rows = []
    for start_hz, end_hz in ((110.0, 880.0), (1760.0, 220.0)):
        frequency = start_hz + (end_hz - start_hz) * time
        phase = 2.0 * np.pi * np.cumsum(frequency) / sample_rate
        rows.append(np.stack((np.sin(phase), np.sin(phase * 1.01))))
    return (0.5 * np.stack(rows)).astype(np.float32)
