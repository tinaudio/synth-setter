"""Pinned PupuJEPA identity, variant configurations, cache, and frame geometry."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from synth_setter.model_cache import checkpoint_files_sha256

PUPUJEPA_UPSTREAM_COMMIT = "54a621e9f879be7659d81b6a3c493bba855cc85f"
PUPUJEPA_TIMM_VERSION = "1.0.28"
DEFAULT_PUPUJEPA_TINY_CHECKPOINT = "spellbrush/PupuJEPA"
PUPUJEPA_CHECKPOINT_REVISION = "2ba230e41440c5b450a8dc8ad5d4a3cc9930f01d"
PUPUJEPA_TINY_ARGS_FILE = "pupujepaV2_25hz_tiny/args.json"
PUPUJEPA_TINY_WEIGHTS_FILE = (
    "pupujepaV2_25hz_tiny/checkpoint/step-0500000_loss-0.125064/model.safetensors"
)
PUPUJEPA_TINY_CHECKPOINT_SHA256 = (
    "7bfd3e04fce4131496362a69eed5b478980181668e918adfaaef4e602bbceb2a"
)
PUPUJEPA_LARGE_ARGS_FILE = "pupujepaV2_25hz_large/args.json"
PUPUJEPA_LARGE_WEIGHTS_FILE = (
    "pupujepaV2_25hz_large/checkpoint/step-0500000_loss-0.176985/model.safetensors"
)
PUPUJEPA_LARGE_CHECKPOINT_SHA256 = (
    "9e16f31ee25371dcb0e7e97264dfeab2d9318f55f6939504d22fc66e29c3fc84"
)

type PupuJepaVariant = Literal["tiny", "large"]


@dataclass(frozen=True)
class PupuJepaConfig:
    """Define the waveform frontend and teacher architecture.

    .. attribute :: sample_rate

        Frontend sample rate in Hz.

    .. attribute :: n_fft

        FFT size in samples.

    .. attribute :: win_length

        Hann window length in samples.

    .. attribute :: hop_length

        STFT hop length in samples.

    .. attribute :: n_mels

        Number of mel bands.

    .. attribute :: fmin

        Lowest mel frequency in Hz.

    .. attribute :: fmax

        Highest mel frequency in Hz.

    .. attribute :: mel_mean

        Training-set log-mel mean.

    .. attribute :: mel_std

        Training-set log-mel standard deviation.

    .. attribute :: patch_time

        Mel frames per patch.

    .. attribute :: patch_frequency

        Mel bands per patch.

    .. attribute :: embed_dim

        Teacher hidden width per patch.

    .. attribute :: depth

        Number of teacher transformer blocks.

    .. attribute :: num_heads

        Attention head count.

    .. attribute :: mlp_ratio

        Transformer MLP expansion ratio.

    .. attribute :: use_swiglu

        Whether teacher blocks use SwiGLU MLPs.

    .. attribute :: qk_norm

        Whether teacher attention normalizes queries and keys.
    """

    sample_rate: int
    n_fft: int
    win_length: int
    hop_length: int
    n_mels: int
    fmin: float
    fmax: float
    mel_mean: float
    mel_std: float
    patch_time: int
    patch_frequency: int
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    use_swiglu: bool
    qk_norm: bool

    def __post_init__(self) -> None:
        """Reject geometry that cannot form the PupuJEPA patch grid.

        :raises ValueError: A signal, patch, or transformer dimension is invalid.
        """
        positive_integers = (
            self.sample_rate,
            self.n_fft,
            self.win_length,
            self.hop_length,
            self.n_mels,
            self.patch_time,
            self.patch_frequency,
            self.embed_dim,
            self.depth,
            self.num_heads,
        )
        if any(value < 1 for value in positive_integers):
            raise ValueError("PupuJEPA dimensions must be positive")
        if self.win_length > self.n_fft or self.hop_length > self.n_fft:
            raise ValueError("PupuJEPA window and hop lengths cannot exceed n_fft")
        if self.n_mels % self.patch_frequency:
            raise ValueError("PupuJEPA n_mels must divide into complete frequency patches")
        if self.embed_dim % self.num_heads:
            raise ValueError("PupuJEPA embed_dim must divide evenly across attention heads")
        if not 0.0 <= self.fmin < self.fmax <= self.sample_rate / 2:
            raise ValueError("PupuJEPA mel bounds must lie within the Nyquist interval")
        if self.mel_std <= 0.0 or self.mlp_ratio <= 0.0:
            raise ValueError("PupuJEPA normalization and MLP scales must be positive")

    @property
    def reflection_padding(self) -> int:
        """Return the symmetric pre-STFT reflection padding in samples."""
        return (self.n_fft - self.hop_length) // 2

    @property
    def frequency_patches(self) -> int:
        """Return the number of frequency patches per time patch."""
        return self.n_mels // self.patch_frequency

    @property
    def output_dim(self) -> int:
        """Return the frequency-concatenated teacher width."""
        return self.frequency_patches * self.embed_dim


PUPUJEPA_TINY_CONFIG = PupuJepaConfig(
    sample_rate=24_000,
    n_fft=1_024,
    win_length=1_024,
    hop_length=240,
    n_mels=128,
    fmin=0.0,
    fmax=12_000.0,
    mel_mean=-4.089994845986366,
    mel_std=2.0242277159094813,
    patch_time=4,
    patch_frequency=16,
    embed_dim=192,
    depth=12,
    num_heads=3,
    mlp_ratio=4.0,
    use_swiglu=True,
    qk_norm=True,
)
PUPUJEPA_LARGE_CONFIG = replace(
    PUPUJEPA_TINY_CONFIG,
    embed_dim=1_024,
    depth=24,
    num_heads=16,
)
PUPUJEPA_SAMPLE_RATE = PUPUJEPA_TINY_CONFIG.sample_rate
PUPUJEPA_TINY_EMBEDDING_DIM = PUPUJEPA_TINY_CONFIG.output_dim
PUPUJEPA_LARGE_EMBEDDING_DIM = PUPUJEPA_LARGE_CONFIG.output_dim
PUPUJEPA_EMBEDDING_DIM = PUPUJEPA_TINY_EMBEDDING_DIM


@dataclass(frozen=True)
class PupuJepaCheckpointSpec:
    """Bind one public variant to its immutable artifacts and geometry.

    .. attribute :: args_file

        Snapshot-relative configuration path.

    .. attribute :: weights_file

        Snapshot-relative safetensors path.

    .. attribute :: checkpoint_sha256

        Framed digest of the selected configuration and weights.

    .. attribute :: config

        Expected frontend and teacher geometry.
    """

    args_file: str
    weights_file: str
    checkpoint_sha256: str
    config: PupuJepaConfig


PUPUJEPA_CHECKPOINT_SPECS: Mapping[PupuJepaVariant, PupuJepaCheckpointSpec] = MappingProxyType(
    {
        "tiny": PupuJepaCheckpointSpec(
            args_file=PUPUJEPA_TINY_ARGS_FILE,
            weights_file=PUPUJEPA_TINY_WEIGHTS_FILE,
            checkpoint_sha256=PUPUJEPA_TINY_CHECKPOINT_SHA256,
            config=PUPUJEPA_TINY_CONFIG,
        ),
        "large": PupuJepaCheckpointSpec(
            args_file=PUPUJEPA_LARGE_ARGS_FILE,
            weights_file=PUPUJEPA_LARGE_WEIGHTS_FILE,
            checkpoint_sha256=PUPUJEPA_LARGE_CHECKPOINT_SHA256,
            config=PUPUJEPA_LARGE_CONFIG,
        ),
    }
)


class _PreprocessArgs(BaseModel):
    """Validate the shape-defining checkpoint frontend values.

    .. attribute :: model_config

        Pydantic model configuration.

    .. attribute :: sample_rate

        Frontend sample rate in Hz.

    .. attribute :: cut_mel_frame

        Training crop length in mel frames.

    .. attribute :: hop_size

        STFT hop length in samples.

    .. attribute :: n_fft

        FFT size in samples.

    .. attribute :: win_size

        Hann window length in samples.

    .. attribute :: fmin

        Lowest mel frequency in Hz.

    .. attribute :: fmax

        Highest mel frequency in Hz.

    .. attribute :: n_mels

        Number of mel bands.

    .. attribute :: normalize

        Whether training-set mel normalization is enabled.

    .. attribute :: flip_ft

        Whether checkpoint images are time-major.
    """

    model_config = ConfigDict(strict=True, extra="ignore")

    sample_rate: int
    cut_mel_frame: int
    hop_size: int
    n_fft: int
    win_size: int
    fmin: int | float
    fmax: int | float
    n_mels: int
    normalize: bool
    flip_ft: bool


class _ModelArgs(BaseModel):
    """Validate the teacher architecture values used by inference.

    .. attribute :: model_config

        Pydantic model configuration.

    .. attribute :: image_size

        Training mel-image dimensions.

    .. attribute :: patch_size

        Time and frequency patch dimensions.

    .. attribute :: embed_dim

        Teacher hidden width.

    .. attribute :: depth

        Number of teacher transformer blocks.

    .. attribute :: num_heads

        Attention head count.

    .. attribute :: mlp_ratio

        Transformer MLP expansion ratio.

    .. attribute :: drop_path_rate

        Stochastic-depth rate.

    .. attribute :: drop_path_uniform

        Whether stochastic depth is uniform across blocks.

    .. attribute :: use_swiglu

        Whether teacher blocks use SwiGLU MLPs.

    .. attribute :: layer_scale_init_value

        Optional initial layer-scale value.

    .. attribute :: qk_norm

        Whether teacher attention normalizes queries and keys.

    .. attribute :: norm_layer

        Final normalization family.

    .. attribute :: frequency_first

        Whether patch tokens are frequency-major.

    .. attribute :: in_chans

        Mel-image channel count.
    """

    model_config = ConfigDict(strict=True, extra="ignore")

    image_size: list[int]
    patch_size: list[int]
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    drop_path_rate: float
    drop_path_uniform: bool
    use_swiglu: bool
    layer_scale_init_value: float | None
    qk_norm: bool
    norm_layer: str
    frequency_first: bool
    in_chans: int


class _CheckpointArgs(BaseModel):
    """Validate the nested PupuJEPA checkpoint configuration.

    .. attribute :: model_config

        Pydantic model configuration.

    .. attribute :: preprocess

        Validated waveform frontend settings.

    .. attribute :: model

        Validated teacher architecture settings.
    """

    model_config = ConfigDict(strict=True, extra="ignore")

    preprocess: _PreprocessArgs
    model: _ModelArgs


def pupujepa_checkpoint_files(
    checkpoint_dir: Path,
    variant: PupuJepaVariant = "tiny",
) -> tuple[Path, Path]:
    """Locate one PupuJEPA variant under a snapshot or direct directory.

    :param checkpoint_dir: Materialized Hugging Face snapshot or local model directory.
    :param variant: Released teacher size to locate.
    :returns: Configuration and safetensors paths.
    :raises FileNotFoundError: Either required file is absent.
    """
    spec = PUPUJEPA_CHECKPOINT_SPECS[variant]
    candidates = (
        (checkpoint_dir / spec.args_file, checkpoint_dir / spec.weights_file),
        (checkpoint_dir / "args.json", checkpoint_dir / "model.safetensors"),
    )
    for args_path, weights_path in candidates:
        if args_path.is_file() and weights_path.is_file():
            return args_path, weights_path
    raise FileNotFoundError(
        f"PupuJEPA {variant} checkpoint lacks {spec.args_file!r} and "
        f"{spec.weights_file!r}: {checkpoint_dir}"
    )


def resolve_pupujepa_checkpoint(
    checkpoint: str = DEFAULT_PUPUJEPA_TINY_CHECKPOINT,
    revision: str = PUPUJEPA_CHECKPOINT_REVISION,
    variant: PupuJepaVariant = "tiny",
) -> Path:
    """Resolve the pinned Hugging Face snapshot or an explicit local checkpoint.

    :param checkpoint: Canonical Hugging Face repo id or local checkpoint directory.
    :param revision: Required immutable Hugging Face commit.
    :param variant: Released teacher size to resolve.
    :returns: Local directory containing the two inference artifacts.
    :raises ValueError: A remote source or revision is not the pinned identity.
    """
    spec = PUPUJEPA_CHECKPOINT_SPECS[variant]
    local = Path(checkpoint).expanduser()
    if local.is_dir():
        pupujepa_checkpoint_files(local, variant)
        return local.resolve()
    if checkpoint != DEFAULT_PUPUJEPA_TINY_CHECKPOINT:
        raise ValueError(
            f"PupuJEPA requires {DEFAULT_PUPUJEPA_TINY_CHECKPOINT!r} or a local directory, "
            f"got {checkpoint!r}"
        )
    if revision != PUPUJEPA_CHECKPOINT_REVISION:
        raise ValueError(
            f"PupuJEPA requires HF revision {PUPUJEPA_CHECKPOINT_REVISION}, got {revision}"
        )

    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=checkpoint,
            revision=revision,
            allow_patterns=[spec.args_file, spec.weights_file],
        )
    )
    artifacts = pupujepa_checkpoint_files(snapshot, variant)
    if snapshot.name != revision:
        raise ValueError(f"Hugging Face resolved PupuJEPA to {snapshot.name}, expected {revision}")
    actual_sha256 = checkpoint_files_sha256(snapshot, artifacts)
    if actual_sha256 != spec.checkpoint_sha256:
        raise ValueError(
            f"PupuJEPA {variant} checkpoint digest mismatch: expected "
            f"{spec.checkpoint_sha256}, got {actual_sha256}"
        )
    return snapshot


def load_pupujepa_config(
    checkpoint_dir: Path,
    variant: PupuJepaVariant = "tiny",
) -> PupuJepaConfig:
    """Parse and validate the frontend and teacher geometry in ``args.json``.

    :param checkpoint_dir: Materialized snapshot or direct model directory.
    :param variant: Released teacher size whose args should be parsed.
    :returns: Validated PupuJEPA architecture.
    :raises ValueError: The checkpoint requests an unsupported inference variant.
    """
    args_path, _ = pupujepa_checkpoint_files(checkpoint_dir, variant)
    args = _CheckpointArgs.model_validate(yaml.safe_load(args_path.read_text()))
    preprocess = args.preprocess
    model = args.model
    frontend_matches = (
        preprocess.normalize
        and preprocess.flip_ft
        and len(model.image_size) == 2
        and model.image_size == [preprocess.cut_mel_frame, preprocess.n_mels]
    )
    if not frontend_matches:
        raise ValueError(
            "PupuJEPA checkpoint does not match the normalized time-first frontend contract"
        )
    if len(model.patch_size) != 2:
        raise ValueError(f"PupuJEPA patch_size must have two dimensions, got {model.patch_size}")
    teacher_matches = (
        model.norm_layer == "layer"
        and not model.frequency_first
        and model.in_chans == 1
        and model.drop_path_rate == 0.0
        and not model.drop_path_uniform
        and model.layer_scale_init_value is None
    )
    if not teacher_matches:
        raise ValueError("PupuJEPA checkpoint requests an unsupported teacher architecture")
    return PupuJepaConfig(
        sample_rate=preprocess.sample_rate,
        n_fft=preprocess.n_fft,
        win_length=preprocess.win_size,
        hop_length=preprocess.hop_size,
        n_mels=preprocess.n_mels,
        fmin=float(preprocess.fmin),
        fmax=float(preprocess.fmax),
        mel_mean=PUPUJEPA_TINY_CONFIG.mel_mean,
        mel_std=PUPUJEPA_TINY_CONFIG.mel_std,
        patch_time=model.patch_size[0],
        patch_frequency=model.patch_size[1],
        embed_dim=model.embed_dim,
        depth=model.depth,
        num_heads=model.num_heads,
        mlp_ratio=model.mlp_ratio,
        use_swiglu=model.use_swiglu,
        qk_norm=model.qk_norm,
    )


def pupujepa_num_time_patches(
    num_samples: int,
    sample_rate: int,
    config: PupuJepaConfig = PUPUJEPA_TINY_CONFIG,
) -> int:
    """Return temporal teacher patches after resampling and reflection padding.

    :param num_samples: Positive source waveform length.
    :param sample_rate: Positive source sample rate in Hz.
    :param config: Shape-defining frontend configuration.
    :returns: Number of complete time patches.
    :raises ValueError: Inputs are non-positive or shorter than one complete time patch.
    """
    if num_samples < 1 or sample_rate < 1:
        raise ValueError(f"need positive num_samples/sample_rate, got {num_samples}/{sample_rate}")
    resampled_samples = math.ceil(num_samples * config.sample_rate / sample_rate)
    padded_samples = resampled_samples + 2 * config.reflection_padding
    mel_frames = 1 + (padded_samples - config.n_fft) // config.hop_length
    time_patches = mel_frames // config.patch_time
    if time_patches < 1:
        minimum_samples = config.patch_time * config.hop_length
        raise ValueError(
            f"PupuJEPA needs one complete time patch ({minimum_samples} samples at "
            f"{config.sample_rate} Hz), got {resampled_samples}"
        )
    return time_patches


def pupujepa_artifact_digest(
    checkpoint: str,
    variant: PupuJepaVariant = "tiny",
) -> str:
    """Identify the pinned source and materialized teacher artifacts.

    :param checkpoint: Canonical Hugging Face repo id or local checkpoint directory.
    :param variant: Released teacher size to identify.
    :returns: Version string suitable for embedding policy metadata.
    """
    spec = PUPUJEPA_CHECKPOINT_SPECS[variant]
    checkpoint_dir = resolve_pupujepa_checkpoint(checkpoint, variant=variant)
    load_pupujepa_config(checkpoint_dir, variant)
    artifacts = pupujepa_checkpoint_files(checkpoint_dir, variant)
    if checkpoint == DEFAULT_PUPUJEPA_TINY_CHECKPOINT:
        identity = (
            f"hf:{checkpoint}@{PUPUJEPA_CHECKPOINT_REVISION};sha256:{spec.checkpoint_sha256}"
        )
    else:
        identity = f"local:sha256:{checkpoint_files_sha256(checkpoint_dir, artifacts)}"
    return f"source:{PUPUJEPA_UPSTREAM_COMMIT};checkpoint:{identity}"
