"""Pinned-source adapter for TinyMU's frozen MATPAC audio encoder.

TinyMU has no detected license file, so this module does not redistribute its source. Runtime
loading requires an external checkout at the exact recorded commit plus the hash-pinned checkpoint.
Usage::

    encoder = load_tinymu_audio_encoder(source_dir=Path(os.environ["TINYMU_SOURCE_DIR"]))
    embeddings = encoder(audio_batch, sample_rate)
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, Self, cast

import numpy as np
import structlog

from synth_setter.model_cache import embedding_model_dir
from synth_setter.pipeline import r2_io
from synth_setter.utils.logging_utils import resolve_git_sha

if TYPE_CHECKING:
    import torch

logger = structlog.get_logger(__name__)

TINYMU_SOURCE_COMMIT = "eadbe2fc96cbbb5cdb9f91604c7a4e63782e6e7b"
TINYMU_SOURCE_MODEL_PATH = Path("src/models/matpac/model.py")
TINYMU_SOURCE_DIR_ENV = "TINYMU_SOURCE_DIR"
TINYMU_CHECKPOINT_REVISION = "0735fc50bc8b881d687dedccdd48b742927611b3"
TINYMU_CHECKPOINT_NAME = "matpac_plus_as_48_1_map_enconly.pt"
TINYMU_CHECKPOINT_SHA256 = "e8cec6847b2d918c8f77f82d79d90adf7dd82f99e80fa12eb3444f87f24bb998"
DEFAULT_TINYMU_CHECKPOINT = (
    "r2://intermediate-data/tinymu/source/pretrained/AndreasXi/TinyMU/"
    f"{TINYMU_CHECKPOINT_REVISION}/{TINYMU_CHECKPOINT_NAME}"
)

TINYMU_ENCODE_MAX_BATCH = 16


@dataclass(frozen=True)
class _TinyMUFrontendConfig:
    """Shape-defining MATPAC frontend contract.

    .. attribute :: sample_rate

        Model sample rate in Hz.

    .. attribute :: n_fft

        STFT window length in samples.

    .. attribute :: hop_length

        STFT hop length in samples.

    .. attribute :: patch_size

        Time/frequency patch width.

    .. attribute :: n_mels

        Mel-bin count.

    .. attribute :: base_embedding_dim

        Per-frequency-patch width.

    .. attribute :: unit_frames

        Precise-mode frame unit.

    .. attribute :: encoder_depth

        Transformer block count.
    """

    sample_rate: int
    n_fft: int
    hop_length: int
    patch_size: int
    n_mels: int
    base_embedding_dim: int
    unit_frames: int
    encoder_depth: int

    @property
    def embedding_dim(self) -> int:
        """Return the concatenated frequency-patch width."""
        return self.n_mels // self.patch_size * self.base_embedding_dim

    @property
    def min_input_samples(self) -> int:
        """Return the shortest waveform producing one complete patch."""
        return self.n_fft + (self.patch_size - 1) * self.hop_length


TINYMU_FRONTEND = _TinyMUFrontendConfig(
    sample_rate=16_000,
    n_fft=400,
    hop_length=160,
    patch_size=16,
    n_mels=80,
    base_embedding_dim=768,
    unit_frames=992,
    encoder_depth=12,
)

_TINYMU_MODULE_NAME = "_synth_setter_external_tinymu_matpac"
_EXPECTED_UNPERSISTED_BUFFERS = frozenset(
    {
        "log_mel.MelSpectrogram.mel_scale.fb",
        "log_mel.MelSpectrogram.spectrogram.window",
    }
)

type TinyMUEncodeFn = Callable[[np.ndarray, int], np.ndarray]


class _EncoderConfig(Protocol):
    """Shape-defining upstream encoder settings.

    .. attribute :: depth

        Transformer block count.

    .. attribute :: embed_dim

        Per-frequency-patch width.
    """

    depth: int
    embed_dim: int


class _MatpacConfig(Protocol):
    """Measured subset of the external MATPAC configuration.

    .. attribute :: encoder

        Encoder architecture settings.

    .. attribute :: n_freq

        Mel-bin count.

    .. attribute :: n_t

        Precise-mode frame unit.

    .. attribute :: patch_size

        Time/frequency patch width.

    .. attribute :: sr

        Model sample rate in Hz.
    """

    encoder: _EncoderConfig
    n_freq: int
    n_t: int
    patch_size: int
    sr: int


class _IncompatibleState(Protocol):
    """State-loading result inspected by the adapter.

    .. attribute :: missing_keys

        Model keys absent from the checkpoint.

    .. attribute :: unexpected_keys

        Checkpoint keys absent from the model.
    """

    missing_keys: list[str]
    unexpected_keys: list[str]


class _MatpacModel(Protocol):
    """Narrow runtime surface required from external MATPAC.

    .. attribute :: cfg

        Shape-defining model configuration.
    """

    cfg: _MatpacConfig

    def __call__(self, inputs: torch.Tensor) -> tuple[torch.Tensor, object]: ...

    def eval(self) -> Self: ...

    def load_state_dict(
        self, state_dict: Mapping[str, torch.Tensor], *, strict: bool
    ) -> _IncompatibleState: ...

    def requires_grad_(self, requires_grad: bool = True) -> Self: ...

    def to(self, device: str) -> Self: ...


class _MatpacModule(Protocol):
    """Constructor exported by the pinned external module."""

    def matpac_wrapper(
        self, *, inference_type: str, pull_time_dimension: bool
    ) -> _MatpacModel: ...


def _file_sha256(path: Path) -> str:
    """Hash a model artifact without loading it into memory.

    :param path: File whose strong identity is required.
    :returns: Lowercase SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_checkpoint(path: Path) -> Path:
    """Require the trusted MATPAC artifact digest at ``path``.

    :param path: Candidate checkpoint.
    :returns: Resolved checkpoint path.
    :raises FileNotFoundError: The candidate is not a file.
    :raises ValueError: The SHA-256 digest differs from the pinned artifact.
    """
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"TinyMU checkpoint does not exist: {resolved}")
    actual = _file_sha256(resolved)
    if actual != TINYMU_CHECKPOINT_SHA256:
        raise ValueError(
            f"TinyMU checkpoint SHA-256 is {actual}, expected {TINYMU_CHECKPOINT_SHA256}: "
            f"{resolved}"
        )
    return resolved


def resolve_tinymu_checkpoint(checkpoint: str = DEFAULT_TINYMU_CHECKPOINT) -> Path:
    """Resolve and strongly verify the immutable MATPAC checkpoint.

    :param checkpoint: Exact pinned R2 URI or a local copy with the pinned SHA-256.
    :returns: Verified local checkpoint path.
    :raises ValueError: An R2 URI is not the pinned identity.
    """
    if not r2_io.is_r2_uri(checkpoint):
        return _verified_checkpoint(Path(checkpoint))
    if checkpoint != DEFAULT_TINYMU_CHECKPOINT:
        raise ValueError(
            f"TinyMU requires the pinned TinyMU checkpoint URI {DEFAULT_TINYMU_CHECKPOINT!r}, "
            f"got {checkpoint!r}"
        )

    cache_dir = embedding_model_dir(f"tinymu-{TINYMU_CHECKPOINT_REVISION}")
    destination = cache_dir / TINYMU_CHECKPOINT_NAME
    if destination.exists():
        return _verified_checkpoint(destination)

    cache_dir.mkdir(parents=True, exist_ok=True)
    r2_io.ensure_r2_env_loaded()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache_dir, prefix=f".{TINYMU_CHECKPOINT_NAME}.", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        # The canonical R2 helper owns bounded retries and transfer I/O timeouts.
        r2_io.download_to_path(checkpoint, temporary_path)
        _verified_checkpoint(temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return _verified_checkpoint(destination)


def _git_output(source_dir: Path, *arguments: str) -> str:
    r"""Run a read-only git identity query in the external checkout.

    :param source_dir: TinyMU repository root.
    :param \*arguments: Git arguments after ``-C <root>``.
    :returns: Stripped stdout.
    :raises ValueError: Git cannot resolve the requested identity.
    """
    result = subprocess.run(  # noqa: S603 — executable and arguments are fixed identity probes
        ["git", "-C", str(source_dir), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"TinyMU source identity probe failed in {source_dir}: {detail}")
    return result.stdout.strip()


def resolve_tinymu_source_model(source_dir: Path) -> Path:
    """Validate the external TinyMU checkout and return its MATPAC model module.

    :param source_dir: External TinyMU repository root.
    :returns: Source model path whose bytes match the pinned commit blob.
    :raises FileNotFoundError: The checkout or model module is absent.
    :raises ValueError: HEAD or the working-tree model differs from the pinned source.
    """
    root = source_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"TinyMU source checkout does not exist: {root}")
    model_path = root / TINYMU_SOURCE_MODEL_PATH
    if not model_path.is_file():
        raise FileNotFoundError(f"TinyMU MATPAC source does not exist: {model_path}")

    head = _git_output(root, "rev-parse", "HEAD")
    if head != TINYMU_SOURCE_COMMIT:
        raise ValueError(f"TinyMU source HEAD is {head}, expected {TINYMU_SOURCE_COMMIT}: {root}")
    expected_blob = _git_output(root, "rev-parse", f"{TINYMU_SOURCE_COMMIT}:{TINYMU_SOURCE_MODEL_PATH}")
    actual_blob = _git_output(root, "hash-object", str(model_path))
    if actual_blob != expected_blob:
        raise ValueError(f"TinyMU MATPAC source differs from pinned commit: {model_path}")
    return model_path


def configured_tinymu_source_model(source_dir: Path | None) -> Path:
    """Resolve the explicit config path or documented environment fallback.

    :param source_dir: Configured checkout root, or ``None`` to read ``TINYMU_SOURCE_DIR``.
    :returns: Validated MATPAC source module path.
    :raises FileNotFoundError: No source boundary is configured.
    """
    configured = source_dir
    if configured is None:
        environment_path = os.environ.get(TINYMU_SOURCE_DIR_ENV)
        configured = Path(environment_path) if environment_path else None
    if configured is None:
        raise FileNotFoundError(
            "TinyMU source is not redistributed because upstream has no detected license file; "
            f"set tinymu_source_dir or {TINYMU_SOURCE_DIR_ENV} to the checkout at "
            f"{TINYMU_SOURCE_COMMIT}"
        )
    return resolve_tinymu_source_model(configured)


def _load_source_module(model_path: Path) -> ModuleType:
    """Load only the pinned upstream MATPAC module from its external checkout.

    :param model_path: Path returned by :func:`resolve_tinymu_source_model`.
    :returns: Executed upstream module.
    :raises ImportError: Python cannot construct or execute the module.
    """
    spec = importlib.util.spec_from_file_location(_TINYMU_MODULE_NAME, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load TinyMU MATPAC source module: {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_TINYMU_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        sys.modules.pop(_TINYMU_MODULE_NAME, None)
        raise ImportError(
            "loading TinyMU requires the optional `tinymu` extra — "
            "install it with `uv sync --extra tinymu`"
        ) from exc
    return module


def _validate_model_contract(model: _MatpacModel) -> None:
    """Reject upstream source whose runtime architecture differs from the measured contract.

    :param model: Instantiated external MATPAC module.
    :raises ValueError: A shape-defining model setting has changed.
    """
    cfg = model.cfg
    actual = {
        "depth": cfg.encoder.depth,
        "embed_dim": cfg.encoder.embed_dim,
        "n_freq": cfg.n_freq,
        "n_t": cfg.n_t,
        "patch_size": cfg.patch_size,
        "sample_rate": cfg.sr,
    }
    expected = {
        "depth": TINYMU_FRONTEND.encoder_depth,
        "embed_dim": TINYMU_FRONTEND.base_embedding_dim,
        "n_freq": TINYMU_FRONTEND.n_mels,
        "n_t": TINYMU_FRONTEND.unit_frames,
        "patch_size": TINYMU_FRONTEND.patch_size,
        "sample_rate": TINYMU_FRONTEND.sample_rate,
    }
    if actual != expected:
        raise ValueError(f"TinyMU MATPAC architecture is {actual}, expected {expected}")


def tinymu_num_latent_frames(num_samples: int, sample_rate: int) -> int:
    """Return MATPAC's precise-mode token count after resampling and patch padding.

    :param num_samples: Positive source clip length in samples.
    :param sample_rate: Positive source sample rate in Hz.
    :returns: Temporal token count for the full clip.
    :raises ValueError: Inputs are non-positive or too short for one MATPAC patch.
    """
    if num_samples < 1 or sample_rate < 1:
        raise ValueError(f"need positive num_samples/sample_rate, got {num_samples}/{sample_rate}")
    resampled_samples = math.ceil(num_samples * TINYMU_FRONTEND.sample_rate / sample_rate)
    if resampled_samples < TINYMU_FRONTEND.min_input_samples:
        raise ValueError(
            f"TinyMU needs at least {TINYMU_FRONTEND.min_input_samples} samples after "
            f"resampling, got {resampled_samples}"
        )
    mel_frames = 1 + (
        resampled_samples - TINYMU_FRONTEND.n_fft
    ) // TINYMU_FRONTEND.hop_length
    return math.ceil(mel_frames / TINYMU_FRONTEND.patch_size)


def tinymu_encoder_input(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Prepare ``(B, C, T)`` audio as finite float32 mono at 16 kHz.

    :param audio: Audio batch with one or two channels.
    :param sample_rate: Positive source sample rate in Hz.
    :returns: Contiguous ``(B, T_16k)`` MATPAC input.
    :raises ValueError: Audio shape, rate, values, or duration is incompatible.
    """
    if audio.ndim != 3:
        raise ValueError(f"expected a (B, C, T) batch for TinyMU, got shape {audio.shape}")
    if audio.shape[0] < 1:
        raise ValueError("TinyMU expects a non-empty batch")
    if audio.shape[1] not in (1, 2):
        raise ValueError(f"TinyMU expects 1 or 2 channels, got shape {audio.shape}")
    if sample_rate < 1:
        raise ValueError(f"TinyMU needs a positive sample_rate, got {sample_rate}")
    if not np.isfinite(audio).all():
        raise ValueError("TinyMU input audio contains non-finite values")
    peak_amplitude = float(np.max(np.abs(audio)))
    if peak_amplitude > 1.0:
        raise ValueError(
            f"TinyMU input audio is outside [-1.0, 1.0]: peak amplitude {peak_amplitude}"
        )

    tinymu_num_latent_frames(audio.shape[-1], sample_rate)
    mono = np.ascontiguousarray(audio.mean(axis=1, dtype=np.float32))
    if sample_rate == TINYMU_FRONTEND.sample_rate:
        return mono

    import torch
    import torchaudio.functional as audio_fn

    resampled = audio_fn.resample(
        torch.from_numpy(mono), sample_rate, TINYMU_FRONTEND.sample_rate
    )
    return np.ascontiguousarray(resampled.numpy(), dtype=np.float32)


def _load_tinymu_model(
    module: _MatpacModule, checkpoint_path: Path, device: str
) -> _MatpacModel:
    """Construct and freeze the pinned MATPAC architecture and state.

    :param module: Verified external module constructor.
    :param checkpoint_path: SHA-256-verified model state.
    :param device: Torch inference device.
    :returns: Frozen eval-mode MATPAC model.
    :raises ValueError: State keys or architecture violate the pinned contract.
    """
    import torch

    model = module.matpac_wrapper(inference_type="precise", pull_time_dimension=False)
    _validate_model_contract(model)
    state = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != _EXPECTED_UNPERSISTED_BUFFERS or unexpected:
        raise ValueError(
            f"TinyMU checkpoint state is incompatible: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return model.to(device).eval().requires_grad_(False)


def _encode_tinymu_chunk(
    model: _MatpacModel, chunk: np.ndarray, device: str
) -> np.ndarray:
    """Encode prepared ``(B, T_16k)`` audio as ``(B, 3840, T_tokens)``.

    :param model: Frozen MATPAC model.
    :param chunk: Finite normalized mono audio at 16 kHz.
    :param device: Torch inference device.
    :returns: Finite contiguous float32 embedding sequences.
    :raises ValueError: MATPAC returns an invalid shape or non-finite values.
    """
    import torch

    with torch.inference_mode():
        embeddings, _ = model(torch.from_numpy(chunk).to(device))
    expected = (
        len(chunk),
        tinymu_num_latent_frames(chunk.shape[-1], TINYMU_FRONTEND.sample_rate),
        TINYMU_FRONTEND.embedding_dim,
    )
    if tuple(embeddings.shape) != expected:
        raise ValueError(f"TinyMU encoder produced shape {tuple(embeddings.shape)}, expected {expected}")
    values = embeddings.float().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("TinyMU encoder produced non-finite values")
    return np.ascontiguousarray(values.transpose(0, 2, 1), dtype=np.float32)


def load_tinymu_audio_encoder(
    checkpoint: str = DEFAULT_TINYMU_CHECKPOINT,
    *,
    source_dir: Path | None = None,
    device: str = "cpu",
) -> TinyMUEncodeFn:
    """Load frozen MATPAC weights through the pinned external TinyMU source boundary.

    :param checkpoint: Exact pinned R2 URI or a hash-identical local file.
    :param source_dir: External TinyMU checkout root, or ``None`` for the env fallback.
    :param device: Explicit Torch device.
    :returns: Encoder accepting finite normalized ``(B, C, T)`` audio and returning
        ``(B, 3840, T_tokens)`` float32 sequences.
    """
    model_path = configured_tinymu_source_model(source_dir)
    checkpoint_path = resolve_tinymu_checkpoint(checkpoint)
    module = cast("_MatpacModule", _load_source_module(model_path))
    model = _load_tinymu_model(module, checkpoint_path, device)
    logger.info(
        "loaded_tinymu_checkpoint",
        checkpoint_revision=TINYMU_CHECKPOINT_REVISION,
        checkpoint_sha256=TINYMU_CHECKPOINT_SHA256,
        device=device,
        source_commit=TINYMU_SOURCE_COMMIT,
        synth_setter_git_sha=resolve_git_sha(),
    )

    def encode(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Encode normalized ``(B, C, T)`` audio as finite ``(B, 3840, T_tokens)``.

        :param audio: One- or two-channel batch with amplitudes in ``[-1, 1]``.
        :param sample_rate: Positive source sample rate in Hz.
        :returns: Contiguous float32 embedding sequences.
        """
        prepared = tinymu_encoder_input(audio, sample_rate)
        chunks = [
            _encode_tinymu_chunk(
                model,
                prepared[start : start + TINYMU_ENCODE_MAX_BATCH],
                device,
            )
            for start in range(0, len(prepared), TINYMU_ENCODE_MAX_BATCH)
        ]
        return np.concatenate(chunks, axis=0)

    return encode
