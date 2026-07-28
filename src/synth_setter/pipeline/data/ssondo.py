"""Pin and adapt S-SONDO for ``embeddings=[ssondo]`` CLI runs.

Typical usage::

    synth-setter-add-embeddings lance_uri=DATASET.lance embeddings=[ssondo]
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import structlog

from synth_setter.model_cache import embedding_model_dir

logger = structlog.get_logger(__name__)

DEFAULT_SSONDO_CHECKPOINT = "mohammedali2501/ssondo"
SSONDO_CHECKPOINT_REVISION = "afc946ee816eb2287b62c7cadadd59e507996b23"
SSONDO_CHECKPOINT_NAME = "matpac_mobilenetv3.ckpt"
SSONDO_CHECKPOINT_SHA256 = "87cff558a9a442e97d630a79f391bce8663d31a3adcbbf0b0a8cc41cb41854fc"
SSONDO_SOURCE_REVISION = "231d260cc1be2eb93b060accc8bfa218feff3a74"
SSONDO_EMBEDDING_DIM = 960
SSONDO_SAMPLE_RATE = 32_000
SSONDO_WINDOW_SECONDS = 10
SSONDO_INPUT_SAMPLES = SSONDO_SAMPLE_RATE * SSONDO_WINDOW_SECONDS
SSONDO_ENCODE_MAX_BATCH = 16

type SSONDOEncodeFn = Callable[[np.ndarray, int], np.ndarray]


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
    """Require the trusted S-SONDO artifact digest at ``path``.

    :param path: Candidate checkpoint.
    :returns: Resolved checkpoint path.
    :raises FileNotFoundError: The candidate is not a file.
    :raises ValueError: The SHA-256 digest differs from the pinned artifact.
    """
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"S-SONDO checkpoint does not exist: {resolved}")
    actual = _file_sha256(resolved)
    if actual != SSONDO_CHECKPOINT_SHA256:
        raise ValueError(
            f"S-SONDO checkpoint SHA-256 is {actual}, expected {SSONDO_CHECKPOINT_SHA256}: "
            f"{resolved}"
        )
    return resolved


def resolve_ssondo_checkpoint(checkpoint: str = DEFAULT_SSONDO_CHECKPOINT) -> Path:
    """Resolve and strongly verify the immutable S-SONDO checkpoint.

    :param checkpoint: Pinned Hugging Face repo id or a hash-identical local file.
    :returns: Verified local checkpoint path.
    :raises ValueError: A non-local checkpoint id is not the pinned repository.
    """
    local = Path(checkpoint).expanduser()
    if local.is_file():
        return _verified_checkpoint(local)
    if checkpoint != DEFAULT_SSONDO_CHECKPOINT:
        raise ValueError(
            f"S-SONDO requires the pinned S-SONDO checkpoint repo "
            f"{DEFAULT_SSONDO_CHECKPOINT!r}, got {checkpoint!r}"
        )

    from huggingface_hub import hf_hub_download

    cache_dir = embedding_model_dir(f"ssondo-{SSONDO_CHECKPOINT_REVISION}")
    # Hugging Face Hub owns 429/5xx/network backoff and atomic cache writes.
    path = hf_hub_download(
        repo_id=DEFAULT_SSONDO_CHECKPOINT,
        filename=SSONDO_CHECKPOINT_NAME,
        revision=SSONDO_CHECKPOINT_REVISION,
        local_dir=cache_dir,
    )
    return _verified_checkpoint(Path(path))


def _require_normalized(audio: np.ndarray) -> None:
    """Reject waveforms outside S-SONDO's normalized input range.

    :param audio: Candidate waveform values.
    :raises ValueError: Any value falls outside ``[-1, 1]``.
    """
    float_audio = np.asarray(audio, dtype=np.float32)
    if np.abs(float_audio).max() > 1.0:
        raise ValueError("S-SONDO input audio must be normalized to [-1, 1]")


def ssondo_encoder_input(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Prepare audio as one finite mono 32 kHz S-SONDO window.

    :param audio: ``(B, C, T)`` audio with one or two channels.
    :param sample_rate: Positive source sample rate in Hz.
    :returns: Contiguous ``(B, 320000)`` float32 waveforms.
    :raises ValueError: Audio shape, rate, values, or duration is incompatible.
    """
    if audio.ndim != 3:
        raise ValueError(f"expected a (B, C, T) batch for S-SONDO, got shape {audio.shape}")
    if audio.shape[1] not in (1, 2):
        raise ValueError(f"S-SONDO expects 1 or 2 channels, got shape {audio.shape}")
    if sample_rate < 1:
        raise ValueError(f"S-SONDO needs a positive sample_rate, got {sample_rate}")
    if audio.shape[-1] < 1:
        raise ValueError("S-SONDO input audio must be non-empty")
    if not np.isfinite(audio).all():
        raise ValueError("S-SONDO input audio contains non-finite values")
    _require_normalized(audio)

    source_exceeds_window = (
        audio.shape[-1] * SSONDO_SAMPLE_RATE > SSONDO_INPUT_SAMPLES * sample_rate
    )
    if source_exceeds_window:
        raise ValueError("S-SONDO accepts at most 10 seconds of audio")

    mono = np.ascontiguousarray(audio.mean(axis=1, dtype=np.float32))
    if sample_rate != SSONDO_SAMPLE_RATE:
        import torch
        import torchaudio.functional as audio_fn

        mono = audio_fn.resample(torch.from_numpy(mono), sample_rate, SSONDO_SAMPLE_RATE).numpy()
        np.clip(mono, -1.0, 1.0, out=mono)
    padding = SSONDO_INPUT_SAMPLES - mono.shape[-1]
    return np.ascontiguousarray(np.pad(mono, ((0, 0), (0, padding))), dtype=np.float32)


def load_ssondo_audio_encoder(
    checkpoint: str = DEFAULT_SSONDO_CHECKPOINT,
    device: str = "cpu",
) -> SSONDOEncodeFn:
    """Load S-SONDO and return an encoder over source audio batches.

    :param checkpoint: Pinned Hugging Face repo id or a hash-identical local file.
    :param device: Explicit Torch device.
    :returns: Encoder producing ``(B, 960)`` float32 vectors.
    """
    import torch
    from importlib.metadata import version
    from ssondo import get_ssondo

    checkpoint_path = resolve_ssondo_checkpoint(checkpoint)
    model = get_ssondo(str(checkpoint_path), device=device)
    model = model.eval().requires_grad_(False)
    logger.info(
        "loaded_ssondo_checkpoint",
        checkpoint_repo=DEFAULT_SSONDO_CHECKPOINT,
        checkpoint_revision=SSONDO_CHECKPOINT_REVISION,
        checkpoint_sha256=SSONDO_CHECKPOINT_SHA256,
        package_version=version("ssondo"),
        source_revision=SSONDO_SOURCE_REVISION,
        device=device,
    )

    @torch.inference_mode()
    def _encode_chunk(chunk: np.ndarray) -> np.ndarray:
        inputs = torch.from_numpy(chunk).to(device)
        embeddings = model.get_embeddings(inputs)
        expected_shape = (len(chunk), SSONDO_EMBEDDING_DIM)
        if tuple(embeddings.shape) != expected_shape:
            raise ValueError(
                f"S-SONDO encoder produced shape {tuple(embeddings.shape)}, "
                f"expected {expected_shape}"
            )
        values = embeddings.float().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("S-SONDO encoder produced non-finite values")
        return np.ascontiguousarray(values, dtype=np.float32)

    def encode(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        prepared = ssondo_encoder_input(audio, sample_rate)
        chunks = [
            _encode_chunk(prepared[start : start + SSONDO_ENCODE_MAX_BATCH])
            for start in range(0, len(prepared), SSONDO_ENCODE_MAX_BATCH)
        ]
        return np.concatenate(chunks, axis=0)

    return encode
