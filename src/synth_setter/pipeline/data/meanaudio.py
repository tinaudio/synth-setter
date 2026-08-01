"""MeanAudio 16 kHz MMAudio VAE adapter for offline audio embeddings.

The adapter imports the pinned upstream package directly, loads only its mel frontend and VAE, and
stores deterministic posterior means without loading a vocoder or flow model. Typical use is
``load_meanaudio_audio_encoder()(audio, sample_rate)``.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, cast

import numpy as np
import structlog
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from synth_setter.data.vst.shapes import AUDIO_FIELD, MEANAUDIO_16K_FIELD
from synth_setter.utils.logging_utils import resolve_git_sha

if TYPE_CHECKING:
    import pyarrow as pa
    import torch

logger = structlog.get_logger(__name__)

MEANAUDIO_PACKAGE_COMMIT = "8740a3e8df4c891a8d9deee1f820d051584d2671"
MEANAUDIO_CHECKPOINT_REPO = "AndreasXi/MeanAudio"
MEANAUDIO_CHECKPOINT_REVISION = "6e072062d4f9af21c647e2bae5aafc1da2c84014"
MEANAUDIO_CHECKPOINT_NAME = "v1-16.pth"
MEANAUDIO_CHECKPOINT_SHA256 = "15ad082c714ccf3771898a771fc6eebdc1d9c8d5c6154726906a97f43603d62c"
DEFAULT_MEANAUDIO_CHECKPOINT = MEANAUDIO_CHECKPOINT_REPO
MEANAUDIO_SAMPLE_RATE = 16_000
MEANAUDIO_EMBEDDING_DIM = 20
MEANAUDIO_INDEX_SUB_VECTORS = 4
MEANAUDIO_MEL_HOP_LENGTH = 256
MEANAUDIO_VAE_DOWNSAMPLE = 2
MEANAUDIO_ENCODE_MAX_BATCH = 4


type MeanAudioEncodeFn = Callable[[np.ndarray, int], np.ndarray]


class _Posterior(Protocol):
    """Posterior surface required from the upstream VAE."""

    def mode(self) -> torch.Tensor:
        """Return the deterministic posterior mean.

        :returns: Posterior mean tensor.
        """
        ...


class _MeanAudioVAE(Protocol):
    """Narrow upstream VAE surface used by the adapter.

    .. attribute :: decoder

        Decoder removed after strict state loading.
    """

    decoder: object

    def encode(self, mel: torch.Tensor) -> _Posterior:
        """Encode canonical mel tensors into a Gaussian posterior.

        :param mel: MeanAudio-normalized mel tensors.
        :returns: Latent posterior.
        """
        ...

    def eval(self) -> Self:
        """Select evaluation mode.

        :returns: This model.
        """
        ...

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> object:
        """Load checkpoint state.

        :param state_dict: Complete VAE checkpoint state.
        :param strict: Whether every model and checkpoint key must match.
        :param assign: Whether checkpoint tensors replace meta-device parameters.
        :returns: Upstream state-loading result.
        """
        ...

    def remove_weight_norm(self) -> Self:
        """Materialize the inference-time normalized weights.

        :returns: This model.
        """
        ...

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        """Set parameter gradient requirements.

        :param requires_grad: Whether parameters require gradients.
        :returns: This model.
        """
        ...

    def to(self, device: str) -> Self:
        """Move the VAE to an inference device.

        :param device: Torch device selector.
        :returns: This model.
        """
        ...


class _MelConverter(Protocol):
    """Canonical upstream mel-converter surface used by the adapter."""

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert mono waveform rows to canonical mel tensors.

        :param waveform: ``(B, T)`` waveform tensor at 16 kHz.
        :returns: ``(B, 80, mel_frames)`` mel tensor.
        """
        ...

    def eval(self) -> Self:
        """Select evaluation mode.

        :returns: This converter.
        """
        ...

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        """Set parameter gradient requirements.

        :param requires_grad: Whether parameters require gradients.
        :returns: This converter.
        """
        ...

    def to(self, device: str) -> Self:
        """Move frontend buffers to an inference device.

        :param device: Torch device selector.
        :returns: This converter.
        """
        ...


def _file_sha256(path: Path) -> str:
    """Hash one checkpoint without loading it into memory.

    :param path: File whose identity is required.
    :returns: Lowercase SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_checkpoint(path: Path) -> Path:
    """Require the pinned MeanAudio checkpoint digest.

    :param path: Candidate local checkpoint.
    :returns: Resolved verified path.
    :raises FileNotFoundError: The candidate is not a file.
    :raises ValueError: Its SHA-256 differs from the pinned checkpoint.
    """
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"MeanAudio checkpoint does not exist: {resolved}")
    actual = _file_sha256(resolved)
    if actual != MEANAUDIO_CHECKPOINT_SHA256:
        raise ValueError(
            f"MeanAudio checkpoint SHA-256 is {actual}, expected "
            f"{MEANAUDIO_CHECKPOINT_SHA256}: {resolved}"
        )
    return resolved


def _is_retryable_download_error(error: BaseException) -> bool:
    """Return whether a transient transport or service failure can succeed on retry.

    :param error: Download failure raised by Hugging Face Hub.
    :returns: Whether the failure is transient.
    """
    from httpx import TransportError
    from huggingface_hub.errors import HfHubHTTPError

    if isinstance(error, TimeoutError | ConnectionError | TransportError):
        return True
    if not isinstance(error, HfHubHTTPError) or error.response is None:
        return False
    return error.response.status_code in {408, 425, 429, 500, 502, 503, 504}


@retry(
    retry=retry_if_exception(_is_retryable_download_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def _download_meanaudio_checkpoint(download: Callable[..., str]) -> str:
    """Download the immutable checkpoint with bounded transient-failure retries.

    :param download: Hugging Face file downloader.
    :returns: Local cache path from the downloader.
    """
    return download(
        repo_id=MEANAUDIO_CHECKPOINT_REPO,
        revision=MEANAUDIO_CHECKPOINT_REVISION,
        filename=MEANAUDIO_CHECKPOINT_NAME,
    )


def resolve_meanaudio_checkpoint(
    checkpoint: str = DEFAULT_MEANAUDIO_CHECKPOINT,
) -> Path:
    """Resolve the pinned Hugging Face checkpoint or a hash-identical local copy.

    :param checkpoint: Exact pinned Hugging Face repo id or a local checkpoint path.
    :returns: Verified local checkpoint path.
    :raises ValueError: A non-local identity is not the pinned repository.
    """
    local = Path(checkpoint).expanduser()
    if local.is_file():
        return _verified_checkpoint(local)
    if checkpoint != DEFAULT_MEANAUDIO_CHECKPOINT:
        raise ValueError(
            f"MeanAudio requires the pinned Hugging Face repo "
            f"{DEFAULT_MEANAUDIO_CHECKPOINT!r} or a hash-identical local file, got "
            f"{checkpoint!r}"
        )

    from huggingface_hub import hf_hub_download

    downloaded = _download_meanaudio_checkpoint(hf_hub_download)
    return _verified_checkpoint(Path(downloaded))


def meanaudio_num_latent_frames(num_samples: int, sample_rate: int) -> int:
    """Return MeanAudio's latent width after resampling, mel conversion, and VAE downsampling.

    :param num_samples: Positive source clip length in samples.
    :param sample_rate: Positive source sample rate in Hz.
    :returns: Latent frame count.
    :raises ValueError: Inputs are non-positive or too short to produce a latent frame.
    """
    if num_samples < 1 or sample_rate < 1:
        raise ValueError(f"need positive num_samples/sample_rate, got {num_samples}/{sample_rate}")
    resampled_samples = math.ceil(num_samples * MEANAUDIO_SAMPLE_RATE / sample_rate)
    frames = resampled_samples // (MEANAUDIO_MEL_HOP_LENGTH * MEANAUDIO_VAE_DOWNSAMPLE)
    if frames < 1:
        raise ValueError(
            "MeanAudio needs at least "
            f"{MEANAUDIO_MEL_HOP_LENGTH * MEANAUDIO_VAE_DOWNSAMPLE} samples after "
            f"resampling, got {resampled_samples}"
        )
    return frames


def meanaudio_encoder_input(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Prepare finite normalized ``(B, C, T)`` audio as 16 kHz float32 mono.

    :param audio: One- or two-channel source audio.
    :param sample_rate: Positive source sample rate in Hz.
    :returns: Contiguous ``(B, T_16k)`` input for the canonical mel converter.
    :raises ValueError: Rank, batch, channels, rate, duration, range, or finiteness is invalid.
    """
    if audio.ndim != 3:
        raise ValueError(f"expected a (B, C, T) batch for MeanAudio, got shape {audio.shape}")
    if audio.shape[0] < 1:
        raise ValueError("MeanAudio expects a non-empty batch")
    if audio.shape[1] not in (1, 2):
        raise ValueError(f"MeanAudio expects 1 or 2 channels, got shape {audio.shape}")
    if sample_rate < 1:
        raise ValueError(f"MeanAudio needs a positive sample_rate, got {sample_rate}")
    meanaudio_num_latent_frames(audio.shape[-1], sample_rate)
    if not np.isfinite(audio).all():
        raise ValueError("MeanAudio input audio contains non-finite values")
    peak_amplitude = float(np.max(np.abs(audio)))
    if peak_amplitude > 1.0:
        raise ValueError(
            f"MeanAudio input audio is outside [-1.0, 1.0]: peak amplitude {peak_amplitude}"
        )

    mono = np.ascontiguousarray(audio.mean(axis=1, dtype=np.float32))
    if sample_rate == MEANAUDIO_SAMPLE_RATE:
        return mono

    import torch
    import torchaudio.functional as audio_fn

    resampled = audio_fn.resample(torch.from_numpy(mono), sample_rate, MEANAUDIO_SAMPLE_RATE)
    values = resampled.numpy()
    np.clip(values, -1.0, 1.0, out=values)
    return np.ascontiguousarray(values, dtype=np.float32)


def _load_meanaudio_vae(
    factory: Callable[[str], _MeanAudioVAE], checkpoint_path: Path, device: str
) -> _MeanAudioVAE:
    """Strict-load, prune, normalize, and freeze the upstream 16 kHz VAE.

    :param factory: Upstream ``get_my_vae`` constructor.
    :param checkpoint_path: SHA-256-verified VAE state.
    :param device: Torch inference device.
    :returns: Frozen encoder-only VAE in evaluation mode.
    """
    import torch

    with torch.device("meta"):
        vae = factory("16k")
    state = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=True,
        mmap=True,
    )
    vae.load_state_dict(state, strict=True, assign=True)
    del vae.decoder
    del state
    vae.remove_weight_norm()
    return vae.to(device).eval().requires_grad_(False)


def _encode_meanaudio_chunk(
    mel_converter: _MelConverter,
    vae: _MeanAudioVAE,
    chunk: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    """Encode prepared mono audio as deterministic channel-major posterior means.

    :param mel_converter: Canonical upstream 16 kHz mel frontend.
    :param vae: Frozen encoder-only MeanAudio VAE.
    :param chunk: ``(B, T_16k)`` finite normalized mono audio.
    :param device: Torch inference device.
    :returns: Contiguous float32 ``(B, 20, F)`` posterior means.
    :raises ValueError: Upstream output has the wrong shape or non-finite values.
    """
    import torch

    with torch.inference_mode():
        waveform = torch.from_numpy(chunk).to(device)
        latents = vae.encode(mel_converter(waveform)).mode()
    expected_shape = (
        len(chunk),
        MEANAUDIO_EMBEDDING_DIM,
        meanaudio_num_latent_frames(chunk.shape[-1], MEANAUDIO_SAMPLE_RATE),
    )
    if tuple(latents.shape) != expected_shape:
        raise ValueError(
            f"MeanAudio encoder produced shape {tuple(latents.shape)}, expected {expected_shape}"
        )
    values = np.ascontiguousarray(latents.float().cpu().numpy(), dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("MeanAudio encoder produced non-finite values")
    return values


def _encode_meanaudio_chunks(
    mel_converter: _MelConverter,
    vae: _MeanAudioVAE,
    prepared: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    """Encode prepared rows in bounded large-model batches.

    :param mel_converter: Canonical upstream 16 kHz mel frontend.
    :param vae: Frozen encoder-only MeanAudio VAE.
    :param prepared: ``(B, T_16k)`` finite normalized mono audio.
    :param device: Torch inference device.
    :returns: Contiguous float32 ``(B, 20, F)`` posterior means.
    """
    chunks = [
        _encode_meanaudio_chunk(
            mel_converter,
            vae,
            prepared[start : start + MEANAUDIO_ENCODE_MAX_BATCH],
            device=device,
        )
        for start in range(0, len(prepared), MEANAUDIO_ENCODE_MAX_BATCH)
    ]
    return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)


def load_meanaudio_audio_encoder(
    checkpoint: str = DEFAULT_MEANAUDIO_CHECKPOINT,
    *,
    device: str = "cpu",
) -> MeanAudioEncodeFn:
    """Load the frozen upstream MeanAudio mel frontend and encoder-only VAE.

    :param checkpoint: Pinned Hugging Face repo or a SHA-identical local checkpoint.
    :param device: Explicit Torch inference device.
    :returns: Encoder accepting ``(B, C, T)`` audio and returning contiguous float32
        ``(B, 20, F)`` posterior means.
    """
    from meanaudio.ext.autoencoder.vae import get_my_vae
    from meanaudio.ext.mel_converter import get_mel_converter

    checkpoint_path = resolve_meanaudio_checkpoint(checkpoint)
    mel_converter = cast("_MelConverter", get_mel_converter("16k"))
    mel_converter = mel_converter.to(device).eval().requires_grad_(False)
    vae = _load_meanaudio_vae(
        cast("Callable[[str], _MeanAudioVAE]", get_my_vae), checkpoint_path, device
    )
    logger.info(
        "loaded_meanaudio_checkpoint",
        checkpoint_revision=MEANAUDIO_CHECKPOINT_REVISION,
        checkpoint_sha256=MEANAUDIO_CHECKPOINT_SHA256,
        device=device,
        source_commit=MEANAUDIO_PACKAGE_COMMIT,
        synth_setter_git_sha=resolve_git_sha(),
    )

    def encode(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Encode normalized source audio as deterministic MeanAudio latents.

        :param audio: One- or two-channel ``(B, C, T)`` audio in ``[-1, 1]``.
        :param sample_rate: Positive source sample rate in Hz.
        :returns: Contiguous float32 ``(B, 20, F)`` posterior means.
        """
        prepared = meanaudio_encoder_input(audio, sample_rate)
        return _encode_meanaudio_chunks(
            mel_converter,
            vae,
            prepared,
            device=device,
        )

    return encode


def encode_meanaudio_column(
    sources: Mapping[str, np.ndarray], sample_rate: int, encoder: MeanAudioEncodeFn
) -> pa.Array:
    """Encode one audio batch as a fixed-shape MeanAudio Lance tensor.

    :param sources: Decoded source columns carrying ``(B, C, T)`` audio.
    :param sample_rate: Dataset sample rate in Hz.
    :param encoder: Loaded MeanAudio audio encoder.
    :returns: Fixed-shape float32 Arrow tensor array with shape ``(20, F)`` per row.
    :raises ValueError: Encoder output has the wrong shape or non-finite values.
    """
    audio = sources[AUDIO_FIELD]
    embeddings = np.asarray(encoder(audio, sample_rate), dtype=np.float32)
    expected_shape = (
        len(audio),
        MEANAUDIO_EMBEDDING_DIM,
        meanaudio_num_latent_frames(audio.shape[-1], sample_rate),
    )
    if embeddings.shape != expected_shape:
        raise ValueError(
            f"{MEANAUDIO_16K_FIELD} encoder produced shape {embeddings.shape}, "
            f"expected {expected_shape}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{MEANAUDIO_16K_FIELD} encoder produced non-finite values")

    from synth_setter.pipeline.data.lance_shard import tensor_array

    return tensor_array(np.ascontiguousarray(embeddings), np.dtype("float32"), expected_shape[1:])


def meanaudio_artifact_digest(checkpoint: str) -> str:
    """Verify and identify the pinned MeanAudio package/checkpoint pair.

    :param checkpoint: Pinned Hugging Face repo or SHA-identical local checkpoint.
    :returns: Package and checkpoint identity for registry policy versioning.
    """
    resolve_meanaudio_checkpoint(checkpoint)
    return f"package:{MEANAUDIO_PACKAGE_COMMIT};checkpoint:sha256:{MEANAUDIO_CHECKPOINT_SHA256}"
