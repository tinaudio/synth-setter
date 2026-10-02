"""Official MeanAudio-S-Full text-to-unnormalized-latent generation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final

import numpy as np
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from synth_setter.pipeline.data.meanaudio import (
    MEANAUDIO_CHECKPOINT_NAME,
    MEANAUDIO_CHECKPOINT_REPO,
    MEANAUDIO_CHECKPOINT_REVISION,
    MEANAUDIO_CHECKPOINT_SHA256,
    MEANAUDIO_PACKAGE_COMMIT,
    MEANAUDIO_SAMPLE_RATE,
    load_meanaudio_audio_encoder,
    resolve_meanaudio_checkpoint,
)

MEANAUDIO_VARIANT: Final = "s-full"
MEANAUDIO_DURATION_SECONDS: Final = 4.0
MEANAUDIO_STEPS: Final = 25
MEANAUDIO_LATENT_SHAPE: Final = (20, 125)
MEANAUDIO_S_FULL_CHECKPOINT_NAME: Final = "meanaudio_s_full.pth"
MEANAUDIO_S_FULL_CHECKPOINT_SHA256: Final = (
    "d1051af33e15f7d98481c2a10b61cd54ab8bca45df59b7079c3bc27dcd1e8ac5"
)
MEANAUDIO_CLAP_PACKAGE_REVISION: Final = "590b52df03a5e480634b8714135d3534313be860"
MEANAUDIO_CLAP_CHECKPOINT_NAME: Final = "music_speech_audioset_epoch_15_esc_89.98.pt"
MEANAUDIO_CLAP_CHECKPOINT_SHA256: Final = (
    "51c68f12f9d7ea25fdaaccf741ec7f81e93ee594455410f3bca4f47f88d8e006"
)
MEANAUDIO_VOCODER_CHECKPOINT_NAME: Final = "best_netG.pt"
MEANAUDIO_VOCODER_CHECKPOINT_SHA256: Final = (
    "970ca75ee4d5ce583e9396a4534acb14971ea2b4f1c22e038f476680c868a789"
)
MEANAUDIO_T5_REPO: Final = "google/flan-t5-large"
MEANAUDIO_T5_REVISION: Final = "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
MEANAUDIO_ROBERTA_REPO: Final = "FacebookAI/roberta-base"
MEANAUDIO_ROBERTA_REVISION: Final = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"

type MeanAudioGenerateFn = Callable[[str, int], np.ndarray]


def meanaudio_s_full_provenance() -> dict[str, str]:
    """Return immutable identities for every generated-latent model input.

    :returns: Source-package and text/model checkpoint identities.
    """
    return {
        "meanaudio_upstream_revision": MEANAUDIO_PACKAGE_COMMIT,
        "meanaudio_checkpoint_revision": MEANAUDIO_CHECKPOINT_REVISION,
        "meanaudio_model_variant": MEANAUDIO_VARIANT,
        "meanaudio_model_checkpoint_name": MEANAUDIO_S_FULL_CHECKPOINT_NAME,
        "meanaudio_model_checkpoint_sha256": MEANAUDIO_S_FULL_CHECKPOINT_SHA256,
        "meanaudio_clap_package_revision": MEANAUDIO_CLAP_PACKAGE_REVISION,
        "meanaudio_clap_checkpoint_name": MEANAUDIO_CLAP_CHECKPOINT_NAME,
        "meanaudio_clap_checkpoint_sha256": MEANAUDIO_CLAP_CHECKPOINT_SHA256,
        "meanaudio_t5_repo": MEANAUDIO_T5_REPO,
        "meanaudio_t5_revision": MEANAUDIO_T5_REVISION,
        "meanaudio_roberta_repo": MEANAUDIO_ROBERTA_REPO,
        "meanaudio_roberta_revision": MEANAUDIO_ROBERTA_REVISION,
    }


def meanaudio_s_full_reencoded_provenance() -> dict[str, str]:
    """Return identities for waveform projection into posterior-mean conditioning.

    :returns: Base generation provenance plus VAE/vocoder projection identity.
    """
    return meanaudio_s_full_provenance() | {
        "meanaudio_projection": "vae-decode-vocode-encode-mode",
        "meanaudio_vae_checkpoint_name": MEANAUDIO_CHECKPOINT_NAME,
        "meanaudio_vae_checkpoint_sha256": MEANAUDIO_CHECKPOINT_SHA256,
        "meanaudio_vocoder_checkpoint_name": MEANAUDIO_VOCODER_CHECKPOINT_NAME,
        "meanaudio_vocoder_checkpoint_sha256": MEANAUDIO_VOCODER_CHECKPOINT_SHA256,
    }


def _file_sha256(path: Path) -> str:
    """Hash one upstream generation asset without loading it into memory.

    :param path: File whose immutable identity is required.
    :returns: Lowercase SHA-256 digest.
    """
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_retryable_download_error(error: BaseException) -> bool:
    """Return whether an upstream asset download can succeed on retry.

    :param error: Hugging Face download failure.
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
def _download_meanaudio_asset(filename: str) -> str:
    """Download one file from the immutable official MeanAudio snapshot.

    :param filename: Repository-relative asset name.
    :returns: Local Hugging Face cache path.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=MEANAUDIO_CHECKPOINT_REPO,
        revision=MEANAUDIO_CHECKPOINT_REVISION,
        filename=filename,
    )


def _resolve_meanaudio_asset(filename: str, expected_sha256: str) -> Path:
    """Resolve and verify one official generation checkpoint.

    :param filename: Repository-relative asset name.
    :param expected_sha256: Required lowercase SHA-256 digest.
    :returns: Verified local cache path.
    :raises ValueError: Downloaded bytes differ from the pinned identity.
    """
    path = Path(_download_meanaudio_asset(filename)).resolve()
    actual = _file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"MeanAudio asset {filename} SHA-256 is {actual}, expected {expected_sha256}"
        )
    return path


def validate_meanaudio_s_full_latent(latent: np.ndarray) -> np.ndarray:
    """Validate an unnormalized S-Full latent for direct inverse conditioning.

    :param latent: Channel-major latent expected to have shape ``(1, 20, 125)``.
    :returns: Finite contiguous float32 latent with the same shape.
    :raises ValueError: Shape or values violate the four-second inverse contract.
    """
    expected_shape = (1, *MEANAUDIO_LATENT_SHAPE)
    if latent.shape != expected_shape:
        raise ValueError(f"MeanAudio-S-Full latent shape {latent.shape}, expected {expected_shape}")
    values = np.ascontiguousarray(latent, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("MeanAudio-S-Full latent contains non-finite values")
    return values


def load_meanaudio_s_full_generator(
    *,
    steps: int = MEANAUDIO_STEPS,
    duration_seconds: float = MEANAUDIO_DURATION_SECONDS,
    device: str = "cuda",
) -> MeanAudioGenerateFn:
    """Load official S-Full weights and return prompt-to-unnormalized-latent inference.

    The returned latent is MeanAudio's state immediately after ``net.unnormalize``;
    neither the VAE decoder nor vocoder is loaded.

    :param steps: Positive MeanFlow integration step count.
    :param duration_seconds: Fixed four-second generation duration.
    :param device: Torch device hosting text and latent-flow inference.
    :returns: Callable accepting prompt and seed and returning ``(1, 20, 125)`` float32.
    :raises ValueError: Steps, duration, device, prompt, or generated output is invalid.
    """
    import torch
    from meanaudio.model.mean_flow import MeanFlow
    from meanaudio.model.networks import get_mean_audio
    from transformers import AutoTokenizer, RobertaTokenizer, T5EncoderModel

    if steps < 1:
        raise ValueError(f"MeanAudio steps must be positive, got {steps}")
    if duration_seconds != MEANAUDIO_DURATION_SECONDS:
        raise ValueError(
            f"MeanAudio-S-Full comparison requires {MEANAUDIO_DURATION_SECONDS}s, "
            f"got {duration_seconds}s"
        )
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("MeanAudio CUDA device requested but CUDA is unavailable")
    dtype = torch.bfloat16 if selected_device.type == "cuda" else torch.float32

    model_path = _resolve_meanaudio_asset(
        MEANAUDIO_S_FULL_CHECKPOINT_NAME,
        MEANAUDIO_S_FULL_CHECKPOINT_SHA256,
    )
    clap_path = _resolve_meanaudio_asset(
        MEANAUDIO_CLAP_CHECKPOINT_NAME,
        MEANAUDIO_CLAP_CHECKPOINT_SHA256,
    )

    net = get_mean_audio("meanaudio_s", use_rope=True, text_c_dim=512)
    state = torch.load(model_path, map_location="cpu", weights_only=True, mmap=True)
    net.load_weights(state)
    del state
    net.update_seq_lengths(MEANAUDIO_LATENT_SHAPE[1])
    net = net.to(selected_device, dtype=dtype).eval().requires_grad_(False)

    t5_tokenizer = AutoTokenizer.from_pretrained(
        MEANAUDIO_T5_REPO,
        revision=MEANAUDIO_T5_REVISION,
    )
    t5 = T5EncoderModel.from_pretrained(
        MEANAUDIO_T5_REPO,
        revision=MEANAUDIO_T5_REVISION,
        dtype=dtype,
    )
    torch.nn.Module.to(t5, device=selected_device, dtype=dtype)
    t5.eval().requires_grad_(False)

    import laion_clap

    clap = laion_clap.CLAP_Module(
        enable_fusion=False,
        device=str(selected_device),
        amodel="HTSAT-base",
    )
    clap.tokenize = RobertaTokenizer.from_pretrained(
        MEANAUDIO_ROBERTA_REPO,
        revision=MEANAUDIO_ROBERTA_REVISION,
    )
    clap.load_ckpt(str(clap_path), verbose=False)
    clap = clap.to(selected_device, dtype=dtype).eval().requires_grad_(False)
    mean_flow = MeanFlow(steps=steps)

    def generate(prompt: str, seed: int) -> np.ndarray:
        """Generate one official unnormalized S-Full latent.

        :param prompt: Non-blank natural-language audio description.
        :param seed: Noise seed for MeanFlow generation.
        :returns: Contiguous float32 ``(1, 20, 125)`` latent.
        :raises ValueError: The prompt is blank or generation violates the latent contract.
        """
        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise ValueError("MeanAudio prompt must contain text")
        tokens = t5_tokenizer(
            [normalized_prompt],
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        t5_inputs = {name: value.to(selected_device) for name, value in tokens.items()}
        rng = torch.Generator(device=selected_device).manual_seed(seed)
        with torch.inference_mode():
            text_features = t5(**t5_inputs).last_hidden_state
            text_features_c = clap.get_text_embedding([normalized_prompt], use_tensor=True)
            conditions = net.preprocess_conditions(text_features, text_features_c)
            empty_conditions = net.get_empty_conditions(1)
            noise = torch.randn(
                1,
                MEANAUDIO_LATENT_SHAPE[1],
                MEANAUDIO_LATENT_SHAPE[0],
                device=selected_device,
                dtype=dtype,
                generator=rng,
            )

            def velocity(*, t: torch.Tensor, r: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                return net.ode_wrapper(t, r, x, conditions, empty_conditions, 0.0)

            unnormalized = net.unnormalize(mean_flow.to_data(velocity, noise))
            channel_major = unnormalized.transpose(1, 2).float().cpu().numpy()
        return validate_meanaudio_s_full_latent(channel_major)

    return generate


def load_meanaudio_s_full_reencoded_generator(
    *,
    steps: int = MEANAUDIO_STEPS,
    duration_seconds: float = MEANAUDIO_DURATION_SECONDS,
    device: str = "cuda",
) -> MeanAudioGenerateFn:
    """Project generated S-Full audio into the posterior means used for inverse training.

    :param steps: Positive MeanFlow integration step count.
    :param duration_seconds: Fixed four-second generation duration.
    :param device: Torch device hosting generation, decoding, and encoding.
    :returns: Callable accepting prompt and seed and returning ``(1, 20, 125)`` float32.
    """
    import torch
    from meanaudio.ext.autoencoder.autoencoder import AutoEncoderModule

    generate_direct = load_meanaudio_s_full_generator(
        steps=steps,
        duration_seconds=duration_seconds,
        device=device,
    )
    selected_device = torch.device(device)
    dtype = torch.bfloat16 if selected_device.type == "cuda" else torch.float32
    vae_path = resolve_meanaudio_checkpoint()
    vocoder_path = _resolve_meanaudio_asset(
        MEANAUDIO_VOCODER_CHECKPOINT_NAME,
        MEANAUDIO_VOCODER_CHECKPOINT_SHA256,
    )
    decoder = AutoEncoderModule(
        vae_ckpt_path=str(vae_path),
        vocoder_ckpt_path=str(vocoder_path),
        mode="16k",
        need_vae_encoder=False,
    )
    decoder = decoder.to(selected_device, dtype=dtype).eval().requires_grad_(False)
    encode_mode = load_meanaudio_audio_encoder(str(vae_path), device=device)

    def generate(prompt: str, seed: int) -> np.ndarray:
        """Generate waveform and re-encode its deterministic posterior mean.

        :param prompt: Non-blank natural-language audio description.
        :param seed: Noise seed for MeanFlow generation.
        :returns: Contiguous float32 ``(1, 20, 125)`` posterior mean.
        :raises ValueError: The decoded waveform violates the four-second mono contract.
        """
        direct = generate_direct(prompt, seed)
        channel_major = torch.from_numpy(direct).to(selected_device, dtype=dtype)
        with torch.inference_mode():
            mel = decoder.decode(channel_major)
            waveform = decoder.vocode(mel).float().cpu().numpy()
        if waveform.ndim == 2:
            waveform = waveform[:, None, :]
        expected_samples = int(MEANAUDIO_DURATION_SECONDS * MEANAUDIO_SAMPLE_RATE)
        expected_shape = (1, 1, expected_samples)
        if waveform.shape != expected_shape:
            raise ValueError(
                f"MeanAudio-S-Full decoded waveform shape {waveform.shape}, "
                f"expected {expected_shape}"
            )
        return validate_meanaudio_s_full_latent(
            encode_mode(waveform, MEANAUDIO_SAMPLE_RATE)
        )

    return generate
