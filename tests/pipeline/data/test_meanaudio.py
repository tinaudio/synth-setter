"""Behavioral and contract tests for the MeanAudio MMAudio VAE adapter."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

import synth_setter.pipeline.data.meanaudio as meanaudio_module
from synth_setter.data.vst.shapes import AUDIO_FIELD, MEANAUDIO_16K_FIELD
from synth_setter.pipeline.data.add_embeddings import EMBEDDING_REGISTRY, IndexSpec
from synth_setter.pipeline.data.meanaudio import (
    DEFAULT_MEANAUDIO_CHECKPOINT,
    MEANAUDIO_CHECKPOINT_SHA256,
    MEANAUDIO_EMBEDDING_DIM,
    MEANAUDIO_INDEX_SUB_VECTORS,
    MEANAUDIO_PACKAGE_COMMIT,
    encode_meanaudio_column,
    meanaudio_encoder_input,
    meanaudio_num_latent_frames,
    resolve_meanaudio_checkpoint,
)

_SAMPLE_RATE = 16_000
_FOUR_SECONDS = 64_000


def test_meanaudio_registry_spec_pins_sequence_and_index_policy() -> None:
    """The registry exposes the immutable MeanAudio sequence contract."""
    spec = EMBEDDING_REGISTRY["meanaudio_16k"]

    assert spec.column == MEANAUDIO_16K_FIELD
    assert spec.default_checkpoint == DEFAULT_MEANAUDIO_CHECKPOINT
    assert spec.co_resident is False
    assert spec.index == IndexSpec(
        pool="mean",
        vector_column=f"{MEANAUDIO_16K_FIELD}_vec",
        vector_dim=MEANAUDIO_EMBEDDING_DIM,
        num_sub_vectors=MEANAUDIO_INDEX_SUB_VECTORS,
    )


def test_meanaudio_four_second_frame_geometry_returns_125() -> None:
    """Four seconds at the standard render rate maps to 125 latent frames."""
    assert meanaudio_num_latent_frames(176_400, 44_100) == 125
    assert meanaudio_num_latent_frames(_FOUR_SECONDS, _SAMPLE_RATE) == 125


def test_meanaudio_frame_geometry_nonintegral_resample_crosses_first_frame_boundary() -> None:
    """Ceiling resample geometry admits the first source length that reaches 512 samples."""
    with pytest.raises(ValueError, match="at least 512 samples"):
        meanaudio_num_latent_frames(1_408, 44_100)

    assert meanaudio_num_latent_frames(1_409, 44_100) == 1


def test_meanaudio_encoder_input_mono_native_rate_preserves_known_values() -> None:
    """Native-rate mono samples reach the canonical frontend unchanged."""
    audio = np.zeros((1, 1, 1_024), dtype=np.float32)
    audio[0, 0, :4] = [-1.0, -0.25, 0.5, 1.0]

    prepared = meanaudio_encoder_input(audio, _SAMPLE_RATE)

    assert prepared.dtype == np.float32
    assert prepared.flags.c_contiguous
    np.testing.assert_array_equal(prepared[0, :4], [-1.0, -0.25, 0.5, 1.0])


def test_meanaudio_encoder_input_stereo_averages_known_values() -> None:
    """Stereo input is downmixed by its sample-wise channel mean."""
    audio = np.zeros((1, 2, 1_024), dtype=np.float32)
    audio[0, 0, :3] = [-1.0, 0.5, 1.0]
    audio[0, 1, :3] = [1.0, -0.25, 0.0]

    prepared = meanaudio_encoder_input(audio, _SAMPLE_RATE)

    np.testing.assert_array_equal(prepared[0, :3], [0.0, 0.125, 0.5])


def test_meanaudio_encoder_input_8khz_constant_resamples_known_signal() -> None:
    """Resampling doubles an 8 kHz clip while preserving its interior level."""
    audio = np.full((1, 1, 1_024), 0.25, dtype=np.float32)

    prepared = meanaudio_encoder_input(audio, 8_000)

    assert prepared.shape == (1, 2_048)
    assert np.isfinite(prepared).all()
    np.testing.assert_allclose(prepared[0, 100:105], 0.25, atol=3e-4)


def test_meanaudio_encoder_input_resampling_matches_canonical_torchaudio_output() -> None:
    """Normalized source audio preserves the canonical resampler output without clipping."""
    import torchaudio.functional as audio_fn

    audio = np.concatenate(
        (
            -np.ones(2_205, dtype=np.float32),
            np.ones(2_205, dtype=np.float32),
        )
    ).reshape(1, 1, -1)
    expected = audio_fn.resample(torch.from_numpy(audio[:, 0]), 44_100, 16_000).numpy()

    prepared = meanaudio_encoder_input(audio, 44_100)

    np.testing.assert_array_equal(prepared, expected)


@pytest.mark.parametrize(
    ("audio", "sample_rate", "message"),
    [
        (np.zeros((2, 1_024), dtype=np.float32), 16_000, r"expected a \(B, C, T\) batch"),
        (np.zeros((0, 1, 1_024), dtype=np.float32), 16_000, "non-empty batch"),
        (np.zeros((1, 3, 1_024), dtype=np.float32), 16_000, "1 or 2 channels"),
        (np.zeros((1, 1, 1_024), dtype=np.float32), 0, "positive sample_rate"),
        (np.full((1, 1, 1_024), np.nan, dtype=np.float32), 16_000, "non-finite"),
        (np.zeros((1, 1, 0), dtype=np.float32), 16_000, "positive num_samples"),
    ],
)
def test_meanaudio_encoder_input_incompatible_audio_raises(
    audio: np.ndarray, sample_rate: int, message: str
) -> None:
    """Malformed or non-finite input fails before upstream inference.

    :param audio: Candidate source batch.
    :param sample_rate: Candidate source rate.
    :param message: Expected failure detail.
    """
    with pytest.raises(ValueError, match=message):
        meanaudio_encoder_input(audio, sample_rate)


@pytest.mark.parametrize("amplitude", [-1.0001, 1.0001])
def test_meanaudio_encoder_input_outside_unit_range_raises(amplitude: float) -> None:
    """Normalized-audio violations fail before downmixing.

    :param amplitude: Invalid signed peak amplitude.
    """
    audio = np.full((1, 1, 1_024), amplitude, dtype=np.float32)

    with pytest.raises(ValueError, match=r"outside \[-1\.0, 1\.0\]"):
        meanaudio_encoder_input(audio, _SAMPLE_RATE)


class _Posterior:
    """Expose a deterministic posterior mean while rejecting sampling."""

    def __init__(self, values: torch.Tensor) -> None:
        """Store the deterministic mean.

        :param values: Posterior mean returned by :meth:`mode`.
        """
        self.values = values

    def mode(self) -> torch.Tensor:
        """Return the posterior mean.

        :returns: Stored posterior mean.
        """
        return self.values

    def sample(self) -> torch.Tensor:
        """Reject stochastic selection.

        :returns: No value; this method always raises.
        :raises AssertionError: Always, because the adapter must use :meth:`mode`.
        """
        raise AssertionError("MeanAudio adapter sampled the posterior")


class _ChunkVAE:
    """Return row-identifiable posterior means for chunk contract tests."""

    def __init__(self, *, frame_delta: int = 0, nonfinite: bool = False) -> None:
        """Configure output corruption.

        :param frame_delta: Offset from the expected latent width.
        :param nonfinite: Whether to inject a non-finite output.
        """
        self.frame_delta = frame_delta
        self.nonfinite = nonfinite
        self.batch_sizes: list[int] = []

    def encode(self, mel: torch.Tensor) -> _Posterior:
        """Build a deterministic posterior from the mel row identity.

        :param mel: Canonical MeanAudio mel tensors.
        :returns: Posterior exposing only its deterministic mean.
        """
        self.batch_sizes.append(len(mel))
        frames = mel.shape[-1] // 2 + self.frame_delta
        values = mel[:, :1, :1].expand(-1, MEANAUDIO_EMBEDDING_DIM, frames).clone()
        if self.nonfinite:
            values[0, 0, 0] = torch.nan
        return _Posterior(values)


class _ChunkMel:
    """Produce canonical-width mel rows without altering row identity."""

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Expand the first sample to a 250-frame mel tensor.

        :param waveform: Prepared mono waveform batch.
        :returns: Deterministic 80-bin mel tensor.
        """
        return waveform[:, :1, None].expand(-1, 80, 250).clone()


def test_encode_meanaudio_chunk_selects_deterministic_posterior_mean() -> None:
    """The adapter stores ``posterior.mode()`` rather than a random sample."""
    chunk = np.stack(
        [
            np.full(_FOUR_SECONDS, 0.25, dtype=np.float32),
            np.full(_FOUR_SECONDS, 0.75, dtype=np.float32),
        ]
    )

    encoded = meanaudio_module._encode_meanaudio_chunk(
        cast("meanaudio_module._MelConverter", _ChunkMel()),
        cast("meanaudio_module._MeanAudioVAE", _ChunkVAE()),
        chunk,
        device="cpu",
    )

    assert encoded.shape == (2, MEANAUDIO_EMBEDDING_DIM, 125)
    assert encoded.dtype == np.float32
    assert encoded.flags.c_contiguous
    np.testing.assert_array_equal(encoded[:, 0, 0], [0.25, 0.75])


def test_encode_meanaudio_chunks_bounds_large_model_batches() -> None:
    """Large-model inference never exceeds the declared four-row chunk bound."""
    vae = _ChunkVAE()
    encoded = meanaudio_module._encode_meanaudio_chunks(
        cast("meanaudio_module._MelConverter", _ChunkMel()),
        cast("meanaudio_module._MeanAudioVAE", vae),
        np.zeros((9, _FOUR_SECONDS), dtype=np.float32),
        device="cpu",
    )

    assert vae.batch_sizes == [4, 4, 1]
    assert encoded.shape == (9, MEANAUDIO_EMBEDDING_DIM, 125)


@pytest.mark.parametrize(
    ("vae", "message"),
    [
        (_ChunkVAE(frame_delta=-1), "produced shape"),
        (_ChunkVAE(nonfinite=True), "non-finite"),
    ],
)
def test_encode_meanaudio_chunk_invalid_output_raises(vae: _ChunkVAE, message: str) -> None:
    """Wrong or non-finite posterior means cannot reach Lance.

    :param vae: Candidate VAE output behavior.
    :param message: Expected failure detail.
    """
    with pytest.raises(ValueError, match=message):
        meanaudio_module._encode_meanaudio_chunk(
            cast("meanaudio_module._MelConverter", _ChunkMel()),
            cast("meanaudio_module._MeanAudioVAE", vae),
            np.zeros((2, _FOUR_SECONDS), dtype=np.float32),
            device="cpu",
        )


def test_encode_meanaudio_column_valid_output_returns_fixed_shape_tensor() -> None:
    """Registry persistence retains MeanAudio's channel-major latent orientation."""
    audio = np.zeros((2, 1, _FOUR_SECONDS), dtype=np.float32)

    def encode(source: np.ndarray, sample_rate: int) -> np.ndarray:
        assert source is audio
        assert sample_rate == _SAMPLE_RATE
        return np.ones((2, MEANAUDIO_EMBEDDING_DIM, 125), dtype=np.float32)

    encoded = encode_meanaudio_column({AUDIO_FIELD: audio}, _SAMPLE_RATE, encode)

    assert encoded.to_numpy_ndarray().shape == (2, MEANAUDIO_EMBEDDING_DIM, 125)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (np.ones((2, 125, MEANAUDIO_EMBEDDING_DIM), dtype=np.float32), "produced shape"),
        (
            np.full((2, MEANAUDIO_EMBEDDING_DIM, 125), np.inf, dtype=np.float32),
            "non-finite",
        ),
    ],
)
def test_encode_meanaudio_column_invalid_output_raises(output: np.ndarray, message: str) -> None:
    """Wrong orientation and non-finite values fail at the Arrow boundary.

    :param output: Candidate adapter output.
    :param message: Expected failure detail.
    """
    audio = np.zeros((2, 1, _FOUR_SECONDS), dtype=np.float32)

    def encode(source: np.ndarray, sample_rate: int) -> np.ndarray:
        del source, sample_rate
        return output

    with pytest.raises(ValueError, match=message):
        encode_meanaudio_column({AUDIO_FIELD: audio}, _SAMPLE_RATE, encode)


def test_resolve_meanaudio_checkpoint_hash_identical_local_file_returns_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A local copy with the pinned digest is an accepted immutable identity.

    :param monkeypatch: Fixture setting the test artifact digest.
    :param tmp_path: Temporary checkpoint location.
    """
    checkpoint = tmp_path / "v1-16.pth"
    checkpoint.write_bytes(b"trusted MeanAudio checkpoint")
    monkeypatch.setattr(
        meanaudio_module,
        "MEANAUDIO_CHECKPOINT_SHA256",
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )

    assert resolve_meanaudio_checkpoint(str(checkpoint)) == checkpoint


def test_resolve_meanaudio_checkpoint_wrong_local_hash_raises(tmp_path: Path) -> None:
    """A local override cannot weaken the pinned checkpoint identity.

    :param tmp_path: Temporary checkpoint location.
    """
    checkpoint = tmp_path / "v1-16.pth"
    checkpoint.write_bytes(b"not the pinned MeanAudio checkpoint")

    with pytest.raises(ValueError, match=MEANAUDIO_CHECKPOINT_SHA256):
        resolve_meanaudio_checkpoint(str(checkpoint))


def test_resolve_meanaudio_checkpoint_unpinned_remote_identity_raises() -> None:
    """Only the revision-pinned default Hugging Face repository is accepted."""
    with pytest.raises(ValueError, match="requires the pinned Hugging Face repo"):
        resolve_meanaudio_checkpoint("other-owner/MeanAudio")


def test_download_meanaudio_checkpoint_transient_failures_retry_to_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient Hugging Face failures retry before surfacing to the pipeline.

    :param monkeypatch: Fixture removing retry waits from this unit test.
    """
    from tenacity import wait_none

    attempts = 0

    def download(**kwargs: str) -> str:
        nonlocal attempts
        attempts += 1
        assert kwargs["repo_id"] == meanaudio_module.MEANAUDIO_CHECKPOINT_REPO
        if attempts < 3:
            raise TimeoutError("transient Hugging Face timeout")
        return "/tmp/v1-16.pth"

    monkeypatch.setattr(meanaudio_module._download_meanaudio_checkpoint.retry, "wait", wait_none())

    assert meanaudio_module._download_meanaudio_checkpoint(download) == "/tmp/v1-16.pth"
    assert attempts == 3


def test_download_meanaudio_checkpoint_permanent_failure_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Permanent downloader failures surface without repeated requests.

    :param monkeypatch: Fixture removing retry waits from this unit test.
    """
    from tenacity import wait_none

    attempts = 0

    def download(**kwargs: str) -> str:
        nonlocal attempts
        del kwargs
        attempts += 1
        raise ValueError("invalid repository identity")

    monkeypatch.setattr(meanaudio_module._download_meanaudio_checkpoint.retry, "wait", wait_none())

    with pytest.raises(ValueError, match="invalid repository identity"):
        meanaudio_module._download_meanaudio_checkpoint(download)
    assert attempts == 1


class _StateLoadingVAE(torch.nn.Module):
    """Record strict loading, decoder deletion, and frozen-eval transitions."""

    def __init__(self) -> None:
        """Initialize a model carrying both encoder and decoder state."""
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1)
        self.decoder = torch.nn.Linear(1, 1)
        self.assign: bool | None = None
        self.events: list[str] = []
        self.strict: bool | None = None

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> SimpleNamespace:
        self.events.append("load")
        self.strict = strict
        self.assign = assign
        assert hasattr(self, "decoder")
        assert state_dict["weight"].item() == pytest.approx(1.5)
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def remove_weight_norm(self) -> _StateLoadingVAE:
        self.events.append("remove_weight_norm")
        assert not hasattr(self, "decoder")
        return self

    def eval(self) -> _StateLoadingVAE:
        self.events.append("eval")
        return cast("_StateLoadingVAE", super().eval())


def test_load_meanaudio_vae_loads_strictly_before_pruning_and_freezes(
    tmp_path: Path,
) -> None:
    """Checkpoint state is strict-loaded before the unused decoder is removed.

    :param tmp_path: Temporary checkpoint location.
    """
    checkpoint = tmp_path / "v1-16.pth"
    torch.save({"weight": torch.tensor(1.5)}, checkpoint)
    vae = _StateLoadingVAE()

    factory = cast(
        "Callable[[str], meanaudio_module._MeanAudioVAE]",
        lambda _mode: vae,
    )
    loaded = meanaudio_module._load_meanaudio_vae(factory, checkpoint, "cpu")

    assert loaded is vae
    assert vae.strict is True
    assert vae.assign is True
    assert vae.events == ["load", "remove_weight_norm", "eval"]
    assert not hasattr(vae, "decoder")
    assert vae.training is False
    assert all(not parameter.requires_grad for parameter in vae.parameters())


def test_meanaudio_artifact_digest_pins_package_and_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Artifact identity binds both immutable upstream inputs.

    :param monkeypatch: Fixture accepting a local test artifact.
    :param tmp_path: Temporary checkpoint location.
    """
    checkpoint = tmp_path / "v1-16.pth"
    checkpoint.write_bytes(b"trusted MeanAudio checkpoint")
    monkeypatch.setattr(
        meanaudio_module,
        "MEANAUDIO_CHECKPOINT_SHA256",
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )

    digest = meanaudio_module.meanaudio_artifact_digest(str(checkpoint))

    assert f"package:{MEANAUDIO_PACKAGE_COMMIT}" in digest
    assert f"checkpoint:sha256:{meanaudio_module.MEANAUDIO_CHECKPOINT_SHA256}" in digest
