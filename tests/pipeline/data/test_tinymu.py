"""Behavioral tests for the pinned TinyMU MATPAC adapter."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import synth_setter.pipeline.data.tinymu as tinymu_module
from hypothesis import given, settings
from hypothesis import strategies as st

from synth_setter.data.vst.shapes import TINYMU_FIELD
from synth_setter.pipeline.data.add_embeddings import EMBEDDING_REGISTRY, IndexSpec
from synth_setter.pipeline.data.tinymu import (
    DEFAULT_TINYMU_CHECKPOINT,
    TINYMU_CHECKPOINT_SHA256,
    TINYMU_FRONTEND,
    TINYMU_SOURCE_COMMIT,
    TINYMU_SOURCE_DIR_ENV,
    TINYMU_SOURCE_MODEL_PATH,
    resolve_tinymu_checkpoint,
    resolve_tinymu_source_model,
    tinymu_encoder_input,
    tinymu_num_latent_frames,
)
from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig

_SAMPLE_RATE = 16_000
_AUDIO_SAMPLES = 16_000


def test_tinymu_registry_spec_pins_checkpoint_and_mean_pooling() -> None:
    """The registry exposes one immutable MATPAC sequence policy."""
    spec = EMBEDDING_REGISTRY["tinymu"]

    assert spec.column == TINYMU_FIELD
    assert spec.default_checkpoint == DEFAULT_TINYMU_CHECKPOINT
    assert spec.co_resident is False
    assert spec.index == IndexSpec(pool="mean", vector_column=f"{TINYMU_FIELD}_vec")


@pytest.mark.parametrize(
    ("num_samples", "sample_rate", "expected_frames"),
    [
        (2_800, 16_000, 1),
        (16_000, 16_000, 7),
        (64_000, 16_000, 25),
        (480_000, 48_000, 63),
    ],
)
def test_tinymu_num_latent_frames_measured_clips_matches_contract(
    num_samples: int, sample_rate: int, expected_frames: int
) -> None:
    """Measured clip lengths map to the upstream MATPAC token counts.

    :param num_samples: Source clip length.
    :param sample_rate: Source sample rate.
    :param expected_frames: Measured MATPAC token count.
    """
    assert tinymu_num_latent_frames(num_samples, sample_rate) == expected_frames


@given(
    channels=st.integers(min_value=1, max_value=2),
    sample_rate=st.sampled_from([8_000, 16_000, 22_050, 44_100, 48_000]),
)
@settings(max_examples=10, deadline=None)
def test_tinymu_encoder_input_valid_audio_returns_finite_mono_float32(
    channels: int, sample_rate: int
) -> None:
    """Supported channel/sample-rate combinations normalize to MATPAC input.

    :param channels: Supported source channel count.
    :param sample_rate: Supported source sample rate.
    """
    samples = sample_rate
    time = np.arange(samples, dtype=np.float32) / sample_rate
    channel_rows = [np.sin(2 * np.pi * frequency * time) for frequency in (220.0, 330.0)]
    audio = np.stack(channel_rows[:channels])[None]

    prepared = tinymu_encoder_input(audio, sample_rate)

    assert prepared.shape == (1, TINYMU_FRONTEND.sample_rate)
    assert prepared.dtype == np.float32
    assert np.isfinite(prepared).all()


def test_tinymu_encoder_input_mono_16khz_preserves_known_values() -> None:
    """Native-rate mono samples reach MATPAC without waveform corruption."""
    samples = np.zeros((1, 1, 2_800), dtype=np.float32)
    samples[0, 0, :4] = [-1.0, -0.25, 0.5, 1.0]

    prepared = tinymu_encoder_input(samples, 16_000)

    np.testing.assert_array_equal(prepared[0, :4], [-1.0, -0.25, 0.5, 1.0])


def test_tinymu_encoder_input_stereo_averages_known_values() -> None:
    """Stereo downmix is the sample-wise mean of both channels."""
    samples = np.zeros((1, 2, 2_800), dtype=np.float32)
    samples[0, 0, :3] = [-1.0, 0.5, 1.0]
    samples[0, 1, :3] = [1.0, -0.25, 0.0]

    prepared = tinymu_encoder_input(samples, 16_000)

    np.testing.assert_array_equal(prepared[0, :3], [0.0, 0.125, 0.5])


def test_tinymu_encoder_input_8khz_constant_resamples_known_signal() -> None:
    """Resampling doubles an 8 kHz constant clip while preserving its interior level."""
    samples = np.full((1, 1, 2_800), 0.25, dtype=np.float32)

    prepared = tinymu_encoder_input(samples, 8_000)

    assert prepared.shape == (1, 5_600)
    np.testing.assert_allclose(prepared[0, 100:105], 0.25, atol=3e-4)


@pytest.mark.parametrize(
    ("audio", "sample_rate", "message"),
    [
        (np.zeros((2, 100), dtype=np.float32), 16_000, r"expected a \(B, C, T\) batch"),
        (np.zeros((1, 3, 3_000), dtype=np.float32), 16_000, "1 or 2 channels"),
        (np.zeros((0, 1, 3_000), dtype=np.float32), 16_000, "non-empty batch"),
        (np.zeros((1, 1, 3_000), dtype=np.float32), 0, "positive sample_rate"),
        (np.full((1, 1, 3_000), np.nan, dtype=np.float32), 16_000, "non-finite"),
        (np.zeros((1, 1, 2_799), dtype=np.float32), 16_000, "at least 2800 samples"),
    ],
)
def test_tinymu_encoder_input_incompatible_audio_raises(
    audio: np.ndarray, sample_rate: int, message: str
) -> None:
    """Malformed, non-finite, and too-short audio fails before inference.

    :param audio: Candidate source batch.
    :param sample_rate: Candidate source sample rate.
    :param message: Expected failure detail.
    """
    with pytest.raises(ValueError, match=message):
        tinymu_encoder_input(audio, sample_rate)


@pytest.mark.parametrize("amplitude", [-1.0001, 1.0001])
def test_tinymu_encoder_input_outside_unit_amplitude_raises(amplitude: float) -> None:
    """Finite samples outside normalized audio bounds fail before inference.

    :param amplitude: Invalid signed peak amplitude.
    """
    audio = np.full((1, 1, 2_800), amplitude, dtype=np.float32)

    with pytest.raises(ValueError, match=r"outside \[-1\.0, 1\.0\]"):
        tinymu_encoder_input(audio, 16_000)


def test_tinymu_registry_encoder_valid_sequence_returns_fixed_shape_tensor() -> None:
    """The registry persists TinyMU sequences in conditioning orientation."""
    audio = np.zeros((2, 1, _AUDIO_SAMPLES), dtype=np.float32)

    def encode(source: np.ndarray, sample_rate: int) -> np.ndarray:
        assert source.shape == audio.shape
        assert sample_rate == _SAMPLE_RATE
        return np.ones((2, TINYMU_FRONTEND.embedding_dim, 7), dtype=np.float32)

    encoded = EMBEDDING_REGISTRY["tinymu"].encode_column(audio, _SAMPLE_RATE, encode)

    assert encoded.to_numpy_ndarray().shape == (2, TINYMU_FRONTEND.embedding_dim, 7)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (np.ones((2, 7, TINYMU_FRONTEND.embedding_dim), dtype=np.float32), "produced shape"),
        (
            np.full((2, TINYMU_FRONTEND.embedding_dim, 7), np.inf, dtype=np.float32),
            "non-finite",
        ),
    ],
)
def test_tinymu_registry_encoder_invalid_output_raises(
    output: np.ndarray, message: str
) -> None:
    """Wrong orientation and non-finite MATPAC outputs cannot reach Lance.

    :param output: Candidate encoder output.
    :param message: Expected failure detail.
    """
    audio = np.zeros((2, 1, _AUDIO_SAMPLES), dtype=np.float32)

    def encode(source: np.ndarray, sample_rate: int) -> np.ndarray:
        del source, sample_rate
        return output

    with pytest.raises(ValueError, match=message):
        EMBEDDING_REGISTRY["tinymu"].encode_column(audio, _SAMPLE_RATE, encode)


def test_resolve_tinymu_checkpoint_hash_identical_local_file_returns_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A local artifact with the configured strong identity resolves without R2.

    :param monkeypatch: Fixture setting the test artifact identity.
    :param tmp_path: Temporary checkpoint location.
    """
    checkpoint = tmp_path / "matpac.pt"
    checkpoint.write_bytes(b"trusted test checkpoint")
    monkeypatch.setattr(
        tinymu_module,
        "TINYMU_CHECKPOINT_SHA256",
        tinymu_module._file_sha256(checkpoint),
    )

    assert resolve_tinymu_checkpoint(str(checkpoint)) == checkpoint


def test_resolve_tinymu_checkpoint_empty_cache_hydrates_atomically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The pinned R2 object becomes a verified canonical cache file.

    :param monkeypatch: Fixture isolating cache and R2 transfer boundaries.
    :param tmp_path: Temporary cache location.
    """
    artifact = b"trusted test checkpoint"
    expected_digest = hashlib.sha256(artifact).hexdigest()
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(tinymu_module, "TINYMU_CHECKPOINT_SHA256", expected_digest)
    monkeypatch.setattr(tinymu_module, "embedding_model_dir", lambda _name: cache_dir)
    monkeypatch.setattr(tinymu_module.r2_io, "ensure_r2_env_loaded", lambda: None)

    def download(_uri: str, destination: Path) -> None:
        destination.write_bytes(artifact)

    monkeypatch.setattr(tinymu_module.r2_io, "download_to_path", download)

    resolved = resolve_tinymu_checkpoint(DEFAULT_TINYMU_CHECKPOINT)

    assert resolved == cache_dir / tinymu_module.TINYMU_CHECKPOINT_NAME
    assert resolved.read_bytes() == artifact
    assert list(cache_dir.glob(".*")) == []


def test_resolve_tinymu_checkpoint_wrong_hash_raises(tmp_path: Path) -> None:
    """A local checkpoint override cannot weaken the pinned artifact identity.

    :param tmp_path: Temporary checkpoint location.
    """
    checkpoint = tmp_path / "matpac.pt"
    checkpoint.write_bytes(b"not the trusted MATPAC checkpoint")

    with pytest.raises(ValueError, match=TINYMU_CHECKPOINT_SHA256):
        resolve_tinymu_checkpoint(str(checkpoint))


def test_resolve_tinymu_checkpoint_unpinned_r2_uri_raises() -> None:
    """A mutable or unrelated R2 object is rejected before hydration."""
    with pytest.raises(ValueError, match="pinned TinyMU checkpoint"):
        resolve_tinymu_checkpoint("r2://intermediate-data/tinymu/main/model.pt")


def _write_test_source_checkout(tmp_path: Path) -> tuple[Path, str]:
    """Create a real Git checkout carrying the expected source-relative model path.

    :param tmp_path: Temporary Git checkout parent.
    :returns: Checkout root and its commit identity.
    """
    source = tmp_path / "TinyMU"
    model = source / TINYMU_SOURCE_MODEL_PATH
    model.parent.mkdir(parents=True)
    model.write_text("class matpac_wrapper: pass\n")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=TinyMU test",
            "-c",
            "user.email=tinymu@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return source, commit


def test_resolve_tinymu_source_model_matching_checkout_returns_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exact commit and source blob identity accepts the external checkout.

    :param monkeypatch: Fixture setting the test source identity.
    :param tmp_path: Temporary Git checkout location.
    """
    source, commit = _write_test_source_checkout(tmp_path)
    monkeypatch.setattr(tinymu_module, "TINYMU_SOURCE_COMMIT", commit)

    assert resolve_tinymu_source_model(source) == source / TINYMU_SOURCE_MODEL_PATH


def test_resolve_tinymu_source_model_modified_blob_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A dirty MATPAC model file fails even when checkout HEAD is pinned.

    :param monkeypatch: Fixture setting the test source identity.
    :param tmp_path: Temporary Git checkout location.
    """
    source, commit = _write_test_source_checkout(tmp_path)
    monkeypatch.setattr(tinymu_module, "TINYMU_SOURCE_COMMIT", commit)
    (source / TINYMU_SOURCE_MODEL_PATH).write_text("modified = True\n")

    with pytest.raises(ValueError, match="differs from pinned commit"):
        resolve_tinymu_source_model(source)


def test_configured_tinymu_source_model_environment_checkout_resolves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The documented environment fallback validates and returns the source module.

    :param monkeypatch: Fixture setting the source identity and environment.
    :param tmp_path: Temporary Git checkout location.
    """
    source, commit = _write_test_source_checkout(tmp_path)
    monkeypatch.setattr(tinymu_module, "TINYMU_SOURCE_COMMIT", commit)
    monkeypatch.setenv(TINYMU_SOURCE_DIR_ENV, str(source))

    assert tinymu_module.configured_tinymu_source_model(None) == (
        source / TINYMU_SOURCE_MODEL_PATH
    )


def test_configured_tinymu_source_model_without_boundary_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing config and environment source paths fail with setup guidance.

    :param monkeypatch: Fixture clearing the source environment variable.
    """
    monkeypatch.delenv(TINYMU_SOURCE_DIR_ENV, raising=False)

    with pytest.raises(FileNotFoundError, match=TINYMU_SOURCE_DIR_ENV):
        tinymu_module.configured_tinymu_source_model(None)


def test_load_tinymu_source_module_valid_python_exports_constructor(tmp_path: Path) -> None:
    """A verified Python module executes under the isolated adapter namespace.

    :param tmp_path: Temporary source module location.
    """
    model_path = tmp_path / "model.py"
    model_path.write_text("matpac_wrapper = object()\n")

    module = tinymu_module._load_source_module(model_path)

    assert module.matpac_wrapper is not None


def test_load_tinymu_source_module_missing_dependency_names_extra(tmp_path: Path) -> None:
    """An unavailable upstream dependency reports the optional installation command.

    :param tmp_path: Temporary source module location.
    """
    model_path = tmp_path / "model.py"
    model_path.write_text("import dependency_that_does_not_exist_for_tinymu\n")

    with pytest.raises(ImportError, match="uv sync --extra tinymu"):
        tinymu_module._load_source_module(model_path)


def _model_contract(*, depth: int = 12) -> tinymu_module._MatpacModel:
    """Build the narrow external config surface used by contract tests.

    :param depth: Candidate upstream transformer depth.
    :returns: Structurally typed MATPAC model placeholder.
    """
    config = SimpleNamespace(
        encoder=SimpleNamespace(depth=depth, embed_dim=768),
        n_freq=80,
        n_t=992,
        patch_size=16,
        sr=16_000,
    )
    return cast("tinymu_module._MatpacModel", SimpleNamespace(cfg=config))


def test_validate_tinymu_model_contract_matching_architecture_returns() -> None:
    """The measured upstream architecture satisfies the narrow adapter protocol."""
    tinymu_module._validate_model_contract(_model_contract())


def test_validate_tinymu_model_contract_changed_depth_raises() -> None:
    """A shape-defining upstream architecture change fails before state loading."""
    with pytest.raises(ValueError, match="architecture"):
        tinymu_module._validate_model_contract(_model_contract(depth=13))


def test_resolve_tinymu_source_model_wrong_commit_raises(tmp_path: Path) -> None:
    """The adapter refuses a checkout whose source does not match the pinned commit.

    :param tmp_path: Temporary Git checkout location.
    """
    source, _ = _write_test_source_checkout(tmp_path)

    with pytest.raises(ValueError, match=TINYMU_SOURCE_COMMIT):
        resolve_tinymu_source_model(source)


def test_add_embeddings_config_tinymu_incompatible_pq_split_raises() -> None:
    """Known TinyMU vector width rejects an invalid PQ split before augmentation."""
    with pytest.raises(
        ValueError,
        match=r"num_sub_vectors \(7\) must divide the tinymu dim \(3840\)",
    ):
        AddEmbeddingsConfig(
            lance_uri="dataset.lance",
            embeddings=("tinymu",),
            num_sub_vectors=7,
        )


def test_add_embeddings_config_tinymu_source_string_coerces_path() -> None:
    """Hydra source-directory strings remain strict Path values."""
    config = AddEmbeddingsConfig(
        lance_uri="dataset.lance",
        embeddings=("tinymu",),
        tinymu_source_dir="/opt/TinyMU",  # type: ignore[arg-type]
    )

    assert config.tinymu_source_dir == Path("/opt/TinyMU")


def test_add_embeddings_config_tinymu_source_env_name_is_stable() -> None:
    """The documented environment fallback remains a single explicit boundary."""
    assert TINYMU_SOURCE_DIR_ENV == "TINYMU_SOURCE_DIR"
    assert TINYMU_FRONTEND.embedding_dim == 3_840
