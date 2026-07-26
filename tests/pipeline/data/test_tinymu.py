"""Behavioral tests for the pinned TinyMU MATPAC adapter."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import synth_setter.pipeline.data.add_embeddings as add_embeddings_module
import synth_setter.pipeline.data.tinymu as tinymu_module
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


def test_resolve_tinymu_checkpoint_missing_local_file_raises(tmp_path: Path) -> None:
    """A missing local override fails before any model-loading side effect.

    :param tmp_path: Temporary checkpoint parent.
    """
    missing = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match=str(missing)):
        resolve_tinymu_checkpoint(str(missing))


def test_resolve_tinymu_checkpoint_verified_cache_hit_avoids_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A verified cached artifact is returned without contacting R2.

    :param monkeypatch: Fixture isolating the model cache and transfer boundary.
    :param tmp_path: Temporary cache location.
    """
    artifact = b"trusted cached checkpoint"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    checkpoint = cache_dir / tinymu_module.TINYMU_CHECKPOINT_NAME
    checkpoint.write_bytes(artifact)
    monkeypatch.setattr(
        tinymu_module, "TINYMU_CHECKPOINT_SHA256", hashlib.sha256(artifact).hexdigest()
    )
    monkeypatch.setattr(tinymu_module, "embedding_model_dir", lambda _name: cache_dir)

    def unexpected_download(_uri: str, _destination: Path) -> None:
        raise AssertionError("cache hit attempted an R2 download")

    monkeypatch.setattr(tinymu_module.r2_io, "download_to_path", unexpected_download)

    assert resolve_tinymu_checkpoint(DEFAULT_TINYMU_CHECKPOINT) == checkpoint


def test_resolve_tinymu_checkpoint_tempfile_failure_preserves_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Failure to allocate a download file is not masked by cleanup.

    :param monkeypatch: Fixture failing temporary-file allocation.
    :param tmp_path: Temporary cache location.
    """
    monkeypatch.setattr(tinymu_module, "embedding_model_dir", lambda _name: tmp_path)
    monkeypatch.setattr(tinymu_module.r2_io, "ensure_r2_env_loaded", lambda: None)

    def fail_tempfile(**_kwargs: object) -> None:
        raise OSError("temporary file unavailable")

    monkeypatch.setattr(tinymu_module.tempfile, "NamedTemporaryFile", fail_tempfile)

    with pytest.raises(OSError, match="temporary file unavailable"):
        resolve_tinymu_checkpoint(DEFAULT_TINYMU_CHECKPOINT)


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


def test_tinymu_git_identity_probe_timeout_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stalled source checkout fails within the adapter's bounded probe.

    :param monkeypatch: Fixture simulating a stalled Git process.
    :param tmp_path: Candidate checkout location.
    """
    seen_timeout: list[float] = []

    def timeout(*_args: object, timeout: float, **_kwargs: object) -> None:
        seen_timeout.append(timeout)
        raise subprocess.TimeoutExpired(cmd=["git"], timeout=timeout)

    monkeypatch.setattr(tinymu_module.subprocess, "run", timeout)

    with pytest.raises(ValueError, match="source identity probe timed out"):
        tinymu_module._git_output(tmp_path, "rev-parse", "HEAD")

    assert seen_timeout == [30]


def test_tinymu_git_identity_probe_non_repository_raises(tmp_path: Path) -> None:
    """A directory without Git identity reports the source-probe failure.

    :param tmp_path: Non-repository source directory.
    """
    with pytest.raises(ValueError, match="source identity probe failed"):
        tinymu_module._git_output(tmp_path, "rev-parse", "HEAD")


def test_resolve_tinymu_source_model_missing_checkout_raises(tmp_path: Path) -> None:
    """A missing external checkout fails with its resolved location.

    :param tmp_path: Temporary source parent.
    """
    missing = tmp_path / "TinyMU"

    with pytest.raises(FileNotFoundError, match=str(missing)):
        resolve_tinymu_source_model(missing)


def test_resolve_tinymu_source_model_missing_module_raises(tmp_path: Path) -> None:
    """A checkout without the pinned model module fails before Git probing.

    :param tmp_path: Temporary source parent.
    """
    source = tmp_path / "TinyMU"
    source.mkdir()

    with pytest.raises(FileNotFoundError, match=str(TINYMU_SOURCE_MODEL_PATH)):
        resolve_tinymu_source_model(source)


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


def test_configured_tinymu_source_model_explicit_checkout_resolves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit source path takes precedence over the environment fallback.

    :param monkeypatch: Fixture setting source identities and a conflicting environment value.
    :param tmp_path: Temporary Git checkout location.
    """
    source, commit = _write_test_source_checkout(tmp_path)
    monkeypatch.setattr(tinymu_module, "TINYMU_SOURCE_COMMIT", commit)
    monkeypatch.setenv(TINYMU_SOURCE_DIR_ENV, str(tmp_path / "wrong-source"))

    assert tinymu_module.configured_tinymu_source_model(source) == (
        source / TINYMU_SOURCE_MODEL_PATH
    )


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


def test_load_tinymu_source_module_without_import_spec_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unsupported source path reports that Python cannot load it.

    :param monkeypatch: Fixture simulating an unavailable import specification.
    :param tmp_path: Temporary source module location.
    """
    model_path = tmp_path / "model.py"
    monkeypatch.setattr(tinymu_module.importlib.util, "spec_from_file_location", lambda *_: None)

    with pytest.raises(ImportError, match="cannot load TinyMU MATPAC source module"):
        tinymu_module._load_source_module(model_path)


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


@pytest.mark.parametrize(
    ("num_samples", "sample_rate"),
    [(0, 16_000), (2_800, 0)],
)
def test_tinymu_num_latent_frames_nonpositive_input_raises(
    num_samples: int, sample_rate: int
) -> None:
    """Non-positive source dimensions fail before frame arithmetic.

    :param num_samples: Candidate source clip length.
    :param sample_rate: Candidate source sample rate.
    """
    with pytest.raises(ValueError, match="need positive num_samples/sample_rate"):
        tinymu_num_latent_frames(num_samples, sample_rate)


class _StateLoadingModel:
    """Record MATPAC state-loading and freezing behavior."""

    def __init__(self, missing_keys: list[str], unexpected_keys: list[str]) -> None:
        """Initialize candidate state incompatibilities.

        :param missing_keys: Model keys absent from the checkpoint.
        :param unexpected_keys: Checkpoint keys absent from the model.
        """
        self.cfg = _model_contract().cfg
        self.missing_keys = missing_keys
        self.unexpected_keys = unexpected_keys
        self.device: str | None = None
        self.is_eval = False
        self.loaded_state: dict[str, torch.Tensor] | None = None
        self.requires_grad: bool | None = None
        self.strict: bool | None = None

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], *, strict: bool
    ) -> SimpleNamespace:
        self.loaded_state = state_dict
        self.strict = strict
        return SimpleNamespace(
            missing_keys=self.missing_keys,
            unexpected_keys=self.unexpected_keys,
        )

    def to(self, device: str) -> _StateLoadingModel:
        self.device = device
        return self

    def eval(self) -> _StateLoadingModel:
        self.is_eval = True
        return self

    def requires_grad_(self, requires_grad: bool = True) -> _StateLoadingModel:
        self.requires_grad = requires_grad
        return self


class _StateLoadingModule:
    """Expose one recording model through the external constructor contract."""

    def __init__(self, model: _StateLoadingModel) -> None:
        """Initialize the model returned by the external constructor.

        :param model: Recording model to return.
        """
        self.model = model
        self.constructor_args: tuple[str, bool] | None = None

    def matpac_wrapper(
        self, *, inference_type: str, pull_time_dimension: bool
    ) -> _StateLoadingModel:
        self.constructor_args = inference_type, pull_time_dimension
        return self.model


def test_load_tinymu_model_valid_state_returns_frozen_eval_model(tmp_path: Path) -> None:
    """A compatible checkpoint is loaded on CPU before the model is frozen.

    :param tmp_path: Temporary checkpoint location.
    """
    checkpoint = tmp_path / "matpac.pt"
    torch.save({"encoder.weight": torch.tensor([1.5])}, checkpoint)
    model = _StateLoadingModel(
        missing_keys=sorted(tinymu_module._EXPECTED_UNPERSISTED_BUFFERS),
        unexpected_keys=[],
    )
    module = _StateLoadingModule(model)

    loaded = tinymu_module._load_tinymu_model(
        cast("tinymu_module._MatpacModule", module), checkpoint, "cpu"
    )

    assert loaded is model
    assert module.constructor_args == ("precise", False)
    assert model.loaded_state is not None
    torch.testing.assert_close(model.loaded_state["encoder.weight"], torch.tensor([1.5]))
    assert model.strict is False
    assert model.device == "cpu"
    assert model.is_eval is True
    assert model.requires_grad is False


@pytest.mark.parametrize(
    ("missing_keys", "unexpected_keys"),
    [
        ([], []),
        (sorted(tinymu_module._EXPECTED_UNPERSISTED_BUFFERS), ["unexpected.weight"]),
    ],
)
def test_load_tinymu_model_incompatible_state_raises(
    missing_keys: list[str], unexpected_keys: list[str], tmp_path: Path
) -> None:
    """Missing persisted state or unexpected tensors fail before device transfer.

    :param missing_keys: Candidate state keys absent from the checkpoint.
    :param unexpected_keys: Candidate checkpoint keys absent from the model.
    :param tmp_path: Temporary checkpoint location.
    """
    checkpoint = tmp_path / "matpac.pt"
    torch.save({}, checkpoint)
    model = _StateLoadingModel(missing_keys, unexpected_keys)

    with pytest.raises(ValueError, match="checkpoint state is incompatible"):
        tinymu_module._load_tinymu_model(
            cast("tinymu_module._MatpacModule", _StateLoadingModule(model)),
            checkpoint,
            "cpu",
        )

    assert model.device is None


class _ChunkModel:
    """Return deterministic MATPAC-oriented embeddings from each source row."""

    def __init__(self, *, frame_delta: int = 0, nonfinite: bool = False) -> None:
        """Initialize output corruption controls.

        :param frame_delta: Offset from the contract token count.
        :param nonfinite: Whether to inject NaN into the output.
        """
        self.frame_delta = frame_delta
        self.nonfinite = nonfinite
        self.batch_sizes: list[int] = []

    def __call__(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.batch_sizes.append(len(inputs))
        frames = tinymu_num_latent_frames(inputs.shape[-1], TINYMU_FRONTEND.sample_rate)
        frames += self.frame_delta
        row_values = inputs[:, :1, None]
        embeddings = row_values.expand(-1, frames, TINYMU_FRONTEND.embedding_dim).clone()
        if self.nonfinite:
            embeddings[0, 0, 0] = torch.nan
        return embeddings, None


def test_encode_tinymu_chunk_valid_output_transposes_to_conditioning_orientation() -> None:
    """MATPAC token-major output becomes contiguous channel-major conditioning."""
    chunk = np.stack(
        [
            np.full(16_000, 0.25, dtype=np.float32),
            np.full(16_000, 0.75, dtype=np.float32),
        ]
    )

    encoded = tinymu_module._encode_tinymu_chunk(
        cast("tinymu_module._MatpacModel", _ChunkModel()), chunk, "cpu"
    )

    assert encoded.shape == (2, TINYMU_FRONTEND.embedding_dim, 7)
    assert encoded.dtype == np.float32
    assert encoded.flags.c_contiguous
    assert encoded[0, 0, 0] == pytest.approx(0.25)
    assert encoded[1, 0, 0] == pytest.approx(0.75)


def test_encode_tinymu_chunk_wrong_token_count_raises() -> None:
    """An upstream token-count drift fails before persistence."""
    chunk = np.zeros((2, 16_000), dtype=np.float32)

    with pytest.raises(ValueError, match="encoder produced shape"):
        tinymu_module._encode_tinymu_chunk(
            cast("tinymu_module._MatpacModel", _ChunkModel(frame_delta=-1)),
            chunk,
            "cpu",
        )


def test_encode_tinymu_chunk_nonfinite_output_raises() -> None:
    """A non-finite model output fails before conditioning persistence."""
    chunk = np.zeros((2, 16_000), dtype=np.float32)

    with pytest.raises(ValueError, match="encoder produced non-finite values"):
        tinymu_module._encode_tinymu_chunk(
            cast("tinymu_module._MatpacModel", _ChunkModel(nonfinite=True)),
            chunk,
            "cpu",
        )


def test_load_tinymu_audio_encoder_batches_and_preserves_row_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The loaded encoder bounds inference chunks and concatenates every row in order.

    :param monkeypatch: Fixture isolating verified source and checkpoint boundaries.
    :param tmp_path: Temporary source and checkpoint paths.
    """
    model = _ChunkModel()
    source_model = tmp_path / "model.py"
    checkpoint = tmp_path / "matpac.pt"
    monkeypatch.setattr(
        tinymu_module,
        "configured_tinymu_source_model",
        lambda source_dir: source_model,
    )
    monkeypatch.setattr(
        tinymu_module,
        "resolve_tinymu_checkpoint",
        lambda checkpoint_uri: checkpoint,
    )
    monkeypatch.setattr(tinymu_module, "_load_source_module", lambda _path: object())
    monkeypatch.setattr(
        tinymu_module,
        "_load_tinymu_model",
        lambda _module, _checkpoint, _device: model,
    )
    monkeypatch.setattr(tinymu_module, "resolve_git_sha", lambda: "test-sha")
    audio = np.stack(
        [np.full((1, 16_000), row / 20, dtype=np.float32) for row in range(17)]
    )

    encode = tinymu_module.load_tinymu_audio_encoder(
        "local-checkpoint", source_dir=tmp_path, device="cpu"
    )
    encoded = encode(audio, 16_000)

    assert model.batch_sizes == [16, 1]
    assert encoded.shape == (17, TINYMU_FRONTEND.embedding_dim, 7)
    np.testing.assert_allclose(encoded[:, 0, 0], np.arange(17) / 20)


def test_tinymu_registry_loader_returns_adapter_encoder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The registry threads config into TinyMU and returns its usable encoder.

    :param monkeypatch: Fixture replacing the already-tested external adapter boundary.
    :param tmp_path: Configured external checkout path.
    """
    seen: list[tuple[str, Path | None, str]] = []

    def load(
        checkpoint: str, *, source_dir: Path | None, device: str
    ) -> tinymu_module.TinyMUEncodeFn:
        seen.append((checkpoint, source_dir, device))

        def encode(audio: np.ndarray, sample_rate: int) -> np.ndarray:
            del sample_rate
            return np.full(
                (len(audio), TINYMU_FRONTEND.embedding_dim, 7),
                0.5,
                dtype=np.float32,
            )

        return encode

    monkeypatch.setattr(add_embeddings_module, "load_tinymu_audio_encoder", load)
    config = AddEmbeddingsConfig(
        lance_uri="dataset.lance",
        embeddings=("tinymu",),
        tinymu_source_dir=tmp_path,
        device="cpu",
    )

    encoder = cast(
        "tinymu_module.TinyMUEncodeFn",
        EMBEDDING_REGISTRY["tinymu"].load_encoder("checkpoint.pt", config),
    )
    encoded = encoder(np.zeros((2, 1, 16_000), dtype=np.float32), 16_000)

    assert seen == [("checkpoint.pt", tmp_path, "cpu")]
    assert encoded.shape == (2, TINYMU_FRONTEND.embedding_dim, 7)
    assert np.all(encoded == 0.5)


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
