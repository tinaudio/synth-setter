"""Behavioral tests for the registry-driven embedding augmenter."""

from __future__ import annotations

import gc
import importlib.util
import os
import subprocess
import sys
import weakref
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import lance
import numpy as np
import pyarrow as pa
import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from structlog.testing import capture_logs

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    CLAP_FIELD,
    M2L_FIELD,
    PARAM_ARRAY_FIELD,
    SAME_L_FIELD,
    SAME_S_FIELD,
)
from synth_setter.pipeline.data.add_embeddings import (
    CLAP_EMBEDDING_DIM,
    DEFAULT_CLAP_CHECKPOINT,
    DEFAULT_LANCE_BATCH_SIZE,
    DEFAULT_SAME_L_CHECKPOINT,
    DEFAULT_SAME_S_CHECKPOINT,
    EMBEDDING_REGISTRY,
    SAME_DOWNSAMPLING_RATIO,
    SAME_EMBEDDING_DIM,
    SAME_LATENT_FRAMES,
    SAME_SAMPLE_RATE,
    EmbeddingSpec,
    Encoder,
    IndexSpec,
    _configure_lance_logging,
    _downmix_to_mono,
    _require_extras,
    _resolve_same_checkpoint_dir,
    _write_columns,
    add_embeddings,
    build_index,
    load_clap_audio_encoder,
    load_m2l_audio_encoder,
    load_same_audio_encoder,
    same_encoder_input,
    same_num_latent_frames,
)
from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig
from synth_setter.workspace import operator_workspace
from tests.helpers.finalize_shards import build_lance_smoke_spec, write_minimal_lance_shard
from tests.helpers.lance_fixtures import write_lance_shard

_SAMPLE_RATE = 44100
_FIXTURE_SAMPLES = 16
_FIXTURE_FRAMES = 2
_M2L_TIME = 3
_LANCE_URI = "r2://bucket/run/train.lance"


def _fake_m2l(audio: np.ndarray) -> np.ndarray:
    """Encode audio as a deterministic ``(B, C*4, 3)`` latent.

    :param audio: ``(B, C, T)`` audio batch.
    :returns: Deterministic m2l-shaped latent batch.
    """
    per_channel = np.repeat(audio.mean(axis=2), 4, axis=1)
    return np.repeat(per_channel[:, :, None], _M2L_TIME, axis=2)


def _fake_clap(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """Encode mono audio as deterministic CLAP-width vectors.

    :param mono: ``(B, T)`` mono batch.
    :param sample_rate: Ignored sample rate.
    :returns: Deterministic CLAP-shaped vectors.
    """
    del sample_rate
    return np.repeat(mono.mean(axis=1, keepdims=True), CLAP_EMBEDDING_DIM, axis=1)


def _distinct_clap(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """Encode each row as a distinct exact-search vector.

    :param mono: ``(B, T)`` mono batch.
    :param sample_rate: Ignored sample rate.
    :returns: Distinct CLAP-width vectors.
    """
    del sample_rate
    output = np.zeros((len(mono), CLAP_EMBEDDING_DIM), dtype=np.float32)
    output[:, 0] = mono.mean(axis=1)
    output[:, 1] = np.arange(len(mono), dtype=np.float32)
    return output


def _fake_same(fill: float) -> Callable[[np.ndarray], np.ndarray]:
    """Build a deterministic SAME encoder.

    :param fill: Constant value used for every latent cell.
    :returns: Encoder over prepared stereo audio.
    """

    def encode(stereo: np.ndarray) -> np.ndarray:
        frames = same_num_latent_frames(stereo.shape[2], SAME_SAMPLE_RATE)
        return np.full((len(stereo), SAME_EMBEDDING_DIM, frames), fill, dtype=np.float32)

    return encode


def _encoder_for(name: str) -> Callable[..., np.ndarray]:
    """Return the fake encoder matching a registry key.

    :param name: Embedding registry key.
    :returns: Matching fake encoder.
    """
    if name == "m2l":
        return _fake_m2l
    if name == "clap":
        return _fake_clap
    return _fake_same(0.25 if name == "same_s" else 0.75)


def _fake_spec(name: str, events: list[str] | None = None) -> EmbeddingSpec:
    """Copy a production spec with a dependency-free loader.

    :param name: Registry key to copy.
    :param events: Optional list receiving loader events.
    :returns: Spec using a fake encoder and no optional-extra gate.
    """

    def load(checkpoint: str, device: str | None) -> Callable[..., np.ndarray]:
        del checkpoint, device
        if events is not None:
            events.append(f"load:{name}")
        return _encoder_for(name)

    return replace(EMBEDDING_REGISTRY[name], requires_extra=None, load_encoder=load)


def _install_fake_specs(
    monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...], events: list[str] | None = None
) -> None:
    """Install dependency-free registry entries for an endpoint test.

    :param monkeypatch: Fixture restoring registry entries after the test.
    :param names: Registry keys to replace.
    :param events: Optional list receiving loader events.
    """
    for name in names:
        monkeypatch.setitem(EMBEDDING_REGISTRY, name, _fake_spec(name, events))


def _audio_dataset(uri: Path, rows: int, *, channels: int = 2) -> np.ndarray:
    """Write a fixed-shape audio Lance dataset.

    :param uri: Output dataset path.
    :param rows: Number of rows to write.
    :param channels: Number of audio channels.
    :returns: Audio values written to the dataset.
    """
    rng = np.random.default_rng(rows)
    audio = rng.random((rows, channels, _FIXTURE_SAMPLES)).astype(np.float16)
    params = rng.random((rows, 3)).astype(np.float32)
    write_lance_shard(uri, {AUDIO_FIELD: audio, PARAM_ARRAY_FIELD: params})
    return audio


def _empty_audio_dataset(uri: Path) -> None:
    """Write an empty fixed-shape audio Lance dataset.

    :param uri: Output dataset path.
    """
    tensor_type = pa.fixed_shape_tensor(pa.float16(), [2, _FIXTURE_SAMPLES])
    storage = pa.array([], type=tensor_type.storage_type)
    lance.write_dataset(
        pa.table({AUDIO_FIELD: pa.ExtensionArray.from_storage(tensor_type, storage)}), str(uri)
    )


def _run_udf_in_process(
    dataset: lance.LanceDataset,
    udf: Callable[[pa.RecordBatch], pa.RecordBatch],
    *,
    read_columns: list[str],
    batch_size: int,
) -> None:
    """Run a Lance batch UDF synchronously for deterministic assertions.

    :param dataset: Local dataset supplying batches.
    :param udf: Batch transform under test.
    :param read_columns: Source columns supplied to the transform.
    :param batch_size: Maximum rows per invocation.
    """
    for batch in dataset.to_batches(columns=read_columns, batch_size=batch_size):
        udf(batch)


def _compose_add_embeddings(*overrides: str) -> DictConfig:
    """Compose the shipped embedding config.

    :param *overrides: Additional Hydra overrides.
    :returns: Composed Hydra config.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        return compose(
            config_name="add_embeddings",
            return_hydra_config=True,
            overrides=[f"lance_uri={_LANCE_URI}", *overrides],
        )


@pytest.mark.parametrize(
    ("audio", "expected"),
    [
        (
            np.array([[[1.0, 3.0], [3.0, 5.0]]], dtype=np.float16),
            np.array([[2.0, 4.0]], dtype=np.float32),
        ),
        (
            np.array([[[1.0, 2.0, 3.0]]], dtype=np.float16),
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        ),
    ],
)
def test_downmix_to_mono_with_any_channel_count_averages_to_float32(
    audio: np.ndarray, expected: np.ndarray
) -> None:
    """CLAP input preparation collapses channels without losing mono values.

    :param audio: One input audio batch.
    :param expected: Expected float32 mono signal.
    """
    mono = _downmix_to_mono(audio)

    assert mono.dtype == np.float32
    np.testing.assert_allclose(mono, expected)


def test_embedding_registry_contains_peer_specs_with_expected_policies() -> None:
    """The registry is the single source of truth for all supported embeddings."""
    assert set(EMBEDDING_REGISTRY) == {"clap", "m2l", "same_s", "same_l"}
    assert EMBEDDING_REGISTRY["clap"].index == IndexSpec()
    assert EMBEDDING_REGISTRY["m2l"].index is None
    assert EMBEDDING_REGISTRY["same_s"].requires_extra == "same"
    assert EMBEDDING_REGISTRY["same_l"].requires_extra == "same"
    assert EMBEDDING_REGISTRY["clap"].co_resident is True
    assert EMBEDDING_REGISTRY["m2l"].co_resident is True
    assert EMBEDDING_REGISTRY["same_s"].co_resident is False
    assert EMBEDDING_REGISTRY["same_l"].co_resident is False


def test_embedding_spec_when_mutated_raises_frozen_instance_error() -> None:
    """Registry policy objects are immutable after construction."""
    with pytest.raises(FrozenInstanceError):
        EMBEDDING_REGISTRY["clap"].column = "changed"  # type: ignore[misc]


def test_add_embeddings_config_composition_surfaces_registry_defaults() -> None:
    """The shipped Hydra config preserves the clap+m2l default behavior."""
    cfg = _compose_add_embeddings()
    try:
        assert cfg.lance_uri == _LANCE_URI
        assert list(cfg.embeddings) == ["clap", "m2l"]
        assert dict(cfg.checkpoints) == {}
        assert cfg.device is None
        assert cfg.batch_size == DEFAULT_LANCE_BATCH_SIZE
        assert cfg.build_index is True
        assert cfg.num_partitions is None
        assert cfg.num_sub_vectors == 16
        assert cfg.metric == "cosine"
        assert cfg.resume_cache is None
        assert cfg.debug is False
        assert AddEmbeddingsConfig.from_hydra_cfg(cfg) == AddEmbeddingsConfig(
            lance_uri=_LANCE_URI
        )
    finally:
        GlobalHydra.instance().clear()


def test_add_embeddings_config_from_hydra_coerces_embedding_list_to_tuple() -> None:
    """A Hydra embedding list validates into an ordered tuple."""
    cfg = _compose_add_embeddings("embeddings=[same_s,clap]")
    try:
        config = AddEmbeddingsConfig.from_hydra_cfg(cfg)
    finally:
        GlobalHydra.instance().clear()
    assert config.embeddings == ("same_s", "clap")


@pytest.mark.parametrize("bad", [["unknown"], ["clap", "unknown"]])
def test_add_embeddings_config_with_unknown_embedding_raises(bad: list[str]) -> None:
    """Unknown embedding tokens fail at the config boundary.

    :param bad: Selection containing an unknown token.
    """
    with pytest.raises(ValueError, match="embeddings"):
        AddEmbeddingsConfig(lance_uri=_LANCE_URI, embeddings=bad)  # type: ignore[arg-type]


def test_add_embeddings_config_with_duplicate_embedding_raises() -> None:
    """A registry entry can be selected at most once."""
    with pytest.raises(ValueError, match="embeddings .* has duplicate entries"):
        AddEmbeddingsConfig(
            lance_uri=_LANCE_URI, embeddings=["clap", "clap"]  # type: ignore[arg-type]
        )


def test_add_embeddings_config_with_empty_embedding_selection_raises() -> None:
    """An empty registry selection is rejected instead of becoming a silent no-op."""
    with pytest.raises(ValueError, match="embeddings must select at least one registry key"):
        AddEmbeddingsConfig(lance_uri=_LANCE_URI, embeddings=())


def test_add_embeddings_config_with_unknown_checkpoint_key_raises() -> None:
    """Checkpoint overrides are constrained to registry keys."""
    with pytest.raises(ValueError, match="checkpoints"):
        AddEmbeddingsConfig(lance_uri=_LANCE_URI, checkpoints={"unknown": "weights"})


def test_add_embeddings_config_with_checkpoint_override_preserves_mapping() -> None:
    """A known checkpoint override remains keyed by its embedding name."""
    config = AddEmbeddingsConfig(
        lance_uri=_LANCE_URI, checkpoints={"same_s": "stabilityai/SAME-S"}
    )
    assert config.checkpoints == {"same_s": "stabilityai/SAME-S"}


def test_add_embeddings_config_with_resume_cache_string_coerces_path() -> None:
    """Hydra string paths become Path values under strict validation."""
    config = AddEmbeddingsConfig(
        lance_uri=_LANCE_URI, resume_cache="cache/embed.cache"  # type: ignore[arg-type]
    )
    assert config.resume_cache == Path("cache/embed.cache")


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("num_sub_vectors", 0, "Input should be greater than or equal to 1"),
        ("num_sub_vectors", -1, "Input should be greater than or equal to 1"),
        ("num_sub_vectors", 15, r"num_sub_vectors \(15\) must divide the clap dim \(512\)"),
        ("num_partitions", 0, "Input should be greater than or equal to 1"),
        ("num_partitions", -1, "Input should be greater than or equal to 1"),
        ("metric", "banana", r"metric 'banana' must be one of \['cosine', 'dot', 'l2'\]"),
    ],
)
def test_add_embeddings_config_with_invalid_index_setting_raises(
    field: str, bad: object, message: str
) -> None:
    """Invalid IVF_PQ settings fail with actionable validator diagnostics.

    :param field: Config field under test.
    :param bad: Invalid field value.
    :param message: Expected diagnostic.
    """
    with pytest.raises(ValueError, match=message):
        AddEmbeddingsConfig.model_validate({"lance_uri": _LANCE_URI, field: bad}, strict=True)


@pytest.mark.parametrize("name", ["clap", "m2l", "same_s", "same_l"])
def test_embedding_spec_encode_column_for_valid_encoder_builds_arrow_array(name: str) -> None:
    """Every registry encoder closure preserves its exact shape and values.

    :param name: Registry key under test.
    """
    audio = np.random.default_rng(0).random((3, 2, _FIXTURE_SAMPLES)).astype(np.float16)
    spec = EMBEDDING_REGISTRY[name]
    encoder = _encoder_for(name)

    array = spec.encode_column(audio, _SAMPLE_RATE, encoder)

    assert len(array) == 3
    if name == "clap":
        assert array.type == pa.list_(pa.float32(), CLAP_EMBEDDING_DIM)
        np.testing.assert_allclose(np.asarray(array.to_pylist()), _fake_clap(_downmix_to_mono(audio), _SAMPLE_RATE))
    else:
        assert isinstance(array.type, pa.FixedShapeTensorType)
        assert array.type.value_type == pa.float32()
        values = pa.chunked_array([array]).combine_chunks().to_numpy_ndarray()
        expected = encoder(audio) if name == "m2l" else encoder(same_encoder_input(audio, _SAMPLE_RATE))
        np.testing.assert_allclose(values, expected)


@pytest.mark.parametrize("value", [np.nan, np.inf])
@pytest.mark.parametrize("name", ["clap", "m2l", "same_s", "same_l"])
def test_embedding_spec_encode_column_with_nonfinite_output_raises(
    name: str, value: float
) -> None:
    """No registry closure permits NaN or infinity to land.

    :param name: Registry key under test.
    :param value: Non-finite cell value emitted by the encoder.
    """
    audio = np.zeros((2, 2, _FIXTURE_SAMPLES), dtype=np.float16)
    base = _encoder_for(name)

    def poisoned(*args: object) -> np.ndarray:
        output = np.array(base(*args), dtype=np.float32)
        output.flat[0] = value
        return output

    with pytest.raises(ValueError, match=f"{EMBEDDING_REGISTRY[name].column} embeddings contain non-finite values"):
        EMBEDDING_REGISTRY[name].encode_column(audio, _SAMPLE_RATE, poisoned)


def test_same_embedding_spec_prepares_stereo_before_encoder_call() -> None:
    """The SAME closure owns mono duplication and float32 conversion."""
    mono = np.random.default_rng(4).random((2, 1, _FIXTURE_SAMPLES)).astype(np.float16)
    seen: list[np.ndarray] = []

    def recording(stereo: np.ndarray) -> np.ndarray:
        seen.append(stereo)
        return _fake_same(1.0)(stereo)

    EMBEDDING_REGISTRY["same_s"].encode_column(mono, SAME_SAMPLE_RATE, recording)

    assert seen[0].shape == (2, 2, _FIXTURE_SAMPLES)
    assert seen[0].dtype == np.float32


@pytest.mark.parametrize(
    ("name", "encoder", "message"),
    [
        ("m2l", lambda audio: _fake_m2l(audio)[:-1], "expected 2 rows"),
        (
            "clap",
            lambda mono, sample_rate: _fake_clap(mono, sample_rate)[:-1],
            r"expected \(2, 512\)",
        ),
        (
            "clap",
            lambda mono, sample_rate: _fake_clap(mono, sample_rate)[:, :256],
            r"expected \(2, 512\)",
        ),
        (
            "same_s",
            lambda stereo: np.zeros((len(stereo), 128, 1), dtype=np.float32),
            r"expected \(2, 256, 2\)",
        ),
    ],
)
def test_embedding_spec_encode_column_with_invalid_shape_raises(
    name: str, encoder: Callable[..., np.ndarray], message: str
) -> None:
    """Each encoder closure rejects outputs outside its row and shape contract.

    :param name: Registry key under test.
    :param encoder: Encoder emitting an invalid shape.
    :param message: Expected diagnostic fragment.
    """
    audio = np.zeros((2, 2, _FIXTURE_SAMPLES), dtype=np.float16)

    with pytest.raises(ValueError, match=message):
        EMBEDDING_REGISTRY[name].encode_column(audio, _SAMPLE_RATE, encoder)


@pytest.mark.parametrize("name", ["clap", "m2l", "same_s", "same_l"])
def test_write_columns_for_single_registry_spec_round_trips_column(
    name: str, tmp_path: Path
) -> None:
    """The unified writer appends each registry column without dropping sources.

    :param name: Registry key under test.
    :param tmp_path: Scratch directory for the dataset.
    """
    uri = tmp_path / f"{name}.lance"
    audio = _audio_dataset(uri, rows=4)
    spec = _fake_spec(name)

    _write_columns(
        lance.dataset(str(uri)),
        [spec],
        _SAMPLE_RATE,
        AddEmbeddingsConfig(lance_uri=str(uri), embeddings=(name,), build_index=False),
    )

    dataset = lance.dataset(str(uri))
    assert {AUDIO_FIELD, PARAM_ARRAY_FIELD, spec.column} <= set(dataset.schema.names)
    column = dataset.to_table(columns=[spec.column]).combine_chunks().column(spec.column).chunk(0)
    if name == "clap":
        values = np.asarray(column.to_pylist(), dtype=np.float32)
        expected = _fake_clap(_downmix_to_mono(audio), _SAMPLE_RATE)
    else:
        values = column.to_numpy_ndarray()
        encoder = _encoder_for(name)
        expected = encoder(audio) if name == "m2l" else encoder(same_encoder_input(audio, _SAMPLE_RATE))
    np.testing.assert_allclose(values, expected)


def test_write_columns_for_co_resident_specs_shares_audio_object(tmp_path: Path) -> None:
    """One UDF decode supplies the same audio object to all co-resident encoders.

    :param tmp_path: Scratch directory for the dataset.
    """
    uri = tmp_path / "shared-decode.lance"
    _audio_dataset(uri, rows=4)
    seen: dict[str, list[int]] = {"clap": [], "m2l": []}

    def recording_spec(name: str) -> EmbeddingSpec:
        original = _fake_spec(name)

        def encode(audio: np.ndarray, sample_rate: int, encoder: Encoder) -> pa.Array:
            seen[name].append(id(audio))
            return original.encode_column(audio, sample_rate, encoder)

        return replace(original, encode_column=encode)

    _write_columns(
        lance.dataset(str(uri)),
        [recording_spec("clap"), recording_spec("m2l")],
        _SAMPLE_RATE,
        AddEmbeddingsConfig(lance_uri=str(uri), build_index=False),
    )

    assert seen["clap"] == seen["m2l"]
    assert lance.dataset(str(uri)).schema.names[-2:] == [CLAP_FIELD, M2L_FIELD]


def test_write_columns_with_empty_spec_group_raises(tmp_path: Path) -> None:
    """The unified writer rejects an empty policy group.

    :param tmp_path: Scratch directory for the dataset.
    """
    uri = tmp_path / "empty-specs.lance"
    _audio_dataset(uri, rows=2)

    with pytest.raises(ValueError, match="no embedding specs given; nothing to write"):
        _write_columns(
            lance.dataset(str(uri)),
            [],
            _SAMPLE_RATE,
            AddEmbeddingsConfig(lance_uri=str(uri)),
        )


def test_write_columns_with_nonpositive_batch_size_raises(tmp_path: Path) -> None:
    """The functional writer rejects a non-positive UDF batch size.

    :param tmp_path: Scratch directory for the dataset.
    """
    uri = tmp_path / "bad-batch.lance"
    _audio_dataset(uri, rows=2)
    config = AddEmbeddingsConfig(lance_uri=str(uri)).model_copy(update={"batch_size": 0})

    with pytest.raises(ValueError, match="batch_size must be >= 1, got 0"):
        _write_columns(lance.dataset(str(uri)), [_fake_spec("m2l")], _SAMPLE_RATE, config)


def test_write_columns_with_existing_target_raises_before_encoder_load(tmp_path: Path) -> None:
    """An existing target fails before any checkpoint loader runs.

    :param tmp_path: Scratch directory for the dataset.
    """
    uri = tmp_path / "existing.lance"
    _audio_dataset(uri, rows=2)
    initial = _fake_spec("same_s")
    config = AddEmbeddingsConfig(
        lance_uri=str(uri), embeddings=("same_s",), build_index=False
    )
    _write_columns(lance.dataset(str(uri)), [initial], _SAMPLE_RATE, config)
    loads: list[str] = []

    with pytest.raises(
        ValueError, match=r"dataset already has embedding column\(s\): \['same_s'\]"
    ):
        _write_columns(
            lance.dataset(str(uri)), [_fake_spec("same_s", loads)], _SAMPLE_RATE, config
        )

    assert loads == []


def test_write_columns_with_empty_dataset_raises(tmp_path: Path) -> None:
    """A rowless source fails before schema inference.

    :param tmp_path: Scratch directory for the dataset.
    """
    uri = tmp_path / "empty.lance"
    _empty_audio_dataset(uri)
    with pytest.raises(ValueError, match="dataset has no rows to embed"):
        _write_columns(
            lance.dataset(str(uri)),
            [_fake_spec("m2l")],
            _SAMPLE_RATE,
            AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("m2l",)),
        )


def test_write_columns_with_missing_audio_raises(tmp_path: Path) -> None:
    """A source without audio fails before a UDF is built.

    :param tmp_path: Scratch directory for the dataset.
    """
    uri = tmp_path / "no-audio.lance"
    write_lance_shard(uri, {PARAM_ARRAY_FIELD: np.zeros((2, 3), dtype=np.float32)})
    with pytest.raises(ValueError, match="dataset has no 'audio' column to embed"):
        _write_columns(
            lance.dataset(str(uri)),
            [_fake_spec("m2l")],
            _SAMPLE_RATE,
            AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("m2l",)),
        )


def test_write_columns_after_success_removes_resume_cache(tmp_path: Path) -> None:
    """A committed UDF pass removes its now-consumed resume cache.

    :param tmp_path: Scratch directory for the dataset and cache.
    """
    uri = tmp_path / "resume.lance"
    resume_cache = tmp_path / "resume.cache"
    _audio_dataset(uri, rows=5)

    _write_columns(
        lance.dataset(str(uri)),
        [_fake_spec("m2l")],
        _SAMPLE_RATE,
        AddEmbeddingsConfig(
            lance_uri=str(uri),
            embeddings=("m2l",),
            resume_cache=resume_cache,
            build_index=False,
        ),
    )

    assert not resume_cache.exists()
    assert M2L_FIELD in lance.dataset(str(uri)).schema.names


@pytest.mark.parametrize("name", ["m2l", "same_s"])
def test_write_columns_with_resume_cache_skips_completed_batches_after_interruption(
    name: str, tmp_path: Path
) -> None:
    """A rerun consumes cached batches instead of re-encoding them.

    :param name: Co-resident or SAME registry path under test.
    :param tmp_path: Scratch directory for the dataset and cache.
    """
    uri = tmp_path / f"resume-{name}.lance"
    resume_cache = tmp_path / f"resume-{name}.cache"
    audio = _audio_dataset(uri, rows=6)
    first_calls: list[int] = []
    second_calls: list[int] = []
    base_spec = _fake_spec(name)

    def load_crashing(checkpoint: str, device: str | None) -> Callable[..., np.ndarray]:
        del checkpoint, device
        base = _encoder_for(name)

        def encode(*args: object) -> np.ndarray:
            first_calls.append(len(cast(np.ndarray, args[0])))
            if len(first_calls) == 4:
                raise RuntimeError("simulated crash")
            return base(*args)

        return encode

    crashing_spec = replace(base_spec, load_encoder=load_crashing)
    config = AddEmbeddingsConfig(
        lance_uri=str(uri),
        embeddings=(name,),
        batch_size=2,
        resume_cache=resume_cache,
        build_index=False,
    )
    with pytest.raises(OSError, match="simulated crash"):
        _write_columns(lance.dataset(str(uri)), [crashing_spec], _SAMPLE_RATE, config)
    assert resume_cache.exists()

    def load_recording(checkpoint: str, device: str | None) -> Callable[..., np.ndarray]:
        del checkpoint, device
        base = _encoder_for(name)

        def encode(*args: object) -> np.ndarray:
            second_calls.append(len(cast(np.ndarray, args[0])))
            return base(*args)

        return encode

    _write_columns(
        lance.dataset(str(uri)),
        [replace(base_spec, load_encoder=load_recording)],
        _SAMPLE_RATE,
        config,
    )

    assert len(second_calls) < len(first_calls)
    assert not resume_cache.exists()
    column = (
        lance.dataset(str(uri))
        .to_table(columns=[base_spec.column])
        .combine_chunks()
        .column(base_spec.column)
        .chunk(0)
    )
    assert len(column.to_numpy_ndarray()) == len(audio)


def test_write_columns_when_resume_cache_cleanup_fails_logs_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache unlink failure after commit does not roll back written columns.

    :param tmp_path: Scratch directory for the dataset and cache.
    :param monkeypatch: Fixture making cache deletion fail.
    """
    uri = tmp_path / "cleanup.lance"
    resume_cache = tmp_path / "cleanup.cache"
    _audio_dataset(uri, rows=4)

    def deny_unlink(self: Path, missing_ok: bool = False) -> None:
        del missing_ok
        raise PermissionError(f"unlink denied: {self}")

    monkeypatch.setattr(Path, "unlink", deny_unlink)
    with capture_logs() as logs:
        _write_columns(
            lance.dataset(str(uri)),
            [_fake_spec("m2l")],
            _SAMPLE_RATE,
            AddEmbeddingsConfig(
                lance_uri=str(uri),
                embeddings=("m2l",),
                resume_cache=resume_cache,
                build_index=False,
            ),
        )

    warning = next(entry for entry in logs if entry["event"] == "resume_cache_cleanup_failed")
    assert warning["resume_cache"] == str(resume_cache)
    assert "unlink denied" in warning["error"]
    assert M2L_FIELD in lance.dataset(str(uri)).schema.names


def test_write_columns_with_default_batch_size_bounds_work_and_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default writes cap encoder batches and emit bounded completion progress.

    :param tmp_path: Scratch directory for the dataset.
    :param monkeypatch: Fixture running Lance UDF batches in-process.
    """
    uri = tmp_path / "default-progress.lance"
    _audio_dataset(uri, rows=300)
    batch_sizes: list[int] = []
    spec = _fake_spec("m2l")

    def encode(audio: np.ndarray, sample_rate: int, encoder: Encoder) -> pa.Array:
        batch_sizes.append(len(audio))
        return spec.encode_column(audio, sample_rate, encoder)

    monkeypatch.setattr(lance.LanceDataset, "add_columns", _run_udf_in_process)
    with capture_logs() as logs:
        _write_columns(
            lance.dataset(str(uri)),
            [replace(spec, encode_column=encode)],
            _SAMPLE_RATE,
            AddEmbeddingsConfig(
                lance_uri=str(uri), embeddings=("m2l",), build_index=False
            ),
        )

    progress = [entry for entry in logs if entry["event"] == "embedding_progress"]
    events = [entry["event"] for entry in logs]
    assert events.index("inferring_embedding_schema") < events.index("inferred_embedding_schema")
    assert events.index("inferred_embedding_schema") < events.index("embedding_write_started")
    assert max(batch_sizes) == DEFAULT_LANCE_BATCH_SIZE
    assert progress[-1]["rows_processed"] == 300
    assert progress[-1]["total_rows"] == 300
    assert progress[-1]["percent"] == 100.0
    assert len(progress) <= 20


def test_write_columns_with_debug_logs_progress_and_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unified writer reports progress between source and committed versions.

    :param tmp_path: Scratch directory for the dataset.
    :param monkeypatch: Fixture running Lance UDF batches in-process.
    """
    uri = tmp_path / "progress.lance"
    _audio_dataset(uri, rows=5)

    monkeypatch.setattr(lance.LanceDataset, "add_columns", _run_udf_in_process)
    with capture_logs() as logs:
        _write_columns(
            lance.dataset(str(uri)),
            [_fake_spec("m2l")],
            _SAMPLE_RATE,
            AddEmbeddingsConfig(
                lance_uri=str(uri), embeddings=("m2l",), batch_size=2, debug=True
            ),
        )

    progress = [entry for entry in logs if entry["event"] == "embedding_progress"]
    assert [entry["rows_processed"] for entry in progress] == [2, 4, 5]
    assert [entry["batch_rows"] for entry in progress] == [2, 2, 1]
    for field in ("m2l_ms", "batch_ms", "interbatch_ms", "rows_per_second"):
        assert progress[-1][field] >= 0.0
    events = [entry["event"] for entry in logs]
    assert events.index("embedding_write_started") < events.index("embedding_progress")
    assert events.index("embedding_progress") < events.index("wrote_embeddings")
    assert "source_version" in next(
        entry for entry in logs if entry["event"] == "embedding_write_started"
    )
    assert "committed_version" in next(
        entry for entry in logs if entry["event"] == "wrote_embeddings"
    )


def test_require_extras_with_missing_same_dependency_names_uv_sync_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing optional dependency fails with its install command.

    :param monkeypatch: Fixture hiding the SAME dependency.
    """
    real_find_spec = importlib.util.find_spec

    def find_spec(name: str) -> object:
        return None if name == "stable_audio_tools" else real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    with pytest.raises(ImportError, match="uv sync --extra same"):
        _require_extras([EMBEDDING_REGISTRY["same_s"]])


def test_add_embeddings_with_mixed_selection_writes_exact_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mixed selection writes only its requested peer registry entries.

    :param tmp_path: Scratch directory for the finalized shard.
    :param monkeypatch: Fixture installing dependency-free specs.
    """
    uri = tmp_path / "mixed.lance"
    write_minimal_lance_shard(uri, build_lance_smoke_spec())
    selected = ("clap", "m2l", "same_s")
    _install_fake_specs(monkeypatch, selected)

    add_embeddings(
        AddEmbeddingsConfig(lance_uri=str(uri), embeddings=selected, build_index=False)
    )

    names = set(lance.dataset(str(uri)).schema.names)
    assert {CLAP_FIELD, M2L_FIELD, SAME_S_FIELD} <= names
    assert SAME_L_FIELD not in names


def test_add_embeddings_with_checkpoint_override_threads_selected_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A selected spec loader receives its keyed checkpoint override.

    :param tmp_path: Scratch directory for the finalized shard.
    :param monkeypatch: Fixture installing a recording spec.
    """
    uri = tmp_path / "checkpoint.lance"
    write_minimal_lance_shard(uri, build_lance_smoke_spec())
    seen: list[tuple[str, str | None]] = []
    spec = _fake_spec("same_s")

    def load(checkpoint: str, device: str | None) -> Callable[..., np.ndarray]:
        seen.append((checkpoint, device))
        return _fake_same(0.25)

    monkeypatch.setitem(EMBEDDING_REGISTRY, "same_s", replace(spec, load_encoder=load))
    add_embeddings(
        AddEmbeddingsConfig(
            lance_uri=str(uri),
            embeddings=("same_s",),
            checkpoints={"same_s": "custom/same-s"},
            device="cpu",
            build_index=False,
        )
    )

    assert seen == [("custom/same-s", "cpu")]


def test_add_embeddings_with_all_specs_commits_grouped_and_loads_same_sequentially(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Co-resident peers share one commit while SAME models load in separate passes.

    :param tmp_path: Scratch directory for the finalized shard.
    :param monkeypatch: Fixture installing dependency-free recording specs.
    """
    uri = tmp_path / "grouped.lance"
    write_minimal_lance_shard(uri, build_lance_smoke_spec())
    selected = ("clap", "m2l", "same_s", "same_l")
    events: list[str] = []
    _install_fake_specs(monkeypatch, selected, events)
    original_add_columns = lance.LanceDataset.add_columns
    commits: list[tuple[int, int]] = []

    def recording_add_columns(dataset: lance.LanceDataset, *args: Any, **kwargs: Any) -> None:
        source_version = dataset.version
        original_add_columns(dataset, *args, **kwargs)
        commits.append((source_version, dataset.version))

    monkeypatch.setattr(lance.LanceDataset, "add_columns", recording_add_columns)
    add_embeddings(
        AddEmbeddingsConfig(lance_uri=str(uri), embeddings=selected, build_index=False)
    )

    assert len(commits) == 3
    assert events == ["load:clap", "load:m2l", "load:same_s", "load:same_l"]
    schema_names = lance.dataset(str(uri)).schema.names
    assert set(schema_names) >= {CLAP_FIELD, M2L_FIELD, SAME_S_FIELD, SAME_L_FIELD}
    assert schema_names[-4:] == [CLAP_FIELD, M2L_FIELD, SAME_S_FIELD, SAME_L_FIELD]


def test_add_embeddings_with_two_same_specs_releases_first_before_second_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second multi-GB SAME model loads only after the first becomes unreachable.

    :param tmp_path: Scratch directory for the finalized shard.
    :param monkeypatch: Fixture installing lifetime-recording specs.
    """
    uri = tmp_path / "same-residency.lance"
    write_minimal_lance_shard(uri, build_lance_smoke_spec())
    first_encoder: weakref.ReferenceType[Callable[..., np.ndarray]] | None = None

    def load_same_s(checkpoint: str, device: str | None) -> Callable[..., np.ndarray]:
        nonlocal first_encoder
        del checkpoint, device
        encoder = _fake_same(0.25)
        first_encoder = weakref.ref(encoder)
        return encoder

    def load_same_l(checkpoint: str, device: str | None) -> Callable[..., np.ndarray]:
        del checkpoint, device
        gc.collect()
        assert first_encoder is not None
        assert first_encoder() is None
        return _fake_same(0.75)

    monkeypatch.setitem(
        EMBEDDING_REGISTRY,
        "same_s",
        replace(_fake_spec("same_s"), load_encoder=load_same_s),
    )
    monkeypatch.setitem(
        EMBEDDING_REGISTRY,
        "same_l",
        replace(_fake_spec("same_l"), load_encoder=load_same_l),
    )

    add_embeddings(
        AddEmbeddingsConfig(
            lance_uri=str(uri),
            embeddings=("same_s", "same_l"),
            build_index=False,
        )
    )

    assert {SAME_S_FIELD, SAME_L_FIELD} <= set(lance.dataset(str(uri)).schema.names)


def test_add_embeddings_with_non_clap_selection_builds_no_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only specs declaring an index trigger index construction.

    :param tmp_path: Scratch directory for the finalized shard.
    :param monkeypatch: Fixture installing fake specs and an index spy.
    """
    uri = tmp_path / "no-index.lance"
    write_minimal_lance_shard(uri, build_lance_smoke_spec())
    selected = ("m2l", "same_s")
    _install_fake_specs(monkeypatch, selected)
    calls: list[str] = []
    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_embeddings.build_index",
        lambda dataset, column, *, index, config: calls.append(column),
    )

    add_embeddings(AddEmbeddingsConfig(lance_uri=str(uri), embeddings=selected))

    assert calls == []


def test_add_embeddings_with_clap_selection_builds_only_clap_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLAP is the sole registry entry whose index policy is applied.

    :param tmp_path: Scratch directory for the finalized shard.
    :param monkeypatch: Fixture installing fake specs and an index spy.
    """
    uri = tmp_path / "clap-index.lance"
    write_minimal_lance_shard(uri, build_lance_smoke_spec())
    selected = ("same_l", "clap", "m2l")
    _install_fake_specs(monkeypatch, selected)
    calls: list[tuple[str, IndexSpec, int, int, str]] = []

    def record_index(
        dataset: lance.LanceDataset,
        column: str,
        *,
        index: IndexSpec,
        config: AddEmbeddingsConfig,
    ) -> bool:
        del dataset
        calls.append(
            (column, index, cast(int, config.num_partitions), config.num_sub_vectors, config.metric)
        )
        return False

    monkeypatch.setattr("synth_setter.pipeline.data.add_embeddings.build_index", record_index)
    add_embeddings(
        AddEmbeddingsConfig(
            lance_uri=str(uri),
            embeddings=selected,
            num_partitions=4,
            num_sub_vectors=8,
            metric="l2",
        )
    )

    assert calls == [(CLAP_FIELD, IndexSpec(), 4, 8, "l2")]


def test_add_embeddings_existing_mixed_target_guards_all_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any existing selected column blocks every model download.

    :param tmp_path: Scratch directory for the finalized shard.
    :param monkeypatch: Fixture installing dependency-free recording specs.
    """
    uri = tmp_path / "guard.lance"
    write_minimal_lance_shard(uri, build_lance_smoke_spec())
    _install_fake_specs(monkeypatch, ("same_s",))
    add_embeddings(
        AddEmbeddingsConfig(
            lance_uri=str(uri), embeddings=("same_s",), build_index=False
        )
    )
    loads: list[str] = []
    _install_fake_specs(monkeypatch, ("clap", "same_s"), loads)

    with pytest.raises(ValueError, match="same_s"):
        add_embeddings(
            AddEmbeddingsConfig(
                lance_uri=str(uri), embeddings=("clap", "same_s"), build_index=False
            )
        )

    assert loads == []


def test_build_index_with_too_few_rows_skips(tmp_path: Path) -> None:
    """A small CLAP dataset retains exact search without training IVF_PQ.

    :param tmp_path: Scratch directory for the dataset.
    """
    uri = tmp_path / "tiny.lance"
    _audio_dataset(uri, rows=8)
    _write_columns(
        lance.dataset(str(uri)),
        [_fake_spec("clap")],
        _SAMPLE_RATE,
        AddEmbeddingsConfig(
            lance_uri=str(uri), embeddings=("clap",), build_index=False
        ),
    )

    built = build_index(
        lance.dataset(str(uri)),
        CLAP_FIELD,
        index=IndexSpec(),
        config=AddEmbeddingsConfig(lance_uri=str(uri)),
    )

    assert built is False
    assert lance.dataset(str(uri)).list_indices() == []


@pytest.mark.slow
def test_build_index_with_enough_rows_creates_searchable_ivf_pq(tmp_path: Path) -> None:
    """A declared CLAP policy builds a queryable IVF_PQ index.

    :param tmp_path: Scratch directory for the dataset.
    """
    uri = tmp_path / "indexed.lance"
    _audio_dataset(uri, rows=300)
    _write_columns(
        lance.dataset(str(uri)),
        [_fake_spec("clap")],
        _SAMPLE_RATE,
        AddEmbeddingsConfig(
            lance_uri=str(uri), embeddings=("clap",), build_index=False
        ),
    )

    built = build_index(
        lance.dataset(str(uri)),
        CLAP_FIELD,
        index=IndexSpec(),
        config=AddEmbeddingsConfig(
            lance_uri=str(uri), num_partitions=4, num_sub_vectors=16
        ),
    )

    dataset = lance.dataset(str(uri))
    indices = cast("list[dict[str, Any]]", dataset.list_indices())
    assert built is True
    assert any(entry["fields"] == [CLAP_FIELD] for entry in indices)
    hits = dataset.to_table(
        nearest={
            "column": CLAP_FIELD,
            "q": np.ones(CLAP_EMBEDDING_DIM, dtype=np.float32),
            "k": 5,
        }
    )
    assert hits.num_rows == 5


@pytest.mark.slow
def test_clap_exact_search_with_stored_vector_returns_queried_row(tmp_path: Path) -> None:
    """Exact CLAP search returns the source row at zero distance.

    :param tmp_path: Scratch directory for the dataset.
    """
    uri = tmp_path / "exact-search.lance"
    _audio_dataset(uri, rows=64)

    def load_distinct(checkpoint: str, device: str | None) -> Callable[..., np.ndarray]:
        del checkpoint, device
        return _distinct_clap

    spec = replace(_fake_spec("clap"), load_encoder=load_distinct)
    _write_columns(
        lance.dataset(str(uri)),
        [spec],
        _SAMPLE_RATE,
        AddEmbeddingsConfig(
            lance_uri=str(uri), embeddings=("clap",), build_index=False
        ),
    )
    dataset = lance.dataset(str(uri))
    stored = dataset.to_table(columns=[CLAP_FIELD, PARAM_ARRAY_FIELD])
    target_row = 37
    query = np.asarray(stored.column(CLAP_FIELD)[target_row].as_py(), dtype=np.float32)
    expected_params = stored.column(PARAM_ARRAY_FIELD)[target_row].as_py()

    hits = dataset.to_table(
        nearest={"column": CLAP_FIELD, "q": query, "k": 1},
        columns=[PARAM_ARRAY_FIELD],
    )

    assert hits.column(PARAM_ARRAY_FIELD)[0].as_py() == expected_params
    np.testing.assert_allclose(hits.column("_distance")[0].as_py(), 0.0, atol=1e-5)


def test_build_index_with_invalid_subvector_count_raises(tmp_path: Path) -> None:
    """Index validation reports a column-specific dimensionality mismatch.

    :param tmp_path: Scratch directory for the dataset.
    """
    uri = tmp_path / "bad-index.lance"
    vectors = pa.array([[0.0] * 10] * 8, type=pa.list_(pa.float32(), 10))
    lance.write_dataset(pa.table({CLAP_FIELD: vectors}), str(uri))
    config = AddEmbeddingsConfig(lance_uri=str(uri), num_sub_vectors=8)

    with pytest.raises(ValueError, match="num_sub_vectors=8 does not divide clap dim 10"):
        build_index(lance.dataset(str(uri)), CLAP_FIELD, index=IndexSpec(), config=config)


def test_same_encoder_input_with_mono_duplicates_channels() -> None:
    """SAME preparation duplicates mono input and upcasts it."""
    mono = np.random.default_rng(0).random((3, 1, 32)).astype(np.float16)
    prepared = same_encoder_input(mono, SAME_SAMPLE_RATE)
    assert prepared.shape == (3, 2, 32)
    assert prepared.dtype == np.float32
    np.testing.assert_array_equal(prepared[:, 0], prepared[:, 1])
    np.testing.assert_allclose(prepared[:, 0], mono[:, 0].astype(np.float32))


def test_same_encoder_input_with_stereo_at_target_rate_preserves_values() -> None:
    """Conformant stereo audio is only upcast to float32."""
    stereo = np.random.default_rng(1).random((2, 2, 32)).astype(np.float16)

    prepared = same_encoder_input(stereo, SAME_SAMPLE_RATE)

    assert prepared.dtype == np.float32
    np.testing.assert_allclose(prepared, stereo.astype(np.float32))


def test_same_encoder_input_with_half_rate_doubles_sample_count() -> None:
    """SAME input preparation resamples source audio to 44.1 kHz."""
    stereo = np.random.default_rng(2).random((2, 2, 512)).astype(np.float16)

    prepared = same_encoder_input(stereo, SAME_SAMPLE_RATE // 2)

    assert prepared.shape == (2, 2, 1024)
    assert prepared.dtype == np.float32
    assert np.isfinite(prepared).all()


def test_same_encoder_input_with_more_than_two_channels_raises() -> None:
    """Audio without a defined stereo mapping fails with its received shape."""
    surround = np.zeros((1, 3, 32), dtype=np.float32)

    with pytest.raises(
        ValueError,
        match=r"expected a \(B, C, T\) batch with 1 or 2 channels.*\(1, 3, 32\)",
    ):
        same_encoder_input(surround, SAME_SAMPLE_RATE)


def test_same_num_latent_frames_for_standard_render_returns_44() -> None:
    """The standard four-second render produces the conditioning profile width."""
    assert same_num_latent_frames(4 * SAME_SAMPLE_RATE, SAME_SAMPLE_RATE) == 44
    assert SAME_LATENT_FRAMES == 44


@pytest.mark.parametrize(
    ("num_samples", "sample_rate", "expected"),
    [
        (2 * SAME_DOWNSAMPLING_RATIO, SAME_SAMPLE_RATE // 2, 4),
        (SAME_SAMPLE_RATE, SAME_SAMPLE_RATE, 12),
        (SAME_DOWNSAMPLING_RATIO, SAME_SAMPLE_RATE, 2),
    ],
)
def test_same_num_latent_frames_resamples_and_pads_even_blocks(
    num_samples: int, sample_rate: int, expected: int
) -> None:
    """SAME frame math follows resampling and two-hop padding.

    :param num_samples: Source clip length.
    :param sample_rate: Source rate in Hz.
    :param expected: Expected even latent-frame count.
    """
    assert same_num_latent_frames(num_samples, sample_rate) == expected


@pytest.mark.parametrize(
    ("num_samples", "sample_rate"), [(0, SAME_SAMPLE_RATE), (SAME_SAMPLE_RATE, 0)]
)
def test_same_num_latent_frames_with_nonpositive_input_raises(
    num_samples: int, sample_rate: int
) -> None:
    """Zero clip lengths and rates fail with both received values.

    :param num_samples: Invalid source length candidate.
    :param sample_rate: Invalid source rate candidate.
    """
    with pytest.raises(
        ValueError,
        match=f"need positive num_samples/sample_rate, got {num_samples}/{sample_rate}",
    ):
        same_num_latent_frames(num_samples, sample_rate)


def test_same_profile_shape_feeds_embedding_pool() -> None:
    """A profile-width SAME latent is consumable by the training encoder."""
    from synth_setter.models.components.embed_pool import EmbeddingPool

    encoder = EmbeddingPool(
        embed_dim=SAME_EMBEDDING_DIM,
        d_model=32,
        num_heads=4,
        max_seq_len=SAME_LATENT_FRAMES,
    )

    pooled = encoder(torch.randn(2, SAME_EMBEDDING_DIM, SAME_LATENT_FRAMES))

    assert pooled.shape == (2, 32)


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "requested", "expected"),
    [
        (True, True, None, "cuda"),
        (False, True, None, "mps"),
        (False, False, None, "cpu"),
        (True, True, "cpu", "cpu"),
    ],
)
def test_load_m2l_audio_encoder_selects_expected_device(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    mps_available: bool,
    requested: str | None,
    expected: str,
) -> None:
    """The m2l loader honors overrides and CUDA-MPS-CPU priority.

    :param monkeypatch: Fixture replacing hardware discovery and model construction.
    :param cuda_available: Whether CUDA is discoverable.
    :param mps_available: Whether MPS is discoverable.
    :param requested: Explicit device override, or ``None``.
    :param expected: Device expected by the encoder constructor.
    """
    selected: list[str | None] = []
    monkeypatch.setattr("torch.cuda.is_available", lambda: cuda_available)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: mps_available)
    monkeypatch.setattr(
        "music2latent.EncoderDecoder", lambda *, device=None: selected.append(device)
    )

    load_m2l_audio_encoder(requested)

    assert selected == [expected]


@pytest.mark.mps
@pytest.mark.slow
def test_m2l_audio_encoder_on_mps_produces_finite_latents() -> None:
    """The real music2latent encoder produces finite MPS latents."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")

    encode = load_m2l_audio_encoder("mps")
    latents = encode(np.zeros((1, 1, _SAMPLE_RATE), dtype=np.float32))

    assert latents.shape[0] == 1
    assert latents.dtype == np.float32
    assert np.isfinite(latents).all()


def test_load_clap_audio_encoder_with_mps_available_selects_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLAP automatically selects MPS when CUDA is unavailable.

    :param monkeypatch: Fixture replacing hardware discovery and model construction.
    """
    selected: list[str] = []
    model = SimpleNamespace()

    def to(device: str) -> SimpleNamespace:
        selected.append(device)
        return model

    model.to = to
    model.eval = lambda: model
    transformers = SimpleNamespace(
        ClapModel=SimpleNamespace(from_pretrained=lambda checkpoint: model),
        ClapProcessor=SimpleNamespace(from_pretrained=lambda checkpoint: SimpleNamespace()),
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    load_clap_audio_encoder()

    assert selected == ["mps"]


def test_load_clap_audio_encoder_uses_checkpoint_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLAP loader passes the selected checkpoint to both model components.

    :param monkeypatch: Fixture replacing transformers checkpoint loading.
    """
    checkpoints: list[str] = []

    class Model:
        def to(self, device: str) -> Model:
            del device
            return self

        def eval(self) -> Model:
            return self

    class Loader:
        @staticmethod
        def from_pretrained(checkpoint: str) -> object:
            checkpoints.append(checkpoint)
            return Model()

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        type("Transformers", (), {"ClapModel": Loader, "ClapProcessor": Loader}),
    )
    load_clap_audio_encoder("custom/clap", "cpu")
    assert checkpoints == ["custom/clap", "custom/clap"]


def test_resolve_same_checkpoint_dir_with_full_r2_path_uses_distinct_cache_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2 mirrors sharing a basename retain distinct local cache directories.

    :param monkeypatch: Fixture replacing credential loading and download.
    """
    downloads: list[tuple[str, Path]] = []
    monkeypatch.setattr("synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda: None)
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_dir_no_overwrite",
        lambda uri, destination: downloads.append((uri, destination)),
    )

    dir_a = _resolve_same_checkpoint_dir("r2://bucket/team-a/same-s")
    dir_b = _resolve_same_checkpoint_dir("r2://bucket/team-b/same-s/")

    assert dir_a != dir_b
    assert downloads == [
        ("r2://bucket/team-a/same-s", dir_a),
        ("r2://bucket/team-b/same-s/", dir_b),
    ]


def test_resolve_same_checkpoint_dir_with_existing_local_path_returns_it(
    tmp_path: Path,
) -> None:
    """An existing local checkpoint directory needs no download.

    :param tmp_path: Existing local checkpoint directory.
    """
    assert _resolve_same_checkpoint_dir(str(tmp_path)) == tmp_path


def test_load_same_audio_encoder_without_extra_names_install_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The direct SAME loader reports the optional-extra install command.

    :param monkeypatch: Fixture hiding the optional dependency.
    :param tmp_path: Local checkpoint placeholder.
    """
    monkeypatch.setitem(sys.modules, "stable_audio_tools", None)
    monkeypatch.setitem(sys.modules, "stable_audio_tools.models", None)
    monkeypatch.setitem(sys.modules, "stable_audio_tools.models.factory", None)

    with pytest.raises(ImportError, match="uv sync --extra same"):
        load_same_audio_encoder(str(tmp_path), device="cpu")


def test_configure_lance_logging_without_debug_defaults_to_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default native logging preserves warnings when no override exists.

    :param monkeypatch: Fixture clearing ambient Lance logging.
    """
    monkeypatch.delenv("LANCE_LOG", raising=False)

    _configure_lance_logging(debug=False)

    assert os.environ["LANCE_LOG"] == "warn"


def test_configure_lance_logging_with_debug_enables_native_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Debug mode overrides ambient native Lance logging.

    :param monkeypatch: Fixture setting ambient Lance logging.
    """
    monkeypatch.setenv("LANCE_LOG", "warn")
    _configure_lance_logging(debug=True)
    assert os.environ["LANCE_LOG"] == "debug"


def test_add_embeddings_with_resume_cache_completes_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Endpoint resume settings reach the writer and clean up after commit.

    :param tmp_path: Scratch directory for the dataset and cache.
    :param monkeypatch: Fixture installing dependency-free specs.
    """
    uri = tmp_path / "resume-endpoint.lance"
    resume_cache = tmp_path / "resume-endpoint.cache"
    write_minimal_lance_shard(uri, build_lance_smoke_spec())
    _install_fake_specs(monkeypatch, ("clap", "m2l"))

    add_embeddings(
        AddEmbeddingsConfig(
            lance_uri=str(uri), resume_cache=resume_cache, build_index=False
        )
    )

    assert not resume_cache.exists()
    assert {CLAP_FIELD, M2L_FIELD} <= set(lance.dataset(str(uri)).schema.names)


def test_add_embeddings_threads_device_and_debug_to_loaders_and_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Endpoint device and debug settings reach both loaders and progress logging.

    :param tmp_path: Scratch directory for the dataset.
    :param monkeypatch: Fixture installing recording registry specs.
    """
    uri = tmp_path / "device-debug.lance"
    write_minimal_lance_shard(uri, build_lance_smoke_spec())
    selected: list[tuple[str, str | None]] = []

    for name in ("clap", "m2l"):
        spec = _fake_spec(name)

        def load(
            checkpoint: str, device: str | None, *, registry_name: str = name
        ) -> Callable[..., np.ndarray]:
            del checkpoint
            selected.append((registry_name, device))
            return _encoder_for(registry_name)

        monkeypatch.setitem(EMBEDDING_REGISTRY, name, replace(spec, load_encoder=load))

    with capture_logs() as logs:
        add_embeddings(
            AddEmbeddingsConfig(
                lance_uri=str(uri), device="mps", debug=True, build_index=False
            )
        )

    assert selected == [("clap", "mps"), ("m2l", "mps")]
    assert any(entry["event"] == "embedding_progress" for entry in logs)


def test_add_embeddings_uses_sample_rate_from_dataset_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dataset sample rate is forwarded into the selected encoder closure.

    :param tmp_path: Scratch directory for the dataset.
    :param monkeypatch: Fixture installing a sample-rate recording CLAP spec.
    """
    dataset_spec = build_lance_smoke_spec()
    uri = tmp_path / "sample-rate.lance"
    write_minimal_lance_shard(uri, dataset_spec)
    seen: list[int] = []
    spec = _fake_spec("clap")

    def encode(audio: np.ndarray, sample_rate: int, encoder: Encoder) -> pa.Array:
        seen.append(sample_rate)
        return spec.encode_column(audio, sample_rate, encoder)

    monkeypatch.setitem(EMBEDDING_REGISTRY, "clap", replace(spec, encode_column=encode))
    add_embeddings(
        AddEmbeddingsConfig(
            lance_uri=str(uri), embeddings=("clap",), build_index=False
        )
    )

    assert seen
    assert set(seen) == {int(dataset_spec.render.sample_rate)}


def test_write_columns_with_mono_same_source_round_trips(tmp_path: Path) -> None:
    """A mono dataset reaches SAME through writer-owned stereo preparation.

    :param tmp_path: Scratch directory for the dataset.
    """
    uri = tmp_path / "mono-same.lance"
    _audio_dataset(uri, rows=2, channels=1)

    _write_columns(
        lance.dataset(str(uri)),
        [_fake_spec("same_s")],
        SAME_SAMPLE_RATE,
        AddEmbeddingsConfig(
            lance_uri=str(uri), embeddings=("same_s",), build_index=False
        ),
    )

    values = (
        lance.dataset(str(uri))
        .to_table(columns=[SAME_S_FIELD])
        .combine_chunks()
        .column(SAME_S_FIELD)
        .chunk(0)
        .to_numpy_ndarray()
    )
    assert values.shape == (2, SAME_EMBEDDING_DIM, _FIXTURE_FRAMES)


def test_add_embeddings_open_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dataset-open failures remain available for the Hydra shell to map.

    :param monkeypatch: Fixture breaking dataset opening.
    """
    def boom(uri: str) -> object:
        raise RuntimeError(f"missing R2 credentials for {uri}")

    monkeypatch.setattr("synth_setter.pipeline.data.add_embeddings._open_lance_dataset", boom)

    with pytest.raises(RuntimeError, match="missing R2 credentials"):
        add_embeddings(AddEmbeddingsConfig(lance_uri="s3://bucket/missing.lance"))


def test_add_embeddings_loader_failure_leaves_dataset_unaugmented(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A co-resident loader failure commits none of the group's columns.

    :param tmp_path: Scratch directory for the dataset.
    :param monkeypatch: Fixture installing a failing registry loader.
    """
    uri = tmp_path / "loader-failure.lance"
    write_minimal_lance_shard(uri, build_lance_smoke_spec())
    _install_fake_specs(monkeypatch, ("clap", "m2l"))
    spec = EMBEDDING_REGISTRY["m2l"]

    def boom(checkpoint: str, device: str | None) -> Encoder:
        del checkpoint, device
        raise RuntimeError("encoder load blew up")

    monkeypatch.setitem(EMBEDDING_REGISTRY, "m2l", replace(spec, load_encoder=boom))

    with pytest.raises(RuntimeError, match="encoder load blew up"):
        add_embeddings(AddEmbeddingsConfig(lance_uri=str(uri), build_index=False))

    assert {CLAP_FIELD, M2L_FIELD}.isdisjoint(lance.dataset(str(uri)).schema.names)


def test_module_import_defers_lance_initialization_until_cli_configures_logging() -> None:
    """Importing the endpoint leaves native Lance initialization deferred."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import synth_setter.pipeline.data.add_embeddings; "
            "sys.exit('lance imported early' if 'lance' in sys.modules else 0)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_add_embeddings_main_when_open_fails_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Hydra shell maps endpoint failures to exit status one.

    :param tmp_path: Scratch directory for Hydra output.
    :param monkeypatch: Fixture breaking dataset opening and replacing argv.
    """
    from synth_setter.pipeline.data.add_embeddings import main

    def boom(uri: str) -> object:
        raise RuntimeError(f"missing R2 credentials for {uri}")

    monkeypatch.setattr("synth_setter.pipeline.data.add_embeddings._open_lance_dataset", boom)
    monkeypatch.setenv("PROJECT_ROOT", str(operator_workspace()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synth-setter-add-embeddings",
            "lance_uri=s3://bucket/missing.lance",
            f"paths.log_dir={tmp_path}",
            f"hydra.run.dir={tmp_path / 'run'}",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1


@pytest.mark.slow
@pytest.mark.parametrize(
    ("selection", "expected", "expected_checkpoints"),
    [
        (None, {CLAP_FIELD, M2L_FIELD}, []),
        ("[same_s]", {SAME_S_FIELD}, [("same_s", DEFAULT_SAME_S_CHECKPOINT)]),
        ("[same_l]", {SAME_L_FIELD}, [("same_l", DEFAULT_SAME_L_CHECKPOINT)]),
        (
            "[same_s,same_l]",
            {SAME_S_FIELD, SAME_L_FIELD},
            [
                ("same_s", DEFAULT_SAME_S_CHECKPOINT),
                ("same_l", DEFAULT_SAME_L_CHECKPOINT),
            ],
        ),
    ],
)
def test_add_embeddings_main_with_registry_mode_writes_exact_columns(
    selection: str | None,
    expected: set[str],
    expected_checkpoints: list[tuple[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Hydra shell dispatches default and SAME-only registry selections.

    :param selection: Hydra embedding-list override, or ``None`` for defaults.
    :param expected: Exact embedding columns expected to land.
    :param expected_checkpoints: SAME loaders and defaults expected in order.
    :param tmp_path: Scratch directory for the shard and Hydra output.
    :param monkeypatch: Fixture installing fake registry specs and argv.
    """
    from synth_setter.pipeline.data.add_embeddings import main

    uri = tmp_path / "registry-shell.lance"
    write_minimal_lance_shard(uri, build_lance_smoke_spec())
    _install_fake_specs(monkeypatch, ("clap", "m2l"))
    seen_checkpoints: list[tuple[str, str]] = []
    for name in ("same_s", "same_l"):
        spec = _fake_spec(name)

        def load(
            checkpoint: str, device: str | None, *, registry_name: str = name
        ) -> Callable[..., np.ndarray]:
            del device
            seen_checkpoints.append((registry_name, checkpoint))
            return _encoder_for(registry_name)

        monkeypatch.setitem(EMBEDDING_REGISTRY, name, replace(spec, load_encoder=load))
    argv = [
        "synth-setter-add-embeddings",
        f"lance_uri={uri}",
        "build_index=false",
        f"paths.log_dir={tmp_path}",
        f"hydra.run.dir={tmp_path / 'run'}",
    ]
    if selection is not None:
        argv.insert(2, f"embeddings={selection}")
    monkeypatch.setenv("PROJECT_ROOT", str(operator_workspace()))
    monkeypatch.setattr(sys, "argv", argv)

    main()

    embedding_columns = {CLAP_FIELD, M2L_FIELD, SAME_S_FIELD, SAME_L_FIELD}
    dataset = lance.dataset(str(uri))
    assert set(dataset.schema.names) & embedding_columns == expected
    assert seen_checkpoints == expected_checkpoints
    for name, fill in ((SAME_S_FIELD, 0.25), (SAME_L_FIELD, 0.75)):
        if name in expected:
            values = (
                dataset.to_table(columns=[name])
                .combine_chunks()
                .column(name)
                .chunk(0)
                .to_numpy_ndarray()
            )
            assert float(values.flat[0]) == fill


@pytest.mark.slow
def test_add_embeddings_main_with_registry_selection_writes_requested_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real Hydra shell accepts the registry selection syntax end to end.

    :param tmp_path: Scratch directory for the shard and Hydra run.
    :param monkeypatch: Fixture installing dependency-free specs and argv.
    """
    from synth_setter.pipeline.data.add_embeddings import main

    uri = tmp_path / "shell.lance"
    write_minimal_lance_shard(uri, build_lance_smoke_spec())
    _install_fake_specs(monkeypatch, ("clap",))
    checkpoints: list[str] = []
    same_spec = _fake_spec("same_s")

    def load_same(checkpoint: str, device: str | None) -> Callable[..., np.ndarray]:
        del device
        checkpoints.append(checkpoint)
        return _fake_same(0.25)

    monkeypatch.setitem(
        EMBEDDING_REGISTRY,
        "same_s",
        replace(same_spec, load_encoder=load_same),
    )
    monkeypatch.setenv("PROJECT_ROOT", str(operator_workspace()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synth-setter-add-embeddings",
            f"lance_uri={uri}",
            "embeddings=[clap,same_s]",
            "checkpoints.same_s=custom/same-s",
            "build_index=false",
            f"paths.log_dir={tmp_path}",
            f"hydra.run.dir={tmp_path / 'run'}",
        ],
    )

    main()

    names = set(lance.dataset(str(uri)).schema.names)
    assert CLAP_FIELD in names
    assert SAME_S_FIELD in names
    assert M2L_FIELD not in names
    assert SAME_L_FIELD not in names
    assert checkpoints == ["custom/same-s"]


def test_registry_default_checkpoints_match_public_sources() -> None:
    """Checkpoint-backed registry entries preserve their established defaults."""
    assert EMBEDDING_REGISTRY["clap"].default_checkpoint == DEFAULT_CLAP_CHECKPOINT
    assert EMBEDDING_REGISTRY["same_s"].default_checkpoint == DEFAULT_SAME_S_CHECKPOINT
    assert EMBEDDING_REGISTRY["same_l"].default_checkpoint == DEFAULT_SAME_L_CHECKPOINT
