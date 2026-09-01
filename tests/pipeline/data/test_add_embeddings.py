"""Behavioral tests for the registry-driven embedding augmenter."""

from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
import threading
import weakref
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Literal, cast

import lance
import numpy as np
import pyarrow as pa
import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from pydantic import ValidationError
from structlog.testing import capture_logs

from synth_setter.clap import (
    DEFAULT_CLAP_TRAINING_CHECKPOINT,
    DEFAULT_CLAP_TRAINING_CHECKPOINT_SHA256,
    clap_checkpoint_sha256,
)
from synth_setter.conditioning import SKETCH_STORAGE_FRAMES
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    CLAP_FIELD,
    M2L_FIELD,
    MEANAUDIO_16K_FIELD,
    PARAM_ARRAY_FIELD,
    PUPUJEPA_LARGE_FIELD,
    SAME_L_FIELD,
    SAME_S_FIELD,
    SKETCH_CENTROID_CHILD,
    SKETCH_CENTROID_ROW,
    SKETCH_LOUDNESS_CHILD,
    SKETCH_LOUDNESS_ROW,
    SKETCH_PITCH_BINS,
    SKETCH_PITCH_CHILD,
    SKETCH_PITCH_SLICE,
    SKETCH_STRUCT_FIELD,
    SKETCH_VEC_CHILD,
    T5GEMMA_FIELD,
    dataset_field_shapes,
)
from synth_setter.features.sketch_controls import (
    NUM_SKETCH_CONTROLS,
    extract_sketch_controls_batch,
    sketch_num_frames,
)
from synth_setter.model_cache import checkpoint_tree_sha256
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.data.add_embeddings import (
    CLAP_EMBEDDING_DIM,
    DEFAULT_CLAP_CHECKPOINT,
    DEFAULT_LANCE_BATCH_SIZE,
    EMBEDDING_REGISTRY,
    SAME_LATENT_FRAMES,
    SKETCH_INDEX_SUB_VECTORS,
    SKETCH_VEC_COLUMN,
    EmbeddingSpec,
    Encoder,
    IndexSpec,
    ParamTextEncodeFn,
    _configure_lance_logging,
    _downmix_to_mono,
    _encode_t5gemma_column,
    _load_clap_spec_encoder,
    _load_m2l_spec_encoder,
    _load_same_spec_encoder,
    _load_t5gemma_spec_encoder,
    _matching_index_exists,
    _missing_embedding_specs,
    _prepare_resume_cache,
    _resolve_artifact_identity,
    _resolve_clap_checkpoint,
    _resume_source_identity,
    _versioned_artifact_identity,
    _write_columns,
    add_embeddings,
    build_index,
    load_clap_audio_encoder,
    load_m2l_audio_encoder,
    load_same_audio_encoder,
    same_encoder_input,
    same_l_num_latent_frames,
    same_s_num_latent_frames,
)
from synth_setter.pipeline.data.matpac_plus import (
    MATPAC_PLUS_FRONTEND,
    matpac_plus_num_latent_frames,
)
from synth_setter.pipeline.data.meanaudio import (
    MEANAUDIO_EMBEDDING_DIM,
    MEANAUDIO_INDEX_SUB_VECTORS,
    meanaudio_num_latent_frames,
)
from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig
from synth_setter.same import (
    DEFAULT_SAME_L_CHECKPOINT,
    DEFAULT_SAME_S_CHECKPOINT,
    SAME_DOWNSAMPLING_RATIO,
    SAME_EMBEDDING_DIM,
    SAME_SAMPLE_RATE,
    resolve_same_checkpoint,
)
from synth_setter.sketch import pool_sketch_controls
from synth_setter.workspace import operator_workspace
from tests.helpers.finalize_shards import build_lance_smoke_spec, write_minimal_lance_shard
from tests.helpers.lance_fixtures import write_lance_shard
from tests.helpers.run_if import RunIf

_SAMPLE_RATE = 44100
_FIXTURE_SAMPLES = 16
_FIXTURE_FRAMES = 2
_M2L_TIME = 3
_LANCE_URI = "r2://bucket/run/train.lance"
_CLAP_MIRROR_FILES = (
    "config.json",
    "preprocessor_config.json",
    "pytorch_model.bin",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
)


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


def _temporal_m2l(audio: np.ndarray) -> np.ndarray:
    """Encode audio as m2l latents whose time-axis mean is observable.

    :param audio: ``(B, C, T)`` audio batch.
    :returns: Deterministic ``(B, 16, 3)`` latents.
    """
    vectors = np.ascontiguousarray(audio.reshape(len(audio), -1)[:, :16], dtype=np.float32)
    offsets = np.arange(_M2L_TIME, dtype=np.float32)
    return vectors[:, :, None] + offsets[None, None, :]


def _fake_same(
    fill: float,
    frame_count: Callable[[int, int], int] = same_s_num_latent_frames,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a deterministic SAME encoder with the selected model's frame contract.

    :param fill: Constant value used for every latent cell.
    :param frame_count: SAME model's latent-frame calculation.
    :returns: Encoder over prepared stereo audio.
    """

    def encode(stereo: np.ndarray) -> np.ndarray:
        frames = frame_count(stereo.shape[2], SAME_SAMPLE_RATE)
        return np.full((len(stereo), SAME_EMBEDDING_DIM, frames), fill, dtype=np.float32)

    return encode


def _fake_sketch(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Encode audio as deterministic sketch-control matrices.

    :param audio: ``(B, C, T)`` audio batch.
    :param sample_rate: Sample rate deciding the frame grid.
    :returns: Deterministic in-bounds ``(B, NUM_SKETCH_CONTROLS, F)`` controls.
    """
    frames = sketch_num_frames(audio.shape[-1], sample_rate)
    # Row steps dominate the clip term, keeping every cell inside [0, 1) while
    # each (clip, row) pair stays distinct.
    rows = np.arange(NUM_SKETCH_CONTROLS, dtype=np.float32) / NUM_SKETCH_CONTROLS
    per_clip = np.clip(audio.mean(axis=(1, 2), dtype=np.float32), -0.5, 0.5) / (
        2 * NUM_SKETCH_CONTROLS
    )
    cells = per_clip[:, None, None] + rows[None, :, None]
    return np.ascontiguousarray(np.repeat(cells, frames, axis=2))


def _fake_meanaudio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Encode audio as deterministic MeanAudio-shaped sequences.

    :param audio: ``(B, C, T)`` audio batch.
    :param sample_rate: Source sample rate in Hz.
    :returns: Deterministic ``(B, 20, F)`` posterior means.
    """
    frames = meanaudio_num_latent_frames(audio.shape[-1], sample_rate)
    fill = audio.astype(np.float32).mean(axis=(1, 2))
    return np.broadcast_to(
        fill[:, None, None], (len(audio), MEANAUDIO_EMBEDDING_DIM, frames)
    ).copy()


def _fake_matpac_plus(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Encode audio as deterministic MATPAC++-shaped sequences.

    :param audio: ``(B, C, T)`` audio batch.
    :param sample_rate: Source sample rate in Hz.
    :returns: Deterministic MATPAC++-shaped embeddings.
    """
    frames = matpac_plus_num_latent_frames(audio.shape[-1], sample_rate)
    fill = audio.astype(np.float32).mean(axis=(1, 2))
    return np.broadcast_to(
        fill[:, None, None], (len(audio), MATPAC_PLUS_FRONTEND.embedding_dim, frames)
    ).copy()


def _encoder_for(name: str) -> Callable[..., np.ndarray]:
    """Return the fake encoder matching a registry key.

    :param name: Embedding registry key.
    :returns: Matching fake encoder.
    :raises ValueError: No fake is registered for ``name``.
    """
    if name == "m2l":
        return _fake_m2l
    if name == "clap":
        return _fake_clap
    if name == "sketch":
        return _fake_sketch
    if name == "same_s":
        return _fake_same(0.25)
    if name == "same_l":
        return _fake_same(0.75, same_l_num_latent_frames)
    if name == "matpac_plus":
        return _fake_matpac_plus
    if name == "meanaudio_16k":
        return _fake_meanaudio
    raise ValueError(f"no fake encoder for {name!r}")


def _fake_spec(name: str, events: list[str] | None = None) -> EmbeddingSpec:
    """Copy a production spec with a dependency-free loader.

    :param name: Registry key to copy.
    :param events: Optional list receiving loader events.
    :returns: Spec using a fake encoder and no optional-extra gate.
    """

    def load(checkpoint: str, config: AddEmbeddingsConfig) -> Callable[..., np.ndarray]:
        del checkpoint, config
        if events is not None:
            events.append(f"load:{name}")
        return _encoder_for(name)

    return replace(
        EMBEDDING_REGISTRY[name],
        load_encoder=load,
        resolve_artifact_identity=lambda checkpoint: f"fake:{name}:{checkpoint}:v1",
    )


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


_REAL_ADD_COLUMNS = lance.LanceDataset.add_columns


def _run_udf_in_process(
    dataset: lance.LanceDataset,
    udf: Callable[[pa.RecordBatch], pa.RecordBatch],
    *,
    read_columns: list[str],
    batch_size: int,
) -> None:
    """Run a Lance batch UDF synchronously, then commit its outputs.

    Committing keeps the writer's post-commit column check satisfied while the UDF invocations stay
    observable in-process.

    :param dataset: Local dataset supplying batches.
    :param udf: Batch transform under test.
    :param read_columns: Source columns supplied to the transform.
    :param batch_size: Maximum rows per invocation.
    """
    outputs = [
        udf(batch)
        for batch in dataset.to_batches(columns=read_columns, batch_size=batch_size)
    ]
    reader = pa.RecordBatchReader.from_batches(outputs[0].schema, outputs)
    _REAL_ADD_COLUMNS(dataset, reader, batch_size=batch_size)


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
    assert set(EMBEDDING_REGISTRY) == {
        "clap",
        "m2l",
        "param_shift",
        "pupujepa_large",
        "pupujepa_tiny",
        "same_l",
        "same_s",
        "sketch",
        "ssondo",
        "t5gemma",
        "matpac_plus",
        "meanaudio_16k",
    }
    assert EMBEDDING_REGISTRY["sketch"].index == IndexSpec(
        pool="none",
        num_sub_vectors=SKETCH_INDEX_SUB_VECTORS,
        vector_column=SKETCH_VEC_COLUMN,
        vector_dim=NUM_SKETCH_CONTROLS,
    )
    assert EMBEDDING_REGISTRY["sketch"].co_resident is True
    assert EMBEDDING_REGISTRY["clap"].index == IndexSpec(
        pool="none", vector_dim=CLAP_EMBEDDING_DIM
    )
    assert EMBEDDING_REGISTRY["m2l"].index == IndexSpec(
        pool="mean", vector_column=f"{M2L_FIELD}_vec"
    )
    assert EMBEDDING_REGISTRY["same_s"].index == IndexSpec(
        pool="mean", vector_column=f"{SAME_S_FIELD}_vec"
    )
    assert EMBEDDING_REGISTRY["same_l"].index == IndexSpec(
        pool="mean", vector_column=f"{SAME_L_FIELD}_vec"
    )
    assert EMBEDDING_REGISTRY["ssondo"].index == IndexSpec(pool="none", vector_dim=960)
    assert EMBEDDING_REGISTRY["t5gemma"].index is None
    assert EMBEDDING_REGISTRY["t5gemma"].input_fields == (PARAM_ARRAY_FIELD,)
    assert EMBEDDING_REGISTRY["clap"].co_resident is True
    assert EMBEDDING_REGISTRY["m2l"].co_resident is True
    assert EMBEDDING_REGISTRY["pupujepa_tiny"].co_resident is False
    assert EMBEDDING_REGISTRY["pupujepa_large"].index == IndexSpec(
        pool="mean", vector_column=f"{PUPUJEPA_LARGE_FIELD}_vec", vector_dim=8192
    )
    assert EMBEDDING_REGISTRY["pupujepa_large"].co_resident is False
    assert EMBEDDING_REGISTRY["same_s"].co_resident is False
    assert EMBEDDING_REGISTRY["same_l"].co_resident is False
    assert EMBEDDING_REGISTRY["ssondo"].co_resident is True
    assert EMBEDDING_REGISTRY["t5gemma"].co_resident is False
    assert EMBEDDING_REGISTRY["matpac_plus"].co_resident is False
    assert EMBEDDING_REGISTRY["meanaudio_16k"].index == IndexSpec(
        pool="mean",
        num_sub_vectors=MEANAUDIO_INDEX_SUB_VECTORS,
        vector_column=f"{MEANAUDIO_16K_FIELD}_vec",
        vector_dim=MEANAUDIO_EMBEDDING_DIM,
    )
    assert EMBEDDING_REGISTRY["meanaudio_16k"].co_resident is False


@pytest.mark.parametrize(
    ("name", "variant"),
    [("pupujepa_tiny", "tiny"), ("pupujepa_large", "large")],
)
def test_pupujepa_registry_loader_threads_variant(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    variant: str,
) -> None:
    """Each PupuJEPA registry adapter selects its released teacher size.

    :param monkeypatch: Fixture replacing heavyweight teacher loading.
    :param name: Registry profile under test.
    :param variant: Expected teacher size.
    """
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_embeddings.load_pupujepa_audio_encoder",
        lambda checkpoint, *, device, variant: (
            calls.append((checkpoint, device, variant)) or (lambda audio, rate: audio)
        ),
    )
    spec = EMBEDDING_REGISTRY[name]

    spec.load_encoder(
        "custom/pupujepa",
        AddEmbeddingsConfig(lance_uri="x.lance", device="cpu"),
    )

    assert calls == [("custom/pupujepa", "cpu", variant)]


@pytest.mark.parametrize(
    ("name", "variant"),
    [("pupujepa_tiny", "tiny"), ("pupujepa_large", "large")],
)
def test_pupujepa_registry_artifact_identity_threads_variant(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    variant: str,
) -> None:
    """Artifact identities hash only the selected teacher size.

    :param monkeypatch: Fixture replacing checkpoint hashing.
    :param name: Registry profile under test.
    :param variant: Expected teacher size.
    """
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_embeddings.pupujepa_artifact_digest",
        lambda checkpoint, selected: calls.append((checkpoint, selected)) or "digest",
    )
    spec = EMBEDDING_REGISTRY[name]

    identity = spec.resolve_artifact_identity("custom/pupujepa")

    assert calls == [("custom/pupujepa", variant)]
    assert "digest" in identity


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
        assert cfg.num_sub_vectors is None
        assert cfg.metric == "cosine"
        assert cfg.resume_cache is None
        assert cfg.debug is False
        assert AddEmbeddingsConfig.from_hydra_cfg(cfg) == AddEmbeddingsConfig(lance_uri=_LANCE_URI)
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
            lance_uri=_LANCE_URI,
            embeddings=["clap", "clap"],  # type: ignore[arg-type]
        )


def test_add_embeddings_config_with_empty_embedding_selection_raises() -> None:
    """An empty registry selection is rejected instead of becoming a silent no-op."""
    with pytest.raises(ValueError, match="embeddings must select at least one registry key"):
        AddEmbeddingsConfig(lance_uri=_LANCE_URI, embeddings=())


def test_add_embeddings_config_with_m2l_checkpoint_override_raises() -> None:
    """M2L cannot label package-owned weights as a caller-selected checkpoint."""
    with pytest.raises(ValidationError, match="does not support checkpoint overrides"):
        AddEmbeddingsConfig(lance_uri="dataset", checkpoints={"m2l": "replacement"})


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
        lance_uri=_LANCE_URI,
        resume_cache="cache/embed.cache",  # type: ignore[arg-type]
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


def test_add_embeddings_config_with_m2l_only_allows_non_clap_subvector_count() -> None:
    """Runtime-width companions defer PQ divisibility checks to index construction."""
    config = AddEmbeddingsConfig(
        lance_uri=_LANCE_URI,
        embeddings=("m2l",),
        num_sub_vectors=15,
    )

    assert config.num_sub_vectors == 15


@pytest.mark.parametrize("name", ["clap", "m2l", "same_s", "same_l"])
def test_embedding_spec_encode_column_for_valid_encoder_builds_arrow_array(name: str) -> None:
    """Every registry encoder closure preserves its exact shape and values.

    :param name: Registry key under test.
    """
    audio = np.random.default_rng(0).random((3, 2, _FIXTURE_SAMPLES)).astype(np.float16)
    spec = EMBEDDING_REGISTRY[name]
    encoder = _encoder_for(name)

    array = spec.encode_column({AUDIO_FIELD: audio}, _SAMPLE_RATE, encoder)

    assert len(array) == 3
    if name == "clap":
        assert array.type == pa.list_(pa.float32(), CLAP_EMBEDDING_DIM)
        np.testing.assert_allclose(
            np.asarray(array.to_pylist()), _fake_clap(_downmix_to_mono(audio), _SAMPLE_RATE)
        )
    else:
        assert isinstance(array.type, pa.FixedShapeTensorType)
        assert array.type.value_type == pa.float32()
        values = pa.chunked_array([array]).combine_chunks().to_numpy_ndarray()
        expected = (
            encoder(audio) if name == "m2l" else encoder(same_encoder_input(audio, _SAMPLE_RATE))
        )
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

    with pytest.raises(
        ValueError, match=f"{EMBEDDING_REGISTRY[name].column} embeddings contain non-finite values"
    ):
        EMBEDDING_REGISTRY[name].encode_column({AUDIO_FIELD: audio}, _SAMPLE_RATE, poisoned)


def test_same_embedding_spec_prepares_stereo_before_encoder_call() -> None:
    """The SAME closure owns mono duplication and float32 conversion."""
    mono = np.random.default_rng(4).random((2, 1, _FIXTURE_SAMPLES)).astype(np.float16)
    seen: list[np.ndarray] = []

    def recording(stereo: np.ndarray) -> np.ndarray:
        seen.append(stereo)
        return _fake_same(1.0)(stereo)

    EMBEDDING_REGISTRY["same_s"].encode_column({AUDIO_FIELD: mono}, SAME_SAMPLE_RATE, recording)

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
        (
            "same_l",
            lambda stereo: np.zeros((len(stereo), 256, 2), dtype=np.float32),
            r"expected \(2, 256, 1\)",
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
        EMBEDDING_REGISTRY[name].encode_column({AUDIO_FIELD: audio}, _SAMPLE_RATE, encoder)


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
        assert f"{CLAP_FIELD}_vec" not in dataset.schema.names
        values = np.asarray(column.to_pylist(), dtype=np.float32)
        expected = _fake_clap(_downmix_to_mono(audio), _SAMPLE_RATE)
    else:
        values = column.to_numpy_ndarray()
        encoder = _encoder_for(name)
        expected = (
            encoder(audio) if name == "m2l" else encoder(same_encoder_input(audio, _SAMPLE_RATE))
        )
    np.testing.assert_allclose(values, expected)


def test_write_columns_with_mean_pool_writes_float32_companion_from_sequence(
    tmp_path: Path,
) -> None:
    """A pooled spec stores its sequence and finite time-axis mean together.

    :param tmp_path: Scratch directory for the dataset.
    """
    uri = tmp_path / "mean-pooled.lance"
    audio = _audio_dataset(uri, rows=4)
    base = _fake_spec("m2l")

    def load(checkpoint: str, config: AddEmbeddingsConfig) -> Callable[..., np.ndarray]:
        del checkpoint, config
        return _temporal_m2l

    _write_columns(
        lance.dataset(str(uri)),
        [replace(base, load_encoder=load)],
        _SAMPLE_RATE,
        AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("m2l",), build_index=False),
    )

    dataset = lance.dataset(str(uri))
    vector_column = f"{M2L_FIELD}_vec"
    vector_type = dataset.schema.field(vector_column).type
    assert vector_type == pa.list_(pa.float32(), 16)
    vectors = np.asarray(
        dataset.to_table(columns=[vector_column]).column(vector_column).to_pylist(),
        dtype=np.float32,
    )
    assert vectors.shape == (4, 16)
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(vectors, _temporal_m2l(audio).mean(axis=-1))


def test_write_columns_with_attention_pool_raises_not_implemented(tmp_path: Path) -> None:
    """An attention policy fails explicitly until an implementation is selected.

    :param tmp_path: Scratch directory for the dataset.
    """
    uri = tmp_path / "attention.lance"
    _audio_dataset(uri, rows=2)
    spec = replace(
        _fake_spec("m2l"),
        index=IndexSpec(pool="attention", vector_column=f"{M2L_FIELD}_vec"),
    )

    with pytest.raises(NotImplementedError, match="attention pooling is not implemented"):
        _write_columns(
            lance.dataset(str(uri)),
            [spec],
            _SAMPLE_RATE,
            AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("m2l",), build_index=False),
        )


def test_write_columns_for_co_resident_specs_shares_audio_object(tmp_path: Path) -> None:
    """One UDF decode supplies the same audio object to all co-resident encoders.

    :param tmp_path: Scratch directory for the dataset.
    """
    uri = tmp_path / "shared-decode.lance"
    _audio_dataset(uri, rows=4)
    seen: dict[str, list[int]] = {"clap": [], "m2l": []}

    def recording_spec(name: str) -> EmbeddingSpec:
        original = _fake_spec(name)

        def encode(
            sources: Mapping[str, np.ndarray], sample_rate: int, encoder: Encoder
        ) -> pa.Array:
            seen[name].append(id(sources[AUDIO_FIELD]))
            return original.encode_column(sources, sample_rate, encoder)

        return replace(original, encode_column=encode)

    _write_columns(
        lance.dataset(str(uri)),
        [recording_spec("clap"), recording_spec("m2l")],
        _SAMPLE_RATE,
        AddEmbeddingsConfig(lance_uri=str(uri), build_index=False),
    )

    assert seen["clap"] == seen["m2l"]
    assert lance.dataset(str(uri)).schema.names[-3:] == [
        CLAP_FIELD,
        M2L_FIELD,
        f"{M2L_FIELD}_vec",
    ]


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
    config = AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("same_s",), build_index=False)
    _write_columns(lance.dataset(str(uri)), [initial], _SAMPLE_RATE, config)
    loads: list[str] = []

    with pytest.raises(
        ValueError,
        match=r"dataset already has embedding column\(s\): \['same_s', 'same_s_vec'\]",
    ):
        _write_columns(
            lance.dataset(str(uri)), [_fake_spec("same_s", loads)], _SAMPLE_RATE, config
        )

    assert loads == []


def test_write_columns_with_existing_companion_raises_before_encoder_load(
    tmp_path: Path,
) -> None:
    """An existing pooled-vector target fails before checkpoint loading.

    :param tmp_path: Scratch directory for the dataset.
    """
    uri = tmp_path / "existing-vector.lance"
    rows = 2
    rng = np.random.default_rng(7)
    write_lance_shard(
        uri,
        {
            AUDIO_FIELD: rng.random((rows, 2, _FIXTURE_SAMPLES)).astype(np.float16),
            PARAM_ARRAY_FIELD: rng.random((rows, 3)).astype(np.float32),
            f"{M2L_FIELD}_vec": np.zeros((rows, 8), dtype=np.float32),
        },
    )
    loads: list[str] = []

    with pytest.raises(
        ValueError, match=rf"dataset already has embedding column\(s\): \['{M2L_FIELD}_vec'\]"
    ):
        _write_columns(
            lance.dataset(str(uri)),
            [_fake_spec("m2l", loads)],
            _SAMPLE_RATE,
            AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("m2l",), build_index=False),
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


def test_checkpoint_tree_identity_ignores_huggingface_download_bookkeeping(
    tmp_path: Path,
) -> None:
    """Download timestamps do not change checkpoint content identity.

    :param tmp_path: Scratch checkpoint tree.
    """
    model = tmp_path / "model.bin"
    metadata = tmp_path / ".cache" / "huggingface" / "download" / "model.metadata"
    metadata.parent.mkdir(parents=True)
    model.write_bytes(b"weights")
    metadata.write_text("first timestamp")
    first = checkpoint_tree_sha256(tmp_path)
    metadata.write_text("different timestamp")

    assert checkpoint_tree_sha256(tmp_path) == first


def test_versioned_artifact_identity_uses_explicit_policy_version() -> None:
    """Unrelated repository revisions do not alter artifact identity."""
    assert (
        _versioned_artifact_identity("matpac_plus", "checkpoint:sha256:abc")
        == "matpac_plus:policy-v1:checkpoint:sha256:abc"
    )


def test_sketch_artifact_identity_tracks_storage_frame_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sketch metadata identifies the active pooled storage grid.

    :param monkeypatch: Replaces the storage frame count for the identity probe.
    """
    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_embeddings.SKETCH_STORAGE_FRAMES",
        17,
    )
    spec = EMBEDDING_REGISTRY["sketch"]

    identity = spec.resolve_artifact_identity(spec.default_checkpoint)

    assert identity.endswith(";storage:avgmax17")


def test_resume_source_identity_changes_with_input_contract(tmp_path: Path) -> None:
    """Output-affecting source settings cannot share cached UDF batches.

    :param tmp_path: Scratch directory for the Lance source.
    """
    uri = tmp_path / "source.lance"
    _audio_dataset(uri, rows=4)
    dataset = lance.dataset(uri)
    baseline = _resume_source_identity(
        dataset,
        sample_rate=_SAMPLE_RATE,
        batch_size=2,
        input_fields=[AUDIO_FIELD],
    )

    assert baseline != _resume_source_identity(
        dataset,
        sample_rate=_SAMPLE_RATE // 2,
        batch_size=2,
        input_fields=[AUDIO_FIELD],
    )
    assert baseline != _resume_source_identity(
        dataset,
        sample_rate=_SAMPLE_RATE,
        batch_size=1,
        input_fields=[AUDIO_FIELD],
    )
    assert baseline != _resume_source_identity(
        dataset,
        sample_rate=_SAMPLE_RATE,
        batch_size=2,
        input_fields=[PARAM_ARRAY_FIELD],
    )


def test_prepare_resume_cache_without_identity_discards_legacy_batches(
    tmp_path: Path,
) -> None:
    """An unverifiable legacy cache restarts instead of poisoning output metadata.

    :param tmp_path: Scratch directory for cache files.
    """
    resume_cache = tmp_path / "resume.cache"
    resume_cache.write_bytes(b"legacy cached batches")

    with capture_logs() as logs:
        _prepare_resume_cache(resume_cache, {"clap": "artifact-a"}, "dataset-a:v1")

    assert not resume_cache.exists()
    assert resume_cache.with_name("resume.cache.identity").is_file()
    warning = next(entry for entry in logs if entry["event"] == "resume_cache_identity_missing")
    assert warning["action"] == "discard"


def test_prepare_resume_cache_with_different_artifact_rejects_stale_batches(
    tmp_path: Path,
) -> None:
    """A cache created for one artifact cannot resume another artifact.

    :param tmp_path: Scratch directory for cache files.
    """
    resume_cache = tmp_path / "resume.cache"
    _prepare_resume_cache(resume_cache, {"clap": "artifact-a"}, "dataset-a:v1")
    resume_cache.write_bytes(b"cached Lance batches")

    with pytest.raises(ValueError, match="resume identity .*does not match"):
        _prepare_resume_cache(resume_cache, {"clap": "artifact-b"}, "dataset-a:v1")


def test_add_embeddings_with_recreated_source_rejects_stale_resume_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache cannot resume after its Lance source is deleted and recreated.

    :param tmp_path: Scratch directory for the source and cache.
    :param monkeypatch: Fixture installing dependency-free registry policies.
    """
    uri = tmp_path / "source.lance"
    resume_cache = tmp_path / "resume.cache"
    write_minimal_lance_shard(uri, build_lance_smoke_spec())
    base_spec = _fake_spec("m2l")
    calls = 0

    def load_crashing(checkpoint: str, device: str | None) -> Callable[..., np.ndarray]:
        del checkpoint, device
        encoder = _encoder_for("m2l")

        def encode(*args: object) -> np.ndarray:
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("simulated crash")
            return encoder(*args)

        return encode

    config = AddEmbeddingsConfig(
        lance_uri=str(uri),
        embeddings=("m2l",),
        batch_size=1,
        resume_cache=resume_cache,
        build_index=False,
    )
    monkeypatch.setitem(
        EMBEDDING_REGISTRY, "m2l", replace(base_spec, load_encoder=load_crashing)
    )
    with pytest.raises(OSError, match="simulated crash"):
        add_embeddings(config)
    assert resume_cache.exists()

    shutil.rmtree(uri)
    write_minimal_lance_shard(uri, build_lance_smoke_spec())
    monkeypatch.setitem(EMBEDDING_REGISTRY, "m2l", base_spec)

    with pytest.raises(ValueError, match="resume identity .*does not match"):
        add_embeddings(config)
    assert M2L_FIELD not in lance.dataset(uri).schema.names


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
    assert not resume_cache.with_name("resume.cache.identity").exists()
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

    def encode(
        sources: Mapping[str, np.ndarray], sample_rate: int, encoder: Encoder
    ) -> pa.Array:
        batch_sizes.append(len(sources[AUDIO_FIELD]))
        return spec.encode_column(sources, sample_rate, encoder)

    monkeypatch.setattr(lance.LanceDataset, "add_columns", _run_udf_in_process)
    with capture_logs() as logs:
        _write_columns(
            lance.dataset(str(uri)),
            [replace(spec, encode_column=encode)],
            _SAMPLE_RATE,
            AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("m2l",), build_index=False),
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
            AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("m2l",), batch_size=2, debug=True),
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

    add_embeddings(AddEmbeddingsConfig(lance_uri=str(uri), embeddings=selected, build_index=False))

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

    def load(checkpoint: str, config: AddEmbeddingsConfig) -> Callable[..., np.ndarray]:
        seen.append((checkpoint, config.device))
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
    add_embeddings(AddEmbeddingsConfig(lance_uri=str(uri), embeddings=selected, build_index=False))

    assert len(commits) == 3
    assert events == ["load:clap", "load:m2l", "load:same_s", "load:same_l"]
    schema_names = lance.dataset(str(uri)).schema.names
    assert set(schema_names) >= {CLAP_FIELD, M2L_FIELD, SAME_S_FIELD, SAME_L_FIELD}
    assert schema_names[-7:] == [
        CLAP_FIELD,
        M2L_FIELD,
        f"{M2L_FIELD}_vec",
        SAME_S_FIELD,
        f"{SAME_S_FIELD}_vec",
        SAME_L_FIELD,
        f"{SAME_L_FIELD}_vec",
    ]


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
        return _fake_same(0.75, same_l_num_latent_frames)

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


def test_add_embeddings_after_index_failure_resumes_without_reencoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry skips committed columns and resumes the missing index.

    :param tmp_path: Scratch directory for the Lance dataset.
    :param monkeypatch: Fixture replacing the encoder and index boundary.
    """
    uri = tmp_path / "index-resume.lance"
    write_minimal_lance_shard(uri, build_lance_smoke_spec(), num_rows=256)
    loads: list[str] = []
    monkeypatch.setitem(EMBEDDING_REGISTRY, "m2l", _fake_spec("m2l", loads))
    config = AddEmbeddingsConfig(
        lance_uri=str(uri),
        embeddings=("m2l",),
        num_partitions=1,
        num_sub_vectors=4,
        metric="l2",
    )
    real_build_index = build_index

    def fail_index(*args: object, **kwargs: object) -> bool:
        del args, kwargs
        raise RuntimeError("index unavailable")

    monkeypatch.setattr("synth_setter.pipeline.data.add_embeddings.build_index", fail_index)
    with pytest.raises(RuntimeError, match="index unavailable"):
        add_embeddings(config)

    committed_version = lance.dataset(uri).version
    monkeypatch.setattr("synth_setter.pipeline.data.add_embeddings.build_index", real_build_index)
    add_embeddings(config)

    dataset = lance.dataset(uri)
    assert loads == ["load:m2l"]
    assert dataset.version > committed_version
    indices = cast("list[dict[str, object]]", dataset.list_indices())
    assert [entry["fields"] for entry in indices] == [[f"{M2L_FIELD}_vec"]]


def test_missing_embedding_specs_with_legacy_metadata_accepts_existing_policy(
    tmp_path: Path,
) -> None:
    """Legacy columns without identity metadata remain selectable.

    :param tmp_path: Scratch directory for the Lance dataset.
    """
    uri = tmp_path / "legacy-clap.lance"
    audio = np.zeros((2, 2, _FIXTURE_SAMPLES), dtype=np.float16)
    clap = np.zeros((2, CLAP_EMBEDDING_DIM), dtype=np.float32)
    write_lance_shard(uri, {AUDIO_FIELD: audio, CLAP_FIELD: clap})
    spec = _fake_spec("clap")
    config = AddEmbeddingsConfig(
        lance_uri=str(uri), embeddings=("clap",), build_index=False
    )

    with capture_logs() as logs:
        missing = _missing_embedding_specs(lance.dataset(uri), [spec], config)

    assert missing == []
    assert any(entry["event"] == "legacy_embedding_identity_missing" for entry in logs)


def test_add_embeddings_existing_artifact_identity_mismatch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed column cannot masquerade as a different checkpoint output.

    :param tmp_path: Scratch directory for the finalized shard.
    :param monkeypatch: Fixture installing a dependency-free CLAP policy.
    """
    uri = tmp_path / "checkpoint-mismatch.lance"
    write_minimal_lance_shard(uri, build_lance_smoke_spec())
    _install_fake_specs(monkeypatch, ("clap",))
    add_embeddings(
        AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("clap",), build_index=False)
    )

    monkeypatch.setitem(
        EMBEDDING_REGISTRY,
        "clap",
        replace(
            _fake_spec("clap"),
            resolve_artifact_identity=lambda checkpoint: f"fake:clap:{checkpoint}:v2",
        ),
    )
    with pytest.raises(ValueError, match="checkpoint identity"):
        add_embeddings(
            AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("clap",), build_index=False)
        )


def test_t5gemma_artifact_identity_includes_parameter_text_policy() -> None:
    """Text embeddings from different parameter policies are incompatible."""
    spec = _fake_spec("t5gemma")
    surge_xt = AddEmbeddingsConfig(
        lance_uri=_LANCE_URI,
        embeddings=("t5gemma",),
        param_spec_name="surge_xt",
        param_text_normalizer="param_names",
    )
    surge_4 = surge_xt.model_copy(update={"param_spec_name": "surge_4"})
    alternate_normalizer = surge_xt.model_copy(update={"param_text_normalizer": "future_policy"})

    assert _resolve_artifact_identity(spec, surge_xt) != _resolve_artifact_identity(spec, surge_4)
    assert _resolve_artifact_identity(spec, surge_xt) != _resolve_artifact_identity(
        spec, alternate_normalizer
    )


def test_add_embeddings_partial_policy_columns_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A policy with only its sequence output cannot be treated as complete.

    :param tmp_path: Scratch directory for the finalized shard.
    :param monkeypatch: Fixture installing a dependency-free M2L policy.
    """
    uri = tmp_path / "partial-policy.lance"
    write_minimal_lance_shard(uri, build_lance_smoke_spec())
    _install_fake_specs(monkeypatch, ("m2l",))
    config = AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("m2l",), build_index=False)
    add_embeddings(config)
    lance.dataset(uri).drop_columns([f"{M2L_FIELD}_vec"])

    with pytest.raises(ValueError, match="partial m2l columns"):
        add_embeddings(config)


def test_add_embeddings_with_index_disabled_writes_companions_without_building(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabling index builds retains pooled vectors while skipping every index.

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

    add_embeddings(AddEmbeddingsConfig(lance_uri=str(uri), embeddings=selected, build_index=False))

    assert calls == []
    dataset = lance.dataset(str(uri))
    assert {f"{M2L_FIELD}_vec", f"{SAME_S_FIELD}_vec"} <= set(dataset.schema.names)
    assert dataset.list_indices() == []


def test_add_embeddings_with_index_enabled_targets_declared_vector_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each index policy targets its vector column rather than a raw sequence.

    :param tmp_path: Scratch directory for the finalized shard.
    :param monkeypatch: Fixture installing fake specs and an index spy.
    """
    uri = tmp_path / "declared-indices.lance"
    write_minimal_lance_shard(uri, build_lance_smoke_spec())
    selected = ("same_l", "clap", "m2l")
    _install_fake_specs(monkeypatch, selected)
    calls: list[tuple[str, IndexSpec, int, int | None, str]] = []

    def record_index(
        dataset: lance.LanceDataset,
        column: str,
        *,
        index: IndexSpec,
        config: AddEmbeddingsConfig,
    ) -> bool:
        del dataset
        calls.append(
            (
                column,
                index,
                cast(int, config.num_partitions),
                config.num_sub_vectors,
                config.metric,
            )
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

    assert calls == [
        (
            f"{SAME_L_FIELD}_vec",
            IndexSpec(pool="mean", vector_column=f"{SAME_L_FIELD}_vec"),
            4,
            8,
            "l2",
        ),
        (CLAP_FIELD, IndexSpec(pool="none", vector_dim=CLAP_EMBEDDING_DIM), 4, 8, "l2"),
        (
            f"{M2L_FIELD}_vec",
            IndexSpec(pool="mean", vector_column=f"{M2L_FIELD}_vec"),
            4,
            8,
            "l2",
        ),
    ]


def test_add_embeddings_existing_mixed_target_writes_only_missing_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry skips a committed policy and writes the remaining selection.

    :param tmp_path: Scratch directory for the finalized shard.
    :param monkeypatch: Fixture installing dependency-free recording specs.
    """
    uri = tmp_path / "guard.lance"
    write_minimal_lance_shard(uri, build_lance_smoke_spec())
    _install_fake_specs(monkeypatch, ("same_s",))
    add_embeddings(
        AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("same_s",), build_index=False)
    )
    loads: list[str] = []
    _install_fake_specs(monkeypatch, ("clap", "same_s"), loads)

    add_embeddings(
        AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("clap", "same_s"), build_index=False)
    )

    assert loads == ["load:clap"]
    assert CLAP_FIELD in lance.dataset(uri).schema.names


@pytest.mark.slow
def test_add_embeddings_existing_index_config_mismatch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed index cannot satisfy a request for different search semantics.

    :param tmp_path: Scratch directory for the finalized shard.
    :param monkeypatch: Fixture installing a dependency-free CLAP policy.
    """
    uri = tmp_path / "index-config-mismatch.lance"
    write_minimal_lance_shard(uri, build_lance_smoke_spec(), num_rows=256)
    _install_fake_specs(monkeypatch, ("clap", "m2l"))
    add_embeddings(
        AddEmbeddingsConfig(
            lance_uri=str(uri),
            embeddings=("clap",),
            num_partitions=2,
            metric="l2",
        )
    )

    with pytest.raises(ValueError, match="index configuration"):
        add_embeddings(
            AddEmbeddingsConfig(
                lance_uri=str(uri),
                embeddings=("clap", "m2l"),
                num_partitions=2,
                metric="cosine",
            )
        )

    assert M2L_FIELD not in lance.dataset(uri).schema.names


def test_matching_index_exists_when_matching_entry_follows_stale_entry() -> None:
    """Index selection checks every same-column entry before rejecting stale policy."""
    dataset = SimpleNamespace(
        count_rows=lambda: 256,
        list_indices=lambda: [
            {"name": "stale", "fields": [CLAP_FIELD]},
            {"name": "current", "fields": [CLAP_FIELD]},
        ],
        index_statistics=lambda name: {
            "indices": [
                {
                    "metric_type": "cosine" if name == "stale" else "l2",
                    "num_partitions": 2,
                    "sub_index": {"num_sub_vectors": 16},
                }
            ]
        },
    )
    config = AddEmbeddingsConfig(
        lance_uri="dataset", num_partitions=2, num_sub_vectors=16, metric="l2"
    )

    assert _matching_index_exists(
        cast("lance.LanceDataset", dataset),
        CLAP_FIELD,
        index=IndexSpec(),
        config=config,
    )


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
        AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("clap",), build_index=False),
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
        AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("clap",), build_index=False),
    )

    built = build_index(
        lance.dataset(str(uri)),
        CLAP_FIELD,
        index=IndexSpec(),
        config=AddEmbeddingsConfig(lance_uri=str(uri), num_partitions=4, num_sub_vectors=16),
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
def test_m2l_exact_then_ann_search_with_stored_vector_returns_queried_row(
    tmp_path: Path,
) -> None:
    """The m2l companion serves exact and indexed self-nearest queries.

    :param tmp_path: Scratch directory for the dataset.
    """
    uri = tmp_path / "m2l-search.lance"
    _audio_dataset(uri, rows=300)
    base = _fake_spec("m2l")

    def load(checkpoint: str, config: AddEmbeddingsConfig) -> Callable[..., np.ndarray]:
        del checkpoint, config
        return _temporal_m2l

    spec = replace(base, load_encoder=load)
    config = AddEmbeddingsConfig(
        lance_uri=str(uri),
        embeddings=("m2l",),
        build_index=False,
        num_partitions=4,
        num_sub_vectors=4,
        metric="l2",
    )
    _write_columns(lance.dataset(str(uri)), [spec], _SAMPLE_RATE, config)
    dataset = lance.dataset(str(uri))
    vector_column = f"{M2L_FIELD}_vec"
    stored = dataset.to_table(columns=[vector_column, PARAM_ARRAY_FIELD])
    target_row = 137
    query = np.asarray(stored.column(vector_column)[target_row].as_py(), dtype=np.float32)
    expected_params = stored.column(PARAM_ARRAY_FIELD)[target_row].as_py()

    exact_hits = dataset.to_table(
        nearest={"column": vector_column, "q": query, "k": 1},
        columns=[PARAM_ARRAY_FIELD],
    )
    assert exact_hits.column(PARAM_ARRAY_FIELD)[0].as_py() == expected_params
    np.testing.assert_allclose(exact_hits.column("_distance")[0].as_py(), 0.0, atol=1e-5)

    built = build_index(dataset, vector_column, index=cast(IndexSpec, spec.index), config=config)

    dataset = lance.dataset(str(uri))
    indices = cast("list[dict[str, Any]]", dataset.list_indices())
    assert built is True
    assert [entry["fields"] for entry in indices] == [[vector_column]]
    assert all(entry["fields"] != [M2L_FIELD] for entry in indices)
    ann_hits = dataset.to_table(
        nearest={"column": vector_column, "q": query, "k": 1},
        columns=[PARAM_ARRAY_FIELD],
    )
    assert ann_hits.column(PARAM_ARRAY_FIELD)[0].as_py() == expected_params


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
        AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("clap",), build_index=False),
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


def test_same_s_num_latent_frames_for_one_second_returns_12() -> None:
    """SAME-S pads one second to six two-frame blocks."""
    assert same_s_num_latent_frames(SAME_SAMPLE_RATE, SAME_SAMPLE_RATE) == 12


def test_same_l_num_latent_frames_for_one_second_returns_11() -> None:
    """SAME-L emits one frame per complete or partial 4096-sample hop."""
    assert same_l_num_latent_frames(SAME_SAMPLE_RATE, SAME_SAMPLE_RATE) == 11


@pytest.mark.parametrize("frame_count", [same_s_num_latent_frames, same_l_num_latent_frames])
def test_same_num_latent_frames_for_four_second_conditioning_returns_44(
    frame_count: Callable[[int, int], int],
) -> None:
    """Both SAME models retain the 44-frame conditioning profile contract.

    :param frame_count: SAME model's latent-frame calculation.
    """
    assert frame_count(4 * SAME_SAMPLE_RATE, SAME_SAMPLE_RATE) == 44
    assert SAME_LATENT_FRAMES == 44


@pytest.mark.parametrize(
    ("num_samples", "sample_rate", "expected"),
    [
        (2 * SAME_DOWNSAMPLING_RATIO, SAME_SAMPLE_RATE // 2, 4),
        (SAME_DOWNSAMPLING_RATIO, SAME_SAMPLE_RATE, 2),
    ],
)
def test_same_s_num_latent_frames_resamples_and_pads_two_frame_blocks(
    num_samples: int, sample_rate: int, expected: int
) -> None:
    """SAME-S frame math follows resampling and two-hop padding.

    :param num_samples: Source clip length.
    :param sample_rate: Source rate in Hz.
    :param expected: Expected even latent-frame count.
    """
    assert same_s_num_latent_frames(num_samples, sample_rate) == expected


@pytest.mark.parametrize("frame_count", [same_s_num_latent_frames, same_l_num_latent_frames])
@pytest.mark.parametrize(
    ("num_samples", "sample_rate"), [(0, SAME_SAMPLE_RATE), (SAME_SAMPLE_RATE, 0)]
)
def test_same_num_latent_frames_with_nonpositive_input_raises(
    frame_count: Callable[[int, int], int], num_samples: int, sample_rate: int
) -> None:
    """Both SAME frame contracts reject non-positive lengths and rates.

    :param frame_count: SAME model's latent-frame calculation.
    :param num_samples: Invalid source length candidate.
    :param sample_rate: Invalid source rate candidate.
    """
    with pytest.raises(
        ValueError,
        match=f"need positive num_samples/sample_rate, got {num_samples}/{sample_rate}",
    ):
        frame_count(num_samples, sample_rate)


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
        ClapProcessor=SimpleNamespace(
            from_pretrained=lambda checkpoint: SimpleNamespace(
                feature_extractor=SimpleNamespace(sampling_rate=48_000)
            )
        ),
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_embeddings._resolve_clap_checkpoint",
        lambda _checkpoint: "/cache/clap",
    )
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
    downloads: list[str] = []

    class Model:
        feature_extractor = SimpleNamespace(sampling_rate=48_000)

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
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda checkpoint: downloads.append(checkpoint) or "/cache/custom-clap",
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        type("Transformers", (), {"ClapModel": Loader, "ClapProcessor": Loader}),
    )
    load_clap_audio_encoder("custom/clap", "cpu")

    assert downloads == ["custom/clap"]
    assert checkpoints == ["/cache/custom-clap", "/cache/custom-clap"]


def test_load_clap_audio_encoder_uses_processor_sample_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checkpoint's feature extractor controls CLAP resampling.

    :param monkeypatch: Fixture installing a dependency-free Transformers boundary.
    """

    class Model:
        def to(self, device: str) -> Model:
            del device
            return self

        def eval(self) -> Model:
            return self

        def get_audio_features(self, input_features: torch.Tensor) -> SimpleNamespace:
            samples = input_features[:, :1]
            return SimpleNamespace(pooler_output=samples.repeat(1, CLAP_EMBEDDING_DIM))

    class Processor:
        feature_extractor = SimpleNamespace(sampling_rate=32_000)

        def __call__(
            self, *, audio: list[np.ndarray], sampling_rate: int, return_tensors: str
        ) -> dict[str, torch.Tensor]:
            assert sampling_rate == 32_000
            assert return_tensors == "pt"
            return {"input_features": torch.tensor([[len(audio[0])]], dtype=torch.float32)}

    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_embeddings._resolve_clap_checkpoint",
        lambda _checkpoint: "/cache/clap",
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            ClapModel=SimpleNamespace(from_pretrained=lambda _checkpoint: Model()),
            ClapProcessor=SimpleNamespace(from_pretrained=lambda _checkpoint: Processor()),
        ),
    )

    encode = load_clap_audio_encoder(device="cpu")
    embedding = encode(np.zeros((1, 16_000), dtype=np.float32), 16_000)

    assert embedding.shape == (1, CLAP_EMBEDDING_DIM)
    assert embedding[0, 0] == 32_000


def test_resolve_clap_checkpoint_with_existing_local_path_returns_it(
    tmp_path: Path,
) -> None:
    """An existing local CLAP directory needs no snapshot download.

    :param tmp_path: Existing local checkpoint directory.
    """
    assert _resolve_clap_checkpoint(str(tmp_path)) == str(tmp_path)


def _materialize_clap_stub(
    downloads: list[tuple[str, Path]], uri: str, destination: Path
) -> None:
    """Record a checkpoint download and materialize the mirror contract.

    :param downloads: Download call ledger.
    :param uri: Source URI.
    :param destination: Local checkpoint directory.
    """
    downloads.append((uri, destination))
    destination.mkdir(parents=True, exist_ok=True)
    for filename in _CLAP_MIRROR_FILES:
        (destination / filename).write_bytes(b"downloaded")


def _write_tiny_clap_checkpoint(destination: Path) -> None:
    """Write an eight-file checkpoint loadable by real Transformers consumers.

    :param destination: Checkpoint directory to materialize.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import ClapConfig, ClapFeatureExtractor, ClapModel, PreTrainedTokenizerFast

    config = ClapConfig(
        projection_dim=8,
        text_config={
            "vocab_size": 8,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "max_position_embeddings": 16,
            "projection_dim": 8,
        },
        audio_config={
            "num_mel_bins": 64,
            "spec_size": 256,
            "hidden_size": 32,
            "projection_dim": 8,
            "depths": [1, 1, 1, 1],
            "num_attention_heads": [1, 2, 4, 8],
            "patch_embeds_hidden_size": 4,
            "patch_size": 4,
            "patch_stride": [4, 4],
            "num_classes": 8,
            "window_size": 4,
        },
    )
    model = ClapModel(config)
    config.save_pretrained(destination)
    torch.save(model.state_dict(), destination / "pytorch_model.bin")
    ClapFeatureExtractor(feature_size=64, sampling_rate=48_000).save_pretrained(destination)
    backend = Tokenizer(
        WordLevel(
            {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3, "sound": 4},
            unk_token="<unk>",
        )
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    tokenizer.save_pretrained(destination)
    (destination / "special_tokens_map.json").write_text(
        '{"unk_token":"<unk>","bos_token":"<s>","eos_token":"</s>","pad_token":"<pad>"}'
    )
    (destination / "vocab.json").write_text('{"<unk>":0}')
    (destination / "merges.txt").write_text("#version: 0.2\n")


def test_clap_checkpoint_defaults_keep_pipeline_and_training_sources_distinct() -> None:
    """Offline augmentation remains on HF while training uses its pinned R2 mirror."""
    assert DEFAULT_CLAP_CHECKPOINT == "laion/clap-htsat-unfused"
    assert (
        DEFAULT_CLAP_TRAINING_CHECKPOINT
        == "r2://intermediate-data/models/encoders/clap-htsat-unfused"
    )
    assert (
        DEFAULT_CLAP_TRAINING_CHECKPOINT_SHA256
        == "4a120dac122423e69160d8653fd9e5505fb35c6a482e564b62ce5ca07a7c54ca"
    )


def test_resolve_clap_checkpoint_with_default_hf_source_uses_legacy_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The add-embeddings default retains its established HF cache path.

    :param monkeypatch: Fixture isolating and recording the HF snapshot download.
    :param tmp_path: XDG cache root for the assertion.
    """
    downloads: list[tuple[str, str | None]] = []
    expected = tmp_path / "synth-setter/models/embeddings/clap-htsat-unfused"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo_id, *, local_dir=None: downloads.append((repo_id, local_dir)) or str(expected),
    )

    resolved = _resolve_clap_checkpoint(DEFAULT_CLAP_CHECKPOINT)

    assert resolved == str(expected)
    assert downloads == [(DEFAULT_CLAP_CHECKPOINT, str(expected))]


def test_resolve_clap_checkpoint_with_training_r2_source_uses_uri_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The training mirror hydrates outside the legacy HF cache.

    :param monkeypatch: Fixture isolating cache location and R2 download.
    :param tmp_path: XDG cache root for the assertion.
    """
    downloads: list[tuple[str, Path]] = []
    legacy = tmp_path / "synth-setter/models/embeddings/clap-htsat-unfused"
    legacy.mkdir(parents=True)
    (legacy / "legacy").write_text("preserve")
    expected = (
        tmp_path
        / "synth-setter/models/r2/intermediate-data/models/encoders/clap-htsat-unfused"
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr("synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda: None)
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_dir_no_overwrite",
        lambda uri, destination: _materialize_clap_stub(downloads, uri, destination),
    )

    resolved = _resolve_clap_checkpoint(DEFAULT_CLAP_TRAINING_CHECKPOINT)

    assert resolved == str(expected)
    assert legacy.joinpath("legacy").read_text() == "preserve"
    assert downloads[0][0] == DEFAULT_CLAP_TRAINING_CHECKPOINT
    assert downloads[0][1].parent == expected.parent
    assert downloads[0][1] != expected


def test_resolve_clap_checkpoint_from_r2_loads_real_transformers_consumers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An atomically published mirror loads as a real model and processor.

    :param monkeypatch: Fixture routing the R2 boundary through a local valid mirror.
    :param tmp_path: Scratch source and XDG cache root.
    """
    from transformers import ClapModel, ClapProcessor

    source = tmp_path / "source"
    _write_tiny_clap_checkpoint(source)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr("synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda: None)
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_dir_no_overwrite",
        lambda uri, destination: shutil.copytree(source, destination, dirs_exist_ok=True),
    )

    resolved = _resolve_clap_checkpoint("r2://bucket/tiny-clap")
    model = ClapModel.from_pretrained(resolved)
    processor = ClapProcessor.from_pretrained(resolved)

    assert model.config.projection_dim == 8
    assert processor.feature_extractor.sampling_rate == 48_000


@pytest.mark.parametrize("missing_file", _CLAP_MIRROR_FILES)
def test_resolve_clap_checkpoint_with_missing_required_file_repairs_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    missing_file: str,
) -> None:
    """Every model and tokenizer artifact participates in completeness.

    :param monkeypatch: Fixture isolating cache location and recording R2 access.
    :param tmp_path: XDG cache root holding the incomplete checkpoint.
    :param missing_file: Required mirror file omitted from the published cache.
    """
    checkpoint_dir = (
        tmp_path
        / "synth-setter/models/r2/intermediate-data/models/encoders/clap-htsat-unfused"
    )
    _materialize_clap_stub([], DEFAULT_CLAP_TRAINING_CHECKPOINT, checkpoint_dir)
    (checkpoint_dir / missing_file).unlink()
    downloads: list[tuple[str, Path]] = []
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr("synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda: None)
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_dir_no_overwrite",
        lambda uri, destination: _materialize_clap_stub(downloads, uri, destination),
    )

    resolved = Path(_resolve_clap_checkpoint(DEFAULT_CLAP_TRAINING_CHECKPOINT))

    assert resolved == checkpoint_dir
    assert {path.name for path in resolved.iterdir()} == set(_CLAP_MIRROR_FILES)
    assert len(downloads) == 1


@pytest.mark.parametrize("tree_state", ["absent", "empty"])
def test_clap_checkpoint_sha256_without_files_raises_value_error(
    tmp_path: Path, tree_state: str
) -> None:
    """A missing checkpoint tree cannot masquerade as the empty-string digest.

    :param tmp_path: Scratch root for the absent or empty tree.
    :param tree_state: Whether the checkpoint directory exists without files.
    """
    checkpoint_dir = tmp_path / "checkpoint"
    if tree_state == "empty":
        checkpoint_dir.mkdir()

    with pytest.raises(ValueError, match="CLAP checkpoint .* has no files"):
        clap_checkpoint_sha256(checkpoint_dir)


def test_resolve_clap_checkpoint_when_download_incomplete_preserves_cache_and_cleans_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed hydration leaves prior data untouched and removes staging.

    :param monkeypatch: Fixture forcing an incomplete R2 download.
    :param tmp_path: XDG cache root holding prior incomplete data.
    """
    checkpoint_dir = tmp_path / "synth-setter/models/r2/bucket/clap"
    checkpoint_dir.mkdir(parents=True)
    sentinel = checkpoint_dir / "prior"
    sentinel.write_text("preserve")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr("synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda: None)

    def download_incomplete(uri: str, destination: Path) -> None:
        del uri
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text("partial")

    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_dir_no_overwrite", download_incomplete
    )

    with pytest.raises(RuntimeError, match="downloaded CLAP checkpoint is incomplete"):
        _resolve_clap_checkpoint("r2://bucket/clap")

    assert sentinel.read_text() == "preserve"
    assert list(checkpoint_dir.parent.glob(".clap.staging-*")) == []


def test_resolve_clap_checkpoint_with_concurrent_callers_downloads_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Concurrent callers observe one atomically published checkpoint.

    :param monkeypatch: Fixture coordinating the R2 boundary.
    :param tmp_path: XDG cache root shared by both callers.
    """
    entered = threading.Event()
    release = threading.Event()
    downloads: list[tuple[str, Path]] = []
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr("synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda: None)

    def controlled_download(uri: str, destination: Path) -> None:
        entered.set()
        assert release.wait(timeout=5)
        _materialize_clap_stub(downloads, uri, destination)

    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_dir_no_overwrite", controlled_download
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_resolve_clap_checkpoint, DEFAULT_CLAP_TRAINING_CHECKPOINT)
        assert entered.wait(timeout=5)
        second = executor.submit(_resolve_clap_checkpoint, DEFAULT_CLAP_TRAINING_CHECKPOINT)
        release.set()
        resolved = [first.result(timeout=5), second.result(timeout=5)]

    assert resolved[0] == resolved[1]
    assert len(downloads) == 1
    assert set(Path(resolved[0]).iterdir()) == {
        Path(resolved[0]) / filename for filename in _CLAP_MIRROR_FILES
    }


def test_resolve_clap_checkpoint_with_full_r2_path_uses_distinct_cache_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """R2 mirrors sharing a basename retain distinct local cache directories.

    :param monkeypatch: Fixture replacing credential loading and download.
    :param tmp_path: XDG cache root for the custom sources.
    """
    downloads: list[tuple[str, Path]] = []
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr("synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda: None)
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_dir_no_overwrite",
        lambda uri, destination: _materialize_clap_stub(downloads, uri, destination),
    )

    resolved_a = _resolve_clap_checkpoint("r2://bucket/team-a/clap")
    resolved_b = _resolve_clap_checkpoint("r2://bucket/team-b/clap")

    assert resolved_a != resolved_b
    assert resolved_a.endswith("models/r2/bucket/team-a/clap")
    assert resolved_b.endswith("models/r2/bucket/team-b/clap")
    assert [download[0] for download in downloads] == [
        "r2://bucket/team-a/clap",
        "r2://bucket/team-b/clap",
    ]


@pytest.mark.parametrize(
    ("checkpoint", "model_name"),
    [
        (DEFAULT_SAME_S_CHECKPOINT, "same-s"),
        (DEFAULT_SAME_L_CHECKPOINT, "same-l"),
    ],
)
def test_resolve_same_checkpoint_with_default_r2_source_uses_canonical_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoint: str,
    model_name: str,
) -> None:
    """Hydrate each default SAME source into its stable model directory.

    :param monkeypatch: Fixture isolating cache location and R2 download.
    :param tmp_path: XDG cache root for the assertion.
    :param checkpoint: Default SAME R2 source under test.
    :param model_name: Canonical local directory name.
    """
    downloads: list[tuple[str, Path]] = []
    expected = tmp_path / "synth-setter/models/embeddings" / model_name
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr("synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda: None)
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_dir_no_overwrite",
        lambda uri, destination: downloads.append((uri, destination)),
    )

    resolved = resolve_same_checkpoint(checkpoint)

    assert resolved == expected
    assert downloads == [(checkpoint, expected)]


def test_resolve_same_checkpoint_with_full_r2_path_uses_distinct_cache_keys(
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

    dir_a = resolve_same_checkpoint("r2://bucket/team-a/same-s")
    dir_b = resolve_same_checkpoint("r2://bucket/team-b/same-s/")

    assert dir_a != dir_b
    assert downloads == [
        ("r2://bucket/team-a/same-s", dir_a),
        ("r2://bucket/team-b/same-s/", dir_b),
    ]


def test_resolve_same_checkpoint_with_existing_local_path_returns_it(
    tmp_path: Path,
) -> None:
    """An existing local checkpoint directory needs no download.

    :param tmp_path: Existing local checkpoint directory.
    """
    assert resolve_same_checkpoint(str(tmp_path)) == tmp_path


def test_resolve_same_checkpoint_with_repo_id_uses_hub_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A HuggingFace repo ID resolves to its downloaded snapshot directory.

    :param monkeypatch: Fixture replacing the Hub download.
    :param tmp_path: Downloaded snapshot directory.
    """
    downloads: list[str] = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo_id: downloads.append(repo_id) or str(tmp_path),
    )

    resolved = resolve_same_checkpoint("org/same-checkpoint")

    assert resolved == tmp_path
    assert downloads == ["org/same-checkpoint"]


def _install_sa3_factory(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[dict[str, object], int], torch.nn.Module],
) -> None:
    """Install a local SA3 factory module at the external dependency boundary.

    :param monkeypatch: Fixture restoring imported modules after the test.
    :param factory: Autoencoder factory exposed by the test module.
    """
    factory_module = ModuleType("stable_audio_3.factory")
    factory_module.create_autoencoder_from_config = factory  # type: ignore[attr-defined]
    package = ModuleType("stable_audio_3")
    package.factory = factory_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stable_audio_3", package)
    monkeypatch.setitem(sys.modules, "stable_audio_3.factory", factory_module)


class _RecordingSameAutoencoder(torch.nn.Module):
    """Record observable inference state for the SAME loader contract test."""

    def __init__(self) -> None:
        """Create a one-parameter encoder with inference-state logs."""
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.chunk_sizes: list[int] = []
        self.grad_enabled: list[bool] = []
        self.training_states: list[bool] = []
        self.parameter_grad_states: list[bool] = []

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode one chunk while recording model and autograd state.

        :param audio: Prepared stereo audio.
        :returns: Float64 first-channel values scaled by the loaded parameter.
        """
        self.chunk_sizes.append(len(audio))
        self.grad_enabled.append(torch.is_grad_enabled())
        self.training_states.append(self.training)
        self.parameter_grad_states.append(self.scale.requires_grad)
        return (audio[:, :1] * self.scale).double()


def _write_same_checkpoint(checkpoint_dir: Path, weights: dict[str, torch.Tensor]) -> None:
    """Write a minimal SA3-shaped checkpoint directory for loader tests.

    :param checkpoint_dir: Destination directory.
    :param weights: Safetensors state dictionary.
    """
    import json

    from safetensors.torch import save_file

    (checkpoint_dir / "model_config.json").write_text(
        json.dumps({"model": {"family": "same"}, "sample_rate": SAME_SAMPLE_RATE})
    )
    save_file(weights, checkpoint_dir / "model.safetensors")


def test_load_same_audio_encoder_uses_sa3_factory_and_preserves_inference_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The SA3 loader preserves config, batching, frozen inference, and output contracts.

    :param monkeypatch: Fixture installing the external SA3 factory boundary.
    :param tmp_path: Local checkpoint directory.
    """
    model = _RecordingSameAutoencoder()
    factory_calls: list[tuple[dict[str, object], int]] = []

    def factory(config: dict[str, object], sample_rate: int) -> torch.nn.Module:
        factory_calls.append((config, sample_rate))
        return model

    _install_sa3_factory(monkeypatch, factory)
    _write_same_checkpoint(tmp_path, model.state_dict())
    encode = load_same_audio_encoder(str(tmp_path), device="cpu")
    audio = np.ones((17, 2, 8), dtype=np.float32)

    latents = encode(audio)

    assert factory_calls == [({"family": "same"}, SAME_SAMPLE_RATE)]
    assert model.chunk_sizes == [16, 1]
    assert model.grad_enabled == [False, False]
    assert model.training_states == [False, False]
    assert model.parameter_grad_states == [False, False]
    assert next(model.parameters()).device.type == "cpu"
    assert latents.shape == (17, 1, 8)
    assert latents.dtype == np.float32
    np.testing.assert_array_equal(latents, np.full((17, 1, 8), 2.0, dtype=np.float32))


def test_load_same_audio_encoder_with_incompatible_state_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Missing and unexpected checkpoint keys fail strict SA3 state loading.

    :param monkeypatch: Fixture installing the external SA3 factory boundary.
    :param tmp_path: Local checkpoint directory.
    """
    _install_sa3_factory(monkeypatch, lambda config, sample_rate: torch.nn.Linear(2, 2))
    _write_same_checkpoint(tmp_path, {"unexpected": torch.ones(1)})

    with pytest.raises(RuntimeError, match=r"(?s)Missing key\(s\).*Unexpected key\(s\)"):
        load_same_audio_encoder(str(tmp_path), device="cpu")


def test_configure_lance_logging_without_debug_defaults_to_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default native logging preserves warnings when no override exists.

    :param monkeypatch: Fixture clearing ambient Lance logging.
    """
    monkeypatch.delenv("LANCE_LOG", raising=False)
    monkeypatch.delenv("LANCE_INCLUDE_VECTOR_CENTROIDS", raising=False)

    _configure_lance_logging(debug=False)

    assert os.environ["LANCE_LOG"] == "warn"
    assert os.environ["LANCE_INCLUDE_VECTOR_CENTROIDS"] == "false"


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
        AddEmbeddingsConfig(lance_uri=str(uri), resume_cache=resume_cache, build_index=False)
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
            checkpoint: str, config: AddEmbeddingsConfig, *, registry_name: str = name
        ) -> Callable[..., np.ndarray]:
            del checkpoint
            selected.append((registry_name, config.device))
            return _encoder_for(registry_name)

        monkeypatch.setitem(EMBEDDING_REGISTRY, name, replace(spec, load_encoder=load))

    with capture_logs() as logs:
        add_embeddings(
            AddEmbeddingsConfig(lance_uri=str(uri), device="mps", debug=True, build_index=False)
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

    def encode(
        sources: Mapping[str, np.ndarray], sample_rate: int, encoder: Encoder
    ) -> pa.Array:
        seen.append(sample_rate)
        return spec.encode_column(sources, sample_rate, encoder)

    monkeypatch.setitem(EMBEDDING_REGISTRY, "clap", replace(spec, encode_column=encode))
    add_embeddings(
        AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("clap",), build_index=False)
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
        AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("same_s",), build_index=False),
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


def test_add_embeddings_hydra_main_with_invalid_config_exits_one() -> None:
    """Trust-boundary validation failures use the structured CLI error path."""
    from synth_setter.pipeline.data.add_embeddings import _hydra_main

    cfg = DictConfig({"lance_uri": 123})

    with capture_logs() as logs, pytest.raises(SystemExit) as exc_info:
        _hydra_main.__wrapped__(cfg)

    assert exc_info.value.code == 1
    assert logs[-1]["event"] == "add_embeddings_failed"


def test_add_embeddings_hydra_main_with_subprocess_failure_exits_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checkpoint subprocess failures use the structured CLI error path.

    :param monkeypatch: Fixture replacing the embedding operation.
    """
    from synth_setter.pipeline.data.add_embeddings import _hydra_main

    def fail(config: AddEmbeddingsConfig) -> None:
        del config
        raise subprocess.CalledProcessError(1, ["rclone", "copy"])

    monkeypatch.setattr("synth_setter.pipeline.data.add_embeddings.add_embeddings", fail)
    cfg = DictConfig({"lance_uri": "dataset.lance"})

    with capture_logs() as logs, pytest.raises(SystemExit) as exc_info:
        _hydra_main.__wrapped__(cfg)

    assert exc_info.value.code == 1
    assert logs[-1]["event"] == "add_embeddings_failed"


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
            checkpoint: str, config: AddEmbeddingsConfig, *, registry_name: str = name
        ) -> Callable[..., np.ndarray]:
            del config
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


def _install_fake_t5gemma(
    monkeypatch: pytest.MonkeyPatch, seen: list[np.ndarray] | None = None
) -> None:
    """Install a dependency-free t5gemma entry recording its encoder input.

    :param monkeypatch: Fixture restoring the registry entry after the test.
    :param seen: Optional list receiving each encoded param batch.
    """

    def load(checkpoint: str, config: AddEmbeddingsConfig) -> Callable[..., np.ndarray]:
        del checkpoint, config

        def encode(rows: np.ndarray) -> np.ndarray:
            if seen is not None:
                seen.append(rows)
            return np.zeros((len(rows), 4, 5), dtype=np.float32)

        return encode

    monkeypatch.setitem(
        EMBEDDING_REGISTRY,
        "t5gemma",
        replace(
            EMBEDDING_REGISTRY["t5gemma"],
            load_encoder=load,
            resolve_artifact_identity=lambda checkpoint: f"fake:t5gemma:{checkpoint}",
        ),
    )


def test_add_embeddings_with_param_array_spec_feeds_encoder_param_rows_not_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A param-sourced embedding reads param_array, leaving audio untouched.

    :param tmp_path: Scratch directory for the shard.
    :param monkeypatch: Fixture installing the dependency-free t5gemma entry.
    """
    uri = tmp_path / "t5gemma.lance"
    write_minimal_lance_shard(uri, build_lance_smoke_spec())
    seen: list[np.ndarray] = []
    _install_fake_t5gemma(monkeypatch, seen)

    add_embeddings(
        AddEmbeddingsConfig(
            lance_uri=str(uri),
            embeddings=("t5gemma",),
            build_index=False,
            param_spec_name="surge_simple",
        )
    )

    expected = (
        lance.dataset(str(uri))
        .to_table(columns=[PARAM_ARRAY_FIELD])
        .column(PARAM_ARRAY_FIELD)
        .combine_chunks()
        .to_numpy_ndarray()
    )
    np.testing.assert_array_equal(seen[-1], expected)


def test_add_embeddings_with_param_array_spec_writes_fixed_shape_tensor_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The param-sourced embedding lands as a fixed-shape tensor column.

    :param tmp_path: Scratch directory for the shard.
    :param monkeypatch: Fixture installing the dependency-free t5gemma entry.
    """
    uri = tmp_path / "t5gemma.lance"
    write_minimal_lance_shard(uri, build_lance_smoke_spec())
    _install_fake_t5gemma(monkeypatch)

    add_embeddings(
        AddEmbeddingsConfig(
            lance_uri=str(uri),
            embeddings=("t5gemma",),
            build_index=False,
            param_spec_name="surge_simple",
        )
    )

    column_type = lance.dataset(str(uri)).schema.field(T5GEMMA_FIELD).type
    assert isinstance(column_type, pa.FixedShapeTensorType)
    assert column_type.shape == [4, 5]


def test_t5gemma_encoder_with_param_rows_wider_than_its_spec_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A param spec that does not describe the dataset fails instead of mislabeling it.

    :param monkeypatch: Fixture replacing the heavyweight text-model load.
    """
    monkeypatch.setattr(
        "synth_setter.pipeline.data.t5gemma.load_t5gemma_text_encoder",
        lambda checkpoint, device: lambda prompts: np.zeros((len(prompts), 4, 5), np.float32),
    )
    config = AddEmbeddingsConfig(
        lance_uri="unused.lance", embeddings=("t5gemma",), param_spec_name="surge_4"
    )
    encode = cast("ParamTextEncodeFn", _load_t5gemma_spec_encoder("unused-checkpoint", config))
    surge_4_width = resolve_param_spec(ParamSpecName("surge_4")).encoded_width

    with pytest.raises(ValueError, match="encoded width"):
        encode(np.zeros((2, surge_4_width + 1), dtype=np.float32))


def test_t5gemma_encoder_with_matching_param_rows_encodes_one_caption_per_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Correctly shaped param rows become one prompt per row.

    :param monkeypatch: Fixture replacing the heavyweight text-model load.
    """
    seen: list[list[str]] = []

    def fake_load(checkpoint: str, device: str) -> Callable[[list[str]], np.ndarray]:
        del checkpoint, device

        def encode_text(prompts: list[str]) -> np.ndarray:
            seen.append(prompts)
            return np.zeros((len(prompts), 4, 5), dtype=np.float32)

        return encode_text

    monkeypatch.setattr("synth_setter.pipeline.data.t5gemma.load_t5gemma_text_encoder", fake_load)
    config = AddEmbeddingsConfig(
        lance_uri="unused.lance", embeddings=("t5gemma",), param_spec_name="surge_4"
    )
    spec = resolve_param_spec(ParamSpecName("surge_4"))
    encode = cast("ParamTextEncodeFn", _load_t5gemma_spec_encoder("unused-checkpoint", config))

    encode(np.zeros((3, spec.encoded_width), dtype=np.float32))

    assert seen == [[", ".join(spec.names)] * 3]


def test_add_embeddings_config_without_lance_uri_raises() -> None:
    """An augmentation run requires one Lance dataset."""
    with pytest.raises(ValidationError, match="lance_uri"):
        AddEmbeddingsConfig.model_validate({})


def test_add_embeddings_config_with_dataset_root_target_raises() -> None:
    """Every embedding uses the same single-Lance-dataset target contract."""
    with pytest.raises(ValidationError, match="dataset_root_uri"):
        AddEmbeddingsConfig.model_validate(
            {"dataset_root_uri": "dataset", "embeddings": ("matpac_plus",)}
        )


def test_add_embeddings_config_with_t5gemma_and_no_param_spec_raises() -> None:
    """A param-sourced embedding cannot run without knowing its parameter space."""
    with pytest.raises(ValidationError, match="require param_spec_name"):
        AddEmbeddingsConfig(lance_uri="x.lance", embeddings=("t5gemma",))


def test_add_embeddings_config_with_audio_embeddings_needs_no_param_spec() -> None:
    """Audio-sourced embeddings are unaffected by the param-spec requirement."""
    config = AddEmbeddingsConfig(lance_uri="x.lance", embeddings=("clap",))

    assert config.param_spec_name is None


def test_add_embeddings_config_with_unknown_param_spec_name_raises() -> None:
    """An unregistered param spec is rejected at config time."""
    with pytest.raises(ValidationError, match="param_spec_name"):
        AddEmbeddingsConfig(
            lance_uri="x.lance", embeddings=("t5gemma",), param_spec_name="not_a_synth"
        )


def test_add_embeddings_config_with_unknown_text_normalizer_raises() -> None:
    """An unregistered text normalizer is rejected at config time."""
    with pytest.raises(ValidationError, match="param_text_normalizer"):
        AddEmbeddingsConfig(
            lance_uri="x.lance",
            embeddings=("t5gemma",),
            param_spec_name="surge_4",
            param_text_normalizer="not_a_strategy",
        )


def test_add_embeddings_config_composition_defaults_the_text_normalizer() -> None:
    """The shipped Hydra config exposes the param-text defaults."""
    cfg = _compose_add_embeddings()
    try:
        config = AddEmbeddingsConfig.from_hydra_cfg(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert (config.param_spec_name, config.param_text_normalizer) == (None, "param_names")


def test_load_m2l_spec_encoder_passes_the_configured_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registry adapter threads the run config's device to the m2l loader.

    :param monkeypatch: Fixture replacing the heavyweight encoder load.
    """
    seen: list[str | None] = []
    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_embeddings.load_m2l_audio_encoder",
        lambda device: seen.append(device) or (lambda audio: audio),
    )

    _load_m2l_spec_encoder("ignored", AddEmbeddingsConfig(lance_uri="x.lance", device="mps"))

    assert seen == ["mps"]


def test_load_clap_spec_encoder_passes_the_checkpoint_and_configured_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registry adapter threads both checkpoint and device to the CLAP loader.

    :param monkeypatch: Fixture replacing the heavyweight encoder load.
    """
    seen: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_embeddings.load_clap_audio_encoder",
        lambda checkpoint, device: (
            seen.append((checkpoint, device)) or (lambda audio, rate: audio)
        ),
    )

    _load_clap_spec_encoder("custom/clap", AddEmbeddingsConfig(lance_uri="x.lance", device="cpu"))

    assert seen == [("custom/clap", "cpu")]


def test_load_same_spec_encoder_passes_the_checkpoint_and_configured_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registry adapter threads both checkpoint and device to the SAME loader.

    :param monkeypatch: Fixture replacing the heavyweight encoder load.
    """
    seen: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_embeddings.load_same_audio_encoder",
        lambda checkpoint, device: seen.append((checkpoint, device)) or (lambda audio: audio),
    )

    _load_same_spec_encoder(
        "custom/same-s", AddEmbeddingsConfig(lance_uri="x.lance", device="cpu")
    )

    assert seen == [("custom/same-s", "cpu")]


def test_load_t5gemma_spec_encoder_without_a_param_spec_raises() -> None:
    """Calling the loader outside config validation still refuses to guess a param spec."""
    config = AddEmbeddingsConfig(lance_uri="x.lance", embeddings=("clap",))

    with pytest.raises(ValueError, match="require param_spec_name"):
        _load_t5gemma_spec_encoder("unused-checkpoint", config)


@pytest.mark.parametrize(
    ("shape", "reason"),
    [((2, 4), "rank"), ((1, 4, 5), "row count")],
)
def test_encode_t5gemma_column_with_malformed_encoder_output_raises(
    shape: tuple[int, ...], reason: str
) -> None:
    """A conditioner returning the wrong rank or row count fails before the Arrow write.

    :param shape: Malformed encoder output shape.
    :param reason: What the shape gets wrong, named for readability.
    """
    del reason
    params = np.zeros((2, 7), dtype=np.float32)

    with pytest.raises(ValueError, match="expected 2 rows"):
        _encode_t5gemma_column(
            {PARAM_ARRAY_FIELD: params}, 44100, lambda _: np.zeros(shape, dtype=np.float32)
        )


def _struct_sketch_controls(struct: pa.StructArray) -> np.ndarray:
    """Reassemble a stored sketch struct into the flat control stack.

    :param struct: Nested sketch column values.
    :returns: ``(rows, NUM_SKETCH_CONTROLS, F)`` float32 controls.
    """
    pitch = cast(
        "pa.FixedShapeTensorArray", struct.field(SKETCH_PITCH_CHILD)
    ).to_numpy_ndarray()
    rows, _, frames = pitch.shape
    stacked = np.empty((rows, NUM_SKETCH_CONTROLS, frames), dtype=np.float32)
    for child, row in (
        (SKETCH_LOUDNESS_CHILD, SKETCH_LOUDNESS_ROW),
        (SKETCH_CENTROID_CHILD, SKETCH_CENTROID_ROW),
    ):
        child_array = struct.field(child)
        stacked[:, row] = np.asarray(child_array.flatten()).reshape(rows, frames)
    stacked[:, SKETCH_PITCH_SLICE] = pitch
    return stacked


def _struct_sketch_vec(struct: pa.StructArray) -> np.ndarray:
    """Extract the nested IVF companion vectors from a stored sketch struct.

    :param struct: Nested sketch column values.
    :returns: ``(rows, NUM_SKETCH_CONTROLS)`` float32 vectors.
    """
    vec = struct.field(SKETCH_VEC_CHILD)
    return np.asarray(vec.flatten()).reshape(len(struct), NUM_SKETCH_CONTROLS)


def _stored_sketch_struct(dataset: lance.LanceDataset) -> pa.StructArray:
    """Read the full sketch struct column from a dataset.

    :param dataset: Dataset carrying the nested sketch column.
    :returns: Combined struct array.
    """
    table = dataset.to_table(columns=[SKETCH_STRUCT_FIELD])
    return cast("pa.StructArray", table.column(SKETCH_STRUCT_FIELD).combine_chunks())


def test_sketch_encode_column_builds_pooled_struct_and_vec() -> None:
    """The sketch closure stores pooled controls and their search vector."""
    audio = np.random.default_rng(7).random((3, 2, _FIXTURE_SAMPLES)).astype(np.float16)
    spec = EMBEDDING_REGISTRY["sketch"]

    array = spec.encode_column({AUDIO_FIELD: audio}, _SAMPLE_RATE, _fake_sketch)

    assert pa.types.is_struct(array.type)
    struct = cast("pa.StructArray", array)
    child_types = {field.name: field.type for field in struct.type}
    assert child_types[SKETCH_LOUDNESS_CHILD] == pa.list_(
        pa.float32(), SKETCH_STORAGE_FRAMES
    )
    assert child_types[SKETCH_CENTROID_CHILD] == pa.list_(
        pa.float32(), SKETCH_STORAGE_FRAMES
    )
    pitch_type = cast("pa.FixedShapeTensorType", child_types[SKETCH_PITCH_CHILD])
    assert list(pitch_type.shape) == [SKETCH_PITCH_BINS, SKETCH_STORAGE_FRAMES]
    assert child_types[SKETCH_VEC_CHILD] == pa.list_(pa.float32(), NUM_SKETCH_CONTROLS)
    full_controls = _fake_sketch(audio, _SAMPLE_RATE)
    expected = pool_sketch_controls(torch.from_numpy(full_controls)).numpy()
    np.testing.assert_array_equal(_struct_sketch_controls(struct), expected)
    np.testing.assert_allclose(_struct_sketch_vec(struct), expected.mean(axis=-1), rtol=1e-6)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_sketch_encode_column_with_nonfinite_output_raises(value: float) -> None:
    """Non-finite control cells never land in the dataset.

    :param value: Non-finite cell value emitted by the encoder.
    """
    audio = np.zeros((2, 2, _FIXTURE_SAMPLES), dtype=np.float16)

    def poisoned(batch: np.ndarray, sample_rate: int) -> np.ndarray:
        output = _fake_sketch(batch, sample_rate)
        output.flat[0] = value
        return output

    with pytest.raises(
        ValueError, match=f"{SKETCH_STRUCT_FIELD} embeddings contain non-finite values"
    ):
        EMBEDDING_REGISTRY["sketch"].encode_column({AUDIO_FIELD: audio}, _SAMPLE_RATE, poisoned)


@pytest.mark.parametrize(
    ("row", "value"),
    [(0, 1.2), (0, -1.2), (2, -0.1), (2, 1.2)],
    ids=["affine-high", "affine-low", "pitch-negative", "pitch-high"],
)
def test_sketch_encode_column_with_out_of_bounds_output_raises(row: int, value: float) -> None:
    """Finite control cells outside the documented bounds never land in the dataset.

    :param row: Control row receiving the out-of-bounds cell.
    :param value: Finite cell value outside that row's documented range.
    """
    audio = np.zeros((2, 2, _FIXTURE_SAMPLES), dtype=np.float16)

    def poisoned(batch: np.ndarray, sample_rate: int) -> np.ndarray:
        output = _fake_sketch(batch, sample_rate)
        output[0, row, 0] = value
        return output

    with pytest.raises(ValueError, match=f"{SKETCH_STRUCT_FIELD} controls out of bounds"):
        EMBEDDING_REGISTRY["sketch"].encode_column({AUDIO_FIELD: audio}, _SAMPLE_RATE, poisoned)


def test_sketch_encode_never_exceeds_extraction_batch_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every extractor invocation stays within SKETCH_ENCODE_MAX_BATCH.

    :param monkeypatch: Fixture recording extractor input batch sizes.
    """
    import synth_setter.features.sketch_controls as sketch_controls
    from synth_setter.pipeline.data.add_embeddings import (
        SKETCH_ENCODE_MAX_BATCH,
        _sketch_encode,
    )

    seen_sizes: list[int] = []

    def record(batch: torch.Tensor, sample_rate: int, device: str = "cpu") -> torch.Tensor:
        del sample_rate, device
        seen_sizes.append(len(batch))
        return torch.zeros(len(batch), NUM_SKETCH_CONTROLS, 1)

    monkeypatch.setattr(sketch_controls, "extract_sketch_controls_batch", record)
    rows = 2 * SKETCH_ENCODE_MAX_BATCH + 5
    audio = np.zeros((rows, 1, _FIXTURE_SAMPLES), dtype=np.float32)

    controls = _sketch_encode(audio, _SAMPLE_RATE)

    assert len(controls) == rows
    assert sum(seen_sizes) == rows
    assert max(seen_sizes) == SKETCH_ENCODE_MAX_BATCH
    assert all(size <= SKETCH_ENCODE_MAX_BATCH for size in seen_sizes)


@pytest.mark.slow
def test_sketch_encode_chunked_batch_matches_single_pass() -> None:
    """Memory-capped chunking preserves control values within float32 kernel jitter.

    Torch reduction kernels can vary by batch shape at approximately 1e-6.
    """
    from synth_setter.pipeline.data.add_embeddings import (
        SKETCH_ENCODE_MAX_BATCH,
        _sketch_encode,
    )

    rows = SKETCH_ENCODE_MAX_BATCH + 3
    # Clips long enough for PESTO's CQT and the loudness STFT windows.
    samples = 8192
    audio = (
        (np.random.default_rng(23).random((rows, 1, samples)) - 0.5) * 0.8
    ).astype(np.float32)

    chunked = _sketch_encode(audio, _SAMPLE_RATE)

    full = (
        extract_sketch_controls_batch(torch.from_numpy(audio), _SAMPLE_RATE).cpu().numpy()
    )
    np.testing.assert_allclose(chunked, full, atol=1e-5)


def test_sketch_encode_column_with_wrong_frame_count_raises() -> None:
    """The sketch closure rejects outputs off the shared mel frame grid."""
    audio = np.zeros((2, 2, _FIXTURE_SAMPLES), dtype=np.float16)

    def off_grid(batch: np.ndarray, sample_rate: int) -> np.ndarray:
        del sample_rate
        return np.zeros((len(batch), NUM_SKETCH_CONTROLS, 5), np.float32)

    with pytest.raises(ValueError, match=r"expected \(2, 386, 1\)"):
        EMBEDDING_REGISTRY["sketch"].encode_column({AUDIO_FIELD: audio}, _SAMPLE_RATE, off_grid)


@pytest.mark.parametrize("storage_version", ["2.1", "2.2"])
def test_write_columns_appends_sketch_struct_to_existing_dataset(
    tmp_path: Path, storage_version: Literal["2.1", "2.2"]
) -> None:
    """The add_columns UDF appends the whole nested struct on either storage format.

    The production dataset may carry Lance file format 2.1 or 2.2; whole-struct append must land
    identically on both.

    :param tmp_path: Scratch directory for the dataset.
    :param storage_version: Lance data storage version of the pre-existing dataset.
    """
    uri = tmp_path / "sketch.lance"
    rng = np.random.default_rng(4)
    audio = rng.random((4, 2, _FIXTURE_SAMPLES)).astype(np.float16)
    base = pa.record_batch(
        {
            AUDIO_FIELD: pa.FixedShapeTensorArray.from_numpy_ndarray(audio),
            PARAM_ARRAY_FIELD: pa.FixedShapeTensorArray.from_numpy_ndarray(
                rng.random((4, 3)).astype(np.float32)
            ),
        }
    )
    lance.write_dataset(base, str(uri), data_storage_version=storage_version)

    _write_columns(
        lance.dataset(str(uri)),
        [_fake_spec("sketch")],
        _SAMPLE_RATE,
        AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("sketch",), build_index=False),
    )

    dataset = lance.dataset(str(uri))
    assert SKETCH_STRUCT_FIELD in dataset.schema.names
    struct_type = dataset.schema.field(SKETCH_STRUCT_FIELD).type
    assert pa.types.is_struct(struct_type)
    pitch_type = cast(
        "pa.FixedShapeTensorType",
        struct_type.field(struct_type.get_field_index(SKETCH_PITCH_CHILD)).type,
    )
    assert list(pitch_type.shape) == [SKETCH_PITCH_BINS, SKETCH_STORAGE_FRAMES]
    struct = _stored_sketch_struct(dataset)
    full_controls = _fake_sketch(audio, _SAMPLE_RATE)
    expected = pool_sketch_controls(torch.from_numpy(full_controls)).numpy()
    np.testing.assert_array_equal(_struct_sketch_controls(struct), expected)
    np.testing.assert_allclose(_struct_sketch_vec(struct), expected.mean(axis=-1), rtol=1e-6)
    # Dotted-path child projection must serve reads without the sibling children.
    pitch_only = lance.dataset(str(uri)).to_table(
        columns=[f"{SKETCH_STRUCT_FIELD}.{SKETCH_PITCH_CHILD}"]
    )
    projected = cast(
        "pa.FixedShapeTensorArray", pitch_only.column(0).combine_chunks()
    ).to_numpy_ndarray()
    np.testing.assert_array_equal(projected, expected[:, SKETCH_PITCH_SLICE])


def test_write_columns_with_noop_add_columns_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write pass whose commit silently lands nothing fails hard, never cleanly.

    Guards the observed field failure mode where a run logs
    ``embedding_write_started`` and exits without committing (#2707).

    :param tmp_path: Scratch directory for the dataset.
    :param monkeypatch: Fixture stubbing the Lance commit to a no-op.
    """
    uri = tmp_path / "sketch-noop.lance"
    _audio_dataset(uri, rows=4)
    dataset = lance.dataset(str(uri))
    monkeypatch.setattr(dataset, "add_columns", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="without committing"):
        _write_columns(
            dataset,
            [_fake_spec("sketch")],
            _SAMPLE_RATE,
            AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("sketch",), build_index=False),
        )


def test_full_struct_rewrite_refreshes_sketch_children(tmp_path: Path) -> None:
    """The whole-struct rewrite path (add + drop + rename) refreshes stored controls.

    The struct is an atomic write unit — no per-child recompute exists — so this add-new-column /
    drop / rename flow is the escape hatch for refreshing any child (#2707).

    :param tmp_path: Scratch directory for the dataset.
    """
    from synth_setter.pipeline.data.lance_shard import sketch_struct_array

    uri = tmp_path / "sketch-rewrite.lance"
    audio = _audio_dataset(uri, rows=4)
    _write_columns(
        lance.dataset(str(uri)),
        [_fake_spec("sketch")],
        _SAMPLE_RATE,
        AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("sketch",), build_index=False),
    )
    full_controls = np.clip(_fake_sketch(audio, _SAMPLE_RATE) + 0.001, 0.0, 1.0)
    refreshed = pool_sketch_controls(torch.from_numpy(full_controls)).numpy()
    replacement = f"{SKETCH_STRUCT_FIELD}_refreshed"

    dataset = lance.dataset(str(uri))

    def rewrite(batch: pa.RecordBatch) -> pa.RecordBatch:
        # Stateless recompute from source rows: Lance also invokes the UDF for
        # schema inference, so call order carries no row-position information.
        decoded = batch.column(AUDIO_FIELD).to_numpy_ndarray()
        rows = np.clip(_fake_sketch(decoded, _SAMPLE_RATE) + 0.001, 0.0, 1.0)
        pooled = pool_sketch_controls(torch.from_numpy(rows)).numpy()
        return pa.RecordBatch.from_arrays([sketch_struct_array(pooled)], names=[replacement])

    dataset.add_columns(rewrite, read_columns=[AUDIO_FIELD])
    dataset.drop_columns([SKETCH_STRUCT_FIELD])
    # Runtime type-checks demand a bare AlterColumn dict although the stub
    # declares Iterable[AlterColumn]; cast bridges the disagreement.
    rename = {"path": replacement, "name": SKETCH_STRUCT_FIELD}
    dataset.alter_columns(cast("Any", rename))

    reread = lance.dataset(str(uri))
    assert SKETCH_STRUCT_FIELD in reread.schema.names
    assert replacement not in reread.schema.names
    pitch_type = cast(
        "pa.FixedShapeTensorType",
        reread.schema.field(SKETCH_STRUCT_FIELD).type.field(SKETCH_PITCH_CHILD).type,
    )
    assert list(pitch_type.shape) == [SKETCH_PITCH_BINS, SKETCH_STORAGE_FRAMES]
    np.testing.assert_array_equal(
        _struct_sketch_controls(_stored_sketch_struct(reread)), refreshed
    )


def test_add_embeddings_main_with_sketch_selection_writes_control_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real Hydra entrypoint lands sketch controls from the CLI selection.

    :param tmp_path: Scratch directory for the shard and Hydra run.
    :param monkeypatch: Fixture installing the entrypoint argv.
    """
    from synth_setter.pipeline.data.add_embeddings import main

    uri = tmp_path / "sketch-cli.lance"
    spec = build_lance_smoke_spec(task_name="sketch-cli-e2e")
    write_minimal_lance_shard(uri, spec)
    monkeypatch.setenv("PROJECT_ROOT", str(operator_workspace()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synth-setter-add-embeddings",
            f"lance_uri={uri}",
            "embeddings=[sketch]",
            "build_index=false",
            f"paths.log_dir={tmp_path}",
            f"hydra.run.dir={tmp_path / 'run'}",
        ],
    )

    main()

    dataset = lance.dataset(str(uri))
    assert SKETCH_STRUCT_FIELD in dataset.schema.names
    assert pa.types.is_struct(dataset.schema.field(SKETCH_STRUCT_FIELD).type)
    controls = _struct_sketch_controls(_stored_sketch_struct(dataset))
    render = spec.render_for_shard(spec.shards[0])
    audio_shape = dataset_field_shapes(render, spec.num_params)[AUDIO_FIELD]
    assert controls.shape == (
        audio_shape[0],
        NUM_SKETCH_CONTROLS,
        SKETCH_STORAGE_FRAMES,
    )
    assert np.isfinite(controls).all()
    assert controls.min() >= -1.0 and controls.max() <= 1.0


def test_add_embeddings_sketch_with_real_pesto_round_trips(tmp_path: Path) -> None:
    """The real extraction path lands controls matching direct extraction.

    :param tmp_path: Scratch directory for the dataset.
    """
    uri = tmp_path / "sketch-real.lance"
    spec = build_lance_smoke_spec(task_name="sketch-real-e2e")
    render = spec.render_for_shard(spec.shards[0])
    audio_shape = dataset_field_shapes(render, spec.num_params)[AUDIO_FIELD]
    rng = np.random.default_rng(11)
    audio = ((rng.random(audio_shape) - 0.5) * 0.8).astype(np.float16)
    write_minimal_lance_shard(uri, spec, audio=audio)

    # Pinned to CPU so the stored column and the reference share a device;
    # PESTO's convolutions drift ~1e-2 between CPU and CUDA kernels.
    add_embeddings(
        AddEmbeddingsConfig(
            lance_uri=str(uri), embeddings=("sketch",), build_index=False, device="cpu"
        )
    )

    controls = _struct_sketch_controls(_stored_sketch_struct(lance.dataset(str(uri))))
    sample_rate = int(render.sample_rate)
    full_controls = extract_sketch_controls_batch(
        torch.from_numpy(audio.astype(np.float32)), sample_rate, device="cpu"
    )
    expected = pool_sketch_controls(full_controls).numpy()
    assert controls.shape == (
        audio_shape[0],
        NUM_SKETCH_CONTROLS,
        SKETCH_STORAGE_FRAMES,
    )
    np.testing.assert_allclose(controls, expected, atol=1e-5)
    assert np.isfinite(controls).all()
    assert controls.min() >= -1.0 and controls.max() <= 1.0


def _nested_vec_dataset(uri: Path, rows: int) -> np.ndarray:
    """Write a dataset whose sketch struct carries only the nested vec child.

    :param uri: Output dataset path.
    :param rows: Number of rows.
    :returns: The vec vectors written into the struct.
    """
    rng = np.random.default_rng(3)
    vectors = rng.random((rows, NUM_SKETCH_CONTROLS)).astype(np.float32)
    vec = pa.FixedSizeListArray.from_arrays(
        pa.array(vectors.reshape(-1), pa.float32()), NUM_SKETCH_CONTROLS
    )
    struct = pa.StructArray.from_arrays([vec], names=[SKETCH_VEC_CHILD])
    table = pa.table(
        {SKETCH_STRUCT_FIELD: struct, "row": pa.array(np.arange(rows), pa.int32())}
    )
    lance.write_dataset(table, str(uri))
    return vectors


def test_build_index_on_nested_vec_child_serves_ann_self_query(tmp_path: Path) -> None:
    """The registry PQ split builds on the dotted vec child and answers ANN queries.

    :param tmp_path: Scratch directory for the dataset.
    """
    uri = tmp_path / "sketch-index.lance"
    vectors = _nested_vec_dataset(uri, rows=300)
    dataset = lance.dataset(str(uri))
    spec_index = EMBEDDING_REGISTRY["sketch"].index
    assert spec_index is not None

    built = build_index(
        dataset,
        SKETCH_VEC_COLUMN,
        index=spec_index,
        config=AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("sketch",)),
    )

    assert built is True
    indices = cast("list[dict[str, object]]", dataset.list_indices())
    index_fields = [index["fields"] for index in indices]
    assert [SKETCH_VEC_COLUMN] in index_fields
    target_row = 137
    hits = dataset.to_table(
        nearest={"column": SKETCH_VEC_COLUMN, "q": vectors[target_row], "k": 1},
        columns=["row"],
    )
    assert hits.column("row")[0].as_py() == target_row


def test_add_embeddings_sketch_end_to_end_writes_struct_and_nested_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The endpoint lands the nested layout and builds the vec index in one run.

    :param tmp_path: Scratch directory for the dataset.
    :param monkeypatch: Fixture installing the dependency-free sketch spec.
    """
    uri = tmp_path / "sketch-e2e.lance"
    spec = build_lance_smoke_spec(task_name="sketch-nested-e2e")
    render = spec.render_for_shard(spec.shards[0])
    audio_shape = dataset_field_shapes(render, spec.num_params)[AUDIO_FIELD]
    rows = 300  # Above MIN_ROWS_FOR_INDEX so the vec index really builds.
    rng = np.random.default_rng(17)
    audio = ((rng.random((rows, *audio_shape[1:])) - 0.5) * 0.8).astype(np.float16)
    write_minimal_lance_shard(uri, spec, num_rows=rows, audio=audio)
    _install_fake_specs(monkeypatch, ("sketch",))

    add_embeddings(
        AddEmbeddingsConfig(lance_uri=str(uri), embeddings=("sketch",), build_index=True)
    )

    dataset = lance.dataset(str(uri))
    assert pa.types.is_struct(dataset.schema.field(SKETCH_STRUCT_FIELD).type)
    indices = cast("list[dict[str, object]]", dataset.list_indices())
    assert [SKETCH_VEC_COLUMN] in [index["fields"] for index in indices]
    struct = _stored_sketch_struct(dataset)
    full_controls = _fake_sketch(audio.astype(np.float32), int(render.sample_rate))
    expected = pool_sketch_controls(torch.from_numpy(full_controls)).numpy()
    np.testing.assert_array_equal(_struct_sketch_controls(struct), expected)
    hits = dataset.to_table(
        nearest={"column": SKETCH_VEC_COLUMN, "q": _struct_sketch_vec(struct)[7], "k": 1}
    )
    assert hits.num_rows == 1


def test_sketch_spec_encoder_loads_pesto_on_the_configured_device() -> None:
    """The registry loader honours config.device instead of pinning PESTO to CPU."""
    from synth_setter.features.sketch_controls import DEFAULT_PESTO_CHECKPOINT, load_pesto_model
    from synth_setter.pipeline.data.add_embeddings import _load_sketch_spec_encoder

    config = AddEmbeddingsConfig(lance_uri=_LANCE_URI, embeddings=("sketch",), device="cpu")
    _load_sketch_spec_encoder(DEFAULT_PESTO_CHECKPOINT, config)

    assert next(load_pesto_model().parameters()).device.type == "cpu"


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_sketch_spec_encoder_with_cuda_config_extracts_on_cuda() -> None:
    """A cuda run config moves PESTO and the extraction itself onto the GPU."""
    from synth_setter.features.sketch_controls import DEFAULT_PESTO_CHECKPOINT, load_pesto_model
    from synth_setter.pipeline.data.add_embeddings import (
        SketchEncodeFn,
        _load_sketch_spec_encoder,
    )

    # Long enough for PESTO's CQT and the loudness STFT windows.
    audio = ((np.random.default_rng(7).random((4, 1, 8192)) - 0.5) * 0.8).astype(np.float32)
    config = AddEmbeddingsConfig(lance_uri=_LANCE_URI, embeddings=("sketch",), device="cuda")

    encode = cast("SketchEncodeFn", _load_sketch_spec_encoder(DEFAULT_PESTO_CHECKPOINT, config))
    controls = encode(audio, _SAMPLE_RATE)

    assert next(load_pesto_model().parameters()).device.type == "cuda"
    on_cpu = (
        extract_sketch_controls_batch(torch.from_numpy(audio), _SAMPLE_RATE, device="cpu")
        .cpu()
        .numpy()
    )
    assert controls.shape == on_cpu.shape
    affine = [SKETCH_LOUDNESS_ROW, SKETCH_CENTROID_ROW]
    np.testing.assert_allclose(controls[:, affine], on_cpu[:, affine], atol=1e-5)
    # Pitch activations are not bitwise portable across devices; the bin is.
    assert np.array_equal(
        controls[:, SKETCH_PITCH_SLICE].argmax(axis=1),
        on_cpu[:, SKETCH_PITCH_SLICE].argmax(axis=1),
    )
