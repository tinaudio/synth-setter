"""Tests for the Lance CLAP-embedding augmentation core.

The augmentation core injects the encoder as a plain callable, so those paths
run without a CLAP checkpoint; :func:`load_clap_audio_encoder` is tested against
fake ``transformers``/``torchaudio`` stand-ins (a real checkpoint is exercised
manually — see the PR's "Verification" checklist).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from lance.file import LanceFileReader

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    CLAP_EMBEDDING_DIM,
    CLAP_FIELD,
    MEL_SPEC_FIELD,
    PARAM_ARRAY_FIELD,
)
from synth_setter.pipeline.data.clap_lance import (
    CLAP_SAMPLE_RATE,
    EncodeFn,
    clap_augment_split,
    clap_augmented_schema,
    load_clap_audio_encoder,
)
from synth_setter.pipeline.data.lance_shard import iter_lance_column_rows, write_lance_file

# Small embedding width so fixtures stay tiny; unrelated to the real 512-d model.
_TEST_DIM = 4
_SAMPLE_RATE = 16000


def _write_shard(path: Path, audio: np.ndarray, mel: np.ndarray, params: np.ndarray) -> None:
    """Write a single-file Lance shard carrying the three core dataset columns.

    :param path: Output ``.lance`` shard file.
    :param audio: ``(N, C, T)`` audio rows.
    :param mel: ``(N, ...)`` mel-spectrogram rows.
    :param params: ``(N, P)`` parameter rows.
    """
    from tests.helpers.lance_fixtures import write_lance_shard

    write_lance_shard(
        path,
        {AUDIO_FIELD: audio, MEL_SPEC_FIELD: mel, PARAM_ARRAY_FIELD: params},
    )


def _read_schema_and_batches(path: Path) -> tuple[pa.Schema, list[pa.RecordBatch]]:
    """Open a Lance shard the way finalize does, returning its schema and batches.

    :param path: Lance shard file to read.
    :returns: The shard's Arrow schema paired with every record batch in it.
    """
    reader = LanceFileReader(str(path))
    return reader.metadata().schema, list(reader.read_all().to_batches())


def _constant_audio(num_rows: int, channels: int, time: int) -> np.ndarray:
    """Build float16 audio whose channel ``c`` is filled with ``c + 1`` (distinct per channel).

    The per-channel constant makes the mono downmix a known value, so downmix
    tests can assert the exact averaged input.

    :param num_rows: Leading row axis ``N``.
    :param channels: Channel axis ``C`` whose distinct fills drive the downmix.
    :param time: Sample axis ``T``.
    :returns: ``(N, C, T)`` float16 audio.
    """
    rows = np.empty((num_rows, channels, time), dtype=np.float16)
    for channel in range(channels):
        rows[:, channel, :] = channel + 1
    return rows


def _row_indexed_encoder(dim: int) -> EncodeFn:
    """Build a deterministic encoder mapping a mono batch to ``mean(row) + arange(dim)``.

    :param dim: Embedding width the encoder emits.
    :returns: An encode callable producing distinct, reproducible rows.
    """

    def encode(mono: np.ndarray, sample_rate: int) -> np.ndarray:  # noqa: ARG001
        return (mono.mean(axis=1, keepdims=True) + np.arange(dim)[None, :]).astype(np.float32)

    return encode


def test_clap_augmented_schema_appends_fixed_shape_tensor_clap_column(tmp_path: Path) -> None:
    """The augmented schema keeps every original column and adds a ``(dim,)`` float32 ``clap``.

    :param tmp_path: Pytest tmp dir hosting the fixture shard.
    """
    _write_shard(
        tmp_path / "shard.lance",
        audio=_constant_audio(2, 2, 8),
        mel=np.zeros((2, 2, 3), dtype=np.float32),
        params=np.zeros((2, 5), dtype=np.float32),
    )
    schema, _ = _read_schema_and_batches(tmp_path / "shard.lance")

    augmented = clap_augmented_schema(schema, _TEST_DIM)

    assert augmented.names == [AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD, CLAP_FIELD]
    clap_type = augmented.field(CLAP_FIELD).type
    assert tuple(clap_type.shape) == (_TEST_DIM,)
    assert clap_type.value_type == pa.float32()


def test_clap_augmented_schema_preserves_shard_metadata() -> None:
    """Schema-level metadata (ShardMetadata payload) survives the column append."""
    base = pa.schema(
        [pa.field(AUDIO_FIELD, pa.fixed_shape_tensor(pa.float16(), (2, 8)), nullable=False)],
        metadata={b"synth_setter.shard_metadata": b'{"sample_rate": 16000}'},
    )

    augmented = clap_augmented_schema(base, _TEST_DIM)

    assert augmented.metadata == base.metadata


def test_clap_augment_split_writes_encoder_output_as_clap_column(tmp_path: Path) -> None:
    """Each row's ``clap`` column equals the injected encoder's output for that row's audio.

    :param tmp_path: Pytest tmp dir hosting the input and output shards.
    """
    audio = _constant_audio(num_rows=3, channels=2, time=8)
    _write_shard(
        tmp_path / "in.lance",
        audio=audio,
        mel=np.zeros((3, 2, 3), dtype=np.float32),
        params=np.zeros((3, 5), dtype=np.float32),
    )
    schema, batches = _read_schema_and_batches(tmp_path / "in.lance")

    out_schema, out_batches = clap_augment_split(
        schema, batches, _row_indexed_encoder(_TEST_DIM), _SAMPLE_RATE, dim=_TEST_DIM
    )
    write_lance_file(tmp_path / "out.lance", out_schema, out_batches)

    clap_rows = list(iter_lance_column_rows(tmp_path / "out.lance", CLAP_FIELD))
    # Every channel mean is (1+2)/2 = 1.5, so each row is 1.5 + arange(dim).
    expected_row = (1.5 + np.arange(_TEST_DIM)).astype(np.float32)
    assert len(clap_rows) == 3
    for row in clap_rows:
        assert row.shape == (_TEST_DIM,)
        assert row.dtype == np.float32
        np.testing.assert_array_equal(row, expected_row)


def test_clap_augment_split_appends_clap_to_every_input_batch(tmp_path: Path) -> None:
    """Multi-batch input: each incoming batch is yielded with its own ``clap`` values.

    Exercises the per-batch boundary that the single-batch Lance-reader fixtures
    do not reach, and pins that each batch is encoded from its own audio.

    :param tmp_path: Pytest tmp dir hosting the seed shard.
    """
    _write_shard(
        tmp_path / "a.lance",
        audio=_constant_audio(1, 2, 4),  # channel mean 1.5
        mel=np.zeros((1, 2, 3), dtype=np.float32),
        params=np.zeros((1, 5), dtype=np.float32),
    )
    _write_shard(
        tmp_path / "b.lance",
        audio=_constant_audio(1, 2, 4) + 2,  # channel mean 3.5
        mel=np.zeros((1, 2, 3), dtype=np.float32),
        params=np.zeros((1, 5), dtype=np.float32),
    )
    schema, batch_a = _read_schema_and_batches(tmp_path / "a.lance")
    _, batch_b = _read_schema_and_batches(tmp_path / "b.lance")

    out_schema, out_batches = clap_augment_split(
        schema,
        [batch_a[0], batch_b[0]],
        _row_indexed_encoder(_TEST_DIM),
        _SAMPLE_RATE,
        dim=_TEST_DIM,
    )
    materialized = list(out_batches)

    assert len(materialized) == 2
    clap_index = out_schema.get_field_index(CLAP_FIELD)
    first = materialized[0].column(clap_index).to_pylist()[0]
    second = materialized[1].column(clap_index).to_pylist()[0]
    np.testing.assert_array_equal(first, (1.5 + np.arange(_TEST_DIM)).astype(np.float32))
    np.testing.assert_array_equal(second, (3.5 + np.arange(_TEST_DIM)).astype(np.float32))


def test_clap_augment_split_downmixes_channels_to_mono_before_encoding(tmp_path: Path) -> None:
    """The encoder receives a mono ``(B, T)`` batch averaged across the channel axis.

    :param tmp_path: Pytest tmp dir hosting the input shard.
    """
    audio = _constant_audio(num_rows=2, channels=2, time=4)
    _write_shard(
        tmp_path / "in.lance",
        audio=audio,
        mel=np.zeros((2, 2, 3), dtype=np.float32),
        params=np.zeros((2, 5), dtype=np.float32),
    )
    schema, batches = _read_schema_and_batches(tmp_path / "in.lance")
    seen: list[np.ndarray] = []

    def recording_encode(mono: np.ndarray, sample_rate: int) -> np.ndarray:  # noqa: ARG001
        seen.append(mono.copy())
        return np.zeros((mono.shape[0], _TEST_DIM), dtype=np.float32)

    _, out_batches = clap_augment_split(
        schema, batches, recording_encode, _SAMPLE_RATE, dim=_TEST_DIM
    )
    list(out_batches)

    assert len(seen) == 1
    assert seen[0].shape == (2, 4)
    assert seen[0].dtype == np.float32
    np.testing.assert_array_equal(seen[0], np.full((2, 4), 1.5, dtype=np.float32))


def test_clap_augment_split_passes_sample_rate_to_encoder(tmp_path: Path) -> None:
    """The sample rate threaded into the iterator reaches the encoder unchanged.

    :param tmp_path: Pytest tmp dir hosting the input shard.
    """
    _write_shard(
        tmp_path / "in.lance",
        audio=_constant_audio(1, 2, 4),
        mel=np.zeros((1, 2, 3), dtype=np.float32),
        params=np.zeros((1, 5), dtype=np.float32),
    )
    schema, batches = _read_schema_and_batches(tmp_path / "in.lance")
    seen_rates: list[int] = []

    def recording_encode(mono: np.ndarray, sample_rate: int) -> np.ndarray:
        seen_rates.append(sample_rate)
        return np.zeros((mono.shape[0], _TEST_DIM), dtype=np.float32)

    _, out_batches = clap_augment_split(schema, batches, recording_encode, 22050, dim=_TEST_DIM)
    list(out_batches)

    assert seen_rates == [22050]


@pytest.mark.parametrize("bad_width", [_TEST_DIM - 1, _TEST_DIM + 1])
def test_clap_augment_split_raises_when_encoder_width_differs_from_dim(
    tmp_path: Path, bad_width: int
) -> None:
    """A width guard rejects an encoder whose output dimension is not the declared ``dim``.

    :param tmp_path: Pytest tmp dir hosting the input shard.
    :param bad_width: A wrong embedding width the encoder emits.
    """
    _write_shard(
        tmp_path / "in.lance",
        audio=_constant_audio(2, 2, 4),
        mel=np.zeros((2, 2, 3), dtype=np.float32),
        params=np.zeros((2, 5), dtype=np.float32),
    )
    schema, batches = _read_schema_and_batches(tmp_path / "in.lance")

    def wrong_width_encode(mono: np.ndarray, sample_rate: int) -> np.ndarray:  # noqa: ARG001
        return np.zeros((mono.shape[0], bad_width), dtype=np.float32)

    _, out_batches = clap_augment_split(
        schema, batches, wrong_width_encode, _SAMPLE_RATE, dim=_TEST_DIM
    )
    with pytest.raises(ValueError, match=r"expected \(2, 4\)"):
        list(out_batches)


def test_clap_augment_split_raises_when_encoder_row_count_differs_from_batch(
    tmp_path: Path,
) -> None:
    """The guard rejects an encoder that returns a different row count than the input batch.

    Without it a row-count mismatch would surface only as a cryptic Arrow length error when the
    batch is reassembled.

    :param tmp_path: Pytest tmp dir hosting the input shard.
    """
    _write_shard(
        tmp_path / "in.lance",
        audio=_constant_audio(2, 2, 4),
        mel=np.zeros((2, 2, 3), dtype=np.float32),
        params=np.zeros((2, 5), dtype=np.float32),
    )
    schema, batches = _read_schema_and_batches(tmp_path / "in.lance")

    def dropped_row_encode(mono: np.ndarray, sample_rate: int) -> np.ndarray:  # noqa: ARG001
        return np.zeros((mono.shape[0] - 1, _TEST_DIM), dtype=np.float32)

    _, out_batches = clap_augment_split(
        schema, batches, dropped_row_encode, _SAMPLE_RATE, dim=_TEST_DIM
    )
    with pytest.raises(ValueError, match=r"expected \(2, 4\)"):
        list(out_batches)


def test_clap_augment_split_preserves_existing_columns(tmp_path: Path) -> None:
    """Augmentation is additive: the original audio/mel/param rows round-trip unchanged.

    :param tmp_path: Pytest tmp dir hosting the input and output shards.
    """
    mel = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    params = np.arange(2 * 5, dtype=np.float32).reshape(2, 5)
    _write_shard(
        tmp_path / "in.lance",
        audio=_constant_audio(2, 2, 4),
        mel=mel,
        params=params,
    )
    schema, batches = _read_schema_and_batches(tmp_path / "in.lance")

    out_schema, out_batches = clap_augment_split(
        schema, batches, _row_indexed_encoder(_TEST_DIM), _SAMPLE_RATE, dim=_TEST_DIM
    )
    write_lance_file(tmp_path / "out.lance", out_schema, out_batches)

    mel_rows = list(iter_lance_column_rows(tmp_path / "out.lance", MEL_SPEC_FIELD))
    param_rows = list(iter_lance_column_rows(tmp_path / "out.lance", PARAM_ARRAY_FIELD))
    np.testing.assert_array_equal(np.stack(mel_rows), mel)
    np.testing.assert_array_equal(np.stack(param_rows), params)


def _install_fake_clap_stack(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Swap the lazily-imported CLAP stack for fakes; return a resample-call log.

    Replaces ``transformers.ClapModel`` / ``ClapProcessor`` and
    ``torchaudio.functional.resample`` with in-memory fakes so the wiring inside
    :func:`load_clap_audio_encoder` runs without a checkpoint download. The fake
    model projects any batch to zeros ``(B, CLAP_EMBEDDING_DIM)`` float32.

    :param monkeypatch: Pytest monkeypatch fixture.
    :returns: List that records ``(orig_freq, new_freq)`` for each ``resample`` call.
    """
    import torch
    import torchaudio.functional as audio_fn
    import transformers

    # Belt-and-suspenders: even though from_pretrained is faked below, force
    # HuggingFace offline so any unintercepted call fails fast instead of hanging
    # on the Hub's 429-retry backoff (which has stalled the conda CI lane).
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    resample_calls: list[tuple[int, int]] = []

    def fake_resample(wav: Any, orig_freq: int, new_freq: int) -> Any:
        resample_calls.append((orig_freq, new_freq))
        return wav

    def get_audio_features(**inputs: Any) -> SimpleNamespace:
        (batch,) = {value.shape[0] for value in inputs.values()}
        pooled = torch.zeros((batch, CLAP_EMBEDDING_DIM), dtype=torch.float32)
        return SimpleNamespace(pooler_output=pooled)

    fake_model = SimpleNamespace(
        to=lambda device: fake_model,  # noqa: ARG005
        eval=lambda: fake_model,
        get_audio_features=get_audio_features,
    )

    def fake_processor(*, audio: list[Any], sampling_rate: int, return_tensors: str) -> dict:  # noqa: ARG001
        return {"input_features": torch.zeros((len(audio), 3), dtype=torch.float32)}

    # Patch ``from_pretrained`` on the real classes rather than swapping the classes
    # on the lazy ``transformers`` module: ``from transformers import ClapModel`` inside
    # the loader resolves the real class object, so only a method patch on that object
    # is seen — a module-attribute swap is bypassed and the real checkpoint downloads.
    monkeypatch.setattr(transformers.ClapModel, "from_pretrained", lambda checkpoint: fake_model)
    monkeypatch.setattr(
        transformers.ClapProcessor, "from_pretrained", lambda checkpoint: fake_processor
    )
    monkeypatch.setattr(audio_fn, "resample", fake_resample)
    return resample_calls


def test_load_clap_audio_encoder_maps_mono_batch_to_embedding_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loaded encoder maps a mono ``(B, T)`` batch to ``(B, CLAP_EMBEDDING_DIM)`` float32.

    The transformers/torchaudio stack is faked so the wiring inside
    :func:`load_clap_audio_encoder` runs without a checkpoint download.

    :param monkeypatch: Pytest monkeypatch fixture swapping in the fake CLAP stack.
    """
    resample_calls = _install_fake_clap_stack(monkeypatch)
    encode = load_clap_audio_encoder(device="cpu")

    embeddings = encode(np.zeros((2, 8), dtype=np.float32), CLAP_SAMPLE_RATE)

    assert embeddings.shape == (2, CLAP_EMBEDDING_DIM)
    assert embeddings.dtype == np.float32
    # At the native rate the encoder must not resample.
    assert resample_calls == []


def test_load_clap_audio_encoder_resamples_when_sample_rate_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-native sample rate is resampled to ``CLAP_SAMPLE_RATE`` before encoding.

    :param monkeypatch: Pytest monkeypatch fixture swapping in the fake CLAP stack.
    """
    resample_calls = _install_fake_clap_stack(monkeypatch)
    encode = load_clap_audio_encoder(device="cpu")

    embeddings = encode(np.zeros((1, 8), dtype=np.float32), _SAMPLE_RATE)

    assert embeddings.shape == (1, CLAP_EMBEDDING_DIM)
    assert resample_calls == [(_SAMPLE_RATE, CLAP_SAMPLE_RATE)]
