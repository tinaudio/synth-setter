"""Behavioral tests for :mod:`synth_setter.pipeline.data.add_embeddings`.

The model loaders are exercised through *injected* encode callables so the suite
never downloads a CLAP or music2latent checkpoint; the functional core, the
Lance ``add_columns`` wiring, and the vector-index path are what these tests pin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import lance
import numpy as np
import pyarrow as pa
import pytest
from click.testing import CliRunner

from synth_setter.data.vst.shapes import AUDIO_FIELD, CLAP_FIELD, M2L_FIELD, PARAM_ARRAY_FIELD
from synth_setter.pipeline.data.add_embeddings import (
    CLAP_EMBEDDING_DIM,
    ClapEncodeFn,
    M2LEncodeFn,
    _downmix_to_mono,
    add_embeddings,
    build_clap_index,
    embeddings_record_batch,
    main,
)
from tests.helpers.finalize_shards import build_lance_smoke_spec, write_minimal_lance_shard
from tests.helpers.lance_fixtures import write_lance_shard

_SAMPLE_RATE = 44100
# fake m2l per-row inner shape: (C*4, 3) — constant across rows (tensor contract).
_M2L_TIME = 3


def _fake_m2l(audio: np.ndarray) -> np.ndarray:
    """Tile the per-channel mean into a constant-shape ``(B, C*4, 3)`` latent.

    :param audio: ``(B, C, T)`` audio batch.
    :returns: ``(B, C*4, 3)`` stand-in latent batch.
    """
    per_channel = np.repeat(audio.mean(axis=2), 4, axis=1)  # (B, C*4)
    return np.repeat(per_channel[:, :, None], _M2L_TIME, axis=2)


def _fake_clap(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """Broadcast each row's grand mean into a ``(B, CLAP_EMBEDDING_DIM)`` embedding.

    :param mono: ``(B, T)`` mono batch.
    :param sample_rate: Ignored.
    :returns: ``(B, CLAP_EMBEDDING_DIM)`` stand-in embedding batch.
    """
    del sample_rate
    return np.repeat(mono.mean(axis=1, keepdims=True), CLAP_EMBEDDING_DIM, axis=1)


def _short_m2l(audio: np.ndarray) -> np.ndarray:
    """M2l encoder that drops a row, mismatching the input row count.

    :param audio: ``(B, C, T)`` audio batch.
    :returns: ``(B-1, C*4, 3)`` latent batch.
    """
    return _fake_m2l(audio)[:-1]


def _short_clap(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """CLAP encoder that drops a row, mismatching the input row count.

    :param mono: ``(B, T)`` mono batch.
    :param sample_rate: Ignored.
    :returns: ``(B-1, CLAP_EMBEDDING_DIM)`` embedding batch.
    """
    return _fake_clap(mono, sample_rate)[:-1]


def _wrong_dim_clap(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """CLAP encoder with the wrong embedding width.

    :param mono: ``(B, T)`` mono batch.
    :param sample_rate: Ignored.
    :returns: ``(B, CLAP_EMBEDDING_DIM // 2)`` embedding batch.
    """
    del sample_rate
    return np.repeat(mono.mean(axis=1, keepdims=True), CLAP_EMBEDDING_DIM // 2, axis=1)


def _nonfinite_m2l(value: float) -> M2LEncodeFn:
    """Build an m2l encoder whose first cell is ``value`` (a NaN/inf injector).

    :param value: Non-finite value to inject at row 0.
    :returns: An encoder poisoning one cell of its output.
    """

    def encode(audio: np.ndarray) -> np.ndarray:
        out = _fake_m2l(audio).astype(np.float32)
        out[0, 0, 0] = value
        return out

    return encode


def _nonfinite_clap(value: float) -> ClapEncodeFn:
    """Build a CLAP encoder whose first cell is ``value`` (a NaN/inf injector).

    :param value: Non-finite value to inject at row 0.
    :returns: An encoder poisoning one cell of its output.
    """

    def encode(mono: np.ndarray, sample_rate: int) -> np.ndarray:
        out = _fake_clap(mono, sample_rate).astype(np.float32)
        out[0, 0] = value
        return out

    return encode


def _audio_dataset(uri: str, rows: int, *, with_params: bool = False) -> np.ndarray:
    """Write a Lance dataset of ``rows`` random-audio rows; return the audio array.

    :param uri: Output ``.lance`` directory.
    :param rows: Row count.
    :param with_params: Also write a ``param_array`` column.
    :returns: The ``(rows, 2, 16)`` float16 audio written.
    """
    rng = np.random.default_rng(rows)
    audio = rng.random((rows, 2, 16)).astype(np.float16)
    columns: dict[str, np.ndarray] = {AUDIO_FIELD: audio}
    if with_params:
        columns[PARAM_ARRAY_FIELD] = rng.random((rows, 3)).astype(np.float32)
    write_lance_shard(Path(uri), columns)
    return audio


def test_downmix_to_mono_averages_channels_to_float32() -> None:
    """Channel averaging yields float32 mono with the channel axis collapsed."""
    audio = np.array([[[1.0, 3.0], [3.0, 5.0]]], dtype=np.float16)
    mono = _downmix_to_mono(audio)
    assert mono.shape == (1, 2)
    assert mono.dtype == np.float32
    np.testing.assert_allclose(mono, [[2.0, 4.0]])


def test_downmix_to_mono_single_channel_passes_signal_through() -> None:
    """A mono ``(B, 1, T)`` input is upcast to float32 with values unchanged."""
    audio = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float16)
    mono = _downmix_to_mono(audio)
    assert mono.shape == (1, 3)
    assert mono.dtype == np.float32
    np.testing.assert_allclose(mono, [[1.0, 2.0, 3.0]])


def test_embeddings_record_batch_builds_tensor_and_fixed_size_list() -> None:
    """M2l lands as a fixed-shape tensor; clap as a fixed-size-list<float32, dim>."""
    audio = np.random.default_rng(0).random((5, 2, 8)).astype(np.float16)
    batch = embeddings_record_batch(audio, _fake_m2l, _fake_clap, _SAMPLE_RATE)
    table = pa.Table.from_batches([batch])

    assert batch.schema.field(CLAP_FIELD).type == pa.list_(pa.float32(), CLAP_EMBEDDING_DIM)
    m2l = table.column(M2L_FIELD).combine_chunks().to_numpy_ndarray()
    clap = np.array(table.column(CLAP_FIELD).to_pylist(), dtype=np.float32)
    assert m2l.shape == (5, 8, _M2L_TIME)  # (B, C*4, T)
    assert clap.shape == (5, CLAP_EMBEDDING_DIM)
    np.testing.assert_allclose(m2l, _fake_m2l(audio))
    assert np.isfinite(m2l).all()
    assert np.isfinite(clap).all()


def test_embeddings_record_batch_rejects_m2l_row_count_mismatch() -> None:
    """An m2l encoder returning fewer rows than the input raises."""
    audio = np.zeros((4, 2, 8), dtype=np.float16)
    with pytest.raises(ValueError, match="row"):
        embeddings_record_batch(audio, _short_m2l, _fake_clap, _SAMPLE_RATE)


def test_embeddings_record_batch_rejects_clap_row_count_mismatch() -> None:
    """A CLAP encoder returning fewer rows than the input raises."""
    audio = np.zeros((4, 2, 8), dtype=np.float16)
    with pytest.raises(ValueError, match="row"):
        embeddings_record_batch(audio, _fake_m2l, _short_clap, _SAMPLE_RATE)


def test_embeddings_record_batch_rejects_wrong_clap_dim() -> None:
    """A CLAP embedding of the wrong width raises before the column is built."""
    audio = np.zeros((4, 2, 8), dtype=np.float16)
    with pytest.raises(ValueError, match="expected"):
        embeddings_record_batch(audio, _fake_m2l, _wrong_dim_clap, _SAMPLE_RATE)


@pytest.mark.parametrize("value", [np.nan, np.inf])
@pytest.mark.parametrize("side", ["m2l", "clap"])
def test_embeddings_record_batch_rejects_non_finite_embeddings(side: str, value: float) -> None:
    """A NaN/inf from either encoder raises rather than landing in the permanent column.

    :param side: Which encoder (``m2l`` or ``clap``) emits the poisoned cell.
    :param value: The non-finite value injected (NaN or inf).
    """
    audio = np.zeros((3, 2, 8), dtype=np.float16)
    m2l = _nonfinite_m2l(value) if side == "m2l" else _fake_m2l
    clap = _nonfinite_clap(value) if side == "clap" else _fake_clap
    with pytest.raises(ValueError, match="non-finite"):
        embeddings_record_batch(audio, m2l, clap, _SAMPLE_RATE)


@pytest.mark.slow
def test_add_embeddings_writes_searchable_columns_and_keeps_params(tmp_path: Path) -> None:
    uri = str(tmp_path / "smoke.lance")
    audio = _audio_dataset(uri, 6, with_params=True)

    add_embeddings(lance.dataset(uri), _fake_m2l, _fake_clap, _SAMPLE_RATE, build_index=False)

    table = lance.dataset(uri).to_table()
    assert set(table.column_names) == {AUDIO_FIELD, PARAM_ARRAY_FIELD, M2L_FIELD, CLAP_FIELD}
    assert table.schema.field(CLAP_FIELD).type == pa.list_(pa.float32(), CLAP_EMBEDDING_DIM)
    m2l = table.column(M2L_FIELD).combine_chunks().to_numpy_ndarray()
    np.testing.assert_allclose(m2l, _fake_m2l(audio))
    # Exact (brute-force) nearest works even without an index.
    hits = lance.dataset(uri).to_table(
        nearest={"column": CLAP_FIELD, "q": np.ones(CLAP_EMBEDDING_DIM, np.float32), "k": 3}
    )
    assert hits.num_rows == 3
    assert "_distance" in hits.column_names


@pytest.mark.slow
def test_add_embeddings_builds_ivf_pq_index_on_clap(tmp_path: Path) -> None:
    uri = str(tmp_path / "indexed.lance")
    _audio_dataset(uri, 300)

    add_embeddings(
        lance.dataset(uri), _fake_m2l, _fake_clap, _SAMPLE_RATE, build_index=True, num_partitions=4
    )

    indices = cast("list[dict[str, Any]]", lance.dataset(uri).list_indices())
    assert any(idx["fields"] == [CLAP_FIELD] for idx in indices)
    hits = lance.dataset(uri).to_table(
        nearest={"column": CLAP_FIELD, "q": np.ones(CLAP_EMBEDDING_DIM, np.float32), "k": 5}
    )
    assert hits.num_rows == 5


@pytest.mark.slow
def test_build_clap_index_skips_when_too_few_rows(tmp_path: Path) -> None:
    uri = str(tmp_path / "tiny.lance")
    _audio_dataset(uri, 8)
    add_embeddings(lance.dataset(uri), _fake_m2l, _fake_clap, _SAMPLE_RATE, build_index=False)

    built = build_clap_index(lance.dataset(uri))

    assert built is False
    assert lance.dataset(uri).list_indices() == []


@pytest.mark.slow
def test_add_embeddings_rejects_rerun_when_columns_already_exist(tmp_path: Path) -> None:
    uri = str(tmp_path / "twice.lance")
    _audio_dataset(uri, 6)
    add_embeddings(lance.dataset(uri), _fake_m2l, _fake_clap, _SAMPLE_RATE, build_index=False)

    with pytest.raises(ValueError, match="already has embedding column"):
        add_embeddings(lance.dataset(uri), _fake_m2l, _fake_clap, _SAMPLE_RATE, build_index=False)


@pytest.mark.slow
def test_main_adds_embeddings_using_sample_rate_from_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = build_lance_smoke_spec()
    uri = tmp_path / "shard.lance"
    write_minimal_lance_shard(uri, spec)

    seen_sample_rate: list[int] = []

    def clap_recording_sr(mono: np.ndarray, sample_rate: int) -> np.ndarray:
        seen_sample_rate.append(sample_rate)
        return _fake_clap(mono, sample_rate)

    # Loaders injected: the real encoders need checkpoints + a GPU (see the notebook).
    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_embeddings.load_m2l_audio_encoder",
        lambda: _fake_m2l,
    )
    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_embeddings.load_clap_audio_encoder",
        lambda checkpoint, device=None: clap_recording_sr,
    )

    result = CliRunner().invoke(main, [str(uri), "--no-build-index"])

    assert result.exit_code == 0, result.output
    assert seen_sample_rate
    assert all(sr == int(spec.render.sample_rate) for sr in seen_sample_rate)
    assert {M2L_FIELD, CLAP_FIELD} <= set(lance.dataset(str(uri)).schema.names)


def test_main_exits_1_when_open_fails_with_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A cloud-creds RuntimeError from the open path must exit 1 cleanly, not traceback.
    def boom(uri: str) -> object:
        raise RuntimeError("missing R2 credentials")

    monkeypatch.setattr("synth_setter.pipeline.data.add_embeddings._open_lance_dataset", boom)

    result = CliRunner().invoke(main, ["s3://bucket/missing.lance"])

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
