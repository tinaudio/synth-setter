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


def _distinct_clap(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """Give every row a unique embedding so each row is its own nearest neighbour.

    Encodes the row's mono mean into channel 0 and the row index into channel 1,
    leaving the rest zero — distinct per row regardless of duplicate audio.

    :param mono: ``(B, T)`` mono batch.
    :param sample_rate: Ignored.
    :returns: ``(B, CLAP_EMBEDDING_DIM)`` embedding batch, distinct per row.
    """
    del sample_rate
    out = np.zeros((mono.shape[0], CLAP_EMBEDDING_DIM), dtype=np.float32)
    out[:, 0] = mono.mean(axis=1)
    out[:, 1] = np.arange(mono.shape[0], dtype=np.float32)
    return out


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
def test_clap_exact_search_returns_queried_row_as_top_hit(tmp_path: Path) -> None:
    """Exact nearest search over ``clap`` returns the queried row itself at distance ~0.

    Uses Lance's exact (brute-force) scan — deterministic regardless of vector distribution — to
    pin the *semantic* contract: a stored vector's nearest neighbour is its own row. (IVF_PQ recall
    on realistic dense embeddings is covered by the real ≥256-row R2 e2e; PQ on synthetic toy
    vectors is degenerate and not a meaningful correctness signal.)

    :param tmp_path: Pytest-provided scratch directory for the dataset.
    """
    uri = str(tmp_path / "semantic.lance")
    _audio_dataset(uri, 64, with_params=True)

    # _distinct_clap gives every row a unique vector, so the exact nearest of a
    # stored vector is unambiguously its own row.
    add_embeddings(lance.dataset(uri), _fake_m2l, _distinct_clap, _SAMPLE_RATE, build_index=False)

    dataset = lance.dataset(uri)
    stored = dataset.to_table(columns=[CLAP_FIELD, PARAM_ARRAY_FIELD])
    target_row = 37
    query = np.array(stored.column(CLAP_FIELD)[target_row].as_py(), dtype=np.float32)
    expected_params = stored.column(PARAM_ARRAY_FIELD)[target_row].as_py()

    hits = dataset.to_table(
        nearest={"column": CLAP_FIELD, "q": query, "k": 1}, columns=[PARAM_ARRAY_FIELD]
    )

    assert hits.num_rows == 1
    assert hits.column(PARAM_ARRAY_FIELD)[0].as_py() == expected_params
    np.testing.assert_allclose(hits.column("_distance")[0].as_py(), 0.0, atol=1e-5)


@pytest.mark.slow
def test_build_clap_index_skips_when_too_few_rows(tmp_path: Path) -> None:
    uri = str(tmp_path / "tiny.lance")
    _audio_dataset(uri, 8)
    add_embeddings(lance.dataset(uri), _fake_m2l, _fake_clap, _SAMPLE_RATE, build_index=False)

    built = build_clap_index(lance.dataset(uri))

    assert built is False
    assert lance.dataset(uri).list_indices() == []


@pytest.mark.slow
def test_add_embeddings_rejects_dataset_without_audio_column(tmp_path: Path) -> None:
    """A dataset lacking the audio column raises before the UDF runs.

    :param tmp_path: Pytest-provided scratch directory for the dataset.
    """
    uri = str(tmp_path / "no_audio.lance")
    rng = np.random.default_rng(0)
    write_lance_shard(Path(uri), {PARAM_ARRAY_FIELD: rng.random((4, 3)).astype(np.float32)})

    with pytest.raises(ValueError, match="no 'audio' column"):
        add_embeddings(lance.dataset(uri), _fake_m2l, _fake_clap, _SAMPLE_RATE, build_index=False)


@pytest.mark.slow
def test_build_clap_index_rejects_num_sub_vectors_not_dividing_dim(tmp_path: Path) -> None:
    """A num_sub_vectors that does not divide the clap dim raises before any index work.

    :param tmp_path: Pytest-provided scratch directory for the dataset.
    """
    uri = str(tmp_path / "indivisible.lance")
    _audio_dataset(uri, 8)
    add_embeddings(lance.dataset(uri), _fake_m2l, _fake_clap, _SAMPLE_RATE, build_index=False)

    # 7 does not divide 512; reject before any index/training work begins.
    with pytest.raises(ValueError, match="does not divide clap dim"):
        build_clap_index(lance.dataset(uri), num_sub_vectors=7)

    assert lance.dataset(uri).list_indices() == []


def test_build_clap_index_rejects_non_positive_index_params(tmp_path: Path) -> None:
    """Non-positive num_sub_vectors / num_partitions raise instead of ZeroDivision/opaque.

    :param tmp_path: Pytest-provided scratch directory for the dataset.
    """
    uri = str(tmp_path / "badparams.lance")
    _audio_dataset(uri, 8)
    add_embeddings(lance.dataset(uri), _fake_m2l, _fake_clap, _SAMPLE_RATE, build_index=False)

    with pytest.raises(ValueError, match="num_sub_vectors must be >= 1"):
        build_clap_index(lance.dataset(uri), num_sub_vectors=0)
    with pytest.raises(ValueError, match="num_partitions must be >= 1"):
        build_clap_index(lance.dataset(uri), num_partitions=0)

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


def test_main_exposes_index_tuning_options() -> None:
    """The CLI surfaces the IVF_PQ tuning flags threaded into ``add_embeddings``."""
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0, result.output
    for flag in ("--num-partitions", "--num-sub-vectors", "--metric"):
        assert flag in result.output


@pytest.mark.slow
def test_main_threads_index_tuning_options_into_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI's index-tuning flags reach ``build_clap_index`` unchanged.

    :param tmp_path: Pytest-provided scratch directory for the dataset.
    :param monkeypatch: Fixture used to inject fake encoders + a spy index builder.
    """
    spec = build_lance_smoke_spec()
    uri = tmp_path / "tuned.lance"
    write_minimal_lance_shard(uri, spec)

    captured: dict[str, object] = {}

    def spy_build_clap_index(
        dataset: object,
        *,
        num_partitions: int | None = None,
        num_sub_vectors: int = 0,
        metric: str = "",
    ) -> bool:
        captured.update(
            num_partitions=num_partitions, num_sub_vectors=num_sub_vectors, metric=metric
        )
        return False

    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_embeddings.load_m2l_audio_encoder", lambda: _fake_m2l
    )
    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_embeddings.load_clap_audio_encoder",
        lambda checkpoint, device=None: _fake_clap,
    )
    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_embeddings.build_clap_index", spy_build_clap_index
    )

    result = CliRunner().invoke(
        main,
        [str(uri), "--num-partitions", "4", "--num-sub-vectors", "8", "--metric", "l2"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {"num_partitions": 4, "num_sub_vectors": 8, "metric": "l2"}


def test_main_exits_1_when_open_fails_with_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A cloud-creds RuntimeError from the open path must exit 1 cleanly, not traceback.
    def boom(uri: str) -> object:
        raise RuntimeError("missing R2 credentials")

    monkeypatch.setattr("synth_setter.pipeline.data.add_embeddings._open_lance_dataset", boom)

    result = CliRunner().invoke(main, ["s3://bucket/missing.lance"])

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)


@pytest.mark.slow
def test_main_exits_1_when_add_step_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = build_lance_smoke_spec()
    uri = tmp_path / "shard.lance"
    write_minimal_lance_shard(uri, spec)

    def boom() -> M2LEncodeFn:
        raise RuntimeError("encoder load blew up")

    # Dataset opens fine; a failure in the encode/add step must still exit 1 cleanly.
    monkeypatch.setattr("synth_setter.pipeline.data.add_embeddings.load_m2l_audio_encoder", boom)

    result = CliRunner().invoke(main, [str(uri)])

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert {M2L_FIELD, CLAP_FIELD}.isdisjoint(lance.dataset(str(uri)).schema.names)
