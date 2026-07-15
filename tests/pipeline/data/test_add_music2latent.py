"""Behavioral tests for :mod:`synth_setter.pipeline.data.add_music2latent`.

The model loader is exercised through an *injected* encode callable so the suite
never downloads a music2latent checkpoint; the functional core, the Lance
``add_columns`` wiring, and the multi-shard CLI selection are what these tests pin.
"""

from __future__ import annotations

from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
from click.testing import CliRunner

from synth_setter.data.vst.shapes import AUDIO_FIELD, PARAM_ARRAY_FIELD
from synth_setter.pipeline.data.add_music2latent import (
    MUSIC2LATENT_FIELD,
    M2LEncodeFn,
    add_music2latent,
    discover_shards,
    get_shard_id,
    main,
    music2latent_record_batch,
)
from tests.helpers.lance_fixtures import write_lance_shard

# fake m2l per-row inner shape: (C*4, 3) — constant across rows (tensor contract).
_M2L_TIME = 3


def _fake_m2l(audio: np.ndarray) -> np.ndarray:
    """Tile the per-channel mean into a constant-shape ``(B, C*4, 3)`` latent.

    :param audio: ``(B, C, T)`` audio batch.
    :returns: ``(B, C*4, 3)`` stand-in latent batch.
    """
    per_channel = np.repeat(audio.mean(axis=2), 4, axis=1)  # (B, C*4)
    return np.repeat(per_channel[:, :, None], _M2L_TIME, axis=2)


def _short_m2l(audio: np.ndarray) -> np.ndarray:
    """M2l encoder that drops a row, mismatching the input row count.

    :param audio: ``(B, C, T)`` audio batch.
    :returns: ``(B-1, C*4, 3)`` latent batch.
    """
    return _fake_m2l(audio)[:-1]


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


def _rank1_m2l(audio: np.ndarray) -> np.ndarray:
    """M2l encoder collapsing each row to a scalar, yielding a ``(B,)`` latent.

    :param audio: ``(B, C, T)`` audio batch.
    :returns: ``(B,)`` latent batch (wrong rank).
    """
    return audio.mean(axis=(1, 2))


def _rank2_m2l(audio: np.ndarray) -> np.ndarray:
    """M2l encoder dropping the time axis, yielding a ``(B, C)`` latent.

    :param audio: ``(B, C, T)`` audio batch.
    :returns: ``(B, C)`` latent batch (wrong rank).
    """
    return audio.mean(axis=2)


def _audio_shard(uri: Path, rows: int, *, with_params: bool = False, seed: int = 0) -> np.ndarray:
    """Write a Lance shard of ``rows`` random-audio rows; return the audio array.

    :param uri: Output ``.lance`` directory.
    :param rows: Row count.
    :param with_params: Also write a ``param_array`` column.
    :param seed: RNG seed for reproducible audio.
    :returns: The ``(rows, 2, 16)`` float16 audio written.
    """
    rng = np.random.default_rng(seed)
    audio = rng.random((rows, 2, 16)).astype(np.float16)
    columns: dict[str, np.ndarray] = {AUDIO_FIELD: audio}
    if with_params:
        columns[PARAM_ARRAY_FIELD] = rng.random((rows, 3)).astype(np.float32)
    write_lance_shard(uri, columns)
    return audio


def _write_shard_dir(root: Path, shard_ids: list[int], rows: int = 6) -> Path:
    """Write ``shard-<id>.lance`` datasets for each id under a fresh directory.

    :param root: Parent directory to create the shard directory in.
    :param shard_ids: Shard ids to materialize (zero-padded to 6 digits).
    :param rows: Rows per shard.
    :returns: The directory holding the shards.
    """
    data_dir = root / "shards"
    data_dir.mkdir()
    for shard_id in shard_ids:
        _audio_shard(data_dir / f"shard-{shard_id:06d}.lance", rows, seed=shard_id)
    return data_dir


def test_get_shard_id_parses_zero_padded_stem() -> None:
    """The integer id is parsed from a zero-padded ``shard-<id>.lance`` name."""
    assert get_shard_id(Path("/data/shard-000042.lance")) == 42


def test_music2latent_record_batch_builds_fixed_shape_tensor() -> None:
    """The latent lands as a fixed-shape tensor with the encoder's per-row shape."""
    audio = np.random.default_rng(0).random((5, 2, 8)).astype(np.float16)
    batch = music2latent_record_batch(audio, _fake_m2l)
    table = pa.Table.from_batches([batch])

    latents = table.column(MUSIC2LATENT_FIELD).combine_chunks().to_numpy_ndarray()
    assert latents.shape == (5, 8, _M2L_TIME)  # (B, C*4, T)
    np.testing.assert_allclose(latents, _fake_m2l(audio))
    assert np.isfinite(latents).all()


def test_music2latent_record_batch_rejects_row_count_mismatch() -> None:
    """An encoder returning fewer rows than the input raises."""
    audio = np.zeros((4, 2, 8), dtype=np.float16)
    with pytest.raises(ValueError, match="row"):
        music2latent_record_batch(audio, _short_m2l)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_music2latent_record_batch_rejects_non_finite_latents(value: float) -> None:
    """A NaN/inf latent raises rather than landing in the permanent column.

    :param value: The non-finite value injected (NaN or inf).
    """
    audio = np.zeros((3, 2, 8), dtype=np.float16)
    with pytest.raises(ValueError, match="non-finite"):
        music2latent_record_batch(audio, _nonfinite_m2l(value))


@pytest.mark.parametrize("encode", [_rank1_m2l, _rank2_m2l], ids=["rank1", "rank2"])
def test_music2latent_record_batch_rejects_wrong_rank_latents(encode: M2LEncodeFn) -> None:
    """A latent batch that is not rank-3 ``(B, C*D, T)`` raises before commit.

    :param encode: Encoder returning a ``(B,)`` or ``(B, C)`` latent batch.
    """
    audio = np.zeros((4, 2, 8), dtype=np.float16)
    with pytest.raises(ValueError, match="rank-3"):
        music2latent_record_batch(audio, encode)


@pytest.mark.slow
def test_add_music2latent_writes_column_and_keeps_other_columns(tmp_path: Path) -> None:
    """The latent column lands while pre-existing columns are left untouched.

    :param tmp_path: Per-test tmp dir holding the Lance shard(s).
    """
    uri = tmp_path / "smoke.lance"
    audio = _audio_shard(uri, 6, with_params=True)

    add_music2latent(lance.dataset(str(uri)), _fake_m2l)

    table = lance.dataset(str(uri)).to_table()
    assert set(table.column_names) == {AUDIO_FIELD, PARAM_ARRAY_FIELD, MUSIC2LATENT_FIELD}
    latents = table.column(MUSIC2LATENT_FIELD).combine_chunks().to_numpy_ndarray()
    np.testing.assert_allclose(latents, _fake_m2l(audio))


@pytest.mark.slow
def test_add_music2latent_rejects_rerun_when_column_exists(tmp_path: Path) -> None:
    """A second run on an augmented shard raises before touching the dataset.

    :param tmp_path: Per-test tmp dir holding the Lance shard(s).
    """
    uri = tmp_path / "twice.lance"
    _audio_shard(uri, 6)
    add_music2latent(lance.dataset(str(uri)), _fake_m2l)

    with pytest.raises(ValueError, match="already has"):
        add_music2latent(lance.dataset(str(uri)), _fake_m2l)


@pytest.mark.slow
def test_add_music2latent_rejects_dataset_without_audio_column(tmp_path: Path) -> None:
    """A dataset lacking the audio column raises before the UDF runs.

    :param tmp_path: Per-test tmp dir holding the Lance shard(s).
    """
    uri = tmp_path / "no_audio.lance"
    rng = np.random.default_rng(0)
    write_lance_shard(uri, {PARAM_ARRAY_FIELD: rng.random((4, 3)).astype(np.float32)})

    with pytest.raises(ValueError, match="no 'audio' column"):
        add_music2latent(lance.dataset(str(uri)), _fake_m2l)


def test_discover_shards_sorts_by_id_and_ignores_non_shards(tmp_path: Path) -> None:
    """Only ``shard-*.lance`` dirs are returned, in ascending id order.

    :param tmp_path: Per-test tmp dir holding the Lance shard(s).
    """
    data_dir = _write_shard_dir(tmp_path, [2, 0, 1])
    (data_dir / "notes.txt").write_text("ignore me")

    ids = [get_shard_id(s) for s in discover_shards(data_dir)]

    assert ids == [0, 1, 2]


def test_discover_shards_rejects_both_shard_and_range(tmp_path: Path) -> None:
    """Passing both a single shard and a range is a usage error.

    :param tmp_path: Per-test tmp dir holding the Lance shard(s).
    """
    data_dir = _write_shard_dir(tmp_path, [0])
    with pytest.raises(ValueError, match="both"):
        discover_shards(data_dir, shard_range=(0, 2), shard=0)


def test_discover_shards_filters_to_selected_shard(tmp_path: Path) -> None:
    """A single-shard selection keeps only the matching id.

    :param tmp_path: Per-test tmp dir holding the Lance shard(s).
    """
    data_dir = _write_shard_dir(tmp_path, [0, 1, 2])
    assert [get_shard_id(s) for s in discover_shards(data_dir, shard=1)] == [1]


def test_discover_shards_filters_to_half_open_range(tmp_path: Path) -> None:
    """A ``[lo, hi)`` range excludes the upper bound.

    :param tmp_path: Per-test tmp dir holding the Lance shard(s).
    """
    data_dir = _write_shard_dir(tmp_path, [0, 1, 2, 3])
    kept = [get_shard_id(s) for s in discover_shards(data_dir, shard_range=(1, 3))]
    assert kept == [1, 2]


@pytest.mark.slow
def test_main_adds_column_to_every_shard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI backfills the latent column across all discovered shards.

    :param tmp_path: Per-test tmp dir holding the Lance shard(s).
    :param monkeypatch: Patches the module-level ``load_m2l_audio_encoder``.
    """
    data_dir = _write_shard_dir(tmp_path, [0, 1, 2])
    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_music2latent.load_m2l_audio_encoder", lambda: _fake_m2l
    )

    result = CliRunner().invoke(main, [str(data_dir)])

    assert result.exit_code == 0, result.output
    for shard in data_dir.glob("shard-*.lance"):
        assert MUSIC2LATENT_FIELD in lance.dataset(str(shard)).schema.names


@pytest.mark.slow
def test_main_skips_shards_already_having_the_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shard already carrying the column is left on its existing version.

    :param tmp_path: Per-test tmp dir holding the Lance shard(s).
    :param monkeypatch: Patches the module-level ``load_m2l_audio_encoder``.
    """
    data_dir = _write_shard_dir(tmp_path, [0, 1])
    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_music2latent.load_m2l_audio_encoder", lambda: _fake_m2l
    )
    done = data_dir / "shard-000000.lance"
    add_music2latent(lance.dataset(str(done)), _fake_m2l)
    version_before = lance.dataset(str(done)).version

    result = CliRunner().invoke(main, [str(data_dir)])

    assert result.exit_code == 0, result.output
    assert lance.dataset(str(done)).version == version_before
    assert MUSIC2LATENT_FIELD in lance.dataset(str(data_dir / "shard-000001.lance")).schema.names


def test_main_warns_and_returns_when_no_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty directory exits 0 without loading the encoder.

    :param tmp_path: Per-test tmp dir holding the Lance shard(s).
    :param monkeypatch: Patches the module-level ``load_m2l_audio_encoder``.
    """
    empty = tmp_path / "empty"
    empty.mkdir()

    def fail_load() -> M2LEncodeFn:
        raise AssertionError("encoder must not load when there are no shards")

    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_music2latent.load_m2l_audio_encoder", fail_load
    )

    result = CliRunner().invoke(main, [str(empty)])

    assert result.exit_code == 0, result.output


def test_main_exits_1_when_encoder_load_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing encoder load exits 1 cleanly rather than raising.

    :param tmp_path: Per-test tmp dir holding the Lance shard(s).
    :param monkeypatch: Patches the module-level ``load_m2l_audio_encoder``.
    """
    data_dir = _write_shard_dir(tmp_path, [0])

    def boom() -> M2LEncodeFn:
        raise RuntimeError("encoder load blew up")

    monkeypatch.setattr("synth_setter.pipeline.data.add_music2latent.load_m2l_audio_encoder", boom)

    result = CliRunner().invoke(main, [str(data_dir)])

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert (
        MUSIC2LATENT_FIELD not in lance.dataset(str(data_dir / "shard-000000.lance")).schema.names
    )


@pytest.mark.slow
def test_main_continues_past_failed_shard_and_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One shard's failure is isolated: later shards still get the column and the CLI exits 1.

    :param tmp_path: Per-test tmp dir holding the Lance shard(s).
    :param monkeypatch: Patches the encoder loader and ``add_music2latent``.
    """
    data_dir = _write_shard_dir(tmp_path, [0, 1, 2])
    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_music2latent.load_m2l_audio_encoder", lambda: _fake_m2l
    )

    real_add = add_music2latent
    attempted: list[str] = []

    def flaky_add(
        dataset: lance.LanceDataset,
        m2l_encode: M2LEncodeFn,
        *,
        batch_size: int | None = None,
    ) -> None:
        name = Path(dataset.uri).name
        attempted.append(name)
        if name == "shard-000001.lance":
            raise RuntimeError("boom on shard 1")
        real_add(dataset, m2l_encode, batch_size=batch_size)

    monkeypatch.setattr("synth_setter.pipeline.data.add_music2latent.add_music2latent", flaky_add)

    result = CliRunner().invoke(main, [str(data_dir)])

    assert result.exit_code == 1
    assert attempted == ["shard-000000.lance", "shard-000001.lance", "shard-000002.lance"]
    assert MUSIC2LATENT_FIELD in lance.dataset(str(data_dir / "shard-000000.lance")).schema.names
    assert MUSIC2LATENT_FIELD in lance.dataset(str(data_dir / "shard-000002.lance")).schema.names
    assert (
        MUSIC2LATENT_FIELD not in lance.dataset(str(data_dir / "shard-000001.lance")).schema.names
    )


@pytest.mark.slow
def test_main_continues_past_unreadable_shard_and_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt (manifest-less) shard between valid ones is isolated, not fatal.

    :param tmp_path: Per-test tmp dir holding the Lance shard(s).
    :param monkeypatch: Patches the module-level ``load_m2l_audio_encoder``.
    """
    data_dir = _write_shard_dir(tmp_path, [0, 2])
    corrupt = data_dir / "shard-000001.lance"
    corrupt.mkdir()
    (corrupt / "garbage.bin").write_bytes(b"not a lance dataset")
    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_music2latent.load_m2l_audio_encoder", lambda: _fake_m2l
    )

    result = CliRunner().invoke(main, [str(data_dir)])

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert MUSIC2LATENT_FIELD in lance.dataset(str(data_dir / "shard-000000.lance")).schema.names
    assert MUSIC2LATENT_FIELD in lance.dataset(str(data_dir / "shard-000002.lance")).schema.names
