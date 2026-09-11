"""Real Lance coverage for dataset-backed pyFDN input audio."""

from __future__ import annotations

from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

from synth_setter.data.vst.input_audio import InputAudioPool
from synth_setter.pipeline.data.lance_shard import SHARD_METADATA_SCHEMA_KEY, tensor_array
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata
from synth_setter.pipeline.schemas.spec import InputAudioSource
from synth_setter.renderer_backend import INPUT_AUDIO_ADAPTATION_POLICY

_FRAMES = 8
_SAMPLE_RATE = 44_100


def _write_source(
    root: Path,
    audio: np.ndarray,
    *,
    field_name: str = "audio",
    sample_rate: int = _SAMPLE_RATE,
) -> str:
    split = root / "train.lance"
    metadata = ShardMetadata(
        velocity=0,
        signal_duration_seconds=_FRAMES / sample_rate,
        sample_rate=sample_rate,
        channels=audio.shape[1],
        min_loudness=-100.0,
    )
    field = pa.field(
        field_name,
        pa.fixed_shape_tensor(pa.float32(), audio.shape[1:]),
        nullable=False,
    )
    schema = pa.schema(
        [field],
        metadata={SHARD_METADATA_SCHEMA_KEY: metadata.model_dump_json().encode()},
    )
    if audio.shape[0] == 0:
        storage = pa.array([], type=field.type.storage_type)
    else:
        storage = tensor_array(audio, np.dtype("float32"), audio.shape[1:]).storage
    audio_array = pa.ExtensionArray.from_storage(field.type, storage)
    table = pa.Table.from_arrays([audio_array], schema=schema)
    dataset = lance.write_dataset(table, str(split))
    transaction = dataset.read_transaction(dataset.version)
    assert transaction is not None
    return transaction.uuid


def _source(root: Path, txid: str) -> InputAudioSource:
    return InputAudioSource(
        dataset_uri=str(root),
        split="train",
        snapshot_txid=txid,
        sampling_seed=7,
    )


def test_input_audio_pool_real_lance_row_returns_mono_waveform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    audio = np.stack(
        [
            np.linspace(-0.5, 0.5, _FRAMES, dtype=np.float32),
            np.linspace(0.5, -0.5, _FRAMES, dtype=np.float32),
        ]
    )[:, None, :]
    txid = _write_source(source_root, audio)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    pool = InputAudioPool(_source(source_root, txid), sample_rate=_SAMPLE_RATE, frames=_FRAMES)

    assert pool.row_count == 2
    assert pool.take(1).shape == (_FRAMES,)
    np.testing.assert_array_equal(pool.take(1), audio[1, 0])
    assert pool.materialized_path.is_dir()


def test_input_audio_pool_real_lance_row_returns_stereo_waveform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    audio = np.arange(2 * _FRAMES, dtype=np.float32).reshape(1, 2, _FRAMES) / 100
    txid = _write_source(source_root, audio)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    pool = InputAudioPool(
        _source(source_root, txid),
        sample_rate=_SAMPLE_RATE,
        frames=_FRAMES,
        input_channels=2,
    )

    assert pool.take(0).shape == (2, _FRAMES)
    np.testing.assert_array_equal(pool.take(0), audio[0])


def test_input_audio_pool_mono_source_for_stereo_renderer_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    audio = np.zeros((1, 1, _FRAMES), dtype=np.float32)
    txid = _write_source(source_root, audio)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    with pytest.raises(ValueError, match="input audio row shape"):
        InputAudioPool(
            _source(source_root, txid),
            sample_rate=_SAMPLE_RATE,
            frames=_FRAMES,
            input_channels=2,
        )


def test_input_audio_pool_row_selection_retry_changes_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    audio = np.zeros((2, 1, _FRAMES), dtype=np.float32)
    txid = _write_source(source_root, audio)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    pool = InputAudioPool(_source(source_root, txid), sample_rate=_SAMPLE_RATE, frames=_FRAMES)

    assert pool.row_index(11, 3, 0) == 1
    assert pool.row_index(11, 3, 1) == 0


def test_input_audio_pool_snapshot_pin_excludes_later_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    audio = np.zeros((1, 1, _FRAMES), dtype=np.float32)
    txid = _write_source(source_root, audio)
    lance.write_dataset(
        pa.table({"audio": tensor_array(np.ones_like(audio), np.dtype("float32"), (1, _FRAMES))}),
        str(source_root / "train.lance"),
        mode="append",
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    pool = InputAudioPool(_source(source_root, txid), sample_rate=_SAMPLE_RATE, frames=_FRAMES)

    assert pool.row_count == 1


def test_input_audio_pool_empty_dataset_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    audio = np.empty((0, 1, _FRAMES), dtype=np.float32)
    txid = _write_source(source_root, audio)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    with pytest.raises(ValueError, match="non-empty"):
        InputAudioPool(_source(source_root, txid), sample_rate=_SAMPLE_RATE, frames=_FRAMES)


def test_input_audio_pool_missing_audio_column_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    audio = np.zeros((1, 1, _FRAMES), dtype=np.float32)
    txid = _write_source(source_root, audio, field_name="waveform")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    with pytest.raises((KeyError, ValueError), match="audio"):
        InputAudioPool(_source(source_root, txid), sample_rate=_SAMPLE_RATE, frames=_FRAMES)


def test_input_audio_pool_longer_row_truncates_to_first_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    audio = np.linspace(-0.5, 0.5, 12, dtype=np.float32).reshape(1, 1, 12)
    txid = _write_source(source_root, audio)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    pool = InputAudioPool(_source(source_root, txid), sample_rate=_SAMPLE_RATE, frames=_FRAMES)

    np.testing.assert_array_equal(pool.take(0), audio[0, 0, :_FRAMES])


def test_input_audio_pool_shorter_row_zero_pads_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    audio = np.full((1, 1, 5), 0.25, dtype=np.float32)
    txid = _write_source(source_root, audio)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    pool = InputAudioPool(_source(source_root, txid), sample_rate=_SAMPLE_RATE, frames=_FRAMES)

    row = pool.take(0)
    assert row.shape == (_FRAMES,)
    np.testing.assert_array_equal(row[:5], np.full(5, 0.25, dtype=np.float32))
    np.testing.assert_array_equal(row[5:], np.zeros(_FRAMES - 5, dtype=np.float32))


def test_input_audio_pool_foreign_rate_resamples_to_target_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    audio = np.full((1, 1, 2205), 0.25, dtype=np.float32)
    txid = _write_source(source_root, audio, sample_rate=22_050)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    pool = InputAudioPool(_source(source_root, txid), sample_rate=_SAMPLE_RATE, frames=_FRAMES)

    row = pool.take(0)
    assert row.shape == (_FRAMES,)
    assert row.dtype == np.float32
    assert np.isfinite(row).all()
    # Loose: the soxr edge transient rings on the first samples of short excerpts.
    np.testing.assert_allclose(row, np.full(_FRAMES, 0.25, dtype=np.float32), atol=5e-2)


def test_input_audio_pool_resample_wires_rates_and_truncation_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import librosa

    source_root = tmp_path / "source"
    ramp = np.linspace(0.0, 1.0, 2205, dtype=np.float32).reshape(1, 1, 2205)
    txid = _write_source(source_root, ramp, sample_rate=22_050)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    pool = InputAudioPool(_source(source_root, txid), sample_rate=_SAMPLE_RATE, frames=_FRAMES)

    expected = librosa.resample(ramp[0], orig_sr=22_050, target_sr=_SAMPLE_RATE)[0, :_FRAMES]
    np.testing.assert_array_equal(pool.take(0), np.ascontiguousarray(expected, dtype=np.float32))


def test_input_audio_pool_stereo_row_downmixes_to_mono_mean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    audio = np.stack(
        [np.zeros(_FRAMES, dtype=np.float32), np.ones(_FRAMES, dtype=np.float32)]
    ).reshape(1, 2, _FRAMES)
    txid = _write_source(source_root, audio)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    pool = InputAudioPool(_source(source_root, txid), sample_rate=_SAMPLE_RATE, frames=_FRAMES)

    np.testing.assert_array_equal(pool.take(0), np.full(_FRAMES, 0.5, dtype=np.float32))


def test_input_audio_pool_matching_geometry_skips_adaptation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    audio = np.zeros((1, 1, _FRAMES), dtype=np.float32)
    txid = _write_source(source_root, audio)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    pool = InputAudioPool(_source(source_root, txid), sample_rate=_SAMPLE_RATE, frames=_FRAMES)

    assert pool.adapted is False
    assert pool.source_sample_rate == _SAMPLE_RATE
    assert (pool.source_channels, pool.source_frames) == (1, _FRAMES)


def test_input_audio_pool_foreign_geometry_reports_adaptation_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    audio = np.zeros((1, 1, 12), dtype=np.float32)
    txid = _write_source(source_root, audio, sample_rate=22_050)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    pool = InputAudioPool(_source(source_root, txid), sample_rate=_SAMPLE_RATE, frames=_FRAMES)

    assert pool.adapted is True
    provenance = pool.adaptation_provenance()
    assert provenance["source_sample_rate"] == 22_050
    assert (provenance["source_channels"], provenance["source_frames"]) == (1, 12)
    assert (provenance["target_sample_rate"], provenance["target_frames"]) == (
        _SAMPLE_RATE,
        _FRAMES,
    )
    assert provenance["policy"] == INPUT_AUDIO_ADAPTATION_POLICY


def test_input_audio_pool_foreign_frame_count_adapts_instead_of_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    audio = np.zeros((1, 1, _FRAMES - 1), dtype=np.float32)
    txid = _write_source(source_root, audio)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    pool = InputAudioPool(_source(source_root, txid), sample_rate=_SAMPLE_RATE, frames=_FRAMES)

    assert pool.adapted is True
    assert pool.take(0).shape == (_FRAMES,)


def test_input_audio_pool_foreign_sample_rate_adapts_instead_of_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    audio = np.zeros((1, 1, _FRAMES), dtype=np.float32)
    txid = _write_source(source_root, audio, sample_rate=48_000)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    pool = InputAudioPool(_source(source_root, txid), sample_rate=_SAMPLE_RATE, frames=_FRAMES)

    assert pool.adapted is True
    assert pool.take(0).shape == (_FRAMES,)


def test_input_audio_pool_nonfinite_row_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    audio = np.zeros((1, 1, _FRAMES), dtype=np.float32)
    audio[0, 0, 3] = np.nan
    txid = _write_source(source_root, audio)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    pool = InputAudioPool(_source(source_root, txid), sample_rate=_SAMPLE_RATE, frames=_FRAMES)

    with pytest.raises(ValueError, match="finite"):
        pool.take(0)
