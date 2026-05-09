"""Tests for ``scripts/replay_h5_to_audio_pairs.py``.

Pure / fast tests cover the on-disk pair-write contract; slow VST e2e tests
prove the orchestrator produces output that ``compute_audio_metrics.py``
consumes and scores within replay thresholds.
"""

from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  side-effect: registers Blosc2 filter for h5py reads
import numpy as np
import pytest
from pedalboard.io import AudioFile

from scripts.compute_audio_metrics import compute_metrics_on_dir, find_possible_subdirs
from scripts.replay_h5_to_audio_pairs import _write_pair, replay_h5_to_audio_pairs
from src.data.vst import param_specs
from src.data.vst.generate_vst_dataset import load_fixed_params_from_h5, make_dataset
from src.data.vst.param_spec import ParamSpec
from tests.data.vst.test_generate_vst_dataset import (
    _CHANNELS,
    _DURATION,
    _MIN_LOUDNESS,
    _MSS_MAX,
    _NUM_SAMPLES,
    _PLUGIN_PATH,
    _PRESET_PATH,
    _RMS_MIN_COSINE,
    _SAMPLE_RATE,
    _SOT_MAX,
    _SPEC_NAME,
    _VELOCITY,
    _WMFCC_MAX,
    _assert_h5_structure_is_valid,
    _assert_round_trip_matches,
    skip_no_vst,
)


def _stereo_sine(frequency: float, duration_seconds: float, sample_rate: float) -> np.ndarray:
    """Deterministic stereo float32 audio in ``(channels, frames)`` layout."""
    n = int(sample_rate * duration_seconds)
    t = np.arange(n) / sample_rate
    left = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    right = (0.5 * np.sin(2 * np.pi * (frequency * 1.5) * t)).astype(np.float32)
    return np.stack([left, right], axis=0)


def test_write_pair_creates_subdir_recognized_by_find_possible_subdirs(tmp_path: Path) -> None:
    """``_write_pair`` output is consumable by ``compute_audio_metrics.find_possible_subdirs``.

    Pins the contract — both files exist under a per-sample subdir — not the
    label-parsing detail (``rsplit("_", 1)[1]``) inside compute_audio_metrics.
    """

    target = _stereo_sine(440.0, 0.1, 44100.0)
    pred = _stereo_sine(880.0, 0.1, 44100.0)

    _write_pair(
        out_dir=tmp_path,
        idx=0,
        target=target,
        pred=pred,
        sample_rate=44100.0,
        channels=2,
    )

    subdirs = find_possible_subdirs(tmp_path)
    assert len(subdirs) == 1
    assert (subdirs[0] / "target.wav").exists()
    assert (subdirs[0] / "pred.wav").exists()


def test_write_pair_round_trips_audio_via_audio_file(tmp_path: Path) -> None:
    """Read both WAVs back; sample_rate, channels, frame count, and content match inputs.

    Pins the (channels, frames) → (frames, channels) transpose at write time and the inverse at
    read time — the convention that bites if forgotten.
    """

    target = _stereo_sine(440.0, 0.1, 44100.0)
    pred = _stereo_sine(880.0, 0.1, 44100.0)

    sample_dir = _write_pair(
        out_dir=tmp_path,
        idx=0,
        target=target,
        pred=pred,
        sample_rate=44100.0,
        channels=2,
    )

    with AudioFile(str(sample_dir / "target.wav")) as f:
        assert f.samplerate == 44100.0
        assert f.num_channels == 2
        assert f.frames == target.shape[1]
        target_read = f.read(f.frames)
    with AudioFile(str(sample_dir / "pred.wav")) as f:
        pred_read = f.read(f.frames)

    assert target_read.shape == target.shape
    assert pred_read.shape == pred.shape
    assert target_read.dtype == np.float32
    assert pred_read.dtype == np.float32
    np.testing.assert_allclose(target_read, target, atol=1e-3)
    np.testing.assert_allclose(pred_read, pred, atol=1e-3)


def test_write_pair_overwrites_existing_files_idempotently(tmp_path: Path) -> None:
    """Calling ``_write_pair`` twice at the same idx leaves the second call's audio on disk."""

    first_target = _stereo_sine(440.0, 0.1, 44100.0)
    first_pred = _stereo_sine(880.0, 0.1, 44100.0)
    second_target = _stereo_sine(220.0, 0.1, 44100.0)
    second_pred = _stereo_sine(660.0, 0.1, 44100.0)

    _write_pair(tmp_path, 0, first_target, first_pred, 44100.0, 2)
    sample_dir = _write_pair(tmp_path, 0, second_target, second_pred, 44100.0, 2)

    with AudioFile(str(sample_dir / "target.wav")) as f:
        target_read = f.read(f.frames)
    with AudioFile(str(sample_dir / "pred.wav")) as f:
        pred_read = f.read(f.frames)

    assert target_read.shape == second_target.shape
    assert pred_read.shape == second_pred.shape
    np.testing.assert_allclose(target_read, second_target, atol=1e-3)
    np.testing.assert_allclose(pred_read, second_pred, atol=1e-3)


def _write_synthetic_h5(
    path: Path,
    num_rows: int,
    spec: ParamSpec,
    sample_rate: float = _SAMPLE_RATE,
    channels: int = _CHANNELS,
    duration: float = _DURATION,
) -> None:
    """Write a zero-filled h5 with the schema replay_h5_to_audio_pairs reads.

    Used to drive the validation path without invoking the VST renderer.
    """
    n_frames = int(sample_rate * duration)
    with h5py.File(path, "w") as f:
        audio = f.create_dataset(
            "audio",
            shape=(num_rows, channels, n_frames),
            dtype=np.float16,
        )
        audio.attrs["sample_rate"] = sample_rate
        audio.attrs["channels"] = channels
        audio.attrs["signal_duration_seconds"] = duration
        audio.attrs["velocity"] = _VELOCITY
        audio.attrs["min_loudness"] = _MIN_LOUDNESS
        f.create_dataset(
            "param_array",
            shape=(num_rows, len(spec)),
            dtype=np.float32,
        )


def test_replay_raises_when_num_samples_exceeds_h5_rows(tmp_path: Path) -> None:
    """``num_samples`` larger than the h5's row count raises before any render is attempted."""
    spec = param_specs[_SPEC_NAME]
    h5_path = tmp_path / "tiny.h5"
    _write_synthetic_h5(h5_path, num_rows=3, spec=spec)

    with pytest.raises(ValueError, match="exceeds h5 row count"):
        replay_h5_to_audio_pairs(
            h5_path=h5_path,
            output_dir=tmp_path / "audio-pairs",
            plugin_path=_PLUGIN_PATH,
            preset_path=_PRESET_PATH,
            param_spec=spec,
            num_samples=5,
        )


def test_load_fixed_params_from_h5_max_rows_decodes_only_first_n(tmp_path: Path) -> None:
    """``max_rows=N`` decodes exactly N rows, even when the h5 has more."""
    spec = param_specs[_SPEC_NAME]
    h5_path = tmp_path / "shard.h5"
    _write_synthetic_h5(h5_path, num_rows=5, spec=spec)

    synth_list, note_list = load_fixed_params_from_h5(str(h5_path), spec, max_rows=2)

    assert len(synth_list) == 2
    assert len(note_list) == 2


def test_load_fixed_params_from_h5_raises_when_max_rows_exceeds_h5_rows(tmp_path: Path) -> None:
    """``max_rows`` larger than the h5's row count raises before any decode runs."""
    spec = param_specs[_SPEC_NAME]
    h5_path = tmp_path / "shard.h5"
    _write_synthetic_h5(h5_path, num_rows=3, spec=spec)

    with pytest.raises(ValueError, match="exceeds h5 param_array row count"):
        load_fixed_params_from_h5(str(h5_path), spec, max_rows=5)


@pytest.mark.slow
@pytest.mark.requires_vst
@skip_no_vst
def test_replay_writes_only_first_n_pairs_when_num_samples_specified(tmp_path: Path) -> None:
    """``num_samples=N`` writes exactly N pairs even when the h5 has more rows."""
    spec = param_specs[_SPEC_NAME]

    h5_path = tmp_path / "candidates.h5"
    with h5py.File(h5_path, "a") as f:
        make_dataset(
            hdf5_file=f,
            num_samples=_NUM_SAMPLES,
            plugin_path=_PLUGIN_PATH,
            preset_path=_PRESET_PATH,
            sample_rate=_SAMPLE_RATE,
            channels=_CHANNELS,
            velocity=_VELOCITY,
            signal_duration_seconds=_DURATION,
            min_loudness=_MIN_LOUDNESS,
            param_spec=spec,
            sample_batch_size=_NUM_SAMPLES,
        )

    out_dir = tmp_path / "audio-pairs"
    limited_n = _NUM_SAMPLES - 2
    pairs_written = replay_h5_to_audio_pairs(
        h5_path=h5_path,
        output_dir=out_dir,
        plugin_path=_PLUGIN_PATH,
        preset_path=_PRESET_PATH,
        param_spec=spec,
        num_samples=limited_n,
    )

    assert pairs_written == limited_n
    subdirs = find_possible_subdirs(out_dir)
    assert len(subdirs) == limited_n


@pytest.mark.slow
@pytest.mark.requires_vst
@skip_no_vst
def test_replay_h5_to_audio_pairs_writes_one_pair_per_row(tmp_path: Path) -> None:
    """``replay_h5_to_audio_pairs`` writes exactly one ``find_possible_subdirs``-recognized pair
    per h5 row."""

    spec = param_specs[_SPEC_NAME]

    h5_path = tmp_path / "candidates.h5"
    with h5py.File(h5_path, "a") as f:
        make_dataset(
            hdf5_file=f,
            num_samples=_NUM_SAMPLES,
            plugin_path=_PLUGIN_PATH,
            preset_path=_PRESET_PATH,
            sample_rate=_SAMPLE_RATE,
            channels=_CHANNELS,
            velocity=_VELOCITY,
            signal_duration_seconds=_DURATION,
            min_loudness=_MIN_LOUDNESS,
            param_spec=spec,
            sample_batch_size=_NUM_SAMPLES,
        )

    out_dir = tmp_path / "audio-pairs"
    pairs_written = replay_h5_to_audio_pairs(
        h5_path=h5_path,
        output_dir=out_dir,
        plugin_path=_PLUGIN_PATH,
        preset_path=_PRESET_PATH,
        param_spec=spec,
    )

    assert pairs_written == _NUM_SAMPLES
    subdirs = find_possible_subdirs(out_dir)
    assert len(subdirs) == _NUM_SAMPLES


@pytest.mark.slow
@pytest.mark.requires_vst
@skip_no_vst
def test_replayed_h5_matches_input_h5_within_phase_robust_tolerances(tmp_path: Path) -> None:
    """The replayed h5 reproduces the input h5's audio/mel/params within project tolerances.

    Reuses ``_assert_round_trip_matches`` from the canonical generate_vst_dataset
    test module — the same helper that pins the in-memory replay round-trip.
    """
    spec = param_specs[_SPEC_NAME]

    h5_path = tmp_path / "candidates.h5"
    with h5py.File(h5_path, "a") as f:
        make_dataset(
            hdf5_file=f,
            num_samples=_NUM_SAMPLES,
            plugin_path=_PLUGIN_PATH,
            preset_path=_PRESET_PATH,
            sample_rate=_SAMPLE_RATE,
            channels=_CHANNELS,
            velocity=_VELOCITY,
            signal_duration_seconds=_DURATION,
            min_loudness=_MIN_LOUDNESS,
            param_spec=spec,
            sample_batch_size=_NUM_SAMPLES,
        )

    out_dir = tmp_path / "audio-pairs"
    replay_h5_to_audio_pairs(
        h5_path=h5_path,
        output_dir=out_dir,
        plugin_path=_PLUGIN_PATH,
        preset_path=_PRESET_PATH,
        param_spec=spec,
    )
    replayed_h5 = out_dir / "replayed.h5"

    expected_audio, expected_mel, expected_params = _assert_h5_structure_is_valid(
        h5_path, spec, _NUM_SAMPLES
    )
    actual_audio, actual_mel, actual_params = _assert_h5_structure_is_valid(
        replayed_h5, spec, _NUM_SAMPLES
    )
    expected_synth_patches, expected_note_patches = load_fixed_params_from_h5(str(h5_path), spec)

    _assert_round_trip_matches(
        actual_audio=actual_audio,
        actual_mel=actual_mel,
        actual_params=actual_params,
        expected_audio=expected_audio,
        expected_mel=expected_mel,
        expected_params=expected_params,
        expected_synth_patches=expected_synth_patches,
        expected_note_patches=expected_note_patches,
        spec=spec,
        num_samples=_NUM_SAMPLES,
    )


@pytest.mark.slow
@pytest.mark.requires_vst
@skip_no_vst
def test_replay_audio_metrics_within_replay_thresholds(tmp_path: Path) -> None:
    """``compute_metrics_on_dir`` of every replay-pair scores within the in-memory replay
    thresholds."""

    spec = param_specs[_SPEC_NAME]

    h5_path = tmp_path / "candidates.h5"
    with h5py.File(h5_path, "a") as f:
        make_dataset(
            hdf5_file=f,
            num_samples=_NUM_SAMPLES,
            plugin_path=_PLUGIN_PATH,
            preset_path=_PRESET_PATH,
            sample_rate=_SAMPLE_RATE,
            channels=_CHANNELS,
            velocity=_VELOCITY,
            signal_duration_seconds=_DURATION,
            min_loudness=_MIN_LOUDNESS,
            param_spec=spec,
            sample_batch_size=_NUM_SAMPLES,
        )

    out_dir = tmp_path / "audio-pairs"
    replay_h5_to_audio_pairs(
        h5_path=h5_path,
        output_dir=out_dir,
        plugin_path=_PLUGIN_PATH,
        preset_path=_PRESET_PATH,
        param_spec=spec,
    )

    subdirs = find_possible_subdirs(out_dir)
    assert len(subdirs) == _NUM_SAMPLES
    for subdir in subdirs:
        metrics = compute_metrics_on_dir(subdir)
        assert all(np.isfinite(v) for v in metrics.values()), f"{subdir.name}: {metrics}"
        assert metrics["mss"] < _MSS_MAX, f"{subdir.name}: mss={metrics['mss']:.4f}"
        assert metrics["wmfcc"] < _WMFCC_MAX, f"{subdir.name}: wmfcc={metrics['wmfcc']:.4f}"
        assert metrics["sot"] < _SOT_MAX, f"{subdir.name}: sot={metrics['sot']:.4f}"
        assert metrics["rms"] > _RMS_MIN_COSINE, f"{subdir.name}: rms={metrics['rms']:.4f}"
