"""Basic e2e test for src/data/vst/generate_vst_dataset.py — verifies HDF5 output."""

import logging
import os
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  # pyright: ignore[reportUnusedImport]  side-effect: registers Blosc2 filter for h5py reads
import numpy as np
import pytest

from src.data.vst import param_specs
from src.data.vst.generate_vst_dataset import make_dataset

log = logging.getLogger(__name__)

_PLUGIN_PATH = os.environ.get("SYNTH_SETTER_PLUGIN_PATH") or "plugins/Surge XT.vst3"
_PRESET_PATH = "presets/surge-base.vstpreset"
_NUM_SAMPLES = 5
_SAMPLE_RATE = 44100.0
_CHANNELS = 2
_DURATION = 4.0
_VELOCITY = 100
_MIN_LOUDNESS = -55.0
_SPEC_NAME = "surge_xt"
_ABSOLUTE_TOLERANCE = 1e-7
_RELATIVE_TOLERANCE = 1e-9

# Per-sample mel shape `(channels, n_mels, n_frames)` hardcoded by the writer.
# Derivation: channels=2 literal in writer; n_mels=128 from librosa kwarg;
# n_frames = _DURATION * 100 + 1 (hop_length = sample_rate/100 → 100 fps + librosa's
# trailing frame). If _CHANNELS, _DURATION, librosa kwargs, or the writer literal
# change, update this constant.
# Pointers:
#   - `create_datasets_and_get_start_idx()` in `src/data/vst/generate_vst_dataset.py`
#     (the literal `(num_samples, 2, 128, 401)` passed to `create_dataset`)
#   - `make_spectrogram()` in `src/data/vst/generate_vst_dataset.py`
#   - `_SURGE_MEL_SHAPE` in `tests/conftest.py` — mirror, keep in sync.
_PER_SAMPLE_MEL_SHAPE = (2, 128, 401)

skip_no_vst = pytest.mark.skipif(
    not Path(_PLUGIN_PATH).exists(),
    reason=f"VST plugin not found at {_PLUGIN_PATH}",
)


def _assert_h5_structure_is_valid(
    out: Path, spec, num_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open ``out`` and assert dataset keys, shapes, dtypes, attrs, and finiteness.

    Returns materialized (audio, mel_spec, param_array) numpy arrays so callers can perform per-row
    decode checks without re-opening the file. Shared by the fixed-params and random-sampling e2e
    tests.
    """
    expected_audio_shape = (num_samples, _CHANNELS, int(_SAMPLE_RATE * _DURATION))
    expected_mel_shape = (num_samples, *_PER_SAMPLE_MEL_SHAPE)

    with h5py.File(out, "r") as f:
        for name in ("audio", "mel_spec", "param_array"):
            assert name in f, f"missing dataset {name!r}"

        audio = f["audio"]
        mel = f["mel_spec"]
        params = f["param_array"]
        assert isinstance(audio, h5py.Dataset)
        assert isinstance(mel, h5py.Dataset)
        assert isinstance(params, h5py.Dataset)

        assert audio.shape == expected_audio_shape
        assert mel.shape == expected_mel_shape
        assert params.shape == (num_samples, len(spec))
        assert params.shape[1] == spec.synth_param_length + spec.note_param_length
        assert spec.note_param_length == 3  # pitch (1) + note_start_and_end (2)

        assert audio.dtype == np.float16
        assert mel.dtype == np.float32
        assert params.dtype == np.float32

        assert audio.attrs["velocity"] == _VELOCITY
        assert audio.attrs["sample_rate"] == _SAMPLE_RATE
        assert audio.attrs["channels"] == _CHANNELS
        assert audio.attrs["signal_duration_seconds"] == _DURATION
        assert audio.attrs["min_loudness"] == _MIN_LOUDNESS

        audio_arr = audio[...].astype(np.float32)
        assert np.isfinite(audio_arr).all()
        assert np.isfinite(mel[...]).all()
        assert np.isfinite(params[...]).all()

        peak = np.abs(audio_arr).reshape(num_samples, -1).max(axis=1)
        assert (peak > 1e-4).all(), f"silent clips: peaks={peak.tolist()}"

        return audio_arr, mel[...], params[...]


@pytest.mark.slow
@pytest.mark.requires_vst
@skip_no_vst
def test_make_dataset_with_fixed_params_round_trips_per_row(tmp_path: Path) -> None:
    """make_dataset with distinct fixed_*_params_list per row recovers each row's input on
    decode."""
    out = tmp_path / "fixed.h5"
    spec = param_specs[_SPEC_NAME]
    log.info("spec %s synth_len=%d note_len=%d", spec.names, spec.synth_param_length,
             spec.note_param_length)

    synth_patches: list[dict[str, float]] = []
    note_patches: list[dict[str, float]] = []
    for _ in range(_NUM_SAMPLES):
        s, n = spec.sample()
        synth_patches.append(s)
        note_patches.append(n)

    with h5py.File(out, "a") as f:
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
            fixed_synth_params_list=synth_patches,
            fixed_note_params_list=note_patches,
        )

    _, _, params = _assert_h5_structure_is_valid(out, spec, _NUM_SAMPLES)

    for i in range(_NUM_SAMPLES):
        decoded_synth_params, decoded_note_params = spec.decode(params[i])
        assert isinstance(decoded_synth_params, dict)
        assert decoded_synth_params == pytest.approx(
            synth_patches[i], rel=_RELATIVE_TOLERANCE, abs=_ABSOLUTE_TOLERANCE
        ), (
            f"sample {i}: decoded synth params {decoded_synth_params} do not match input "
            f"{synth_patches[i]} within tolerances (abs={_ABSOLUTE_TOLERANCE}, "
            f"rel={_RELATIVE_TOLERANCE})"
        )
        assert decoded_note_params == pytest.approx(
            note_patches[i], rel=_RELATIVE_TOLERANCE, abs=_ABSOLUTE_TOLERANCE
        ), (
            f"sample {i}: decoded note params {decoded_note_params} do not match input "
            f"{note_patches[i]} within tolerances (abs={_ABSOLUTE_TOLERANCE}, "
            f"rel={_RELATIVE_TOLERANCE})"
        )
        assert isinstance(decoded_note_params, dict)
        assert decoded_note_params.keys() == {"pitch", "note_start_and_end"}
        assert isinstance(decoded_note_params["pitch"], int)
        assert isinstance(decoded_note_params["note_start_and_end"], tuple)
        start, end = decoded_note_params["note_start_and_end"]
        assert isinstance(start, np.floating)
        assert isinstance(end, np.floating)


@pytest.mark.slow
@pytest.mark.requires_vst
@skip_no_vst
def test_make_dataset_with_random_sampling_produces_valid_h5(tmp_path: Path) -> None:
    """make_dataset without fixed_*_params_list samples params internally and writes a valid h5."""
    out = tmp_path / "random.h5"
    spec = param_specs[_SPEC_NAME]

    with h5py.File(out, "a") as f:
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

    _, _, params = _assert_h5_structure_is_valid(out, spec, _NUM_SAMPLES)

    decoded_rows = [spec.decode(params[i]) for i in range(_NUM_SAMPLES)]
    for synth, note in decoded_rows:
        assert isinstance(synth, dict)
        assert isinstance(note, dict)
        assert note.keys() == {"pitch", "note_start_and_end"}
