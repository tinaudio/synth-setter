"""Basic e2e test for src/data/vst/generate_vst_dataset.py — verifies HDF5 output."""

import os
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  # pyright: ignore[reportUnusedImport]  side-effect: registers Blosc2 filter for h5py reads
import numpy as np
import pytest
import logging
log = logging.getLogger(__name__)
from rich import print   # shadows builtin print


from src.data.vst import param_specs
from src.data.vst.generate_vst_dataset import make_dataset

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


@pytest.mark.slow
@pytest.mark.requires_vst
@skip_no_vst
def test_make_dataset_writes_valid_h5(tmp_path: Path) -> None:
    """make_dataset produces an h5 with correct datasets, shapes, attrs, and finite samples."""
    out = tmp_path / "data.h5"
    spec = param_specs[_SPEC_NAME]
    log.info(spec.names, spec.synth_param_length, spec.note_param_length)
    SYNTH_PATCHES = []
    NOTE_PATCHES = []
    single_synth_patch, single_set_note_params = spec.sample()
    for i in range(_NUM_SAMPLES):
        # patch, _ = spec.sample()
        # for param_name, param_value in patch.items():
        #     patch[param_name] = float(0)
        # print(f"Sampled synth params for sample {i}: {patch}")

        SYNTH_PATCHES.append(single_synth_patch)
        NOTE_PATCHES.append(single_set_note_params)

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
            fixed_synth_params_list=SYNTH_PATCHES,
            fixed_note_params_list=NOTE_PATCHES,
        )

    expected_audio_shape = (_NUM_SAMPLES, _CHANNELS, int(_SAMPLE_RATE * _DURATION))
    expected_mel_shape = (_NUM_SAMPLES, *_PER_SAMPLE_MEL_SHAPE)

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
        assert params.shape == (_NUM_SAMPLES, len(spec))

        log.info(
            f"params dtype: {params.dtype}, param.shape: {params.shape}, params[0].shape: {params[0].shape}, params[0]: {params[0]}, param[0][0]: {params[0][0]}"
        )
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

        peak = np.abs(audio_arr).reshape(_NUM_SAMPLES, -1).max(axis=1)
        assert (peak > 1e-4).all(), f"silent clips: peaks={peak.tolist()}"

        for i in range(_NUM_SAMPLES):
            encoded = params[i]
            decoded_synth_params, decoded_note_params = spec.decode(encoded)
            assert isinstance(decoded_synth_params, dict)
            original_synth_params = SYNTH_PATCHES[i]
            assert decoded_synth_params == pytest.approx(
                original_synth_params, rel=_RELATIVE_TOLERANCE, abs=_ABSOLUTE_TOLERANCE
            ), (
                f"decoded synth param `{name}` value {decoded_synth_params} does not match original {original_synth_params} within tolerances (abs={_ABSOLUTE_TOLERANCE}, rel={_RELATIVE_TOLERANCE})"
            )
            assert decoded_note_params == pytest.approx(
                NOTE_PATCHES[i], rel=_RELATIVE_TOLERANCE, abs=_ABSOLUTE_TOLERANCE
            ), (
                f"decoded note params {decoded_note_params} do not match original {NOTE_PATCHES[i]} within tolerances (abs={_ABSOLUTE_TOLERANCE}, rel={_RELATIVE_TOLERANCE})"
            )

            log.info(f"Decoded synth params for sample {i}: {decoded_synth_params}")
            assert isinstance(decoded_note_params, dict)
            log.info(f"Decoded note params for sample {i}: {decoded_note_params}")
            assert decoded_note_params.keys() == {"pitch", "note_start_and_end"}
            assert isinstance(decoded_note_params["pitch"], int)
            assert isinstance(decoded_note_params["note_start_and_end"], tuple)
            log.info(f"type of note_start_and_end: {type(decoded_note_params['note_start_and_end'])}")
            start, end = decoded_note_params["note_start_and_end"]
            log.info(f"type of start: {type(start)}, type of end: {type(end)}")
            assert isinstance(start, np.floating)
            assert isinstance(end, np.floating)
