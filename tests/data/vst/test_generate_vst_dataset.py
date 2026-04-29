"""Basic e2e test for src/data/vst/generate_vst_dataset.py — verifies HDF5 output."""

import logging
import os
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  # pyright: ignore[reportUnusedImport]  side-effect: registers Blosc2 filter for h5py reads
import numpy as np
import pytest
from rich import print

from scripts.compute_audio_metrics import compute_mss, compute_rms, compute_sot, compute_wmfcc
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

# Phase-robust audio similarity thresholds for fixed-params replay vs. candidates.
# Two independent renders of identical params differ at the sample level (Surge XT's
# oscillator phase init is nondeterministic across the per-call plugin reloads in
# ``render_params``), but should remain perceptually close. Tune downward if the
# metrics consistently come in tighter than these caps.
_MSS_MAX = 5.0           # multi-scale log-mel L1 distance (dB)
_WMFCC_MAX = 5.0         # DTW-aligned MFCC L1 distance
_SOT_MAX = 0.05          # spectral optimal transport (Wasserstein on STFT mags)
_RMS_MIN_COSINE = 0.95   # RMS envelope cosine similarity (1.0 = identical)
_MEL_MEAN_ABS_MAX = 5.0  # mean abs diff on stored mel_spec (log-power dB)

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
    """make_dataset with fixed_*_params_list reproduces a previous dataset within audio metrics.

    Two-stage e2e test:

    1. Build a "candidates" dataset by calling ``make_dataset`` *without* any
       ``fixed_*_params_list``. Internally ``generate_sample`` samples params via
       ``param_spec.sample()`` and rejects renders below ``_MIN_LOUDNESS`` in a
       ``while True`` loop. Each surviving row is therefore guaranteed to be
       loud enough to pass the loudness gate.
    2. Decode the candidate ``param_array`` rows back into ``synth_patches`` /
       ``note_patches`` and feed those into a second ``make_dataset`` call as
       ``fixed_synth_params_list`` / ``fixed_note_params_list``. Because the
       params are guaranteed-loud (step 1), this run can't infinite-loop on the
       loudness gate the way it would for arbitrary fixed params.

    Assertions:

    - ``param_array`` matches the candidates exactly within float32 numeric
      tolerance — params are deterministic.
    - Per-row audio matches within phase-robust perceptual metrics (MSS,
      wMFCC, SOT, RMS-envelope cosine). Element-wise equality is *not* a
      property of the system: ``render_params`` reloads Surge XT per call to
      work around the silent-output bug (commits 086d80f, 9ff7f16), and each
      reload yields a different oscillator phase init, so the two renders of
      the same params are phase-shifted variants of the same waveform.
    - Mel spec matches within mean absolute log-power error.
    """
    spec = param_specs[_SPEC_NAME]

    # Stage 1: random-sampled "candidates" dataset (loudness-filtered).
    expected_dataset = tmp_path / "candidates.h5"
    with h5py.File(expected_dataset, "a") as expected_file:
        make_dataset(
            hdf5_file=expected_file,
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

    expected_audio, expected_mel, expected_params = _assert_h5_structure_is_valid(
        expected_dataset, spec, _NUM_SAMPLES
    )

    log.info(
        "spec %s synth_len=%d note_len=%d",
        spec.names,
        spec.synth_param_length,
        spec.note_param_length,
    )

    # Decode the candidate rows back into synth/note params dicts. These are the
    # inputs for the second ``make_dataset`` run — guaranteed past the loudness
    # gate by construction (the candidate render survived stage 1).
    synth_patches: list[dict[str, float]] = []
    note_patches: list[dict[str, float]] = []
    for i in range(_NUM_SAMPLES):
        decoded_synth_params, decoded_note_params = spec.decode(expected_params[i])
        synth_patches.append(decoded_synth_params)
        note_patches.append(decoded_note_params)
    print("synth_patches", synth_patches)
    print("note_patches", note_patches)

    # Stage 2: replay the candidates as fixed inputs and verify reproducibility.
    got_dataset = tmp_path / "fixed.h5"
    with h5py.File(got_dataset, "a") as f:
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

    actual_audio, actual_mel, actual_params = _assert_h5_structure_is_valid(
        got_dataset, spec, _NUM_SAMPLES
    )

    # Encoded param rows must round-trip exactly to bit-precision float32.
    np.testing.assert_allclose(
        actual_params,
        expected_params,
        rtol=_RELATIVE_TOLERANCE,
        atol=_ABSOLUTE_TOLERANCE,
        err_msg="param_array does not match candidates row-for-row",
    )

    # Per-row audio: phase-robust metrics. Element-wise equality fails because
    # Surge XT's oscillator phase init is nondeterministic across the per-call
    # plugin reloads in ``render_params``; two renders of identical params are
    # phase-shifted variants of the same waveform.
    for i in range(_NUM_SAMPLES):
        target = expected_audio[i]
        pred = actual_audio[i]
        mss = compute_mss(target, pred)
        wmfcc = compute_wmfcc(target, pred)
        sot = compute_sot(target, pred)
        rms_cos = compute_rms(target, pred)
        log.info(
            "sample %d audio metrics: mss=%.4f wmfcc=%.4f sot=%.4f rms_cos=%.4f",
            i,
            mss,
            wmfcc,
            sot,
            rms_cos,
        )
        assert mss < _MSS_MAX, f"sample {i}: mss={mss:.4f} exceeds {_MSS_MAX}"
        assert wmfcc < _WMFCC_MAX, f"sample {i}: wmfcc={wmfcc:.4f} exceeds {_WMFCC_MAX}"
        assert sot < _SOT_MAX, f"sample {i}: sot={sot:.4f} exceeds {_SOT_MAX}"
        assert rms_cos > _RMS_MIN_COSINE, (
            f"sample {i}: rms cosine similarity {rms_cos:.4f} below {_RMS_MIN_COSINE}"
        )

    # Stored mel_spec rows: simple mean abs diff (log-power dB). Same reasoning
    # as the audio metrics — phase drift dominates element-wise comparison.
    mel_dist = float(np.mean(np.abs(actual_mel - expected_mel)))
    log.info("mel mean abs diff: %.4f (max=%.4f)", mel_dist, _MEL_MEAN_ABS_MAX)
    assert mel_dist < _MEL_MEAN_ABS_MAX, (
        f"mel mean abs diff {mel_dist:.4f} exceeds {_MEL_MEAN_ABS_MAX}"
    )

    # Decoded params should match the inputs we fed in within the same tight
    # numeric tolerance used for the encoded array.
    for i in range(_NUM_SAMPLES):
        decoded_synth_params, decoded_note_params = spec.decode(actual_params[i])
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
