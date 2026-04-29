"""Basic e2e test for src/data/vst/generate_vst_dataset.py — verifies HDF5 output."""

import logging
import os
from pathlib import Path
from unittest.mock import patch

import h5py
import hdf5plugin  # noqa: F401  # pyright: ignore[reportUnusedImport]  side-effect: registers Blosc2 filter for h5py reads
import numpy as np
import pytest

from scripts.compute_audio_metrics import compute_mss, compute_rms, compute_sot, compute_wmfcc
from src.data.vst import param_specs
from src.data.vst.generate_vst_dataset import make_dataset
from src.data.vst.param_spec import ParamSpec

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
_MSS_MAX = 10.0           # multi-scale log-mel L1 distance (dB)
_WMFCC_MAX = 18.0         # DTW-aligned MFCC L1 distance
_SOT_MAX = 0.15          # spectral optimal transport (Wasserstein on STFT mags)
_RMS_MIN_COSINE = 0.8   # RMS envelope cosine similarity (1.0 = identical)
_MEL_MEAN_ABS_MAX = 5.0  # mean abs diff on stored mel_spec (log-power dB)

# Peak amplitude floor below which a clip is treated as silent.
_AUDIO_PEAK_SILENCE_FLOOR = 1e-4

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

# Known-good loudness-passing patch captured from a prior random-sampled run
# (sample 4 of a 5-sample candidates dataset for the surge_xt spec). Used by
# ``test_make_dataset_replays_via_param_spec_sample_mock_with_hardcoded_params``
# to drive both stages with a fixed, single-sample param set rather than
# random sampling. If the spec changes (keys added/removed), this dict must
# be regenerated; values can be edited freely as long as the resulting render
# clears ``_MIN_LOUDNESS``.
_HARDCODED_SYNTH_PARAMS: dict[str, float] = {
    "a_amp_eg_attack": 0.6115446960926056,
    "a_amp_eg_decay": 0.7035384780168533,
    "a_amp_eg_sustain": 0.8458655476570129,
    "a_amp_eg_release": 0.7526835530996323,
    "a_amp_eg_envelope_mode": 0.75,
    "a_filter_eg_attack": 0.6334269267320634,
    "a_filter_eg_decay": 0.498066001534462,
    "a_filter_eg_sustain": 0.20409274101257324,
    "a_filter_eg_release": 0.2183585774898529,
    "a_filter_eg_envelope_mode": 0.75,
    "a_feedback": 0.5,
    "a_filter_balance": 0.5,
    "a_filter_configuration": 0.7125,
    "a_highpass": 0.8109021186828613,
    "a_filter_1_cutoff": 0.18654786050319672,
    "a_filter_1_feg_mod_amount": 0.6178140640258789,
    "a_filter_1_resonance": 0.9254387617111206,
    "a_filter_1_type": 0.6955,
    "a_filter_2_cutoff": 0.222905233502388,
    "a_filter_2_feg_mod_amount": 0.5447665452957153,
    "a_filter_2_resonance": 0.4141045808792114,
    "a_filter_2_type": 0.2455,
    "a_waveshaper_drive": 0.5353075826990605,
    "a_waveshaper_type": 0.1,
    "a_osc_1_mute": 0.2505,
    "a_osc_1_octave": 0.5005,
    "a_osc_1_pitch": 0.27579981088638306,
    "a_osc_1_route": 0.874,
    "a_osc_1_sawtooth": 0.8939391374588013,
    "a_osc_1_width": 0.7921043634414673,
    "a_osc_1_sync": 0.008044014684855938,
    "a_osc_1_unison_detune": 0.2343703955411911,
    "a_osc_1_unison_voices": 0.0195,
    "a_osc_1_volume": 0.22278083860874176,
    "a_osc_1_pulse": 0.6516783237457275,
    "a_osc_1_triangle": 0.24101869761943817,
    "a_osc_2_mute": 0.2505,
    "a_osc_2_octave": 0.5,
    "a_osc_2_pitch": 0.5,
    "a_osc_2_route": 0.1265,
    "a_osc_2_sawtooth": 0.7841401696205139,
    "a_osc_2_width": 0.651004433631897,
    "a_osc_2_sync": 0.8729211688041687,
    "a_osc_2_unison_detune": 0.9950850605964661,
    "a_osc_2_unison_voices": 0.0195,
    "a_osc_2_volume": 0.8628856539726257,
    "a_osc_2_pulse": 0.5585712194442749,
    "a_osc_2_triangle": 0.7885411381721497,
    "a_osc_3_mute": 0.2505,
    "a_osc_3_octave": 0.5,
    "a_osc_3_pitch": 0.8914749622344971,
    "a_osc_3_route": 0.874,
    "a_osc_3_sawtooth": 0.8178988695144653,
    "a_osc_3_width": 0.25304195284843445,
    "a_osc_3_sync": 0.32539647817611694,
    "a_osc_3_unison_detune": 0.5072322487831116,
    "a_osc_3_unison_voices": 0.0195,
    "a_osc_3_volume": 0.9149643778800964,
    "a_osc_3_pulse": 0.03317605331540108,
    "a_osc_3_triangle": 0.02597862109541893,
    "a_osc_drift": 0.9930819272994995,
    "a_fm_depth": 0.21858084201812744,
    "a_fm_routing": 0.9155,
    "a_lfo_1_amplitude": 0.0,
    "a_lfo_1_attack": 0.0022514958889223637,
    "a_lfo_1_decay": 0.2558778440952301,
    "a_lfo_1_deform": 0.31317904591560364,
    "a_lfo_1_hold": 0.5517071908712388,
    "a_lfo_1_phase": 0.6524000763893127,
    "a_lfo_1_rate": 0.45813828706741333,
    "a_lfo_1_release": 0.5639569038152695,
    "a_lfo_1_sustain": 0.7934410572052002,
    "a_lfo_1_type": 0.5549999999999999,
    "a_lfo_2_amplitude": 0.0,
    "a_lfo_2_attack": 0.2449902430176735,
    "a_lfo_2_decay": 0.004002517010085285,
    "a_lfo_2_deform": 0.6100139021873474,
    "a_lfo_2_hold": 0.1760928821563721,
    "a_lfo_2_phase": 0.7932401299476624,
    "a_lfo_2_rate": 0.09311769902706146,
    "a_lfo_2_release": 0.490568408370018,
    "a_lfo_2_sustain": 0.04693538323044777,
    "a_lfo_2_type": 0.5549999999999999,
    "a_lfo_3_amplitude": 0.0,
    "a_lfo_3_attack": 0.4771536821126938,
    "a_lfo_3_decay": 0.22750570356845856,
    "a_lfo_3_deform": 0.5475855469703674,
    "a_lfo_3_hold": 0.2858144730329514,
    "a_lfo_3_phase": 0.4254957139492035,
    "a_lfo_3_rate": 0.870485246181488,
    "a_lfo_3_release": 0.597478711605072,
    "a_lfo_3_sustain": 0.8878445029258728,
    "a_lfo_3_type": 0.3355,
    "a_lfo_4_amplitude": 0.011512083001434803,
    "a_lfo_4_attack": 0.2023567634820938,
    "a_lfo_4_decay": 0.7672159743309022,
    "a_lfo_4_deform": 0.22317861020565033,
    "a_lfo_4_hold": 0.2421818467974663,
    "a_lfo_4_phase": 0.17402644455432892,
    "a_lfo_4_rate": 0.32441940903663635,
    "a_lfo_4_release": 0.09115911923348904,
    "a_lfo_4_sustain": 0.8703923225402832,
    "a_lfo_4_type": 0.0305,
    "a_lfo_5_amplitude": 0.0,
    "a_lfo_5_attack": 0.7430108767747879,
    "a_lfo_5_decay": 0.70498992562294,
    "a_lfo_5_deform": 0.3357408046722412,
    "a_lfo_5_hold": 0.4660577839612961,
    "a_lfo_5_phase": 0.14568571746349335,
    "a_lfo_5_rate": 0.1889970600605011,
    "a_lfo_5_release": 0.19390373915433884,
    "a_lfo_5_sustain": 0.1149880513548851,
    "a_lfo_5_type": 0.0305,
    "a_lfo_6_amplitude": 0.0,
    "a_lfo_6_attack": 0.27028446555137636,
    "a_lfo_6_decay": 0.31658956825733187,
    "a_lfo_6_deform": 0.20451003313064575,
    "a_lfo_6_hold": 0.3530474078655243,
    "a_lfo_6_release": 0.5311448252201081,
    "a_lfo_6_sustain": 0.9014169573783875,
    "a_noise_color": 0.06865139305591583,
    "a_noise_mute": 0.2505,
    "a_noise_route": 0.5005,
    "a_noise_volume": 0.25137007236480713,
    "a_pan": 0.5,
    "a_ring_modulation_1x2_mute": 0.7505,
    "a_ring_modulation_1x2_route": 0.1265,
    "a_ring_modulation_1x2_volume": 0.05965404957532883,
    "a_ring_modulation_2x3_mute": 0.7505,
    "a_ring_modulation_2x3_route": 0.1265,
    "a_ring_modulation_2x3_volume": 0.39768943190574646,
    "a_vca_gain": 0.5358291573184729,
    "a_width": 0.793622612953186,
    "fx_a1_delay_time": 0.2910090684890747,
    "fx_a1_modulation_rate": 0.28712332248687744,
    "fx_a1_modulation_depth": 0.6283016800880432,
    "fx_a1_delay_feedback": 0.18823447823524475,
    "fx_a1_eq_low_cut": 0.5102251768112183,
    "fx_a1_eq_high_cut": 0.07518906146287918,
    "fx_a1_output_mix": 0.0,
    "fx_a1_output_width": 0.5458093285560608,
    "fx_a2_delay_time_left": 0.673,
    "fx_a2_delay_time_right": 0.3225,
    "fx_a2_feedback_eq_feedback": 0.09261893033981324,
    "fx_a2_feedback_eq_crossfeed": 0.33431210517883303,
    "fx_a2_feedback_eq_low_cut": 0.7649092674255371,
    "fx_a2_feedback_eq_high_cut": 0.445017546415329,
    "fx_a2_modulation_rate": 0.8504968881607056,
    "fx_a2_modulation_depth": 0.5354740619659424,
    "fx_a2_input_channel": 0.9212818741798401,
    "fx_a2_output_mix": 0.0,
    "fx_a2_output_width": 0.9808350205421448,
    "fx_a3_pre_delay_pre_delay": 0.2418496161699295,
    "fx_a3_reverb_room_size": 0.6668235659599304,
    "fx_a3_reverb_decay_time": 0.9710032939910889,
    "fx_a3_reverb_diffusion": 0.2893294394016266,
    "fx_a3_reverb_buildup": 0.799410879611969,
    "fx_a3_reverb_modulation": 0.7270567417144775,
    "fx_a3_eq_lf_damping": 0.9389781951904297,
    "fx_a3_eq_hf_damping": 0.398930162191391,
    "fx_a3_output_width": 0.7004019618034363,
    "fx_a3_output_mix": 0.0,
}

_HARDCODED_NOTE_PARAMS: dict[str, int | tuple[float, float]] = {
    "pitch": 64,
    "note_start_and_end": (0.77033705, 2.2995389),
}


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
        assert (peak > _AUDIO_PEAK_SILENCE_FLOOR).all(), (
            f"silent clips: peaks={peak.tolist()}"
        )

        return audio_arr, mel[...], params[...]


def _assert_round_trip_matches(
    actual_audio: np.ndarray,
    actual_mel: np.ndarray,
    actual_params: np.ndarray,
    expected_audio: np.ndarray,
    expected_mel: np.ndarray,
    expected_params: np.ndarray,
    expected_synth_patches: list[dict[str, float]],
    expected_note_patches: list[dict[str, int | tuple[float, float]]],
    spec: ParamSpec,
    num_samples: int,
) -> None:
    """Assert a Stage-2 dataset reproduces a Stage-1 dataset within phase-robust tolerances.

    Five checks: ``param_array`` exact equality, per-row phase-robust audio metrics
    (MSS / wMFCC / SOT / RMS-envelope cosine), mel mean-abs-diff, per-row decoded-params
    equality vs. the expected patches, and decoded shape/type sanity. Element-wise audio
    equality is *not* a property of the system — see the docstring of
    ``test_make_dataset_with_fixed_params_round_trips_per_row`` for the phase-init
    nondeterminism background.

    ``expected_synth_patches`` / ``expected_note_patches`` are length-``num_samples``
    lists. For tests that replay a single fixed patch across all rows, callers should
    pass the patch repeated ``num_samples`` times.
    """
    np.testing.assert_allclose(
        actual_params,
        expected_params,
        rtol=_RELATIVE_TOLERANCE,
        atol=_ABSOLUTE_TOLERANCE,
        err_msg="param_array row-for-row mismatch",
    )

    for i in range(num_samples):
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

    mel_dist = float(np.mean(np.abs(actual_mel - expected_mel)))
    log.info("mel mean abs diff: %.4f (max=%.4f)", mel_dist, _MEL_MEAN_ABS_MAX)
    assert mel_dist < _MEL_MEAN_ABS_MAX, (
        f"mel mean abs diff {mel_dist:.4f} exceeds {_MEL_MEAN_ABS_MAX}"
    )

    for i in range(num_samples):
        decoded_synth_params, decoded_note_params = spec.decode(actual_params[i])
        assert isinstance(decoded_synth_params, dict)
        assert decoded_synth_params == pytest.approx(
            expected_synth_patches[i], rel=_RELATIVE_TOLERANCE, abs=_ABSOLUTE_TOLERANCE
        ), (
            f"sample {i}: decoded synth params {decoded_synth_params} do not match input "
            f"{expected_synth_patches[i]} within tolerances "
            f"(abs={_ABSOLUTE_TOLERANCE}, rel={_RELATIVE_TOLERANCE})"
        )
        assert decoded_note_params == pytest.approx(
            expected_note_patches[i], rel=_RELATIVE_TOLERANCE, abs=_ABSOLUTE_TOLERANCE
        ), (
            f"sample {i}: decoded note params {decoded_note_params} do not match input "
            f"{expected_note_patches[i]} within tolerances "
            f"(abs={_ABSOLUTE_TOLERANCE}, rel={_RELATIVE_TOLERANCE})"
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
def test_make_dataset_replays_via_param_spec_sample_mock_with_hardcoded_params(
    tmp_path: Path,
) -> None:
    """make_dataset round-trips a single hardcoded param set when ``param_spec.sample`` is patched.

    Variant of ``test_make_dataset_replays_via_param_spec_sample_mock`` that removes
    the random Stage 1 entirely. Both stages patch ``param_spec.sample`` to return the
    same hardcoded ``(_HARDCODED_SYNTH_PARAMS, _HARDCODED_NOTE_PARAMS)`` tuple, so the
    test pins reproducibility on a fixed, version-controlled patch rather than a
    random candidate. ``num_samples=1``: a single render per stage, total two renders.

    The hardcoded values are a known-good loudness-passing capture from a prior
    surge_xt run; if the spec changes, they must be regenerated.
    """
    spec = param_specs[_SPEC_NAME]
    num_samples = 8

    pull_count = [0]

    def fake_sample() -> tuple[dict[str, float], dict[str, int | tuple[float, float]]]:
        pull_count[0] += 1
        return _HARDCODED_SYNTH_PARAMS, _HARDCODED_NOTE_PARAMS

    expected_dataset = tmp_path / "expected.h5"
    with (
        h5py.File(expected_dataset, "a") as f,
        patch.object(spec, "sample", side_effect=fake_sample),
    ):
        make_dataset(
            hdf5_file=f,
            num_samples=num_samples,
            plugin_path=_PLUGIN_PATH,
            preset_path=_PRESET_PATH,
            sample_rate=_SAMPLE_RATE,
            channels=_CHANNELS,
            velocity=_VELOCITY,
            signal_duration_seconds=_DURATION,
            min_loudness=_MIN_LOUDNESS,
            param_spec=spec,
            sample_batch_size=num_samples,
        )

    assert pull_count[0] == num_samples, (
        f"Stage 1: expected exactly {num_samples} param_spec.sample calls; got "
        f"{pull_count[0]} (loudness loop retried — hardcoded params may be too quiet)"
    )

    expected_audio, expected_mel, expected_params = _assert_h5_structure_is_valid(
        expected_dataset, spec, num_samples
    )

    pull_count[0] = 0
    got_dataset = tmp_path / "replayed.h5"
    with (
        h5py.File(got_dataset, "a") as f,
        patch.object(spec, "sample", side_effect=fake_sample),
    ):
        make_dataset(
            hdf5_file=f,
            num_samples=num_samples,
            plugin_path=_PLUGIN_PATH,
            preset_path=_PRESET_PATH,
            sample_rate=_SAMPLE_RATE,
            channels=_CHANNELS,
            velocity=_VELOCITY,
            signal_duration_seconds=_DURATION,
            min_loudness=_MIN_LOUDNESS,
            param_spec=spec,
            sample_batch_size=num_samples,
        )

    assert pull_count[0] == num_samples, (
        f"Stage 2: expected exactly {num_samples} param_spec.sample calls; got "
        f"{pull_count[0]} (loudness loop retried — hardcoded params may be too quiet)"
    )

    actual_audio, actual_mel, actual_params = _assert_h5_structure_is_valid(
        got_dataset, spec, num_samples
    )

    _assert_round_trip_matches(
        actual_audio=actual_audio,
        actual_mel=actual_mel,
        actual_params=actual_params,
        expected_audio=expected_audio,
        expected_mel=expected_mel,
        expected_params=expected_params,
        expected_synth_patches=[_HARDCODED_SYNTH_PARAMS] * num_samples,
        expected_note_patches=[_HARDCODED_NOTE_PARAMS] * num_samples,
        spec=spec,
        num_samples=num_samples,
    )

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
    note_patches: list[dict[str, int | tuple[float, float]]] = []
    for i in range(_NUM_SAMPLES):
        decoded_synth_params, decoded_note_params = spec.decode(expected_params[i])
        synth_patches.append(decoded_synth_params)
        note_patches.append(decoded_note_params)
    log.info("synth_patches: %s", synth_patches)
    log.info("note_patches: %s", note_patches)

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

    _assert_round_trip_matches(
        actual_audio=actual_audio,
        actual_mel=actual_mel,
        actual_params=actual_params,
        expected_audio=expected_audio,
        expected_mel=expected_mel,
        expected_params=expected_params,
        expected_synth_patches=synth_patches,
        expected_note_patches=note_patches,
        spec=spec,
        num_samples=_NUM_SAMPLES,
    )


@pytest.mark.slow
@pytest.mark.requires_vst
@skip_no_vst
def test_make_dataset_replays_via_param_spec_sample_mock(tmp_path: Path) -> None:
    """make_dataset round-trips when the random source is patched, not via fixed_*_params_list.

    Same two-stage e2e shape as ``test_make_dataset_with_fixed_params_round_trips_per_row``,
    but the Stage 2 replay is driven by patching ``param_spec.sample`` to yield the Stage 1
    candidate ``(synth, note)`` tuples in order, instead of feeding them through the public
    ``fixed_synth_params_list`` / ``fixed_note_params_list`` API. ``generate_sample`` still
    runs end-to-end (real ``render_params``, real loudness gate, real mel computation, real
    writer) — only the random source is controlled. This pins the same reproducibility
    invariant via the function's internal sampling seam.

    The replay assumes ``generate_sample`` calls ``param_spec.sample`` exactly once per output
    sample. If the loudness ``while True`` loop retries (a re-render of an already-survived
    candidate dipping below ``_MIN_LOUDNESS``), the iterator would desync; we detect that by
    asserting the total pull count equals ``_NUM_SAMPLES`` after the run.
    """
    spec = param_specs[_SPEC_NAME]

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

    synth_patches: list[dict[str, float]] = []
    note_patches: list[dict[str, int | tuple[float, float]]] = []
    for i in range(_NUM_SAMPLES):
        decoded_synth_params, decoded_note_params = spec.decode(expected_params[i])
        synth_patches.append(decoded_synth_params)
        note_patches.append(decoded_note_params)

    replay_iter = iter(zip(synth_patches, note_patches, strict=True))
    pull_count = [0]

    def fake_sample() -> tuple[dict[str, float], dict[str, int | tuple[float, float]]]:
        pull_count[0] += 1
        # Extra pulls (loudness retry) raise StopIteration, surfacing as a test failure.
        return next(replay_iter)

    got_dataset = tmp_path / "replayed.h5"
    with (
        h5py.File(got_dataset, "a") as f,
        patch.object(spec, "sample", side_effect=fake_sample),
    ):
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

    assert pull_count[0] == _NUM_SAMPLES, (
        f"expected exactly {_NUM_SAMPLES} param_spec.sample calls; got {pull_count[0]} "
        "(loudness loop retried — replay may have desynced)"
    )

    actual_audio, actual_mel, actual_params = _assert_h5_structure_is_valid(
        got_dataset, spec, _NUM_SAMPLES
    )

    _assert_round_trip_matches(
        actual_audio=actual_audio,
        actual_mel=actual_mel,
        actual_params=actual_params,
        expected_audio=expected_audio,
        expected_mel=expected_mel,
        expected_params=expected_params,
        expected_synth_patches=synth_patches,
        expected_note_patches=note_patches,
        spec=spec,
        num_samples=_NUM_SAMPLES,
    )


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
