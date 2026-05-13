"""Basic e2e test for src/synth_setter/data/vst/generate_vst_dataset.py — verifies HDF5 output."""

import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import h5py
import hdf5plugin  # noqa: F401  side-effect: registers Blosc2 filter for h5py reads
_ = hdf5plugin  # keep type checkers from flagging the side-effect import
import numpy as np
import pytest

from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.evaluation.compute_audio_metrics import compute_mss, compute_rms, compute_sot, compute_wmfcc
from synth_setter.data.vst import param_specs
from synth_setter.data.vst.core import render_params
from synth_setter.data.vst.generate_vst_dataset import make_dataset
from synth_setter.data.vst.param_spec import ParamSpec

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
_RENDERER_VERSION = "1.3.4"
_ABSOLUTE_TOLERANCE = 1e-7
_RELATIVE_TOLERANCE = 1e-9


def _render_cfg(num_samples: int, sample_batch_size: int | None = None) -> RenderConfig:
    """Build a RenderConfig with this module's test defaults.

    ``batch_per_shard`` is set to ``num_samples`` (one shard = one batch in tests);
    ``sample_batch_size`` defaults to the same so each test renders in a single batch.
    """
    return RenderConfig(
        plugin_path=_PLUGIN_PATH,
        preset_path=_PRESET_PATH,
        param_spec_name=_SPEC_NAME,
        renderer_version=_RENDERER_VERSION,
        sample_rate=int(_SAMPLE_RATE),
        channels=_CHANNELS,
        velocity=_VELOCITY,
        signal_duration_seconds=_DURATION,
        min_loudness=_MIN_LOUDNESS,
        sample_batch_size=sample_batch_size if sample_batch_size is not None else num_samples,
        batch_per_shard=num_samples,
    )

# Phase-robust audio similarity thresholds for replayed-params vs. candidates.
# Two independent renders of identical params differ at the sample level on main
# (issue #489 — Surge XT's oscillator phase init is nondeterministic across renders),
# but should remain perceptually close. Tune downward if the metrics consistently come
# in tighter than these caps.
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
#   - `create_datasets_and_get_start_idx()` in `src/synth_setter/data/vst/generate_vst_dataset.py`
#     (the literal `(num_samples, 2, 128, 401)` passed to `create_dataset`)
#   - `make_spectrogram()` in `src/synth_setter/data/vst/generate_vst_dataset.py`
#   - `_SURGE_MEL_SHAPE` in `tests/conftest.py` — mirror, keep in sync.
_PER_SAMPLE_MEL_SHAPE = (2, 128, 401)

skip_no_vst = pytest.mark.skipif(
    not Path(_PLUGIN_PATH).exists(),
    reason=f"VST plugin not found at {_PLUGIN_PATH}",
)

# Known-good loudness-passing patch captured from a prior random-sampled run
# (sample 4 of a 5-sample candidates dataset for the surge_xt spec). Used by
# ``test_datasets_from_hardcoded_params_are_identical`` to drive both stages
# with a fixed, single-sample param set rather than random sampling. If the
# spec changes (keys added/removed), this dict must be regenerated; values
# can be edited freely as long as the resulting render clears ``_MIN_LOUDNESS``.
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
    decode checks without re-opening the file. Shared by the round-trip and random-sampling e2e
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
        mel_arr = mel[...]
        params_arr = params[...]
        assert np.isfinite(audio_arr).all()
        assert np.isfinite(mel_arr).all()
        assert np.isfinite(params_arr).all()

        peak = np.abs(audio_arr).reshape(num_samples, -1).max(axis=1)
        assert (peak > _AUDIO_PEAK_SILENCE_FLOOR).all(), (
            f"silent clips: peaks={peak.tolist()}"
        )

        return audio_arr, mel_arr, params_arr


@contextmanager
def _patched_sample(
    spec: ParamSpec,
    replay: list[tuple[dict[str, float], dict[str, int | tuple[float, float]]]],
) -> Iterator[None]:
    """Patch ``spec.sample`` with a deterministic replay over the lifetime of the block.

    Each call to the patched ``sample`` yields the next ``(synth, note)`` tuple from
    ``replay``. If the loudness ``while True`` loop in ``generate_sample`` retries (an
    unexpected extra call), ``next(replay_iter)`` raises ``StopIteration`` which
    surfaces as a test failure rather than a silent desync. On exit, asserts the
    total pull count equals ``len(replay)``.
    """
    replay_iter = iter(replay)
    pull_count = [0]

    def fake_sample() -> tuple[dict[str, float], dict[str, int | tuple[float, float]]]:
        pull_count[0] += 1
        return next(replay_iter)

    with patch.object(spec, "sample", side_effect=fake_sample):
        yield

    assert pull_count[0] == len(replay), (
        f"expected exactly {len(replay)} param_spec.sample calls; got {pull_count[0]} "
        "(loudness loop retried — replay may have desynced)"
    )


def _audio_metrics(
    target: np.ndarray, pred: np.ndarray
) -> tuple[float, float, float, float]:
    """Compute (mss, wmfcc, sot, rms_cos) for one (target, pred) audio pair."""
    return (
        compute_mss(target, pred),
        compute_wmfcc(target, pred),
        compute_sot(target, pred),
        compute_rms(target, pred),
    )


def _assert_audio_metrics_within_thresholds(
    target: np.ndarray, pred: np.ndarray, label: str
) -> tuple[float, float, float, float]:
    """Assert phase-robust audio metrics for one pair are within the module thresholds.

    Returns the four metric values so callers can accumulate them for trend reporting.
    """
    mss, wmfcc, sot, rms_cos = _audio_metrics(target, pred)
    log.info(
        "%s audio metrics: mss=%.4f wmfcc=%.4f sot=%.4f rms_cos=%.4f",
        label,
        mss,
        wmfcc,
        sot,
        rms_cos,
    )
    assert mss < _MSS_MAX, f"{label}: mss={mss:.4f} exceeds {_MSS_MAX}"
    assert wmfcc < _WMFCC_MAX, f"{label}: wmfcc={wmfcc:.4f} exceeds {_WMFCC_MAX}"
    assert sot < _SOT_MAX, f"{label}: sot={sot:.4f} exceeds {_SOT_MAX}"
    assert rms_cos > _RMS_MIN_COSINE, (
        f"{label}: rms cosine similarity {rms_cos:.4f} below {_RMS_MIN_COSINE}"
    )
    return mss, wmfcc, sot, rms_cos


@dataclass(frozen=True)
class RoundTripMetrics:
    """Per-pair (``expected[i]`` vs ``actual[i]``) audio-similarity summary.

    Computed by ``_assert_round_trip_matches`` over ``num_samples`` matched
    rows. Values are worst-case across rows for distance-style metrics
    (max for MSS / wMFCC / SOT, min for RMS cosine), and a global mean for
    the mel diff.
    """

    mss_max: float
    wmfcc_max: float
    sot_max: float
    rms_cos_min: float
    mel_mean_abs: float
    num_samples: int


@dataclass(frozen=True)
class AllPairsMetrics:
    """Worst-case across all (i, j) pairs in a single audio array.

    Computed by ``_assert_all_pairs_audio_metrics_within_thresholds``.
    Use this when every row is a render of the *same* params — the
    worst-pair signal is the #489 fingerprint (every-other-render variance)
    and is the right thing to track on the trend chart for that bucket.
    """

    mss_max: float
    wmfcc_max: float
    sot_max: float
    rms_cos_min: float
    pair_count: int


def _emit_audio_similarity_benchmark_metrics(
    prefix: str,
    round_trip: RoundTripMetrics | None = None,
    all_pairs: AllPairsMetrics | None = None,
    total_render_seconds: float | None = None,
) -> None:
    """Convert metric structs into bench entries and write them under ``prefix``.

    Caller supplies whichever structs are meaningful for the bucket:

    - ``round_trip`` for buckets where each row uses different params; the
      meaningful comparison is per-pair ``expected[i]`` vs ``actual[i]``.
      Series prefix: ``<prefix>/`` (no extra qualifier).
    - ``all_pairs`` for buckets where every row uses identical params; the
      meaningful comparison is the worst-case across the union of renders
      (the #489 reproducer signal). Series prefix: ``<prefix>/all-pairs-``.
    - ``total_render_seconds`` adds a ``wall-clock-seconds-per-render``
      entry derived from ``round_trip.num_samples`` (so it's only emitted
      when ``round_trip`` is also supplied).

    No-op when ``$BENCHMARK_OUTPUT_DIR`` is unset (local pytest runs).
    """
    entries: list[dict[str, float | str]] = []
    if round_trip is not None:
        entries.extend(
            [
                {
                    "name": f"{prefix}/multi-scale-spectral-loss-max",
                    "unit": "dB",
                    "value": round_trip.mss_max,
                },
                {
                    "name": f"{prefix}/dtw-aligned-mfcc-distance-max",
                    "unit": "L1",
                    "value": round_trip.wmfcc_max,
                },
                {
                    "name": f"{prefix}/spectral-optimal-transport-max",
                    "unit": "Wasserstein",
                    "value": round_trip.sot_max,
                },
                {
                    "name": f"{prefix}/rms-envelope-cosine-distance-max",
                    "unit": "1-cos",
                    "value": 1.0 - round_trip.rms_cos_min,
                },
                {
                    "name": f"{prefix}/mel-spectrogram-mean-absolute-error",
                    "unit": "dB",
                    "value": round_trip.mel_mean_abs,
                },
                {
                    # Static input parameter; emitting it as a series surfaces
                    # accidental fixture-size regressions alongside the metric
                    # drift they would silently cause.
                    "name": f"{prefix}/num-samples",
                    "unit": "count",
                    "value": round_trip.num_samples,
                },
            ]
        )
        if total_render_seconds is not None:
            # Wall-clock time of both stages divided by total renders
            # (= 2 × num_samples, since each stage renders num_samples).
            # Captures renderer perf drift even when audio metrics stay flat.
            # Includes loudness-loop retries on Stage 1 of the random-replay
            # test, so it's a real-throughput number not a render-only one.
            entries.append(
                {
                    "name": f"{prefix}/wall-clock-seconds-per-render",
                    "unit": "seconds",
                    "value": total_render_seconds / (2 * round_trip.num_samples),
                }
            )
    if all_pairs is not None:
        entries.extend(
            [
                {
                    "name": f"{prefix}/all-pairs-multi-scale-spectral-loss-max",
                    "unit": "dB",
                    "value": all_pairs.mss_max,
                },
                {
                    "name": f"{prefix}/all-pairs-dtw-aligned-mfcc-distance-max",
                    "unit": "L1",
                    "value": all_pairs.wmfcc_max,
                },
                {
                    "name": f"{prefix}/all-pairs-spectral-optimal-transport-max",
                    "unit": "Wasserstein",
                    "value": all_pairs.sot_max,
                },
                {
                    "name": f"{prefix}/all-pairs-rms-envelope-cosine-distance-max",
                    "unit": "1-cos",
                    "value": 1.0 - all_pairs.rms_cos_min,
                },
                {
                    "name": f"{prefix}/all-pairs-pair-count",
                    "unit": "count",
                    "value": all_pairs.pair_count,
                },
            ]
        )
    if not entries:
        return
    _emit_benchmark_metrics(entries=entries, bench_filename=f"{prefix}.json")


def _emit_benchmark_metrics(
    entries: list[dict[str, float | str]], bench_filename: str
) -> None:
    """Append metrics to ``$BENCHMARK_OUTPUT_DIR/<bench_filename>`` for the trend chart.

    No-op when the env var is unset (e.g. local pytest runs). One file per
    benchmark "bucket" because ``benchmark-action/github-action-benchmark@v1``
    keys all entries from a single file under one chart page; multiple charts
    require multiple publish steps reading distinct files. See
    ``docs/reference/audio-similarity-benchmarks.md`` for the wiring.

    The on-disk format is the ``customSmallerIsBetter`` schema:
    https://github.com/benchmark-action/github-action-benchmark#examples
    """
    out_dir = os.environ.get("BENCHMARK_OUTPUT_DIR")
    if not out_dir:
        return
    out = Path(out_dir) / bench_filename
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        try:
            existing = json.loads(out.read_text())
        except json.JSONDecodeError:
            # Truncated file from a previous interrupted run — start fresh.
            log.warning("benchmark file %s contains invalid JSON; resetting", out)
            existing = []
    else:
        existing = []
    out.write_text(json.dumps(existing + entries, indent=2))


def _assert_all_pairs_audio_metrics_within_thresholds(
    audio: np.ndarray, label_prefix: str
) -> AllPairsMetrics:
    """For every (i, j) pair with i < j in ``audio``, assert metrics within thresholds.

    Use this when every row is expected to be a render of the *same* params — every pair should be
    perceptually identical, and the worst pair across the run is the interesting datum. Asserts on
    the worst measurement so a single failure surfaces the most-divergent rendering rather than the
    first iteration to trip a threshold. Identifies the (i, j) that produced each worst measurement
    to make debugging easy.

    Returns an ``AllPairsMetrics`` summarising the worst-case observations so callers can pass it
    to ``_emit_audio_similarity_benchmark_metrics`` without recomputing.
    """
    n = audio.shape[0]
    pair_count = n * (n - 1) // 2
    worst_mss = (-1.0, -1, -1)
    worst_wmfcc = (-1.0, -1, -1)
    worst_sot = (-1.0, -1, -1)
    worst_rms_cos = (1.0 + 1e-9, -1, -1)  # min target — start above 1.0
    for i in range(n):
        for j in range(i + 1, n):
            mss, wmfcc, sot, rms_cos = _audio_metrics(audio[i], audio[j])
            if mss > worst_mss[0]:
                worst_mss = (mss, i, j)
            if wmfcc > worst_wmfcc[0]:
                worst_wmfcc = (wmfcc, i, j)
            if sot > worst_sot[0]:
                worst_sot = (sot, i, j)
            if rms_cos < worst_rms_cos[0]:
                worst_rms_cos = (rms_cos, i, j)
    log.info(
        "%s worst-pair over %d pairs: mss=%.4f@(%d,%d) wmfcc=%.4f@(%d,%d) "
        "sot=%.4f@(%d,%d) rms_cos=%.4f@(%d,%d)",
        label_prefix,
        pair_count,
        *worst_mss,
        *worst_wmfcc,
        *worst_sot,
        *worst_rms_cos,
    )
    assert worst_mss[0] < _MSS_MAX, (
        f"{label_prefix}: worst mss={worst_mss[0]:.4f} at pair {worst_mss[1:]} "
        f"exceeds {_MSS_MAX}"
    )
    assert worst_wmfcc[0] < _WMFCC_MAX, (
        f"{label_prefix}: worst wmfcc={worst_wmfcc[0]:.4f} at pair {worst_wmfcc[1:]} "
        f"exceeds {_WMFCC_MAX}"
    )
    assert worst_sot[0] < _SOT_MAX, (
        f"{label_prefix}: worst sot={worst_sot[0]:.4f} at pair {worst_sot[1:]} "
        f"exceeds {_SOT_MAX}"
    )
    assert worst_rms_cos[0] > _RMS_MIN_COSINE, (
        f"{label_prefix}: worst rms cosine={worst_rms_cos[0]:.4f} at pair "
        f"{worst_rms_cos[1:]} below {_RMS_MIN_COSINE}"
    )
    return AllPairsMetrics(
        mss_max=float(worst_mss[0]),
        wmfcc_max=float(worst_wmfcc[0]),
        sot_max=float(worst_sot[0]),
        rms_cos_min=float(worst_rms_cos[0]),
        pair_count=pair_count,
    )


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
) -> RoundTripMetrics:
    """Assert a Stage-2 dataset reproduces a Stage-1 dataset within phase-robust tolerances.

    Five checks: ``param_array`` exact equality, per-row phase-robust audio metrics
    (MSS / wMFCC / SOT / RMS-envelope cosine), mel mean-abs-diff, per-row decoded-params
    equality vs. the expected patches, and decoded shape/type sanity. Element-wise audio
    equality is *not* a property of the system: Surge XT's oscillator phase init is
    nondeterministic across renders of identical params (issue #489), so two renders are
    phase-shifted variants of the same waveform.

    ``expected_synth_patches`` / ``expected_note_patches`` are length-``num_samples``
    lists. For tests that replay a single fixed patch across all rows, callers should
    pass the patch repeated ``num_samples`` times.

    Returns a ``RoundTripMetrics`` summarising the per-row worst-case observations so
    callers can pass it to ``_emit_audio_similarity_benchmark_metrics`` without
    recomputing.
    """
    np.testing.assert_allclose(
        actual_params,
        expected_params,
        rtol=_RELATIVE_TOLERANCE,
        atol=_ABSOLUTE_TOLERANCE,
        err_msg="param_array row-for-row mismatch",
    )

    mss_values: list[float] = []
    wmfcc_values: list[float] = []
    sot_values: list[float] = []
    rms_cos_values: list[float] = []
    for i in range(num_samples):
        mss, wmfcc, sot, rms_cos = _assert_audio_metrics_within_thresholds(
            expected_audio[i], actual_audio[i], label=f"sample {i}"
        )
        mss_values.append(float(mss))
        wmfcc_values.append(float(wmfcc))
        sot_values.append(float(sot))
        rms_cos_values.append(float(rms_cos))

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

    return RoundTripMetrics(
        mss_max=max(mss_values),
        wmfcc_max=max(wmfcc_values),
        sot_max=max(sot_values),
        rms_cos_min=min(rms_cos_values),
        mel_mean_abs=mel_dist,
        num_samples=num_samples,
    )


@pytest.mark.slow
@pytest.mark.requires_vst
@skip_no_vst
def test_datasets_from_hardcoded_params_are_identical(
    tmp_path: Path,
) -> None:
    """make_dataset round-trips a single hardcoded param set when ``param_spec.sample`` is patched.

    Both stages of ``make_dataset`` patch ``param_spec.sample`` to return the same
    hardcoded ``(_HARDCODED_SYNTH_PARAMS, _HARDCODED_NOTE_PARAMS)`` tuple, so every
    one of the ``2 × num_samples`` renders uses identical inputs. This pins
    reproducibility on a fixed, version-controlled patch — no random sampling, no
    Stage-1-then-decode — and the rendered audio across rows should be perceptually
    identical on every pair.

    ``num_samples=6`` (12 renders total) so the all-pairs check has enough pairs to
    surface every-other-render variance — bug #489 only shows up across multiple
    renders, single-pair checks miss it.

    The hardcoded values are a known-good loudness-passing capture from a prior
    surge_xt run; if the spec changes, they must be regenerated.
    """
    spec = param_specs[_SPEC_NAME]
    num_samples = 6
    replay = [(_HARDCODED_SYNTH_PARAMS, _HARDCODED_NOTE_PARAMS)] * num_samples

    expected_dataset = tmp_path / "expected.h5"
    t0 = time.perf_counter()
    with (
        h5py.File(expected_dataset, "a") as f,
        _patched_sample(spec, replay),
    ):
        make_dataset(hdf5_file=f, render_cfg=_render_cfg(num_samples))
    stage1_seconds = time.perf_counter() - t0

    expected_audio, expected_mel, expected_params = _assert_h5_structure_is_valid(
        expected_dataset, spec, num_samples
    )

    got_dataset = tmp_path / "replayed.h5"
    t0 = time.perf_counter()
    with (
        h5py.File(got_dataset, "a") as f,
        _patched_sample(spec, replay),
    ):
        make_dataset(hdf5_file=f, render_cfg=_render_cfg(num_samples))
    stage2_seconds = time.perf_counter() - t0

    actual_audio, actual_mel, actual_params = _assert_h5_structure_is_valid(
        got_dataset, spec, num_samples
    )

    round_trip_metrics = _assert_round_trip_matches(
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

    # Every render in expected ∪ actual uses identical params, so every pair across
    # the union should be perceptually identical. Asserts on the worst-case pair to
    # surface the most-divergent rendering (e.g. an every-other-render glitch). This is
    # the #489 fingerprint, so the trend chart for this bucket tracks all-pairs metrics
    # (the per-row round_trip metrics are emitted alongside for context, not as the
    # primary regression signal).
    all_pairs_metrics = _assert_all_pairs_audio_metrics_within_thresholds(
        np.concatenate([expected_audio, actual_audio], axis=0),
        label_prefix="hardcoded all-pairs",
    )

    # Bench filename is the prefix; workflow's publish step reads
    # ``vst-noise-floor-1-preset-n-renders.json``. See
    # ``docs/reference/audio-similarity-benchmarks.md``.
    _emit_audio_similarity_benchmark_metrics(
        prefix="vst-noise-floor-1-preset-n-renders",
        round_trip=round_trip_metrics,
        all_pairs=all_pairs_metrics,
        total_render_seconds=stage1_seconds + stage2_seconds,
    )


@pytest.mark.slow
@pytest.mark.requires_vst
@skip_no_vst
def test_datasets_from_sampled_params_are_identical(tmp_path: Path) -> None:
    """make_dataset reproduces a previous dataset row-for-row when params are replayed.

    Two-stage e2e test:

    1. Build a "candidates" dataset by calling ``make_dataset`` with the natural
       random source. ``generate_sample`` samples params via ``param_spec.sample()``
       and rejects renders below ``_MIN_LOUDNESS`` in a ``while True`` loop, so each
       surviving row is guaranteed to be loud enough to pass the loudness gate.
    2. Decode the candidate ``param_array`` rows back into ``synth_patches`` /
       ``note_patches`` and replay them in Stage 2 by patching ``param_spec.sample``
       (via ``_patched_sample``) to yield the candidate tuples in order. Because the
       params are guaranteed-loud (step 1), this run can't infinite-loop on the
       loudness gate.

    Assertions:

    - ``param_array`` matches the candidates exactly within float32 numeric
      tolerance — params are deterministic.
    - Per-row audio matches within phase-robust perceptual metrics (MSS,
      wMFCC, SOT, RMS-envelope cosine). Element-wise equality is *not* a
      property of the system: Surge XT's oscillator phase init is
      nondeterministic across renders of identical params (issue #489), so
      the two renders of the same params are phase-shifted variants of the
      same waveform.
    - Mel spec matches within mean absolute log-power error.
    """
    spec = param_specs[_SPEC_NAME]

    # Stage 1: random-sampled "candidates" dataset (loudness-filtered).
    expected_dataset = tmp_path / "candidates.h5"
    t0 = time.perf_counter()
    with h5py.File(expected_dataset, "a") as expected_file:
        make_dataset(hdf5_file=expected_file, render_cfg=_render_cfg(_NUM_SAMPLES))
    stage1_seconds = time.perf_counter() - t0

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
        # ParamSpec.decode is annotated dict[str, float] on main but actually returns
        # dict[str, int | tuple[float, float]] at runtime (pitch is int, note_start_and_end
        # is a tuple). Annotation gets corrected in a sibling PR.
        note_patches.append(decoded_note_params)  # pyright: ignore[reportArgumentType]
    log.info("synth_patches: %s", synth_patches)
    log.info("note_patches: %s", note_patches)

    # Stage 2: replay the candidates via ``_patched_sample`` and verify reproducibility.
    got_dataset = tmp_path / "replayed.h5"
    replay = list(zip(synth_patches, note_patches, strict=True))
    t0 = time.perf_counter()
    with (
        h5py.File(got_dataset, "a") as f,
        _patched_sample(spec, replay),
    ):
        make_dataset(hdf5_file=f, render_cfg=_render_cfg(_NUM_SAMPLES))
    stage2_seconds = time.perf_counter() - t0

    actual_audio, actual_mel, actual_params = _assert_h5_structure_is_valid(
        got_dataset, spec, _NUM_SAMPLES
    )

    round_trip_metrics = _assert_round_trip_matches(
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

    # Each row uses different random params, so cross-row pairs differ legitimately —
    # no all-pairs check applies here. Per-row round-trip metrics are the right
    # regression signal for this bucket. Bench filename is the prefix; workflow's
    # publish step reads ``vst-noise-floor-random-preset-replay.json``. See
    # ``docs/reference/audio-similarity-benchmarks.md``.
    _emit_audio_similarity_benchmark_metrics(
        prefix="vst-noise-floor-random-preset-replay",
        round_trip=round_trip_metrics,
        total_render_seconds=stage1_seconds + stage2_seconds,
    )


@pytest.mark.slow
@pytest.mark.requires_vst
@skip_no_vst
def test_make_dataset(tmp_path: Path) -> None:
    """make_dataset with the natural random source writes a valid h5."""
    out = tmp_path / "random.h5"
    spec = param_specs[_SPEC_NAME]

    with h5py.File(out, "a") as f:
        make_dataset(hdf5_file=f, render_cfg=_render_cfg(_NUM_SAMPLES))

    _, _, params = _assert_h5_structure_is_valid(out, spec, _NUM_SAMPLES)

    decoded_rows = [spec.decode(params[i]) for i in range(_NUM_SAMPLES)]
    for synth, note in decoded_rows:
        assert isinstance(synth, dict)
        assert isinstance(note, dict)
        assert note.keys() == {"pitch", "note_start_and_end"}


@pytest.mark.slow
@pytest.mark.requires_vst
@skip_no_vst
def test_show_editor_warmup_does_not_change_rendered_audio() -> None:
    """``show_editor`` warm-up in ``load_plugin`` does not change rendered audio.

    Empirical characterisation of ``load_plugin``'s ``show_editor`` warm-up.
    Renders the same hardcoded patch N times each with the warm-up enabled
    and disabled (by swapping ``VST3Plugin.show_editor`` to a no-op around
    the second batch), then asserts every cross-path pair (with[i] vs no[j])
    is within the same audio-similarity thresholds the round-trip tests use.

    If this passes, the warm-up is not load-bearing for the per-render
    reload path — it can be dropped without changing output. That matters
    for #714 (macOS SIGTRAP from per-render ``show_editor`` accumulating
    AppKit/CGS state in unbundled python).
    """
    from pedalboard import VST3Plugin

    n_renders = 3
    pitch = _HARDCODED_NOTE_PARAMS["pitch"]
    note_window = _HARDCODED_NOTE_PARAMS["note_start_and_end"]
    assert isinstance(pitch, int)
    assert isinstance(note_window, tuple)

    def _render_n(n: int) -> list[np.ndarray]:
        return [
            render_params(
                _PLUGIN_PATH,
                _HARDCODED_SYNTH_PARAMS,
                pitch,
                _VELOCITY,
                note_window,
                _DURATION,
                _SAMPLE_RATE,
                _CHANNELS,
                preset_path=_PRESET_PATH,
            )
            for _ in range(n)
        ]

    with_editor = _render_n(n_renders)

    original_show_editor = VST3Plugin.show_editor
    VST3Plugin.show_editor = lambda self, *a, **kw: None
    try:
        no_editor = _render_n(n_renders)
    finally:
        VST3Plugin.show_editor = original_show_editor

    for i, target in enumerate(with_editor):
        for j, pred in enumerate(no_editor):
            _assert_audio_metrics_within_thresholds(
                target, pred, label=f"with-editor[{i}] vs no-editor[{j}]"
            )


def test_make_dataset_raises_when_fixed_params_list_is_too_short(
    tmp_path: Path,
) -> None:
    """make_dataset rejects fixed_*_params_list shorter than batch_per_shard - start_idx."""
    out = tmp_path / "should_not_write.h5"
    with h5py.File(out, "a") as f:
        with pytest.raises(ValueError, match="fixed_synth_params_list has length"):
            make_dataset(
                hdf5_file=f,
                render_cfg=_render_cfg(num_samples=3),
                fixed_synth_params_list=[_HARDCODED_SYNTH_PARAMS],
            )


# Unit tests for the loudness-loop retry/raise semantics. Mocking
# ``render_params`` and ``param_spec.sample`` keeps these CPU-only and fast —
# no VST plugin, no real audio rendering. They guard the asymmetric guard in
# ``generate_sample``: silent renders raise immediately when the synth is
# fixed, but retry (still unboundedly) when only note params are fixed —
# synth is re-sampled each iteration, so retrying remains meaningful — see
# issue #724.

_SILENT_AUDIO_PEAK = 0.0
_LOUD_AUDIO_PEAK = 0.5


def _silent_audio() -> np.ndarray:
    """Return a silent stereo render with the module's standard shape."""
    return np.zeros((_CHANNELS, int(_SAMPLE_RATE * _DURATION)), dtype=np.float32)


def _loud_audio() -> np.ndarray:
    """Return a clearly-loud stereo render: a 440 Hz sine at half-scale."""
    n = int(_SAMPLE_RATE * _DURATION)
    t = np.arange(n) / _SAMPLE_RATE
    sine = (_LOUD_AUDIO_PEAK * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    return np.stack([sine, sine], axis=0)


def test_generate_sample_raises_when_fixed_synth_params_renders_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silent render with ``fixed_synth_params`` (no note fix) raises instead of looping.

    Pre-fix, this call would loop forever: ``fully_fixed`` was False (note params
    not fixed) so the loudness gate fell through to ``continue``, but re-sampling
    note params alone never lifts a silent synth above threshold. Post-fix, the
    "fixed synth" branch raises immediately. See issue #724.
    """
    from synth_setter.data.vst import generate_vst_dataset

    spec = param_specs[_SPEC_NAME]
    monkeypatch.setattr(generate_vst_dataset, "render_params", lambda *a, **kw: _silent_audio())

    with pytest.raises(ValueError, match="fixed_synth_params render produced loudness"):
        generate_vst_dataset.generate_sample(
            plugin_path=_PLUGIN_PATH,
            velocity=_VELOCITY,
            signal_duration_seconds=_DURATION,
            sample_rate=_SAMPLE_RATE,
            channels=_CHANNELS,
            min_loudness=_MIN_LOUDNESS,
            param_spec=spec,
            preset_path=_PRESET_PATH,
            fixed_synth_params=_HARDCODED_SYNTH_PARAMS,
            fixed_note_params=None,
        )


def test_generate_sample_raises_when_both_fixed_renders_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both-fixed silent render still raises (existing fully-fixed behavior preserved)."""
    from synth_setter.data.vst import generate_vst_dataset

    spec = param_specs[_SPEC_NAME]
    monkeypatch.setattr(generate_vst_dataset, "render_params", lambda *a, **kw: _silent_audio())

    with pytest.raises(ValueError, match="fixed_synth_params render produced loudness"):
        generate_vst_dataset.generate_sample(
            plugin_path=_PLUGIN_PATH,
            velocity=_VELOCITY,
            signal_duration_seconds=_DURATION,
            sample_rate=_SAMPLE_RATE,
            channels=_CHANNELS,
            min_loudness=_MIN_LOUDNESS,
            param_spec=spec,
            preset_path=_PRESET_PATH,
            fixed_synth_params=_HARDCODED_SYNTH_PARAMS,
            fixed_note_params=_HARDCODED_NOTE_PARAMS,
        )


def test_generate_sample_retries_when_only_fixed_note_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only-note-fixed retries: synth is re-sampled each iteration, loop is meaningful.

    First render is silent (rejected); second is loud (accepted). The function should
    retry rather than raise — re-sampling the synth each iteration can lift loudness
    above threshold.
    """
    from synth_setter.data.vst import generate_vst_dataset

    spec = param_specs[_SPEC_NAME]

    render_outputs = iter([_silent_audio(), _loud_audio()])
    monkeypatch.setattr(
        generate_vst_dataset,
        "render_params",
        lambda *a, **kw: next(render_outputs),
    )

    sample_returns = iter([
        (_HARDCODED_SYNTH_PARAMS, _HARDCODED_NOTE_PARAMS),
        (_HARDCODED_SYNTH_PARAMS, _HARDCODED_NOTE_PARAMS),
    ])
    monkeypatch.setattr(spec, "sample", lambda: next(sample_returns))

    sample = generate_vst_dataset.generate_sample(
        plugin_path=_PLUGIN_PATH,
        velocity=_VELOCITY,
        signal_duration_seconds=_DURATION,
        sample_rate=_SAMPLE_RATE,
        channels=_CHANNELS,
        min_loudness=_MIN_LOUDNESS,
        param_spec=spec,
        preset_path=_PRESET_PATH,
        fixed_synth_params=None,
        fixed_note_params=_HARDCODED_NOTE_PARAMS,
    )
    # Exhausted both render outputs ⇒ retried exactly once.
    assert next(render_outputs, None) is None
    assert sample.note_params == _HARDCODED_NOTE_PARAMS


@pytest.mark.slow
@pytest.mark.requires_vst
@skip_no_vst
def test_make_dataset_uses_fixed_params_lists_when_provided(
    tmp_path: Path,
) -> None:
    """make_dataset writes the supplied fixed params verbatim, bypassing param_spec.sample()."""
    spec = param_specs[_SPEC_NAME]
    out = tmp_path / "fixed.h5"
    num_samples = 3
    with h5py.File(out, "a") as f:
        make_dataset(
            hdf5_file=f,
            render_cfg=_render_cfg(num_samples),
            fixed_synth_params_list=[_HARDCODED_SYNTH_PARAMS] * num_samples,
            fixed_note_params_list=[_HARDCODED_NOTE_PARAMS] * num_samples,
        )

    _, _, params = _assert_h5_structure_is_valid(out, spec, num_samples)
    # ParamSpec.encode is annotated dict[str, float] on main but accepts the runtime
    # note-param shape (pitch is int, note_start_and_end is a tuple); annotation gets
    # corrected in a sibling PR.
    expected = spec.encode(_HARDCODED_SYNTH_PARAMS, _HARDCODED_NOTE_PARAMS)  # pyright: ignore[reportArgumentType]
    for i in range(num_samples):
        assert np.allclose(params[i], expected, atol=_ABSOLUTE_TOLERANCE), (
            f"row {i} did not match fixed params"
        )


# Unit tests for ``_emit_audio_similarity_benchmark_metrics`` — pure JSON
# serialization, no VST needed. Fast feedback while iterating on the
# emission schema; complements the slow integration coverage above.


def test_emit_benchmark_skips_when_env_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No-op when ``BENCHMARK_OUTPUT_DIR`` isn't set (local pytest runs)."""
    monkeypatch.delenv("BENCHMARK_OUTPUT_DIR", raising=False)
    _emit_audio_similarity_benchmark_metrics(
        prefix="x",
        round_trip=RoundTripMetrics(
            mss_max=1.0, wmfcc_max=1.0, sot_max=0.001,
            rms_cos_min=0.99, mel_mean_abs=0.5, num_samples=4,
        ),
    )
    assert list(tmp_path.iterdir()) == []  # nothing written


def test_emit_benchmark_round_trip_only_writes_expected_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-pair-only emission (sampled bucket): seven entries, no all-pairs.

    Includes ``wall-clock-seconds-per-render`` because ``total_render_seconds``
    is supplied. ``rms-envelope-cosine-distance`` is reported as ``1 - rms_cos_min``
    so the schema reads smaller-is-better.
    """
    monkeypatch.setenv("BENCHMARK_OUTPUT_DIR", str(tmp_path))
    _emit_audio_similarity_benchmark_metrics(
        prefix="bucket-a",
        round_trip=RoundTripMetrics(
            mss_max=2.5, wmfcc_max=3.5, sot_max=0.02,
            rms_cos_min=0.97, mel_mean_abs=1.5, num_samples=4,
        ),
        total_render_seconds=8.0,  # 8 / (2 × 4) = 1.0 s/render
    )
    out = tmp_path / "bucket-a.json"
    assert out.exists()
    entries = json.loads(out.read_text())
    by_name = {e["name"]: e for e in entries}
    assert by_name["bucket-a/multi-scale-spectral-loss-max"]["value"] == 2.5
    assert by_name["bucket-a/dtw-aligned-mfcc-distance-max"]["value"] == 3.5
    assert by_name["bucket-a/spectral-optimal-transport-max"]["value"] == 0.02
    assert (
        by_name["bucket-a/rms-envelope-cosine-distance-max"]["value"]
        == pytest.approx(1.0 - 0.97)
    )
    assert by_name["bucket-a/mel-spectrogram-mean-absolute-error"]["value"] == 1.5
    assert by_name["bucket-a/num-samples"]["value"] == 4
    assert by_name["bucket-a/wall-clock-seconds-per-render"]["value"] == 1.0
    # Units are spelled out, not abbreviated.
    assert by_name["bucket-a/spectral-optimal-transport-max"]["unit"] == "Wasserstein"
    # No all-pairs entries when only round_trip is supplied.
    assert not any("all-pairs" in name for name in by_name)


def test_emit_benchmark_all_pairs_only_skips_round_trip_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All-pairs-only emission: five entries, none of the round-trip series."""
    monkeypatch.setenv("BENCHMARK_OUTPUT_DIR", str(tmp_path))
    _emit_audio_similarity_benchmark_metrics(
        prefix="bucket-b",
        all_pairs=AllPairsMetrics(
            mss_max=4.0, wmfcc_max=6.0, sot_max=0.03,
            rms_cos_min=0.95, pair_count=66,
        ),
    )
    entries = json.loads((tmp_path / "bucket-b.json").read_text())
    by_name = {e["name"]: e for e in entries}
    assert by_name["bucket-b/all-pairs-multi-scale-spectral-loss-max"]["value"] == 4.0
    assert (
        by_name["bucket-b/all-pairs-rms-envelope-cosine-distance-max"]["value"]
        == pytest.approx(1.0 - 0.95)
    )
    assert by_name["bucket-b/all-pairs-pair-count"]["value"] == 66
    # No round-trip entries when only all_pairs is supplied.
    assert "bucket-b/multi-scale-spectral-loss-max" not in by_name
    assert "bucket-b/num-samples" not in by_name


def test_emit_benchmark_both_structs_writes_both_namespaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hardcoded bucket emits both per-row and all-pairs (the latter is the #489 signal)."""
    monkeypatch.setenv("BENCHMARK_OUTPUT_DIR", str(tmp_path))
    _emit_audio_similarity_benchmark_metrics(
        prefix="bucket-c",
        round_trip=RoundTripMetrics(
            mss_max=1.0, wmfcc_max=1.0, sot_max=0.001,
            rms_cos_min=0.99, mel_mean_abs=0.5, num_samples=6,
        ),
        all_pairs=AllPairsMetrics(
            mss_max=4.0, wmfcc_max=6.0, sot_max=0.03,
            rms_cos_min=0.95, pair_count=66,
        ),
    )
    entries = json.loads((tmp_path / "bucket-c.json").read_text())
    names = {e["name"] for e in entries}
    # Both namespaces present.
    assert "bucket-c/multi-scale-spectral-loss-max" in names
    assert "bucket-c/all-pairs-multi-scale-spectral-loss-max" in names


def test_emit_benchmark_no_args_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both structs None ⇒ no file written, no error."""
    monkeypatch.setenv("BENCHMARK_OUTPUT_DIR", str(tmp_path))
    _emit_audio_similarity_benchmark_metrics(prefix="bucket-d")
    assert list(tmp_path.iterdir()) == []


def test_emit_benchmark_appends_to_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Subsequent emissions for the same prefix accumulate in the same file."""
    monkeypatch.setenv("BENCHMARK_OUTPUT_DIR", str(tmp_path))
    rt = RoundTripMetrics(
        mss_max=1.0, wmfcc_max=1.0, sot_max=0.001,
        rms_cos_min=0.99, mel_mean_abs=0.5, num_samples=2,
    )
    _emit_audio_similarity_benchmark_metrics(prefix="bucket-e", round_trip=rt)
    _emit_audio_similarity_benchmark_metrics(prefix="bucket-e", round_trip=rt)
    entries = json.loads((tmp_path / "bucket-e.json").read_text())
    # Six round-trip entries per call (no render-seconds), twice = 12 total.
    assert len(entries) == 12
