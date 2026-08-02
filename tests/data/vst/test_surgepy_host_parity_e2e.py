"""Real three-host Surge rendering parity and throughput benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Literal, TypedDict, cast

import lance
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.io import wavfile

from synth_setter.data.vst.generate_vst_dataset import make_spectrogram
from synth_setter.data.vst.param_map import load_param_map
from synth_setter.data.vst.param_spec import NoteParams
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.data.vst.renderers import _sample_index_at_or_after
from synth_setter.data.vst.shapes import AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD
from synth_setter.data.vst.surgepy_runtime import surge_component_state
from synth_setter.data.vst.writers import make_lance_dataset
from synth_setter.evaluation.compute_audio_metrics import (
    compute_mss,
    compute_rms,
    compute_sot,
    compute_wmfcc,
)
from synth_setter.pipeline.schemas.spec import RenderConfig
from tests._vst import TEST_SYNTH
from tests.data.vst.test_dawdreamer_dataset_e2e import _dawdreamer_experiment_config
from tests.data.vst.test_generate_vst_dataset import (
    _HARDCODED_NOTE_PARAMS,
    _HARDCODED_SYNTH_PARAMS,
    _emit_benchmark_metrics,
)

type BenchmarkEntry = dict[str, float | str]
type ParityBackend = Literal["dawdreamer", "pedalboard", "surgepy"]


class MetricRow(TypedDict):
    """One matched-render quality diagnostic.

    .. attribute :: sample

        Zero-based workload row.

    .. attribute :: mel_rmse

        Persisted mel root mean squared error.

    .. attribute :: mss

        Multi-scale spectral loss.

    .. attribute :: rms

        RMS-envelope cosine similarity.

    .. attribute :: rms_distance

        One minus RMS-envelope cosine similarity.

    .. attribute :: sot

        Spectral optimal transport distance.

    .. attribute :: wmfcc

        Warped MFCC distance.
    """

    sample: int
    mel_rmse: float
    mss: float
    rms: float
    rms_distance: float
    sot: float
    wmfcc: float


class DiagnosticOnsetRow(TypedDict):
    """One diagnostic onset, with ``None`` when a patch remains silent.

    .. attribute :: backend

        Rendering host.

    .. attribute :: sample

        Zero-based workload row.

    .. attribute :: onset_sample

        First audible sample, or ``None`` when no sample crosses the diagnostic amplitude.

    .. attribute :: requested_sample

        Quantized MIDI note-start sample.
    """

    backend: ParityBackend
    sample: int
    onset_sample: int | None
    requested_sample: int


class OnsetRow(TypedDict):
    """One backend render's absolute onset diagnostic.

    .. attribute :: backend

        Rendering host.

    .. attribute :: sample

        Zero-based workload row.

    .. attribute :: onset_sample

        First audible sample.

    .. attribute :: requested_sample

        Quantized MIDI note-start sample.
    """

    backend: ParityBackend
    sample: int
    onset_sample: int
    requested_sample: int


_BACKENDS: tuple[ParityBackend, ...] = ("pedalboard", "dawdreamer", "surgepy")
_TRUSTED_BACKENDS: tuple[ParityBackend, ...] = ("pedalboard", "dawdreamer")
_REPEATED_RENDER_COUNT = 30
_RANDOM_MIN_LOUDNESS = float("-inf")
_RANDOM_PATCH_COUNT = 30
_RANDOM_PATCH_SEED = 20260330
_RANDOM_WORKLOAD = "random-patches"
_SURGE_SYNTH_PARAMETER_COUNT = 162
_RANDOM_LOUDNESS_OVERRIDE_RATIONALE = (
    "Sample 1 falls below the normal -55 dB validity floor; retaining the fixed corpus "
    "requires accepting every diagnostic render without rejection or resampling."
)
_PARITY_SYNTH_PARAMS = {**_HARDCODED_SYNTH_PARAMS, "a_osc_drift": 0.0}
_DIVERSE_PATCH_VALUES = (
    (0.08, 0.044),
    (0.18, 0.1705),
    (0.28, 0.3355),
    (0.38, 0.5005),
    (0.52, 0.6655),
    (0.64, 0.8305),
    (0.76, 0.9565),
    (0.88, 0.9565),
)
_EXPECTED_MEL_SHAPE = (2, 128, 401)
_ADJACENT_MEL_RMSE_MIN = 2.5
_CAUSAL_CENTROID_SHIFT_MIN = 1.0
_CAUSAL_OCTAVE_FREQUENCY_RATIO_MIN = 6.0
_DIVERSE_CENTROID_SHIFT_MIN = 7.0
_DOMINANT_FREQUENCY_MAX_HZ = 5_000.0
_DOMINANT_FREQUENCY_MIN_HZ = 40.0
_MEL_RMSE_MAX = 5.0
_MSS_MAX = 6.0
_ONSET_AMPLITUDE = 1e-8
_ONSET_CONTROL_LAG_TOLERANCE_SAMPLES = 2
_PARAMETER_MAP_PATH = Path("src/synth_setter/data/vst/surge_xt_param_map.json")
_RMS_MIN = 0.99
_SOT_MAX = 0.01
_SURGEPY_PRESET_PATH = Path("presets/surge-base.fxp")
_VST_PRESET_PATH = Path("presets/surge-base.vstpreset")
_WMFCC_MAX = 2.0
_OUT_OF_LIMIT_METRIC_ROW: MetricRow = {
    "sample": 7,
    "mel_rmse": 50.0,
    "mss": 60.0,
    "rms": 0.5,
    "rms_distance": 0.5,
    "sot": 0.1,
    "wmfcc": 20.0,
}


@dataclass(frozen=True)
class _BackendResult:
    """Materialized production artifact and elapsed generation time.

    .. attribute :: audio
        :type: np.ndarray

        Channel-leading waveforms consumed from Lance.

    .. attribute :: mel
        :type: np.ndarray

        Persisted mel tensors consumed from Lance.

    .. attribute :: params
        :type: np.ndarray

        Encoded normalized parameter rows consumed from Lance.

    .. attribute :: total_seconds
        :type: float

        End-to-end dataset generation time.
    """

    audio: np.ndarray
    mel: np.ndarray
    params: np.ndarray
    total_seconds: float


def _config(
    backend: ParityBackend,
    render_count: int,
    *,
    min_loudness: float | None = None,
) -> RenderConfig:
    """Return one production render configuration for a workload.

    :param backend: Host selected for the real dataset path.
    :param render_count: Number of rows in the workload.
    :param min_loudness: Workload-specific loudness floor override, when required.
    :returns: Validated fixed-workload render configuration.
    """
    values = {
        **_dawdreamer_experiment_config().model_dump(),
        "audio_dtype": "float32",
        "gui_toggle_cadence": "never",
        "plugin_reload_cadence": "render",
        "samples_per_shard": render_count,
        "renderer_backend": backend,
    }
    if min_loudness is not None:
        values["min_loudness"] = min_loudness
    if backend == "surgepy":
        values["synth"] = {
            **values["synth"],
            "plugin_path": "surgepy",
            "plugin_state_path": str(_SURGEPY_PRESET_PATH),
            "synth_version": "1.3.master.f7b97c68",
        }
    return RenderConfig.model_validate(values)


def _random_patch_config(backend: ParityBackend) -> RenderConfig:
    """Return the diagnostic config that retains the fixed random corpus.

    :param backend: Host selected for the real dataset path.
    :returns: Random-workload config with loudness rejection disabled.
    """
    return _config(backend, _RANDOM_PATCH_COUNT, min_loudness=_RANDOM_MIN_LOUDNESS)


def _sample_random_patches(config: RenderConfig, *, seed: int) -> list[dict[str, float]]:
    """Sample the ordered random corpus directly from the resolved ParamSpec.

    :param config: Render config selecting the production ParamSpec.
    :param seed: NumPy generator seed defining the complete corpus.
    :returns: Full-dimensional normalized synth patches in sample order.
    """
    param_spec = resolve_param_spec(config.param_spec_name)
    rng = np.random.default_rng(seed)
    return [param_spec.sample(rng)[0] for _ in range(_RANDOM_PATCH_COUNT)]


def _render_dataset(
    backend: ParityBackend,
    path: Path,
    *,
    synth_params: list[dict[str, float]],
    note_params: list[NoteParams],
    min_loudness: float | None = None,
) -> _BackendResult:
    """Render and consume one real fixed-workload Lance dataset.

    :param backend: Host selected for rendering.
    :param path: Lance dataset destination.
    :param synth_params: Exact normalized patches shared by all hosts.
    :param note_params: Exact MIDI events shared by all hosts.
    :param min_loudness: Workload-specific loudness floor override, when required.
    :returns: Materialized columns and elapsed production-path time.
    """
    render_count = len(synth_params)
    started = time.perf_counter()
    config = _config(backend, render_count, min_loudness=min_loudness)
    make_lance_dataset(
        path,
        config,
        fixed_synth_params_list=synth_params,
        fixed_note_params_list=note_params,
    )
    elapsed = time.perf_counter() - started
    columns = lance.dataset(str(path)).to_table(
        columns=[AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD]
    )
    audio = columns.column(AUDIO_FIELD).combine_chunks().to_numpy_ndarray()
    mel = columns.column(MEL_SPEC_FIELD).combine_chunks().to_numpy_ndarray()
    assert audio.dtype == np.dtype(config.audio_dtype)
    assert mel.dtype == np.dtype(config.mel_spec_dtype)
    return _BackendResult(
        audio=audio,
        mel=mel,
        params=columns.column(PARAM_ARRAY_FIELD).combine_chunks().to_numpy_ndarray(),
        total_seconds=elapsed,
    )


def _render_workload(
    tmp_path: Path,
    workload: str,
    synth_params: list[dict[str, float]],
    *,
    min_loudness: float | None = None,
) -> dict[ParityBackend, _BackendResult]:
    """Render one exact patch corpus through every backend.

    :param tmp_path: Temporary dataset root.
    :param workload: Name used to isolate Lance paths.
    :param synth_params: Exact normalized patches shared by all hosts.
    :param min_loudness: Workload-specific loudness floor override, when required.
    :returns: Consumed results keyed by backend.
    """
    note_params = [_HARDCODED_NOTE_PARAMS.copy() for _ in synth_params]
    return {
        backend: _render_dataset(
            backend,
            tmp_path / f"{workload}-{backend}.lance",
            synth_params=synth_params,
            note_params=note_params,
            min_loudness=min_loudness,
        )
        for backend in _BACKENDS
    }


def _pair_metric_rows(
    reference: _BackendResult,
    candidate: _BackendResult,
) -> list[MetricRow]:
    """Return quality metrics for every matched artifact row.

    :param reference: First host's consumed dataset.
    :param candidate: Second host's consumed dataset.
    :returns: Per-row audio and persisted-mel comparison metrics.
    """
    rows: list[MetricRow] = []
    for index in range(len(reference.audio)):
        mel_delta = reference.mel[index] - candidate.mel[index]
        rms = float(np.clip(compute_rms(reference.audio[index], candidate.audio[index]), -1.0, 1.0))
        rows.append(
            {
                "sample": index,
                "mel_rmse": float(np.sqrt(np.mean(np.square(mel_delta)))),
                "mss": float(compute_mss(reference.audio[index], candidate.audio[index])),
                "rms": rms,
                "rms_distance": 1.0 - rms,
                "sot": float(compute_sot(reference.audio[index], candidate.audio[index])),
                "wmfcc": float(compute_wmfcc(reference.audio[index], candidate.audio[index])),
            }
        )
    return rows


def _pair_metrics(
    results: dict[ParityBackend, _BackendResult],
) -> dict[str, list[MetricRow]]:
    """Compute all pairwise diagnostics for a three-host workload.

    :param results: Materialized artifacts keyed by backend.
    :returns: Per-sample metric rows keyed by backend pair.
    """
    return {
        f"{reference}-vs-{candidate}": _pair_metric_rows(results[reference], results[candidate])
        for reference, candidate in combinations(results, 2)
    }


def _onset_samples(audio: np.ndarray) -> np.ndarray:
    """Return the first audible sample in each channel-leading waveform.

    :param audio: Waveforms shaped ``(rows, channels, samples)``.
    :returns: Integer onset sample per row; a silent row fails the test.
    :raises ValueError: If ``audio`` does not have three nonempty dimensions.
    """
    if audio.ndim != 3 or any(size == 0 for size in audio.shape):
        raise ValueError("audio must have shape (rows, channels, samples)")
    onsets: list[int] = []
    for index, waveform in enumerate(audio):
        audible = np.flatnonzero(np.max(np.abs(waveform), axis=0) > _ONSET_AMPLITUDE)
        assert len(audible) > 0, f"sample {index} is silent"
        onsets.append(int(audible[0]))
    return np.asarray(onsets)


def _onset_rows(
    results: dict[ParityBackend, _BackendResult],
) -> list[OnsetRow]:
    """Return requested and observed onset samples for every render.

    :param results: Materialized artifacts keyed by backend.
    :returns: Per-backend and per-sample onset diagnostics.
    """
    render_count = len(results["pedalboard"].audio)
    note_start = _HARDCODED_NOTE_PARAMS["note_start_and_end"][0]
    requested_sample = _sample_index_at_or_after(
        note_start,
        _config("pedalboard", render_count).sample_rate,
    )
    rows: list[OnsetRow] = []
    for backend, result in results.items():
        rows.extend(
            {
                "backend": backend,
                "sample": sample,
                "onset_sample": int(onset),
                "requested_sample": requested_sample,
            }
            for sample, onset in enumerate(_onset_samples(result.audio))
        )
    return rows


def _diagnostic_onset_rows(
    results: dict[ParityBackend, _BackendResult],
    sample_rate: int,
) -> list[DiagnosticOnsetRow]:
    """Return optional onset diagnostics without rejecting silent random rows.

    :param results: Materialized artifacts keyed by backend.
    :param sample_rate: Rendered audio sample rate in Hz.
    :returns: Per-render onset samples, retaining ``None`` for silence.
    """
    requested_sample = _sample_index_at_or_after(
        _HARDCODED_NOTE_PARAMS["note_start_and_end"][0],
        sample_rate,
    )
    rows: list[DiagnosticOnsetRow] = []
    for backend, result in results.items():
        for sample, waveform in enumerate(result.audio):
            audible = np.flatnonzero(np.max(np.abs(waveform), axis=0) > _ONSET_AMPLITUDE)
            rows.append(
                {
                    "backend": backend,
                    "sample": sample,
                    "onset_sample": int(audible[0]) if len(audible) else None,
                    "requested_sample": requested_sample,
                }
            )
    return rows


def _assert_onset_parity(workload: str, rows: list[OnsetRow]) -> None:
    """Gate absolute onset and lag from independent trusted hosts.

    :param workload: Workload identity included in failures.
    :param rows: Per-render onset diagnostics.
    """
    early = [row for row in rows if int(row["onset_sample"]) < int(row["requested_sample"])]
    assert not early, f"early host onset(s): workload={workload}, rows={early}"

    onset_by_identity = {
        (int(row["sample"]), str(row["backend"])): int(row["onset_sample"]) for row in rows
    }
    late: list[OnsetRow] = []
    for row in rows:
        sample = int(row["sample"])
        backend = cast(ParityBackend, row["backend"])
        controls = [
            onset_by_identity[(sample, control)]
            for control in _TRUSTED_BACKENDS
            if control != backend
        ]
        if int(row["onset_sample"]) > min(controls) + _ONSET_CONTROL_LAG_TOLERANCE_SAMPLES:
            late.append(row)
    assert not late, f"late host onset(s): workload={workload}, rows={late}"


def _assert_pair_metrics(workload: str, pair_rows: dict[str, list[MetricRow]]) -> None:
    """Gate every matched render with case-level diagnostics.

    :param workload: Workload identity included in failures.
    :param pair_rows: Per-sample metrics keyed by backend pair.
    """
    for pair, rows in pair_rows.items():
        for row in rows:
            identity = {
                "workload": workload,
                "backend_pair": pair,
                "sample": int(row["sample"]),
                "metrics": row,
            }
            assert float(row["mel_rmse"]) <= _MEL_RMSE_MAX, identity
            assert float(row["mss"]) <= _MSS_MAX, identity
            assert float(row["rms"]) >= _RMS_MIN, identity
            assert float(row["sot"]) <= _SOT_MAX, identity
            assert float(row["wmfcc"]) <= _WMFCC_MAX, identity


def _assert_repeated_backend_stability(
    results: dict[ParityBackend, _BackendResult],
) -> None:
    """Gate every repeated mel against its backend's first render.

    :param results: Repeated-patch artifacts keyed by backend.
    """
    for backend, result in results.items():
        for sample, mel in enumerate(result.mel):
            mel_rmse = float(np.sqrt(np.mean(np.square(mel - result.mel[0]))))
            identity = {"backend": backend, "sample": sample, "mel_rmse": mel_rmse}
            assert mel_rmse <= _MEL_RMSE_MAX, identity


def _worst_pair_metrics(rows: list[MetricRow]) -> dict[str, float]:
    """Reduce per-row metrics to benchmark summaries.

    :param rows: Per-row metrics for one backend pair.
    :returns: Worst distance values and minimum RMS-envelope cosine.
    """
    return {
        "mel_rmse": max(float(row["mel_rmse"]) for row in rows),
        "mss": max(float(row["mss"]) for row in rows),
        "rms": min(float(row["rms"]) for row in rows),
        "rms_distance": max(float(row["rms_distance"]) for row in rows),
        "sot": max(float(row["sot"]) for row in rows),
        "wmfcc": max(float(row["wmfcc"]) for row in rows),
    }


def _benchmark_entries(
    workload: str,
    *,
    results: dict[ParityBackend, _BackendResult],
    pair_rows: dict[str, list[MetricRow]],
) -> list[BenchmarkEntry]:
    """Build quality and throughput entries for one workload.

    :param workload: Stable benchmark-series workload name.
    :param results: Real artifact results keyed by host.
    :param pair_rows: Pairwise quality diagnostics.
    :returns: Benchmark-action custom metric entries.
    """
    render_count = len(next(iter(results.values())).audio)
    prefix = f"surge-host-parity/{workload}"
    entries: list[BenchmarkEntry] = [
        {"name": f"{prefix}/render-count", "unit": "renders", "value": render_count}
    ]
    for backend, result in results.items():
        seconds_per_render = result.total_seconds / render_count
        entries.extend(
            [
                {
                    "name": f"{prefix}/{backend}/dataset-seconds-per-render",
                    "unit": "seconds",
                    "value": seconds_per_render,
                },
                {
                    "name": f"{prefix}/{backend}/dataset-realtime-factor",
                    "unit": "ratio",
                    "value": seconds_per_render
                    / _config(backend, render_count).signal_duration_seconds,
                },
            ]
        )
    for pair, rows in pair_rows.items():
        metrics = _worst_pair_metrics(rows)
        for metric in ("mel_rmse", "mss", "sot", "wmfcc"):
            entries.append(
                {
                    "name": f"{prefix}/{pair}/{metric}-max",
                    "unit": metric,
                    "value": metrics[metric],
                }
            )
        entries.append(
            {
                "name": f"{prefix}/{pair}/rms-envelope-cosine-distance-max",
                "unit": "1-cos",
                "value": 1.0 - metrics["rms"],
            }
        )
    return entries


def _diagnostic_benchmark_entries(
    workload: str,
    *,
    results: dict[ParityBackend, _BackendResult],
    pair_rows: dict[str, list[MetricRow]],
) -> list[BenchmarkEntry]:
    """Build complete per-row and worst-case diagnostic metric series.

    :param workload: Stable diagnostic workload name.
    :param results: Real artifact results keyed by host.
    :param pair_rows: Pairwise quality diagnostics retained without limits.
    :returns: Benchmark-action entries including every observed metric row.
    """
    entries = _benchmark_entries(workload, results=results, pair_rows=pair_rows)
    prefix = f"surge-host-parity/{workload}"
    for pair, rows in pair_rows.items():
        pair_prefix = f"{prefix}/{pair}"
        for row in rows:
            sample_prefix = f"{pair_prefix}/sample-{int(row['sample']):02d}"
            metric_values = (
                ("mel-rmse", "mel_rmse", float(row["mel_rmse"])),
                ("mss", "mss", float(row["mss"])),
                ("rms-envelope-cosine-distance", "1-cos", float(row["rms_distance"])),
                ("sot", "sot", float(row["sot"])),
                ("wmfcc", "wmfcc", float(row["wmfcc"])),
            )
            entries.extend(
                {
                    "name": f"{sample_prefix}/{metric}",
                    "unit": unit,
                    "value": value,
                }
                for metric, unit, value in metric_values
            )
    return entries


def _write_audio_artifacts(
    output_dir: Path,
    results: dict[ParityBackend, _BackendResult],
) -> None:
    """Write one backend-named WAV per workload row.

    :param output_dir: Workload comparison artifact root.
    :param results: Materialized real artifacts keyed by rendering backend.
    """
    render_count = len(next(iter(results.values())).audio)
    sample_rate = _config("pedalboard", render_count).sample_rate
    for index in range(render_count):
        sample_dir = output_dir / "audio" / f"sample_{index:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for backend, result in results.items():
            wavfile.write(
                sample_dir / f"{backend}.wav",
                sample_rate,
                result.audio[index].T,
            )


def _write_mel_artifacts(
    output_dir: Path,
    results: dict[ParityBackend, _BackendResult],
) -> None:
    """Write persisted mel arrays and viewable previews.

    :param output_dir: Workload comparison artifact root.
    :param results: Materialized real artifacts keyed by rendering backend.
    """
    render_count = len(next(iter(results.values())).audio)
    for index in range(render_count):
        mel_dir = output_dir / "mel" / f"sample_{index:02d}"
        mel_dir.mkdir(parents=True, exist_ok=True)
        for backend, result in results.items():
            np.save(mel_dir / f"{backend}.npy", result.mel[index])
            plt.imsave(mel_dir / f"{backend}.png", np.concatenate(result.mel[index]), cmap="magma")


def _comparison_manifest(
    workload: str,
    results: dict[ParityBackend, _BackendResult],
) -> dict[str, object]:
    """Build provenance and threshold metadata for one workload.

    :param workload: Listening workload identity.
    :param results: Materialized real artifacts keyed by rendering backend.
    :returns: JSON-serializable comparison manifest.
    """
    parameter_map = load_param_map(_PARAMETER_MAP_PATH)
    render_count = len(next(iter(results.values())).audio)
    config = _config("pedalboard", render_count)
    manifest: dict[str, object] = {
        "artifact_schema_version": 2,
        "workload": workload,
        "backends": {
            backend: {
                "filename": f"{backend}.wav",
                "renderer_version": _config(backend, render_count).synth.synth_version,
            }
            for backend in results
        },
        "container_image": os.environ.get("SYNTH_SETTER_BENCHMARK_IMAGE"),
        "git_sha": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "parameter_map": str(_PARAMETER_MAP_PATH),
        "parameter_map_preset_sha256": parameter_map.preset_sha256,
        "surgepy_preset": str(_SURGEPY_PRESET_PATH),
        "surgepy_preset_sha256": parameter_map.surgepy_preset_sha256,
        "render_count": render_count,
        "sample_rate": config.sample_rate,
        "signal_duration_seconds": config.signal_duration_seconds,
        "onset_gate": {
            "amplitude": _ONSET_AMPLITUDE,
            "control_backends": list(_TRUSTED_BACKENDS),
            "control_lag_tolerance_samples": _ONSET_CONTROL_LAG_TOLERANCE_SAMPLES,
        },
        "workload_gates": {
            "adjacent_mel_rmse_min": _ADJACENT_MEL_RMSE_MIN,
            "diverse_centroid_shift_min": _DIVERSE_CENTROID_SHIFT_MIN,
        },
        "thresholds": {
            "mel_rmse_max": _MEL_RMSE_MAX,
            "mss_max": _MSS_MAX,
            "rms_envelope_cosine_min": _RMS_MIN,
            "sot_max": _SOT_MAX,
            "wmfcc_max": _WMFCC_MAX,
        },
    }
    if workload == _RANDOM_WORKLOAD:
        manifest.update(
            {
                "diagnostic_only": True,
                "green_definition": (
                    f"All {_RANDOM_PATCH_COUNT} fixed random patches rendered and diagnostics "
                    "were recorded; audio quality parity is not asserted."
                ),
                "sampler": {
                    "api": "resolve_param_spec(config.param_spec_name).sample(rng)",
                    "generator": "numpy.random.default_rng",
                    "seed": _RANDOM_PATCH_SEED,
                    "synth_parameter_count": _SURGE_SYNTH_PARAMETER_COUNT,
                },
                "loudness_policy": {
                    "min_loudness_db": str(_RANDOM_MIN_LOUDNESS),
                    "rationale": _RANDOM_LOUDNESS_OVERRIDE_RATIONALE,
                },
                "onset_diagnostics": {
                    "amplitude": _ONSET_AMPLITUDE,
                    "asserted": False,
                },
                "onset_gate": None,
                "thresholds": {},
                "workload_gates": {},
            }
        )
    return manifest


def _write_comparison_directory(
    output_dir: Path,
    workload: str,
    *,
    results: dict[ParityBackend, _BackendResult],
    synth_params: list[dict[str, float]],
    pair_rows: dict[str, list[MetricRow]],
    onset_rows: list[OnsetRow] | list[DiagnosticOnsetRow],
) -> None:
    """Write pairwise listening files and per-render diagnostics.

    :param output_dir: Persistent workload comparison destination.
    :param workload: Listening workload identity.
    :param results: Materialized real artifacts keyed by rendering backend.
    :param synth_params: Exact normalized patches shared by every backend.
    :param pair_rows: Per-sample quality metrics keyed by backend pair.
    :param onset_rows: Per-render onset diagnostics.
    """
    shutil.rmtree(output_dir, ignore_errors=True)
    _write_audio_artifacts(output_dir, results)
    _write_mel_artifacts(output_dir, results)
    parameters = [
        {
            "sample": index,
            "encoded_normalized_vector": results["pedalboard"]
            .params[index]
            .astype(float)
            .tolist(),
            "midi_event": {
                "note": _HARDCODED_NOTE_PARAMS["pitch"],
                "note_start_and_end_seconds": _HARDCODED_NOTE_PARAMS["note_start_and_end"],
                "velocity": _config("pedalboard", len(synth_params)).velocity,
            },
            "normalized_synth_parameters": patch,
        }
        for index, patch in enumerate(synth_params)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    metrics: dict[str, object] = {"onsets": onset_rows, "pairwise": pair_rows}
    if workload == _RANDOM_WORKLOAD:
        metrics["worst_pairwise"] = {
            pair: _worst_pair_metrics(rows) for pair, rows in pair_rows.items()
        }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output_dir / "manifest.json").write_text(
        json.dumps(_comparison_manifest(workload, results), indent=2) + "\n"
    )


def _export_workload_artifacts(
    workload: str,
    *,
    results: dict[ParityBackend, _BackendResult],
    synth_params: list[dict[str, float]],
    pair_rows: dict[str, list[MetricRow]],
    onset_rows: list[OnsetRow] | list[DiagnosticOnsetRow],
) -> None:
    """Write and consume schema-v2 artifacts when an output root is configured.

    :param workload: Listening workload identity.
    :param results: Materialized real artifacts keyed by rendering backend.
    :param synth_params: Exact normalized patches shared by every backend.
    :param pair_rows: Per-sample quality metrics keyed by backend pair.
    :param onset_rows: Per-render onset diagnostics.
    """
    if not (output_dir := os.environ.get("SURGE_PARITY_OUTPUT_DIR")):
        return
    workload_dir = Path(output_dir) / workload
    _write_comparison_directory(
        workload_dir,
        workload,
        results=results,
        synth_params=synth_params,
        pair_rows=pair_rows,
        onset_rows=onset_rows,
    )
    _assert_comparison_artifact(
        workload_dir,
        workload,
        synth_params=synth_params,
        pairs=list(pair_rows),
    )


def _assert_structural_artifact_contract(
    workload: str,
    results: dict[ParityBackend, _BackendResult],
    render_count: int,
) -> None:
    """Validate every backend's finite Lance output and parameter identity.

    :param workload: Workload identity included in failures.
    :param results: Materialized artifacts keyed by backend.
    :param render_count: Expected row count.
    """
    config = _config("pedalboard", render_count)
    expected_samples = int(config.sample_rate * config.signal_duration_seconds)
    expected_param_width = resolve_param_spec(config.param_spec_name).encoded_width
    for backend, result in results.items():
        identity = {"workload": workload, "backend": backend}
        assert result.audio.shape == (render_count, 2, expected_samples), identity
        assert result.mel.shape == (render_count, *_EXPECTED_MEL_SHAPE), identity
        assert result.params.shape == (render_count, expected_param_width), identity
        assert result.params.dtype == np.float32, identity
        for sample in range(render_count):
            row_identity = {**identity, "sample": sample}
            assert np.isfinite(result.audio[sample]).all(), row_identity
            assert np.isfinite(result.mel[sample]).all(), row_identity
            assert np.isfinite(result.params[sample]).all(), row_identity
            assert np.all((result.params[sample] >= 0.0) & (result.params[sample] <= 1.0)), (
                row_identity
            )
            assert np.max(np.abs(result.audio[sample])) <= 1.0, row_identity
    np.testing.assert_array_equal(results["pedalboard"].params, results["dawdreamer"].params)
    np.testing.assert_array_equal(results["pedalboard"].params, results["surgepy"].params)


def _assert_artifact_contract(
    workload: str,
    results: dict[ParityBackend, _BackendResult],
    render_count: int,
) -> None:
    """Gate finite structure and signal amplitude for parity workloads.

    :param workload: Workload identity included in failures.
    :param results: Materialized artifacts keyed by backend.
    :param render_count: Expected row count.
    """
    _assert_structural_artifact_contract(workload, results, render_count)
    for backend, result in results.items():
        for sample in range(render_count):
            identity = {"workload": workload, "backend": backend, "sample": sample}
            assert np.max(np.abs(result.audio[sample])) > 1e-4, identity


def _assert_manifest_artifact(output_dir: Path, workload: str, render_count: int) -> None:
    """Consume workload identity and quality limits.

    :param output_dir: Workload comparison artifact root.
    :param workload: Expected workload identity.
    :param render_count: Expected workload row count.
    """
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["artifact_schema_version"] == 2
    assert manifest["workload"] == workload
    assert manifest["render_count"] == render_count
    assert set(manifest["backends"]) == set(_BACKENDS)
    for backend in _BACKENDS:
        assert manifest["backends"][backend]["filename"] == f"{backend}.wav"
    assert "container_image" in manifest
    assert "git_sha" in manifest
    assert "github_run_id" in manifest
    if workload == _RANDOM_WORKLOAD:
        assert manifest["diagnostic_only"] is True
        assert "rendered and diagnostics were recorded" in manifest["green_definition"]
        assert "quality parity is not asserted" in manifest["green_definition"]
        assert manifest["sampler"] == {
            "api": "resolve_param_spec(config.param_spec_name).sample(rng)",
            "generator": "numpy.random.default_rng",
            "seed": _RANDOM_PATCH_SEED,
            "synth_parameter_count": _SURGE_SYNTH_PARAMETER_COUNT,
        }
        assert manifest["loudness_policy"] == {
            "min_loudness_db": str(_RANDOM_MIN_LOUDNESS),
            "rationale": _RANDOM_LOUDNESS_OVERRIDE_RATIONALE,
        }
        assert manifest["onset_diagnostics"]["asserted"] is False
        assert manifest["onset_gate"] is None
        assert manifest["workload_gates"] == {}
        assert manifest["thresholds"] == {}
        return
    assert manifest["workload_gates"] == {
        "adjacent_mel_rmse_min": _ADJACENT_MEL_RMSE_MIN,
        "diverse_centroid_shift_min": _DIVERSE_CENTROID_SHIFT_MIN,
    }
    assert manifest["thresholds"] == {
        "mel_rmse_max": _MEL_RMSE_MAX,
        "mss_max": _MSS_MAX,
        "rms_envelope_cosine_min": _RMS_MIN,
        "sot_max": _SOT_MAX,
        "wmfcc_max": _WMFCC_MAX,
    }


def _assert_parameter_artifact(
    output_dir: Path,
    synth_params: list[dict[str, float]],
) -> None:
    """Consume persisted patches, MIDI events, and encoded vectors.

    :param output_dir: Workload comparison artifact root.
    :param synth_params: Exact normalized patches expected in row order.
    """
    parameters = json.loads((output_dir / "parameters.json").read_text())
    render_count = len(synth_params)
    assert len(parameters) == render_count
    assert {row["sample"] for row in parameters} == set(range(render_count))
    config = _config("pedalboard", render_count)
    param_spec = resolve_param_spec(config.param_spec_name)
    persisted_midi = {
        "note": _HARDCODED_NOTE_PARAMS["pitch"],
        "note_start_and_end_seconds": list(_HARDCODED_NOTE_PARAMS["note_start_and_end"]),
        "velocity": config.velocity,
    }
    encoded_midi = {
        "pitch": persisted_midi["note"],
        "note_start_and_end": tuple(persisted_midi["note_start_and_end_seconds"]),
        "velocity": persisted_midi["velocity"],
    }
    for row, expected_patch in zip(parameters, synth_params, strict=True):
        assert set(row) == {
            "sample",
            "encoded_normalized_vector",
            "midi_event",
            "normalized_synth_parameters",
        }
        assert row["normalized_synth_parameters"] == expected_patch
        assert row["midi_event"] == persisted_midi
        expected_vector = param_spec.encode(expected_patch, encoded_midi)
        np.testing.assert_array_equal(row["encoded_normalized_vector"], expected_vector)


def _assert_metrics_artifact(
    output_dir: Path,
    workload: str,
    render_count: int,
    pairs: list[str],
) -> None:
    """Consume complete finite onset and pairwise diagnostic rows.

    :param output_dir: Workload comparison artifact root.
    :param workload: Expected workload identity.
    :param render_count: Expected workload row count.
    :param pairs: Ordered backend pair names.
    """
    metrics = json.loads((output_dir / "metrics.json").read_text())
    expected_keys = {"onsets", "pairwise"}
    if workload == _RANDOM_WORKLOAD:
        expected_keys.add("worst_pairwise")
    assert set(metrics) == expected_keys
    assert len(metrics["onsets"]) == render_count * len(_BACKENDS)
    for row in metrics["onsets"]:
        assert set(row) == {
            "backend",
            "sample",
            "onset_sample",
            "requested_sample",
        }
        assert row["backend"] in _BACKENDS
        assert type(row["sample"]) is int
        if workload == _RANDOM_WORKLOAD:
            assert row["onset_sample"] is None or type(row["onset_sample"]) is int
        else:
            assert type(row["onset_sample"]) is int
        assert type(row["requested_sample"]) is int
    assert {(row["backend"], row["sample"]) for row in metrics["onsets"]} == set(
        product(_BACKENDS, range(render_count))
    )
    assert set(metrics["pairwise"]) == set(pairs)
    for rows in metrics["pairwise"].values():
        assert len(rows) == render_count
        assert {row["sample"] for row in rows} == set(range(render_count))
        for row in rows:
            values = [
                row[metric]
                for metric in ("mel_rmse", "mss", "rms", "rms_distance", "sot", "wmfcc")
            ]
            assert all(map(math.isfinite, values))
    if workload == _RANDOM_WORKLOAD:
        assert metrics["worst_pairwise"] == {
            pair: _worst_pair_metrics(rows) for pair, rows in metrics["pairwise"].items()
        }


def _assert_comparison_json_artifacts(
    output_dir: Path,
    workload: str,
    *,
    synth_params: list[dict[str, float]],
    pairs: list[str],
) -> None:
    """Consume every persisted JSON contract for one workload.

    :param output_dir: Workload comparison artifact root.
    :param workload: Expected workload identity.
    :param synth_params: Exact normalized patches expected in row order.
    :param pairs: Ordered backend pair names.
    """
    render_count = len(synth_params)
    _assert_manifest_artifact(output_dir, workload, render_count)
    _assert_parameter_artifact(output_dir, synth_params)
    _assert_metrics_artifact(output_dir, workload, render_count, pairs)


def _assert_audio_and_mel_artifacts(output_dir: Path, render_count: int) -> None:
    """Consume each backend WAV and its same-named persisted mel.

    :param output_dir: Workload comparison artifact root.
    :param render_count: Expected workload row count.
    """
    config = _config("pedalboard", render_count)
    expected_samples = int(config.sample_rate * config.signal_duration_seconds)
    wav_paths = sorted((output_dir / "audio").glob("sample_*/*.wav"))
    assert len(wav_paths) == render_count * len(_BACKENDS)
    assert {path.parent.name for path in wav_paths} == {
        f"sample_{index:02d}" for index in range(render_count)
    }
    assert {path.name for path in wav_paths} == {f"{backend}.wav" for backend in _BACKENDS}
    for path in wav_paths:
        sample_rate, audio = wavfile.read(path)
        assert sample_rate == config.sample_rate, path
        assert audio.shape == (expected_samples, 2), path
        assert audio.dtype == np.float32, path
        assert np.isfinite(audio).all(), path
        mel_path = output_dir / "mel" / path.parent.name / f"{path.stem}.npy"
        persisted_mel = np.load(mel_path)
        assert persisted_mel.shape == _EXPECTED_MEL_SHAPE, mel_path
        assert persisted_mel.dtype == np.float32, mel_path
        recomputed_mel = make_spectrogram(audio.T, sample_rate)
        np.testing.assert_allclose(persisted_mel, recomputed_mel, rtol=1e-6, atol=1e-6)
    assert len(list((output_dir / "mel").glob("sample_*/*.npy"))) == len(wav_paths)
    assert len(list((output_dir / "mel").glob("sample_*/*.png"))) == len(wav_paths)


def _assert_comparison_artifact(
    output_dir: Path,
    workload: str,
    *,
    synth_params: list[dict[str, float]],
    pairs: list[str],
) -> None:
    """Consume every file in one schema-v2 listening artifact.

    :param output_dir: Workload comparison artifact root.
    :param workload: Expected workload identity.
    :param synth_params: Exact normalized patches expected in row order.
    :param pairs: Ordered backend pair names.
    """
    _assert_audio_and_mel_artifacts(output_dir, len(synth_params))
    _assert_comparison_json_artifacts(
        output_dir,
        workload,
        synth_params=synth_params,
        pairs=pairs,
    )


def _db_to_power(values: np.ndarray) -> np.ndarray:
    """Convert power-decibel values to linear power.

    :param values: Arbitrarily shaped decibel values.
    :returns: Linear power values with the input shape.
    """
    return np.power(10.0, values / 10.0)


def _mel_centroids(mel: np.ndarray) -> np.ndarray:
    """Return one energy-weighted mel-bin centroid per render.

    :param mel: Persisted mel tensors shaped ``(rows, channels, bins, frames)``.
    :returns: Floating-point centroid per row.
    """
    mel_bins = np.arange(mel.shape[2], dtype=np.float64)[None, :, None]
    energy = _db_to_power(mel.astype(np.float64))
    return np.sum(energy * mel_bins, axis=(1, 2, 3)) / np.sum(energy, axis=(1, 2, 3))


def _assert_directional_audio_diversity(
    results: dict[ParityBackend, _BackendResult],
) -> None:
    """Require every patch step to change spectrum and end at a higher centroid.

    :param results: Materialized diverse-patch artifacts keyed by backend.
    """
    for backend, result in results.items():
        centroids = _mel_centroids(result.mel)
        adjacent_mel_rmse = np.sqrt(np.mean(np.diff(result.mel, axis=0) ** 2, axis=(1, 2, 3)))
        assert np.all(adjacent_mel_rmse > _ADJACENT_MEL_RMSE_MIN), {
            "workload": "diverse-patches",
            "backend": backend,
            "adjacent_mel_rmse": adjacent_mel_rmse.tolist(),
        }
        assert centroids[-1] - centroids[0] > _DIVERSE_CENTROID_SHIFT_MIN, {
            "workload": "diverse-patches",
            "backend": backend,
            "mel_centroids": centroids.tolist(),
        }


def _dominant_frequencies(audio: np.ndarray) -> np.ndarray:
    """Return the strongest audible FFT-bin frequency per render.

    :param audio: Channel-leading waveforms shaped ``(rows, channels, samples)``.
    :returns: Dominant frequency in Hz per row.
    """
    mono = audio.mean(axis=1)
    magnitudes = np.abs(np.fft.rfft(mono, axis=1))
    sample_rate = _config("pedalboard", len(audio)).sample_rate
    frequencies = np.fft.rfftfreq(audio.shape[2], 1.0 / sample_rate)
    audible = (frequencies >= _DOMINANT_FREQUENCY_MIN_HZ) & (
        frequencies <= _DOMINANT_FREQUENCY_MAX_HZ
    )
    peak_indexes = np.argmax(magnitudes[:, audible], axis=1)
    return frequencies[audible][peak_indexes]


def _assert_causal_parameter_response(
    results: dict[ParityBackend, _BackendResult],
) -> None:
    """Require independent cutoff and oscillator-one-octave changes per backend.

    :param results: Baseline, cutoff-only, and octave-only renders by backend.
    """
    for backend, result in results.items():
        centroids = _mel_centroids(result.mel)
        cutoff_centroid_shift = centroids[1] - centroids[0]
        dominant_frequencies = _dominant_frequencies(result.audio)
        octave_frequency_ratio = dominant_frequencies[2] / dominant_frequencies[0]
        assert cutoff_centroid_shift > _CAUSAL_CENTROID_SHIFT_MIN, {
            "workload": "causal-parameters",
            "backend": backend,
            "sample": 1,
            "cutoff_centroid_shift": cutoff_centroid_shift,
        }
        assert octave_frequency_ratio > _CAUSAL_OCTAVE_FREQUENCY_RATIO_MIN, {
            "workload": "causal-parameters",
            "backend": backend,
            "sample": 2,
            "octave_frequency_ratio": octave_frequency_ratio,
        }


def _evaluate_workload(
    tmp_path: Path,
    workload: str,
    synth_params: list[dict[str, float]],
) -> tuple[
    dict[ParityBackend, _BackendResult],
    dict[str, list[MetricRow]],
    list[OnsetRow],
]:
    """Render and gate one real three-host workload.

    :param tmp_path: Temporary dataset root.
    :param workload: Stable workload name.
    :param synth_params: Exact normalized patches shared by all hosts.
    :returns: Validated results, pairwise metrics, and onset rows.
    """
    results = _render_workload(tmp_path, workload, synth_params)
    _assert_artifact_contract(workload, results, len(synth_params))
    pair_rows = _pair_metrics(results)
    onset_rows = _onset_rows(results)
    _assert_pair_metrics(workload, pair_rows)
    _assert_onset_parity(workload, onset_rows)
    return results, pair_rows, onset_rows


def _collect_random_diagnostics(
    tmp_path: Path,
    synth_params: list[dict[str, float]],
) -> tuple[
    dict[ParityBackend, _BackendResult],
    dict[str, list[MetricRow]],
    list[DiagnosticOnsetRow],
]:
    """Render random patches and collect diagnostics without quality gates.

    :param tmp_path: Temporary dataset root.
    :param synth_params: Exact normalized ParamSpec samples shared by all hosts.
    :returns: Structurally valid results, pairwise metrics, and onset rows.
    """
    workload = _RANDOM_WORKLOAD
    results = _render_workload(
        tmp_path,
        workload,
        synth_params,
        min_loudness=_RANDOM_MIN_LOUDNESS,
    )
    _assert_structural_artifact_contract(workload, results, _RANDOM_PATCH_COUNT)
    assert np.unique(results["pedalboard"].params, axis=0).shape[0] == _RANDOM_PATCH_COUNT
    sample_rate = _random_patch_config("pedalboard").sample_rate
    return results, _pair_metrics(results), _diagnostic_onset_rows(results, sample_rate)


def _run_parity_workload(
    tmp_path: Path,
    workload: str,
    synth_params: list[dict[str, float]],
) -> dict[ParityBackend, _BackendResult]:
    """Render, benchmark, export, and gate one parity workload.

    :param tmp_path: Temporary dataset root.
    :param workload: Stable workload name.
    :param synth_params: Exact normalized patches shared by all hosts.
    :returns: Validated results keyed by backend.
    """
    results, pair_rows, onset_rows = _evaluate_workload(tmp_path, workload, synth_params)
    _emit_benchmark_metrics(
        entries=_benchmark_entries(workload, results=results, pair_rows=pair_rows),
        bench_filename=f"surge-host-parity-{workload}.json",
    )
    _export_workload_artifacts(
        workload,
        results=results,
        synth_params=synth_params,
        pair_rows=pair_rows,
        onset_rows=onset_rows,
    )
    return results


def _run_random_diagnostic_workload(
    tmp_path: Path,
    synth_params: list[dict[str, float]],
) -> dict[ParityBackend, _BackendResult]:
    """Render and export the fixed random corpus without quality assertions.

    :param tmp_path: Temporary dataset root.
    :param synth_params: Exact normalized ParamSpec samples shared by all hosts.
    :returns: Structurally validated results keyed by backend.
    """
    workload = _RANDOM_WORKLOAD
    results, pair_rows, onset_rows = _collect_random_diagnostics(tmp_path, synth_params)
    _emit_benchmark_metrics(
        entries=_diagnostic_benchmark_entries(
            workload,
            results=results,
            pair_rows=pair_rows,
        ),
        bench_filename="surge-host-parity-random-patches.json",
    )
    _export_workload_artifacts(
        workload,
        results=results,
        synth_params=synth_params,
        pair_rows=pair_rows,
        onset_rows=onset_rows,
    )
    return results


def _require_surge_xt() -> None:
    """Skip this real fixture when the selected CI synth is not Surge XT."""
    if TEST_SYNTH != "surge_xt":
        pytest.skip("three-host parity fixture uses Surge XT")
    assert surge_component_state(_VST_PRESET_PATH) == surge_component_state(_SURGEPY_PRESET_PATH)


def test_onset_gate_rejects_common_early_render() -> None:
    """Absolute timing rejects common-mode early onset."""
    rows: list[OnsetRow] = [
        {
            "backend": backend,
            "sample": 0,
            "onset_sample": 335,
            "requested_sample": 336,
        }
        for backend in _BACKENDS
    ]

    with pytest.raises(AssertionError, match="early host onset"):
        _assert_onset_parity("repeated-patch", rows)


def test_onset_gate_rejects_lag_from_either_trusted_host() -> None:
    """A later trusted control cannot hide lag from the earlier control."""
    rows: list[OnsetRow] = [
        {
            "backend": "pedalboard",
            "sample": 0,
            "onset_sample": 336,
            "requested_sample": 336,
        },
        {
            "backend": "dawdreamer",
            "sample": 0,
            "onset_sample": 338,
            "requested_sample": 336,
        },
        {
            "backend": "surgepy",
            "sample": 0,
            "onset_sample": 340,
            "requested_sample": 336,
        },
    ]

    with pytest.raises(AssertionError, match="late host onset"):
        _assert_onset_parity("diverse-patches", rows)


def test_onset_gate_rejects_backend_lag() -> None:
    """Trusted-host timing rejects a delayed SurgePy render."""
    rows: list[OnsetRow] = [
        {
            "backend": backend,
            "sample": 0,
            "onset_sample": 340 if backend == "surgepy" else 336,
            "requested_sample": 336,
        }
        for backend in _BACKENDS
    ]

    with pytest.raises(AssertionError, match="late host onset"):
        _assert_onset_parity("diverse-patches", rows)


def test_random_patch_sampler_same_seed_reproduces_full_param_corpus() -> None:
    """Pin the corpus hash so sampler changes cannot alter benchmark comparability."""
    config = _config("pedalboard", _RANDOM_PATCH_COUNT)

    patches = _sample_random_patches(config, seed=_RANDOM_PATCH_SEED)
    repeated = _sample_random_patches(config, seed=_RANDOM_PATCH_SEED)
    assert patches == repeated
    param_spec = resolve_param_spec(config.param_spec_name)
    expected_names = {parameter.name for parameter in param_spec.synth_params}
    assert len(patches) == _RANDOM_PATCH_COUNT
    assert {frozenset(patch) for patch in patches} == {frozenset(expected_names)}
    assert {len(patch) for patch in patches} == {_SURGE_SYNTH_PARAMETER_COUNT}

    parameter_order = sorted(expected_names)
    corpus_values = np.asarray(
        [[patch[name] for name in parameter_order] for patch in patches],
        dtype="<f8",
    )
    assert np.unique(corpus_values, axis=0).shape[0] == _RANDOM_PATCH_COUNT
    assert hashlib.sha256(corpus_values.tobytes()).hexdigest() == (
        "baa7c0a6511ff99f86074d13814078b217f5ef3bd6506b76351e2b1a875a9694"
    )


def test_random_patch_sampler_different_seed_changes_ordered_corpus() -> None:
    """Ensure deterministic replay still depends on the supplied generator seed."""
    config = _config("pedalboard", _RANDOM_PATCH_COUNT)

    assert _sample_random_patches(config, seed=_RANDOM_PATCH_SEED) != _sample_random_patches(
        config, seed=20260331
    )


def test_random_diagnostic_entries_retain_out_of_limit_metric_rows() -> None:
    """Preserve diagnostic values that would fail the strict parity gates."""
    row = _OUT_OF_LIMIT_METRIC_ROW
    result = _BackendResult(
        audio=np.zeros((1, 2, 8), dtype=np.float32),
        mel=np.zeros((1, *_EXPECTED_MEL_SHAPE), dtype=np.float32),
        params=np.zeros((1, 1), dtype=np.float32),
        total_seconds=1.0,
    )

    entries = _diagnostic_benchmark_entries(
        "random-patches",
        results={backend: result for backend in _BACKENDS},
        pair_rows={"pedalboard-vs-surgepy": [row]},
    )

    values = {entry["name"]: entry["value"] for entry in entries}
    prefix = "surge-host-parity/random-patches/pedalboard-vs-surgepy"
    assert values[f"{prefix}/sample-07/mel-rmse"] == 50.0
    assert values[f"{prefix}/sample-07/mss"] == 60.0
    assert values[f"{prefix}/sample-07/rms-envelope-cosine-distance"] == 0.5
    assert values[f"{prefix}/sample-07/sot"] == 0.1
    assert values[f"{prefix}/sample-07/wmfcc"] == 20.0
    assert values[f"{prefix}/mel_rmse-max"] == 50.0


@pytest.mark.parametrize("workload", ["repeated-patch", "diverse-patches"])
def test_gated_workloads_reject_out_of_limit_random_metric_row(workload: str) -> None:
    """Protect strict parity gates from diagnostic-workload relaxation.

    :param workload: Published workload whose quality gates must remain active.
    """
    row = _OUT_OF_LIMIT_METRIC_ROW

    with pytest.raises(AssertionError):
        _assert_pair_metrics(workload, {"pedalboard-vs-surgepy": [row]})


def test_random_patch_config_scopes_loudness_override() -> None:
    """Only random diagnostics disable the production loudness floor."""
    regular = _config("pedalboard", _RANDOM_PATCH_COUNT)
    diagnostic = _random_patch_config("pedalboard")

    assert regular.min_loudness == -55.0
    assert diagnostic.min_loudness == _RANDOM_MIN_LOUDNESS


def test_pair_metric_rows_store_cosine_distance_as_one_minus_similarity() -> None:
    """Keep persisted RMS similarity and distance mathematically paired."""
    config = _config("pedalboard", 1)
    sample_count = int(config.sample_rate * config.signal_duration_seconds)
    time_axis = np.arange(sample_count, dtype=np.float32) / config.sample_rate
    mono = np.sin(2 * np.pi * 220.0 * time_axis, dtype=np.float32)
    audio = np.stack([[mono, mono]])
    result = _BackendResult(
        audio=audio,
        mel=np.zeros((1, *_EXPECTED_MEL_SHAPE), dtype=np.float32),
        params=np.zeros((1, 1), dtype=np.float32),
        total_seconds=0.0,
    )

    row = _pair_metric_rows(result, result)[0]

    assert row["rms"] == pytest.approx(1.0)
    assert row["rms_distance"] == pytest.approx(0.0)
    assert row["rms_distance"] == pytest.approx(1.0 - row["rms"])


def test_pair_gate_reports_workload_backend_pair_and_sample() -> None:
    """A single anomalous render fails with complete case identity."""
    rows: list[MetricRow] = [
        {
            "sample": 7,
            "mel_rmse": 5.1,
            "mss": 0.0,
            "rms": 1.0,
            "rms_distance": 0.0,
            "sot": 0.0,
            "wmfcc": 0.0,
        }
    ]

    with pytest.raises(AssertionError) as error:
        _assert_pair_metrics("diverse-patches", {"pedalboard-vs-surgepy": rows})

    message = str(error.value)
    assert "diverse-patches" in message
    assert "pedalboard-vs-surgepy" in message
    assert "'sample': 7" in message


def test_repeated_backend_stability_rejects_anomalous_render() -> None:
    """One divergent repeated mel fails with backend and sample identity."""
    result = _BackendResult(
        audio=np.zeros((2, 2, 8), dtype=np.float32),
        mel=np.stack(
            [
                np.zeros(_EXPECTED_MEL_SHAPE, dtype=np.float32),
                np.full(_EXPECTED_MEL_SHAPE, 5.1, dtype=np.float32),
            ]
        ),
        params=np.zeros((2, 1), dtype=np.float32),
        total_seconds=0.0,
    )

    with pytest.raises(AssertionError, match="pedalboard.*sample.*1"):
        _assert_repeated_backend_stability({"pedalboard": result})


def test_repeated_backend_stability_accepts_documented_boundary() -> None:
    """Inclusive repeated-render limit accepts mel RMSE at the boundary."""
    result = _BackendResult(
        audio=np.zeros((2, 2, 8), dtype=np.float32),
        mel=np.stack(
            [
                np.zeros(_EXPECTED_MEL_SHAPE, dtype=np.float32),
                np.full(_EXPECTED_MEL_SHAPE, 5.0, dtype=np.float32),
            ]
        ),
        params=np.zeros((2, 1), dtype=np.float32),
        total_seconds=0.0,
    )

    _assert_repeated_backend_stability({"pedalboard": result})


def test_pair_metrics_accept_documented_boundary_values() -> None:
    """Inclusive quality limits accept metrics exactly on each boundary."""
    rows: list[MetricRow] = [
        {
            "sample": 0,
            "mel_rmse": 5.0,
            "mss": 6.0,
            "rms": 0.99,
            "rms_distance": 0.01,
            "sot": 0.01,
            "wmfcc": 2.0,
        }
    ]

    _assert_pair_metrics("boundary", {"pedalboard-vs-surgepy": rows})


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(
            {
                "sample": 0,
                "mel_rmse": 5.1,
                "mss": 0.0,
                "rms": 1.0,
                "rms_distance": 0.0,
                "sot": 0.0,
                "wmfcc": 0.0,
            },
            id="mel-rmse",
        ),
        pytest.param(
            {
                "sample": 0,
                "mel_rmse": 0.0,
                "mss": 6.1,
                "rms": 1.0,
                "rms_distance": 0.0,
                "sot": 0.0,
                "wmfcc": 0.0,
            },
            id="mss",
        ),
        pytest.param(
            {
                "sample": 0,
                "mel_rmse": 0.0,
                "mss": 0.0,
                "rms": 0.98,
                "rms_distance": 0.02,
                "sot": 0.0,
                "wmfcc": 0.0,
            },
            id="rms",
        ),
        pytest.param(
            {
                "sample": 0,
                "mel_rmse": 0.0,
                "mss": 0.0,
                "rms": 1.0,
                "rms_distance": 0.0,
                "sot": 0.02,
                "wmfcc": 0.0,
            },
            id="sot",
        ),
        pytest.param(
            {
                "sample": 0,
                "mel_rmse": 0.0,
                "mss": 0.0,
                "rms": 1.0,
                "rms_distance": 0.0,
                "sot": 0.0,
                "wmfcc": 2.1,
            },
            id="wmfcc",
        ),
    ],
)
def test_pair_metrics_reject_documented_limit_violation(row: MetricRow) -> None:
    """A metric outside its documented quality limit fails independently.

    :param row: One parity row with exactly one out-of-contract metric.
    """
    with pytest.raises(AssertionError):
        _assert_pair_metrics("boundary", {"pedalboard-vs-surgepy": [row]})


def test_manifest_rejects_backend_filename_swap(tmp_path: Path) -> None:
    """Manifest backend keys must identify their same-named WAVs.

    :param tmp_path: Temporary schema-v2 artifact root.
    """
    manifest = {
        "artifact_schema_version": 2,
        "workload": "diverse-patches",
        "backends": {
            "pedalboard": {"filename": "surgepy.wav"},
            "dawdreamer": {"filename": "dawdreamer.wav"},
            "surgepy": {"filename": "pedalboard.wav"},
        },
        "container_image": "local-vst",
        "git_sha": "abc123",
        "github_run_id": "local",
        "render_count": 1,
        "workload_gates": {
            "adjacent_mel_rmse_min": _ADJACENT_MEL_RMSE_MIN,
            "diverse_centroid_shift_min": _DIVERSE_CENTROID_SHIFT_MIN,
        },
        "thresholds": {
            "mel_rmse_max": _MEL_RMSE_MAX,
            "mss_max": _MSS_MAX,
            "rms_envelope_cosine_min": _RMS_MIN,
            "sot_max": _SOT_MAX,
            "wmfcc_max": _WMFCC_MAX,
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(AssertionError):
        _assert_manifest_artifact(tmp_path, "diverse-patches", 1)


def test_comparison_json_rejects_missing_pair_metrics(tmp_path: Path) -> None:
    """The artifact consumer rejects incomplete persisted diagnostics.

    :param tmp_path: Temporary comparison artifact root.
    """
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": 2,
                "workload": "diverse-patches",
                "backends": {backend: {"filename": f"{backend}.wav"} for backend in _BACKENDS},
                "container_image": "local-vst",
                "git_sha": "abc123",
                "github_run_id": "local",
                "render_count": 1,
                "workload_gates": {
                    "adjacent_mel_rmse_min": _ADJACENT_MEL_RMSE_MIN,
                    "diverse_centroid_shift_min": _DIVERSE_CENTROID_SHIFT_MIN,
                },
                "thresholds": {
                    "mel_rmse_max": _MEL_RMSE_MAX,
                    "mss_max": _MSS_MAX,
                    "rms_envelope_cosine_min": _RMS_MIN,
                    "sot_max": _SOT_MAX,
                    "wmfcc_max": _WMFCC_MAX,
                },
            }
        )
    )
    config = _config("pedalboard", 1)
    persisted_midi = {
        "note": _HARDCODED_NOTE_PARAMS["pitch"],
        "note_start_and_end_seconds": list(_HARDCODED_NOTE_PARAMS["note_start_and_end"]),
        "velocity": config.velocity,
    }
    encoded_midi = {
        "pitch": persisted_midi["note"],
        "note_start_and_end": tuple(persisted_midi["note_start_and_end_seconds"]),
        "velocity": persisted_midi["velocity"],
    }
    vector = resolve_param_spec(config.param_spec_name).encode(
        _PARITY_SYNTH_PARAMS,
        encoded_midi,
    )
    (tmp_path / "parameters.json").write_text(
        json.dumps(
            [
                {
                    "sample": 0,
                    "encoded_normalized_vector": vector.tolist(),
                    "midi_event": persisted_midi,
                    "normalized_synth_parameters": _PARITY_SYNTH_PARAMS,
                }
            ]
        )
    )
    onset_rows = [
        {
            "backend": backend,
            "sample": 0,
            "onset_sample": 336,
            "requested_sample": 336,
        }
        for backend in _BACKENDS
    ]
    (tmp_path / "metrics.json").write_text(json.dumps({"onsets": onset_rows, "pairwise": {}}))

    with pytest.raises(AssertionError):
        _assert_comparison_json_artifacts(
            tmp_path,
            "diverse-patches",
            synth_params=[_PARITY_SYNTH_PARAMS],
            pairs=["pedalboard-vs-surgepy"],
        )


def test_backend_named_artifact_rejects_wav_label_swap(tmp_path: Path) -> None:
    """A WAV moved under another backend name cannot match its persisted mel.

    :param tmp_path: Temporary schema-v2 artifact root.
    """
    config = _config("pedalboard", 1)
    samples = int(config.sample_rate * config.signal_duration_seconds)
    time_axis = np.arange(samples, dtype=np.float32) / config.sample_rate
    audio_by_backend = {
        "pedalboard": np.sin(2 * np.pi * 220.0 * time_axis, dtype=np.float32),
        "dawdreamer": np.sin(2 * np.pi * 440.0 * time_axis, dtype=np.float32),
        "surgepy": np.sin(2 * np.pi * 880.0 * time_axis, dtype=np.float32),
    }
    sample_dir = tmp_path / "audio" / "sample_00"
    mel_dir = tmp_path / "mel" / "sample_00"
    sample_dir.mkdir(parents=True)
    mel_dir.mkdir(parents=True)
    swapped_labels = {
        "pedalboard": "surgepy",
        "dawdreamer": "dawdreamer",
        "surgepy": "pedalboard",
    }
    for backend, mono in audio_by_backend.items():
        stereo = np.stack([mono, mono])
        wav_backend = swapped_labels[backend]
        wavfile.write(sample_dir / f"{wav_backend}.wav", config.sample_rate, stereo.T)
        np.save(mel_dir / f"{backend}.npy", make_spectrogram(stereo, config.sample_rate))

    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        _assert_audio_and_mel_artifacts(tmp_path, 1)


@pytest.mark.slow
@pytest.mark.requires_vst
@pytest.mark.requires_surgepy
def test_surge_hosts_repeated_patch_have_per_render_parity(tmp_path: Path) -> None:
    """Every deterministic repeated render satisfies three-host parity.

    :param tmp_path: Temporary destination for all three real Lance datasets.
    """
    _require_surge_xt()
    synth_params = [_PARITY_SYNTH_PARAMS.copy() for _ in range(_REPEATED_RENDER_COUNT)]

    results = _run_parity_workload(tmp_path, "repeated-patch", synth_params)

    _assert_repeated_backend_stability(results)


@pytest.mark.slow
@pytest.mark.requires_vst
@pytest.mark.requires_surgepy
def test_surge_hosts_diverse_patches_have_per_render_parity(tmp_path: Path) -> None:
    """Distinct deterministic patches retain timing and quality parity.

    :param tmp_path: Temporary destination for all three real Lance datasets.
    """
    _require_surge_xt()
    synth_params = [
        {
            **_PARITY_SYNTH_PARAMS,
            "a_filter_1_cutoff": cutoff,
            "a_osc_1_octave": octave,
        }
        for cutoff, octave in _DIVERSE_PATCH_VALUES
    ]

    results = _run_parity_workload(tmp_path, "diverse-patches", synth_params)

    assert np.unique(results["pedalboard"].params, axis=0).shape[0] == len(synth_params)
    _assert_directional_audio_diversity(results)


@pytest.mark.slow
@pytest.mark.requires_vst
@pytest.mark.requires_surgepy
def test_surge_hosts_random_patches_record_diagnostics_without_quality_gate(
    tmp_path: Path,
) -> None:
    """Exercise production Lance persistence for ungated random-corpus diagnostics.

    :param tmp_path: Temporary destination for all three real Lance datasets.
    """
    _require_surge_xt()
    config = _random_patch_config("pedalboard")
    synth_params = _sample_random_patches(config, seed=_RANDOM_PATCH_SEED)

    results = _run_random_diagnostic_workload(tmp_path, synth_params)

    assert all(len(patch) == _SURGE_SYNTH_PARAMETER_COUNT for patch in synth_params)
    assert np.unique(results["pedalboard"].params, axis=0).shape[0] == _RANDOM_PATCH_COUNT


@pytest.mark.slow
@pytest.mark.requires_vst
@pytest.mark.requires_surgepy
def test_surge_hosts_apply_cutoff_and_octave_independently(tmp_path: Path) -> None:
    """Each host responds independently to cutoff and oscillator-one octave.

    :param tmp_path: Temporary destination for all three real Lance datasets.
    """
    _require_surge_xt()
    baseline = {
        **_PARITY_SYNTH_PARAMS,
        "a_filter_1_cutoff": 0.2,
        "a_osc_1_octave": 0.1705,
        "a_osc_1_volume": 1.0,
        "a_osc_2_mute": 0.7505,
        "a_osc_3_mute": 0.7505,
    }
    synth_params = [
        baseline,
        {**baseline, "a_filter_1_cutoff": 0.8},
        {**baseline, "a_osc_1_octave": 0.8305},
    ]

    results, _, _ = _evaluate_workload(tmp_path, "causal-parameters", synth_params)

    _assert_causal_parameter_response(results)
