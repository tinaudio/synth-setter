"""Real three-host Surge rendering parity and listening artifacts."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Literal

import lance
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.io import wavfile

from synth_setter.data.vst.generate_vst_dataset import make_spectrogram
from synth_setter.data.vst.param_map import load_param_map
from synth_setter.data.vst.param_spec import NoteParams
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
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
type MetricRow = dict[str, float | int]

_BACKENDS: tuple[ParityBackend, ...] = ("pedalboard", "dawdreamer", "surgepy")
_REPEATED_RENDER_COUNT = 30
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
_MODIFIED_Z_FACTOR = 0.67448975
_MODIFIED_Z_MAX = 3.5
_ADJACENT_MEL_RMSE_MIN = 2.5
_CAUSAL_CENTROID_SHIFT_MIN = 1.0
_CAUSAL_OCTAVE_FREQUENCY_RATIO_MIN = 6.0
_DIVERSE_CENTROID_SHIFT_MIN = 7.0
_ONSET_AMPLITUDE = 1e-8
_ONSET_SCALE_FLOOR_SAMPLES = 2.0
_HOST_PAIR_THRESHOLDS = {
    "mel_rmse_max": 3.5,
    "mss_max": 1.0,
    "rms_min": 0.995,
    "sot_max": 0.01,
    "wmfcc_max": 2.0,
}
_PAIR_THRESHOLDS = {
    "dawdreamer-vs-surgepy": _HOST_PAIR_THRESHOLDS,
    "pedalboard-vs-dawdreamer": _HOST_PAIR_THRESHOLDS,
    "pedalboard-vs-surgepy": _HOST_PAIR_THRESHOLDS,
}
_PARAMETER_MAP_PATH = Path("src/synth_setter/data/vst/surge_xt_param_map.json")
_SURGEPY_PRESET_PATH = Path("presets/surge-base.fxp")
_VST_PRESET_PATH = Path("presets/surge-base.vstpreset")


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


def _config(backend: ParityBackend, render_count: int) -> RenderConfig:
    """Return one fixed production render configuration.

    :param backend: Host selected for the real dataset path.
    :param render_count: Number of rows in the workload.
    :returns: Validated render configuration.
    """
    values = {
        **_dawdreamer_experiment_config().model_dump(),
        "audio_dtype": "float32",
        "gui_toggle_cadence": "never",
        "plugin_reload_cadence": "render",
        "samples_per_shard": render_count,
        "renderer_backend": backend,
    }
    if backend == "surgepy":
        values["synth"] = {
            **values["synth"],
            "plugin_path": "surgepy",
            "plugin_state_path": str(_SURGEPY_PRESET_PATH),
            "synth_version": "1.3.master.f7b97c68",
        }
    return RenderConfig.model_validate(values)


def _render_dataset(
    backend: ParityBackend,
    path: Path,
    *,
    synth_params: list[dict[str, float]],
    note_params: list[NoteParams],
) -> _BackendResult:
    """Render and consume one real Lance dataset.

    :param backend: Host selected for rendering.
    :param path: Lance dataset destination.
    :param synth_params: Exact normalized patches shared by all hosts.
    :param note_params: Exact MIDI events shared by all hosts.
    :returns: Materialized columns and elapsed production-path time.
    """
    render_count = len(synth_params)
    started = time.perf_counter()
    make_lance_dataset(
        path,
        _config(backend, render_count),
        fixed_synth_params_list=synth_params,
        fixed_note_params_list=note_params,
    )
    elapsed = time.perf_counter() - started
    columns = lance.dataset(str(path)).to_table(
        columns=[AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD]
    )
    audio = columns.column(AUDIO_FIELD).combine_chunks().to_numpy_ndarray()
    mel = columns.column(MEL_SPEC_FIELD).combine_chunks().to_numpy_ndarray()
    config = _config(backend, render_count)
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
) -> dict[ParityBackend, _BackendResult]:
    """Render one exact patch corpus through every backend.

    :param tmp_path: Temporary dataset root.
    :param workload: Name used to isolate Lance paths.
    :param synth_params: Exact normalized patches shared by all hosts.
    :returns: Consumed results keyed by backend.
    """
    note_params = [_HARDCODED_NOTE_PARAMS.copy() for _ in synth_params]
    return {
        backend: _render_dataset(
            backend,
            tmp_path / f"{workload}-{backend}.lance",
            synth_params=synth_params,
            note_params=note_params,
        )
        for backend in _BACKENDS
    }


def _pair_metric_rows(
    reference: _BackendResult,
    candidate: _BackendResult,
) -> list[MetricRow]:
    """Return quality diagnostics for every matched artifact row.

    :param reference: First host's consumed dataset.
    :param candidate: Second host's consumed dataset.
    :returns: Per-row audio and persisted-mel metrics.
    """
    rows: list[MetricRow] = []
    for index in range(len(reference.audio)):
        mel_delta = reference.mel[index] - candidate.mel[index]
        rows.append(
            {
                "sample": index,
                "mel_rmse": float(np.sqrt(np.mean(np.square(mel_delta)))),
                "mss": float(compute_mss(reference.audio[index], candidate.audio[index])),
                "rms": float(compute_rms(reference.audio[index], candidate.audio[index])),
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


def _modified_z_scores(calibration: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Score one-dimensional observations against robust calibration values.

    :param calibration: Nonempty one-dimensional trusted-control values.
    :param observed: One-dimensional values to score without affecting calibration.
    :returns: One-sided modified z-scores for upper outliers.
    :raises ValueError: If either input violates its shape contract.
    """
    if calibration.ndim != 1 or observed.ndim != 1 or len(calibration) == 0:
        raise ValueError("modified z-score inputs must be nonempty one-dimensional arrays")
    median = float(np.median(calibration))
    mad = float(np.median(np.abs(calibration - median)))
    scale = max(mad, _ONSET_SCALE_FLOOR_SAMPLES)
    excess = np.maximum(observed - median, 0.0)
    return _MODIFIED_Z_FACTOR * excess / scale


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


def _early_onset_z_scores(
    controls: np.ndarray,
    *,
    observed: np.ndarray,
    expected: np.ndarray,
) -> np.ndarray:
    """Score per-patch early audio against independent trusted-host controls.

    :param controls: Onsets shaped ``(control backends, patches)``.
    :param observed: One-dimensional backend onset samples to score.
    :param expected: One-dimensional requested onset samples.
    :returns: One-sided upper modified z-score per patch.
    :raises ValueError: If inputs are empty or have incompatible shapes.
    """
    if (
        controls.ndim != 2
        or controls.shape[0] == 0
        or controls.shape[1:] != expected.shape
        or observed.ndim != 1
        or observed.shape != expected.shape
        or expected.ndim != 1
        or len(expected) == 0
    ):
        raise ValueError("matched onset inputs must contain controls and equal patch vectors")
    control_earliness = np.maximum(expected[None, :] - controls, 0)
    observed_earliness = np.maximum(expected - observed, 0)
    return np.asarray(
        [
            _modified_z_scores(control_earliness[:, index], observed_earliness[index : index + 1])[0]
            for index in range(len(expected))
        ]
    )


def _onset_rows(
    results: dict[ParityBackend, _BackendResult],
) -> list[dict[str, float | int | str]]:
    """Score every backend onset against Pedalboard/DawDreamer controls.

    :param results: Materialized artifacts keyed by backend.
    :returns: Per-backend, per-sample onset values and modified z-scores.
    """
    onsets = {backend: _onset_samples(result.audio) for backend, result in results.items()}
    render_count = len(onsets["pedalboard"])
    note_start = _HARDCODED_NOTE_PARAMS["note_start_and_end"][0]
    requested_sample = math.ceil(note_start * _config("pedalboard", render_count).sample_rate)
    expected = np.full(render_count, requested_sample)
    rows: list[dict[str, float | int | str]] = []
    for backend, values in onsets.items():
        control_backends = tuple(
            control for control in ("pedalboard", "dawdreamer") if control != backend
        )
        controls = np.stack([onsets[control] for control in control_backends])
        scores = _early_onset_z_scores(controls, observed=values, expected=expected)
        rows.extend(
            {
                "backend": backend,
                "sample": sample,
                "onset_sample": int(values[sample]),
                "requested_sample": requested_sample,
                "modified_z": float(scores[sample]),
            }
            for sample in range(len(values))
        )
    return rows


def _assert_no_onset_outliers(rows: list[dict[str, float | int | str]]) -> None:
    """Fail with backend and sample identity when any onset is anomalous.

    :param rows: Per-render onset scores.
    """
    early = [row for row in rows if int(row["onset_sample"]) < int(row["requested_sample"])]
    assert not early, f"early host onset(s): {early}"
    outliers = [row for row in rows if float(row["modified_z"]) > _MODIFIED_Z_MAX]
    assert not outliers, f"host onset outlier(s): {outliers}"


def _assert_pair_metrics(pair_rows: dict[str, list[MetricRow]]) -> None:
    """Fail on any matched render outside its host-pair quality bounds.

    :param pair_rows: Per-sample metrics keyed by backend pair.
    """
    for pair, rows in pair_rows.items():
        thresholds = _PAIR_THRESHOLDS[pair]
        for row in rows:
            identity = (pair, int(row["sample"]), row)
            assert float(row["mel_rmse"]) < thresholds["mel_rmse_max"], identity
            assert float(row["mss"]) < thresholds["mss_max"], identity
            assert float(row["rms"]) > thresholds["rms_min"], identity
            assert float(row["sot"]) < thresholds["sot_max"], identity
            assert float(row["wmfcc"]) < thresholds["wmfcc_max"], identity


def _worst_pair_metrics(rows: list[MetricRow]) -> dict[str, float]:
    """Reduce diagnostics for benchmark publication.

    :param rows: Per-row metrics for one backend pair.
    :returns: Worst distance values and minimum RMS-envelope cosine.
    """
    return {
        "mel_rmse": max(float(row["mel_rmse"]) for row in rows),
        "mss": max(float(row["mss"]) for row in rows),
        "rms": min(float(row["rms"]) for row in rows),
        "sot": max(float(row["sot"]) for row in rows),
        "wmfcc": max(float(row["wmfcc"]) for row in rows),
    }


def _backend_benchmark_entries(
    prefix: str,
    render_count: int,
    *,
    results: dict[ParityBackend, _BackendResult],
    onset_rows: list[dict[str, float | int | str]],
) -> list[BenchmarkEntry]:
    """Build per-backend timing and onset benchmark entries.

    :param prefix: Stable benchmark-series prefix.
    :param render_count: Number of workload rows.
    :param results: Real artifact results keyed by host.
    :param onset_rows: Per-render onset scores.
    :returns: Benchmark-action custom metric entries.
    """
    entries: list[BenchmarkEntry] = []
    for backend, result in results.items():
        entries.extend(
            [
                {
                    "name": f"{prefix}/{backend}/dataset-seconds-per-render",
                    "unit": "seconds",
                    "value": result.total_seconds / render_count,
                },
                {
                    "name": f"{prefix}/{backend}/onset-modified-z-max",
                    "unit": "z",
                    "value": max(
                        float(row["modified_z"]) for row in onset_rows if row["backend"] == backend
                    ),
                },
            ]
        )
    return entries


def _pair_benchmark_entries(
    prefix: str,
    pair_rows: dict[str, list[MetricRow]],
) -> list[BenchmarkEntry]:
    """Build per-pair quality benchmark entries.

    :param prefix: Stable benchmark-series prefix.
    :param pair_rows: Pairwise quality diagnostics.
    :returns: Benchmark-action custom metric entries.
    """
    entries: list[BenchmarkEntry] = []
    for pair, rows in pair_rows.items():
        metrics = _worst_pair_metrics(rows)
        thresholds = _PAIR_THRESHOLDS[pair]
        for metric in ("mel_rmse", "mss", "sot", "wmfcc"):
            entries.extend(
                (
                    {
                        "name": f"{prefix}/{pair}/{metric}-max",
                        "unit": metric,
                        "value": metrics[metric],
                    },
                    {
                        "name": f"{prefix}/{pair}/{metric}-threshold-overrun-max",
                        "unit": metric,
                        "value": metrics[metric] - thresholds[f"{metric}_max"],
                    },
                )
            )
        entries.extend(
            (
                {
                    "name": f"{prefix}/{pair}/rms-envelope-cosine-distance-max",
                    "unit": "1-cos",
                    "value": 1.0 - metrics["rms"],
                },
                {
                    "name": f"{prefix}/{pair}/rms-threshold-overrun-max",
                    "unit": "cosine",
                    "value": thresholds["rms_min"] - metrics["rms"],
                },
            )
        )
    return entries


def _benchmark_entries(
    workload: str,
    *,
    results: dict[ParityBackend, _BackendResult],
    pair_rows: dict[str, list[MetricRow]],
    onset_rows: list[dict[str, float | int | str]],
) -> list[BenchmarkEntry]:
    """Build benchmark-action entries for one workload.

    :param workload: Stable benchmark-series workload name.
    :param results: Real artifact results keyed by host.
    :param pair_rows: Pairwise quality diagnostics.
    :param onset_rows: Per-render onset scores.
    :returns: Benchmark-action custom metric entries.
    """
    render_count = len(next(iter(results.values())).audio)
    prefix = f"surge-host-parity/{workload}"
    return [
        {"name": f"{prefix}/render-count", "unit": "renders", "value": render_count},
        *_backend_benchmark_entries(
            prefix,
            render_count,
            results=results,
            onset_rows=onset_rows,
        ),
        *_pair_benchmark_entries(prefix, pair_rows),
    ]


def _write_audio(
    output_dir: Path,
    results: dict[ParityBackend, _BackendResult],
) -> None:
    """Write backend-named WAVs for unambiguous three-way listening.

    :param output_dir: Workload artifact root.
    :param results: Materialized artifacts keyed by backend.
    """
    render_count = len(next(iter(results.values())).audio)
    sample_rate = _config("pedalboard", render_count).sample_rate
    for index in range(render_count):
        sample_dir = output_dir / "audio" / f"sample_{index:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for backend, result in results.items():
            wavfile.write(sample_dir / f"{backend}.wav", sample_rate, result.audio[index].T)


def _write_mels(
    output_dir: Path,
    results: dict[ParityBackend, _BackendResult],
) -> None:
    """Write persisted mel arrays and previews.

    :param output_dir: Workload artifact root.
    :param results: Materialized artifacts keyed by backend.
    """
    render_count = len(next(iter(results.values())).audio)
    for index in range(render_count):
        mel_dir = output_dir / "mel" / f"sample_{index:02d}"
        mel_dir.mkdir(parents=True, exist_ok=True)
        for backend, result in results.items():
            np.save(mel_dir / f"{backend}.npy", result.mel[index])
            plt.imsave(
                mel_dir / f"{backend}.png",
                np.concatenate(result.mel[index]),
                cmap="magma",
            )


def _manifest(
    workload: str,
    results: dict[ParityBackend, _BackendResult],
) -> dict[str, object]:
    """Build provenance and threshold metadata.

    :param workload: Listening workload name.
    :param results: Materialized artifacts keyed by backend.
    :returns: JSON-serializable manifest.
    """
    render_count = len(next(iter(results.values())).audio)
    parameter_map = load_param_map(_PARAMETER_MAP_PATH)
    config = _config("pedalboard", render_count)
    return {
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
        "parameter_map": str(_PARAMETER_MAP_PATH),
        "parameter_map_preset_sha256": parameter_map.preset_sha256,
        "surgepy_preset": str(_SURGEPY_PRESET_PATH),
        "surgepy_preset_sha256": parameter_map.surgepy_preset_sha256,
        "pair_thresholds": _PAIR_THRESHOLDS,
        "render_count": render_count,
        "sample_rate": config.sample_rate,
        "signal_duration_seconds": config.signal_duration_seconds,
        "velocity": config.velocity,
        "onset_gate": {
            "amplitude": _ONSET_AMPLITUDE,
            "calibration_backends": ["pedalboard", "dawdreamer"],
            "modified_z_max": _MODIFIED_Z_MAX,
            "scale_floor_samples": _ONSET_SCALE_FLOOR_SAMPLES,
        },
    }


def _parameter_rows(
    results: dict[ParityBackend, _BackendResult],
    synth_params: list[dict[str, float]],
) -> list[dict[str, object]]:
    """Build the listening artifact's parameter rows.

    :param results: Materialized artifacts keyed by backend.
    :param synth_params: Exact normalized patches rendered by every backend.
    :returns: JSON-compatible parameter records in render order.
    """
    velocity = _config("pedalboard", len(synth_params)).velocity
    midi_event = {**_HARDCODED_NOTE_PARAMS, "velocity": velocity}
    return [
        {
            "sample": index,
            "encoded_normalized_vector": results["pedalboard"]
            .params[index]
            .astype(float)
            .tolist(),
            "midi_event": midi_event,
            "normalized_synth_parameters": patch,
        }
        for index, patch in enumerate(synth_params)
    ]


def _write_json(path: Path, payload: object) -> None:
    """Write one stable, newline-terminated JSON artifact.

    :param path: Artifact destination.
    :param payload: JSON-compatible value.
    """
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_workload(
    output_root: Path,
    workload: str,
    *,
    results: dict[ParityBackend, _BackendResult],
    synth_params: list[dict[str, float]],
    pair_rows: dict[str, list[MetricRow]],
    onset_rows: list[dict[str, float | int | str]],
) -> None:
    """Write one three-way listening and diagnostic artifact.

    :param output_root: Persistent artifact root shared by both workloads.
    :param workload: Workload subdirectory name.
    :param results: Materialized artifacts keyed by backend.
    :param synth_params: Exact normalized patches rendered by every backend.
    :param pair_rows: Pairwise quality diagnostics.
    :param onset_rows: Per-render onset scores.
    """
    output_dir = output_root / workload
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_audio(output_dir, results)
    _write_mels(output_dir, results)
    _write_json(output_dir / "parameters.json", _parameter_rows(results, synth_params))
    _write_json(
        output_dir / "metrics.json",
        {"onsets": onset_rows, "pairwise": pair_rows},
    )
    _write_json(output_dir / "manifest.json", _manifest(workload, results))


def _assert_wav_artifacts(output_dir: Path, render_count: int) -> None:
    """Validate every backend-named WAV through SciPy's real reader.

    :param output_dir: Workload artifact directory.
    :param render_count: Expected sample-directory count.
    """
    config = _config("surgepy", render_count)
    expected_shape = (
        int(config.sample_rate * config.signal_duration_seconds),
        config.channels,
    )
    wav_paths = sorted((output_dir / "audio").glob("sample_*/*.wav"))
    assert len(wav_paths) == render_count * len(_BACKENDS)
    assert {path.name for path in wav_paths} == {f"{backend}.wav" for backend in _BACKENDS}
    for path in wav_paths:
        sample_rate, audio = wavfile.read(path)
        assert sample_rate == config.sample_rate
        assert audio.shape == expected_shape
        assert audio.dtype == np.float32
        assert np.isfinite(audio).all()
        assert np.all((audio >= -1.0) & (audio <= 1.0))


def _assert_mel_artifacts(output_dir: Path, render_count: int) -> None:
    """Validate every persisted mel array and preview.

    :param output_dir: Workload artifact directory.
    :param render_count: Expected sample-directory count.
    """
    mel_paths = sorted((output_dir / "mel").glob("sample_*/*.npy"))
    png_paths = sorted((output_dir / "mel").glob("sample_*/*.png"))
    sample_rate = _config("pedalboard", render_count).sample_rate
    assert len(mel_paths) == render_count * len(_BACKENDS)
    assert len(png_paths) == render_count * len(_BACKENDS)
    for path in mel_paths:
        mel = np.load(path)
        assert mel.shape == (2, 128, 401)
        assert mel.dtype == np.float32
        assert np.isfinite(mel).all()
        _, audio = wavfile.read(output_dir / "audio" / path.parent.name / f"{path.stem}.wav")
        recomputed = make_spectrogram(audio.T, sample_rate)
        np.testing.assert_allclose(mel, recomputed, rtol=1e-6, atol=1e-6)


def _assert_parameter_artifact(output_dir: Path, render_count: int) -> None:
    """Validate parameter rows against the production parameter contract.

    :param output_dir: Workload artifact directory.
    :param render_count: Expected row count.
    """
    config = _config("surgepy", render_count)
    parameters = json.loads((output_dir / "parameters.json").read_text())
    assert len(parameters) == render_count
    assert {row["sample"] for row in parameters} == set(range(render_count))
    expected_midi = {
        "pitch": _HARDCODED_NOTE_PARAMS["pitch"],
        "note_start_and_end": list(_HARDCODED_NOTE_PARAMS["note_start_and_end"]),
        "velocity": config.velocity,
    }
    param_spec = resolve_param_spec(config.param_spec_name)
    expected_width = param_spec.encoded_width
    for row in parameters:
        assert set(row) == {
            "sample",
            "encoded_normalized_vector",
            "midi_event",
            "normalized_synth_parameters",
        }
        assert row["midi_event"] == expected_midi
        vector = np.asarray(row["encoded_normalized_vector"])
        assert vector.shape == (expected_width,)
        assert np.issubdtype(vector.dtype, np.number)
        assert np.isfinite(vector).all()
        assert np.all((vector >= 0.0) & (vector <= 1.0))
        patch = row["normalized_synth_parameters"]
        assert set(patch) == set(_PARITY_SYNTH_PARAMS)
        assert all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0.0 <= value <= 1.0
            for value in patch.values()
        )
        expected_vector = param_spec.encode(patch, expected_midi)
        np.testing.assert_array_equal(vector, expected_vector)


def _assert_metrics_artifact(output_dir: Path, render_count: int) -> None:
    """Validate the pairwise and onset metric row counts.

    :param output_dir: Workload artifact directory.
    :param render_count: Expected row count per backend or pair.
    """
    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert set(metrics) == {"onsets", "pairwise"}
    onset_fields = {
        "backend",
        "sample",
        "onset_sample",
        "requested_sample",
        "modified_z",
    }
    assert len(metrics["onsets"]) == render_count * len(_BACKENDS)
    for row in metrics["onsets"]:
        assert set(row) == onset_fields
        assert row["backend"] in _BACKENDS
        assert type(row["sample"]) is int
        assert type(row["onset_sample"]) is int
        assert type(row["requested_sample"]) is int
        assert isinstance(row["modified_z"], (int, float))
        assert not isinstance(row["modified_z"], bool)
        assert math.isfinite(row["modified_z"])
    assert {
        (row["backend"], row["sample"]) for row in metrics["onsets"]
    } == set(product(_BACKENDS, range(render_count)))

    assert set(metrics["pairwise"]) == set(_PAIR_THRESHOLDS)
    pair_fields = {"sample", "mel_rmse", "mss", "rms", "sot", "wmfcc"}
    for rows in metrics["pairwise"].values():
        assert len(rows) == render_count
        assert {row["sample"] for row in rows} == set(range(render_count))
        for row in rows:
            assert set(row) == pair_fields
            assert type(row["sample"]) is int
            for field in pair_fields - {"sample"}:
                assert isinstance(row[field], (int, float))
                assert not isinstance(row[field], bool)
                assert math.isfinite(row[field])


def _assert_manifest_artifact(output_dir: Path, workload: str, render_count: int) -> None:
    """Validate listening-artifact schema and provenance metadata.

    :param output_dir: Workload artifact directory.
    :param workload: Expected workload identity.
    :param render_count: Expected render count.
    """
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["artifact_schema_version"] == 2
    assert manifest["workload"] == workload
    assert manifest["render_count"] == render_count
    assert manifest["pair_thresholds"] == _PAIR_THRESHOLDS
    assert {entry["filename"] for entry in manifest["backends"].values()} == {
        f"{backend}.wav" for backend in _BACKENDS
    }


def _assert_json_artifacts(output_dir: Path, workload: str, render_count: int) -> None:
    """Validate schema-v2 diagnostics and provenance documents.

    :param output_dir: Workload artifact directory.
    :param workload: Expected workload identity.
    :param render_count: Expected row count in each document.
    """
    _assert_parameter_artifact(output_dir, render_count)
    _assert_metrics_artifact(output_dir, render_count)
    _assert_manifest_artifact(output_dir, workload, render_count)


def _assert_listening_artifact(
    output_root: Path,
    workload: str,
    render_count: int,
) -> None:
    """Consume every boundary in the exported schema-v2 artifact.

    :param output_root: Persistent artifact root shared by both workloads.
    :param workload: Workload subdirectory name.
    :param render_count: Expected sample-directory count.
    """
    output_dir = output_root / workload
    _assert_wav_artifacts(output_dir, render_count)
    _assert_mel_artifacts(output_dir, render_count)
    _assert_json_artifacts(output_dir, workload, render_count)


def _assert_artifact_contract(
    results: dict[ParityBackend, _BackendResult],
    render_count: int,
) -> None:
    """Validate shared Lance output shape and parameter identity.

    :param results: Materialized artifacts keyed by backend.
    :param render_count: Expected row count.
    """
    config = _config("pedalboard", render_count)
    expected_samples = int(config.sample_rate * config.signal_duration_seconds)
    expected_param_width = resolve_param_spec(config.param_spec_name).encoded_width
    for result in results.values():
        assert result.audio.shape == (render_count, 2, expected_samples)
        assert result.mel.shape == (render_count, 2, 128, 401)
        assert result.params.shape == (render_count, expected_param_width)
        assert result.params.dtype == np.float32
        assert np.isfinite(result.audio).all()
        assert np.isfinite(result.mel).all()
        assert np.isfinite(result.params).all()
        assert np.all((result.params >= 0.0) & (result.params <= 1.0))
        assert np.max(np.abs(result.audio)) > 1e-4
        assert np.max(np.abs(result.audio)) <= 1.0
    np.testing.assert_array_equal(results["pedalboard"].params, results["dawdreamer"].params)
    np.testing.assert_array_equal(results["pedalboard"].params, results["surgepy"].params)


def _db_to_power(values: np.ndarray) -> np.ndarray:
    """Convert power-decibel values to linear power.

    :param values: Decibel values.
    :returns: Linear power values.
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
        adjacent_mel_rmse = np.sqrt(
            np.mean(np.diff(result.mel, axis=0) ** 2, axis=(1, 2, 3))
        )
        assert np.all(adjacent_mel_rmse > _ADJACENT_MEL_RMSE_MIN), (
            backend,
            adjacent_mel_rmse.tolist(),
        )
        assert centroids[-1] - centroids[0] > _DIVERSE_CENTROID_SHIFT_MIN, (
            backend,
            centroids.tolist(),
        )


def _dominant_frequencies(audio: np.ndarray) -> np.ndarray:
    """Return the strongest audible FFT-bin frequency per render.

    :param audio: Channel-leading waveforms shaped ``(rows, channels, samples)``.
    :returns: Dominant frequency in Hz per row.
    """
    mono = audio.mean(axis=1)
    magnitudes = np.abs(np.fft.rfft(mono, axis=1))
    sample_rate = _config("pedalboard", len(audio)).sample_rate
    frequencies = np.fft.rfftfreq(audio.shape[2], 1.0 / sample_rate)
    audible = (frequencies >= 40.0) & (frequencies <= 5_000.0)
    peak_indexes = np.argmax(magnitudes[:, audible], axis=1)
    return frequencies[audible][peak_indexes]


def _assert_causal_parameter_response(
    results: dict[ParityBackend, _BackendResult],
) -> None:
    """Require independent cutoff and oscillator-octave changes per backend.

    :param results: Baseline, cutoff-only, and octave-only renders by backend.
    """
    for backend, result in results.items():
        centroids = _mel_centroids(result.mel)
        cutoff_centroid_shift = centroids[1] - centroids[0]
        dominant_frequencies = _dominant_frequencies(result.audio)
        octave_frequency_ratio = dominant_frequencies[2] / dominant_frequencies[0]
        assert cutoff_centroid_shift > _CAUSAL_CENTROID_SHIFT_MIN, (
            backend,
            cutoff_centroid_shift,
        )
        assert octave_frequency_ratio > _CAUSAL_OCTAVE_FREQUENCY_RATIO_MIN, (
            backend,
            octave_frequency_ratio,
        )


def _run_parity_workload(
    tmp_path: Path,
    workload: str,
    synth_params: list[dict[str, float]],
) -> dict[ParityBackend, _BackendResult]:
    """Render, export, benchmark, and gate one real three-host workload.

    :param tmp_path: Temporary dataset root.
    :param workload: Stable workload name.
    :param synth_params: Exact normalized patches shared by all hosts.
    :returns: Validated results keyed by backend.
    """
    results = _render_workload(tmp_path, workload, synth_params)
    _assert_artifact_contract(results, len(synth_params))
    pair_rows = _pair_metrics(results)
    onset_rows = _onset_rows(results)
    _emit_benchmark_metrics(
        entries=_benchmark_entries(
            workload,
            results=results,
            pair_rows=pair_rows,
            onset_rows=onset_rows,
        ),
        bench_filename=f"surge-host-parity-{workload}.json",
    )
    if output_dir := os.environ.get("SURGE_PARITY_OUTPUT_DIR"):
        output_root = Path(output_dir)
        _write_workload(
            output_root,
            workload,
            results=results,
            synth_params=synth_params,
            pair_rows=pair_rows,
            onset_rows=onset_rows,
        )
        _assert_listening_artifact(output_root, workload, len(synth_params))
    _assert_no_onset_outliers(onset_rows)
    _assert_pair_metrics(pair_rows)
    return results


def _require_surge_xt() -> None:
    """Skip this real fixture when the selected CI synth is not Surge XT."""
    if TEST_SYNTH != "surge_xt":
        pytest.skip("three-host parity fixture uses Surge XT")
    assert surge_component_state(_VST_PRESET_PATH) == surge_component_state(_SURGEPY_PRESET_PATH)


def test_onset_gate_rejects_all_hosts_early_together() -> None:
    """Absolute timing rejects common-mode early onset hidden by z-scores."""
    rows = [
        {
            "backend": backend,
            "sample": 0,
            "onset_sample": 335,
            "requested_sample": 336,
            "modified_z": 0.0,
        }
        for backend in _BACKENDS
    ]

    with pytest.raises(AssertionError, match="early host onset"):
        _assert_no_onset_outliers(rows)


def test_pair_benchmark_entries_include_threshold_overruns() -> None:
    """Published pair metrics expose signed distance beyond every gate."""
    rows: list[MetricRow] = [
        {
            "sample": 0,
            "mel_rmse": 1.0,
            "mss": 0.5,
            "rms": 0.999,
            "sot": 0.005,
            "wmfcc": 1.0,
        }
    ]

    entries = _pair_benchmark_entries(
        "surge-host-parity/test",
        {"pedalboard-vs-surgepy": rows},
    )

    assert {
        str(entry["name"])
        for entry in entries
        if "threshold-overrun" in str(entry["name"])
    } == {
        "surge-host-parity/test/pedalboard-vs-surgepy/mel_rmse-threshold-overrun-max",
        "surge-host-parity/test/pedalboard-vs-surgepy/mss-threshold-overrun-max",
        "surge-host-parity/test/pedalboard-vs-surgepy/rms-threshold-overrun-max",
        "surge-host-parity/test/pedalboard-vs-surgepy/sot-threshold-overrun-max",
        "surge-host-parity/test/pedalboard-vs-surgepy/wmfcc-threshold-overrun-max",
    }


def test_metrics_artifact_rejects_nonfinite_metric(tmp_path: Path) -> None:
    """Persisted diagnostics require complete finite metric rows.

    :param tmp_path: Temporary listening-artifact root.
    """
    onset_rows = [
        {
            "backend": backend,
            "sample": 0,
            "onset_sample": 336,
            "requested_sample": 336,
            "modified_z": 0.0,
        }
        for backend in _BACKENDS
    ]
    pair_row = {
        "sample": 0,
        "mel_rmse": float("nan"),
        "mss": 0.0,
        "rms": 1.0,
        "sot": 0.0,
        "wmfcc": 0.0,
    }
    _write_json(
        tmp_path / "metrics.json",
        {
            "onsets": onset_rows,
            "pairwise": {pair: [pair_row] for pair in _PAIR_THRESHOLDS},
        },
    )

    with pytest.raises(AssertionError):
        _assert_metrics_artifact(tmp_path, 1)


def test_wav_artifacts_reject_out_of_range_audio(tmp_path: Path) -> None:
    """Listening WAVs retain the normalized production audio contract.

    :param tmp_path: Temporary listening-artifact root.
    """
    config = _config("surgepy", 1)
    shape = (
        int(config.sample_rate * config.signal_duration_seconds),
        config.channels,
    )
    for backend in _BACKENDS:
        path = tmp_path / "audio" / "sample_00" / f"{backend}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        audio = np.zeros(shape, dtype=np.float32)
        if backend == "surgepy":
            audio[0, 0] = 1.01
        wavfile.write(path, config.sample_rate, audio)

    with pytest.raises(AssertionError):
        _assert_wav_artifacts(tmp_path, 1)


def test_parameter_artifact_rejects_vector_misaligned_with_patch(tmp_path: Path) -> None:
    """An encoded vector must represent its paired normalized patch.

    :param tmp_path: Temporary listening-artifact root.
    """
    config = _config("surgepy", 1)
    midi_event = {
        "pitch": _HARDCODED_NOTE_PARAMS["pitch"],
        "note_start_and_end": list(_HARDCODED_NOTE_PARAMS["note_start_and_end"]),
        "velocity": config.velocity,
    }
    vector = resolve_param_spec(config.param_spec_name).encode(
        _PARITY_SYNTH_PARAMS,
        midi_event,
    )
    vector[0] = 1.0 - vector[0]
    _write_json(
        tmp_path / "parameters.json",
        [
            {
                "sample": 0,
                "encoded_normalized_vector": vector.tolist(),
                "midi_event": midi_event,
                "normalized_synth_parameters": _PARITY_SYNTH_PARAMS,
            }
        ],
    )

    with pytest.raises(AssertionError):
        _assert_parameter_artifact(tmp_path, 1)


def test_early_onset_scores_accept_on_time_and_late_audio() -> None:
    """Only audio preceding the requested event contributes to the score."""
    scores = _early_onset_z_scores(
        np.asarray([[100.0, 200.0], [100.0, 200.0]]),
        observed=np.asarray([120.0, 220.0]),
        expected=np.asarray([100.0, 200.0]),
    )

    np.testing.assert_array_equal(scores, [0.0, 0.0])


def test_early_onset_scores_keep_patch_identity() -> None:
    """Matched requests expose a systematic early backend onset."""
    scores = _early_onset_z_scores(
        np.asarray([[100.0, 200.0], [100.0, 200.0]]),
        observed=np.asarray([80.0, 180.0]),
        expected=np.asarray([100.0, 200.0]),
    )

    np.testing.assert_allclose(scores, [6.7448975, 6.7448975])
    assert np.all(scores > _MODIFIED_Z_MAX)


def test_early_onset_scores_use_matched_patch_controls() -> None:
    """Heterogeneous controls cannot hide a per-patch early onset."""
    scores = _early_onset_z_scores(
        np.asarray([[80.0, 200.0], [80.0, 200.0]]),
        observed=np.asarray([80.0, 180.0]),
        expected=np.asarray([100.0, 200.0]),
    )

    np.testing.assert_allclose(scores, [0.0, 6.7448975])
    assert scores[1] > _MODIFIED_Z_MAX


def test_early_onset_scores_reject_trusted_backend_outlier() -> None:
    """A trusted backend is scored against the other control, not itself."""
    scores = _early_onset_z_scores(
        np.asarray([[336.0]]),
        observed=np.asarray([0.0]),
        expected=np.asarray([336.0]),
    )

    np.testing.assert_allclose(scores, [113.314278])
    assert scores[0] > _MODIFIED_Z_MAX


def test_early_onset_scores_mismatched_shapes_raise() -> None:
    """Patch rows cannot broadcast across a mismatched onset vector."""
    with pytest.raises(ValueError, match="controls and equal patch vectors"):
        _early_onset_z_scores(
            np.asarray([[100.0]]),
            observed=np.asarray([100.0, 200.0]),
            expected=np.asarray([100.0, 200.0]),
        )


def test_onset_samples_without_batch_dimension_raises() -> None:
    """A single waveform cannot be mistaken for a batch of rows."""
    with pytest.raises(ValueError, match=r"\(rows, channels, samples\)"):
        _onset_samples(np.zeros((2, 128), dtype=np.float32))


def test_modified_z_scores_flag_only_divergent_onset() -> None:
    """The robust gate tolerates controls and rejects a divergent onset."""
    scores = _modified_z_scores(
        np.asarray([100.0, 100.0, 100.0, 100.0]),
        np.asarray([100.0, 101.0, 112.0]),
    )

    np.testing.assert_allclose(scores, [0.0, 0.337244875, 4.0469385])
    assert scores[0] < _MODIFIED_Z_MAX
    assert scores[1] < _MODIFIED_Z_MAX
    assert scores[2] > _MODIFIED_Z_MAX


@pytest.mark.slow
@pytest.mark.requires_vst
@pytest.mark.requires_surgepy
def test_surge_hosts_repeated_patch_have_no_per_render_onset_outliers(
    tmp_path: Path,
) -> None:
    """Every repeated render satisfies independent trusted-host onset controls.

    :param tmp_path: Temporary destination for all three real Lance datasets.
    """
    _require_surge_xt()
    synth_params = [_PARITY_SYNTH_PARAMS.copy() for _ in range(_REPEATED_RENDER_COUNT)]

    _run_parity_workload(tmp_path, "repeated-patch", synth_params)


@pytest.mark.slow
@pytest.mark.requires_vst
@pytest.mark.requires_surgepy
def test_surge_hosts_diverse_patches_have_no_per_render_onset_outliers(
    tmp_path: Path,
) -> None:
    """Distinct patches retain three-host timing parity and listening diversity.

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
def test_surge_hosts_apply_cutoff_and_octave_independently(tmp_path: Path) -> None:
    """Each host dispatches independent cutoff and oscillator-three octave changes.

    :param tmp_path: Temporary destination for all three real Lance datasets.
    """
    _require_surge_xt()
    baseline = {
        **_PARITY_SYNTH_PARAMS,
        "a_filter_1_cutoff": 0.2,
        "a_osc_3_octave": 0.1705,
    }
    synth_params = [
        baseline,
        {**baseline, "a_filter_1_cutoff": 0.8},
        {**baseline, "a_osc_3_octave": 0.8305},
    ]

    results = _render_workload(tmp_path, "causal-parameters", synth_params)

    _assert_artifact_contract(results, len(synth_params))
    _assert_no_onset_outliers(_onset_rows(results))
    _assert_causal_parameter_response(results)
