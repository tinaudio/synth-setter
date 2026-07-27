"""Real three-host Surge rendering parity and listening artifacts."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal

import lance
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.io import wavfile

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

type ParityBackend = Literal["dawdreamer", "pedalboard", "surgepy"]
type MetricRow = dict[str, float | int]

_BACKENDS: tuple[ParityBackend, ...] = ("pedalboard", "dawdreamer", "surgepy")
_REPEATED_RENDER_COUNT = 30
_DIVERSE_PATCH_VALUES = (
    (0.08, 0.12),
    (0.18, 0.24),
    (0.28, 0.36),
    (0.38, 0.48),
    (0.52, 0.62),
    (0.64, 0.74),
    (0.76, 0.86),
    (0.88, 0.96),
)
_MODIFIED_Z_FACTOR = 0.67448975
_MODIFIED_Z_MAX = 3.5
_ONSET_AMPLITUDE = 1e-8
_ONSET_SCALE_FLOOR_SAMPLES = 2.0
_CLOSE_PAIR_THRESHOLDS = {
    "mel_rmse_max": 9.0,
    "mss_max": 6.0,
    "rms_min": 0.94,
    "sot_max": 0.05,
    "wmfcc_max": 8.0,
}
_VST_PAIR_THRESHOLDS = {
    "mel_rmse_max": 24.0,
    "mss_max": 21.0,
    "rms_min": 0.8,
    "sot_max": 0.34,
    "wmfcc_max": 22.5,
}
_PAIR_THRESHOLDS = {
    "dawdreamer-vs-surgepy": _VST_PAIR_THRESHOLDS,
    "pedalboard-vs-dawdreamer": _VST_PAIR_THRESHOLDS,
    "pedalboard-vs-surgepy": _CLOSE_PAIR_THRESHOLDS,
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
    """Score observations against a robust independent calibration.

    :param calibration: Trusted control observations used to fit location and scale.
    :param observed: Values to score without affecting calibration.
    :returns: One-sided modified z-scores for upper outliers.
    """
    median = float(np.median(calibration))
    mad = float(np.median(np.abs(calibration - median)))
    scale = max(mad, _ONSET_SCALE_FLOOR_SAMPLES)
    excess = np.maximum(observed - median, 0.0)
    return _MODIFIED_Z_FACTOR * excess / scale


def _onset_samples(audio: np.ndarray) -> np.ndarray:
    """Return the first audible sample in each channel-leading waveform.

    :param audio: Batch of channel-leading waveforms.
    :returns: Integer onset sample per row; a silent row fails the test.
    """
    onsets: list[int] = []
    for index, waveform in enumerate(audio):
        audible = np.flatnonzero(np.max(np.abs(waveform), axis=0) > _ONSET_AMPLITUDE)
        assert len(audible) > 0, f"sample {index} is silent"
        onsets.append(int(audible[0]))
    return np.asarray(onsets)


def _matched_onset_z_scores(
    pedalboard: np.ndarray,
    dawdreamer: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Score matched onsets against each row's trusted-host center.

    :param pedalboard: Pedalboard onset sample per patch.
    :param dawdreamer: DawDreamer onset sample per patch.
    :param observed: Backend onset samples to score row by row.
    :returns: One-sided upper modified z-score per patch.
    """
    centers = (pedalboard + dawdreamer) / 2.0
    control_deviations = np.concatenate(
        (np.abs(pedalboard - centers), np.abs(dawdreamer - centers))
    )
    return _modified_z_scores(control_deviations, np.abs(observed - centers))


def _onset_rows(
    results: dict[ParityBackend, _BackendResult],
) -> list[dict[str, float | int | str]]:
    """Score every backend onset against Pedalboard/DawDreamer controls.

    :param results: Materialized artifacts keyed by backend.
    :returns: Per-backend, per-sample onset values and modified z-scores.
    """
    onsets = {backend: _onset_samples(result.audio) for backend, result in results.items()}
    rows: list[dict[str, float | int | str]] = []
    for backend, values in onsets.items():
        scores = _matched_onset_z_scores(onsets["pedalboard"], onsets["dawdreamer"], values)
        rows.extend(
            {
                "backend": backend,
                "sample": sample,
                "onset_sample": int(values[sample]),
                "modified_z": float(scores[sample]),
            }
            for sample in range(len(values))
        )
    return rows


def _assert_no_onset_outliers(rows: list[dict[str, float | int | str]]) -> None:
    """Fail with backend and sample identity when any onset is anomalous.

    :param rows: Per-render onset scores.
    """
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


def _benchmark_entries(
    workload: str,
    *,
    results: dict[ParityBackend, _BackendResult],
    pair_rows: dict[str, list[MetricRow]],
    onset_rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | str]]:
    """Build benchmark-action entries for quality, timing, and onset parity.

    :param workload: Stable benchmark-series workload name.
    :param results: Real artifact results keyed by host.
    :param pair_rows: Pairwise quality diagnostics.
    :param onset_rows: Per-render onset scores.
    :returns: Benchmark-action custom metric entries.
    """
    render_count = len(next(iter(results.values())).audio)
    prefix = f"surge-host-parity/{workload}"
    entries: list[dict[str, float | str]] = [
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
                    "name": f"{prefix}/{backend}/onset-modified-z-max",
                    "unit": "z",
                    "value": max(
                        float(row["modified_z"]) for row in onset_rows if row["backend"] == backend
                    ),
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
    _write_audio(output_dir, results)
    _write_mels(output_dir, results)
    config = _config("pedalboard", len(synth_params))
    midi_event = {**_HARDCODED_NOTE_PARAMS, "velocity": config.velocity}
    parameters = [
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
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    (output_dir / "metrics.json").write_text(
        json.dumps({"onsets": onset_rows, "pairwise": pair_rows}, indent=2) + "\n"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(_manifest(workload, results), indent=2) + "\n"
    )


def _assert_listening_artifact(
    output_root: Path,
    workload: str,
    render_count: int,
) -> None:
    """Consume the exported schema-v2 listening artifact.

    :param output_root: Persistent artifact root shared by both workloads.
    :param workload: Workload subdirectory name.
    :param render_count: Expected sample-directory count.
    """
    output_dir = output_root / workload
    wav_paths = sorted((output_dir / "audio").glob("sample_*/*.wav"))
    assert len(wav_paths) == render_count * len(_BACKENDS)
    assert {path.name for path in wav_paths} == {f"{backend}.wav" for backend in _BACKENDS}
    sample_rate, audio = wavfile.read(output_dir / "audio" / "sample_00" / "surgepy.wav")
    config = _config("surgepy", render_count)
    assert sample_rate == config.sample_rate
    assert audio.shape == (
        int(config.sample_rate * config.signal_duration_seconds),
        config.channels,
    )
    parameters = json.loads((output_dir / "parameters.json").read_text())
    assert len(parameters) == render_count
    assert {row["midi_event"]["velocity"] for row in parameters} == {config.velocity}


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


def test_matched_onset_scores_accept_trusted_host_midpoint() -> None:
    """An onset closer than either trusted control is not an upper outlier."""
    scores = _matched_onset_z_scores(
        np.asarray([100.0]),
        np.asarray([122.0]),
        np.asarray([111.0]),
    )

    np.testing.assert_array_equal(scores, [0.0])


def test_matched_onset_scores_keep_patch_identity() -> None:
    """Patch-dependent controls cannot hide a systematic backend offset."""
    scores = _matched_onset_z_scores(
        np.asarray([100.0, 200.0]),
        np.asarray([100.0, 200.0]),
        np.asarray([120.0, 220.0]),
    )

    np.testing.assert_allclose(scores, [6.7448975, 6.7448975])
    assert np.all(scores > _MODIFIED_Z_MAX)


def test_modified_z_scores_flag_only_divergent_onset() -> None:
    """The frozen robust gate tolerates controls and rejects a 12-sample offset."""
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
    """Every repeated render starts with the two trusted host controls.

    :param tmp_path: Temporary destination for all three real Lance datasets.
    """
    _require_surge_xt()
    synth_params = [_HARDCODED_SYNTH_PARAMS.copy() for _ in range(_REPEATED_RENDER_COUNT)]

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
            **_HARDCODED_SYNTH_PARAMS,
            "a_filter_1_cutoff": cutoff,
            "a_osc_1_pitch": pitch,
        }
        for cutoff, pitch in _DIVERSE_PATCH_VALUES
    ]

    results = _run_parity_workload(tmp_path, "diverse-patches", synth_params)

    assert np.unique(results["pedalboard"].params, axis=0).shape[0] == len(synth_params)
