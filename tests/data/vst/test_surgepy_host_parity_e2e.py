"""Real three-host Surge rendering parity and throughput benchmarks."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal, cast

import lance
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.io import wavfile

from synth_setter.data.vst.param_map import load_param_map
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.data.vst.shapes import AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD
from synth_setter.data.vst.surgepy_runtime import surge_component_state
from synth_setter.data.vst.writers import make_lance_dataset
from synth_setter.evaluation.compute_audio_metrics import (
    compute_mss,
    compute_rms,
    compute_sot,
    compute_wmfcc,
    find_possible_subdirs,
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

_BACKENDS: tuple[ParityBackend, ...] = ("pedalboard", "dawdreamer", "surgepy")
_RENDER_COUNT = 30
_MEL_RMSE_MAX = 25.0
_MSS_MAX = 22.0
_PARAMETER_MAP_PATH = Path("src/synth_setter/data/vst/surge_xt_param_map.json")
_SURGEPY_PRESET_PATH = Path("presets/surge-base.fxp")
_VST_PRESET_PATH = Path("presets/surge-base.vstpreset")
_RMS_MIN = 0.8
_SOT_MAX = 0.35
_WMFCC_MAX = 25.0


@dataclass(frozen=True)
class _BackendResult:
    """Materialized production artifact and elapsed dataset-generation time.

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


def _config(backend: ParityBackend) -> RenderConfig:
    """Return one fixed production render configuration for a parity backend.

    :param backend: Host selected for the real dataset path.
    :returns: Validated fixed-workload render configuration.
    """
    values = {
        **_dawdreamer_experiment_config().model_dump(),
        "audio_dtype": "float32",
        "gui_toggle_cadence": "never",
        "plugin_reload_cadence": "render",
        "samples_per_shard": _RENDER_COUNT,
        "renderer_backend": backend,
    }
    if backend == "surgepy":
        values["synth"] = {
            **values["synth"],
            "plugin_path": "surgepy",
            "plugin_state_path": str(_SURGEPY_PRESET_PATH),
        }
        values["synth"]["synth_version"] = "1.3.master.f7b97c68"
    return RenderConfig.model_validate(values)


def _render_dataset(
    backend: ParityBackend,
    path: Path,
) -> _BackendResult:
    """Render and consume one real fixed-workload Lance dataset.

    :param backend: Host selected for rendering.
    :param path: Lance dataset destination.
    :returns: Materialized columns and elapsed production-path time.
    """
    started = time.perf_counter()
    make_lance_dataset(
        path,
        _config(backend),
        fixed_synth_params_list=[_HARDCODED_SYNTH_PARAMS] * _RENDER_COUNT,
        fixed_note_params_list=[_HARDCODED_NOTE_PARAMS] * _RENDER_COUNT,
    )
    elapsed = time.perf_counter() - started
    dataset = lance.dataset(str(path))
    columns = dataset.to_table(columns=[AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD])
    audio = columns.column(AUDIO_FIELD).combine_chunks().to_numpy_ndarray()
    mel = columns.column(MEL_SPEC_FIELD).combine_chunks().to_numpy_ndarray()
    config = _config(backend)
    assert audio.dtype == np.dtype(config.audio_dtype)
    assert mel.dtype == np.dtype(config.mel_spec_dtype)
    return _BackendResult(
        audio=audio,
        mel=mel,
        params=columns.column(PARAM_ARRAY_FIELD).combine_chunks().to_numpy_ndarray(),
        total_seconds=elapsed,
    )


def _pair_metric_rows(
    reference: _BackendResult,
    candidate: _BackendResult,
) -> list[dict[str, float | int]]:
    """Return quality metrics for every matched artifact row.

    :param reference: First host's consumed dataset.
    :param candidate: Second host's consumed dataset.
    :returns: Per-row audio and persisted-mel comparison metrics.
    """
    rows: list[dict[str, float | int]] = []
    for index in range(_RENDER_COUNT):
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


def _worst_pair_metrics(rows: list[dict[str, float | int]]) -> dict[str, float]:
    """Reduce per-row metrics to threshold and benchmark summaries.

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
    results: dict[ParityBackend, _BackendResult],
    pair_metrics: dict[str, dict[str, float]],
) -> list[dict[str, float | str]]:
    """Build customSmallerIsBetter entries for quality and throughput trends.

    :param results: Real artifact results keyed by host.
    :param pair_metrics: Pairwise quality summaries keyed by host pair.
    :returns: Benchmark-action custom metric entries.
    """
    entries: list[dict[str, float | str]] = [
        {
            "name": "surge-host-parity/render-count",
            "unit": "renders",
            "value": float(_RENDER_COUNT),
        }
    ]
    for backend, result in results.items():
        seconds_per_render = result.total_seconds / _RENDER_COUNT
        entries.extend(
            [
                {
                    "name": f"surge-host-parity/{backend}/dataset-seconds-per-render",
                    "unit": "seconds",
                    "value": seconds_per_render,
                },
                {
                    "name": f"surge-host-parity/{backend}/dataset-realtime-factor",
                    "unit": "ratio",
                    "value": seconds_per_render / _config(backend).signal_duration_seconds,
                },
            ]
        )
    for pair, metrics in pair_metrics.items():
        for metric in ("mel_rmse", "mss", "sot", "wmfcc"):
            entries.append(
                {
                    "name": f"surge-host-parity/{pair}/{metric}-max",
                    "unit": metric,
                    "value": metrics[metric],
                }
            )
        entries.append(
            {
                "name": f"surge-host-parity/{pair}/rms-envelope-cosine-distance-max",
                "unit": "1-cos",
                "value": 1.0 - metrics["rms"],
            }
        )
    return entries


def _write_audio_comparisons(
    output_dir: Path,
    results: dict[ParityBackend, _BackendResult],
    pairs: list[str],
) -> None:
    """Write pairwise WAV directories accepted by the repository evaluator.

    :param output_dir: Comparison artifact root.
    :param results: Materialized real artifacts keyed by rendering backend.
    :param pairs: Ordered backend pair names.
    """
    sample_rate = _config("pedalboard").sample_rate
    for pair in pairs:
        raw_reference_name, raw_candidate_name = pair.split("-vs-")
        reference_name = cast(ParityBackend, raw_reference_name)
        candidate_name = cast(ParityBackend, raw_candidate_name)
        for index in range(_RENDER_COUNT):
            sample_dir = output_dir / "audio" / pair / f"sample_{index:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            wavfile.write(
                sample_dir / "target.wav",
                sample_rate,
                results[reference_name].audio[index].T,
            )
            wavfile.write(
                sample_dir / "pred.wav",
                sample_rate,
                results[candidate_name].audio[index].T,
            )


def _write_mel_artifacts(
    output_dir: Path,
    results: dict[ParityBackend, _BackendResult],
) -> None:
    """Write persisted mel arrays and viewable previews.

    :param output_dir: Comparison artifact root.
    :param results: Materialized real artifacts keyed by rendering backend.
    """
    for index in range(_RENDER_COUNT):
        mel_dir = output_dir / "mel" / f"{index:02d}"
        mel_dir.mkdir(parents=True, exist_ok=True)
        for backend, result in results.items():
            np.save(mel_dir / f"{backend}.npy", result.mel[index])
            plt.imsave(
                mel_dir / f"{backend}.png",
                np.concatenate(result.mel[index]),
                cmap="magma",
            )


def _comparison_manifest(
    results: dict[ParityBackend, _BackendResult],
) -> dict[str, object]:
    """Build provenance and threshold metadata for an exported comparison.

    :param results: Materialized real artifacts keyed by rendering backend.
    :returns: JSON-serializable comparison manifest.
    """
    parameter_map = load_param_map(_PARAMETER_MAP_PATH)
    config = _config("pedalboard")
    return {
        "backends": {
            backend: {"renderer_version": _config(backend).synth.synth_version}
            for backend in results
        },
        "container_image": os.environ.get("SYNTH_SETTER_BENCHMARK_IMAGE"),
        "git_sha": os.environ.get("GITHUB_SHA"),
        "parameter_map": str(_PARAMETER_MAP_PATH),
        "parameter_map_preset_sha256": parameter_map.preset_sha256,
        "surgepy_preset": str(_SURGEPY_PRESET_PATH),
        "surgepy_preset_sha256": parameter_map.surgepy_preset_sha256,
        "render_count": _RENDER_COUNT,
        "sample_rate": config.sample_rate,
        "signal_duration_seconds": config.signal_duration_seconds,
        "thresholds": {
            "mel_rmse_max": _MEL_RMSE_MAX,
            "mss_max": _MSS_MAX,
            "rms_envelope_cosine_min": _RMS_MIN,
            "sot_max": _SOT_MAX,
            "wmfcc_max": _WMFCC_MAX,
        },
    }


def _write_comparison_directory(
    output_dir: Path,
    results: dict[ParityBackend, _BackendResult],
    pair_rows: dict[str, list[dict[str, float | int]]],
    pair_metrics: dict[str, dict[str, float]],
) -> None:
    """Write evaluator-friendly files derived from consumed Lance artifacts.

    :param output_dir: Persistent benchmark comparison destination.
    :param results: Materialized real artifacts keyed by rendering backend.
    :param pair_rows: Per-sample quality metrics keyed by backend pair.
    :param pair_metrics: Worst-case quality metrics keyed by backend pair.
    """
    shutil.rmtree(output_dir, ignore_errors=True)
    _write_audio_comparisons(output_dir, results, list(pair_rows))
    _write_mel_artifacts(output_dir, results)
    parameters = {
        "encoded_normalized_vector": results["pedalboard"].params[0].astype(float).tolist(),
        "midi_event": {
            "note": _HARDCODED_NOTE_PARAMS["pitch"],
            "note_start_and_end_seconds": _HARDCODED_NOTE_PARAMS["note_start_and_end"],
            "velocity": 100,
        },
        "normalized_synth_parameters": _HARDCODED_SYNTH_PARAMS,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    (output_dir / "metrics.json").write_text(
        json.dumps({"per_sample": pair_rows, "worst": pair_metrics}, indent=2) + "\n"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(_comparison_manifest(results), indent=2) + "\n"
    )


@pytest.mark.slow
@pytest.mark.requires_vst
@pytest.mark.requires_surgepy
def test_surgepy_pedalboard_dawdreamer_have_real_artifact_parity_and_benchmarks(
    tmp_path: Path,
) -> None:
    """Supported real hosts render equivalent state through the production Lance path.

    :param tmp_path: Temporary destination for all three real Lance datasets.
    """
    if TEST_SYNTH != "surge_xt":
        pytest.skip("three-host parity fixture uses Surge XT")
    assert surge_component_state(_VST_PRESET_PATH) == surge_component_state(
        _SURGEPY_PRESET_PATH
    )

    results: dict[ParityBackend, _BackendResult] = {
        backend: _render_dataset(backend, tmp_path / f"{backend}.lance")
        for backend in _BACKENDS
    }
    config = _config("pedalboard")
    expected_samples = int(config.sample_rate * config.signal_duration_seconds)
    expected_param_width = resolve_param_spec(config.param_spec_name).encoded_width
    for result in results.values():
        assert result.audio.shape == (_RENDER_COUNT, 2, expected_samples)
        assert result.mel.shape == (_RENDER_COUNT, 2, 128, 401)
        assert result.params.shape == (_RENDER_COUNT, expected_param_width)
        assert result.params.dtype == np.float32
        assert np.isfinite(result.audio).all()
        assert np.isfinite(result.mel).all()
        assert np.isfinite(result.params).all()
        assert np.all((result.params >= 0.0) & (result.params <= 1.0))
        assert np.max(np.abs(result.audio)) > 1e-4
        assert np.max(np.abs(result.audio)) <= 1.0
    np.testing.assert_array_equal(results["pedalboard"].params, results["dawdreamer"].params)
    np.testing.assert_array_equal(results["pedalboard"].params, results["surgepy"].params)

    pair_rows: dict[str, list[dict[str, float | int]]] = {}
    pair_metrics: dict[str, dict[str, float]] = {}
    for reference_name, candidate_name in combinations(results, 2):
        pair = f"{reference_name}-vs-{candidate_name}"
        rows = _pair_metric_rows(results[reference_name], results[candidate_name])
        metrics = _worst_pair_metrics(rows)
        pair_rows[pair] = rows
        pair_metrics[pair] = metrics

    _emit_benchmark_metrics(
        entries=_benchmark_entries(results, pair_metrics),
        bench_filename="surge-host-parity.json",
    )
    if output_dir := os.environ.get("SURGE_PARITY_OUTPUT_DIR"):
        comparison_dir = Path(output_dir)
        _write_comparison_directory(comparison_dir, results, pair_rows, pair_metrics)
        assert len(list((comparison_dir / "audio").glob("*/sample_*/*.wav"))) == 180
        assert len(list((comparison_dir / "mel").glob("*/*.npy"))) == 90
        assert len(list((comparison_dir / "mel").glob("*/*.png"))) == 90
        for pair in pair_rows:
            sample_dirs = find_possible_subdirs(comparison_dir / "audio" / pair)
            assert len(sample_dirs) == _RENDER_COUNT
            sample_rate, audio = wavfile.read(sample_dirs[0] / "target.wav")
            assert sample_rate == config.sample_rate
            assert audio.shape == (expected_samples, 2)
        assert np.load(comparison_dir / "mel" / "00" / "surgepy.npy").shape == (
            2,
            128,
            401,
        )

    for pair, metrics in pair_metrics.items():
        assert metrics["mel_rmse"] < _MEL_RMSE_MAX, (pair, metrics)
        assert metrics["mss"] < _MSS_MAX, (pair, metrics)
        assert metrics["wmfcc"] < _WMFCC_MAX, (pair, metrics)
        assert metrics["sot"] < _SOT_MAX, (pair, metrics)
        assert metrics["rms"] > _RMS_MIN, (pair, metrics)
