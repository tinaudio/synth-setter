"""Cross-version diagnostics for native SurgePy and Surge WASM rendering."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from hashlib import sha256
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import sh
from pydantic import BaseModel, Field
from scipy.io import wavfile

from synth_setter.data.vst.generate_vst_dataset import make_spectrogram
from synth_setter.data.vst.param_map import SurgePyParamRef, load_param_map
from synth_setter.data.vst.param_spec import NoteParams
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.data.vst.renderers import _sample_index_at_or_after
from tests.data.vst.test_generate_vst_dataset import _HARDCODED_NOTE_PARAMS
from tests.data.vst.test_surgepy_host_parity_e2e import (
    _DIVERSE_PATCH_VALUES,
    _ONSET_AMPLITUDE,
    _PARAMETER_MAP_PATH,
    _PARITY_SYNTH_PARAMS,
    _RANDOM_PATCH_SEED,
    _REPEATED_RENDER_COUNT,
    _SURGEPY_PRESET_PATH,
    _BackendResult,
    _config,
    _onset_samples,
    _pair_metric_rows,
    _random_patch_config,
    _render_dataset,
    _sample_random_patches,
    _worst_pair_metrics,
    _write_audio_artifacts,
    _write_mel_artifacts,
)

_WASM_BACKEND = "surgewasm"
_WASM_BUNDLE_DEFAULT = Path("build/surge-engine")
_WASM_CLI = Path("src/synth_setter/surgewasm/render.mjs")
_WASM_REQUIRED_FILES = (
    "manifest.json",
    "surge-host.mjs",
    "surge-host.wasm",
    "surge-xt.clap.wasm",
)
_WORKLOADS = ("repeated-patch", "diverse-patches", "random-patches")


# Strict models reject drift at the external Node CLI JSON boundary.
class _WasmParameterDescriptor(BaseModel, strict=True, extra="forbid"):
    id: int
    name: str
    value: float


class _WasmReportRow(BaseModel, strict=True, extra="forbid"):
    sample: int
    filename: str
    sample_rate: int = Field(alias="sampleRate")
    frames: int
    render_seconds: float = Field(alias="renderSeconds")
    requested_parameters: list[_WasmParameterDescriptor] = Field(alias="parameters")


class _WasmReport(BaseModel, strict=True, extra="forbid"):
    source: Literal["surge-wasm"]
    engine_commit: str = Field(alias="engineCommit")
    renderer_version: str = Field(alias="rendererVersion")
    preset_sha256: str = Field(alias="presetSha256")
    sample_count: int = Field(alias="sampleCount")
    rows: list[_WasmReportRow]


def _workload_patches(workload: str) -> list[dict[str, float]]:
    """Return the canonical full corpus for one diagnostic workload.

    :param workload: Canonical parity workload name.
    :returns: Ordered normalized patches shared by native and WASM renderers.
    :raises ValueError: If ``workload`` is not canonical.
    """
    if workload == "repeated-patch":
        return [_PARITY_SYNTH_PARAMS.copy() for _ in range(_REPEATED_RENDER_COUNT)]
    if workload == "diverse-patches":
        return [
            {
                **_PARITY_SYNTH_PARAMS,
                "a_filter_1_cutoff": cutoff,
                "a_osc_1_octave": octave,
            }
            for cutoff, octave in _DIVERSE_PATCH_VALUES
        ]
    if workload == "random-patches":
        config = _random_patch_config("surgepy")
        return _sample_random_patches(config, seed=_RANDOM_PATCH_SEED)
    raise ValueError(f"unknown workload: {workload}")


def _require_wasm_prerequisites() -> Path:
    """Return the built WASM bundle or skip for an explicitly missing prerequisite.

    :returns: Existing Surge WASM bundle directory.
    """
    if shutil.which("node") is None:
        pytest.skip("Surge WASM parity requires Node.js on PATH")
    assert _WASM_CLI.is_file(), f"missing committed Surge WASM CLI: {_WASM_CLI}"
    assert _SURGEPY_PRESET_PATH.is_file(), f"missing committed preset: {_SURGEPY_PRESET_PATH}"
    bundle = Path(
        os.environ.get(
            "SURGE_WASM_BUNDLE",
            os.environ.get("SYNTH_SETTER_SURGE_WASM_BUNDLE", str(_WASM_BUNDLE_DEFAULT)),
        )
    )
    missing = [name for name in _WASM_REQUIRED_FILES if not (bundle / name).is_file()]
    if missing:
        pytest.skip(f"Surge WASM parity requires built bundle files in {bundle}: {missing}")
    return bundle


def _wasm_requests(
    synth_params: list[dict[str, float]],
    parameter_map: dict[str, SurgePyParamRef],
) -> list[dict[str, object]]:
    """Map canonical logical patches into the public WASM CLI request schema.

    :param synth_params: Ordered normalized logical parameter patches.
    :param parameter_map: Logical names mapped to native Surge identities.
    :returns: Complete ordered render requests for the WASM CLI.
    """
    config = _config("surgepy", len(synth_params))
    note_start, note_end = _HARDCODED_NOTE_PARAMS["note_start_and_end"]
    frames = int(config.sample_rate * config.signal_duration_seconds)
    return [
        {
            "parameters": [
                {
                    "id": parameter_map[name].synth_side_id,
                    "name": parameter_map[name].name,
                    "value": patch[name],
                }
                for name in patch
            ],
            "note": _HARDCODED_NOTE_PARAMS["pitch"],
            "velocity": config.velocity,
            "noteStart": note_start,
            "noteEnd": note_end,
            "sampleRate": config.sample_rate,
            "frames": frames,
        }
        for patch in synth_params
    ]


def _logical_patch_from_descriptors(
    descriptors: list[_WasmParameterDescriptor],
    parameter_map: dict[str, SurgePyParamRef],
) -> dict[str, float]:
    """Recover logical values from persisted requested native identities.

    :param descriptors: Parameter descriptors consumed from the CLI report.
    :param parameter_map: Logical names mapped to expected native identities.
    :returns: Requested values keyed by canonical logical name.
    """
    logical_by_identity = {
        (reference.synth_side_id, reference.name): logical_name
        for logical_name, reference in parameter_map.items()
    }
    assert len(descriptors) == len(logical_by_identity)
    patch = {
        logical_by_identity[(descriptor.id, descriptor.name)]: descriptor.value
        for descriptor in descriptors
    }
    assert len(patch) == len(logical_by_identity)
    return patch


def _encoded_report_params(
    rows: list[_WasmReportRow],
    parameter_map: dict[str, SurgePyParamRef],
    render_count: int,
) -> np.ndarray:
    """Encode persisted WASM requests through the canonical ParamSpec.

    :param rows: CLI report rows in sample order.
    :param parameter_map: Logical names mapped to native Surge identities.
    :param render_count: Expected report row count.
    :returns: Canonically encoded parameter matrix.
    """
    config = _config("surgepy", render_count)
    param_spec = resolve_param_spec(config.param_spec_name)
    note_params = {
        **_HARDCODED_NOTE_PARAMS,
        "velocity": config.velocity,
    }
    return np.stack(
        [
            param_spec.encode(
                _logical_patch_from_descriptors(row.requested_parameters, parameter_map),
                note_params,
            )
            for row in rows
        ]
    )


def _render_wasm(
    path: Path,
    synth_params: list[dict[str, float]],
    bundle: Path,
) -> tuple[_BackendResult, _WasmReport]:
    """Render and consume one real WASM CLI batch.

    :param path: Temporary root for request and output artifacts.
    :param synth_params: Exact normalized corpus to render.
    :param bundle: Built Surge WASM bundle directory.
    :returns: Materialized audio/features/parameters and consumed CLI report.
    """
    parameter_map = load_param_map(_PARAMETER_MAP_PATH).surgepy_params()
    requests = _wasm_requests(synth_params, parameter_map)
    requests_path = path / "requests.json"
    requests_path.parent.mkdir(parents=True)
    requests_path.write_text(json.dumps(requests, indent=2) + "\n")
    output_dir = path / "rendered"

    started = time.perf_counter()
    sh.Command("node")(
        str(_WASM_CLI),
        str(bundle),
        str(_SURGEPY_PRESET_PATH),
        str(requests_path),
        str(output_dir),
    )
    elapsed = time.perf_counter() - started

    report = _WasmReport.model_validate_json((output_dir / "report.json").read_text())
    rows = report.rows
    assert report.sample_count == len(synth_params)
    assert len(rows) == len(synth_params)
    assert [row.sample for row in rows] == list(range(len(synth_params)))
    requested_parameters = [
        [parameter.model_dump() for parameter in row.requested_parameters] for row in rows
    ]
    assert requested_parameters == [request["parameters"] for request in requests]
    assert all(math.isfinite(row.render_seconds) and row.render_seconds >= 0 for row in rows)
    audio_rows = []
    for row in rows:
        sample_rate, audio = wavfile.read(output_dir / row.filename)
        assert sample_rate == row.sample_rate
        assert audio.shape == (row.frames, 2)
        audio_rows.append(audio.T)
    audio = np.stack(audio_rows).astype(np.float32, copy=False)
    mel = np.stack([make_spectrogram(waveform, rows[0].sample_rate) for waveform in audio])
    params = _encoded_report_params(rows, parameter_map, len(synth_params))
    return _BackendResult(audio=audio, mel=mel, params=params, total_seconds=elapsed), report


def _timing_rows(
    workload: str,
    results: dict[str, _BackendResult],
) -> list[dict[str, int | str | None]]:
    """Record requested and observed onsets without changing either waveform.

    :param workload: Canonical workload name.
    :param results: Materialized native and WASM audio.
    :returns: Per-backend onset diagnostics in sample coordinates.
    """
    config = _config("surgepy", len(next(iter(results.values())).audio))
    requested_sample = _sample_index_at_or_after(
        _HARDCODED_NOTE_PARAMS["note_start_and_end"][0], config.sample_rate
    )
    rows: list[dict[str, int | str | None]] = []
    for backend, result in results.items():
        if workload == "random-patches":
            onsets: list[int | None] = []
            for waveform in result.audio:
                audible = np.flatnonzero(np.max(np.abs(waveform), axis=0) > _ONSET_AMPLITUDE)
                onsets.append(int(audible[0]) if len(audible) else None)
        else:
            onsets = _onset_samples(result.audio).tolist()
        rows.extend(
            {
                "backend": backend,
                "sample": sample,
                "requested_sample": requested_sample,
                "actual_onset_sample": onset,
            }
            for sample, onset in enumerate(onsets)
        )
    return rows


def _artifact_root(tmp_path: Path, workload: str) -> Path:
    """Return the retained or temporary workload artifact directory.

    :param tmp_path: Pytest-owned temporary root.
    :param workload: Canonical workload name.
    :returns: Isolated cross-version artifact path.
    """
    base = Path(os.environ.get("SURGE_PARITY_OUTPUT_DIR", str(tmp_path)))
    return base / "surgepy-wasm" / workload


def _write_diagnostics(
    *,
    output_dir: Path,
    workload: str,
    synth_params: list[dict[str, float]],
    results: dict[str, _BackendResult],
    report: _WasmReport,
    bundle: Path,
) -> None:
    """Persist and consume cross-version audio, features, metrics, and provenance.

    :param output_dir: Workload diagnostic destination.
    :param workload: Canonical workload name.
    :param synth_params: Exact normalized corpus in row order.
    :param results: Materialized native and WASM results.
    :param report: Consumed WASM CLI report.
    :param bundle: Built WASM bundle directory.
    """
    shutil.rmtree(output_dir, ignore_errors=True)
    pair_rows = _pair_metric_rows(results["surgepy"], results[_WASM_BACKEND])
    _write_audio_artifacts(output_dir, results)
    _write_mel_artifacts(output_dir, results)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = report.rows
    config = _config("surgepy", len(synth_params))
    note_start, note_end = _HARDCODED_NOTE_PARAMS["note_start_and_end"]
    midi_event = {
        "note": _HARDCODED_NOTE_PARAMS["pitch"],
        "velocity": config.velocity,
        "note_start_seconds": note_start,
        "note_end_seconds": note_end,
    }
    parameters = [
        {
            "sample": index,
            "normalized_synth_parameters": patch,
            "wasm_requested_parameters": [
                parameter.model_dump() for parameter in row.requested_parameters
            ],
            "wasm_requested_encoded_normalized_vector": results[_WASM_BACKEND]
            .params[index]
            .tolist(),
            "native_persisted_encoded_normalized_vector": results["surgepy"].params[index].tolist(),
            "midi_event": midi_event,
        }
        for index, (patch, row) in enumerate(zip(synth_params, rows, strict=True))
    ]
    (output_dir / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    metrics = {
        "pair": "surgepy-vs-surgewasm",
        "rows": pair_rows,
        "timing": _timing_rows(workload, results),
        "worst": _worst_pair_metrics(pair_rows),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    parameter_map = load_param_map(_PARAMETER_MAP_PATH)
    assert parameter_map.surgepy is not None
    bundle_manifest = json.loads((bundle / "manifest.json").read_text())
    manifest = {
        "artifact_schema_version": 1,
        "workload": workload,
        "diagnostic_only": True,
        "green_definition": (
            "All canonical rows rendered through native SurgePy 1.3 and Surge WASM 1.4; "
            "persisted requests, audio, mel features, metrics, and provenance were consumed."
        ),
        "thresholds": {},
        "parameter_evidence": "Lance-stored and CLI-reported requests, not engine parameter readback",
        "timing_diagnostic": {
            "cross_version": True,
            "asserted": False,
            "onset_amplitude": _ONSET_AMPLITUDE,
            "requested_sample_rule": "ceil(noteStart * sampleRate)",
            "native_policy": "ceil note-on followed by native attack alignment",
            "wasm_policy": "note-on floored to the 32-frame processing quantum without final alignment",
            "waveforms_shifted_for_comparison": False,
        },
        "render_count": len(synth_params),
        "corpus": {
            "count": len(synth_params),
            "random_seed": _RANDOM_PATCH_SEED if workload == "random-patches" else None,
            "random_loudness_policy_db": "-inf" if workload == "random-patches" else None,
        },
        "backends": {
            "surgepy": {
                "renderer_version": _config("surgepy", len(synth_params)).synth.synth_version,
                "engine_commit": _config("surgepy", len(synth_params)).synth.synth_version.split(
                    "."
                )[-1],
                "timing_scope": "configuration and production Lance dataset write wall time; read excluded",
                "total_seconds": results["surgepy"].total_seconds,
            },
            _WASM_BACKEND: {
                "renderer_version": bundle_manifest["version"],
                "host_identity": report.renderer_version,
                "engine_commit": report.engine_commit,
                "bundle_version": bundle_manifest["version"],
                "bundle_commit": bundle_manifest["source"]["commit"],
                "timing_scope": "Node CLI batch wall time",
                "total_seconds": results[_WASM_BACKEND].total_seconds,
                "row_render_seconds": [row.render_seconds for row in rows],
            },
        },
        "preset": {
            "path": str(_SURGEPY_PRESET_PATH),
            "sha256": sha256(_SURGEPY_PRESET_PATH.read_bytes()).hexdigest(),
            "wasm_report_sha256": report.preset_sha256,
            "parameter_map_preset_sha256": parameter_map.surgepy_preset_sha256,
        },
        "parameter_map": {
            "path": str(_PARAMETER_MAP_PATH),
            "sha256": sha256(_PARAMETER_MAP_PATH.read_bytes()).hexdigest(),
            "surgepy_version": parameter_map.surgepy.plugin_version,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _assert_diagnostics(
    output_dir: Path,
    workload: str,
    synth_params: list[dict[str, float]],
    native: _BackendResult,
) -> None:
    """Consume persisted diagnostic files and verify their cross-version contract.

    :param output_dir: Workload diagnostic directory.
    :param workload: Expected canonical workload name.
    :param synth_params: Expected ordered normalized corpus.
    :param native: Native production artifact used for parameter identity.
    """
    render_count = len(synth_params)
    config = _config("surgepy", render_count)
    expected_frames = int(config.sample_rate * config.signal_duration_seconds)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["workload"] == workload
    assert manifest["diagnostic_only"] is True
    assert manifest["thresholds"] == {}
    assert manifest["render_count"] == render_count
    assert manifest["timing_diagnostic"]["cross_version"] is True
    assert manifest["timing_diagnostic"]["asserted"] is False
    assert manifest["timing_diagnostic"]["waveforms_shifted_for_comparison"] is False
    assert manifest["backends"]["surgepy"]["renderer_version"].startswith("1.3.")
    assert manifest["backends"][_WASM_BACKEND]["renderer_version"].startswith("Surge XT 1.4.")
    assert (
        manifest["backends"][_WASM_BACKEND]["engine_commit"]
        == manifest["backends"][_WASM_BACKEND]["bundle_commit"]
    )
    for backend in ("surgepy", _WASM_BACKEND):
        assert math.isfinite(manifest["backends"][backend]["total_seconds"])
        assert manifest["backends"][backend]["total_seconds"] > 0
        assert manifest["backends"][backend]["timing_scope"]
    assert not manifest["backends"][_WASM_BACKEND]["engine_commit"].startswith(
        manifest["backends"]["surgepy"]["engine_commit"]
    )
    assert manifest["preset"]["sha256"] == manifest["preset"]["wasm_report_sha256"]
    assert manifest["preset"]["sha256"] == manifest["preset"]["parameter_map_preset_sha256"]

    parameters = json.loads((output_dir / "parameters.json").read_text())
    assert [row["sample"] for row in parameters] == list(range(render_count))
    assert [row["normalized_synth_parameters"] for row in parameters] == synth_params
    requested_params = np.asarray(
        [row["wasm_requested_encoded_normalized_vector"] for row in parameters],
        dtype=np.float32,
    )
    persisted_params = np.asarray(
        [row["native_persisted_encoded_normalized_vector"] for row in parameters],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(requested_params, native.params)
    np.testing.assert_array_equal(persisted_params, native.params)

    source_report = _WasmReport.model_validate_json((output_dir / "wasm-report.json").read_text())
    assert source_report.sample_count == render_count

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["pair"] == "surgepy-vs-surgewasm"
    assert [row["sample"] for row in metrics["rows"]] == list(range(render_count))
    assert len(metrics["timing"]) == render_count * 2
    requested_sample = _sample_index_at_or_after(
        _HARDCODED_NOTE_PARAMS["note_start_and_end"][0], config.sample_rate
    )
    assert {(row["backend"], row["sample"]) for row in metrics["timing"]} == {
        (backend, sample)
        for backend in ("surgepy", _WASM_BACKEND)
        for sample in range(render_count)
    }
    assert {row["requested_sample"] for row in metrics["timing"]} == {requested_sample}
    assert all(
        row["actual_onset_sample"] is None or type(row["actual_onset_sample"]) is int
        for row in metrics["timing"]
    )
    if workload != "random-patches":
        assert all(row["actual_onset_sample"] is not None for row in metrics["timing"])
    for row in metrics["rows"]:
        assert all(
            math.isfinite(row[name])
            for name in ("mel_rmse", "mss", "rms", "rms_distance", "sot", "wmfcc")
        )

    wav_paths = sorted((output_dir / "audio").glob("sample_*/*.wav"))
    assert len(wav_paths) == render_count * 2
    for path in wav_paths:
        sample_rate, audio = wavfile.read(path)
        assert sample_rate == config.sample_rate
        assert audio.shape == (expected_frames, 2)
        assert np.isfinite(audio).all()
        persisted_mel = np.load(output_dir / "mel" / path.parent.name / f"{path.stem}.npy")
        assert np.isfinite(persisted_mel).all()
        np.testing.assert_allclose(
            persisted_mel,
            make_spectrogram(audio.T, sample_rate),
            rtol=1e-6,
            atol=1e-6,
        )


@pytest.mark.slow
@pytest.mark.requires_surgepy
@pytest.mark.parametrize("workload", _WORKLOADS)
def test_surgepy_wasm_full_corpus_records_cross_version_diagnostics(
    tmp_path: Path,
    workload: str,
) -> None:
    """Render and consume the canonical corpus without cross-version quality gates.

    :param tmp_path: Temporary render and artifact root.
    :param workload: Canonical full-corpus workload selected by parametrization.
    """
    bundle = _require_wasm_prerequisites()
    synth_params = _workload_patches(workload)
    note_params: list[NoteParams] = [_HARDCODED_NOTE_PARAMS.copy() for _ in synth_params]
    min_loudness = float("-inf") if workload == "random-patches" else None
    native = _render_dataset(
        "surgepy",
        tmp_path / f"{workload}-surgepy.lance",
        synth_params=synth_params,
        note_params=note_params,
        min_loudness=min_loudness,
    )
    wasm_path = tmp_path / f"{workload}-wasm"
    wasm, report = _render_wasm(wasm_path, synth_params, bundle)
    results = {"surgepy": native, _WASM_BACKEND: wasm}

    assert native.audio.shape == wasm.audio.shape
    assert native.mel.shape == wasm.mel.shape
    np.testing.assert_array_equal(native.params, wasm.params)
    assert np.isfinite(native.audio).all() and np.isfinite(wasm.audio).all()
    assert np.isfinite(native.mel).all() and np.isfinite(wasm.mel).all()
    if workload != "random-patches":
        assert np.all(np.max(np.abs(native.audio), axis=(1, 2)) > 0.0)
        assert np.all(np.max(np.abs(wasm.audio), axis=(1, 2)) > 0.0)
    if workload != "repeated-patch":
        assert np.unique(native.params, axis=0).shape[0] == len(synth_params)

    output_dir = _artifact_root(tmp_path, workload)
    _write_diagnostics(
        output_dir=output_dir,
        workload=workload,
        synth_params=synth_params,
        results=results,
        report=report,
        bundle=bundle,
    )
    shutil.copyfile(wasm_path / "rendered" / "report.json", output_dir / "wasm-report.json")
    _assert_diagnostics(output_dir, workload, synth_params, native)
