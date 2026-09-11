"""Real DawDreamer, Faust C++, and FaustWasm host parity coverage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import lance
import numpy as np
import pytest

from synth_setter.data.vst.core import extract_backend_version
from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    MEL_SPEC_FIELD,
    PARAM_ARRAY_FIELD,
    make_spectrogram,
)
from synth_setter.data.vst.writers import make_lance_dataset
from synth_setter.evaluation.compute_audio_metrics import (
    compute_mss,
    compute_rms,
    compute_sot,
    compute_wmfcc,
)
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.synth_spec import SYNTHS, SynthName

type FaustBackend = Literal["dawdreamer", "faustcpp", "faustwasm"]

pytestmark = pytest.mark.slow

_BACKENDS: tuple[FaustBackend, ...] = ("dawdreamer", "faustcpp", "faustwasm")
_COMPARISON_SAMPLES = 22_050
_PARITY_PAIRS: tuple[tuple[FaustBackend, FaustBackend], ...] = (
    ("dawdreamer", "faustwasm"),
    ("faustcpp", "faustwasm"),
)
_RENDER_SAMPLES = 176_400
_TAIL_LIMITS = {"mss_max": 0.65, "rms_min": 0.9998, "wmfcc_max": 1.1}
_BACKEND_VERSIONS: dict[FaustBackend, str] = {
    "dawdreamer": "0.8.3",
    "faustwasm": "0.18.3",
}
_NOTE_PARAMS = {"pitch": 60, "note_start_and_end": (0.1, 0.35)}
_ONSET_AMPLITUDE = 1e-8
_ONSET_ALIGNMENT_TOLERANCE_SAMPLES = 1
_REQUESTED_ONSET_SAMPLE = 4_410
_VOLUME_ADDRESS = "/Sequencer/DSP1/brightOrgan/Main/volume"
_PATCH = {
    "/Sequencer/DSP1/brightOrgan/Main/volume": 0.5,
    "/Sequencer/DSP1/brightOrgan/Reverb/Amount": 0.5,
    "/Sequencer/DSP1/brightOrgan/Reverb/Damp": 0.5,
    "/Sequencer/DSP1/brightOrgan/Reverb/Size": 0.5,
    "/Sequencer/DSP1/brightOrgan/Stops/Fifteenth_2'": 0.5,
    "/Sequencer/DSP1/brightOrgan/Stops/Flute_8'": 0.5,
    "/Sequencer/DSP1/brightOrgan/Stops/Foundation_8'": 0.5,
    "/Sequencer/DSP1/brightOrgan/Stops/Nasard_2_2/3'": 0.5,
    "/Sequencer/DSP1/brightOrgan/Stops/Principal_4'": 0.5,
    "/Sequencer/DSP1/brightOrgan/Stops/Tierce_1_3/5'": 0.5,
}
_PARITY_LIMITS = {
    ("dawdreamer", "faustwasm"): {
        "mel_rmse_max": 0.1,
        "mss_max": 0.06,
        "rms_min": 0.999,
        "sot_max": 0.0001,
        "wmfcc_max": 0.06,
    },
    # Three Faust 2.37.3 A/B/A runs had maxima 3.120689, 1.174139, 0.997783, 0.001013, and 1.906417.
    ("faustcpp", "faustwasm"): {
        "mel_rmse_max": 3.75,
        "mss_max": 1.5,
        "rms_min": 0.997,
        "sot_max": 0.0013,
        "wmfcc_max": 2.3,
    },
}


@dataclass(frozen=True)
class _HostResult:
    audio: np.ndarray
    mel: np.ndarray
    params: np.ndarray


def _backend_version(backend: FaustBackend) -> str:
    """Resolve the version required by one real host.

    :param backend: Real Faust host selected for rendering.
    :returns: Installed or dependency-pinned backend version.
    """
    if backend == "faustcpp":
        return extract_backend_version(backend)
    return _BACKEND_VERSIONS[backend]


def _config(backend: FaustBackend) -> RenderConfig:
    """Build one host config for the shared fixed workload.

    :param backend: Real Faust host selected for rendering.
    :returns: Validated render configuration.
    """
    return RenderConfig(
        synth=SYNTHS[SynthName("faust_bright_organ")],
        renderer_backend=backend,
        backend_version=_backend_version(backend),
        block_size=128 if backend in {"faustcpp", "faustwasm"} else None,
        render_contract_version=2,
        sample_rate=44_100,
        channels=2,
        velocity=100,
        signal_duration_seconds=4.0,
        min_loudness=-100.0,
        samples_per_render_batch=1,
        samples_per_shard=3,
        plugin_reload_cadence="render",
        gui_toggle_cadence="never",
        audio_dtype="float32",
    )


def _render_dataset(
    root: Path,
    backend: FaustBackend,
    patches: list[dict[str, float]],
) -> _HostResult:
    """Render and consume one host's production Lance dataset.

    :param root: Temporary root for the generated dataset.
    :param backend: Real Faust host selected for rendering.
    :param patches: Complete native Faust patches in row order.
    :returns: Audio, mel, and normalized parameters consumed from Lance.
    """
    path = root / f"{backend}.lance"
    make_lance_dataset(
        path,
        _config(backend),
        fixed_synth_params_list=patches,
        fixed_note_params_list=[_NOTE_PARAMS.copy() for _ in patches],
    )
    table = lance.dataset(str(path)).to_table(
        columns=[AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD]
    )
    return _HostResult(
        audio=table.column(AUDIO_FIELD).combine_chunks().to_numpy_ndarray(),
        mel=table.column(MEL_SPEC_FIELD).combine_chunks().to_numpy_ndarray(),
        params=table.column(PARAM_ARRAY_FIELD).combine_chunks().to_numpy_ndarray(),
    )


@pytest.fixture(scope="module")
def host_results(tmp_path_factory: pytest.TempPathFactory) -> dict[FaustBackend, _HostResult]:
    """Render the shared A/B/A workload once through all production hosts.

    :param tmp_path_factory: Module-scoped temporary directory factory.
    :returns: Consumed Lance results keyed by host.
    """
    patch_a = {**_PATCH, _VOLUME_ADDRESS: 0.25}
    patch_b = {**_PATCH, _VOLUME_ADDRESS: 1.0}
    patches = [patch_a, patch_b, patch_a]
    root = tmp_path_factory.mktemp("faust-host-parity")
    return {backend: _render_dataset(root, backend, patches) for backend in _BACKENDS}


def _metrics(reference: _HostResult, candidate: _HostResult, sample: int) -> dict[str, float]:
    """Compute existing host-parity metrics for one matched row.

    :param reference: Reference host result consumed from Lance.
    :param candidate: Candidate host result consumed from Lance.
    :param sample: Matched row index.
    :returns: Named metrics over the calibrated comparison window.
    """
    reference_audio = reference.audio[sample, :, :_COMPARISON_SAMPLES]
    candidate_audio = candidate.audio[sample, :, :_COMPARISON_SAMPLES]
    mel_delta = make_spectrogram(reference_audio, 44_100) - make_spectrogram(
        candidate_audio, 44_100
    )
    return {
        "mel_rmse": float(np.sqrt(np.mean(np.square(mel_delta)))),
        "mss": float(compute_mss(reference_audio, candidate_audio)),
        "rms": float(compute_rms(reference_audio, candidate_audio)),
        "sot": float(compute_sot(reference_audio, candidate_audio)),
        "wmfcc": float(compute_wmfcc(reference_audio, candidate_audio)),
    }


@pytest.mark.parametrize("backend", _BACKENDS)
def test_faust_host_dataset_persists_valid_audio_and_mel(
    host_results: dict[FaustBackend, _HostResult],
    backend: FaustBackend,
) -> None:
    """Each real host persists finite, audible, bounded audio and mel rows.

    :param host_results: Shared production Lance results.
    :param backend: Host whose persisted tensors are checked.
    """
    result = host_results[backend]
    assert result.audio.shape == (3, 2, _RENDER_SAMPLES)
    assert result.audio.dtype == np.float32
    assert np.isfinite(result.audio).all()
    assert np.isfinite(result.mel).all()
    assert float(np.max(np.abs(result.audio))) > 1e-4
    assert float(np.max(np.abs(result.audio))) <= 1.0


@pytest.mark.parametrize("backend", _BACKENDS)
def test_faust_host_a_b_a_dataset_is_state_isolated(
    host_results: dict[FaustBackend, _HostResult],
    backend: FaustBackend,
) -> None:
    """Each real host repeats A exactly while preserving B's causal change.

    :param host_results: Shared production Lance results.
    :param backend: Host whose persisted state is checked.
    """
    audio = host_results[backend].audio
    assert np.array_equal(audio[0], audio[2])
    assert not np.array_equal(audio[0], audio[1])


def test_faust_hosts_have_aligned_non_early_onsets(
    host_results: dict[FaustBackend, _HostResult],
) -> None:
    """All real hosts begin together at or after the requested note frame.

    :param host_results: Shared production Lance results.
    """
    onsets = []
    for backend in _BACKENDS:
        waveform = host_results[backend].audio[0]
        audible = np.flatnonzero(np.max(np.abs(waveform), axis=0) > _ONSET_AMPLITUDE)
        onsets.append(int(audible[0]))

    assert min(onsets) >= _REQUESTED_ONSET_SAMPLE
    assert max(onsets) - min(onsets) <= _ONSET_ALIGNMENT_TOLERANCE_SAMPLES


def test_faust_hosts_write_identical_normalized_parameter_rows(
    host_results: dict[FaustBackend, _HostResult],
) -> None:
    """All production paths persist the same normalized conditioning rows.

    :param host_results: Shared production Lance results.
    """
    for backend in _BACKENDS[1:]:
        assert np.array_equal(host_results["dawdreamer"].params, host_results[backend].params)


@pytest.mark.parametrize(("reference_backend", "candidate_backend"), _PARITY_PAIRS)
@pytest.mark.parametrize("sample", [0, 1, 2])
def test_faust_hosts_a_b_a_workload_has_per_render_parity(
    host_results: dict[FaustBackend, _HostResult],
    reference_backend: FaustBackend,
    candidate_backend: FaustBackend,
    sample: int,
) -> None:
    """One matched A/B/A row satisfies every calibrated parity limit.

    :param host_results: Shared production Lance results.
    :param reference_backend: Host providing the reference waveform.
    :param candidate_backend: Host compared with the reference.
    :param sample: Matched workload row.
    """
    metrics = _metrics(host_results[reference_backend], host_results[candidate_backend], sample)
    limits = _PARITY_LIMITS[(reference_backend, candidate_backend)]
    diagnostic = {
        "reference_backend": reference_backend,
        "candidate_backend": candidate_backend,
        "sample": sample,
        "metrics": metrics,
    }
    assert metrics["mel_rmse"] <= limits["mel_rmse_max"], diagnostic
    assert metrics["mss"] <= limits["mss_max"], diagnostic
    assert metrics["rms"] >= limits["rms_min"], diagnostic
    assert metrics["sot"] <= limits["sot_max"], diagnostic
    assert metrics["wmfcc"] <= limits["wmfcc_max"], diagnostic


@pytest.mark.parametrize("sample", [0, 1, 2])
def test_faustcpp_matches_faustwasm_release_tail(
    host_results: dict[FaustBackend, _HostResult],
    sample: int,
) -> None:
    """Native C++ preserves the post-0.5-second release and reverb behavior.

    :param host_results: Shared production Lance results.
    :param sample: Matched workload row.
    """
    native_tail = host_results["faustcpp"].audio[sample, :, _COMPARISON_SAMPLES:]
    wasm_tail = host_results["faustwasm"].audio[sample, :, _COMPARISON_SAMPLES:]
    metrics = {
        "mss": float(compute_mss(native_tail, wasm_tail)),
        "rms": float(compute_rms(native_tail, wasm_tail)),
        "wmfcc": float(compute_wmfcc(native_tail, wasm_tail)),
    }
    diagnostic = {"sample": sample, "metrics": metrics}
    assert metrics["mss"] <= _TAIL_LIMITS["mss_max"], diagnostic
    assert metrics["rms"] >= _TAIL_LIMITS["rms_min"], diagnostic
    assert metrics["wmfcc"] <= _TAIL_LIMITS["wmfcc_max"], diagnostic
