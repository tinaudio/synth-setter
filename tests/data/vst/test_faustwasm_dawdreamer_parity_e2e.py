"""Real FaustWasm and DawDreamer Faust host parity coverage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import lance
import numpy as np
import pytest

from synth_setter.data.vst.shapes import AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD
from synth_setter.data.vst.writers import make_lance_dataset
from synth_setter.evaluation.compute_audio_metrics import (
    compute_mss,
    compute_rms,
    compute_sot,
    compute_wmfcc,
)
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.synth_spec import SYNTHS, SynthName

type FaustBackend = Literal["dawdreamer", "faustwasm"]

pytestmark = pytest.mark.slow

_BACKENDS: tuple[FaustBackend, ...] = ("dawdreamer", "faustwasm")
_BACKEND_VERSIONS: dict[FaustBackend, str] = {
    "dawdreamer": "0.8.3",
    "faustwasm": "0.18.3",
}
_NOTE_PARAMS = {"pitch": 60, "note_start_and_end": (0.1, 0.35)}
_ONSET_AMPLITUDE = 1e-8
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
_MEL_RMSE_MAX = 0.1
_MSS_MAX = 0.06
_RMS_MIN = 0.999
_SOT_MAX = 0.0001
_WMFCC_MAX = 0.06


@dataclass(frozen=True)
class _HostResult:
    audio: np.ndarray
    mel: np.ndarray
    params: np.ndarray


def _config(backend: FaustBackend) -> RenderConfig:
    """Build one host config for the shared fixed workload.

    :param backend: Real Faust host selected for rendering.
    :returns: Validated render configuration.
    """
    return RenderConfig(
        synth=SYNTHS[SynthName("faust_bright_organ")],
        renderer_backend=backend,
        backend_version=_BACKEND_VERSIONS[backend],
        block_size=128 if backend == "faustwasm" else None,
        render_contract_version=2,
        sample_rate=44_100,
        channels=2,
        velocity=100,
        signal_duration_seconds=0.5,
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
    """Render the shared A/B/A workload once through both production hosts.

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

    :param reference: DawDreamer result consumed from Lance.
    :param candidate: FaustWasm result consumed from Lance.
    :param sample: Matched row index.
    :returns: Named audio and persisted-mel parity metrics.
    """
    mel_delta = reference.mel[sample] - candidate.mel[sample]
    return {
        "mel_rmse": float(np.sqrt(np.mean(np.square(mel_delta)))),
        "mss": float(compute_mss(reference.audio[sample], candidate.audio[sample])),
        "rms": float(compute_rms(reference.audio[sample], candidate.audio[sample])),
        "sot": float(compute_sot(reference.audio[sample], candidate.audio[sample])),
        "wmfcc": float(compute_wmfcc(reference.audio[sample], candidate.audio[sample])),
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
    assert result.audio.shape == (3, 2, 22_050)
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
    """Both real hosts begin together at or after the requested note frame.

    :param host_results: Shared production Lance results.
    """
    onsets = []
    for backend in _BACKENDS:
        waveform = host_results[backend].audio[0]
        audible = np.flatnonzero(np.max(np.abs(waveform), axis=0) > _ONSET_AMPLITUDE)
        onsets.append(int(audible[0]))

    assert min(onsets) >= _REQUESTED_ONSET_SAMPLE
    assert len(set(onsets)) == 1


def test_faust_hosts_write_identical_normalized_parameter_rows(
    host_results: dict[FaustBackend, _HostResult],
) -> None:
    """Both production paths persist the same normalized conditioning rows.

    :param host_results: Shared production Lance results.
    """
    assert np.array_equal(host_results["dawdreamer"].params, host_results["faustwasm"].params)


@pytest.mark.parametrize("sample", [0, 1, 2])
def test_faust_hosts_a_b_a_workload_has_per_render_parity(
    host_results: dict[FaustBackend, _HostResult],
    sample: int,
) -> None:
    """One matched A/B/A row satisfies every calibrated parity limit.

    :param host_results: Shared production Lance results.
    :param sample: Matched workload row.
    """
    metrics = _metrics(host_results["dawdreamer"], host_results["faustwasm"], sample)
    diagnostic = {"sample": sample, "metrics": metrics}
    assert metrics["mel_rmse"] <= _MEL_RMSE_MAX, diagnostic
    assert metrics["mss"] <= _MSS_MAX, diagnostic
    assert metrics["rms"] >= _RMS_MIN, diagnostic
    assert metrics["sot"] <= _SOT_MAX, diagnostic
    assert metrics["wmfcc"] <= _WMFCC_MAX, diagnostic
