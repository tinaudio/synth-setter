"""Real DawDreamer dataset generation and host-to-host audio comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import h5py
import numpy as np
import pytest

from synth_setter.data.vst.writers import make_hdf5_dataset
from synth_setter.data.vst.renderers import DawDreamerRenderer, PedalboardRenderer
from synth_setter.evaluation.compute_audio_metrics import compute_mss, compute_rms, compute_sot, compute_wmfcc
from synth_setter.pipeline.schemas.spec import RenderConfig
from tests.data.vst.test_generate_vst_dataset import _HARDCODED_NOTE_PARAMS, _HARDCODED_SYNTH_PARAMS
from tests._vst import PLUGIN_PATH, TEST_PARAM_SPEC_NAME, TEST_PRESET_PATH, TEST_RENDERER_VERSION, TEST_SYNTH


def _config(backend: Literal["pedalboard", "dawdreamer"]) -> RenderConfig:
    return RenderConfig(
        plugin_path=PLUGIN_PATH,
        preset_path=TEST_PRESET_PATH,
        param_spec_name=TEST_PARAM_SPEC_NAME,
        renderer_version=TEST_RENDERER_VERSION,
        renderer_backend=backend,
        sample_rate=44100,
        channels=2,
        velocity=100,
        signal_duration_seconds=4.0,
        min_loudness=-55.0,
        samples_per_render_batch=1,
        samples_per_shard=1,
        base_seed=1808,
        attempts_per_sample=1,
        plugin_reload_cadence="once",
        gui_toggle_cadence="never",
    )


@pytest.mark.slow
@pytest.mark.requires_vst
def test_dawdreamer_dataset_audio_is_similar_to_pedalboard(tmp_path: Path) -> None:
    """Both hosts generate a real dataset row with perceptually similar audio.

    :param tmp_path: Temporary directory for generated HDF5 shards.
    """
    if TEST_SYNTH != "surge_xt":
        pytest.skip("DawDreamer comparison fixture uses the Surge XT parameter map")

    pedalboard_path = tmp_path / "pedalboard.h5"
    dawdreamer_path = tmp_path / "dawdreamer.h5"
    fixed_synth = [_HARDCODED_SYNTH_PARAMS]
    fixed_note = [_HARDCODED_NOTE_PARAMS]
    make_hdf5_dataset(
        pedalboard_path,
        _config("pedalboard"),
        fixed_synth_params_list=fixed_synth,
        fixed_note_params_list=fixed_note,
    )
    make_hdf5_dataset(
        dawdreamer_path,
        _config("dawdreamer"),
        fixed_synth_params_list=fixed_synth,
        fixed_note_params_list=fixed_note,
    )

    with h5py.File(pedalboard_path, "r") as pedalboard_file, h5py.File(
        dawdreamer_path, "r"
    ) as dawdreamer_file:
        pedalboard_audio = cast(h5py.Dataset, pedalboard_file["audio"])[0].astype(np.float32)
        dawdreamer_audio = cast(h5py.Dataset, dawdreamer_file["audio"])[0].astype(np.float32)
        pedalboard_params = cast(h5py.Dataset, pedalboard_file["param_array"])[0]
        dawdreamer_params = cast(h5py.Dataset, dawdreamer_file["param_array"])[0]

    assert np.array_equal(pedalboard_params, dawdreamer_params)
    assert np.max(np.abs(pedalboard_audio)) > 1e-4
    assert np.max(np.abs(dawdreamer_audio)) > 1e-4

    plugin_path = str(Path(PLUGIN_PATH).resolve())
    preset_path = str(Path(TEST_PRESET_PATH).resolve())
    pedalboard_audio = PedalboardRenderer(plugin_path, 44100, 2, 4.0, preset_path).render(
        {}, _HARDCODED_NOTE_PARAMS["pitch"], 100, _HARDCODED_NOTE_PARAMS["note_start_and_end"]
    )
    dawdreamer_audio = DawDreamerRenderer(plugin_path, 44100, 2, 4.0, preset_path).render(
        {}, _HARDCODED_NOTE_PARAMS["pitch"], 100, _HARDCODED_NOTE_PARAMS["note_start_and_end"]
    )
    assert compute_mss(pedalboard_audio, dawdreamer_audio) < 10.0
    assert compute_wmfcc(pedalboard_audio, dawdreamer_audio) < 18.0
    assert compute_sot(pedalboard_audio, dawdreamer_audio) < 0.15
    assert compute_rms(pedalboard_audio, dawdreamer_audio) > 0.8
