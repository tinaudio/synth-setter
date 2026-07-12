"""Real DawDreamer dataset generation and host-to-host audio comparison."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from synth_setter.data.vst.renderers import DawDreamerRenderer
from synth_setter.data.vst.writers import make_hdf5_dataset
from synth_setter.evaluation.compute_audio_metrics import (
    compute_mss,
    compute_rms,
    compute_sot,
    compute_wmfcc,
)
from synth_setter.pipeline.schemas.spec import RenderConfig
from tests._vst import (
    PLUGIN_PATH,
    TEST_PARAM_SPEC_NAME,
    TEST_PRESET_PATH,
    TEST_RENDERER_VERSION,
    TEST_SYNTH,
)
from tests.data.vst.test_generate_vst_dataset import (
    _HARDCODED_NOTE_PARAMS,
    _HARDCODED_SYNTH_PARAMS,
)


def _dawdreamer_experiment_config() -> RenderConfig:
    """Compose the DawDreamer smoke experiment with the test plugin paths.

    :returns: Render configuration selected by the experiment.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="dataset",
            overrides=[
                "experiment=generate_dataset/surge-xt-dawdreamer-smoke",
                f"render.plugin_path={PLUGIN_PATH}",
                f"render.preset_path={TEST_PRESET_PATH}",
                f"render.param_spec_name={TEST_PARAM_SPEC_NAME}",
                f"render.renderer_version={TEST_RENDERER_VERSION}",
            ],
        )
    config = RenderConfig.model_validate(OmegaConf.to_container(cfg.render, resolve=True))
    # base_seed / attempts_per_sample are RenderConfig fields, not render-group keys,
    # so pin them post-validation the way the launcher injects the per-shard seed.
    return config.model_copy(update={"base_seed": 1808, "attempts_per_sample": 1})


@pytest.mark.slow
@pytest.mark.requires_vst
def test_dawdreamer_dataset_audio_is_similar_to_pedalboard(tmp_path: Path) -> None:
    """Both hosts generate a real dataset row with perceptually similar audio.

    :param tmp_path: Temporary directory for generated HDF5 shards.
    """
    if TEST_SYNTH != "surge_xt":
        pytest.skip("DawDreamer comparison fixture uses the Surge XT parameter map")

    dawdreamer_config = _dawdreamer_experiment_config()
    pedalboard_config = dawdreamer_config.model_copy(update={"renderer_backend": "pedalboard"})
    pedalboard_path = tmp_path / "pedalboard.h5"
    dawdreamer_path = tmp_path / "dawdreamer.h5"
    fixed_synth = [_HARDCODED_SYNTH_PARAMS]
    fixed_note = [_HARDCODED_NOTE_PARAMS]
    dawdreamer_renderer = DawDreamerRenderer(
        str(Path(PLUGIN_PATH).resolve()),
        dawdreamer_config.sample_rate,
        dawdreamer_config.channels,
        dawdreamer_config.signal_duration_seconds,
        str(Path(TEST_PRESET_PATH).resolve()),
    )
    missing_keys = _HARDCODED_SYNTH_PARAMS.keys() - dawdreamer_renderer._parameter_indices.keys()
    assert not missing_keys
    host_indices = [dawdreamer_renderer._parameter_indices[key] for key in _HARDCODED_SYNTH_PARAMS]
    assert len(host_indices) == len(set(host_indices))
    make_hdf5_dataset(
        pedalboard_path,
        pedalboard_config,
        fixed_synth_params_list=fixed_synth,
        fixed_note_params_list=fixed_note,
    )
    make_hdf5_dataset(
        dawdreamer_path,
        dawdreamer_config,
        fixed_synth_params_list=fixed_synth,
        fixed_note_params_list=fixed_note,
    )

    with (
        h5py.File(pedalboard_path, "r") as pedalboard_file,
        h5py.File(dawdreamer_path, "r") as dawdreamer_file,
    ):
        pedalboard_audio = cast(h5py.Dataset, pedalboard_file["audio"])[0].astype(np.float32)
        dawdreamer_audio = cast(h5py.Dataset, dawdreamer_file["audio"])[0].astype(np.float32)
        pedalboard_params = cast(h5py.Dataset, pedalboard_file["param_array"])[0]
        dawdreamer_params = cast(h5py.Dataset, dawdreamer_file["param_array"])[0]

    assert np.array_equal(pedalboard_params, dawdreamer_params)
    assert np.max(np.abs(pedalboard_audio)) > 1e-4
    assert np.max(np.abs(dawdreamer_audio)) > 1e-4

    metrics = {
        "mss": compute_mss(pedalboard_audio, dawdreamer_audio),
        "rms": compute_rms(pedalboard_audio, dawdreamer_audio),
        "sot": compute_sot(pedalboard_audio, dawdreamer_audio),
        "wmfcc": compute_wmfcc(pedalboard_audio, dawdreamer_audio),
    }
    assert metrics["mss"] < 22.0, metrics
    assert metrics["wmfcc"] < 25.0, metrics
    assert metrics["sot"] < 0.35, metrics
    assert metrics["rms"] > 0.8, metrics
