"""Native Dexed config-to-dataset-row integration."""

import numpy as np
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf, open_dict

from synth_setter.data.vst.generate_vst_dataset import SampleSeed, generate_sample
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.renderer_factory import make_audio_renderer
from synth_setter.synth_spec import SYNTHS, SynthName


def test_dexed_hydra_config_generates_audio_features_and_parameters() -> None:
    """The real sampling pipeline accepts native audio and encodes its training row."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        config = compose(config_name="render/dexed").render
    with open_dict(config):
        config.synth = SYNTHS[SynthName("dexed_py")].model_dump()
    render = RenderConfig.model_validate(OmegaConf.to_container(config, resolve=True))
    sample = generate_sample(
        make_audio_renderer(render),
        render.velocity,
        render.min_loudness,
        param_specs["dexed_py"],
        fixed_note_params={"pitch": 60, "note_start_and_end": (0.0, 1.0)},
        seed=SampleSeed(master_seed=42, max_attempts=20),
        audio_dtype=render.audio_dtype,
    )
    assert sample.audio.shape == (176400, 2)
    assert np.isfinite(sample.audio).all()
    assert 0.001 < np.max(np.abs(sample.audio)) <= 1.0
    assert sample.param_array.shape == (148,)
    assert np.all((sample.param_array >= 0) & (sample.param_array <= 1))
    assert sample.mel_spec.shape == (2, 128, 401)
    assert np.isfinite(sample.mel_spec).all()
    assert len(sample.audio_mp3) > 1000
