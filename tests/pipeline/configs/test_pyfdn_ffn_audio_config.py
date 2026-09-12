"""Hydra contract for pyFDN FFN training with FLAMO audio feedback."""

from hydra import compose, initialize_config_module


def test_pyfdn_ffn_audio_flamo_experiment_wires_feedback_training() -> None:
    """The experiment supplies target audio and an uncompiled FLAMO-backed FFN."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="train.yaml", overrides=["experiment=pyfdn/ffn_audio_flamo"])

    assert cfg.model._target_ == "synth_setter.models.vst_ff_module.VSTFeedForwardModule"
    assert cfg.model.audio_loss.renderer._target_.endswith(
        "FlamoFDNDifferentiableRenderer.from_param_spec"
    )
    assert cfg.model.audio_loss.renderer.param_spec == "pyfdn_n8_mono_householder"
    assert cfg.datamodule.include_audio is True
    assert cfg.model.compile is False
