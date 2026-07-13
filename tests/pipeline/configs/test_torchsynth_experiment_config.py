"""Composed-config contracts for the TorchSynth train experiment."""

from hydra import compose, initialize_config_module


def test_torchsynth_ffn_experiment_monitors_lsd_on_four_second_audio() -> None:
    """The production TorchSynth experiment checkpoints on ``val/lsd`` over 4 s audio."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="train.yaml", overrides=["experiment=torchsynth/ffn"])

    assert cfg.callbacks.model_checkpoint.monitor == "val/lsd"
    assert cfg.datamodule.signal_length == 176_400
    assert cfg.datamodule.sample_rate == 44_100
    assert cfg.datamodule.resample_train_per_epoch is True
