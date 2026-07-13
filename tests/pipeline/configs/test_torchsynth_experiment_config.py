"""Composed-config contracts for the TorchSynth train experiment."""

from hydra import compose, initialize_config_module


def test_torchsynth_datamodule_defaults_to_four_seconds_of_audio() -> None:
    """The datamodule group defaults to 4 s so envelope/LFO params are identifiable."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="train.yaml", overrides=["datamodule=torchsynth", "model=ffn"])

    assert cfg.datamodule.signal_length == 176_400
    assert cfg.datamodule.sample_rate == 44_100


def test_torchsynth_ffn_experiment_monitors_lsd_with_runnable_input_length() -> None:
    """The FFN experiment checkpoints on ``val/lsd`` and pins an instantiable ``in_dim``.

    The raw-waveform head scales O(in_dim**2) (``LazyLinear(in_dim // 2)`` on flattened
    conv features), so the experiment must not inherit the 4 s datamodule default until
    the spectral front-end (#1848) lands.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="train.yaml", overrides=["experiment=torchsynth/ffn"])

    assert cfg.callbacks.model_checkpoint.monitor == "val/lsd"
    assert cfg.datamodule.signal_length == 4_410
    assert cfg.model.net.in_dim == 4_410
    assert cfg.datamodule.resample_train_per_epoch is True
