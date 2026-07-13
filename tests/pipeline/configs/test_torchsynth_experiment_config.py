"""Composed-config contracts for the TorchSynth train experiment."""

from hydra import compose, initialize_config_module
from hydra.utils import instantiate


def test_torchsynth_datamodule_defaults_to_four_seconds_of_audio() -> None:
    """The datamodule group defaults to 4 s so envelope/LFO params are identifiable."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="train.yaml", overrides=["datamodule=torchsynth", "model=ffn"])

    assert cfg.datamodule.signal_length == 176_400
    assert cfg.datamodule.sample_rate == 44_100


def test_torchsynth_ffn_experiment_uses_four_second_log_mel_frontend() -> None:
    """The FFN experiment uses 4 s audio with the bounded log-mel encoder."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="train.yaml", overrides=["experiment=torchsynth/ffn"])

    assert cfg.callbacks.model_checkpoint.monitor == "val/lsd"
    assert cfg.datamodule.signal_length == 176_400
    assert cfg.model.net.in_dim == 176_400
    assert cfg.model.net.frontend == "log_mel"
    assert cfg.model.net.sample_rate == 44_100
    assert cfg.datamodule.resample_train_per_epoch is True


def test_torchsynth_ffn_four_second_model_has_bounded_parameter_count() -> None:
    """The production 4 s network stays well below the former 36.9 B parameters."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="train.yaml", overrides=["experiment=torchsynth/ffn"])

    network = instantiate(cfg.model.net)

    assert sum(parameter.numel() for parameter in network.parameters()) < 3_000_000
