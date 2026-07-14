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
    """The production experiment uses a memory-bounded four-second frontend."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="train.yaml", overrides=["experiment=torchsynth/ffn"])

    assert cfg.callbacks.model_checkpoint.monitor == "val/lsd"
    assert cfg.datamodule.signal_length == 176_400
    assert cfg.model.net.in_dim == 176_400
    assert cfg.model.net.frontend == "log_mel"
    assert cfg.model.net.center is True
    assert cfg.model.net.f_max is None
    assert cfg.model.net.f_min == 0.0
    assert cfg.model.net.hop_length == 441
    assert cfg.model.net.mel_norm == "slaney"
    assert cfg.model.net.mel_scale == "slaney"
    assert cfg.model.net.n_fft == 1_102
    assert cfg.model.net.n_mels == 128
    assert cfg.model.net.pad_mode == "constant"
    assert cfg.model.net.power == 2.0
    assert cfg.model.net.sample_rate == 44_100
    assert cfg.model.net.top_db == 80.0
    assert cfg.model.net.window == "hamming"
    assert cfg.datamodule.resample_train_per_epoch is True


def test_torchsynth_ffn_four_second_model_has_bounded_parameter_count() -> None:
    """Keep the production network below its memory-safe parameter limit."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="train.yaml", overrides=["experiment=torchsynth/ffn"])

    network = instantiate(cfg.model.net)

    assert sum(parameter.numel() for parameter in network.parameters()) < 3_000_000
