"""Composed-config contracts for the TorchSynth train experiment."""

from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import DictConfig

from synth_setter.data.vst.shapes import mel_hop_length, mel_n_fft


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

    assert cfg.callbacks.model_checkpoint.monitor == "val/param_mse"
    assert cfg.model._target_ == "synth_setter.models.vst_ff_module.VSTFeedForwardModule"
    assert cfg.datamodule.signal_length == 176_400
    assert cfg.model.net.in_dim == 176_400
    assert (
        cfg.model.net._target_
        == "synth_setter.models.components.residual_mlp.LogMelCNNResidualMLP"
    )
    assert cfg.model.net.center is True
    assert cfg.model.net.f_max is None
    assert cfg.model.net.f_min == 0.0
    assert cfg.model.net.mel_norm == "slaney"
    assert cfg.model.net.mel_scale == "slaney"
    assert cfg.model.net.n_mels == 128
    assert cfg.model.net.pad_mode == "constant"
    assert cfg.model.net.power == 2.0
    assert cfg.model.net.sample_rate == 44_100
    assert cfg.model.net.top_db == 80.0
    assert cfg.model.net.window == "hamming"
    assert cfg.datamodule.resample_train_per_epoch is True


def test_torchsynth_ffn_derives_log_mel_geometry_from_sample_rate() -> None:
    """Sample-rate overrides retain the dataset's derived mel geometry."""
    sample_rate = 16_000
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=torchsynth/ffn", f"datamodule.sample_rate={sample_rate}"],
        )

    network = instantiate(cfg.model.net)

    assert network.encoder.mel.n_fft == mel_n_fft(sample_rate)
    assert network.encoder.mel.hop_length == mel_hop_length(sample_rate)


def test_torchsynth_ffn_four_second_model_has_bounded_parameter_count() -> None:
    """Keep the production network below its memory-safe parameter limit."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="train.yaml", overrides=["experiment=torchsynth/ffn"])

    network = instantiate(cfg.model.net)

    assert sum(parameter.numel() for parameter in network.parameters()) < 3_000_000


def test_torchsynth_ffn_composed_model_trains_on_an_online_dict_batch() -> None:
    """The migrated FFN module consumes a real online torchsynth batch end to end.

    Instantiates the composed experiment model (not just its target string) and checks a training
    step produces a finite loss with gradients in the network.
    """
    import torch

    signal_length = 4_410
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=torchsynth/ffn",
                f"datamodule.signal_length={signal_length}",
                "datamodule.train_val_test_sizes=[4,2,2]",
                "datamodule.batch_size=4",
                "datamodule.num_workers=0",
            ],
        )

    datamodule = instantiate(cfg.datamodule)
    datamodule.setup("fit")
    model = instantiate(cfg.model)
    batch = next(iter(datamodule.train_dataloader()))

    loss, preds, targets, _ = model.model_step(batch)
    loss.backward()

    assert preds.shape == targets.shape == batch["params"].shape
    assert torch.isfinite(loss)
    gradients = [p.grad for p in model.net.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def _flow_cfg() -> DictConfig:
    """Compose the torchsynth flow experiment.

    :returns: The composed Hydra config.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        return compose(config_name="train.yaml", overrides=["experiment=torchsynth/flow"])


def test_torchsynth_flow_experiment_uses_the_vst_flow_module() -> None:
    """Torchsynth trains through the same flow module as every VST synth."""
    cfg = _flow_cfg()
    assert (
        cfg.model._target_ == "synth_setter.models.vst_flow_matching_module.VSTFlowMatchingModule"
    )
    assert (
        cfg.datamodule._target_ == "synth_setter.data.torchsynth_datamodule.TorchSynthDataModule"
    )


def test_torchsynth_flow_experiment_observes_raw_audio() -> None:
    """Online rendering has no stored mel column, so the encoder takes the waveform."""
    cfg = _flow_cfg()
    assert cfg.model.conditioning == "audio"
    assert cfg.model.encoder._target_ == "synth_setter.models.components.cnn.LogMelEncoder"


def test_torchsynth_flow_experiment_composes_the_synth_identity_for_the_probe() -> None:
    """The val-audio probe needs both a render group and the root synth identity."""
    cfg = _flow_cfg()
    assert cfg.synth.param_spec_name == "torchsynth_full"
    assert cfg.training.val_audio_probe == "auto"


def test_torchsynth_flow_experiment_carries_no_audio_loss() -> None:
    """The base flow experiment is the control arm and must not pay for renders."""
    cfg = _flow_cfg()
    assert cfg.model.get("audio_loss") is None


def test_torchsynth_flow_audio_experiment_attaches_the_mslm_audio_loss() -> None:
    """The audio arm swaps in the render-feedback term with the datamodule's geometry."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="train.yaml", overrides=["experiment=torchsynth/flow_audio"])

    assert (
        cfg.model.audio_loss._target_
        == "synth_setter.models.components.audio_feedback.AudioFeedbackLoss"
    )
    assert cfg.model.audio_loss.distance == "mslm"
    assert cfg.model.audio_loss.sample_rate == 44_100
    assert cfg.model.audio_loss.signal_length == 176_400
    assert cfg.model.audio_loss.midi_pitch == 60
    # torch.compile traces through the renderer's functional_call and miscompiles.
    assert cfg.model.compile is False
