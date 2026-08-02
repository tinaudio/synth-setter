"""Composed-config contracts for the TorchSynth train experiment."""

import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from synth_setter.clap import (
    DEFAULT_CLAP_TRAINING_CHECKPOINT,
    DEFAULT_CLAP_TRAINING_CHECKPOINT_SHA256,
)
from synth_setter.data.vst.shapes import mel_hop_length, mel_n_fft
from synth_setter.data.vst.torchsynth_param_spec import TORCHSYNTH_FULL_PARAM_SPEC
from synth_setter.models.components.spec_encoder import SpecEncoder
from synth_setter.models.components.transformer import (
    ApproxEquivTransformer,
    AudioSpectrogramTransformer,
    LearntProjection,
)
from synth_setter.pupujepa import (
    DEFAULT_PUPUJEPA_TINY_CHECKPOINT,
    PUPUJEPA_CHECKPOINT_REVISION,
)
from synth_setter.same import (
    DEFAULT_SAME_L_CHECKPOINT,
    DEFAULT_SAME_L_CHECKPOINT_SHA256,
    DEFAULT_SAME_S_CHECKPOINT,
    DEFAULT_SAME_S_CHECKPOINT_SHA256,
)


def test_torchsynth_datamodule_defaults_to_four_seconds_of_audio() -> None:
    """The datamodule group defaults to 4 s so envelope/LFO params are identifiable."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="train.yaml", overrides=["datamodule=torchsynth", "model=ffn"])

    assert cfg.datamodule.signal_length == 176_400
    assert cfg.datamodule.sample_rate == 44_100
    assert "emit_mel" not in cfg.datamodule


def test_torchsynth_flow_finetune_persists_four_workers_by_default() -> None:
    """Gradient-spectral finetuning reuses each render worker across epochs."""
    cfg = _experiment_cfg("flow_finetune")

    assert cfg.datamodule.num_workers == 4
    assert cfg.datamodule.persistent_workers is True


def test_torchsynth_flow_finetune_null_persists_four_workers_by_default() -> None:
    """Null-control finetuning reuses each render worker across epochs."""
    cfg = _experiment_cfg("flow_finetune_null")

    assert cfg.datamodule.num_workers == 4
    assert cfg.datamodule.persistent_workers is True


def test_torchsynth_persistent_workers_cli_override_disables_persistence() -> None:
    """The Hydra override can opt out of worker reuse for troubleshooting."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=torchsynth/flow_finetune_null",
                "datamodule.persistent_workers=false",
            ],
        )

    assert cfg.datamodule.persistent_workers is False


def test_default_callbacks_monitor_parameter_mse() -> None:
    """The shared callback group monitors the metric emitted by default train modules."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["datamodule=torchsynth", "model=ffn", "callbacks=default"],
        )

    assert cfg.callbacks.model_checkpoint.monitor == "val/param_mse"


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
    assert cfg.datamodule.drop_last is False
    assert (
        cfg.datamodule.collate_fn._target_
        == "synth_setter.data.torchsynth_datamodule.collate_audio_dict"
    )


def test_torchsynth_ffn_derives_log_mel_geometry_from_sample_rate() -> None:
    """Sample-rate overrides retain the dataset's derived mel geometry."""
    sample_rate = 16_000
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=torchsynth/ffn", f"datamodule.sample_rate={sample_rate}"],
        )

    network = instantiate(cfg.model.net)

    assert network.encoder.frontend.mel.n_fft == mel_n_fft(sample_rate)
    assert network.encoder.frontend.mel.hop_length == mel_hop_length(sample_rate)


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


def _experiment_cfg(experiment: str) -> DictConfig:
    """Compose one TorchSynth experiment.

    :param experiment: Experiment group member below ``torchsynth/``.
    :returns: The composed Hydra config.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        return compose(config_name="train.yaml", overrides=[f"experiment=torchsynth/{experiment}"])


def _flow_cfg() -> DictConfig:
    """Compose the TorchSynth flow experiment.

    :returns: The composed Hydra config.
    """
    return _experiment_cfg("flow")


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
    assert cfg.model.encoder._target_ == "synth_setter.models.components.spec_encoder.SpecEncoder"
    assert cfg.model.encoder.backbone._target_ == "synth_setter.models.components.cnn.MelCNN"
    assert (
        cfg.datamodule.collate_fn._target_
        == "synth_setter.data.torchsynth_datamodule.collate_audio_dict"
    )


def test_torchsynth_flow_experiment_composes_the_synth_identity_for_the_probe() -> None:
    """The val-audio probe needs both a render group and the root synth identity."""
    cfg = _flow_cfg()
    assert cfg.synth.param_spec_name == "torchsynth_full"
    assert cfg.render is not None


def test_torchsynth_flow_experiment_carries_no_audio_loss() -> None:
    """The base flow experiment is the control arm and must retain every train row."""
    cfg = _flow_cfg()
    assert cfg.model.get("audio_loss") is None
    assert cfg.datamodule.drop_last is False


def test_torchsynth_flow_experiment_uses_online_parameter_width() -> None:
    """The module and projection both match the online dataset's encoded row width."""
    cfg = _flow_cfg()
    width = TORCHSYNTH_FULL_PARAM_SPEC.encoded_width

    assert cfg.datamodule.num_params == width
    assert cfg.model.num_params == width
    assert cfg.model.vector_field.projection.num_params == width


def test_torchsynth_flow_experiment_resamples_training_rows() -> None:
    """The long-running flow experiment draws fresh online rows each epoch."""
    assert _flow_cfg().datamodule.resample_train_per_epoch is True


def test_torchsynth_flow_composed_predict_step_preserves_datamodule_width() -> None:
    """The production encoder, transformer, and projection predict all online parameters."""
    signal_length = 800
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=torchsynth/flow_audio",
                "datamodule.sample_rate=8000",
                f"datamodule.signal_length={signal_length}",
                "model.encoder.backbone.hidden_dim=2",
                "model.encoder.backbone.out_dim=8",
                "model.encoder.frontend.n_mels=16",
                "model.encoder.backbone.num_blocks=1",
                "model.encoder.backbone.kernel_size=3",
                "model.vector_field.d_model=8",
                "model.vector_field.num_heads=2",
                "model.vector_field.d_ff=8",
                "model.vector_field.num_layers=1",
                "model.vector_field.projection.num_tokens=2",
                "model.test_sample_steps=1",
            ],
        )

    model = instantiate(cfg.model)
    model.eval()
    predictions, _ = model.predict_step({"audio": torch.randn(2, signal_length)}, 0)

    assert isinstance(model.encoder, SpecEncoder)
    assert isinstance(model.vector_field, ApproxEquivTransformer)
    assert isinstance(model.vector_field.projection, LearntProjection)
    assert (
        cfg.model.num_params
        == cfg.datamodule.num_params
        == TORCHSYNTH_FULL_PARAM_SPEC.encoded_width
    )
    assert predictions.shape == (2, cfg.datamodule.num_params)
    assert torch.isfinite(predictions).all()


def test_clap_online_conditioning_composes_frozen_backbone_and_projection_head() -> None:
    """The online profile feeds raw audio through frozen CLAP before the trained head."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=torchsynth/flow_audio",
                "conditioning=clap_online",
                "model/encoder=clap_online",
            ],
        )

    assert cfg.model.conditioning == "audio"
    assert cfg.datamodule.conditioning == "audio"
    assert (
        cfg.model.encoder._target_
        == "synth_setter.models.components.pretrained_encoder.PretrainedConditioningEncoder"
    )
    assert (
        cfg.model.encoder.backbone._target_
        == "synth_setter.models.components.pretrained_encoder.ClapAudioEncoder.from_pretrained"
    )
    assert cfg.model.encoder.backbone.sample_rate == cfg.datamodule.sample_rate
    assert cfg.model.encoder.backbone.checkpoint == DEFAULT_CLAP_TRAINING_CHECKPOINT
    assert cfg.model.encoder.backbone.checkpoint_sha256 == DEFAULT_CLAP_TRAINING_CHECKPOINT_SHA256
    assert (
        cfg.model.encoder.head._target_
        == "synth_setter.models.components.vector_projection.VectorProjection"
    )
    assert cfg.model.encoder.head.input_dim == 512
    assert cfg.model.vector_field.conditioning_dim == cfg.model.encoder.out_dim


def test_ast_online_shares_the_vst_backbone_definition() -> None:
    """The online profile reuses the stored-mel AST rather than restating its geometry.

    Parity is the point of the split: only the mel's source (computed vs stored) and the
    channel count may differ, so a hyperparameter that drifts between the two paths means
    the online arm is no longer comparable to the VST arm.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        stored = compose(
            config_name="train.yaml", overrides=["datamodule=surge", "model=vst_flow"]
        )
        online = compose(
            config_name="train.yaml",
            overrides=["experiment=torchsynth/flow", "conditioning=ast_online"],
        )

    stored_ast = OmegaConf.to_container(stored.model.encoder, resolve=True)
    online_ast = OmegaConf.to_container(online.model.encoder.backbone, resolve=True)
    assert isinstance(stored_ast, dict) and isinstance(online_ast, dict)
    # Only the mel's channel count may differ; #2751 tracks collapsing that too.
    differing = {"input_channels"}
    assert {k: v for k, v in stored_ast.items() if k not in differing} == {
        k: v for k, v in online_ast.items() if k not in differing
    }


def test_ast_online_conditioning_derives_its_spectrogram_shape_from_the_datamodule() -> None:
    """The online AST is sized by the render geometry, not a hardcoded stored-mel shape."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=torchsynth/flow",
                "conditioning=ast_online",
                "datamodule.sample_rate=8000",
                "datamodule.signal_length=800",
            ],
        )

    assert cfg.model.conditioning == "audio"
    assert cfg.datamodule.conditioning == "audio"
    assert list(cfg.model.encoder.backbone.spec_shape) == [128, 11]
    # Online renders are mono; the stored-mel profile's patch embedding takes two channels.
    assert cfg.model.encoder.backbone.input_channels == 1
    assert cfg.model.vector_field.conditioning_dim == cfg.model.encoder.backbone.d_model


def test_ast_online_conditioning_predicts_from_online_audio() -> None:
    """The stored-mel transformer conditions on a waveform with no stored mel column."""
    signal_length = 800
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=torchsynth/flow",
                "conditioning=ast_online",
                "datamodule.sample_rate=8000",
                f"datamodule.signal_length={signal_length}",
                "model.encoder.frontend.n_mels=16",
                "model.encoder.backbone.d_model=8",
                "model.encoder.backbone.n_heads=2",
                "model.encoder.backbone.n_layers=1",
                "model.encoder.backbone.n_conditioning_outputs=2",
                "model.encoder.backbone.patch_size=4",
                "model.encoder.backbone.patch_stride=3",
                "model.encoder.backbone.spec_shape=[16,11]",
                "model.vector_field.d_model=8",
                "model.vector_field.num_heads=2",
                "model.vector_field.d_ff=8",
                "model.vector_field.num_layers=1",
                "model.vector_field.projection.num_tokens=2",
                "model.test_sample_steps=1",
            ],
        )

    model = instantiate(cfg.model)
    model.eval()
    predictions, _ = model.predict_step({"audio": torch.randn(2, signal_length)}, 0)

    assert isinstance(model.encoder, SpecEncoder)
    assert isinstance(model.encoder.backbone, AudioSpectrogramTransformer)
    assert predictions.shape == (2, cfg.datamodule.num_params)
    assert torch.isfinite(predictions).all()


@pytest.mark.parametrize(
    ("profile", "checkpoint", "checkpoint_sha256"),
    [
        (
            "same_l_online",
            DEFAULT_SAME_L_CHECKPOINT,
            DEFAULT_SAME_L_CHECKPOINT_SHA256,
        ),
        (
            "same_s_online",
            DEFAULT_SAME_S_CHECKPOINT,
            DEFAULT_SAME_S_CHECKPOINT_SHA256,
        ),
    ],
)
def test_same_online_conditioning_composes_frozen_backbone_and_temporal_pool(
    profile: str, checkpoint: str, checkpoint_sha256: str
) -> None:
    """Online SAME profiles pool frozen waveform latents into flow conditioning.

    :param profile: SAME conditioning profile under test.
    :param checkpoint: Expected pretrained SAME checkpoint.
    :param checkpoint_sha256: Expected materialized checkpoint digest.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=torchsynth/flow", f"conditioning={profile}"],
        )

    assert cfg.model.conditioning == "audio"
    assert cfg.datamodule.conditioning == "audio"
    assert (
        cfg.model.encoder._target_
        == "synth_setter.models.components.pretrained_encoder.PretrainedConditioningEncoder"
    )
    assert (
        cfg.model.encoder.backbone._target_
        == "synth_setter.models.components.same_encoder.SameAudioEncoder.from_pretrained"
    )
    assert cfg.model.encoder.backbone.sample_rate == cfg.datamodule.sample_rate
    assert cfg.model.encoder.backbone.checkpoint == checkpoint
    assert cfg.model.encoder.backbone.checkpoint_sha256 == checkpoint_sha256
    assert (
        cfg.model.encoder.head._target_
        == "synth_setter.models.components.embed_pool.EmbeddingPool"
    )
    assert cfg.model.encoder.head.embed_dim == 256
    assert cfg.model.encoder.head.max_seq_len == 44
    assert cfg.model.vector_field.conditioning_dim == cfg.model.encoder.out_dim


def test_pupujepa_online_conditioning_composes_frozen_teacher_and_temporal_pool() -> None:
    """Online PupuJEPA pools the pinned waveform teacher into flow conditioning."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=torchsynth/flow", "conditioning=pupujepa_tiny_online"],
        )

    assert cfg.model.conditioning == "audio"
    assert cfg.datamodule.conditioning == "audio"
    assert (
        cfg.model.encoder._target_
        == "synth_setter.models.components.pretrained_encoder.PretrainedConditioningEncoder"
    )
    assert (
        cfg.model.encoder.backbone._target_
        == "synth_setter.models.components.pupujepa_encoder.PupuJepaAudioEncoder.from_pretrained"
    )
    assert cfg.model.encoder.backbone.sample_rate == cfg.datamodule.sample_rate
    assert cfg.model.encoder.backbone.checkpoint == DEFAULT_PUPUJEPA_TINY_CHECKPOINT
    assert cfg.model.encoder.backbone.revision == PUPUJEPA_CHECKPOINT_REVISION
    assert (
        cfg.model.encoder.head._target_
        == "synth_setter.models.components.embed_pool.EmbeddingPool"
    )
    assert cfg.model.encoder.head.embed_dim == 1536
    assert cfg.model.encoder.head.max_seq_len == 256
    assert cfg.model.vector_field.conditioning_dim == cfg.model.encoder.out_dim


def test_torchsynth_flow_audio_experiment_attaches_the_latent_audio_loss() -> None:
    """The audio arm swaps in the render-feedback term with the datamodule's geometry."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="train.yaml", overrides=["experiment=torchsynth/flow_audio"])

    assert (
        cfg.model.audio_loss._target_
        == "synth_setter.models.components.audio_feedback.AudioFeedbackLoss"
    )
    assert cfg.model.audio_loss.sample_rate == 44_100
    assert cfg.model.audio_loss.signal_length == 176_400
    assert cfg.model.audio_loss.render_batch_size == cfg.datamodule.batch_size
    # torch.compile traces through the renderer's functional_call and miscompiles.
    assert cfg.model.compile is False


@pytest.mark.parametrize("experiment", ["ffn", "flow", "flow_audio"])
def test_torchsynth_audio_experiment_collation_returns_exact_keys(experiment: str) -> None:
    """Every audio-conditioned experiment emits only params, noise, and audio.

    :param experiment: TorchSynth experiment group member under test.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                f"experiment=torchsynth/{experiment}",
                "datamodule.signal_length=4410",
                "datamodule.train_val_test_sizes=[2,1,1]",
                "datamodule.batch_size=2",
                "datamodule.num_workers=0",
            ],
        )

    datamodule = instantiate(cfg.datamodule)
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))

    assert set(batch) == {"params", "noise", "audio"}
    width = TORCHSYNTH_FULL_PARAM_SPEC.encoded_width
    assert batch["params"].shape == (2, width)
    assert batch["noise"].shape == (2, width)
    assert batch["audio"].shape == (2, 4_410)
    assert all(value.dtype == torch.float32 for value in batch.values())


def test_torchsynth_flow_audio_tiny_split_retains_partial_batch() -> None:
    """An undersized split still yields its partial batch; the render pads it up."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=torchsynth/flow_audio",
                "datamodule.signal_length=4410",
                "datamodule.train_val_test_sizes=[1,1,1]",
                "datamodule.batch_size=2",
                "datamodule.num_workers=0",
            ],
        )

    datamodule = instantiate(cfg.datamodule)
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()

    assert loader.drop_last is False
    assert len(next(iter(loader))["audio"]) == 1


@pytest.mark.parametrize("experiment", ["ffn", "flow", "flow_audio"])
def test_torchsynth_experiment_checkpoint_monitor_is_param_mse(experiment: str) -> None:
    """Every TorchSynth experiment explicitly resolves the parameter-MSE monitor.

    :param experiment: TorchSynth experiment group member under test.
    """
    assert _experiment_cfg(experiment).callbacks.model_checkpoint.monitor == "val/param_mse"


def test_clap_audio_loss_composes_with_stored_embedding_conditioning() -> None:
    """The feedback space is selectable independently of what conditions the flow."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=torchsynth/flow_audio",
                "conditioning=m2l",
                "model/audio_loss=clap",
            ],
        )

    assert cfg.model.encoder._target_ == "synth_setter.models.components.embed_pool.EmbeddingPool"
    assert (
        cfg.model.audio_loss.distance.encoder._target_
        == "synth_setter.models.components.pretrained_encoder.ClapAudioEncoder.from_pretrained"
    )


def test_torchsynth_flow_validates_often_enough_to_checkpoint_within_an_epoch() -> None:
    """The trainer default outlives an epoch of this size, so nothing would be saved."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="train.yaml", overrides=["experiment=torchsynth/flow"])

    assert cfg.trainer.val_check_interval == 2000
    assert cfg.training.val_audio_probe is True


def test_mss_audio_loss_measures_in_the_reported_metric_space() -> None:
    """The default feedback space is the figure evaluation reports, in the same units."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="train.yaml", overrides=["experiment=torchsynth/flow_audio"])

    assert (
        cfg.model.audio_loss.distance._target_
        == "synth_setter.models.components.audio_distance.MultiScaleSpectralDistance"
    )


def test_conditioning_profile_alone_selects_its_encoder() -> None:
    """The experiment must not pin an encoder the conditioning profile owns."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=torchsynth/flow_audio", "conditioning=clap_online"],
        )

    assert (
        cfg.model.encoder._target_
        == "synth_setter.models.components.pretrained_encoder.PretrainedConditioningEncoder"
    )


def test_same_audio_loss_measures_in_the_stored_conditioning_space() -> None:
    """The SAME arm scores renders in the space the `same_s` column is written in."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml", overrides=["experiment=torchsynth/flow_audio_same"]
        )

    assert (
        cfg.model.audio_loss.distance._target_
        == "synth_setter.models.components.audio_distance.LatentMseDistance"
    )
    assert (
        cfg.model.audio_loss.distance.encoder._target_
        == "synth_setter.models.components.same_encoder.SameAudioEncoder.from_pretrained"
    )
    assert cfg.model.audio_loss.distance.encoder.checkpoint == DEFAULT_SAME_S_CHECKPOINT
    assert cfg.model.audio_loss.sample_rate == 44_100
    assert cfg.model.audio_loss.render_batch_size == cfg.datamodule.batch_size
    assert cfg.model.compile is False


def test_same_audio_loss_experiment_conditions_on_online_same_s() -> None:
    """The SAME arm uses frozen SAME-S for both conditioning and render distance."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml", overrides=["experiment=torchsynth/flow_audio_same"]
        )

    assert cfg.model.conditioning == "audio"
    assert cfg.datamodule.conditioning == "audio"
    assert (
        cfg.model.encoder.backbone._target_
        == "synth_setter.models.components.same_encoder.SameAudioEncoder.from_pretrained"
    )
    assert cfg.model.encoder.backbone.checkpoint == DEFAULT_SAME_S_CHECKPOINT
    assert cfg.model.encoder.backbone.checkpoint_sha256 == DEFAULT_SAME_S_CHECKPOINT_SHA256
    assert (
        cfg.model.encoder.head._target_
        == "synth_setter.models.components.embed_pool.EmbeddingPool"
    )
