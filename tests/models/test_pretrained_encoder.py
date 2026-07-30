"""Behavior tests for frozen online CLAP conditioning."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torchaudio.functional as audio_fn
from transformers import ClapConfig, ClapFeatureExtractor, ClapModel

from synth_setter.clap import clap_checkpoint_sha256
from synth_setter.models.components import pretrained_encoder as pretrained_encoder_module
from synth_setter.models.components.pretrained_encoder import (
    ClapAudioEncoder,
    PretrainedConditioningEncoder,
)
from synth_setter.models.components.vector_field import VectorField
from synth_setter.models.components.vector_projection import VectorProjection
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule

_SAMPLE_RATE = 48_000
_PROJECTION_DIM = 8
_TINY_CLAP_CONFIG: dict[str, object] = {
    "projection_dim": _PROJECTION_DIM,
    "text_config": {
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "max_position_embeddings": 16,
        "projection_dim": _PROJECTION_DIM,
    },
    "audio_config": {
        "num_mel_bins": 64,
        "spec_size": 256,
        "hidden_size": 32,
        "projection_dim": _PROJECTION_DIM,
        "depths": [1, 1, 1, 1],
        "num_attention_heads": [1, 2, 4, 8],
        "patch_embeds_hidden_size": 4,
        "patch_size": 4,
        "patch_stride": [4, 4],
        "num_classes": 8,
        "window_size": 4,
    },
}


@pytest.fixture(scope="module")
def clap_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Create an offline feature-extractor checkpoint with production geometry.

    :param tmp_path_factory: Module-scoped temporary directory factory.
    :returns: Local CLAP feature-extractor directory.
    """
    checkpoint = tmp_path_factory.mktemp("clap-feature-extractor")
    ClapFeatureExtractor(
        feature_size=64,
        sampling_rate=_SAMPLE_RATE,
        hop_length=480,
        fft_window_size=1024,
        max_length_s=10,
        frequency_min=50,
        frequency_max=14_000,
        truncation="rand_trunc",
        padding="repeatpad",
    ).save_pretrained(checkpoint)
    text_config = _TINY_CLAP_CONFIG["text_config"]
    audio_config = _TINY_CLAP_CONFIG["audio_config"]
    assert isinstance(text_config, dict)
    assert isinstance(audio_config, dict)
    ClapModel(
        ClapConfig(
            projection_dim=_PROJECTION_DIM,
            text_config=text_config,
            audio_config=audio_config,
        )
    ).save_pretrained(checkpoint)
    return str(checkpoint)


def _checkpoint_with_extractor_overrides(
    checkpoint: str, destination: Path, overrides: dict[str, object]
) -> str:
    """Copy a tiny checkpoint and override feature-extractor fields.

    :param checkpoint: Source checkpoint directory.
    :param destination: Destination directory.
    :param overrides: ``preprocessor_config.json`` values to replace.
    :returns: Copied checkpoint path.
    """
    shutil.copytree(checkpoint, destination)
    config_path = destination / "preprocessor_config.json"
    config = json.loads(config_path.read_text())
    config.update(overrides)
    config_path.write_text(json.dumps(config))
    return str(destination)


@pytest.fixture(scope="module")
def clap_encoder(clap_checkpoint: str) -> ClapAudioEncoder:
    """Build a deterministic small random CLAP network.

    :param clap_checkpoint: Self-contained feature-extractor checkpoint.
    :returns: Frozen waveform encoder suitable for CPU behavior tests.
    """
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return ClapAudioEncoder.from_random_config(
            sample_rate=_SAMPLE_RATE,
            checkpoint=clap_checkpoint,
            backbone_config=_TINY_CLAP_CONFIG,
        )


def test_clap_training_defaults_supply_checkpoint_and_digest(
    clap_checkpoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitted factory values resolve through the training-only shared identity.

    :param clap_checkpoint: Self-contained tiny CLAP checkpoint.
    :param monkeypatch: Pytest patcher for the shared training defaults.
    """
    monkeypatch.setattr(
        pretrained_encoder_module, "DEFAULT_CLAP_TRAINING_CHECKPOINT", clap_checkpoint
    )
    monkeypatch.setattr(
        pretrained_encoder_module,
        "DEFAULT_CLAP_TRAINING_CHECKPOINT_SHA256",
        clap_checkpoint_sha256(Path(clap_checkpoint)),
    )

    encoder = ClapAudioEncoder.from_random_config(
        sample_rate=_SAMPLE_RATE,
        backbone_config=_TINY_CLAP_CONFIG,
    )

    assert encoder.target_sample_rate == _SAMPLE_RATE


def test_clap_from_pretrained_loads_checkpoint_weights(clap_checkpoint: str) -> None:
    """The explicit pretrained factory loads the checkpoint model.

    :param clap_checkpoint: Self-contained tiny CLAP checkpoint.
    """
    encoder = ClapAudioEncoder.from_pretrained(
        sample_rate=_SAMPLE_RATE,
        checkpoint=clap_checkpoint,
    )

    expected = ClapModel.from_pretrained(clap_checkpoint).state_dict()[
        "audio_projection.linear1.weight"
    ]
    actual = encoder.clap.state_dict()["audio_projection.linear1.weight"]
    assert torch.equal(actual, expected)


def test_clap_random_factory_requires_backbone_config(clap_checkpoint: str) -> None:
    """Random construction cannot silently fall back to Transformers defaults.

    :param clap_checkpoint: Self-contained tiny CLAP checkpoint.
    """
    with pytest.raises(TypeError, match="backbone_config"):
        ClapAudioEncoder.from_random_config(  # pyright: ignore[reportCallIssue]
            sample_rate=_SAMPLE_RATE,
            checkpoint=clap_checkpoint,
        )


def test_clap_checkpoint_with_wrong_digest_raises_without_deleting_cache(
    clap_checkpoint: str,
) -> None:
    """A digest mismatch fails without mutating the shared checkpoint cache.

    :param clap_checkpoint: Self-contained tiny CLAP checkpoint.
    """
    marker = Path(clap_checkpoint) / "must-survive-digest-failure"
    marker.write_text("shared cache")

    with pytest.raises(ValueError, match="digest mismatch"):
        ClapAudioEncoder.from_random_config(
            sample_rate=_SAMPLE_RATE,
            checkpoint=clap_checkpoint,
            checkpoint_sha256="0" * 64,
            backbone_config=_TINY_CLAP_CONFIG,
        )

    assert marker.read_text() == "shared cache"


def test_clap_offline_backbone_with_unknown_config_key_raises(clap_checkpoint: str) -> None:
    """Transformers rejects malformed offline config at construction.

    :param clap_checkpoint: Self-contained feature-extractor checkpoint.
    """
    with pytest.raises((TypeError, ValueError), match="unknown_option"):
        ClapAudioEncoder.from_random_config(
            sample_rate=_SAMPLE_RATE,
            checkpoint=clap_checkpoint,
            backbone_config={"unknown_option": True},
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"truncation": "fusion"}, "truncation"),
        ({"padding": "repeat"}, "padding"),
        ({"feature_size": 63}, "num_mel_bins"),
    ],
)
def test_clap_random_factory_with_unsupported_extractor_geometry_raises(
    clap_checkpoint: str,
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    """Only the deterministic frontend geometry is accepted.

    :param clap_checkpoint: Self-contained tiny CLAP checkpoint.
    :param tmp_path: Temporary checkpoint destination.
    :param overrides: Unsupported extractor fields.
    :param message: Expected validation field.
    """
    checkpoint = _checkpoint_with_extractor_overrides(
        clap_checkpoint, tmp_path / "unsupported", overrides
    )

    with pytest.raises(ValueError, match=message):
        ClapAudioEncoder.from_random_config(
            sample_rate=_SAMPLE_RATE,
            checkpoint=checkpoint,
            backbone_config=_TINY_CLAP_CONFIG,
        )


def test_clap_features_match_huggingface_short_audio_path(
    clap_encoder: ClapAudioEncoder, clap_checkpoint: str
) -> None:
    """Online torch features agree with the extractor used for stored CLAP vectors.

    :param clap_encoder: Differentiable online encoder under test.
    :param clap_checkpoint: Self-contained reference extractor checkpoint.
    """
    audio = torch.sin(torch.arange(_SAMPLE_RATE, dtype=torch.float32) * 0.01).unsqueeze(0)
    extractor = ClapFeatureExtractor.from_pretrained(clap_checkpoint)
    expected = extractor(list(audio.numpy()), sampling_rate=_SAMPLE_RATE, return_tensors="pt")[
        "input_features"
    ]

    actual = clap_encoder.features(audio)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-5, rtol=0.0)


def test_clap_target_sample_rate_comes_from_feature_extractor(
    clap_checkpoint: str,
    tmp_path: Path,
) -> None:
    """A nonstandard checkpoint controls the encoder's resampling target.

    :param clap_checkpoint: Self-contained tiny CLAP checkpoint.
    :param tmp_path: Temporary checkpoint destination.
    """
    target_sample_rate = 32_000
    checkpoint = _checkpoint_with_extractor_overrides(
        clap_checkpoint,
        tmp_path / "32khz",
        {"sampling_rate": target_sample_rate, "nb_max_samples": 320_000},
    )
    encoder = ClapAudioEncoder.from_random_config(
        sample_rate=44_100,
        checkpoint=checkpoint,
        backbone_config=_TINY_CLAP_CONFIG,
    )

    assert encoder.sample_rate == 44_100
    assert encoder.target_sample_rate == target_sample_rate


def test_clap_features_match_huggingface_after_44100_hz_resampling(
    clap_checkpoint: str,
) -> None:
    """The shipped TorchSynth rate matches HF after the documented resampling.

    :param clap_checkpoint: Self-contained reference extractor checkpoint.
    """
    sample_rate = 44_100
    audio = torch.sin(torch.arange(sample_rate, dtype=torch.float32) * 0.01).unsqueeze(0)
    encoder = ClapAudioEncoder.from_random_config(
        sample_rate=sample_rate,
        checkpoint=clap_checkpoint,
        backbone_config=_TINY_CLAP_CONFIG,
    )
    extractor = ClapFeatureExtractor.from_pretrained(clap_checkpoint)
    resampled = audio_fn.resample(audio, sample_rate, _SAMPLE_RATE).clamp(-1.0, 1.0)
    expected = extractor(list(resampled.numpy()), sampling_rate=_SAMPLE_RATE, return_tensors="pt")[
        "input_features"
    ]

    actual = encoder.features(audio)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=0.0)


def test_clap_stereo_features_match_mono_channel_mean(clap_encoder: ClapAudioEncoder) -> None:
    """Stored stereo audio uses the same mono downmix as offline CLAP embeddings.

    :param clap_encoder: Small frozen CLAP encoder under test.
    """
    left = torch.sin(torch.arange(4_800, dtype=torch.float32) * 0.01)
    right = torch.cos(torch.arange(4_800, dtype=torch.float32) * 0.01)
    stereo = torch.stack((left, right)).unsqueeze(0)

    assert torch.equal(clap_encoder.features(stereo), clap_encoder.features(stereo.mean(dim=1)))


def test_clap_forward_backpropagates_to_waveform(clap_encoder: ClapAudioEncoder) -> None:
    """The frozen backbone retains the graph from embedding to input waveform.

    :param clap_encoder: Small frozen CLAP encoder under test.
    """
    audio = torch.sin(torch.arange(4_800, dtype=torch.float32) * 0.01).unsqueeze(0)
    audio.requires_grad_()

    embedding = clap_encoder(audio)
    (gradient,) = torch.autograd.grad(embedding.square().sum(), audio)

    assert embedding.shape == (1, _PROJECTION_DIM)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient).item() > 0


def test_clap_batch_jacobian_is_row_isolated(clap_encoder: ClapAudioEncoder) -> None:
    """Each frontend row depends only on its matching waveform row.

    :param clap_encoder: Small frozen CLAP encoder under test.
    """
    samples = torch.arange(4_800, dtype=torch.float32)
    audio = torch.stack((torch.sin(samples * 0.01), torch.cos(samples * 0.01))).requires_grad_()
    features = clap_encoder.features(audio)

    first_gradient = torch.autograd.grad(features[0].square().sum(), audio, retain_graph=True)[0]
    second_gradient = torch.autograd.grad(features[1].square().sum(), audio)[0]

    assert torch.count_nonzero(first_gradient[0]).item() > 0
    assert torch.equal(first_gradient[1], torch.zeros_like(first_gradient[1]))
    assert torch.equal(second_gradient[0], torch.zeros_like(second_gradient[0]))
    assert torch.count_nonzero(second_gradient[1]).item() > 0


def test_clap_backbone_stays_frozen_and_in_eval_mode(clap_encoder: ClapAudioEncoder) -> None:
    """Parent training-mode changes cannot move CLAP weights or stochastic state.

    :param clap_encoder: Small frozen CLAP encoder under test.
    """
    clap_encoder.train()

    assert not clap_encoder.clap.training
    assert all(not parameter.requires_grad for parameter in clap_encoder.clap.parameters())


def test_pretrained_conditioning_encoder_exposes_metric_and_conditioning_taps(
    clap_encoder: ClapAudioEncoder,
) -> None:
    """The frozen embedding and trained projection are usable from one forward pass.

    :param clap_encoder: Small frozen CLAP encoder under test.
    """
    encoder = PretrainedConditioningEncoder(
        backbone=clap_encoder,
        head=VectorProjection(input_dim=_PROJECTION_DIM, d_model=6),
        out_dim=6,
    )
    audio = torch.sin(torch.arange(4_800, dtype=torch.float32) * 0.01).unsqueeze(0)

    embedding = encoder.embed(audio)
    conditioning = encoder.project(embedding)

    assert embedding.shape == (1, _PROJECTION_DIM)
    assert conditioning.shape == (1, 6)
    assert torch.equal(embedding, encoder.frozen_audio_embedder(audio))
    assert torch.equal(conditioning, encoder(audio))


def _flow_module(clap_encoder: ClapAudioEncoder) -> VSTFlowMatchingModule:
    """Build a tiny flow module with a frozen CLAP conditioning encoder.

    :param clap_encoder: Small frozen CLAP encoder.
    :returns: Flow module with trainable projection and vector field.
    """
    encoder = PretrainedConditioningEncoder(
        backbone=clap_encoder,
        head=VectorProjection(input_dim=_PROJECTION_DIM, d_model=6),
        out_dim=6,
    )
    return VSTFlowMatchingModule(
        encoder=encoder,
        vector_field=VectorField(field_dim=4, hidden_dim=8, conditioning_dim=6, num_blocks=1),
        optimizer=torch.optim.Adam,  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=4,
        conditioning="audio",
        cfg_dropout_rate=0.0,
        warmup_steps=0,
    )


def test_clap_projection_conditioning_overfits_fixed_batch(
    clap_encoder: ClapAudioEncoder,
) -> None:
    """The trainable projection learns a fixed mapping from frozen CLAP embeddings.

    :param clap_encoder: Small frozen CLAP encoder producing the conditioning vectors.
    """
    torch.manual_seed(0)
    projection = VectorProjection(input_dim=_PROJECTION_DIM, d_model=4)
    encoder = PretrainedConditioningEncoder(
        backbone=clap_encoder,
        head=projection,
        out_dim=4,
    )
    predictor = torch.nn.Linear(4, 2)
    samples = torch.arange(4_800, dtype=torch.float32)
    audio = torch.stack((torch.sin(samples * 0.01), torch.cos(samples * 0.01)))
    with torch.no_grad():
        embeddings = encoder.embed(audio)
    targets = torch.tensor(((-1.0, 1.0), (1.0, -1.0)))
    initial_projection = projection.projection.weight.detach().clone()
    # 3e-3 over 2000 steps: at 3e-2 the loss oscillates past 1e-3 late in training, so a
    # single sampled step passes or fails on which side of the swing it lands.
    optimizer = torch.optim.Adam((*projection.parameters(), *predictor.parameters()), lr=3e-3)

    initial_loss = torch.nn.functional.mse_loss(predictor(encoder.project(embeddings)), targets)
    loss = initial_loss
    for _ in range(2000):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(predictor(encoder.project(embeddings)), targets)
        loss.backward()
        optimizer.step()

    assert loss.item() < 1e-3
    assert loss.item() < initial_loss.item()
    assert not torch.equal(projection.projection.weight, initial_projection)


def test_optimizer_excludes_frozen_clap_backbone(clap_encoder: ClapAudioEncoder) -> None:
    """Lightning optimizes the projection and field without registering CLAP parameters.

    :param clap_encoder: Small frozen CLAP encoder under test.
    """
    module = _flow_module(clap_encoder)
    module._trainer = SimpleNamespace(model=module)  # pyright: ignore[reportAttributeAccessIssue]

    optimizer = cast(torch.optim.Optimizer, module.configure_optimizers()["optimizer"])
    optimized = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }

    assert optimized
    assert all(id(parameter) not in optimized for parameter in clap_encoder.parameters())
    assert all(
        id(parameter) in optimized for parameter in module.parameters() if parameter.requires_grad
    )


def test_flow_loss_backprops_into_trainable_head_and_field(
    clap_encoder: ClapAudioEncoder,
) -> None:
    """Both trainable aggregates receive a finite nonzero flow gradient.

    :param clap_encoder: Small frozen CLAP encoder under test.
    """
    module = _flow_module(clap_encoder)
    samples = torch.arange(4_800, dtype=torch.float32)
    batch = {
        "audio": torch.stack((torch.sin(samples * 0.01), torch.cos(samples * 0.01))),
        "noise": torch.randn(2, 4),
        "params": torch.randn(2, 4),
    }

    module._train_step(batch).loss.backward()

    encoder = cast(PretrainedConditioningEncoder, module.encoder)
    vector_field = cast(torch.nn.Module, module.vector_field)
    head_gradients = [parameter.grad for parameter in encoder.head.parameters()]
    field_gradients = [
        parameter.grad for parameter in vector_field.parameters() if parameter.grad is not None
    ]
    assert all(
        gradient is not None and torch.isfinite(gradient).all() for gradient in head_gradients
    )
    assert sum(gradient.abs().sum() for gradient in head_gradients if gradient is not None) > 0
    assert field_gradients
    assert all(torch.isfinite(gradient).all() for gradient in field_gradients)
    assert sum(gradient.abs().sum() for gradient in field_gradients) > 0


def test_checkpoint_excludes_frozen_backbone_and_encoder_hyperparameter(
    clap_encoder: ClapAudioEncoder,
) -> None:
    """Lightning checkpoints retain trainable state without serializing CLAP twice.

    :param clap_encoder: Small frozen CLAP encoder under test.
    """
    module = _flow_module(clap_encoder)
    checkpoint: dict[str, object] = {
        "state_dict": {key: value.clone() for key, value in module.state_dict().items()},
        "hyper_parameters": {"encoder": module.encoder, "num_params": 4},
    }

    module.on_save_checkpoint(checkpoint)

    state = checkpoint["state_dict"]
    hyperparameters = checkpoint["hyper_parameters"]
    assert isinstance(state, dict)
    assert isinstance(hyperparameters, dict)
    assert not any(key.startswith("encoder.backbone.") for key in state)
    assert "encoder.head.projection.weight" in state
    assert "vector_field.input.weight" in state
    assert "encoder" not in hyperparameters
    assert "encoder" not in module.hparams


def test_checkpoint_load_restores_current_backbone_for_strict_state_loading(
    clap_encoder: ClapAudioEncoder,
) -> None:
    """Filtered checkpoints strictly restore trainable state onto a resolved backbone.

    :param clap_encoder: Small frozen CLAP encoder under test.
    """
    module = _flow_module(clap_encoder)
    checkpoint: dict[str, object] = {
        "state_dict": {key: value.clone() for key, value in module.state_dict().items()}
    }
    encoder = cast(PretrainedConditioningEncoder, module.encoder)
    head = cast(VectorProjection, encoder.head)
    saved_head = head.projection.weight.detach().clone()
    module.on_save_checkpoint(checkpoint)
    original_backbone = encoder.backbone.mel_filters.detach().clone()
    with torch.no_grad():
        head.projection.weight.add_(1.0)
        encoder.backbone.mel_filters.add_(1.0)
    current_backbone = encoder.backbone.mel_filters.detach().clone()

    module.on_load_checkpoint(checkpoint)
    state = checkpoint["state_dict"]
    assert isinstance(state, dict)
    module.load_state_dict(state, strict=True)

    assert torch.equal(head.projection.weight, saved_head)
    assert torch.equal(encoder.backbone.mel_filters, current_backbone)
    encoder.backbone.mel_filters.copy_(original_backbone)


def test_checkpoint_load_with_missing_trainable_key_remains_strict(
    clap_encoder: ClapAudioEncoder,
) -> None:
    """Backbone restoration cannot mask a missing projection or field weight.

    :param clap_encoder: Small frozen CLAP encoder under test.
    """
    module = _flow_module(clap_encoder)
    checkpoint: dict[str, object] = {
        "state_dict": {key: value.clone() for key, value in module.state_dict().items()}
    }
    module.on_save_checkpoint(checkpoint)
    state = checkpoint["state_dict"]
    assert isinstance(state, dict)
    del state["encoder.head.projection.weight"]

    module.on_load_checkpoint(checkpoint)

    with pytest.raises(RuntimeError, match="encoder.head.projection.weight"):
        module.load_state_dict(state, strict=True)


def test_clap_features_clamp_finite_values_before_and_after_resampling(
    clap_checkpoint: str,
) -> None:
    """Finite overshoot follows the extractor path with both boundary clamps applied.

    :param clap_checkpoint: Self-contained tiny CLAP checkpoint.
    """
    source_sample_rate = 44_100
    audio = 2.0 * torch.sin(torch.arange(source_sample_rate, dtype=torch.float32) * 0.01)
    encoder = ClapAudioEncoder.from_random_config(
        sample_rate=source_sample_rate,
        checkpoint=clap_checkpoint,
        backbone_config=_TINY_CLAP_CONFIG,
    )
    resampled = audio_fn.resample(audio.clamp(-1.0, 1.0), source_sample_rate, _SAMPLE_RATE)
    expected = ClapFeatureExtractor.from_pretrained(clap_checkpoint)(
        [resampled.clamp(-1.0, 1.0).numpy()],
        sampling_rate=_SAMPLE_RATE,
        return_tensors="pt",
    )["input_features"]

    actual = encoder.features(audio.unsqueeze(0))

    assert torch.allclose(actual, expected, atol=1e-5, rtol=0.0)


def test_clap_finite_value_clamp_preserves_gradient_for_saturated_samples(
    clap_encoder: ClapAudioEncoder,
) -> None:
    """The clamp changes forward values without trapping an overshooting waveform.

    :param clap_encoder: Small frozen CLAP encoder under test.
    """
    audio = (2.0 * torch.sin(torch.arange(4_800, dtype=torch.float32) * 0.01)).unsqueeze(0)
    audio.requires_grad_()

    (gradient,) = torch.autograd.grad(clap_encoder.features(audio).square().mean(), audio)

    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient).item() > 0


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
def test_clap_finite_value_clamp_preserves_non_finite_propagation(
    clap_encoder: ClapAudioEncoder,
    non_finite: float,
) -> None:
    """NaN and infinity remain observable instead of being hidden by clipping.

    :param clap_encoder: Small frozen CLAP encoder under test.
    :param non_finite: Non-finite sample under test.
    """
    audio = torch.zeros(1, 4_800)
    audio[0, 0] = non_finite

    features = clap_encoder.features(audio)

    assert not torch.isfinite(features).all()


def test_clap_features_with_empty_audio_raises(clap_encoder: ClapAudioEncoder) -> None:
    """An empty waveform fails at the input boundary instead of dividing by zero.

    :param clap_encoder: Small frozen CLAP encoder under test.
    """
    with pytest.raises(ValueError, match="empty"):
        clap_encoder.features(torch.empty(1, 0))
