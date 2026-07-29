"""Behavioral tests for generic embedding encoders and model routing."""

from collections.abc import Callable
from functools import partial

import pytest
import torch

from synth_setter.conditioning import ConditioningMode, EmbeddingConditioningSpec
from synth_setter.data.vst import param_specs
from synth_setter.data.vst.param_spec import ContinuousParameter, decode_model_output
from synth_setter.models.components.embed_pool import EmbeddingPool
from synth_setter.models.components.vector_projection import VectorProjection
from synth_setter.models.vst_ff_module import VSTFeedForwardModule
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.pipeline.data.tinymu import TINYMU_FRONTEND

_ModelFactory = Callable[[], VSTFeedForwardModule | VSTFlowMatchingModule]
_BatchFactory = Callable[[], dict[str, torch.Tensor]]

# Steps needed for the tiny modules below to memorize one batch on CPU.
_OVERFIT_STEPS = 300


def _flow_module(
    conditioning: ConditioningMode | EmbeddingConditioningSpec,
) -> VSTFlowMatchingModule:
    """Build a tiny module for conditioning-key selection tests.

    :param conditioning: Legacy literal or embedding spec under test.
    :returns: Flow module with inert child networks.
    """
    return VSTFlowMatchingModule(
        encoder=torch.nn.Identity(),
        vector_field=torch.nn.Identity(),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=1,
        conditioning=conditioning,
    )


class _TinyVectorField(torch.nn.Module):
    """Minimal differentiable field for exercising flow training."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(7, 2)

    def apply_dropout(self, conditioning: torch.Tensor, dropout_rate: float) -> torch.Tensor:
        """Return conditioning unchanged for deterministic training.

        :param conditioning: Encoded conditioning vectors.
        :param dropout_rate: Unused classifier-free guidance dropout probability.
        :returns: The unchanged vectors.
        """
        return conditioning

    def forward(
        self, params: torch.Tensor, time: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        """Predict a field from parameters, time, and conditioning.

        :param params: Noisy parameter vectors.
        :param time: Sampled flow times.
        :param conditioning: Encoded conditioning vectors.
        :returns: Predicted parameter-space field.
        """
        return self.projection(torch.cat((params, time, conditioning), dim=1))

    def penalty(self) -> torch.Tensor:
        """Return a differentiable zero regularization penalty.

        :returns: Scalar zero connected to the field parameters.
        """
        return self.projection.weight.sum() * 0


def _ff_embedding_module() -> VSTFeedForwardModule:
    """Build a feed-forward module over a cached sequence embedding.

    :returns: Feed-forward module configured for generic conditioning.
    """
    return VSTFeedForwardModule(
        net=torch.nn.Identity(),
        encoder=EmbeddingPool(embed_dim=4, d_model=2, num_heads=1, max_seq_len=3),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        conditioning=EmbeddingConditioningSpec(column="cached", input_shape=(4, 3)),
    )


def _flow_embedding_module() -> VSTFlowMatchingModule:
    """Build a flow module over a cached sequence embedding.

    :returns: Flow module configured for generic conditioning.
    """
    return VSTFlowMatchingModule(
        encoder=EmbeddingPool(embed_dim=4, d_model=4, num_heads=1, max_seq_len=3),
        vector_field=_TinyVectorField(),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=2,
        conditioning=EmbeddingConditioningSpec(column="cached", input_shape=(4, 3)),
    )


def _ff_embedding_batch() -> dict[str, torch.Tensor]:
    """Build a non-mel feed-forward training batch.

    :returns: Batch containing cached conditioning and target parameters.
    """
    return {
        "conditioning": torch.randn(2, 4, 3),
        "params": torch.randn(2, 2),
    }


def _tinymu_flow_module() -> VSTFlowMatchingModule:
    """Build a tiny flow module over production-shaped TinyMU conditioning.

    :returns: Flow module using TinyMU's embedding width and sequence length.
    """
    return VSTFlowMatchingModule(
        encoder=EmbeddingPool(
            embed_dim=TINYMU_FRONTEND.embedding_dim,
            d_model=4,
            num_heads=1,
            max_seq_len=25,
        ),
        vector_field=_TinyVectorField(),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=2,
        conditioning=EmbeddingConditioningSpec(
            column="tinymu", input_shape=(TINYMU_FRONTEND.embedding_dim, 25)
        ),
    )


def _tinymu_flow_batch() -> dict[str, torch.Tensor]:
    """Build one production-shaped TinyMU conditioning batch.

    :returns: Batch containing TinyMU conditioning, targets, and flow noise.
    """
    return {
        "conditioning": torch.randn(2, TINYMU_FRONTEND.embedding_dim, 25),
        "noise": torch.randn(2, 2),
        "params": torch.randn(2, 2),
    }


def _flow_embedding_batch() -> dict[str, torch.Tensor]:
    """Build a non-mel flow training batch.

    :returns: Batch containing cached conditioning, targets, and flow noise.
    """
    return {
        "conditioning": torch.randn(2, 4, 3),
        "noise": torch.randn(2, 2),
        "params": torch.randn(2, 2),
    }


_cached_embedding_arms = pytest.mark.parametrize(
    ("module_factory", "batch_factory"),
    [
        (_flow_embedding_module, _flow_embedding_batch),
        (_ff_embedding_module, _ff_embedding_batch),
        (_tinymu_flow_module, _tinymu_flow_batch),
    ],
    ids=["flow", "feed_forward", "tinymu_flow"],
)


def test_ff_cached_conditioning_without_encoder_raises() -> None:
    """Cached feed-forward conditioning requires an explicit production encoder."""
    with pytest.raises(ValueError, match="cached conditioning requires an encoder"):
        VSTFeedForwardModule(
            net=torch.nn.Identity(),
            optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
            scheduler=None,  # pyright: ignore[reportArgumentType]
            conditioning=EmbeddingConditioningSpec(column="cached", input_shape=(4, 3)),
        )


def test_ff_cached_encoder_wrong_output_width_raises() -> None:
    """A cached encoder incompatible with parameter targets fails before MSE broadcasting."""
    module = VSTFeedForwardModule(
        net=torch.nn.Identity(),
        encoder=EmbeddingPool(embed_dim=4, d_model=3, num_heads=1, max_seq_len=3),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        conditioning=EmbeddingConditioningSpec(column="cached", input_shape=(4, 3)),
    )

    with pytest.raises(ValueError, match=r"encoder output shape .*3.*target shape .*2"):
        module.model_step(_ff_embedding_batch())


def test_ff_default_mel_conditioning_uses_legacy_network() -> None:
    """Default feed-forward configuration retains the mel-to-network contract."""
    module = VSTFeedForwardModule(
        net=torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(6, 2)),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
    )
    mel_spec = torch.randn(2, 1, 2, 3)

    loss, predictions, _, conditioning = module.model_step(
        {"mel_spec": mel_spec, "params": torch.randn(2, 2)}
    )

    assert conditioning is mel_spec
    assert predictions.shape == (2, 2)
    assert torch.isfinite(loss)


@_cached_embedding_arms
def test_vst_module_training_step_cached_embedding_returns_finite_loss(
    module_factory: _ModelFactory,
    batch_factory: _BatchFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both inversion paths train directly from a cached non-mel tensor.

    :param module_factory: Flow or feed-forward module factory under test.
    :param batch_factory: Matching cached-conditioning batch factory.
    :param monkeypatch: Pytest fixture used to detach Lightning logging from a Trainer.
    """
    module = module_factory()
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)

    loss = module.training_step(batch_factory(), batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


@_cached_embedding_arms
def test_vst_module_cached_embedding_backward_reaches_every_trainable_parameter(
    module_factory: _ModelFactory,
    batch_factory: _BatchFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disconnected encoder arm fails here rather than reporting a finite loss.

    :param module_factory: Flow or feed-forward module factory under test.
    :param batch_factory: Matching cached-conditioning batch factory.
    :param monkeypatch: Pytest fixture used to detach Lightning logging from a Trainer.
    """
    module = module_factory()
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)

    module.training_step(batch_factory(), batch_idx=0).backward()

    gradients = {
        name: parameter.grad
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    assert gradients
    detached = [name for name, gradient in gradients.items() if gradient is None]
    assert not detached, f"parameters without gradients: {detached}"
    unmoved = [
        name
        for name, gradient in gradients.items()
        if gradient is not None and gradient.abs().sum().item() == 0.0
    ]
    assert not unmoved, f"parameters with all-zero gradients: {unmoved}"


@_cached_embedding_arms
def test_vst_module_training_step_zeroed_cached_embedding_changes_loss(
    module_factory: _ModelFactory,
    batch_factory: _BatchFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cached tensor drives the loss, so a bias-only predictor cannot pass.

    Both losses are computed under the same seed, so the flow model's sampled time and noise are
    identical across the two calls and only the conditioning differs.

    :param module_factory: Flow or feed-forward module factory under test.
    :param batch_factory: Matching cached-conditioning batch factory.
    :param monkeypatch: Pytest fixture used to detach Lightning logging from a Trainer.
    """
    module = module_factory()
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
    batch = batch_factory()
    zeroed = {**batch, "conditioning": torch.zeros_like(batch["conditioning"])}

    torch.manual_seed(0)
    conditioned_loss = module.training_step(batch, batch_idx=0)
    torch.manual_seed(0)
    zeroed_loss = module.training_step(zeroed, batch_idx=0)

    assert not torch.isclose(conditioned_loss, zeroed_loss)


@_cached_embedding_arms
def test_vst_module_cached_embedding_single_batch_overfits_to_near_zero_loss(
    module_factory: _ModelFactory,
    batch_factory: _BatchFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both cached paths can learn one batch, not merely emit a finite loss.

    A finite first-step loss survives an architecture or optimizer that cannot learn the mapping at
    all; memorizing a single batch is the weakest check that rules that out.

    :param module_factory: Flow or feed-forward module factory under test.
    :param batch_factory: Matching cached-conditioning batch factory.
    :param monkeypatch: Pytest fixture used to detach Lightning logging from a Trainer.
    """
    torch.manual_seed(0)
    module = module_factory()
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
    batch = batch_factory()
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-2)

    loss = module.training_step(batch, batch_idx=0)
    initial_loss = loss.item()
    for _ in range(_OVERFIT_STEPS):
        optimizer.zero_grad()
        loss = module.training_step(batch, batch_idx=0)
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.01
    assert loss.item() < initial_loss


def test_ff_cached_embedding_predict_step_returns_encoder_predictions() -> None:
    """Feed-forward prediction reads the cached tensor, with no mel key in the batch."""
    module = _ff_embedding_module()
    batch = _ff_embedding_batch()

    predictions, returned_batch = module.predict_step(batch, batch_idx=0)

    assert predictions.shape == (2, 2)
    assert returned_batch is batch


@pytest.mark.parametrize("conditioning_scale", [1.0, 1e3], ids=["typical", "extreme"])
def test_ff_cached_embedding_predictions_decode_inside_param_spec_domain(
    conditioning_scale: float,
) -> None:
    """Unbounded cached predictions still decode to renderable parameter values.

    The models emit unbounded values; :func:`decode_model_output` owns the ``[-1, 1]``
    rescale and clip, so the domain guarantee is asserted where a renderer consumes it.

    :param conditioning_scale: Multiplier driving typical and far-out-of-range encoder outputs.
    """
    spec = param_specs["surge_4"]
    torch.manual_seed(0)
    module = VSTFeedForwardModule(
        net=torch.nn.Identity(),
        encoder=EmbeddingPool(embed_dim=4, d_model=len(spec), num_heads=1, max_seq_len=3),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        conditioning=EmbeddingConditioningSpec(column="cached", input_shape=(4, 3)),
    )
    bounds = {
        parameter.name: (parameter.min, parameter.max)
        for parameter in spec.synth_params
        if isinstance(parameter, ContinuousParameter)
    }
    assert bounds

    predictions, _ = module.predict_step(
        {"conditioning": torch.randn(2, 4, 3) * conditioning_scale}, batch_idx=0
    )

    for row in predictions.detach().numpy():
        decoded, _ = decode_model_output(row, spec)
        out_of_domain = {
            name: value
            for name, value in decoded.items()
            if name in bounds and not bounds[name][0] <= value <= bounds[name][1]
        }
        assert not out_of_domain, f"decoded outside the spec domain: {out_of_domain}"


def test_model_embedding_spec_reads_generic_conditioning_key() -> None:
    """Spec-driven models consume the canonical embedding tensor."""
    module = _flow_module(EmbeddingConditioningSpec(column="clap", input_shape=(5,)))
    expected = torch.randn(2, 5)

    actual = module._get_conditioning_from_batch(  # noqa: SLF001
        {"conditioning": expected, "mel_spec": torch.randn(2, 1)}
    )

    assert actual is expected


def test_model_legacy_m2l_hparams_stay_string_while_routing_generic() -> None:
    """Old checkpoint hparams remain m2l while selecting the canonical tensor."""
    module = _flow_module("m2l")
    expected = torch.randn(2, 128, 42)

    actual = module._get_conditioning_from_batch(  # noqa: SLF001
        {"conditioning": expected, "m2l": torch.randn_like(expected)}
    )

    assert module.hparams["conditioning"] == "m2l"
    assert actual is expected


def test_vector_projection_maps_fixed_vectors_to_output_width() -> None:
    """CLAP-style vectors retain their batch axis and receive the configured width."""
    encoder = VectorProjection(input_dim=7, d_model=11)

    output = encoder(torch.randn(3, 7))

    assert output.shape == (3, 11)


def test_vector_projection_wrong_input_width_raises() -> None:
    """A configured vector width mismatch fails with the shape in the message."""
    encoder = VectorProjection(input_dim=7, d_model=11)

    with pytest.raises(ValueError, match=r"expected .*7.*got .*8"):
        encoder(torch.randn(3, 8))


def test_embedding_pool_seq_len_configurable() -> None:
    """A fixed sequence longer than the legacy default pools when configured."""
    encoder = EmbeddingPool(
        embed_dim=8,
        d_model=12,
        num_heads=3,
        max_seq_len=64,
    )

    assert encoder(torch.randn(2, 8, 64)).shape == (2, 12)


def test_embedding_pool_default_42_unchanged() -> None:
    """Omitting max_seq_len preserves the legacy 42-position contract."""
    encoder = EmbeddingPool(embed_dim=8, d_model=12, num_heads=3)

    assert encoder(torch.randn(2, 8, 42)).shape == (2, 12)
    with pytest.raises(ValueError, match=r"sequence length 43 exceeds max_seq_len 42"):
        encoder(torch.randn(2, 8, 43))
