"""Regression tests for compile-enabled VST module stage transitions."""

from functools import partial

import pytest
import torch

from synth_setter.models.vst_fake_oracle_module import FakeOracleNet, VSTFakeOracleModule
from synth_setter.models.vst_ff_module import VSTFeedForwardModule
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.models.vst_flowvae_module import VSTFlowVAEModule


def _feed_forward_module() -> VSTFeedForwardModule:
    """Build a compile-enabled feed-forward module with a tiny real network.

    :returns: Module suitable for setup-stage tests.
    """
    return VSTFeedForwardModule(
        net=torch.nn.Linear(1, 1),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        compile=True,
    )


def _flow_matching_module() -> VSTFlowMatchingModule:
    """Build a compile-enabled flow module with tiny real component networks.

    :returns: Module suitable for setup-stage tests.
    """
    return VSTFlowMatchingModule(
        encoder=torch.nn.Linear(1, 1),
        vector_field=torch.nn.Linear(1, 1),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        compile=True,
        num_params=1,
    )


def _fake_oracle_module() -> VSTFakeOracleModule:
    """Build a compile-enabled fake-oracle module with its real stand-in net.

    :returns: Module suitable for setup-stage tests.
    """
    return VSTFakeOracleModule(
        net=FakeOracleNet(d_out=1),
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        scheduler=None,
        compile=True,
    )


def test_feed_forward_setup_fit_then_test_compiles_net_once() -> None:
    """The test stage preserves the network wrapper created during fit setup."""
    module = _feed_forward_module()
    original_net = module.net

    module.setup("fit")
    compiled_net = module.net
    module.setup("test")

    assert compiled_net is not original_net
    assert module.net is compiled_net


def test_flow_matching_constructor_without_num_params_raises_type_error() -> None:
    """Flow models require an explicit target width from configuration."""
    with pytest.raises(TypeError, match="num_params"):
        VSTFlowMatchingModule(  # pyright: ignore[reportCallIssue]
            encoder=torch.nn.Linear(1, 1),
            vector_field=torch.nn.Linear(1, 1),
            optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
            scheduler=None,  # pyright: ignore[reportArgumentType]
        )


def test_flow_matching_constructor_num_params_positional_raises_type_error() -> None:
    """A positional fifth argument must fail to prevent a bogus training width."""
    with pytest.raises(TypeError, match="positional"):
        VSTFlowMatchingModule(
            torch.nn.Linear(1, 1),
            torch.nn.Linear(1, 1),
            partial(torch.optim.Adam, lr=1e-3),
            None,
            1,  # pyright: ignore[reportCallIssue]
        )


def test_flow_matching_setup_fit_then_test_compiles_components_once() -> None:
    """The test stage preserves both component wrappers created during fit setup."""
    module = _flow_matching_module()
    original_encoder = module.encoder
    original_vector_field = module.vector_field

    module.setup("fit")
    compiled_encoder = module.encoder
    compiled_vector_field = module.vector_field
    module.setup("test")

    assert compiled_encoder is not original_encoder
    assert compiled_vector_field is not original_vector_field
    assert module.encoder is compiled_encoder
    assert module.vector_field is compiled_vector_field


def _flowvae_module() -> VSTFlowVAEModule:
    """Build a compile-enabled Flow-VAE module with a tiny real network.

    :returns: Module suitable for setup-stage tests.
    """
    return VSTFlowVAEModule(
        net=torch.nn.Linear(1, 1),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        param_spec="surge_simple",
        compile=True,
    )


def test_fake_oracle_setup_fit_then_test_compiles_net_once() -> None:
    """The test stage preserves the network wrapper created during fit setup."""
    module = _fake_oracle_module()
    original_net = module.net

    module.setup("fit")
    compiled_net = module.net
    module.setup("test")

    assert compiled_net is not original_net
    assert module.net is compiled_net


def test_flowvae_setup_fit_then_test_compiles_net_once() -> None:
    """The test stage preserves the network wrapper created during fit setup."""
    module = _flowvae_module()
    original_net = module.net

    module.setup("fit")
    compiled_net = module.net
    module.setup("test")

    assert compiled_net is not original_net
    assert module.net is compiled_net
