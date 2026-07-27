"""Regression tests for compile-enabled VST module stage transitions."""

from functools import partial

import pytest
import torch

from synth_setter.models.ksin_flow_matching_module import KSinFlowMatchingModule
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


def _assert_compiled_in_place_with_clean_keys(
    module: torch.nn.Module,
    net: torch.nn.Module,
) -> None:
    """Assert fit setup compiled a child in place without renaming state keys.

    :param module: LightningModule whose ``setup("fit")`` already ran.
    :param net: Child expected to remain the same object after compilation.
    """
    assert net._compiled_call_impl is not None  # only signal that in-place compile took effect
    assert all("_orig_mod" not in key for key in module.state_dict())


def test_feed_forward_setup_fit_then_test_compiles_net_in_place() -> None:
    """Fit setup compiles the network without replacing it or renaming keys."""
    module = _feed_forward_module()
    original_net = module.net

    module.setup("fit")
    module.setup("test")

    assert module.net is original_net
    _assert_compiled_in_place_with_clean_keys(module, module.net)


def test_feed_forward_compiled_state_dict_loads_strict_into_uncompiled_module() -> None:
    """An eval-stage module strictly consumes weights saved by compiled training."""
    trained = _feed_forward_module()
    trained.setup("fit")
    evaluated = _feed_forward_module()

    evaluated.load_state_dict(trained.state_dict())

    inputs = torch.tensor([[2.0]])
    assert torch.equal(evaluated.net(inputs), trained.net(inputs))


def test_feed_forward_uncompiled_state_dict_loads_strict_into_compiled_module() -> None:
    """A compiled resume-stage module strictly consumes uncompiled weights."""
    trained = _feed_forward_module()
    resumed = _feed_forward_module()
    resumed.setup("fit")

    resumed.load_state_dict(trained.state_dict())

    inputs = torch.tensor([[2.0]])
    assert torch.equal(resumed.net(inputs), trained.net(inputs))


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


def test_flow_matching_setup_fit_then_test_compiles_components_in_place() -> None:
    """Fit setup compiles both components without replacing them or renaming keys."""
    module = _flow_matching_module()
    original_encoder = module.encoder
    original_vector_field = module.vector_field

    module.setup("fit")
    module.setup("test")

    assert module.encoder is original_encoder
    assert module.vector_field is original_vector_field
    _assert_compiled_in_place_with_clean_keys(module, module.encoder)
    _assert_compiled_in_place_with_clean_keys(module, module.vector_field)


def test_flow_matching_compiled_state_dict_loads_strict_into_uncompiled_module() -> None:
    """Flow eval strictly consumes weights saved with both components compiled."""
    trained = _flow_matching_module()
    trained.setup("fit")
    evaluated = _flow_matching_module()

    evaluated.load_state_dict(trained.state_dict())

    evaluated_state = evaluated.state_dict()
    trained_state = trained.state_dict()
    assert torch.equal(evaluated_state["encoder.weight"], trained_state["encoder.weight"])
    assert torch.equal(
        evaluated_state["vector_field.weight"],
        trained_state["vector_field.weight"],
    )


def _ksin_flow_matching_module() -> KSinFlowMatchingModule:
    """Build a compile-enabled KSin flow module with tiny real component networks.

    :returns: Module suitable for setup-stage tests.
    """
    return KSinFlowMatchingModule(
        encoder=torch.nn.Linear(1, 1),
        vector_field=torch.nn.Linear(1, 1),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        compile=True,
    )


def test_ksin_flow_matching_setup_fit_then_test_compiles_components_in_place() -> None:
    """Fit setup compiles both components without replacing them or renaming keys."""
    module = _ksin_flow_matching_module()
    original_encoder = module.encoder
    original_vector_field = module.vector_field

    module.setup("fit")
    module.setup("test")

    assert module.encoder is original_encoder
    assert module.vector_field is original_vector_field
    _assert_compiled_in_place_with_clean_keys(module, module.encoder)
    _assert_compiled_in_place_with_clean_keys(module, module.vector_field)


def test_fake_oracle_setup_fit_then_test_compiles_net_in_place() -> None:
    """Fit setup compiles the network without replacing it or renaming keys."""
    module = _fake_oracle_module()
    original_net = module.net

    module.setup("fit")
    module.setup("test")

    assert module.net is original_net
    _assert_compiled_in_place_with_clean_keys(module, module.net)


def test_flowvae_setup_fit_then_test_compiles_net_in_place() -> None:
    """Fit setup compiles the network without replacing it or renaming keys."""
    module = _flowvae_module()
    original_net = module.net

    module.setup("fit")
    module.setup("test")

    assert module.net is original_net
    _assert_compiled_in_place_with_clean_keys(module, module.net)
