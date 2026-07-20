"""Regression tests for compile-enabled VST module stage transitions."""

import logging
from functools import partial
from typing import cast

import pytest
import torch

from synth_setter.models.compiled_checkpoint_module import CompiledCheckpointModule
from synth_setter.models.vst_fake_oracle_module import FakeOracleNet, VSTFakeOracleModule
from synth_setter.models.vst_ff_module import VSTFeedForwardModule
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.models.vst_flowvae_module import VSTFlowVAEModule


class _NestedNet(torch.nn.Module):
    """Tiny network with a nested compilation boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.block: torch.nn.Module = torch.nn.Linear(1, 1)


class _CompileCompatibleFixture(CompiledCheckpointModule):
    """Minimal module exposing the shared checkpoint hook."""

    def __init__(self) -> None:
        super().__init__()
        self.net: torch.nn.Module = _NestedNet()


class _LegitimateOrigModNet(torch.nn.Module):
    """Network whose real child shares the compile wrapper's reserved name."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self._orig_mod = torch.nn.Linear(1, 1, bias=False)


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


def test_feed_forward_compiled_checkpoint_loads_into_uncompiled_module(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An eval-stage module consumes weights saved through ``torch.compile``.

    :param caplog: Captures the one-line normalization summary.
    """
    trained = _feed_forward_module()
    trained.setup("fit")
    checkpoint = {"state_dict": trained.state_dict()}
    evaluated = _feed_forward_module()

    with caplog.at_level(logging.INFO):
        evaluated.on_load_checkpoint(checkpoint)
    evaluated.load_state_dict(checkpoint["state_dict"])

    inputs = torch.tensor([[2.0]])
    assert torch.equal(evaluated.net(inputs), trained.net(inputs))
    normalization_logs = [
        record.getMessage()
        for record in caplog.records
        if record.name == "synth_setter.models.compiled_checkpoint_module"
    ]
    assert normalization_logs == ["Normalized 2 compiled checkpoint keys"]


def test_feed_forward_uncompiled_checkpoint_loads_into_compiled_module() -> None:
    """A compiled resume-stage module consumes weights saved without compilation."""
    trained = _feed_forward_module()
    checkpoint = {"state_dict": trained.state_dict()}
    resumed = _feed_forward_module()
    resumed.setup("fit")

    resumed.on_load_checkpoint(checkpoint)
    resumed.load_state_dict(checkpoint["state_dict"])

    inputs = torch.tensor([[2.0]])
    assert torch.equal(resumed.net(inputs), trained.net(inputs))


def test_checkpoint_load_wrapper_moved_between_nested_modules_succeeds() -> None:
    """Canonical keys load when compile wraps a different module boundary."""
    trained = _CompileCompatibleFixture()
    trained_net = cast(_NestedNet, trained.net)
    trained_net.block = cast(torch.nn.Module, torch.compile(trained_net.block))
    checkpoint = {"state_dict": trained.state_dict()}
    resumed = _CompileCompatibleFixture()
    resumed.net = cast(torch.nn.Module, torch.compile(cast(_NestedNet, resumed.net)))

    resumed.on_load_checkpoint(checkpoint)
    resumed.load_state_dict(checkpoint["state_dict"])

    assert torch.equal(
        resumed.state_dict()["net._orig_mod.block.weight"],
        trained.state_dict()["net.block._orig_mod.weight"],
    )


def test_checkpoint_load_preserves_legitimate_orig_mod_child() -> None:
    """Compile wrapper keys remain distinct from a real ``_orig_mod`` child."""
    trained = _CompileCompatibleFixture()
    trained.net = cast(torch.nn.Module, torch.compile(_LegitimateOrigModNet()))
    trained_state = trained.state_dict()
    expected_weight = trained_state["net._orig_mod.weight"].clone()
    expected_child_weight = trained_state["net._orig_mod._orig_mod.weight"].clone()
    checkpoint = {"state_dict": trained_state}
    evaluated = _CompileCompatibleFixture()
    evaluated.net = _LegitimateOrigModNet()

    evaluated.on_load_checkpoint(checkpoint)
    evaluated.load_state_dict(checkpoint["state_dict"])

    evaluated_state = evaluated.state_dict()
    assert torch.equal(evaluated_state["net.weight"], expected_weight)
    assert torch.equal(evaluated_state["net._orig_mod.weight"], expected_child_weight)


def test_compiled_checkpoint_metadata_normalizes_to_uncompiled_module() -> None:
    """Underlying-module metadata replaces compile-wrapper metadata on eval."""
    trained = _feed_forward_module()
    trained.setup("fit")
    state_dict = trained.state_dict()
    metadata = getattr(state_dict, "_metadata")
    expected_net_metadata = metadata["net._orig_mod"].copy()
    evaluated = _feed_forward_module()

    evaluated.on_load_checkpoint({"state_dict": state_dict})

    normalized_metadata = getattr(state_dict, "_metadata")
    assert normalized_metadata["net"] == expected_net_metadata
    assert "net._orig_mod" not in normalized_metadata


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


def test_flow_matching_compiled_checkpoint_loads_into_uncompiled_module() -> None:
    """Flow eval consumes checkpoints with both compiled component prefixes."""
    trained = _flow_matching_module()
    expected_state = trained.state_dict()
    trained.setup("fit")
    checkpoint = {"state_dict": trained.state_dict()}
    evaluated = _flow_matching_module()

    evaluated.on_load_checkpoint(checkpoint)
    evaluated.load_state_dict(checkpoint["state_dict"])

    evaluated_state = evaluated.state_dict()
    assert torch.equal(evaluated_state["encoder.weight"], expected_state["encoder.weight"])
    assert torch.equal(
        evaluated_state["vector_field.weight"],
        expected_state["vector_field.weight"],
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
