"""Compile-stage regression for the Flow-VAE module — needs the optional ``nflows`` dep."""

from functools import partial

import pytest
import torch

pytest.importorskip("nflows", reason="vst_flowvae_module pulls the undeclared optional nflows dep")

from synth_setter.models.vst_flowvae_module import VSTFlowVAEModule  # noqa: E402


def _flowvae_module() -> VSTFlowVAEModule:
    """Build a compile-enabled Flow-VAE module with a tiny real network.

    :returns: Module suitable for setup-stage tests.
    """
    return VSTFlowVAEModule(
        net=torch.nn.Linear(1, 1),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        compile=True,
    )


def test_flowvae_setup_fit_then_test_compiles_net_once() -> None:
    """The test stage preserves the network wrapper created during fit setup."""
    module = _flowvae_module()
    original_net = module.net

    module.setup("fit")
    compiled_net = module.net
    module.setup("test")

    assert compiled_net is not original_net
    assert module.net is compiled_net
