"""Contract tests for ``vae.param_loss``'s per-parameter encoded-span dispatch.

Drives the real loss over real registered ParamSpecs — the onehot branch selects cross-entropy and
the continuous branch MSE, so a mis-sliced span would score the wrong columns against the wrong
objective rather than raising.
"""

from __future__ import annotations

import pytest
import torch

from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.models.components.vae import param_loss


@pytest.mark.parametrize("param_spec", ["surge_4", "surge_simple", "surge_xt", "obxf"])
def test_param_loss_over_a_perfect_reconstruction_is_minimal(param_spec: str) -> None:
    """Reconstructing the input exactly cannot score worse than a corrupted guess.

    :param param_spec: Registered ParamSpec name under test.
    """
    width = param_specs[param_spec].encoded_width
    x = torch.rand(4, width)

    perfect = param_loss(x.clone(), x, param_spec)
    corrupted = param_loss(1.0 - x, x, param_spec)

    assert perfect < corrupted


def test_param_loss_rejects_a_row_wider_than_the_spec() -> None:
    """A row whose width disagrees with the spec fails rather than scoring a prefix."""
    x = torch.rand(2, param_specs["surge_4"].encoded_width + 3)

    with pytest.raises((AssertionError, ValueError)):
        param_loss(x.clone(), x, "surge_4")


def test_param_loss_scores_every_encoded_column() -> None:
    """Corrupting only the final column changes the loss, so no column is skipped."""
    spec = param_specs["surge_4"]
    x = torch.rand(4, spec.encoded_width)
    x_hat = x.clone()
    x_hat[:, -1] = 1.0 - x_hat[:, -1]

    assert param_loss(x_hat, x, "surge_4") != param_loss(x.clone(), x, "surge_4")
