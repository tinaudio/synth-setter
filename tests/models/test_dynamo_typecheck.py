"""Behavior tests for the jaxtyping/Dynamo type-check bypass."""

import pytest
import torch
from beartype import beartype
from jaxtyping import Float, TypeCheckError, jaxtyped
from jaxtyping import _config as jaxtyping_config
from torch import Tensor, nn

from synth_setter.models.dynamo_typecheck import install_dynamo_typecheck_bypass


class _Doubler(nn.Module):
    """Typed module standing in for any annotated model the bypass has to leave checked."""

    @jaxtyped(typechecker=beartype)
    def forward(self, audio: Float[Tensor, "batch samples"]) -> Float[Tensor, "batch samples"]:
        """Double the waveform.

        :param audio: Mono waveform batch.
        :returns: The input scaled by two.
        """
        return audio * 2.0


def test_explicit_disable_still_turns_runtime_checking_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator's ``JAXTYPING_DISABLE=1`` must keep silencing checks after installation.

    :param monkeypatch: Restores the global switch once the test returns.
    """
    install_dynamo_typecheck_bypass()
    monkeypatch.setattr(jaxtyping_config.config, "jaxtyping_disable", True)

    assert torch.equal(_Doubler()(torch.ones(2, 3, dtype=torch.int64)), torch.full((2, 3), 2))


def test_repeated_installation_keeps_eager_type_checking_on() -> None:
    """Installing twice must not stack wrappers or silence the eager check."""
    install_dynamo_typecheck_bypass()
    install_dynamo_typecheck_bypass()

    with pytest.raises(TypeCheckError):
        _Doubler()(torch.zeros(2, 3, dtype=torch.int64))
