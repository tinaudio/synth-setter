"""Identity tests pinning the deprecated ``Surge*`` model-class aliases.

Archived W&B run configs and external job scripts resolve the old
``_target_`` paths, so each alias must stay bound to the renamed ``VST*`` class.
The Flow-VAE module pulls the optional ``nflows`` dependency at import, so its
alias is verified only where ``nflows`` is installed (``importorskip``).
"""

from __future__ import annotations

import pytest

from synth_setter.models.surge_fake_oracle_module import (
    SurgeFakeOracleModule,
    VSTFakeOracleModule,
)
from synth_setter.models.surge_ff_module import SurgeFeedForwardModule, VSTFeedForwardModule
from synth_setter.models.surge_flow_matching_module import (
    SurgeFlowMatchingModule,
    VSTFlowMatchingModule,
)


@pytest.mark.parametrize(
    ("alias", "renamed"),
    [
        (SurgeFeedForwardModule, VSTFeedForwardModule),
        (SurgeFlowMatchingModule, VSTFlowMatchingModule),
        (SurgeFakeOracleModule, VSTFakeOracleModule),
    ],
)
def test_deprecated_alias_is_renamed_class(alias: type, renamed: type) -> None:
    """Each ``Surge*`` alias resolves to its renamed ``VST*`` class by identity.

    :param alias: Deprecated ``Surge*`` symbol an old ``_target_`` resolves.
    :param renamed: The ``VST*`` class the alias must be bound to.
    """
    assert alias is renamed


def test_flowvae_deprecated_alias_is_renamed_class() -> None:
    """``SurgeFlowVAEModule`` resolves to ``VSTFlowVAEModule`` (skips without ``nflows``)."""
    pytest.importorskip("nflows")
    from synth_setter.models.surge_flowvae_module import (
        SurgeFlowVAEModule,
        VSTFlowVAEModule,
    )

    assert SurgeFlowVAEModule is VSTFlowVAEModule
