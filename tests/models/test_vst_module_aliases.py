"""Identity tests pinning the deprecated ``Surge*`` model-class aliases.

Archived W&B run configs and external job scripts resolve the old
``_target_`` paths, so each alias must stay bound to the renamed ``VST*`` class.
The Flow-VAE module pulls the optional ``nflows`` dependency at import —
undeclared in this project — so its alias is pinned at the AST level instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import synth_setter.models
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


def test_flowvae_deprecated_alias_assigned_in_module_source() -> None:
    """``surge_flowvae_module`` binds ``SurgeFlowVAEModule`` to ``VSTFlowVAEModule``.

    Importing the module needs the undeclared optional ``nflows`` dependency, so
    the alias assignment is pinned in the module AST rather than by identity.
    """
    source = (Path(synth_setter.models.__file__).parent / "surge_flowvae_module.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "SurgeFlowVAEModule" for t in node.targets)
            and isinstance(node.value, ast.Name)
        ):
            assert node.value.id == "VSTFlowVAEModule"
            return
    pytest.fail("no module-level `SurgeFlowVAEModule = <Name>` assignment found")
