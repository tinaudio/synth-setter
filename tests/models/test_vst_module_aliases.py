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


def _flowvae_module_ast() -> ast.Module:
    # Parse the source instead of importing it: the module pulls the undeclared optional nflows dep.
    source = (Path(synth_setter.models.__file__).parent / "surge_flowvae_module.py").read_text()
    return ast.parse(source)


def test_flowvae_renamed_class_defined_in_module_source() -> None:
    """Pin the renamed ``VSTFlowVAEModule`` class definition via AST.

    Identity import needs the optional ``nflows`` dep, so a typo'd class name would
    otherwise pass the suite and fail only at launch.
    """
    tree = _flowvae_module_ast()
    assert any(
        isinstance(node, ast.ClassDef) and node.name == "VSTFlowVAEModule"
        for node in ast.walk(tree)
    ), "no `class VSTFlowVAEModule` definition found"


def test_flowvae_deprecated_alias_assigned_in_module_source() -> None:
    """``surge_flowvae_module`` binds ``SurgeFlowVAEModule`` to ``VSTFlowVAEModule``.

    Importing the module needs the undeclared optional ``nflows`` dependency, so
    the alias assignment is pinned in the module AST rather than by identity.
    """
    for node in ast.walk(_flowvae_module_ast()):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "SurgeFlowVAEModule" for t in node.targets)
            and isinstance(node.value, ast.Name)
        ):
            assert node.value.id == "VSTFlowVAEModule"
            return
    pytest.fail("no module-level `SurgeFlowVAEModule = <Name>` assignment found")
