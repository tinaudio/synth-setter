"""Identity tests pinning the deprecated ``surge_*`` module shims and ``Surge*`` aliases.

Archived W&B run configs and external job scripts resolve the old ``_target_``
paths, so each ``surge_*`` module must stay importable (a re-export shim) and each
``Surge*`` alias must stay bound to the renamed ``VST*`` class. The Flow-VAE module
pulls the optional ``nflows`` dependency at import — undeclared in this project —
so its shim and alias are pinned at the AST level instead. See #1664.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import synth_setter.models

# ``Surge*`` symbols are imported from the deprecated ``surge_*`` shim paths so a
# broken shim fails collection; the ``VST*`` symbols come from the renamed modules.
from synth_setter.data.surge_datamodule import SurgeDataModule, SurgeXTDataset
from synth_setter.data.vst_datamodule import VSTDataModule, VSTDataset
from synth_setter.models.surge_fake_oracle_module import SurgeFakeOracleModule
from synth_setter.models.surge_ff_module import SurgeFeedForwardModule
from synth_setter.models.surge_flow_matching_module import SurgeFlowMatchingModule
from synth_setter.models.vst_fake_oracle_module import VSTFakeOracleModule
from synth_setter.models.vst_ff_module import VSTFeedForwardModule
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule


@pytest.mark.parametrize(
    ("alias", "renamed"),
    [
        (SurgeFeedForwardModule, VSTFeedForwardModule),
        (SurgeFlowMatchingModule, VSTFlowMatchingModule),
        (SurgeFakeOracleModule, VSTFakeOracleModule),
        (SurgeDataModule, VSTDataModule),
        (SurgeXTDataset, VSTDataset),
    ],
)
def test_deprecated_alias_is_renamed_class(alias: type, renamed: type) -> None:
    """Each ``Surge*`` alias resolves to its renamed ``VST*`` class by identity.

    :param alias: Deprecated ``Surge*`` symbol an old ``_target_`` resolves, imported
        through the ``surge_*`` shim module.
    :param renamed: The ``VST*`` class the alias must be bound to.
    """
    assert alias is renamed


def _flowvae_module_ast(filename: str) -> ast.Module:
    # Parse the source instead of importing it: the module pulls the undeclared optional nflows dep.
    source = (Path(synth_setter.models.__file__).parent / filename).read_text()
    return ast.parse(source)


def test_flowvae_renamed_class_defined_in_module_source() -> None:
    """Pin the renamed ``VSTFlowVAEModule`` class definition via AST.

    Identity import needs the optional ``nflows`` dep, so a typo'd class name would
    otherwise pass the suite and fail only at launch.
    """
    tree = _flowvae_module_ast("vst_flowvae_module.py")
    assert any(
        isinstance(node, ast.ClassDef) and node.name == "VSTFlowVAEModule"
        for node in ast.walk(tree)
    ), "no `class VSTFlowVAEModule` definition found"


def test_flowvae_deprecated_alias_assigned_in_module_source() -> None:
    """``vst_flowvae_module`` binds ``SurgeFlowVAEModule`` to ``VSTFlowVAEModule``.

    Importing the module needs the undeclared optional ``nflows`` dependency, so
    the alias assignment is pinned in the module AST rather than by identity.
    """
    for node in ast.walk(_flowvae_module_ast("vst_flowvae_module.py")):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "SurgeFlowVAEModule" for t in node.targets)
            and isinstance(node.value, ast.Name)
        ):
            assert node.value.id == "VSTFlowVAEModule"
            return
    pytest.fail("no module-level `SurgeFlowVAEModule = <Name>` assignment found")


def test_flowvae_shim_reexports_deprecated_alias() -> None:
    """The ``surge_flowvae_module`` shim re-exports ``SurgeFlowVAEModule``.

    The shim keeps the old ``_target_`` module path resolving; its import needs the
    optional ``nflows`` dependency, so the re-export is pinned at the AST level.
    """
    for node in ast.walk(_flowvae_module_ast("surge_flowvae_module.py")):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "synth_setter.models.vst_flowvae_module"
            and any(name.name == "SurgeFlowVAEModule" for name in node.names)
        ):
            return
    pytest.fail("shim does not re-export `SurgeFlowVAEModule` from vst_flowvae_module")
