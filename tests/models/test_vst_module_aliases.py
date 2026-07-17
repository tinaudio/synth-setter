"""Identity tests pinning the deprecated ``surge_*`` module shims and ``Surge*`` aliases.

Archived W&B run configs and external job scripts resolve the old ``_target_``
paths, so each ``surge_*`` module must stay importable (a re-export shim) and each
``Surge*`` alias must stay bound to a concrete runnable class — the renamed
``VST*`` model classes, and the Lance-backed data classes now that Lance is the
only storage format.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# ``Surge*`` symbols are imported from the deprecated ``surge_*`` shim paths so a
# broken shim fails collection; the concrete symbols come from the renamed modules.
from synth_setter.data.lance_datamodule import LanceVSTDataModule, LanceVSTDataset
from synth_setter.data.surge_datamodule import SurgeDataModule, SurgeXTDataset
from synth_setter.data.vst_datamodule import SurgeDataModule as VSTPathSurgeDataModule
from synth_setter.data.vst_datamodule import SurgeXTDataset as VSTPathSurgeXTDataset
from synth_setter.models.surge_fake_oracle_module import SurgeFakeOracleModule
from synth_setter.models.surge_ff_module import SurgeFeedForwardModule
from synth_setter.models.surge_flow_matching_module import SurgeFlowMatchingModule
from synth_setter.models.surge_flowvae_module import SurgeFlowVAEModule
from synth_setter.models.vst_fake_oracle_module import VSTFakeOracleModule
from synth_setter.models.vst_ff_module import VSTFeedForwardModule
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.models.vst_flowvae_module import VSTFlowVAEModule


def test_vst_flow_matching_import_does_not_initialize_data_vst_package() -> None:
    """The model's shared conditioning type must not load the VST runtime package."""
    script = (
        "import sys\n"
        "import synth_setter.models.vst_flow_matching_module  # noqa: F401\n"
        "assert 'synth_setter.data.vst' not in sys.modules\n"
    )
    result = subprocess.run(  # noqa: S603 — sys.executable + literal script
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("alias", "renamed"),
    [
        (SurgeFeedForwardModule, VSTFeedForwardModule),
        (SurgeFlowMatchingModule, VSTFlowMatchingModule),
        (SurgeFakeOracleModule, VSTFakeOracleModule),
        (SurgeFlowVAEModule, VSTFlowVAEModule),
        (SurgeDataModule, LanceVSTDataModule),
        (SurgeXTDataset, LanceVSTDataset),
        (VSTPathSurgeDataModule, LanceVSTDataModule),
        (VSTPathSurgeXTDataset, LanceVSTDataset),
    ],
)
def test_deprecated_alias_is_renamed_class(alias: type, renamed: type) -> None:
    """Each ``Surge*`` alias resolves to its concrete renamed class by identity.

    :param alias: Deprecated ``Surge*`` symbol an old ``_target_`` resolves, imported
        through the ``surge_*`` shim module.
    :param renamed: The concrete class the alias must be bound to.
    """
    assert alias is renamed
