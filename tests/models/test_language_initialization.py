"""Parameter transforms require an explicit, checkpointable initialization boundary."""

from pathlib import Path

import pytest
import torch

from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.models.components.language_projection import LanguageParameterProjection


def test_forward_uninitialized_missing_path_reports_initialization_not_io(tmp_path: Path) -> None:
    """Forward rejects uninitialized state without opening the configured absent artifact.

    :param tmp_path: Directory without an embedding artifact.
    """
    projection = LanguageParameterProjection(
        16, "surge_4", "surge_4", embedding_path=str(tmp_path / "absent.npz")
    )
    with pytest.raises(ValueError, match="initialize_embeddings"):
        projection.param_to_token(torch.zeros(2, param_specs["surge_4"].encoded_width))
