"""Real finalize publishes field embeddings consumable independently of the text model."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from huggingface_hub import get_token

from synth_setter.pipeline import r2_io, spec_io
from synth_setter.pipeline.data.param_language import load_param_language
from synth_setter.pipeline.schemas.spec import DatasetSpec
from tests.integration.test_finalize_dataset_r2 import staged_lance_spec as staged_lance_spec

pytestmark = [pytest.mark.integration_r2, pytest.mark.r2, pytest.mark.slow]


@pytest.mark.timeout(240)
def test_finalize_language_real_r2_preserves_field_embeddings(
    staged_lance_spec: DatasetSpec, tmp_path: Path
) -> None:
    """Ensure finalize publishes normalized field embeddings to R2.

    :param staged_lance_spec: Dataset specification with staged Lance fragments.
    :param tmp_path: Temporary directory for finalization artifacts.
    """
    if get_token() is None:
        pytest.skip("requires accepted embedding-model license and HF_TOKEN authentication")

    payload = staged_lance_spec.model_dump(mode="json")
    payload["param_language_dimension"] = 128
    spec = DatasetSpec.model_validate_json(json.dumps(payload))
    spec_io.upload_spec(spec)
    subprocess.run(  # noqa: S603 — args are test-controlled literals
        [
            str(Path(sys.executable).parent / "synth-setter-finalize-dataset"),
            f"dataset_root_uri=r2://{spec.r2.bucket}/{spec.r2.prefix}",
            "logger=[]",
            f"paths.output_dir={tmp_path}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    uri = f"r2://{spec.r2.bucket}/{spec.r2.prefix}param_language.npz"
    with r2_io.downloaded_to_tempfile(uri) as path:
        embeddings, metadata = load_param_language(path, "surge_simple", "surge_simple")
    assert embeddings.shape == (len(metadata.descriptions), 128)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-6)
    assert not np.allclose(embeddings[0], embeddings[1])
    assert r2_io.object_size(spec.r2.dataset_complete_marker_uri()) is not None
