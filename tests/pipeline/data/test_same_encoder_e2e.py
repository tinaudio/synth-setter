"""Compare the production SAME loader with legacy stable-audio-tools references."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download

from synth_setter.pipeline.data.add_embeddings import (
    SAME_EMBEDDING_DIM,
    SAME_SAMPLE_RATE,
    load_same_audio_encoder,
)
from tests.helpers.same_reference import (
    SAME_HF_CHECKPOINTS,
    SAME_REFERENCE_DIR,
    SAME_REFERENCE_RANDOM_SEED,
    SAME_REFERENCE_ROWS,
    same_reference_audio,
)

pytestmark = [pytest.mark.slow, pytest.mark.network, pytest.mark.same_e2e]

@pytest.mark.parametrize("model_name", SAME_HF_CHECKPOINTS)
def test_same_sa3_loader_matches_legacy_reference(model_name: str) -> None:
    """SA3 reproduces the legacy runtime's finite float32 latents exactly.

    The compressed references were generated with stable-audio-tools 0.0.20, torch 2.12.0+cpu, and
    the immutable HF revision embedded in each archive.

    :param model_name: Registry-style SAME model name.
    """
    repo_id, revision = SAME_HF_CHECKPOINTS[model_name]
    checkpoint_dir = snapshot_download(repo_id, revision=revision)
    encode = load_same_audio_encoder(checkpoint_dir, device="cpu")
    torch.manual_seed(SAME_REFERENCE_RANDOM_SEED)

    actual = encode(same_reference_audio(SAME_SAMPLE_RATE))

    with np.load(
        SAME_REFERENCE_DIR / f"{model_name}_legacy_reference.npz", allow_pickle=False
    ) as archive:
        reference = archive["latents"]
        assert archive["hf_repo"].item() == repo_id
        assert archive["hf_revision"].item() == revision
        assert archive["reference_runtime"].item() == "stable-audio-tools==0.0.20"
        assert archive["torch_version"].item() == "2.12.0+cpu"
        assert archive["random_seed"].item() == SAME_REFERENCE_RANDOM_SEED

    assert actual.shape == reference.shape
    assert actual.shape[:2] == (SAME_REFERENCE_ROWS, SAME_EMBEDDING_DIM)
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    assert actual.std() > 0.0
    assert not np.array_equal(actual[0], actual[1])
    np.testing.assert_array_equal(actual, reference)
