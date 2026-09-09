"""Run the public SAME Hydra pipeline against legacy reference archives."""

from __future__ import annotations

import sys
from pathlib import Path

import lance
import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download

from synth_setter.pipeline.data.add_embeddings import (
    SAME_EMBEDDING_DIM,
    SAME_SAMPLE_RATE,
    main,
)
from synth_setter.workspace import operator_workspace
from tests.helpers.finalize_shards import build_lance_smoke_spec, write_minimal_lance_shard
from tests.helpers.same_reference import (
    SAME_HF_CHECKPOINTS,
    SAME_REFERENCE_DIR,
    SAME_REFERENCE_RANDOM_SEED,
    SAME_REFERENCE_ROWS,
    same_reference_audio,
)

pytestmark = [pytest.mark.slow, pytest.mark.network, pytest.mark.same_e2e]

_PARITY_ATOL = 3e-4
_PARITY_RTOL = 1e-4


def _write_same_reference_shard(uri: Path) -> None:
    """Write the archived float32 audio input as a production-schema Lance shard.

    :param uri: Destination Lance dataset.
    """
    base_spec = build_lance_smoke_spec()
    render = base_spec.render.model_copy(
        update={
            "audio_dtype": "float32",
            "sample_rate": SAME_SAMPLE_RATE,
            "samples_per_render_batch": SAME_REFERENCE_ROWS,
            "samples_per_shard": SAME_REFERENCE_ROWS,
            "signal_duration_seconds": 1.0,
        }
    )
    spec = build_lance_smoke_spec(
        task_name="same-parity",
        train_val_test_sizes=(SAME_REFERENCE_ROWS, 0, 0),
        render=render,
    )
    write_minimal_lance_shard(
        uri,
        spec,
        audio=same_reference_audio(SAME_SAMPLE_RATE),
    )


@pytest.mark.parametrize(
    ("model_name", "expected_frames"),
    [("same_s", 12), ("same_l", 11)],
)
def test_same_hydra_main_writes_legacy_matching_lance_column(
    model_name: str,
    expected_frames: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real CLI persists finite SA3 latents within measured legacy drift.

    :param model_name: Registry key selecting one SAME checkpoint and column.
    :param expected_frames: Model-specific one-second latent width.
    :param tmp_path: Scratch space for the input shard and Hydra logs.
    :param monkeypatch: Fixture supplying the user-visible process arguments.
    """
    repo_id, revision = SAME_HF_CHECKPOINTS[model_name]
    checkpoint_dir = Path(snapshot_download(repo_id, revision=revision))
    uri = tmp_path / f"{model_name}.lance"
    _write_same_reference_shard(uri)
    monkeypatch.setenv("PROJECT_ROOT", str(operator_workspace()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synth-setter-add-embeddings",
            "logger=[]",
            f"lance_uri={uri}",
            f"embeddings=[{model_name}]",
            f"checkpoints.{model_name}={checkpoint_dir}",
            "device=cpu",
            "batch_size=2",
            "build_index=false",
            f"paths.log_dir={tmp_path}",
            f"hydra.run.dir={tmp_path / 'hydra'}",
        ],
    )
    torch.manual_seed(SAME_REFERENCE_RANDOM_SEED)

    main()

    column = (
        lance.dataset(str(uri))
        .to_table(columns=[model_name])
        .column(model_name)
        .combine_chunks()
    )
    actual = column.to_numpy_ndarray()
    with np.load(
        SAME_REFERENCE_DIR / f"{model_name}_legacy_reference.npz", allow_pickle=False
    ) as archive:
        reference = archive["latents"]
        assert archive["hf_repo"].item() == repo_id
        assert archive["hf_revision"].item() == revision
        assert archive["reference_runtime"].item() == "stable-audio-tools==0.0.20"
        assert archive["torch_version"].item() == "2.12.0+cpu"
        assert archive["platform_system"].item() == "Linux"
        assert archive["platform_machine"].item() == "x86_64"
        assert archive["random_seed"].item() == SAME_REFERENCE_RANDOM_SEED

    assert actual.shape == reference.shape
    assert actual.shape == (SAME_REFERENCE_ROWS, SAME_EMBEDDING_DIM, expected_frames)
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    assert actual.std() > 0.0
    assert not np.array_equal(actual[0], actual[1])
    # Local output is exact; 3e-4 covers the 2.13e-4 cross-runner SAME-S drift
    # measured in PR #2537 while remaining far below the latent scale.
    np.testing.assert_allclose(actual, reference, rtol=_PARITY_RTOL, atol=_PARITY_ATOL)
