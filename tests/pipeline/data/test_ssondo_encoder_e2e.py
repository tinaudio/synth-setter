"""Run the public embedding CLI with the real pinned S-SONDO checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import lance
import numpy as np
import pytest

from synth_setter.data.vst.shapes import SSONDO_FIELD
from synth_setter.pipeline.data.add_embeddings import main
from synth_setter.pipeline.data.ssondo import SSONDO_EMBEDDING_DIM
from synth_setter.workspace import operator_workspace
from tests.helpers.finalize_shards import build_lance_smoke_spec, write_minimal_lance_shard

pytestmark = [pytest.mark.slow, pytest.mark.network]

_REFERENCE_FLOAT16_220_HZ_PREFIX = np.array(
    [
        -0.1108615,
        -0.02561227,
        0.06278306,
        -0.15679714,
        0.27326956,
        -0.0541441,
        0.05115473,
        -0.03197528,
        0.10415208,
        -0.05268817,
        0.04172202,
        -0.06808273,
        0.11262967,
        0.00293486,
        -0.04455963,
        0.00404362,
    ],
    dtype=np.float32,
)


def test_ssondo_hydra_main_real_checkpoint_writes_distinct_finite_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real CLI persists usable vectors from two distinct source rows.

    :param tmp_path: Scratch space for the Lance shard and Hydra output.
    :param monkeypatch: Fixture supplying user-visible CLI arguments.
    """
    uri = tmp_path / "ssondo.lance"
    spec = build_lance_smoke_spec(train_val_test_sizes=(4, 0, 0))
    time = np.arange(3_200, dtype=np.float32) / 8_000
    frequencies = np.array([220.0, 330.0, 440.0, 550.0], dtype=np.float32)
    mono = np.sin(2 * np.pi * frequencies[:, None] * time[None, :])
    audio = np.repeat(mono[:, None, :], 2, axis=1)
    write_minimal_lance_shard(uri, spec, audio=audio)
    monkeypatch.setenv("PROJECT_ROOT", str(operator_workspace()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synth-setter-add-embeddings",
            f"lance_uri={uri}",
            "embeddings=[ssondo]",
            "device=cpu",
            "batch_size=2",
            "build_index=false",
            f"paths.log_dir={tmp_path}",
            f"hydra.run.dir={tmp_path / 'hydra'}",
        ],
    )

    main()

    column = (
        lance.dataset(str(uri))
        .to_table(columns=[SSONDO_FIELD])
        .column(SSONDO_FIELD)
        .combine_chunks()
    )
    vectors = np.stack(column.to_numpy(zero_copy_only=False))
    assert vectors.shape == (4, SSONDO_EMBEDDING_DIM)
    assert vectors.dtype == np.float32
    assert np.isfinite(vectors).all()
    assert np.linalg.norm(vectors, axis=1).min() > 0
    assert not np.array_equal(vectors[0], vectors[1])
    np.testing.assert_allclose(
        vectors[0, : len(_REFERENCE_FLOAT16_220_HZ_PREFIX)],
        _REFERENCE_FLOAT16_220_HZ_PREFIX,
        rtol=1e-3,
        atol=3e-4,
    )
