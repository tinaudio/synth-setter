"""End-to-end add-embeddings-against-real-R2 test (no mocks).

Drives the two production CLIs back to back: the real VST renderer
(``generate_vst_dataset.py``) writes a tiny Lance shard that is uploaded to a
unique R2 prefix, then ``synth-setter-add-embeddings`` runs the real
music2latent + LAION-CLAP encoders against that remote URI. The augmented
dataset is reopened from R2 and its ``m2l`` / ``clap`` columns, indexability,
and ``nearest=`` query path are asserted. The prefix is purged on teardown
regardless of pass/fail.

Auto-skips when the VST plugin is absent (``requires_vst``) or R2 credentials
are missing (``integration_r2``); also skips when R2 is unreachable at runtime.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import lance
import numpy as np
import pyarrow as pa
import pytest

from synth_setter.cli.generate_dataset import build_generate_args
from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    CLAP_FIELD,
    M2L_FIELD,
    PARAM_ARRAY_FIELD,
)
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.add_embeddings import (
    CLAP_EMBEDDING_DIM,
    MIN_ROWS_FOR_INDEX,
)
from synth_setter.pipeline.schemas.spec import DatasetSpec, ShardSpec
from synth_setter.resources import as_file, vst_headless_wrapper
from tests._vst import (
    PLUGIN_PATH,
    TEST_PARAM_SPEC_NAME,
    TEST_PRESET_PATH,
    TEST_RENDERER_VERSION,
    VST_SUBPROCESS_TIMEOUT_SECONDS,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_vst,
    pytest.mark.integration_r2,
    pytest.mark.r2,
]

# Kept tiny so the real encoders stay fast: one 4-row shard. 4 < MIN_ROWS_FOR_INDEX,
# so the IVF_PQ build is skipped and the test asserts the exact ``nearest`` fallback.
_SAMPLES_PER_SHARD = 4
# Short clips keep both the VST render and the CLAP/m2l forward pass cheap;
# CLAP resamples to 48 kHz internally, so 1 s still yields a valid embedding.
_SIGNAL_DURATION_SECONDS = 1.0
_SAMPLE_RATE = 44100
_CHANNELS = 2

# The add_embeddings CLI is the system under test; invoke it as the console
# script the operator runs, against the uploaded ``r2://`` dataset directory.
_ADD_EMBEDDINGS_CMD = "synth-setter-add-embeddings"
# Generous: covers a real VST render plus the first-run checkpoint downloads
# and CPU/GPU forward passes of music2latent + CLAP.
_EMBED_SUBPROCESS_TIMEOUT_SECONDS = 1800


def _unique_test_prefix() -> str:
    """Build a per-run ``ci-add-embeddings/<run_id>/<attempt>/<uuid>/`` R2 prefix.

    Mirrors the layout in :mod:`tests.integration.test_finalize_dataset_r2` so
    concurrent CI runs and local dev runs never collide, and the leading
    ``ci-add-embeddings/`` segment makes a bulk ``rclone purge`` of stale
    artifacts straightforward.

    :returns: Trailing-slash-terminated R2 prefix string.
    """
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "0")
    nonce = uuid.uuid4().hex[:8]
    return f"ci-add-embeddings/{run_id}/{run_attempt}/{nonce}/"


def _lance_embed_spec(prefix: str) -> DatasetSpec:
    """Build a 1-shard Lance ``DatasetSpec`` pinned to the real test synth + R2 prefix.

    :param prefix: Unique R2 prefix the shard is rendered + uploaded under.
    :returns: A frozen Lance spec whose single train shard is renderable by the
        real VST and whose ``r2`` layout is safe to ``purge_prefix`` on teardown.
    """
    spec_kwargs: dict[str, Any] = {
        "task_name": "it-add-embeddings",
        "output_format": "lance",
        "train_val_test_sizes": [_SAMPLES_PER_SHARD, 0, 0],
        "base_seed": 42,
        # Constant mel bins over so few samples; mask so the spec stays valid.
        "mask_degenerate_bins": True,
        "r2": {"bucket": "intermediate-data", "prefix": prefix},
        "render": {
            "plugin_path": PLUGIN_PATH,
            "preset_path": TEST_PRESET_PATH,
            "param_spec_name": TEST_PARAM_SPEC_NAME,
            "renderer_version": TEST_RENDERER_VERSION,
            "sample_rate": _SAMPLE_RATE,
            "channels": _CHANNELS,
            "velocity": 100,
            "signal_duration_seconds": _SIGNAL_DURATION_SECONDS,
            "min_loudness": -55.0,
            "samples_per_render_batch": _SAMPLES_PER_SHARD,
            "samples_per_shard": _SAMPLES_PER_SHARD,
            "gui_toggle_cadence": "never",
        },
    }
    return DatasetSpec(**spec_kwargs)  # type: ignore[arg-type]


def _render_shard_locally(spec: DatasetSpec, shard: ShardSpec, work_dir: Path) -> Path:
    """Render one Lance shard via the real ``generate_vst_dataset.py`` CLI.

    Wraps the renderer in the X11 headless bootstrap (as the production
    dispatcher does on Linux) and shells out with the repo's own
    ``build_generate_args`` so the flag set tracks ``RenderConfig`` exactly.

    :param spec: Spec supplying the render config + shard layout.
    :param shard: The single train shard to render.
    :param work_dir: Local dir the ``.lance`` dataset directory is written into.
    :returns: Path to the produced local Lance dataset directory.
    """
    with ExitStack() as stack:
        args: list[str] = []
        if sys.platform == "linux":
            wrapper = stack.enter_context(as_file(vst_headless_wrapper()))
            args.append(str(wrapper))
        args += build_generate_args(spec, shard, work_dir)
        subprocess.run(  # noqa: S603 — args from a validated spec + repo wrapper
            args, check=True, timeout=VST_SUBPROCESS_TIMEOUT_SECONDS
        )
    shard_path = work_dir / shard.filename
    assert shard_path.is_dir(), f"renderer wrote no Lance dataset at {shard_path}"
    return shard_path


@pytest.fixture()
def remote_lance_dataset_uri() -> Iterator[str]:
    """Render a tiny Lance shard, upload it to a unique R2 prefix, yield its ``r2://`` URI.

    Exercises the real generate path end-to-end: VST render → local Lance
    dataset → ``upload_dir`` to R2. The prefix is purged on teardown regardless
    of pass/fail so a failed assertion never leaks artifacts.

    :yields str: ``r2://bucket/prefix/shard-000000.lance`` of the uploaded dataset.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone not on PATH or rclone lsd r2: failed)")
    r2_io.ensure_r2_env_loaded()

    prefix = _unique_test_prefix()
    spec = _lance_embed_spec(prefix)
    shard = spec.shards[0]
    shard_uri = spec.r2.shard_uri(shard)
    try:
        with tempfile.TemporaryDirectory() as raw_work_dir:
            local_shard = _render_shard_locally(spec, shard, Path(raw_work_dir))
            r2_io.upload_dir(local_shard, shard_uri)
        assert r2_io.r2_directory_exists(shard_uri), f"upload left nothing at {shard_uri}"
        yield shard_uri
    finally:
        # Best-effort teardown; a purge failure must not mask a test assertion.
        r2_io.purge_prefix(spec.r2.bucket, prefix)


def _open_remote_dataset(r2_uri: str) -> lance.LanceDataset:
    """Open a Lance dataset on R2, mirroring ``add_embeddings._open_lance_dataset``.

    :param r2_uri: Canonical ``r2://bucket/key`` dataset directory URI.
    :returns: The credentialed, opened dataset.
    """
    return lance.dataset(r2_io.to_s3_uri(r2_uri), storage_options=r2_io.r2_storage_options())


def test_add_embeddings_cli_against_real_r2_writes_indexable_clap_and_m2l(
    remote_lance_dataset_uri: str,
) -> None:
    """``synth-setter-add-embeddings`` on a real R2 Lance dataset writes searchable columns.

    Runs the two production CLIs back to back with no mocks: the fixture renders
    + uploads a tiny Lance shard via the VST renderer, then this test invokes the
    real ``add_embeddings`` CLI (real music2latent + LAION-CLAP encoders) against
    that ``r2://`` URI. The augmented dataset is reopened from R2 and asserted to
    carry a ``FixedSizeList<float32, 512>`` ``clap`` column, a
    ``fixed_shape_tensor<float32, ...>`` ``m2l`` column, finite values, one row
    per audio row, preserved source columns, and a working exact ``nearest=``
    query (the 4-row shard is below the IVF_PQ training floor, so no index is
    expected).

    :param remote_lance_dataset_uri: Fixture-provided ``r2://`` Lance dataset URI.
    """
    result = subprocess.run(  # noqa: S603 — literal cmd + a validated r2:// URI
        [_ADD_EMBEDDINGS_CMD, remote_lance_dataset_uri],
        check=False,
        capture_output=True,
        text=True,
        timeout=_EMBED_SUBPROCESS_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, (
        f"{_ADD_EMBEDDINGS_CMD} exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    dataset = _open_remote_dataset(remote_lance_dataset_uri)
    names = set(dataset.schema.names)
    assert {AUDIO_FIELD, PARAM_ARRAY_FIELD} <= names, (
        f"source columns dropped: schema is {sorted(names)}"
    )
    assert {M2L_FIELD, CLAP_FIELD} <= names, f"embedding columns absent: schema is {sorted(names)}"

    rows = dataset.count_rows()
    assert rows == _SAMPLES_PER_SHARD, f"row count changed to {rows}"

    clap_type = dataset.schema.field(CLAP_FIELD).type
    assert pa.types.is_fixed_size_list(clap_type), f"clap is {clap_type}, not a fixed-size list"
    assert clap_type.value_type == pa.float32(), f"clap value type is {clap_type.value_type}"
    assert clap_type.list_size == CLAP_EMBEDDING_DIM, (
        f"clap width is {clap_type.list_size}, expected {CLAP_EMBEDDING_DIM}"
    )

    m2l_type = dataset.schema.field(M2L_FIELD).type
    assert isinstance(m2l_type, pa.FixedShapeTensorType), (
        f"m2l is {m2l_type}, not a fixed-shape tensor"
    )
    assert m2l_type.value_type == pa.float32(), f"m2l value type is {m2l_type.value_type}"

    table = dataset.to_table(columns=[CLAP_FIELD, M2L_FIELD])
    clap = np.stack(table.column(CLAP_FIELD).to_numpy(zero_copy_only=False))
    assert clap.shape == (rows, CLAP_EMBEDDING_DIM), f"clap materialized as {clap.shape}"
    assert np.isfinite(clap).all(), "clap embeddings contain non-finite values"
    m2l = table.column(M2L_FIELD).combine_chunks().to_numpy_ndarray()
    assert len(m2l) == rows, f"m2l has {len(m2l)} rows, expected {rows}"
    assert np.isfinite(m2l).all(), "m2l embeddings contain non-finite values"

    # 4 rows is below the IVF_PQ training floor, so the CLI skips the index and
    # exact (brute-force) nearest must still resolve. clap is the only column the
    # CLI ever indexes, so an empty index list pins the skip directly.
    assert rows < MIN_ROWS_FOR_INDEX
    assert dataset.list_indices() == [], (
        f"unexpected index for a {rows}-row dataset: {dataset.list_indices()}"
    )

    query = clap[0].astype(np.float32)
    neighbours = dataset.to_table(nearest={"column": CLAP_FIELD, "q": query, "k": rows})
    assert neighbours.num_rows >= 1, "nearest query returned no rows"
