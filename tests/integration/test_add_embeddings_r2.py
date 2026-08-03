"""End-to-end add-embeddings-against-real-R2 test (no mocks).

Exercises ``synth-setter-add-embeddings`` against real R2 Lance datasets. The
encoder tests upload a tiny real-VST shard before running the real models; the
index test uploads schema-compatible vectors and drives the production IVF-PQ
resume path. Each dataset is reopened through Lance and consumed by a nearest
query before its unique R2 prefix is purged.

The encoder tests skip when the VST plugin is absent; all tests skip when R2
credentials are missing or R2 is unreachable at runtime.
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
from typing import Any, cast

import lance
import numpy as np
import pyarrow as pa
import pytest

from synth_setter.cli.generate_dataset import build_generate_args
from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    CLAP_FIELD,
    M2L_FIELD,
    MATPAC_PLUS_FIELD,
    PARAM_ARRAY_FIELD,
    T5GEMMA_FIELD,
)
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.add_embeddings import (
    CLAP_EMBEDDING_DIM,
    MIN_ROWS_FOR_INDEX,
)
from synth_setter.pipeline.data.lance_shard import (
    SHARD_METADATA_SCHEMA_KEY,
    write_lance_dataset,
)
from synth_setter.pipeline.data.t5gemma import (
    T5GEMMA_EMBEDDING_DIM,
    T5GEMMA_MAX_LENGTH,
)
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata
from synth_setter.pipeline.schemas.spec import DatasetSpec, ShardSpec
from synth_setter.resources import as_file, vst_headless_wrapper
from tests._vst import (
    PLUGIN_PATH,
    TEST_PARAM_SPEC_NAME,
    TEST_PRESET_PATH,
    TEST_SYNTH_VERSION,
    VST_SUBPROCESS_TIMEOUT_SECONDS,
)

pytestmark = [pytest.mark.slow, pytest.mark.integration_r2, pytest.mark.r2]

# Keep real-encoder coverage below the IVF-PQ floor; indexing has its own fixture.
_SAMPLES_PER_SHARD = 4
# Short clips keep rendering and inference cheap while remaining valid after CLAP resampling.
_SIGNAL_DURATION_SECONDS = 1.0
_SAMPLE_RATE = 44100
_CHANNELS = 2
_R2_BUCKET = "intermediate-data"
_INDEX_EMBEDDING_DIM = CLAP_EMBEDDING_DIM
_INDEX_M2L_FRAMES = 2
_INDEX_NUM_PARTITIONS = 4
_INDEX_NUM_SUB_VECTORS = 16
_INDEX_RANDOM_SEED = 0
_INDEX_SUBPROCESS_TIMEOUT_SECONDS = 90
_INDEX_TEST_TIMEOUT_SECONDS = 120

# The add_embeddings CLI is the system under test; invoke it as the console
# script the operator runs, against the uploaded ``r2://`` dataset directory.
_ADD_EMBEDDINGS_CMD = "synth-setter-add-embeddings"
# Generous: covers a real VST render plus the first-run checkpoint downloads
# and CPU/GPU forward passes of music2latent + CLAP.
_EMBED_SUBPROCESS_TIMEOUT_SECONDS = 1800
# Account for parameter loading and host flushes beyond the fixed VST startup budget.
_RENDER_SECONDS_PER_SAMPLE = 5


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


def _lance_embed_spec(prefix: str, rows: int = _SAMPLES_PER_SHARD) -> DatasetSpec:
    """Build a 1-shard Lance ``DatasetSpec`` pinned to the real test synth + R2 prefix.

    :param prefix: Unique R2 prefix the shard is rendered + uploaded under.
    :param rows: Train samples in the rendered shard.
    :returns: A frozen Lance spec whose single train shard is renderable by the
        real VST and whose ``r2`` layout is safe to ``purge_prefix`` on teardown.
    """
    spec_kwargs: dict[str, Any] = {
        "task_name": "it-add-embeddings",
        "output_format": "lance",
        "train_val_test_sizes": [rows, 0, 0],
        "base_seed": 42,
        # Constant mel bins over so few samples; mask so the spec stays valid.
        "mask_degenerate_bins": True,
        "r2": {"bucket": _R2_BUCKET, "prefix": prefix},
        "render": {
            "synth": {
                "name": TEST_PARAM_SPEC_NAME,
                "param_spec_name": TEST_PARAM_SPEC_NAME,
                "plugin_path": PLUGIN_PATH,
                "plugin_state_path": TEST_PRESET_PATH,
                "synth_version": TEST_SYNTH_VERSION,
            },
            "sample_rate": _SAMPLE_RATE,
            "channels": _CHANNELS,
            "velocity": 100,
            "signal_duration_seconds": _SIGNAL_DURATION_SECONDS,
            "min_loudness": -55.0,
            # One batch avoids repeated plugin startup in this serial fixture.
            "samples_per_render_batch": rows,
            "samples_per_shard": rows,
            "plugin_reload_cadence": "once",
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
    # Scale the fixed host-startup budget by the requested render work.
    timeout = (
        VST_SUBPROCESS_TIMEOUT_SECONDS + _RENDER_SECONDS_PER_SAMPLE * spec.render.samples_per_shard
    )
    with ExitStack() as stack:
        args: list[str] = []
        if sys.platform == "linux":
            wrapper = stack.enter_context(as_file(vst_headless_wrapper()))
            args.append(str(wrapper))
        args += build_generate_args(spec, shard, work_dir)
        subprocess.run(  # noqa: S603 — args from a validated spec + repo wrapper
            args, check=True, timeout=timeout
        )
    shard_path = work_dir / shard.filename
    assert shard_path.is_dir(), f"renderer wrote no Lance dataset at {shard_path}"
    return shard_path


def _require_r2() -> None:
    """Skip the requesting fixture unless the real R2 remote is usable."""
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone not on PATH or rclone lsd r2: failed)")
    r2_io.ensure_r2_env_loaded()


def _render_and_upload(rows: int) -> Iterator[str]:
    """Render a Lance shard of ``rows`` samples, upload it to a unique R2 prefix, yield its URI.

    Exercises the real generate path end-to-end: VST render → local Lance
    dataset → ``upload_dir`` to R2. The prefix is purged on teardown regardless
    of pass/fail so a failed assertion never leaks artifacts.

    :param rows: Samples in the single rendered shard.
    :yields str: ``r2://bucket/prefix/shard-000000.lance`` of the uploaded dataset.
    """
    prefix = _unique_test_prefix()
    spec = _lance_embed_spec(prefix, rows)
    shard = spec.shards[0]
    shard_uri = spec.r2.shard_uri(shard)
    try:
        with tempfile.TemporaryDirectory() as raw_work_dir:
            local_shard = _render_shard_locally(spec, shard, Path(raw_work_dir))
            r2_io.upload_dir(local_shard, shard_uri)
        assert r2_io.r2_directory_exists(shard_uri), f"upload left nothing at {shard_uri}"
        yield shard_uri
    finally:
        r2_io.purge_prefix(_R2_BUCKET, prefix)


@pytest.fixture()
def remote_lance_dataset_uri() -> Iterator[str]:
    """Yield a tiny (``_SAMPLES_PER_SHARD``-row) uploaded Lance dataset URI.

    :yields str: ``r2://`` URI of the uploaded dataset.
    """
    _require_r2()
    yield from _render_and_upload(_SAMPLES_PER_SHARD)


def _index_embedding_schema(m2l_shape: tuple[int, ...]) -> pa.Schema:
    """Build the schema and source metadata needed by the index-only CLI path.

    :param m2l_shape: Inner shape of the sequence embedding column.
    :returns: Schema for CLAP, M2L, and pooled M2L vectors.
    """
    metadata = ShardMetadata(
        velocity=100,
        signal_duration_seconds=_SIGNAL_DURATION_SECONDS,
        sample_rate=_SAMPLE_RATE,
        channels=_CHANNELS,
        min_loudness=-55.0,
    )
    return pa.schema(
        [
            pa.field(
                CLAP_FIELD,
                pa.list_(pa.float32(), _INDEX_EMBEDDING_DIM),
                nullable=False,
            ),
            pa.field(M2L_FIELD, pa.fixed_shape_tensor(pa.float32(), m2l_shape), nullable=False),
            pa.field(
                f"{M2L_FIELD}_vec",
                pa.list_(pa.float32(), _INDEX_EMBEDDING_DIM),
                nullable=False,
            ),
        ],
        metadata={SHARD_METADATA_SCHEMA_KEY: metadata.model_dump_json().encode()},
    )


def _indexable_embedding_table() -> pa.Table:
    """Build deterministic schema-compatible embedding columns at the index floor.

    :returns: Table carrying CLAP, M2L, and pooled M2L vectors plus shard metadata.
    """
    rng = np.random.default_rng(_INDEX_RANDOM_SEED)
    clap = rng.standard_normal((MIN_ROWS_FOR_INDEX, _INDEX_EMBEDDING_DIM), dtype=np.float32)
    m2l = rng.standard_normal(
        (MIN_ROWS_FOR_INDEX, _INDEX_EMBEDDING_DIM, _INDEX_M2L_FRAMES),
        dtype=np.float32,
    )
    m2l_vectors = m2l.mean(axis=-1, dtype=np.float32)
    schema = _index_embedding_schema(m2l.shape[1:])
    return pa.Table.from_arrays(
        [
            pa.FixedSizeListArray.from_arrays(pa.array(clap.reshape(-1)), _INDEX_EMBEDDING_DIM),
            pa.FixedShapeTensorArray.from_numpy_ndarray(m2l),
            pa.FixedSizeListArray.from_arrays(
                pa.array(m2l_vectors.reshape(-1)), _INDEX_EMBEDDING_DIM
            ),
        ],
        schema=schema,
    )


@pytest.fixture()
def remote_indexed_lance_dataset_uri() -> Iterator[str]:
    """Yield a real-R2 Lance dataset with index-ready embedding columns.

    :yields str: ``r2://`` URI of the uploaded dataset.
    """
    _require_r2()
    prefix = _unique_test_prefix()
    shard_uri = r2_io.shard_uri(_R2_BUCKET, prefix, "shard-000000.lance")
    storage_options = r2_io.r2_storage_options()
    try:
        table = _indexable_embedding_table()
        write_lance_dataset(
            r2_io.to_s3_uri(shard_uri),
            table.schema,
            table.to_batches(),
            storage_options=storage_options,
        )
        assert _open_remote_dataset(shard_uri).count_rows() == MIN_ROWS_FOR_INDEX
        yield shard_uri
    finally:
        r2_io.purge_prefix(_R2_BUCKET, prefix)


def _open_remote_dataset(r2_uri: str) -> lance.LanceDataset:
    """Open a Lance dataset on R2, mirroring ``add_embeddings._open_lance_dataset``.

    :param r2_uri: Canonical ``r2://bucket/key`` dataset directory URI.
    :returns: The credentialed, opened dataset.
    """
    return lance.dataset(r2_io.to_s3_uri(r2_uri), storage_options=r2_io.r2_storage_options())


def _run_add_embeddings(
    r2_uri: str,
    overrides: tuple[str, ...],
    timeout_seconds: int = _EMBED_SUBPROCESS_TIMEOUT_SECONDS,
) -> None:
    """Run the public CLI against R2 and report captured output on failure.

    :param r2_uri: Canonical ``r2://`` Lance dataset URI.
    :param overrides: Additional Hydra overrides for this scenario.
    :param timeout_seconds: Maximum CLI runtime in seconds.
    """
    command = [_ADD_EMBEDDINGS_CMD, f"lance_uri={r2_uri}", *overrides]
    try:
        result = subprocess.run(  # noqa: S603 — literal command and validated fixture URI
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"{_ADD_EMBEDDINGS_CMD} exceeded {timeout_seconds}s\n"
            f"stdout:\n{exc.stdout or ''}\nstderr:\n{exc.stderr or ''}",
            pytrace=False,
        )
    assert result.returncode == 0, (
        f"{_ADD_EMBEDDINGS_CMD} exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.requires_vst
def test_add_embeddings_matpac_plus_against_real_r2_uses_registry_path(
    remote_lance_dataset_uri: str,
) -> None:
    """The public CLI writes real MATPAC columns through the common Lance target.

    :param remote_lance_dataset_uri: Fixture-provided ``r2://`` Lance dataset URI.
    """
    _run_add_embeddings(
        remote_lance_dataset_uri,
        ("embeddings=[matpac_plus]", "build_index=false"),
    )

    dataset = _open_remote_dataset(remote_lance_dataset_uri)
    assert dataset.count_rows() == _SAMPLES_PER_SHARD
    assert {MATPAC_PLUS_FIELD, f"{MATPAC_PLUS_FIELD}_vec"} <= set(dataset.schema.names)
    values = dataset.to_table(columns=[MATPAC_PLUS_FIELD]).column(MATPAC_PLUS_FIELD)
    assert np.isfinite(values.combine_chunks().to_numpy_ndarray()).all()


@pytest.mark.requires_vst
def test_add_embeddings_cli_against_real_r2_writes_clap_m2l_and_t5gemma(
    remote_lance_dataset_uri: str,
) -> None:
    """``synth-setter-add-embeddings`` on a real R2 Lance dataset writes searchable columns.

    Runs the two production CLIs back to back with no mocks: the fixture renders
    + uploads a tiny Lance shard via the VST renderer, then this test invokes the
    real ``add_embeddings`` CLI with the music2latent, LAION-CLAP, and SA3
    T5Gemma encoders against that ``r2://`` URI. The augmented dataset is reopened
    from R2 and asserted to carry the fixed-size ``clap``, ``m2l``, ``m2l_vec``,
    and ``t5gemma`` columns with finite values and one row per audio row. Source
    columns are preserved and exact ``nearest=`` remains usable; the 4-row shard
    is below the IVF_PQ training floor, so no index is expected.

    :param remote_lance_dataset_uri: Fixture-provided ``r2://`` Lance dataset URI.
    """
    _run_add_embeddings(
        remote_lance_dataset_uri,
        ("embeddings=[clap,m2l,t5gemma]", f"param_spec_name={TEST_PARAM_SPEC_NAME}"),
    )

    dataset = _open_remote_dataset(remote_lance_dataset_uri)
    names = set(dataset.schema.names)
    assert {AUDIO_FIELD, PARAM_ARRAY_FIELD} <= names, (
        f"source columns dropped: schema is {sorted(names)}"
    )
    m2l_vector_field = f"{M2L_FIELD}_vec"
    assert {M2L_FIELD, m2l_vector_field, CLAP_FIELD, T5GEMMA_FIELD} <= names, (
        f"embedding columns absent: schema is {sorted(names)}"
    )

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

    m2l_vector_type = dataset.schema.field(m2l_vector_field).type
    assert pa.types.is_fixed_size_list(m2l_vector_type), (
        f"m2l companion is {m2l_vector_type}, not a fixed-size list"
    )
    assert m2l_vector_type.value_type == pa.float32(), (
        f"m2l companion value type is {m2l_vector_type.value_type}"
    )

    t5gemma_type = dataset.schema.field(T5GEMMA_FIELD).type
    assert isinstance(t5gemma_type, pa.FixedShapeTensorType), (
        f"t5gemma is {t5gemma_type}, not a fixed-shape tensor"
    )
    assert t5gemma_type.value_type == pa.float32(), (
        f"t5gemma value type is {t5gemma_type.value_type}"
    )

    table = dataset.to_table(columns=[CLAP_FIELD, M2L_FIELD, m2l_vector_field, T5GEMMA_FIELD])
    clap = np.stack(table.column(CLAP_FIELD).to_numpy(zero_copy_only=False))
    assert clap.shape == (rows, CLAP_EMBEDDING_DIM), f"clap materialized as {clap.shape}"
    assert np.isfinite(clap).all(), "clap embeddings contain non-finite values"
    m2l = table.column(M2L_FIELD).combine_chunks().to_numpy_ndarray()
    assert len(m2l) == rows, f"m2l has {len(m2l)} rows, expected {rows}"
    assert np.isfinite(m2l).all(), "m2l embeddings contain non-finite values"
    m2l_vectors = np.stack(table.column(m2l_vector_field).to_numpy(zero_copy_only=False))
    assert m2l_vectors.shape == (rows, m2l.shape[1]), (
        f"m2l companion materialized as {m2l_vectors.shape}"
    )
    assert np.isfinite(m2l_vectors).all(), "m2l companion contains non-finite values"
    np.testing.assert_allclose(m2l_vectors, m2l.mean(axis=-1), rtol=1e-5, atol=1e-6)
    t5gemma = table.column(T5GEMMA_FIELD).combine_chunks().to_numpy_ndarray()
    assert t5gemma.shape == (
        rows,
        T5GEMMA_EMBEDDING_DIM,
        T5GEMMA_MAX_LENGTH,
    ), f"t5gemma materialized as {t5gemma.shape}"
    assert np.isfinite(t5gemma).all(), "t5gemma embeddings contain non-finite values"
    assert np.any(t5gemma != 0), "t5gemma embeddings are all zero"
    np.testing.assert_array_equal(t5gemma, np.broadcast_to(t5gemma[0], t5gemma.shape))

    # Below the training floor, every index skips while exact nearest still resolves.
    assert rows < MIN_ROWS_FOR_INDEX
    assert dataset.list_indices() == [], (
        f"unexpected index for a {rows}-row dataset: {dataset.list_indices()}"
    )

    query = clap[0].astype(np.float32)
    neighbours = dataset.to_table(nearest={"column": CLAP_FIELD, "q": query, "k": rows})
    assert neighbours.num_rows >= 1, "nearest query returned no rows"


@pytest.mark.timeout(_INDEX_TEST_TIMEOUT_SECONDS)
def test_add_embeddings_cli_against_real_r2_builds_ivf_pq_index(
    remote_indexed_lance_dataset_uri: str,
) -> None:
    """Train IVF-PQ indexes through the public CLI against a real R2 dataset.

    Uploads schema-compatible embedding columns at ``MIN_ROWS_FOR_INDEX``, runs
    the real ``add_embeddings`` CLI with ``build_index=true``, then reopens the
    remote dataset and asserts IVF_PQ indexes exist on ``clap`` and ``m2l_vec``;
    a CLAP ANN query returns a stored row's own vector as the top hit.

    :param remote_indexed_lance_dataset_uri: Fixture-provided ``r2://`` URI of a
        dataset with enough rows to train the index.
    """
    _run_add_embeddings(
        remote_indexed_lance_dataset_uri,
        (
            "build_index=true",
            f"num_partitions={_INDEX_NUM_PARTITIONS}",
            f"num_sub_vectors={_INDEX_NUM_SUB_VECTORS}",
        ),
        timeout_seconds=_INDEX_SUBPROCESS_TIMEOUT_SECONDS,
    )

    dataset = _open_remote_dataset(remote_indexed_lance_dataset_uri)
    rows = dataset.count_rows()
    assert rows == MIN_ROWS_FOR_INDEX, f"row count changed to {rows}"

    indices = cast("list[dict[str, Any]]", dataset.list_indices())
    assert indices, f"no index built for a {rows}-row dataset"
    expected_index_fields = {(CLAP_FIELD,), (f"{M2L_FIELD}_vec",)}
    assert {tuple(idx["fields"]) for idx in indices} == expected_index_fields, (
        f"expected clap and m2l companion indexes, got {indices}"
    )

    clap_table = dataset.to_table(columns=[CLAP_FIELD])
    clap = np.stack(clap_table.column(CLAP_FIELD).to_numpy(zero_copy_only=False))
    target = rows // 2
    query = clap[target].astype(np.float32)
    # PQ is lossy, so the ANN top hit need not be the exact stored row; assert a
    # near-zero cosine distance, which a self-query yields for any close vector.
    hits = dataset.to_table(nearest={"column": CLAP_FIELD, "q": query, "k": 1})
    assert hits.num_rows == 1, "ANN query returned no rows"
    assert hits.column("_distance")[0].as_py() < 1e-2, (
        f"self-query top hit distance {hits.column('_distance')[0].as_py()} not near zero"
    )
