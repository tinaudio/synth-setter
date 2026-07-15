"""End-to-end finalize-against-real-R2 tests.

Auto-skips when R2 is unreachable (rclone missing, no creds, network down)
via ``r2_io.is_r2_reachable``. Stages a single tiny Lance shard attempt under a
unique R2 prefix, runs ``finalize_lance`` (which commits the staged winner
fragment into the split dataset and writes ``stats.npz`` + ``dataset.json``),
then asserts the artifacts land at the canonical URIs the consumer reads. The
prefix is purged on teardown regardless of pass/fail.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import lance
import numpy as np
import pytest

from synth_setter.cli import finalize_dataset
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.lance_shard import LANCE_DATA_STORAGE_VERSION
from synth_setter.pipeline.schemas.spec import DatasetSpec
from tests.helpers.finalize_shards import write_minimal_lance_shard

pytestmark = [pytest.mark.integration_r2, pytest.mark.r2, pytest.mark.slow]


def _unique_test_prefix_suffix() -> str:
    """Build a ``ci-finalize/<run_id>/<run_attempt>/<uuid>/`` suffix for ``R2Location.prefix``.

    Matches the layout convention of ``test_local_launcher_roundtrip``'s
    ``_unique_r2_prefix`` so concurrent CI runs and local dev runs do not
    collide. Includes the leading ``ci-finalize/`` so a bulk
    ``rclone purge r2:<bucket>/ci-finalize/`` reclaims stale artifacts.

    :returns: Trailing-slash-terminated R2 prefix string.
    """
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "0")
    nonce = uuid.uuid4().hex[:8]
    return f"ci-finalize/{run_id}/{run_attempt}/{nonce}/"


@pytest.fixture()
def staged_lance_spec() -> Iterator[DatasetSpec]:
    """Yield a 1-shard Lance ``DatasetSpec`` with one staged attempt on real R2.

    Mirrors :func:`staged_wds_spec` but for the Lance fragment path (#1776):
    the shard renders locally, then ``stage_lance_shard_attempt`` writes its
    uncommitted fragment into ``train.lance/data/`` on R2 and uploads the
    sidecar + stats + ``.valid`` staging set — the same worker code path
    production runs.

    :yields DatasetSpec: A frozen spec whose train split is one 4-row Lance
        shard staged under ``metadata/workers/shards/shard-000000/``.
    """
    from synth_setter.pipeline.data.lance_staging import stage_lance_shard_attempt

    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone not on PATH or rclone lsd r2: failed)")
    r2_io.ensure_r2_env_loaded()

    prefix = _unique_test_prefix_suffix()
    bucket = "intermediate-data"
    spec_kwargs: dict[str, Any] = {
        "task_name": "finalize-it-test",
        "output_format": "lance",
        "train_val_test_sizes": [4, 0, 0],
        "base_seed": 42,
        "r2": {"bucket": bucket, "prefix": prefix},
        "render": {
            "plugin_path": "/fake/Plugin.vst3",
            "plugin_state_path": "presets/surge-base.vstpreset",
            "param_spec_name": "surge_simple",
            "renderer_version": "1.0.0-test",
            "sample_rate": 44100,
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "samples_per_render_batch": 4,
            "samples_per_shard": 4,
            "gui_toggle_cadence": "never",
        },
    }
    spec = DatasetSpec(**spec_kwargs)  # type: ignore[arg-type]

    with tempfile.TemporaryDirectory() as raw_local:
        local = Path(raw_local) / spec.shards[0].filename
        write_minimal_lance_shard(local, spec)
        stage_lance_shard_attempt(
            spec, spec.shards[0], local, worker_id="it-worker", attempt_uuid=uuid.uuid4().hex[:8]
        )
    try:
        yield spec
    finally:
        subprocess.run(  # noqa: S603 — args are literal strings
            ["rclone", "purge", f"r2:{bucket}/{prefix}"],  # noqa: S607
            capture_output=True,
            check=False,
        )


def test_finalize_lance_commits_staged_fragment_on_real_r2(
    staged_lance_spec: DatasetSpec,
) -> None:
    """``finalize_lance`` commits the staged winner fragment into the R2 split dataset.

    Exercises the fragment path end-to-end over object storage: winner
    selection from the real R2 listing (``LastModified``), structural checks,
    a manifest-only ``Overwrite`` commit of in-place fragment data, Welford
    reduction into ``stats.npz``, and the ``dataset.json`` audit record. The
    split's rows are actually read back (not ``count_rows``, which cannot see
    a dangling fragment) at the pinned on-disk version.

    :param staged_lance_spec: Fixture-provided spec with one staged Lance attempt on R2.
    """
    from synth_setter.data.vst.shapes import (
        MEL_SPEC_FIELD,
        PARAM_ARRAY_FIELD,
        dataset_field_shapes,
    )
    from synth_setter.pipeline.data.lance_shard import iter_lance_column_rows

    spec = staged_lance_spec
    with tempfile.TemporaryDirectory() as raw_work_dir:
        finalize_dataset.finalize_lance(spec, Path(raw_work_dir))

    assert r2_io.object_size(spec.r2.stats_uri()) is not None, (
        f"expected stats.npz at {spec.r2.stats_uri()} after finalize"
    )
    with r2_io.downloaded_to_tempfile(spec.r2.stats_uri()) as stats_path:
        with np.load(stats_path) as stats:
            assert set(stats.files) == {"mean", "std"}
            expected_shape = dataset_field_shapes(spec.render, spec.num_params)[MEL_SPEC_FIELD][1:]
            for key in ("mean", "std"):
                assert stats[key].shape == expected_shape
                assert np.issubdtype(stats[key].dtype, np.floating)
                assert np.isfinite(stats[key]).all()
    assert r2_io.object_size(spec.r2.dataset_card_uri()) is not None, (
        f"expected dataset.json at {spec.r2.dataset_card_uri()} after finalize"
    )
    split_s3, storage_options = r2_io.lance_target(spec.r2.split_lance_uri("train"))
    rows = list(
        iter_lance_column_rows(split_s3, PARAM_ARRAY_FIELD, storage_options=storage_options)
    )
    assert len(rows) == spec.render.samples_per_shard
    dataset = lance.dataset(split_s3, storage_options=storage_options)
    assert dataset.data_storage_version == LANCE_DATA_STORAGE_VERSION
