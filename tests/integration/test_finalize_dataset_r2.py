"""End-to-end finalize-against-real-R2 tests.

Auto-skips when R2 is unreachable (rclone missing, no creds, network down)
via ``r2_io.is_r2_reachable``. Stages a single tiny wds shard under a
unique R2 prefix, runs ``finalize_wds`` + the ``dataset.complete`` upload
the entrypoint emits last, then asserts both artifacts land at the
canonical URIs the consumer reads. The prefix is purged on teardown
regardless of pass/fail.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from synth_setter.cli import finalize_dataset
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.schemas.spec import DatasetSpec
from tests.helpers.finalize_shards import write_minimal_wds_shard

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
def staged_wds_spec() -> Iterator[DatasetSpec]:
    """Yield a 1-shard wds ``DatasetSpec`` with its shard pre-uploaded to R2.

    Pins ``r2.prefix`` to a per-run unique value so the prefix is safe to
    ``rclone purge`` on teardown without touching neighbours. The shard is
    materialized locally then ``rclone copyto``'d to ``spec.r2.shard_uri(shard)``
    so the test exercises the real download path during ``finalize_wds``.

    :yields DatasetSpec: A frozen spec whose train split is one 4-sample wds shard
        already present on R2 at ``spec.r2.shard_uri(spec.shards[0])``.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone not on PATH or rclone lsd r2: failed)")
    r2_io.ensure_r2_env_loaded()

    prefix = _unique_test_prefix_suffix()
    bucket = "intermediate-data"
    spec_kwargs: dict[str, Any] = {
        "task_name": "finalize-it-test",
        "output_format": "wds",
        "train_val_test_sizes": [4, 0, 0],
        "base_seed": 42,
        "r2": {"bucket": bucket, "prefix": prefix},
        "render": {
            "plugin_path": "/fake/Plugin.vst3",
            "preset_path": "presets/surge-base.vstpreset",
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
        write_minimal_wds_shard(local)
        r2_io.upload_to_uri(local, spec.r2.shard_uri(spec.shards[0]))
    try:
        yield spec
    finally:
        # Best-effort cleanup; a non-zero purge exit leaves test artifacts
        # behind but never masks a real test failure.
        subprocess.run(  # noqa: S603 — args are literal strings
            ["rclone", "purge", f"r2:{bucket}/{prefix}"],  # noqa: S607
            capture_output=True,
            check=False,
        )


def test_finalize_wds_uploads_stats_and_marker_to_real_r2(
    staged_wds_spec: DatasetSpec,
) -> None:
    """``finalize_wds`` + marker upload land ``stats.npz`` and ``dataset.complete`` on R2.

    Exercises the production code path end-to-end against the configured
    ``r2:`` remote: shard download → stats computation → stats upload →
    marker upload. Replaces the prior plan-doc manual verification step.

    :param staged_wds_spec: Fixture-provided spec whose train shard is already on R2.
    """
    spec = staged_wds_spec
    with tempfile.TemporaryDirectory() as raw_work_dir:
        work_dir = Path(raw_work_dir)
        finalize_dataset.finalize_wds(spec, work_dir)
        marker_local = work_dir / "dataset.complete"
        marker_local.touch()
        r2_io.upload(marker_local, spec.r2.dataset_complete_marker_uri())

    assert r2_io.object_size(spec.r2.stats_uri()) is not None, (
        f"expected stats.npz at {spec.r2.stats_uri()} after finalize"
    )
    # Marker is a trust anchor, not a payload — 0-byte is the canonical shape.
    marker_size = r2_io.object_size(spec.r2.dataset_complete_marker_uri())
    assert marker_size == 0, (
        f"expected zero-byte marker at {spec.r2.dataset_complete_marker_uri()}; "
        f"got size={marker_size}"
    )
    # stats.npz must carry the keys the VSTDataset reader pulls
    # (vst_datamodule.py:62-64) — pull it back and validate the schema.
    with tempfile.TemporaryDirectory() as raw_verify_dir:
        verify_dir = Path(raw_verify_dir)
        local_stats = verify_dir / "stats.npz"
        r2_io.download_to_path(spec.r2.stats_uri(), local_stats)
        with np.load(local_stats) as stats:
            assert set(stats.files) == {"mean", "std"}, (
                f"stats.npz keys are {set(stats.files)}, expected {{'mean', 'std'}}"
            )
            assert stats["mean"].dtype.kind == "f"
            assert stats["std"].dtype.kind == "f"
