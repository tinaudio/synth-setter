"""End-to-end native-R2-streaming round-trip for the Lance dataset path.

Stages shards under a unique R2 prefix, runs streaming ``finalize_lance``, then
streams the splits back via ``LanceVSTDataModule``. Auto-skips when R2 is
unreachable; the prefix is purged on teardown regardless of outcome.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from lance.file import LanceFileReader

from synth_setter.cli import finalize_dataset
from synth_setter.data.lance_datamodule import LanceVSTDataModule
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.schemas.spec import DatasetSpec
from tests.helpers.finalize_shards import write_minimal_lance_shard

pytestmark = [pytest.mark.integration_r2, pytest.mark.r2, pytest.mark.slow]


def _unique_test_prefix_suffix() -> str:
    """Build a ``ci-finalize/<run_id>/<run_attempt>/<uuid>/`` suffix for ``R2Location.prefix``.

    Shares the ``ci-finalize/`` root with the wds finalize integration test so a
    bulk ``rclone purge r2:<bucket>/ci-finalize/`` reclaims stale artifacts from
    either lane.

    :returns: Trailing-slash-terminated R2 prefix string.
    """
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "0")
    nonce = uuid.uuid4().hex[:8]
    return f"ci-finalize/{run_id}/{run_attempt}/{nonce}/"


@pytest.fixture()
def staged_lance_spec() -> Iterator[DatasetSpec]:
    """Yield a 2-shard lance ``DatasetSpec`` with both shards pre-uploaded to R2.

    Pins ``r2.prefix`` to a per-run unique value so the prefix is safe to
    ``rclone purge`` on teardown without touching neighbours. Each shard is
    materialized locally then ``rclone copyto``'d to ``spec.r2.shard_uri(shard)``
    so finalize must stream them back natively.

    :yields DatasetSpec: A frozen lance spec whose train+val shards are already
        on R2 at ``spec.r2.shard_uri(...)``.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone not on PATH or rclone lsd r2: failed)")
    r2_io.ensure_r2_env_loaded()

    prefix = _unique_test_prefix_suffix()
    spec_kwargs: dict[str, Any] = {
        "task_name": "lance-stream-it",
        "output_format": "lance",
        "train_val_test_sizes": [4, 4, 4],
        "base_seed": 42,
        "r2": {"bucket": "intermediate-data", "prefix": prefix},
        # sample_rate=100 keeps the mel front end at its minimum hop so shards stay tiny.
        "render": {
            "plugin_path": "/fake/Plugin.vst3",
            "preset_path": "presets/surge-base.vstpreset",
            "param_spec_name": "surge_simple",
            "renderer_version": "1.0.0-test",
            "sample_rate": 100,
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 1.0,
            "min_loudness": -55.0,
            "samples_per_render_batch": 4,
            "samples_per_shard": 4,
            "gui_toggle_cadence": "never",
        },
    }
    spec = DatasetSpec(**spec_kwargs)  # type: ignore[arg-type]

    with tempfile.TemporaryDirectory() as raw_local:
        for shard in spec.shards:
            local = Path(raw_local) / shard.filename
            write_minimal_lance_shard(local, spec)
            r2_io.upload_to_uri(local, spec.r2.shard_uri(shard))
    try:
        yield spec
    finally:
        subprocess.run(  # noqa: S603 — args are literal strings
            ["rclone", "purge", f"r2:{spec.r2.bucket}/{prefix}"],  # noqa: S607
            capture_output=True,
            check=False,
        )


def test_finalize_lance_streams_to_r2_then_datamodule_streams_back(
    staged_lance_spec: DatasetSpec, tmp_path: Path
) -> None:
    """Streaming finalize writes splits to R2; streaming datamodule reads a batch back.

    Drives the full native path against the configured ``r2:`` remote: shards
    are read over the S3 API, the split files and ``stats.npz`` are written to
    R2, then ``LanceVSTDataModule(stream_from_r2=True)`` opens ``train.lance``
    from its ``s3://`` URI and decodes one batch.

    :param staged_lance_spec: Fixture-provided spec whose shards are already on R2.
    :param tmp_path: Local cache root for the streamed ``stats.npz``.
    """
    spec = staged_lance_spec
    with tempfile.TemporaryDirectory() as raw_work_dir:
        finalize_dataset.finalize_lance(spec, Path(raw_work_dir))

    for split in ("train", "val", "test"):
        assert r2_io.object_size(spec.r2.split_lance_uri(split)) is not None
    assert r2_io.object_size(spec.r2.stats_uri()) is not None

    # The split must be a well-formed Lance file, not merely a non-empty object:
    # read it back natively and pin its row count + schema.
    train_reader = LanceFileReader(
        r2_io.to_s3_uri(spec.r2.split_lance_uri("train")),
        storage_options=r2_io.r2_storage_options(),
    )
    train_meta = train_reader.metadata()
    assert train_meta.num_rows == spec.train_val_test_sizes[0]
    assert {field.name for field in train_meta.schema} == {
        "audio",
        "mel_spec",
        "param_array",
    }

    module = LanceVSTDataModule(
        dataset_root=str(tmp_path / "cache"),
        download_dataset_root_uri=r2_io.shard_uri(spec.r2.bucket, spec.r2.prefix, ""),
        stream_from_r2=True,
        batch_size=2,
        ot=False,
        conditioning="mel",
        param_spec_name=spec.render.param_spec_name,
    )
    module.prepare_data()
    module.setup()
    try:
        batch = module.train_dataset[0]
    finally:
        module.teardown()

    assert batch["params"] is not None
    assert batch["params"].shape == (2, spec.num_params)
    assert batch["mel_spec"] is not None
    assert batch["mel_spec"].shape[0] == 2
    # Normalization ran against the streamed stats.npz: no NaN/Inf leaked through.
    assert torch.isfinite(batch["mel_spec"]).all()
