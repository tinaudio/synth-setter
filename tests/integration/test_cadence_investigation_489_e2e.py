"""End-to-end smoke run of the #489 cadence investigation, no mocks.

Drives the real orchestrator entry point with the ``local`` launcher at
``smoke`` scale: it generates the copy-source dataset with the **real Surge XT
plugin**, derives its R2 URI, and runs the ``copy_repro`` probe whose cells
replay that source. Everything is real — the plugin renders, ``rclone`` uploads
to a unique throwaway R2 prefix, and the inline finalize + oracle eval run in
the generate subprocess (so a non-zero exit fails the test).

Asserts on real R2 state: the source shard exists, the copy probe's output shard
exists, and the copy shard's ``param_array`` equals the source's — proving the
derived URI was piped through and the source patches were replayed verbatim,
which is the whole point of folding the six sweep files into one script.

Gated on ``requires_vst`` (real plugin) and ``integration_r2`` (real R2); both
auto-skip via ``conftest`` when the resources are absent. The unique prefix is
purged on teardown even when the body raises.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import h5py
import numpy as np
import pytest

from synth_setter.data.vst.shapes import PARAM_ARRAY_FIELD
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.spec_io import join_uri
from synth_setter.tools import cadence_investigation_489 as inv

pytestmark = [
    pytest.mark.integration_r2,
    pytest.mark.r2,
    pytest.mark.requires_vst,
    pytest.mark.slow,
]

# The smoke copy-source partitions into one shard per split; shard 0 is the
# first train shard, present under both the source and every copy run root.
_FIRST_SHARD = "shard-000000.h5"


def _param_array(shard_uri: str) -> np.ndarray:
    """Download an R2 HDF5 shard and return its ``param_array`` dataset.

    :param shard_uri: ``r2://`` URI of the shard object.
    :returns: The shard's ``param_array`` as an in-memory array.
    """
    with r2_io.downloaded_to_tempfile(shard_uri) as local:
        with h5py.File(local, "r") as handle:
            params = handle[PARAM_ARRAY_FIELD]
            assert isinstance(params, h5py.Dataset)
            return params[:]


def test_smoke_investigation_local_replays_source_params_into_copy_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local smoke run generates the source and replays its params into a copy cell.

    :param monkeypatch: Sets ``WANDB_MODE=offline`` and the worktree ``src`` on
        ``PYTHONPATH`` so the generate subprocesses run offline against this code.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone not on PATH or `rclone lsd r2:` failed)")

    prefix_root = (
        f"test-runs/test_smoke_investigation_local_replays_source_params/{uuid.uuid4().hex[:12]}"
    )
    worktree_src = Path(__file__).resolve().parents[2] / "src"
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("PYTHONPATH", f"{worktree_src}:{os.environ.get('PYTHONPATH', '')}")

    try:
        inv.main(
            [
                "--launcher",
                "local",
                "--scale",
                "smoke",
                "--prefix-root",
                prefix_root,
                "--only",
                "copy_repro",
                "--count",
                "1",
            ]
        )

        source_root = inv.reference_copy_uri(prefix_root=prefix_root)
        source_shard = join_uri(source_root, _FIRST_SHARD)
        assert r2_io.object_size(source_shard), f"source shard absent: {source_shard}"

        copy_root = f"r2://{inv.BUCKET}/{prefix_root}/copy-paired-repro-surge-xt/paired-repro-t1"
        copy_shard = join_uri(copy_root, _FIRST_SHARD)
        assert r2_io.object_size(copy_shard), f"copy shard absent: {copy_shard}"

        assert np.array_equal(_param_array(source_shard), _param_array(copy_shard)), (
            "copy probe did not replay the source params verbatim — the derived "
            "copy_dataset_root_uri was not piped through"
        )
    finally:
        r2_io.purge_prefix(inv.BUCKET, f"{prefix_root}/")
