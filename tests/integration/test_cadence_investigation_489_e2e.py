"""End-to-end smoke run of the #489 cadence investigation, no mocks.

Drives the real orchestrator entry point with the ``local`` launcher at
``smoke`` scale: it generates the copy-source dataset with the **real Surge XT
plugin**, derives its R2 URI, and runs the ``copy_repro`` probe whose cells
replay that source. Everything is real — the plugin renders, ``rclone`` uploads
to a unique throwaway R2 prefix, and the inline finalize + oracle eval run in
the generate subprocess (so a non-zero exit fails the test).

Asserts on real R2 state: the source shard exists, the copy probe's output shard
exists, and the copy shard's ``param_array`` equals the source's — proving the
derived URI was piped through and the source patches were replayed verbatim.

Gated on ``requires_vst`` (real plugin) and ``integration_r2`` (real R2); both
auto-skip via ``conftest`` when the resources are absent. The unique prefix is
purged on teardown even when the body raises.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

import h5py
import numpy as np
import pytest

from synth_setter.data.vst.shapes import PARAM_ARRAY_FIELD
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.schemas.prefix import make_r2_prefix
from synth_setter.pipeline.spec_io import join_uri
from synth_setter.tools import cadence_investigation_489 as inv

pytestmark = [
    pytest.mark.integration_r2,
    pytest.mark.r2,
    pytest.mark.requires_vst,
    pytest.mark.slow,
]

# shard-000000 is the first train shard, present under both the source and copy run roots.
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

    copy_probe = next(e for e in inv.build_experiments(inv.SMOKE) if e.name == "copy_repro")

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
                copy_probe.name,
                "--count",
                "1",
            ]
        )

        source_root = inv.reference_copy_uri(prefix_root=prefix_root)
        source_shard = join_uri(source_root, _FIRST_SHARD)
        assert r2_io.object_size(source_shard) is not None, f"source shard absent: {source_shard}"

        # --count 1 runs the first grid cell, so the run_id is the grid's first value.
        first_run_id = str(copy_probe.grid["run_id"][0])
        copy_prefix = make_r2_prefix(copy_probe.task_name, first_run_id, prefix_root=prefix_root)
        copy_shard = join_uri(f"r2://{inv.BUCKET}/{copy_prefix.rstrip('/')}", _FIRST_SHARD)
        assert r2_io.object_size(copy_shard) is not None, f"copy shard absent: {copy_shard}"

        assert np.array_equal(_param_array(source_shard), _param_array(copy_shard)), (
            "copy probe did not replay the source params verbatim — the derived "
            "copy_dataset_root_uri was not piped through"
        )
    finally:
        r2_io.purge_prefix(inv.BUCKET, f"{prefix_root}/")


def _shard_count(prefix_uri: str) -> int:
    """Count ``.h5`` shard objects under an R2 prefix via ``rclone``.

    :param prefix_uri: ``r2://`` prefix to list recursively.
    :returns: The number of ``.h5`` objects found under the prefix.
    """
    rclone_path = prefix_uri.replace("r2://", "r2:", 1)
    # check=True so an auth/network failure raises instead of being misreported
    # as zero shards (empty stdout) by the callers' `>= 1` assertions.
    result = subprocess.run(  # noqa: S603 — args are test-controlled literals
        ["rclone", "lsf", "-R", "--include", "*.h5", rclone_path],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return len([line for line in result.stdout.splitlines() if line.strip()])


@pytest.mark.integration_r2
@pytest.mark.r2
@pytest.mark.requires_vst
@pytest.mark.slow
@pytest.mark.network
def test_full_smoke_investigation_wandb_online_runs_every_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the full investigation at smoke size via a real wandb-online sweep + agent.

    All five experiments and every grid cell run; the test asserts on real R2 state.

    It creates real wandb sweeps/runs and spends real render compute, gated only by
    the standard resource markers — ``requires_vst`` + ``integration_r2`` auto-skip
    without the plugin / R2, and it skips inline without ``WANDB_API_KEY``. Sweeps go
    to the ``WANDB_ENTITY`` / ``WANDB_PROJECT`` the creds can write (the script's
    ``ENTITY`` is a temporary personal pin until #1560 stands up ``tinaudio``).
    Source + experiment outputs land under a unique throwaway R2 prefix purged on
    teardown; the wandb sweeps stay for review in the W&B UI.

    :param monkeypatch: Forces wandb online, points the sweeps at the env's
        entity/project, and puts the worktree ``src`` on the agents' ``PYTHONPATH``.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone not on PATH or `rclone lsd r2:` failed)")
    if not os.environ.get("WANDB_API_KEY"):
        pytest.skip("WANDB_API_KEY required for the online wandb run")

    prefix_root = f"test-runs/test_full_smoke_investigation_wandb/{uuid.uuid4().hex[:12]}"
    worktree_src = Path(__file__).resolve().parents[2] / "src"
    monkeypatch.setenv("PYTHONPATH", f"{worktree_src}:{os.environ.get('PYTHONPATH', '')}")
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(inv, "ENTITY", os.environ.get("WANDB_ENTITY", inv.ENTITY))
    monkeypatch.setattr(inv, "PROJECT", os.environ.get("WANDB_PROJECT", inv.PROJECT))

    experiments = inv.build_experiments(inv.SMOKE)

    try:
        inv.main(["--launcher", "wandb", "--scale", "smoke", "--prefix-root", prefix_root])

        source_root = inv.reference_copy_uri(prefix_root=prefix_root)
        source_shard = join_uri(source_root, _FIRST_SHARD)
        assert r2_io.object_size(source_shard) is not None, f"source shard absent: {source_shard}"

        # Every experiment's agent ran and uploaded at least one shard.
        for exp in experiments:
            task_prefix = f"r2://{inv.BUCKET}/{prefix_root}/{exp.task_name}"
            assert _shard_count(task_prefix) >= 1, (
                f"{exp.name} produced no shards under {task_prefix}"
            )

        # The repro copy probe replayed the source params verbatim (URI piped through).
        repro = next(e for e in experiments if e.name == "copy_repro")
        first_run_id = str(repro.grid["run_id"][0])
        copy_prefix = make_r2_prefix(repro.task_name, first_run_id, prefix_root=prefix_root)
        copy_shard = join_uri(f"r2://{inv.BUCKET}/{copy_prefix.rstrip('/')}", _FIRST_SHARD)
        assert np.array_equal(_param_array(source_shard), _param_array(copy_shard)), (
            "copy probe did not replay the source params verbatim — the derived "
            "copy_dataset_root_uri was not piped through"
        )
    finally:
        r2_io.purge_prefix(inv.BUCKET, f"{prefix_root}/")
