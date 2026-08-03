"""Real R2 reopen E2E: extend a finalized dataset without re-rendering it (#2862).

Generates a finalized Faust dataset with non-zero val/test, reopens it at a
larger train size, re-runs the production generate + finalize CLIs against the
reopened root, and pins the two properties the design rests on: preserved
shards are skipped rather than re-rendered, and their rows survive byte-identical.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import lance
import numpy as np
import pytest

from synth_setter.data.vst.shapes import PARAM_ARRAY_FIELD
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.dataset_reopen import reopen_dataset
from synth_setter.pipeline.data.lance_staging import (
    complete_attempt_names,
    shard_has_complete_attempt,
)
from synth_setter.pipeline.schemas.spec import DatasetSpec
from synth_setter.pipeline.spec_io import load_spec_from_root

pytestmark = [pytest.mark.integration_r2, pytest.mark.r2, pytest.mark.slow]

_TASK_NAME = "reopen-e2e"
_BASE_SEED = 2862
_SOURCE_SIZES = (2, 2, 2)
_GROWN_TRAIN_SIZE = 4
_GENERATE_TIMEOUT_SECONDS = 300
_FINALIZE_TIMEOUT_SECONDS = 180


def _run_id(label: str) -> str:
    """Return an R2-isolated run identifier.

    :param label: Stage label distinguishing the source and reopened runs.
    :returns: Run identifier unique across local and CI attempts.
    """
    github_run_id = os.environ.get("GITHUB_RUN_ID", "local")
    github_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "0")
    return f"reopen-{label}-{github_run_id}-{github_attempt}-{uuid.uuid4().hex[:8]}"


def _cli(name: str) -> str:
    """Resolve one installed command-line executable.

    :param name: Executable basename.
    :returns: Absolute executable path.
    :raises RuntimeError: If the active environment does not expose ``name``.
    """
    executable = Path(sys.executable).with_name(name)
    if executable.is_file():
        return str(executable)
    resolved = shutil.which(name)
    if resolved is None:
        raise RuntimeError(f"required public CLI is unavailable: {name}")
    return resolved


def _run_cli(argv: list[str], *, timeout: int, stage: str) -> None:
    """Run one production CLI and expose both output streams on failure.

    :param argv: Complete public-CLI argv.
    :param timeout: Wall-clock ceiling in seconds.
    :param stage: Human-readable pipeline stage for failure output.
    :raises AssertionError: If the command fails or exceeds ``timeout``.
    """
    try:
        result = subprocess.run(  # noqa: S603
            argv, capture_output=True, text=True, check=False, timeout=timeout
        )
    except subprocess.TimeoutExpired as expired:
        raise AssertionError(f"{stage} exceeded {timeout}s") from expired
    assert result.returncode == 0, (
        f"{stage} failed with exit {result.returncode}\n"
        f"--- STDOUT ---\n{result.stdout[-4000:]}\n"
        f"--- STDERR ---\n{result.stderr[-4000:]}"
    )


def _purge_r2_prefix(spec: DatasetSpec) -> None:
    """Best-effort exact-prefix cleanup after the production-path run.

    :param spec: Dataset identity whose prefix can be deleted safely.
    """
    result = subprocess.run(  # noqa: S603
        [
            _cli("rclone"),
            "purge",
            f"r2:{spec.r2.bucket}/{spec.r2.prefix}",
            "--checksum",
            "--contimeout=10s",
            "--timeout=60s",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        sys.stderr.write(
            f"WARN: exact-prefix cleanup exited {result.returncode} for "
            f"r2:{spec.r2.bucket}/{spec.r2.prefix}\n{result.stderr[-1000:]}\n"
        )


def _generate(run_id: str, train_size: int, output_dir: Path, stage: str) -> None:
    """Run the production generation CLI for one run identity.

    :param run_id: Run id owning the R2 prefix.
    :param train_size: Train split row count.
    :param output_dir: Local Hydra output directory.
    :param stage: Human-readable stage label for failure output.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_cli(
        [
            _cli("synth-setter-generate-dataset"),
            "experiment=generate_dataset/smoke-shard-lance",
            f"task_name={_TASK_NAME}",
            f"run_id={run_id}",
            "synth=faust_bright_organ",
            "render=faust",
            f"train_val_test_sizes=[{train_size},2,2]",
            "render.samples_per_shard=2",
            "render.samples_per_render_batch=1",
            "mask_degenerate_bins=true",
            "~logger",
            f"base_seed={_BASE_SEED}",
            f"train_val_test_seeds=[{_BASE_SEED},{_BASE_SEED + 1},{_BASE_SEED + 2}]",
            f"paths.output_dir={output_dir}",
            "hydra.job.chdir=false",
        ],
        timeout=_GENERATE_TIMEOUT_SECONDS,
        stage=stage,
    )


def _finalize(root_uri: str, output_dir: Path, stage: str) -> None:
    """Run the production finalize CLI against one dataset root.

    :param root_uri: ``r2://`` root of the dataset to finalize.
    :param output_dir: Local Hydra output directory.
    :param stage: Human-readable stage label for failure output.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_cli(
        [
            _cli("synth-setter-finalize-dataset"),
            f"dataset_root_uri={root_uri}",
            "~logger",
            f"paths.output_dir={output_dir}",
            "hydra.job.chdir=false",
        ],
        timeout=_FINALIZE_TIMEOUT_SECONDS,
        stage=stage,
    )


def _split_params(spec: DatasetSpec, split: str) -> np.ndarray:
    """Read one finalized split's parameter matrix from R2.

    :param spec: Finalized dataset identity.
    :param split: Split name.
    :returns: ``(rows, params)`` float array of stored parameters.
    """
    target, storage_options = r2_io.lance_target(spec.r2.split_lance_uri(split))  # type: ignore[arg-type]
    table = lance.dataset(target, storage_options=storage_options).to_table(
        columns=[PARAM_ARRAY_FIELD]
    )
    return table.column(PARAM_ARRAY_FIELD).combine_chunks().to_numpy_ndarray()


@contextmanager
def _finalized_source(tmp_path: Path) -> Iterator[DatasetSpec]:
    """Generate and finalize a real source dataset, then yield its spec.

    :yields DatasetSpec: Finalized source spec, purged after the test.
    :param tmp_path: Per-test local work area.
    """
    probe = subprocess.run(  # noqa: S603
        [_cli("rclone"), "lsd", "r2:", "--checksum"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert probe.returncode == 0, (
        "real R2 prerequisite probe failed\n"
        f"--- STDOUT ---\n{probe.stdout[-2000:]}\n"
        f"--- STDERR ---\n{probe.stderr[-2000:]}"
    )

    run_id = _run_id("source")
    _generate(run_id, _SOURCE_SIZES[0], tmp_path / "generate-source", "source generation")
    _finalize(
        f"r2://intermediate-data/data/{_TASK_NAME}/{run_id}/",
        tmp_path / "finalize-source",
        "source finalization",
    )
    spec = load_spec_from_root(f"r2://intermediate-data/data/{_TASK_NAME}/{run_id}/")
    try:
        assert spec.train_val_test_sizes == _SOURCE_SIZES
        assert spec.split_shard_ranges == {"train": (0, 1), "val": (1, 2), "test": (2, 3)}
        yield spec
    finally:
        _purge_r2_prefix(spec)


def test_reopen_extends_a_finalized_dataset_without_re_rendering_preserved_shards(
    tmp_path: Path,
) -> None:
    """Reopen, extend, and finalize for real — preserved rows must survive untouched.

    :param tmp_path: Per-test directory for real CLI output.
    """
    with _finalized_source(tmp_path) as source_spec:
        source_train_params = _split_params(source_spec, "train")
        source_val_params = _split_params(source_spec, "val")

        plan = reopen_dataset(
            source_spec.r2.dataset_root_uri(),
            (_GROWN_TRAIN_SIZE, 2, 2),
            dest_run_id=_run_id("grown"),
        )
        dest_spec = plan.dest_spec
        try:
            assert plan.preserved_shard_ids == range(0, 1)
            assert plan.discarded_shard_ids == range(1, 3)
            assert plan.pending_shard_ids == range(1, 4)

            # The preserved shard keeps its staged attempt; the renumbered
            # val/test ids must not, or the skip-probe would skip real train work.
            preserved_attempts_before = complete_attempt_names(
                [
                    entry.path
                    for entry in r2_io.list_entries(
                        dest_spec.r2.shard_staging_dir_uri(0), recursive=True
                    )
                ]
            )
            assert preserved_attempts_before
            assert not shard_has_complete_attempt(dest_spec, 1)
            assert not shard_has_complete_attempt(dest_spec, 3)

            _generate(
                dest_spec.run_id,
                _GROWN_TRAIN_SIZE,
                tmp_path / "generate-grown",
                "grown generation",
            )

            preserved_attempts_after = complete_attempt_names(
                [
                    entry.path
                    for entry in r2_io.list_entries(
                        dest_spec.r2.shard_staging_dir_uri(0), recursive=True
                    )
                ]
            )
            assert preserved_attempts_after == preserved_attempts_before, (
                "preserved shard 0 was re-rendered instead of skipped"
            )
            for shard_id in plan.pending_shard_ids:
                assert shard_has_complete_attempt(dest_spec, shard_id)

            _finalize(
                dest_spec.r2.dataset_root_uri(),
                tmp_path / "finalize-grown",
                "grown finalization",
            )

            grown_train_params = _split_params(dest_spec, "train")
            grown_val_params = _split_params(dest_spec, "val")

            assert grown_train_params.shape[0] == _GROWN_TRAIN_SIZE
            assert grown_val_params.shape[0] == 2
            assert np.array_equal(
                grown_train_params[: source_train_params.shape[0]], source_train_params
            ), "preserved train rows changed across the reopen"
            assert np.array_equal(grown_val_params, source_val_params), (
                "renumbered val rows differ from the source's despite split-local offsets"
            )
        finally:
            _purge_r2_prefix(dest_spec)
