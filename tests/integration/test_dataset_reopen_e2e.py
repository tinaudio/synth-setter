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
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import lance
import numpy as np
import pytest

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    DATASET_FIELD_NAMES,
    MEL_SPEC_FIELD,
    PARAM_ARRAY_FIELD,
    dataset_field_dtypes,
)
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.lance_staging import (
    complete_attempt_names,
    shard_has_complete_attempt,
)
from synth_setter.pipeline.schemas.spec import DatasetSpec, Split
from synth_setter.pipeline.spec_io import load_spec_from_root

pytestmark = [pytest.mark.integration_r2, pytest.mark.r2, pytest.mark.slow]

_TASK_NAME = "reopen-e2e"
_BASE_SEED = 2862
_SOURCE_SIZES = (2, 2, 2)
_GROWN_TRAIN_SIZE = 4
_GENERATE_TIMEOUT_SECONDS = 300
_FINALIZE_TIMEOUT_SECONDS = 180
_SPLITS: tuple[Split, ...] = ("train", "val", "test")


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


def _purge_r2_prefix(root_uri: str) -> None:
    """Best-effort exact-prefix cleanup after the production-path run.

    :param root_uri: Non-root ``r2://`` dataset prefix ending in ``/``.
    """
    assert root_uri.startswith("r2://intermediate-data/data/") and root_uri.endswith("/")
    result = subprocess.run(  # noqa: S603
        [
            _cli("rclone"),
            "purge",
            f"r2:{root_uri.removeprefix('r2://')}",
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
            f"{root_uri}\n{result.stderr[-1000:]}\n"
        )


def _generate(run_id: str, train_size: int, output_dir: Path, *, stage: str) -> None:
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


def _split_arrays(spec: DatasetSpec, split: Split) -> dict[str, np.ndarray]:
    """Read and validate every writer-emitted array from one finalized split.

    :param spec: Finalized dataset identity.
    :param split: Split name.
    :returns: Core arrays keyed by writer field name.
    """
    target, storage_options = r2_io.lance_target(spec.r2.split_lance_uri(split))
    dataset = lance.dataset(target, storage_options=storage_options)
    assert set(DATASET_FIELD_NAMES).issubset(dataset.schema.names)
    table = dataset.to_table(columns=list(DATASET_FIELD_NAMES))
    arrays = {
        name: table.column(name).combine_chunks().to_numpy_ndarray()
        for name in DATASET_FIELD_NAMES
    }
    expected_dtypes = dataset_field_dtypes(spec.render)
    assert {name: array.dtype for name, array in arrays.items()} == expected_dtypes
    return arrays


def _assert_stats_match_train(spec: DatasetSpec, train_arrays: Mapping[str, np.ndarray]) -> None:
    """Compare persisted normalization stats with direct grown-train recomputation.

    :param spec: Finalized grown dataset identity.
    :param train_arrays: Every core array read from its train split.
    """
    train_mel = train_arrays[MEL_SPEC_FIELD].astype(np.float64)
    expected_mean = train_mel.mean(axis=0)
    expected_std = train_mel.std(axis=0)
    if spec.mask_degenerate_bins:
        expected_std[expected_std == 0] = 1
    with r2_io.downloaded_to_tempfile(spec.r2.stats_uri()) as stats_path:
        with np.load(stats_path) as stats:
            actual_mean = stats["mean"]
            actual_std = stats["std"]
    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_std, expected_std, rtol=1e-6, atol=5e-6)


def _assert_preserved_and_heldout_arrays_match(
    source: Mapping[Split, Mapping[str, np.ndarray]],
    grown: Mapping[Split, Mapping[str, np.ndarray]],
) -> None:
    """Compare copied train rows exactly and deterministic held-out rerenders numerically.

    Faust rerenders can differ below storage precision while preserving the exact parameter row, so
    held-out signal arrays use a bounded tolerance.

    :param source: Core arrays from the finalized source splits.
    :param grown: Core arrays from the finalized grown splits.
    """
    for field in DATASET_FIELD_NAMES:
        np.testing.assert_array_equal(
            grown["train"][field][: _SOURCE_SIZES[0]],
            source["train"][field],
            err_msg=f"preserved train {field} rows changed across the reopen",
        )
    for split in ("val", "test"):
        np.testing.assert_array_equal(
            grown[split][PARAM_ARRAY_FIELD],
            source[split][PARAM_ARRAY_FIELD],
            err_msg=f"renumbered {split} parameter rows changed",
        )
        for field in (AUDIO_FIELD, MEL_SPEC_FIELD):
            np.testing.assert_allclose(
                grown[split][field],
                source[split][field],
                rtol=1e-3,
                atol=1e-3,
                err_msg=f"renumbered {split} {field} rows changed",
            )


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
    root_uri = f"r2://intermediate-data/data/{_TASK_NAME}/{run_id}/"
    try:
        _generate(
            run_id,
            _SOURCE_SIZES[0],
            tmp_path / "generate-source",
            stage="source generation",
        )
        _finalize(
            root_uri,
            tmp_path / "finalize-source",
            "source finalization",
        )
        spec = load_spec_from_root(root_uri)
        assert spec.train_val_test_sizes == _SOURCE_SIZES
        assert spec.split_shard_ranges == {"train": (0, 1), "val": (1, 2), "test": (2, 3)}
        yield spec
    finally:
        _purge_r2_prefix(root_uri)


def test_reopen_extends_a_finalized_dataset_without_re_rendering_preserved_shards(
    tmp_path: Path,
) -> None:
    """Reopen, extend, and finalize for real — preserved rows must survive untouched.

    :param tmp_path: Per-test directory for real CLI output.
    """
    with _finalized_source(tmp_path) as source_spec:
        source_arrays: dict[Split, dict[str, np.ndarray]] = {
            split: _split_arrays(source_spec, split) for split in _SPLITS
        }
        dest_run_id = _run_id("grown")
        dest_root_uri = f"r2://intermediate-data/data/{_TASK_NAME}/{dest_run_id}/"
        try:
            _run_cli(
                [
                    _cli("synth-setter-reopen-dataset"),
                    "--source",
                    source_spec.r2.dataset_root_uri(),
                    "--train-size",
                    str(_GROWN_TRAIN_SIZE),
                    "--dest-run-id",
                    dest_run_id,
                    "--apply",
                ],
                timeout=_FINALIZE_TIMEOUT_SECONDS,
                stage="dataset reopen",
            )
            dest_spec = load_spec_from_root(dest_root_uri)
            assert dest_spec.train_val_test_sizes == (_GROWN_TRAIN_SIZE, 2, 2)
            assert dest_spec.split_shard_ranges == {
                "train": (0, 2),
                "val": (2, 3),
                "test": (3, 4),
            }

            # The preserved shard keeps its staged attempt; the renumbered
            # val/test ids must not, or the skip-probe would skip real train work.
            preserved_entries_before = r2_io.list_entries(
                dest_spec.r2.shard_staging_dir_uri(0), recursive=True
            )
            preserved_attempts_before = complete_attempt_names(
                [entry.path for entry in preserved_entries_before]
            )
            assert preserved_attempts_before
            assert not shard_has_complete_attempt(dest_spec, 1)
            assert not shard_has_complete_attempt(dest_spec, 3)

            _generate(
                dest_spec.run_id,
                _GROWN_TRAIN_SIZE,
                tmp_path / "generate-grown",
                stage="grown generation",
            )

            preserved_entries_after = r2_io.list_entries(
                dest_spec.r2.shard_staging_dir_uri(0), recursive=True
            )
            preserved_attempts_after = complete_attempt_names(
                [entry.path for entry in preserved_entries_after]
            )
            assert preserved_attempts_after == preserved_attempts_before
            assert preserved_entries_after == preserved_entries_before, (
                "preserved shard 0 was re-rendered instead of skipped"
            )
            for shard_id in range(1, 4):
                assert shard_has_complete_attempt(dest_spec, shard_id)

            _finalize(
                dest_spec.r2.dataset_root_uri(),
                tmp_path / "finalize-grown",
                "grown finalization",
            )

            grown_arrays: dict[Split, dict[str, np.ndarray]] = {
                split: _split_arrays(dest_spec, split) for split in _SPLITS
            }
            assert grown_arrays["train"][PARAM_ARRAY_FIELD].shape[0] == _GROWN_TRAIN_SIZE
            assert grown_arrays["val"][PARAM_ARRAY_FIELD].shape[0] == 2
            assert grown_arrays["test"][PARAM_ARRAY_FIELD].shape[0] == 2
            _assert_preserved_and_heldout_arrays_match(source_arrays, grown_arrays)

            train_rows = {row.tobytes() for row in grown_arrays["train"][PARAM_ARRAY_FIELD]}
            val_rows = {row.tobytes() for row in grown_arrays["val"][PARAM_ARRAY_FIELD]}
            test_rows = {row.tobytes() for row in grown_arrays["test"][PARAM_ARRAY_FIELD]}
            assert len(train_rows) == _GROWN_TRAIN_SIZE
            assert len(val_rows) == 2
            assert len(test_rows) == 2
            assert train_rows.isdisjoint(val_rows)
            assert train_rows.isdisjoint(test_rows)
            assert val_rows.isdisjoint(test_rows)
            _assert_stats_match_train(dest_spec, grown_arrays["train"])
        finally:
            _purge_r2_prefix(dest_root_uri)
