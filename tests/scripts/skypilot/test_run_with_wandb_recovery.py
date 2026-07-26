"""Behavior tests for the remote W&B recovery wrapper."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WRAPPER = _REPO_ROOT / "scripts/skypilot/run_with_wandb_recovery.sh"


def _run_wrapper(worker_root: Path, child: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — the test owns the child command.
        [  # noqa: S607 — image-provided bash is the wrapper's runtime contract.
            "bash",
            str(_WRAPPER),
            str(worker_root / "wandb"),
            "--",
            "bash",
            "-c",
            child,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_wrapper_relative_wandb_root_uploads_canonical_run(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    """A cwd-relative W&B root resolves to the same containment contract.

    :param fake_r2_remote: Local filesystem backing the real ``r2:`` remote.
    :param tmp_path: Temporary worker directory.
    """
    child = """
mkdir -p worker/wandb/run-20260726_120000-relative1
printf history > worker/wandb/run-20260726_120000-relative1/run-relative1.wandb
ln -s run-20260726_120000-relative1 worker/wandb/latest-run
"""

    result = subprocess.run(  # noqa: S603 — the test owns the child command.
        [  # noqa: S607 — image-provided bash is the wrapper's runtime contract.
            "bash",
            str(_WRAPPER),
            "worker/wandb",
            "--",
            "bash",
            "-c",
            child,
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    uri = next(
        line.removeprefix("WANDB_RECOVERY_URI=")
        for line in result.stdout.splitlines()
        if line.startswith("WANDB_RECOVERY_URI=")
    )
    assert (fake_r2_remote / uri.removeprefix("r2://")).is_file()


def test_wrapper_worker_removed_bundle_remains_retrievable(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    """The uploaded archive retains the canonical W&B files after worker deletion.

    :param fake_r2_remote: Local filesystem backing the real ``r2:`` remote.
    :param tmp_path: Temporary worker and retrieval directories.
    """
    worker_root = tmp_path / "worker"
    run_dir = worker_root / "wandb" / "run-20260726_120000-run123"
    child = f"""
set -e
mkdir -p {run_dir}/logs {run_dir}/files
printf history > {run_dir}/run-run123.wandb
printf debug > {run_dir}/logs/debug.log
printf internal > {run_dir}/logs/debug-internal.log
printf config > {run_dir}/files/config.yaml
ln -s {run_dir.name} {worker_root}/wandb/latest-run
"""

    result = _run_wrapper(worker_root, child)

    assert result.returncode == 0, result.stderr
    uri_line = next(
        line for line in result.stdout.splitlines() if line.startswith("WANDB_RECOVERY_URI=")
    )
    uri = uri_line.removeprefix("WANDB_RECOVERY_URI=")
    assert uri.startswith("r2://intermediate-data/diagnostics/wandb/training/run123/")
    assert not any(character in uri for character in ("@", "?", "#"))
    archive = fake_r2_remote / uri.removeprefix("r2://")
    assert archive.is_file()

    shutil.rmtree(worker_root)
    downloaded = tmp_path / "downloaded.tar.gz"
    subprocess.run(  # noqa: S603 — URI comes from the tested wrapper.
        [  # noqa: S607 — rclone is the production binary under test.
            "rclone",
            "copyto",
            "--checksum",
            f"r2:{uri.removeprefix('r2://')}",
            str(downloaded),
        ],
        check=True,
    )
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(downloaded, "r:gz") as bundle:
        bundle.extractall(extracted, filter="data")

    recovered_run = extracted / run_dir.name
    assert (recovered_run / "run-run123.wandb").read_text() == "history"
    assert (recovered_run / "logs/debug.log").read_text() == "debug"
    assert (recovered_run / "logs/debug-internal.log").read_text() == "internal"
    assert (recovered_run / "files/config.yaml").read_text() == "config"


def test_wrapper_real_wandb_sdk_bundle_contains_syncable_datastore(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    """A real W&B run produces the datastore and SDK diagnostics in the bundle.

    :param fake_r2_remote: Local filesystem backing the real ``r2:`` remote.
    :param tmp_path: Temporary worker directory.
    """
    worker_root = tmp_path / "worker"
    wandb_root = worker_root / "wandb"
    python_code = (
        "import wandb; "
        f"run = wandb.init(dir={str(worker_root)!r}, mode='offline', id='sdkreal1'); "
        "run.log({'train/loss': 1.25}, step=1); "
        "run.finish()"
    )
    child = f"{shlex.quote(sys.executable)} -c {shlex.quote(python_code)}"

    result = _run_wrapper(worker_root, child)

    assert result.returncode == 0, result.stderr
    uri = next(
        line.removeprefix("WANDB_RECOVERY_URI=")
        for line in result.stdout.splitlines()
        if line.startswith("WANDB_RECOVERY_URI=")
    )
    archive = fake_r2_remote / uri.removeprefix("r2://")
    shutil.rmtree(wandb_root)
    extracted = tmp_path / "sdk-extracted"
    extracted.mkdir()
    with tarfile.open(archive, "r:gz") as bundle:
        run_prefix = next(
            member.name.split("/", maxsplit=1)[0]
            for member in bundle.getmembers()
            if member.name.endswith("/run-sdkreal1.wandb")
        )
        required = [
            f"{run_prefix}/run-sdkreal1.wandb",
            f"{run_prefix}/logs/debug.log",
            f"{run_prefix}/logs/debug-internal.log",
        ]
        for name in required:
            bundle.extract(name, extracted, filter="data")

    run_dir = extracted / run_prefix
    assert (run_dir / "run-sdkreal1.wandb").stat().st_size > 0
    assert (run_dir / "logs/debug.log").stat().st_size > 0
    assert (run_dir / "logs/debug-internal.log").stat().st_size > 0


def test_wrapper_child_failure_uploads_bundle_and_preserves_status(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    """A failed training process remains failed after its diagnostics upload.

    :param fake_r2_remote: Local filesystem backing the real ``r2:`` remote.
    :param tmp_path: Temporary worker directory.
    """
    worker_root = tmp_path / "worker"
    run_dir = worker_root / "wandb" / "run-20260726_120000-failed123"
    child = f"""
mkdir -p {run_dir}/logs
printf history > {run_dir}/run-failed123.wandb
printf transport-failed > {run_dir}/logs/debug-internal.log
ln -s {run_dir.name} {worker_root}/wandb/latest-run
exit 23
"""

    result = _run_wrapper(worker_root, child)

    assert result.returncode == 23
    uri = next(
        line.removeprefix("WANDB_RECOVERY_URI=")
        for line in result.stdout.splitlines()
        if line.startswith("WANDB_RECOVERY_URI=")
    )
    assert (fake_r2_remote / uri.removeprefix("r2://")).is_file()


def test_wrapper_success_without_wandb_run_fails_durability_gate(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    """A successful training command cannot pass without a recovery bundle.

    :param fake_r2_remote: Activates the real local-backed rclone remote.
    :param tmp_path: Temporary worker directory.
    """
    result = _run_wrapper(tmp_path / "worker", "exit 0")

    assert result.returncode != 0
    assert "no canonical latest-run directory" in result.stderr
    assert "WANDB_RECOVERY_URI=" not in result.stdout


def test_wrapper_upload_removes_recovery_bundles_older_than_retention(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    """Each upload enforces the fixed retention window within its private prefix.

    :param fake_r2_remote: Local filesystem backing the real ``r2:`` remote.
    :param tmp_path: Temporary worker directory.
    """
    expired = (
        fake_r2_remote
        / "intermediate-data/diagnostics/wandb/training/old-run/attempt/wandb-run.tar.gz"
    )
    expired.parent.mkdir(parents=True)
    expired.write_text("expired")
    old = (datetime.now(UTC) - timedelta(days=31)).timestamp()
    os.utime(expired, (old, old))

    worker_root = tmp_path / "worker"
    run_dir = worker_root / "wandb" / "run-20260726_120000-retained1"
    child = f"""
mkdir -p {run_dir}
printf history > {run_dir}/run-retained1.wandb
ln -s {run_dir.name} {worker_root}/wandb/latest-run
"""

    result = _run_wrapper(worker_root, child)

    assert result.returncode == 0, result.stderr
    assert not expired.exists()


def test_wrapper_same_run_id_uses_distinct_attempt_uris(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    """Retries of one W&B run never overwrite an earlier recovery bundle.

    :param fake_r2_remote: Local filesystem backing the real ``r2:`` remote.
    :param tmp_path: Temporary worker directories.
    """
    uris: list[str] = []
    for worker_name in ("worker-a", "worker-b"):
        worker_root = tmp_path / worker_name
        run_dir = worker_root / "wandb" / "run-20260726_120000-shared123"
        child = f"""
mkdir -p {run_dir}
printf history > {run_dir}/run-shared123.wandb
ln -s {run_dir.name} {worker_root}/wandb/latest-run
"""
        result = _run_wrapper(worker_root, child)
        assert result.returncode == 0, result.stderr
        uris.append(
            next(
                line.removeprefix("WANDB_RECOVERY_URI=")
                for line in result.stdout.splitlines()
                if line.startswith("WANDB_RECOVERY_URI=")
            )
        )

    assert uris[0] != uris[1]
    assert all((fake_r2_remote / uri.removeprefix("r2://")).is_file() for uri in uris)
