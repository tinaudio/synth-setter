"""Behavioral tests for the RunPod network-volume staging script."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[2]
_SCRIPT = _REPO_ROOT / "scripts/stage_runpod_network_volume.sh"


def _write_rclone_stub(bin_dir: Path) -> Path:
    """Write an rclone stub that records calls and can fail validation.

    :param bin_dir: Directory prepended to ``PATH``.
    :returns: Path to the call log.
    """
    call_log = bin_dir / "rclone.calls"
    stub = bin_dir / "rclone"
    stub.write_text(
        "#!/bin/bash\n"
        'printf \'%s\\n\' "$*" >> "${RCLONE_CALL_LOG}"\n'
        'if [[ "${1}" == check && "${FAIL_RCLONE_CHECK:-0}" == 1 ]]; then\n'
        "  exit 9\n"
        "fi\n"
    )
    stub.chmod(0o755)
    return call_log


def test_stage_script_copies_checksums_and_marks_complete(tmp_path: Path) -> None:
    """A successful stage validates source parity before writing its marker.

    :param tmp_path: Temporary stub-bin and destination parent.
    """
    call_log = _write_rclone_stub(tmp_path)
    destination = tmp_path / "network-volume" / "dataset"

    result = subprocess.run(  # noqa: S603 - fixed shell and repository script
        [
            "/bin/bash",
            str(_SCRIPT),
            "r2://bucket/finalized-dataset/",
            str(destination),
        ],
        cwd=_REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "RCLONE_CALL_LOG": str(call_log),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert call_log.read_text().splitlines() == [
        f"copy --immutable --checksum r2:bucket/finalized-dataset/ {destination}",
        f"check --one-way --checksum r2:bucket/finalized-dataset/ {destination}",
    ]
    assert (destination / ".synth-setter-stage-complete").read_bytes() == b""


def test_stage_script_remote_root_source_is_rejected(tmp_path: Path) -> None:
    """A remote root cannot be mistaken for one versioned dataset prefix.

    :param tmp_path: Temporary stub-bin and destination parent.
    """
    call_log = _write_rclone_stub(tmp_path)
    destination = tmp_path / "network-volume" / "dataset"

    result = subprocess.run(  # noqa: S603 - fixed shell and repository script
        ["/bin/bash", str(_SCRIPT), "r2://", str(destination)],
        cwd=_REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "RCLONE_CALL_LOG": str(call_log),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert not call_log.exists()


def test_stage_script_failed_check_leaves_dataset_unmarked(tmp_path: Path) -> None:
    """A failed parity check never publishes the completion marker.

    :param tmp_path: Temporary stub-bin and destination parent.
    """
    call_log = _write_rclone_stub(tmp_path)
    destination = tmp_path / "network-volume" / "dataset"

    result = subprocess.run(  # noqa: S603 - fixed shell and repository script
        [
            "/bin/bash",
            str(_SCRIPT),
            "r2://bucket/finalized-dataset/",
            str(destination),
        ],
        cwd=_REPO_ROOT,
        env={
            **os.environ,
            "FAIL_RCLONE_CHECK": "1",
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "RCLONE_CALL_LOG": str(call_log),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 9
    assert not (destination / ".synth-setter-stage-complete").exists()
