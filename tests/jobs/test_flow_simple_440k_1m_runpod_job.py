"""Behavioral coverage for the 440k conditioning RunPod job launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "jobs/train/surge/launch_flow_simple_440k_1m_runpod.sh"
EXPECTED_ARMS = {
    "clap",
    "clap_online",
    "log_mel",
    "m2l",
    "matpac_plus",
    "mel",
    "same_l",
    "same_l_online",
    "same_s",
    "same_s_online",
    "ssondo",
}


def test_launcher_without_execute_prints_plan_without_submitting(tmp_path: Path) -> None:
    """Default invocation lists every arm without calling the paid launcher.

    :param tmp_path: Isolates the executable lookup and call marker.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "uv-called"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(f"#!/bin/bash\ntouch {marker}\n")
    fake_uv.chmod(0o755)

    result = subprocess.run(  # noqa: S603 — repository script and test-owned environment
        [SCRIPT],
        cwd=REPO_ROOT,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
        check=True,
        capture_output=True,
        text=True,
    )

    assert not marker.exists()
    assert set(result.stdout.splitlines()) == {
        f"DRY RUN: surge/flow_simple_440k_1m_{arm}" for arm in EXPECTED_ARMS
    }


def test_launcher_execute_submits_every_arm_without_sky_api_credentials(tmp_path: Path) -> None:
    """Execute mode starts one launch per arm with Sky API credentials removed.

    :param tmp_path: Isolates the fake launcher boundary and captured calls.
    """
    fake_bin = tmp_path / "bin"
    calls = tmp_path / "calls"
    fake_bin.mkdir()
    calls.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        """#!/bin/bash
set -euo pipefail
selector="${5}"
arm="${selector##*_1m_}"
{
  printf 'args=%s\\n' "$*"
  printf 'endpoint=%s\\n' "${SKYPILOT_API_SERVER_ENDPOINT-unset}"
  printf 'token=%s\\n' "${SKYPILOT_API_SERVER_TOKEN-unset}"
  printf 'key=%s\\n' "${SKYPILOT_API_SERVER_KEY-unset}"
} >"${CALLS_DIR}/${arm}"
"""
    )
    fake_uv.chmod(0o755)
    env = {
        **os.environ,
        "CALLS_DIR": str(calls),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SKYPILOT_API_SERVER_ENDPOINT": "must-be-removed",
        "SKYPILOT_API_SERVER_KEY": "must-be-removed",
        "SKYPILOT_API_SERVER_TOKEN": "must-be-removed",
    }

    subprocess.run(  # noqa: S603 — repository script and test-owned environment
        [SCRIPT, "--execute"], cwd=REPO_ROOT, env=env, check=True
    )

    assert {path.name for path in calls.iterdir()} == EXPECTED_ARMS
    for arm in EXPECTED_ARMS:
        call = (calls / arm).read_text()
        assert f"EXPERIMENT surge/flow_simple_440k_1m_{arm}" in call
        assert "train-runpod-flow-simple-440k-1m.yaml" in call
        assert "endpoint=unset" in call
        assert "token=unset" in call
        assert "key=unset" in call


def test_launcher_unknown_argument_exits_nonzero() -> None:
    """Unknown options fail closed instead of launching jobs."""
    result = subprocess.run(  # noqa: S603 — repository script and fixed argument
        [SCRIPT, "--unknown"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Usage:" in result.stderr
