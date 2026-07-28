"""Subprocess E2E for the packaged NSynth import and verification CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import lance
import pytest

from tests.helpers.nsynth_fixtures import write_tiny_source


@pytest.mark.pipeline
def test_nsynth_cli_ingest_upload_then_verify_download_real_lance_rclone(
    fake_r2_remote: Path,
    tmp_path: Path,
) -> None:
    """The public command round-trips three splits through Lance and real rclone.

    :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
    :param tmp_path: Pytest temporary directory.
    """
    source_root = tmp_path / "source"
    write_tiny_source(source_root)
    output_root = tmp_path / "local-import"
    download_root = tmp_path / "downloaded-import"
    executable = Path(sys.executable).parent / "synth-setter-import-nsynth"
    expected = "train=1,valid=1,test=1"

    ingest = subprocess.run(  # noqa: S603 — installed repository entrypoint
        [
            str(executable),
            "ingest",
            str(source_root),
            str(output_root),
            "--batch-size",
            "1",
            "--expected-counts",
            expected,
        ],
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )

    assert ingest.returncode == 0, ingest.stderr
    remote_root = fake_r2_remote / "experiments" / "third_party" / "NSynth"
    assert (remote_root / "manifest.json").is_file()
    assert lance.dataset(remote_root / "train.lance").count_rows() == 1

    verify = subprocess.run(  # noqa: S603 — installed repository entrypoint
        [
            str(executable),
            "verify",
            str(source_root),
            str(download_root),
            "--batch-size",
            "1",
            "--expected-counts",
            expected,
        ],
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )

    assert verify.returncode == 0, verify.stderr
    assert "Verified 3 NSynth rows" in verify.stdout
    assert "0 mismatches" in verify.stdout
    assert (download_root / "manifest.json").read_bytes() == (
        remote_root / "manifest.json"
    ).read_bytes()
    assert lance.dataset(download_root / "test.lance").take_blobs("audio", indices=[0])[0].read()
