"""Installed-CLI round trip for the BBC Sound Effects importer."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def _installed_cli() -> str:
    """Resolve the installed BBC importer executable or fail the test.

    :returns: Absolute or PATH-resolved executable location.
    """
    executable = shutil.which("synth-setter-bbc-sfx")
    if executable is None:
        pytest.fail("synth-setter-bbc-sfx is not installed in the test environment")
    return executable


@pytest.mark.skipif(shutil.which("rclone") is None, reason="rclone binary not available on PATH")
def test_installed_cli_convert_upload_download_verify_publishes_manifest_last(
    tmp_path: Path,
) -> None:
    """The installed CLI round trip publishes completion only after verification.

    :param tmp_path: Per-test source, release, download, and local-remote roots.
    """
    source_root = tmp_path / "source"
    item_root = source_root / "BBCSoundEffectsComplete"
    wav_path = item_root / "sounds" / "A long path & an apostrophe's effect.wav"
    wav_path.parent.mkdir(parents=True)
    sf.write(wav_path, np.array([0.0, 0.25, -0.25, 0.0], dtype=np.float32), 8_000)
    payload = wav_path.read_bytes()
    metadata = {
        "metadata": {"identifier": "BBCSoundEffectsComplete"},
        "files": [
            {
                "name": "sounds/A long path & an apostrophe's effect.wav",
                "size": str(len(payload)),
                "md5": hashlib.md5(payload).hexdigest(),  # noqa: S324 -- IA inventory contract.
                "sha1": hashlib.sha1(payload).hexdigest(),  # noqa: S324 -- IA inventory contract.
                "mtime": "1700000000",
            }
        ],
    }
    (item_root / "ia-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    release_root = tmp_path / "release"
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    env = {
        **os.environ,
        "RCLONE_CONFIG_R2_TYPE": "local",
    }
    cli = _installed_cli()

    convert = subprocess.run(  # noqa: S603 -- executes the resolved installed CLI.
        [
            cli,
            "convert",
            str(source_root),
            str(release_root),
            "--shard-target-bytes",
            "1000000",
        ],
        cwd=remote_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "converted 1 rows" in convert.stdout
    manifest_copy = tmp_path / "release-manifest.json"
    shutil.copy2(release_root / "release-manifest.json", manifest_copy)

    subprocess.run(  # noqa: S603 -- executes the resolved installed CLI.
        [cli, "upload", str(release_root)],
        cwd=remote_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    remote = remote_root / "experiments" / "third_party" / "BBCSoundEffectsComplete"
    assert not (remote / "manifest.json").exists()
    shutil.rmtree(release_root)

    download_root = tmp_path / "full download"
    verified = subprocess.run(  # noqa: S603 -- executes the resolved installed CLI.
        [
            cli,
            "verify",
            str(source_root),
            str(download_root),
            "--release-manifest",
            str(manifest_copy),
        ],
        cwd=remote_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"verified 1 rows and {len(payload)} bytes; 0 mismatches" in verified.stdout
    assert (remote / "manifest.json").read_bytes() == manifest_copy.read_bytes()
    assert (download_root / "metadata" / "inventory.jsonl").is_file()
