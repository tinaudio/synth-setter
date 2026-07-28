"""State tests for immutable rclone upload helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from synth_setter.pipeline.r2_io import upload_dir_immutable, upload_to_uri_immutable


def test_upload_to_uri_immutable_existing_different_file_refused(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    """Immutable single-file upload refuses different existing remote bytes.

    :param fake_r2_remote: Filesystem-backed real rclone remote root.
    :param tmp_path: Per-test local source root.
    """
    source = tmp_path / "manifest.json"
    source.write_bytes(b"first")
    upload_to_uri_immutable(source, "r2://bucket/release/manifest.json")
    source.write_bytes(b"second")

    with pytest.raises(subprocess.CalledProcessError):
        upload_to_uri_immutable(source, "r2://bucket/release/manifest.json")
    assert (fake_r2_remote / "bucket" / "release" / "manifest.json").read_bytes() == b"first"


def test_upload_to_uri_immutable_existing_matching_file_is_idempotent(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    """Immutable single-file upload accepts matching existing remote bytes.

    :param fake_r2_remote: Filesystem-backed real rclone remote root.
    :param tmp_path: Per-test local source root.
    """
    source = tmp_path / "manifest.json"
    source.write_bytes(b"same")

    upload_to_uri_immutable(source, "r2://bucket/release/manifest.json")
    upload_to_uri_immutable(source, "r2://bucket/release/manifest.json")

    assert (fake_r2_remote / "bucket" / "release" / "manifest.json").read_bytes() == b"same"


def test_upload_dir_immutable_existing_different_file_refused(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    """Immutable directory upload refuses different existing remote bytes.

    :param fake_r2_remote: Filesystem-backed real rclone remote root.
    :param tmp_path: Per-test local source root.
    """
    source = tmp_path / "tree"
    source.mkdir()
    (source / "data.bin").write_bytes(b"first")
    upload_dir_immutable(source, "r2://bucket/release")
    (source / "data.bin").write_bytes(b"second")

    with pytest.raises(subprocess.CalledProcessError):
        upload_dir_immutable(source, "r2://bucket/release")
    assert (fake_r2_remote / "bucket" / "release" / "data.bin").read_bytes() == b"first"


def test_upload_dir_immutable_existing_matching_tree_is_idempotent(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    """Immutable directory upload accepts matching existing remote bytes.

    :param fake_r2_remote: Filesystem-backed real rclone remote root.
    :param tmp_path: Per-test local source root.
    """
    source = tmp_path / "tree"
    source.mkdir()
    (source / "data.bin").write_bytes(b"same")

    upload_dir_immutable(source, "r2://bucket/release")
    upload_dir_immutable(source, "r2://bucket/release")

    assert (fake_r2_remote / "bucket" / "release" / "data.bin").read_bytes() == b"same"
