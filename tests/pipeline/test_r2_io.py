"""Tests for pipeline.r2_io — rclone-backed R2 I/O helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.pipeline import r2_io


class TestIsR2Uri:
    """Tests for is_r2_uri scheme detection."""

    def test_r2_uri_returns_true(self) -> None:
        """An r2://bucket/key URI is recognized."""
        assert r2_io.is_r2_uri("r2://bucket/key.json")

    def test_local_path_returns_false(self) -> None:
        """An absolute local path is not an r2 URI."""
        assert not r2_io.is_r2_uri("local-spec.json")

    def test_relative_path_returns_false(self) -> None:
        """A relative local path is not an r2 URI."""
        assert not r2_io.is_r2_uri("spec.json")

    def test_other_scheme_returns_false(self) -> None:
        """A non-r2 URI scheme (s3://) is not recognized."""
        assert not r2_io.is_r2_uri("s3://bucket/key.json")

    def test_empty_string_returns_false(self) -> None:
        """Empty string is not an r2 URI."""
        assert not r2_io.is_r2_uri("")


class TestToRclonePath:
    """Tests for _to_rclone_path URI translation."""

    def test_strips_r2_scheme(self) -> None:
        """`r2://bucket/key` becomes rclone's `r2:bucket/key`."""
        assert r2_io._to_rclone_path("r2://bucket/key.json") == "r2:bucket/key.json"

    def test_handles_nested_keys(self) -> None:
        """Nested key paths under the bucket are preserved."""
        assert r2_io._to_rclone_path("r2://bucket/a/b/c.json") == "r2:bucket/a/b/c.json"

    def test_rejects_local_path(self) -> None:
        """Non-r2 URIs raise ValueError so callers branch on is_r2_uri first."""
        with pytest.raises(ValueError, match="not an r2:// URI"):
            r2_io._to_rclone_path("local-spec.json")


class TestDownloadToPath:
    """Tests for download_to_path — file→file copy."""

    def test_invokes_rclone_copyto_with_checksum(self, tmp_path: Path) -> None:
        """Verifies the rclone command shape (copyto + --checksum + URI translation)."""
        dest = tmp_path / "out.json"
        with patch.object(r2_io.subprocess, "check_call") as mock_call:
            r2_io.download_to_path("r2://bucket/key.json", dest)
        args = mock_call.call_args[0][0]
        assert args[:3] == ["rclone", "copyto", "--checksum"]
        assert args[3] == "r2:bucket/key.json"
        assert args[4] == str(dest)

    def test_rejects_non_r2_uri(self, tmp_path: Path) -> None:
        """A local-path source is rejected — caller must branch on is_r2_uri."""
        with pytest.raises(ValueError, match="not an r2:// URI"):
            r2_io.download_to_path("local-spec.json", tmp_path / "out.json")


class TestUploadToUri:
    """Tests for upload_to_uri — file→file upload with reliability flags."""

    def test_invokes_rclone_copyto_with_reliability_flags(self, tmp_path: Path) -> None:
        """Upload command includes -vv, --contimeout, --timeout, --retries, --checksum."""
        src = tmp_path / "in.json"
        src.write_text("{}")
        with patch.object(r2_io.subprocess, "check_call") as mock_call:
            r2_io.upload_to_uri(src, "r2://bucket/key.json")
        args = mock_call.call_args[0][0]
        assert args[:2] == ["rclone", "copyto"]
        assert "-vv" in args
        assert "--checksum" in args
        assert "--contimeout=30s" in args
        assert "--timeout=300s" in args
        assert "--retries=3" in args
        assert args[-2] == str(src)
        assert args[-1] == "r2:bucket/key.json"

    def test_rejects_non_r2_uri(self, tmp_path: Path) -> None:
        """A local-path destination is rejected."""
        with pytest.raises(ValueError, match="not an r2:// URI"):
            r2_io.upload_to_uri(tmp_path / "in.json", "local-dest.json")


class TestDownloadedToTempfile:
    """Tests for the downloaded_to_tempfile context manager."""

    def test_yields_local_path_named_after_uri_basename(self) -> None:
        """The yielded Path's name matches the URI's last path segment, and the tempdir is cleaned
        on exit."""
        seen: list[Path] = []

        def fake_check_call(args: list[str]) -> None:
            Path(args[-1]).write_text('{"ok": true}')

        with patch.object(r2_io.subprocess, "check_call", side_effect=fake_check_call):
            with r2_io.downloaded_to_tempfile("r2://bucket/path/spec.json") as local:
                seen.append(local)
                assert local.name == "spec.json"
                assert local.read_text() == '{"ok": true}'

        assert not seen[0].exists()

    def test_cleanup_runs_even_on_exception(self) -> None:
        """If the with-block raises, the tempdir is still cleaned up."""
        seen: list[Path] = []

        def fake_check_call(args: list[str]) -> None:
            Path(args[-1]).write_text("{}")

        with patch.object(r2_io.subprocess, "check_call", side_effect=fake_check_call):
            with pytest.raises(RuntimeError, match="boom"):
                with r2_io.downloaded_to_tempfile("r2://bucket/spec.json") as local:
                    seen.append(local)
                    raise RuntimeError("boom")

        assert not seen[0].exists()
