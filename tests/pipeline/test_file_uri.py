"""Tests for synth_setter.pipeline.file_uri — file:// scheme + bare-path dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_setter.pipeline.file_uri import (
    FILE_URI_SCHEME,
    file_uri_to_path,
    is_file_uri,
    local_path_from_arg,
)


class TestFileUriScheme:
    """The exported scheme constant is the canonical ``file://`` prefix."""

    def test_scheme_constant_is_canonical_file_prefix(self) -> None:
        """The module-level constant matches the RFC 8089 prefix."""
        assert FILE_URI_SCHEME == "file://"


class TestIsFileUri:
    """``is_file_uri`` recognises ``file://`` URIs, including localhost-host form."""

    def test_empty_authority_is_recognised(self) -> None:
        """``file:///abs/path`` (empty authority) is a file URI."""
        assert is_file_uri("file:///abs/path") is True

    def test_localhost_authority_is_recognised(self) -> None:
        """``file://localhost/abs/path`` (localhost authority) is a file URI."""
        assert is_file_uri("file://localhost/abs/path") is True

    def test_r2_uri_is_not_a_file_uri(self) -> None:
        """An ``r2://`` URI is not a file URI."""
        assert is_file_uri("r2://bucket/key.json") is False

    def test_bare_local_path_is_not_a_file_uri(self) -> None:
        """A bare filesystem path is not a file URI."""
        assert is_file_uri("/data/spec.json") is False

    def test_empty_string_is_not_a_file_uri(self) -> None:
        """An empty argument is not a file URI."""
        assert is_file_uri("") is False


class TestFileUriToPath:
    """``file_uri_to_path`` decodes a ``file://`` URI to a local filesystem Path."""

    def test_empty_authority_returns_absolute_path(self) -> None:
        """``file:///data/spec.json`` decodes to ``Path('/data/spec.json')``."""
        assert file_uri_to_path("file:///data/spec.json") == Path("/data/spec.json")

    def test_localhost_authority_returns_absolute_path(self) -> None:
        """``file://localhost/data/spec.json`` decodes to ``Path('/data/spec.json')``."""
        assert file_uri_to_path("file://localhost/data/spec.json") == Path("/data/spec.json")

    def test_percent_encoded_path_segments_are_decoded(self) -> None:
        """Percent-encoded characters in the path are decoded into the Path."""
        assert file_uri_to_path("file:///with%20space/spec.json") == Path("/with space/spec.json")

    def test_non_localhost_host_is_rejected(self) -> None:
        """A remote host (anything other than ``localhost``) is rejected."""
        with pytest.raises(ValueError, match="host must be empty or 'localhost'"):
            file_uri_to_path("file://example.com/abs/path")

    def test_non_file_scheme_is_rejected(self) -> None:
        """A non-``file://`` argument (e.g. ``r2://``) is rejected."""
        with pytest.raises(ValueError, match="not a file:// URI"):
            file_uri_to_path("r2://bucket/key.json")

    def test_empty_path_is_rejected(self) -> None:
        """A ``file://localhost`` URI with no path component is rejected."""
        with pytest.raises(ValueError, match="must carry an absolute path"):
            file_uri_to_path("file://localhost")


class TestLocalPathFromArg:
    """``local_path_from_arg`` accepts either a bare local path or a ``file://`` URI."""

    def test_bare_local_path_is_passed_through(self) -> None:
        """A bare filesystem path is passed through to ``pathlib.Path`` unchanged."""
        assert local_path_from_arg("/data/spec.json") == Path("/data/spec.json")

    def test_file_uri_is_decoded(self) -> None:
        """A ``file://`` URI is decoded via ``file_uri_to_path``."""
        assert local_path_from_arg("file:///data/spec.json") == Path("/data/spec.json")

    def test_malformed_file_uri_propagates_value_error(self) -> None:
        """A malformed ``file://`` URI propagates ``ValueError`` from the helper."""
        with pytest.raises(ValueError, match="host must be empty or 'localhost'"):
            local_path_from_arg("file://example.com/abs/path")
