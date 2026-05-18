"""Tests for ``synth_setter.pipeline.schemas.r2_location.R2Location``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from synth_setter.pipeline.schemas.r2_location import R2Location
from synth_setter.pipeline.schemas.spec import ShardSpec


def _shard(filename: str = "shard-000042.h5") -> ShardSpec:  # noqa: DOC101,DOC103,DOC201,DOC203
    """Return a minimal ``ShardSpec`` for URI-construction tests."""
    return ShardSpec(shard_id=42, filename=filename, seed=42)


class TestR2LocationConstruction:
    """Field defaults, validators, and frozen-model invariants."""

    def test_required_fields_only(self) -> None:
        """Bucket + prefix are required; prefix_root defaults to ``data``."""
        loc = R2Location(bucket="intermediate-data", prefix="data/foo/bar/")
        assert loc.bucket == "intermediate-data"
        assert loc.prefix_root == "data"
        assert loc.prefix == "data/foo/bar/"

    def test_missing_bucket_raises(self) -> None:
        """``bucket`` is required — no default."""
        with pytest.raises(ValidationError):
            R2Location(prefix="data/foo/bar/")  # type: ignore[call-arg]

    def test_missing_prefix_raises(self) -> None:
        """``prefix`` is required — no default."""
        with pytest.raises(ValidationError):
            R2Location(bucket="b")  # type: ignore[call-arg]

    def test_blank_bucket_raises(self) -> None:
        """Blank bucket → rclone would emit a malformed destination."""
        for blank in ("", "   ", "\t\n"):
            with pytest.raises(ValidationError, match="r2_bucket must not be blank"):
                R2Location(bucket=blank, prefix="data/x/y/")

    def test_blank_prefix_root_raises(self) -> None:
        """Blank prefix_root would yield a derived prefix starting with ``/``."""
        for blank in ("", "   ", "\t\n"):
            with pytest.raises(ValidationError, match="r2_prefix_root must not be blank"):
                R2Location(bucket="b", prefix_root=blank, prefix="data/x/y/")

    def test_prefix_must_end_with_slash(self) -> None:
        """Prefix without trailing ``/`` would concatenate to ``.../prefixfilename``."""
        with pytest.raises(ValidationError, match="r2_prefix must end with"):
            R2Location(bucket="b", prefix="data/no/slash")

    def test_extra_fields_forbidden(self) -> None:
        """Strict mode + extra=forbid: unknown keys at the trust boundary fail-fast."""
        with pytest.raises(ValidationError):
            R2Location(bucket="b", prefix="p/", unexpected="x")  # type: ignore[call-arg]

    def test_frozen_post_construction(self) -> None:
        """Attribute reassignment is rejected — the materialized location is immutable."""
        loc = R2Location(bucket="b", prefix="p/")
        with pytest.raises(ValidationError):
            loc.bucket = "other"  # type: ignore[misc]


class TestR2LocationURIMethods:
    """``uri``, ``rclone_prefix``, ``shard_uri`` build the canonical R2 strings."""

    def test_uri_under_arbitrary_key(self) -> None:
        """``uri(key)`` is ``r2://<bucket>/<key>`` regardless of ``prefix``.

        Used by the launcher for top-level keys like ``skypilot-launcher-specs/<job>.json``
        that live outside the spec's own data prefix.
        """
        loc = R2Location(bucket="intermediate-data", prefix="data/run/")
        assert (
            loc.uri("skypilot-launcher-specs/job-1.json")
            == "r2://intermediate-data/skypilot-launcher-specs/job-1.json"
        )

    def test_rclone_prefix_uses_rclone_form(self) -> None:
        """``rclone_prefix()`` returns rclone's ``r2:bucket/prefix`` (single colon, no //)."""
        loc = R2Location(bucket="b", prefix="data/run/")
        assert loc.rclone_prefix() == "r2:b/data/run/"

    def test_shard_uri_joins_bucket_prefix_filename(self) -> None:
        """``shard_uri(shard)`` is ``r2://<bucket>/<prefix><shard.filename>``."""
        loc = R2Location(bucket="intermediate-data", prefix="data/run-x/")
        assert (
            loc.shard_uri(_shard("shard-000007.h5"))
            == "r2://intermediate-data/data/run-x/shard-000007.h5"
        )

    def test_shard_uri_preserves_nested_prefix(self) -> None:
        """Multi-segment prefixes are joined verbatim; caller controls trailing slash."""
        loc = R2Location(bucket="b", prefix="a/b/c/")
        assert loc.shard_uri(_shard("shard-000000.h5")) == "r2://b/a/b/c/shard-000000.h5"
