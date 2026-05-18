"""Behavioral tests for the R2Location nested model."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from synth_setter.pipeline.schemas.r2_location import R2Location


def _valid_kwargs(**overrides: Any) -> dict[str, Any]:  # noqa: DOC101,DOC103
    """Return a fresh kwargs dict for R2Location with the production defaults.

    :returns: Kwargs dict ready to splat into ``R2Location(**kwargs)``.
    :rtype: dict[str, Any]
    """
    kwargs: dict[str, Any] = {
        "bucket": "intermediate-data",
        "prefix_root": "data",
        "prefix": "data/run-x/run-x-20260101T000000000Z/",
    }
    kwargs.update(overrides)
    return kwargs


class TestR2LocationFieldValidators:
    """Field-level validators carried over from the legacy ``DatasetSpec`` flat fields."""

    def test_bucket_blank_raises(self) -> None:
        """Blank or whitespace-only bucket raises (rclone would receive a malformed URI)."""
        for blank in ("", "   ", "\t\n"):
            with pytest.raises(ValidationError, match="bucket must not be blank"):
                R2Location(**_valid_kwargs(bucket=blank))

    def test_prefix_root_blank_raises(self) -> None:
        """Blank ``prefix_root`` raises (prevents derived prefix starting with stray ``/``)."""
        for blank in ("", "   ", "\t\n"):
            with pytest.raises(ValidationError, match="prefix_root must not be blank"):
                R2Location(**_valid_kwargs(prefix_root=blank))

    def test_prefix_missing_trailing_slash_raises(self) -> None:
        """Prefix lacking trailing ``/`` raises (rclone would concat into ``…/prefixfilename``)."""
        with pytest.raises(ValidationError, match="prefix must end with"):
            R2Location(**_valid_kwargs(prefix="data/run-x/no-slash"))

    def test_empty_prefix_raises_via_trailing_slash_check(self) -> None:
        """Empty prefix is rejected without a dedicated empty-check.

        The trailing-slash validator catches it: ``""`` doesn't end with ``"/"``.
        """
        with pytest.raises(ValidationError, match="prefix must end with"):
            R2Location(**_valid_kwargs(prefix=""))

    def test_prefix_root_default_is_data(self) -> None:
        """The default ``prefix_root`` matches the production R2 layout root ``data``."""
        loc = R2Location(bucket="b", prefix="data/x/y/")
        assert loc.prefix_root == "data"


class TestR2LocationFrozen:
    """``R2Location`` is frozen so the materialized artifact stays immutable."""

    def test_assignment_after_construction_raises(self) -> None:
        """Mutating a field after construction raises ValidationError under ``frozen=True``."""
        loc = R2Location(**_valid_kwargs())
        with pytest.raises(ValidationError):
            loc.bucket = "other"  # type: ignore[misc]


class TestR2LocationUri:
    """``uri`` builds canonical ``r2://bucket/prefix/name`` URIs."""

    def test_uri_concatenates_prefix_and_name(self) -> None:
        """The URI follows the ``r2://{bucket}/{prefix}{name}`` convention exactly."""
        loc = R2Location(
            bucket="intermediate-data",
            prefix="data/run-x/run-x-20260101T000000000Z/",
        )
        assert (
            loc.uri("shard-000007.h5")
            == "r2://intermediate-data/data/run-x/run-x-20260101T000000000Z/shard-000007.h5"
        )

    def test_uri_preserves_nested_prefix(self) -> None:
        """Multi-segment prefixes are joined verbatim (the trailing ``/`` lives on prefix)."""
        loc = R2Location(bucket="bucket", prefix="a/b/c/")
        assert loc.uri("shard-000000.h5") == "r2://bucket/a/b/c/shard-000000.h5"


class TestR2LocationRclonePrefix:
    """``rclone_prefix`` builds the rclone-syntax destination (``<remote>:bucket/prefix``)."""

    def test_rclone_prefix_uses_central_remote_constant(self) -> None:
        """The remote name comes from the central constant, not a hardcoded ``r2:``.

        Pins ``RCLONE_REMOTE`` as the single source of truth so a future rename
        (``r2`` → ``r2-eu``) is a one-edit change rather than a global search.
        """
        loc = R2Location(
            bucket="intermediate-data",
            prefix="data/run-x/run-x-20260101T000000000Z/",
        )
        assert loc.rclone_prefix() == "r2:intermediate-data/data/run-x/run-x-20260101T000000000Z/"


class TestR2LocationEquality:
    """Two locations with the same fields compare equal (pydantic semantics)."""

    def test_equality_by_value(self) -> None:
        """Same-field instances compare equal; different-bucket instances do not."""
        a = R2Location(**_valid_kwargs())
        b = R2Location(**_valid_kwargs())
        c = R2Location(**_valid_kwargs(bucket="other-bucket"))
        assert a == b
        assert a != c
