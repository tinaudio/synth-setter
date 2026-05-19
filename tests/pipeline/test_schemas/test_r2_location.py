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
            with pytest.raises(ValidationError, match=r"r2\.bucket must not be blank"):
                R2Location(bucket=blank, prefix="data/x/y/")

    def test_blank_prefix_root_raises(self) -> None:
        """Blank prefix_root would yield a derived prefix starting with ``/``."""
        for blank in ("", "   ", "\t\n"):
            with pytest.raises(
                ValidationError, match=r"r2\.prefix_root must not be blank or slash-only"
            ):
                R2Location(bucket="b", prefix_root=blank, prefix="data/x/y/")

    def test_slash_only_prefix_root_raises(self) -> None:
        """Slash-only prefix_root (e.g. ``"////"``) is rejected — matches make_r2_prefix."""
        for slash_only in ("/", "//", "////"):
            with pytest.raises(
                ValidationError, match=r"r2\.prefix_root must not be blank or slash-only"
            ):
                R2Location(bucket="b", prefix_root=slash_only, prefix="data/x/y/")

    def test_whitespace_wrapped_slash_only_prefix_root_raises(self) -> None:
        """Whitespace-wrapped slash-only values (e.g. ``" / "``) are also rejected.

        Regression: the prior validator stripped slashes before whitespace, so
        ``" / "`` would strip to ``" / "`` then ``"/"`` and pass. Stripping
        whitespace first closes the gap.
        """
        for value in (" / ", "\t//\n", "  ///  "):
            with pytest.raises(
                ValidationError, match=r"r2\.prefix_root must not be blank or slash-only"
            ):
                R2Location(bucket="b", prefix_root=value, prefix="data/x/y/")

    def test_slash_wrapped_whitespace_prefix_root_raises(self) -> None:
        """Values that are effectively whitespace once slashes are stripped are rejected.

        Regression: a value like ``"/ /"`` survives ``str.strip()`` (no surrounding
        whitespace) and ``str.strip("/")`` leaves a lone space (truthy), so the
        prior validator passed it through. Stripping whitespace again after the
        slash strip catches the case.
        """
        for value in ("/ /", "/\t/", "// //"):
            with pytest.raises(
                ValidationError, match=r"r2\.prefix_root must not be blank or slash-only"
            ):
                R2Location(bucket="b", prefix_root=value, prefix="data/x/y/")

    def test_prefix_root_strips_surrounding_whitespace(self) -> None:
        """Leading/trailing whitespace is stripped from the stored value.

        Regression: a value like ``" data "`` would otherwise survive into
        ``make_r2_prefix`` (which only strips slashes, not whitespace) and
        produce a malformed prefix like ``" data /task/run/"``.
        """
        loc = R2Location(bucket="b", prefix_root="  data  ", prefix="data/x/y/")
        assert loc.prefix_root == "data"

    def test_prefix_must_end_with_slash(self) -> None:
        """Prefix without trailing ``/`` would concatenate to ``.../prefixfilename``."""
        with pytest.raises(ValidationError, match=r"r2\.prefix must end with"):
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


class TestR2LocationLayoutHelpers:
    """Per-object URI helpers for the canonical R2 layout under ``self.prefix``."""

    def test_input_spec_uri(self) -> None:
        """``input_spec_uri()`` returns ``<prefix>input_spec.json`` (flat today, see #385)."""
        loc = R2Location(bucket="intermediate-data", prefix="data/run/")
        assert loc.input_spec_uri() == "r2://intermediate-data/data/run/input_spec.json"

    def test_config_yaml_uri(self) -> None:
        """``config_yaml_uri()`` returns ``<prefix>config.yaml`` (flat today, see #385)."""
        loc = R2Location(bucket="intermediate-data", prefix="data/run/")
        assert loc.config_yaml_uri() == "r2://intermediate-data/data/run/config.yaml"

    def test_dataset_card_uri(self) -> None:
        """``dataset_card_uri()`` returns ``<prefix>dataset.json`` (flat today, see #385)."""
        loc = R2Location(bucket="intermediate-data", prefix="data/run/")
        assert loc.dataset_card_uri() == "r2://intermediate-data/data/run/dataset.json"

    def test_dataset_complete_marker_uri(self) -> None:
        """``dataset_complete_marker_uri()`` returns ``<prefix>dataset.complete``."""
        loc = R2Location(bucket="intermediate-data", prefix="data/run/")
        assert (
            loc.dataset_complete_marker_uri() == "r2://intermediate-data/data/run/dataset.complete"
        )

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_split_uri(self, split: str) -> None:  # noqa: DOC101,DOC103
        """``split_uri(<split>)`` returns ``<prefix><split>.h5`` for each split."""
        loc = R2Location(bucket="intermediate-data", prefix="data/run/")
        assert loc.split_uri(split) == f"r2://intermediate-data/data/run/{split}.h5"

    def test_stats_uri(self) -> None:
        """``stats_uri()`` returns ``<prefix>stats.npz``."""
        loc = R2Location(bucket="intermediate-data", prefix="data/run/")
        assert loc.stats_uri() == "r2://intermediate-data/data/run/stats.npz"

    def test_worker_staged_shard_uri(self) -> None:
        """Joins shard_id, worker_id, attempt_uuid, ext under metadata/workers/shards/."""
        loc = R2Location(bucket="intermediate-data", prefix="data/run/")
        assert loc.worker_staged_shard_uri(
            shard_id=7, worker_id="rank0", attempt_uuid="abc123", ext=".h5"
        ) == (
            "r2://intermediate-data/data/run/metadata/workers/shards/shard-000007/rank0-abc123.h5"
        )

    def test_worker_staged_shard_uri_tar_extension(self) -> None:
        """Tar shards round-trip through the same helper with ``ext=".tar"``."""
        loc = R2Location(bucket="b", prefix="p/")
        assert (
            loc.worker_staged_shard_uri(shard_id=0, worker_id="w", attempt_uuid="u", ext=".tar")
            == "r2://b/p/metadata/workers/shards/shard-000000/w-u.tar"
        )

    def test_worker_attempt_report_uri(self) -> None:
        """Joins worker_id and attempt_uuid under metadata/workers/attempts/.../report.json."""
        loc = R2Location(bucket="intermediate-data", prefix="data/run/")
        assert loc.worker_attempt_report_uri(worker_id="rank0", attempt_uuid="abc123") == (
            "r2://intermediate-data/data/run/metadata/workers/attempts/rank0-abc123/report.json"
        )

    def test_layout_helpers_share_uri_under_prefix(self) -> None:
        """All under-prefix helpers route through ``self.prefix`` (no flat-key escape).

        Smoke test: every layout helper's URI starts with ``r2://<bucket>/<prefix>``,
        so a future flat→nested migration that retargets ``_under_prefix`` updates
        all of them atomically.
        """
        loc = R2Location(bucket="b", prefix="p/")
        expected_root = f"r2://{loc.bucket}/{loc.prefix}"
        assert loc.input_spec_uri().startswith(expected_root)
        assert loc.config_yaml_uri().startswith(expected_root)
        assert loc.dataset_card_uri().startswith(expected_root)
        assert loc.dataset_complete_marker_uri().startswith(expected_root)
        assert loc.split_uri("train").startswith(expected_root)
        assert loc.stats_uri().startswith(expected_root)
        assert loc.worker_staged_shard_uri(
            shard_id=0, worker_id="w", attempt_uuid="u", ext=".h5"
        ).startswith(expected_root)
        assert loc.worker_attempt_report_uri(worker_id="w", attempt_uuid="u").startswith(
            expected_root
        )
