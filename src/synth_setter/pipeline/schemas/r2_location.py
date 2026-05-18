"""R2 location (bucket + prefix) and URI helpers.

``R2Location`` centralizes the bucket / prefix-root / prefix triple and the
``r2://`` URI + rclone-syntax prefix construction. Every call site that used to
glue these together with f-strings now goes through one method on the model,
so a future change to the URI shape (e.g. moving shards under
``metadata/workers/`` per docs/design/data-pipeline.md §6) lands in one place.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from synth_setter.pipeline.constants import R2_URI_SCHEME, RCLONE_REMOTE
from synth_setter.pipeline.schemas.prefix import DEFAULT_R2_PREFIX_ROOT

__all__ = ["R2Location"]


class R2Location(BaseModel):  # noqa: DOC601,DOC603
    """Cloudflare R2 location: ``bucket`` + ``prefix_root`` + ``prefix``.

    ``prefix`` is the full object-prefix under the bucket (ending with ``/``);
    ``prefix_root`` is its top-level segment, carried for round-trips so a
    materialized spec preserves the launcher's choice of root.

    Strict + frozen so the materialized artifact is immutable post-construction
    and the trust boundary at JSON-from-R2 stays tight.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    bucket: str = Field(
        description="Cloudflare R2 bucket name where shards and metadata are written."
    )
    prefix_root: str = Field(
        default=DEFAULT_R2_PREFIX_ROOT,
        description=(
            "Top-level prefix segment under the bucket; slashes are stripped by "
            "``make_r2_prefix``."
        ),
    )
    prefix: str = Field(
        description=(
            "Full R2 object prefix (``<root>/<task_name>/<run_id>/``); must end with ``/``."
        )
    )

    @field_validator("bucket")
    @classmethod
    def _bucket_must_not_be_blank(cls, value: str) -> str:  # noqa: DOC101,DOC103,DOC201,DOC203,DOC501,DOC503
        """Reject blank buckets so rclone never receives a malformed ``r2:/...`` destination."""
        if not value.strip():
            raise ValueError("bucket must not be blank")
        return value

    @field_validator("prefix_root")
    @classmethod
    def _prefix_root_must_not_be_blank(cls, value: str) -> str:  # noqa: DOC101,DOC103,DOC201,DOC203,DOC501,DOC503
        """Reject blank prefix roots so derived ``prefix`` doesn't start with a stray ``/``."""
        if not value.strip():
            raise ValueError("prefix_root must not be blank")
        return value

    @field_validator("prefix")
    @classmethod
    def _prefix_must_end_with_slash(cls, value: str) -> str:  # noqa: DOC101,DOC103,DOC201,DOC203,DOC501,DOC503
        """Reject prefixes lacking ``/`` so rclone never gets ".../prefixfilename"."""
        if not value.endswith("/"):
            raise ValueError(f"prefix must end with '/' (got: {value!r})")
        return value

    def uri(self, name: str) -> str:
        """Build the canonical ``r2://{bucket}/{prefix}{name}`` URI for one object.

        :param name: Object basename to append to the prefix (e.g. ``shard-000000.h5``).
        :returns: Fully-qualified R2 URI for the named object.
        :rtype: str
        """
        return f"{R2_URI_SCHEME}{self.bucket}/{self.prefix}{name}"

    def rclone_prefix(self) -> str:
        """Build the rclone-syntax destination prefix ``<remote>:{bucket}/{prefix}``.

        Used as the ``rclone copy`` destination directory — the trailing slash on
        ``prefix`` makes rclone preserve the source basename inside that prefix.

        :returns: rclone-syntax prefix (no trailing object name).
        :rtype: str
        """
        return f"{RCLONE_REMOTE}:{self.bucket}/{self.prefix}"
