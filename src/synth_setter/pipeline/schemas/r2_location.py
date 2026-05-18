"""``R2Location`` Pydantic model — bucket / prefix / prefix_root + URI helpers.

Replaces the three flat ``r2_bucket`` / ``r2_prefix_root`` / ``r2_prefix`` fields
that previously lived on ``DatasetSpec``. Centralizes URI construction so the
worker upload, the launcher spec upload, and the CI shard validator agree on
one URI shape (see ``r2_io.shard_uri`` for the historical helper). The methods
are the only sanctioned way to build an R2 URI from a bucket+prefix pair.

``prefix`` is a required field on this model — when the caller omits it,
``DatasetSpec``'s own ``model_validator(mode="before")`` derives it via the
same ``make_r2_prefix`` callers used through the prior
``DatasetSpec._default_r2_prefix`` factory and injects it into the nested
dict before ``R2Location`` validation runs. That same ``before`` validator
also promotes the flat-form ``{r2_bucket, r2_prefix_root, r2_prefix}`` input
dict into the nested ``r2: R2Location`` shape so JSON specs already written
to R2 continue to parse. Constructing ``R2Location`` directly bypasses both
shims — callers must supply ``prefix`` explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from synth_setter.pipeline.constants import R2_URI_SCHEME, RCLONE_REMOTE
from synth_setter.pipeline.schemas.prefix import DEFAULT_R2_PREFIX_ROOT

if TYPE_CHECKING:
    from synth_setter.pipeline.schemas.spec import ShardSpec

__all__ = ["R2Location"]


class R2Location(BaseModel):  # noqa: DOC601,DOC603
    """R2 storage location: bucket + prefix_root + materialized prefix.

    Strict + frozen at the trust boundary — the same JSON-from-R2 round-trip
    contract that ``DatasetSpec`` honors. Field validators reject blanks and
    enforce the ``prefix`` trailing slash so rclone never receives
    ``r2:bucket/prefixshardname`` (concatenation trap).
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    bucket: str = Field(
        description="Cloudflare R2 bucket name where shards and metadata are written."
    )
    prefix_root: str = Field(
        default=DEFAULT_R2_PREFIX_ROOT,
        description=(
            "Top-level prefix segment under the bucket; slashes stripped by ``make_r2_prefix``."
        ),
    )
    prefix: str = Field(
        description="Full R2 object prefix (``<root>/<task_name>/<run_id>/``); must end with ``/``.",
    )

    @field_validator("bucket")
    @classmethod
    def _bucket_must_not_be_blank(cls, value: str) -> str:  # noqa: DOC101,DOC103,DOC201,DOC203,DOC501,DOC503
        """Reject blank buckets so rclone never receives a malformed ``r2:/...`` destination."""
        if not value.strip():
            raise ValueError("r2_bucket must not be blank")
        return value

    @field_validator("prefix_root")
    @classmethod
    def _prefix_root_must_not_be_blank(cls, value: str) -> str:  # noqa: DOC101,DOC103,DOC201,DOC203,DOC501,DOC503
        """Reject blank prefix roots so derived ``prefix`` doesn't start with a stray ``/``."""
        if not value.strip():
            raise ValueError("r2_prefix_root must not be blank")
        return value

    @field_validator("prefix")
    @classmethod
    def _prefix_must_end_with_slash(cls, value: str) -> str:  # noqa: DOC101,DOC103,DOC201,DOC203,DOC501,DOC503
        """Reject prefixes lacking a trailing ``/`` so rclone never gets ".../prefixfilename"."""
        if not value.endswith("/"):
            raise ValueError(f"r2_prefix must end with '/' (got: {value!r})")
        return value

    def uri(self, key: str) -> str:  # noqa: DOC203
        """Return the canonical ``r2://<bucket>/<key>`` URI for the given object key.

        ``key`` is an absolute object key under the bucket, **not** relative to
        ``self.prefix``. Use :meth:`shard_uri` for keys inside ``self.prefix``.

        :param key: Absolute object key (e.g. ``skypilot-launcher-specs/job-1.json``).
        :returns: ``r2://<bucket>/<key>`` URI string.
        """
        return f"{R2_URI_SCHEME}{self.bucket}/{key}"

    def rclone_prefix(self) -> str:  # noqa: DOC203
        """Return the rclone-form destination ``r2:<bucket>/<prefix>`` for ``rclone copy``.

        rclone's CLI takes ``r2:bucket/key`` (no ``//``); see
        ``r2_io.to_rclone_path`` for the URI→rclone-form translator.

        :returns: ``r2:<bucket>/<prefix>`` string for use as an rclone destination.
        """
        return f"{RCLONE_REMOTE}:{self.bucket}/{self.prefix}"

    def shard_uri(self, shard: ShardSpec) -> str:  # noqa: DOC203
        """Return the canonical R2 URI for ``shard``: ``r2://<bucket>/<prefix><filename>``.

        :param shard: A ``ShardSpec`` whose ``filename`` lives directly under
            this location's ``prefix``.
        :returns: ``r2://<bucket>/<prefix><shard.filename>`` URI string.
        """
        return f"{R2_URI_SCHEME}{self.bucket}/{self.prefix}{shard.filename}"
