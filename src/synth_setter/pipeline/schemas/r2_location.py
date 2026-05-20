"""``R2Location`` Pydantic model — bucket / prefix / prefix_root + URI helpers.

Replaces the three flat ``r2_bucket`` / ``r2_prefix_root`` / ``r2_prefix`` fields
that previously lived on ``DatasetSpec``. Centralizes URI construction so the
worker upload, the launcher spec upload, and the CI shard validator agree on
one URI shape (see ``r2_io.shard_uri`` for the historical helper). The methods
on this model are the only sanctioned way to build R2 URIs for the dataset
layout — callers must not concatenate ``self.prefix`` with filename literals.

The canonical R2 object layout (paths, filenames, and the flat → nested
migration plan #385 / #406) is defined in
``docs/design/storage-provenance-spec.md`` §2 + §3a — that doc is the
authoritative source. The per-helper docstrings below name the specific
file each method targets and the helper API stays stable across the
flat → nested migration.

``prefix`` is required. When constructed via ``DatasetSpec`` it is
auto-derived and the legacy flat ``{r2_bucket, r2_prefix_root, r2_prefix}``
input shape is promoted to ``r2: R2Location`` — see ``DatasetSpec``'s
``_normalize_r2_input`` model_validator and ``_default_r2_location`` /
``_fill_default_r2_prefix`` in ``spec.py`` for the mechanics. Constructing
``R2Location`` directly bypasses both shims; callers must supply ``prefix``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from synth_setter.pipeline.constants import (
    DATASET_COMPLETE_FILENAME,
    INPUT_SPEC_FILENAME,
    R2_URI_SCHEME,
    RCLONE_REMOTE,
    STATS_NPZ_FILENAME,
)
from synth_setter.pipeline.schemas.prefix import DEFAULT_R2_PREFIX_ROOT

if TYPE_CHECKING:
    from synth_setter.pipeline.schemas.spec import ShardSpec, Split

__all__ = ["R2Location"]


class R2Location(BaseModel):
    """R2 storage location: bucket + prefix_root + materialized prefix.

    Strict + frozen at the trust boundary — the same JSON-from-R2 round-trip
    contract that ``DatasetSpec`` honors. Field validators reject blanks and
    enforce the ``prefix`` trailing slash so rclone never receives
    ``r2:bucket/prefixshardname`` (concatenation trap).

    .. attribute :: model_config

        Pydantic model config sentinel — see ``ConfigDict(...)`` below for active settings.

    .. attribute :: bucket

        Cloudflare R2 bucket name where shards and metadata are written.

    .. attribute :: prefix_root

        Top-level prefix segment under the bucket.

    .. attribute :: prefix

        Full R2 object prefix (``<root>/<task_name>/<run_id>/``); must end with ``/``.
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
    def _bucket_must_not_be_blank(cls, value: str) -> str:
        """Reject blank buckets so rclone never receives a malformed ``r2:/...`` destination.

        :param value: Candidate ``bucket`` value pre-validation.
        :return: ``value`` unchanged when non-blank.
        :raises ValueError: ``value`` is blank/whitespace-only.
        """
        if not value.strip():
            raise ValueError("r2.bucket must not be blank")
        return value

    @field_validator("prefix_root")
    @classmethod
    def _prefix_root_must_not_be_blank(cls, value: str) -> str:
        """Reject blank/slash-only prefix roots; normalize leading/trailing whitespace.

        ``make_r2_prefix`` strips leading/trailing ``/`` and then rejects an empty
        result, so a value like ``"////"`` would otherwise survive this validator
        and crash later inside the prefix factory. Surrounding whitespace is
        stripped first, then slashes are stripped, then any remaining whitespace
        is stripped — catching ``" / "`` (whitespace-wrapped slash) AND ``"/ /"``
        (slashes-wrapping-whitespace), both of which would otherwise survive a
        single-pass strip and produce a malformed derived prefix. The stored
        value is the whitespace-stripped form (slashes preserved for
        ``make_r2_prefix`` to handle), so a caller passing ``" data "`` doesn't
        end up with ``" data /task/run/"`` as the derived prefix. Catching this
        here keeps error attribution at the ``r2.prefix_root`` boundary.

        :param value: Candidate ``prefix_root`` value pre-validation.
        :return: ``value`` with surrounding whitespace stripped.
        :raises ValueError: ``value`` is blank or slash-only after stripping.
        """
        stripped = value.strip()
        if not stripped.strip("/").strip():
            raise ValueError("r2.prefix_root must not be blank or slash-only")
        return stripped

    @field_validator("prefix")
    @classmethod
    def _prefix_must_end_with_slash(cls, value: str) -> str:
        """Reject prefixes lacking a trailing ``/`` so rclone never gets ".../prefixfilename".

        :param value: Candidate ``prefix`` value pre-validation.
        :return: ``value`` unchanged when it ends with ``/``.
        :raises ValueError: ``value`` does not end with ``/``.
        """
        if not value.endswith("/"):
            raise ValueError(f"r2.prefix must end with '/' (got: {value!r})")
        return value

    def uri(self, key: str) -> str:
        """Return the canonical ``r2://<bucket>/<key>`` URI for the given object key.

        ``key`` is an absolute object key under the bucket, **not** relative to
        ``self.prefix``. Use :meth:`shard_uri` for keys inside ``self.prefix``.

        :param key: Absolute object key (e.g. ``data/cfg-1/run-9/input_spec.json``).
        :returns: ``r2://<bucket>/<key>`` URI string.
        """
        return f"{R2_URI_SCHEME}{self.bucket}/{key}"

    def rclone_prefix(self) -> str:
        """Return the rclone-form destination ``r2:<bucket>/<prefix>`` for ``rclone copy``.

        rclone's CLI takes ``r2:bucket/key`` (no ``//``); see
        ``r2_io.to_rclone_path`` for the URI→rclone-form translator.

        :returns: ``r2:<bucket>/<prefix>`` string for use as an rclone destination.
        """
        return f"{RCLONE_REMOTE}:{self.bucket}/{self.prefix}"

    def _under_prefix(self, name: str) -> str:
        """Build ``r2://<bucket>/<prefix><name>`` — the canonical under-prefix URI shape.

        :param name: Relative key under ``self.prefix`` (may contain ``/``).
        :returns: ``r2://<bucket>/<prefix><name>`` URI string.
        """
        return f"{R2_URI_SCHEME}{self.bucket}/{self.prefix}{name}"

    def shard_uri(self, shard: ShardSpec) -> str:
        """Return the canonical R2 URI for ``shard``: ``r2://<bucket>/<prefix><filename>``.

        :param shard: A ``ShardSpec`` whose ``filename`` lives directly under
            this location's ``prefix``. Future state (#406) will relocate
            shards under a ``shards/`` subdirectory; this helper's API stays
            stable across that migration.
        :returns: ``r2://<bucket>/<prefix><shard.filename>`` URI string.
        """
        return self._under_prefix(shard.filename)

    def input_spec_uri(self) -> str:
        """R2 URI of the frozen ``input_spec.json`` (the materialized ``DatasetSpec``).

        Currently lives flat at ``<prefix>input_spec.json``; future state (#385)
        relocates it under ``<prefix>metadata/input_spec.json``.

        :returns: ``r2://<bucket>/<prefix>input_spec.json`` URI string.
        """
        return self._under_prefix(INPUT_SPEC_FILENAME)

    def config_yaml_uri(self) -> str:
        """R2 URI of the frozen Hydra-pipeline-config provenance copy (``config.yaml``).

        Currently flat; future state (#385) places it under ``metadata/``.

        :returns: ``r2://<bucket>/<prefix>config.yaml`` URI string.
        """
        return self._under_prefix("config.yaml")

    def dataset_card_uri(self) -> str:
        """R2 URI of the self-describing dataset card (``dataset.json``, planned — #74).

        Currently flat; future state (#385) places it under ``metadata/``.

        :returns: ``r2://<bucket>/<prefix>dataset.json`` URI string.
        """
        return self._under_prefix("dataset.json")

    def dataset_complete_marker_uri(self) -> str:
        """R2 URI of the ``dataset.complete`` completion marker (written last by finalize).

        Currently flat; future state (#385) places it under ``metadata/``.

        :returns: ``r2://<bucket>/<prefix>dataset.complete`` URI string.
        """
        return self._under_prefix(DATASET_COMPLETE_FILENAME)

    def split_h5_uri(self, split: Split) -> str:
        """R2 URI of a split virtual-dataset file (``train.h5`` / ``val.h5`` / ``test.h5``).

        Reshard produces these locally today; the URI is where they land once
        finalize uploads them (#408). Paired with :meth:`split_wds_brace_uri`
        for the wds variant; callers branch on ``DatasetSpec.output_format``.

        :param split: Split name; ``Literal["train","val","test"]`` (see
            ``synth_setter.pipeline.schemas.spec.Split``).
        :returns: ``r2://<bucket>/<prefix><split>.h5`` URI string.
        """
        return self._under_prefix(f"{split}.h5")

    def split_wds_brace_uri(self, shard_range: tuple[int, int]) -> str:
        """R2 URI carrying the webdataset brace pattern for ``[lo, hi)`` shards.

        WebDataset readers expand the ``{LO..HI}`` form natively; ``HI`` here
        is inclusive (``shard_range[1] - 1``) per the
        ``webdataset.WebDataset`` contract.

        :param shard_range: Half-open shard-index range, typically from
            ``DatasetSpec.split_shard_ranges[split]``.
        :returns: ``r2://<bucket>/<prefix>shard-{LO..HI}.tar`` with zero-padded
            six-digit indices matching ``ShardSpec.filename``'s format.
        :raises ValueError: ``shard_range`` is empty (``lo >= hi``) — would
            emit a malformed brace like ``{000003..000002}`` that wds reads
            as an empty set instead of raising.
        """
        lo, hi = shard_range
        if lo >= hi:
            raise ValueError(
                f"split_wds_brace_uri requires lo < hi (got {shard_range!r}); "
                f"an empty shard range has no brace pattern."
            )
        return self._under_prefix(f"shard-{{{lo:06d}..{hi - 1:06d}}}.tar")

    def stats_uri(self) -> str:
        """R2 URI of ``stats.npz`` (normalization statistics).

        Stats are written locally today; the URI is where finalize uploads
        them (#408).

        :returns: ``r2://<bucket>/<prefix>stats.npz`` URI string.
        """
        return self._under_prefix(STATS_NPZ_FILENAME)

    def worker_staged_shard_uri(
        self,
        shard_id: int,
        worker_id: str,
        attempt_uuid: str,
        ext: str,
    ) -> str:
        """R2 URI of a per-attempt staged shard under ``metadata/workers/shards/``.

        Future state (#406): workers upload each shard attempt here before
        finalize promotes one canonical copy to the run prefix root.
        No current consumers; included so the staging code path can call
        through this method when #406 lands.

        :param shard_id: Logical shard id (rendered as ``shard-NNNNNN`` directory).
        :param worker_id: Worker identifier issued by the launcher.
        :param attempt_uuid: Per-attempt UUID distinguishing retries.
        :param ext: File extension with leading dot (``".h5"`` or ``".tar"``).
        :returns: ``r2://<bucket>/<prefix>metadata/workers/shards/shard-NNNNNN/<worker>-<attempt>.<ext>``.
        """
        return self._under_prefix(
            f"metadata/workers/shards/shard-{shard_id:06d}/{worker_id}-{attempt_uuid}{ext}"
        )

    def worker_attempt_report_uri(self, worker_id: str, attempt_uuid: str) -> str:
        """R2 URI of a per-attempt worker report under ``metadata/workers/attempts/``.

        Future state (#406): workers write a ``report.json`` for each attempt
        here; finalize reads them to reconcile per-shard status. No current
        consumers.

        :param worker_id: Worker identifier issued by the launcher.
        :param attempt_uuid: Per-attempt UUID distinguishing retries.
        :returns: ``r2://<bucket>/<prefix>metadata/workers/attempts/<worker>-<attempt>/report.json``.
        """
        return self._under_prefix(
            f"metadata/workers/attempts/{worker_id}-{attempt_uuid}/report.json"
        )
