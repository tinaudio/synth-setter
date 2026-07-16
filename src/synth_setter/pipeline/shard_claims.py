"""Lance-backed claim table distributing dataset shards across machines.

One tiny Lance table under the run prefix holds one row per logical shard
(``shard_id``, ``status``, ``owner``, ``lease_expiry_s``, ``attempts``,
``claim_gen``). A worker claims a row with a conditional ``UPDATE`` whose
predicate is the compare-and-set: Lance re-evaluates the predicate against
the latest version when a commit conflicts, so a lost race updates zero rows
and the winner is whoever the post-update read shows as ``owner``. There is
no heartbeat — a single lease expiry written at claim time makes a crashed
worker's claim reclaimable once the lease lapses, and ``claim_gen`` fences a
stale owner's ``complete`` after a takeover. Shard completion truth stays
with the per-shard R2 existence probe; this table only routes work.

Every claim and complete commits a new table version (manifest + fragment
churn) — cosmetic for a table this small and disposable, and claims are
minutes apart per worker. Renders longer than the lease may be double-
rendered by a reclaiming peer; idempotent output keys keep that safe.

Typical usage::

    claims = ShardClaims.for_run(*lance_target(spec.r2.shard_claims_uri()))
    claims.populate(shard.shard_id for shard in spec.shards)  # operator, once
    while (claimed := claims.claim()) is not None:
        render(claimed.shard_id)
        claims.complete(claimed)
"""

from __future__ import annotations

import dataclasses
import os
import re
import secrets
import socket
import time
import uuid
from collections import Counter
from collections.abc import Iterable
from datetime import timedelta
from typing import TYPE_CHECKING, Final

import structlog

if TYPE_CHECKING:
    import lance
    import pyarrow as pa

_logger = structlog.get_logger(__name__)

CLAIM_LEASE: Final = timedelta(hours=2)

_STATUS_AVAILABLE: Final = "available"
_STATUS_CLAIMED: Final = "claimed"
_STATUS_DONE: Final = "done"

# Lance raises both cases as plain OSError (no structured type), so they are
# matched as substrings; the storm-injection tests pin the messages.
_CONTENTION_ERROR: Final = "Too many concurrent writers"
_ALREADY_EXISTS_ERROR: Final = "Dataset already exists"
# Flat jitter suffices: claims are minutes apart in production, so storms are
# short-lived and exponential growth would only delay recovery.
_CONTENTION_BACKOFF_RANGE_S: Final = (0.05, 0.25)

# Jitter/shard picks are load-spreading, not security; SystemRandom only
# because bandit (S311) flags the ``random`` module.
_rng: Final = secrets.SystemRandom()


def _claims_schema() -> pa.Schema:
    """Build the claim-row schema.

    :returns: Arrow schema with one row per logical shard.
    """
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("shard_id", pa.int64()),
            pa.field("status", pa.string()),
            pa.field("owner", pa.string()),
            pa.field("lease_expiry_s", pa.int64()),
            pa.field("attempts", pa.int64()),
            pa.field("claim_gen", pa.int64()),
        ]
    )


def _available_rows(shard_ids: list[int]) -> pa.Table:
    """Build unclaimed rows for ``shard_ids``.

    :param shard_ids: Logical shard IDs to seed.
    :returns: Table shaped per ``_claims_schema()`` of ``available`` rows with
        zeroed lease/attempt counters.
    """
    import pyarrow as pa

    count = len(shard_ids)
    return pa.table(
        {
            "shard_id": pa.array(shard_ids, pa.int64()),
            "status": pa.array([_STATUS_AVAILABLE] * count),
            "owner": pa.array([""] * count),
            "lease_expiry_s": pa.array([0] * count, pa.int64()),
            "attempts": pa.array([0] * count, pa.int64()),
            "claim_gen": pa.array([0] * count, pa.int64()),
        },
        schema=_claims_schema(),
    )


def _claimable_predicate(now_s: int) -> str:
    """Build the SQL predicate selecting rows a worker may take.

    :param now_s: Current epoch seconds; leases are compared as plain data, which is fine against
        worker clocks at hours-long leases.
    :returns: Predicate matching unclaimed rows and lapsed-lease claims.
    """
    return (
        f"status = '{_STATUS_AVAILABLE}' "
        f"OR (status = '{_STATUS_CLAIMED}' AND lease_expiry_s < {now_s})"
    )


def default_owner() -> str:
    """Build a claimant identity unique across machines and process restarts.

    :returns: ``<host>-<pid>-<uuid8>`` with the host reduced to SQL-literal-safe
        characters (the owner is embedded in update predicates).
    """
    host = "".join(c for c in socket.gethostname() if c.isalnum() or c in "-._")
    return f"{host}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


@dataclasses.dataclass(frozen=True)
class ClaimedShard:
    """One won claim: the shard to render plus its fencing token.

    .. attribute :: shard_id

        Logical shard ID to render (index into ``DatasetSpec.shards``).

    .. attribute :: claim_gen

        Generation this claim won; ``complete`` matches nothing if a peer
        has re-claimed the shard since (the lease lapsed).
    """

    shard_id: int
    claim_gen: int


@dataclasses.dataclass(frozen=True)
class ShardClaims:
    """Claim table over one Lance dataset shared by every worker in a run.

    .. attribute :: uri

        Lance dataset URI or local path, from ``r2_io.lance_target``.

    .. attribute :: storage_options

        Lance object-store options, or ``None`` on the local filesystem.

    .. attribute :: owner

        This claimant's identity, recorded on rows it wins.

    .. attribute :: lease

        How long a won claim holds off reclaiming peers. Must exceed the
        slowest expected render; there is no renewal.
    """

    uri: str
    storage_options: dict[str, str] | None
    owner: str
    lease: timedelta = CLAIM_LEASE

    def __post_init__(self) -> None:
        """Reject owners that could corrupt the SQL predicates fencing relies on.

        :raises ValueError: ``owner`` holds characters outside ``[A-Za-z0-9._-]``.
        """
        if not re.fullmatch(r"[A-Za-z0-9._-]+", self.owner):
            raise ValueError(f"owner {self.owner!r} must match [A-Za-z0-9._-]+")

    @classmethod
    def for_run(cls, uri: str, storage_options: dict[str, str] | None) -> ShardClaims:
        """Build a claims facade with a generated per-process owner identity.

        :param uri: Lance dataset URI or local path, from ``r2_io.lance_target``.
        :param storage_options: Lance object-store options, or ``None`` locally.
        :returns: Claims facade ready to populate or claim.
        """
        return cls(uri=uri, storage_options=storage_options, owner=default_owner())

    def _dataset(self) -> lance.LanceDataset:
        """Open the claims table at its latest version.

        Raises ``ValueError`` (from Lance) when the table does not exist —
        the operator never populated the run.

        :returns: Freshly resolved dataset handle.
        """
        import lance

        return lance.dataset(self.uri, storage_options=self.storage_options)

    def populate(self, shard_ids: Iterable[int]) -> int:
        """Ensure one claim row exists per requested shard ID (operator, once).

        Creating the table and inserting stragglers are both conditional Lance commits, so racing
        operators cannot clobber one another; rows that already exist keep their status, preserving
        in-flight claims across a relaunch (crashed claims recover via lease lapse, not re-
        seeding).

        :param shard_ids: Logical shard IDs the run renders; duplicates ignored.
        :returns: Number of missing rows inserted.
        :raises OSError: A storage failure other than the table already existing.
        """
        import lance

        ids = list(dict.fromkeys(shard_ids))
        rows = _available_rows(ids)
        try:
            lance.write_dataset(
                rows, self.uri, mode="create", storage_options=self.storage_options
            )
        except OSError as exc:
            if _ALREADY_EXISTS_ERROR not in str(exc):
                raise
            # Typically a relaunch: top up missing rows.
            merged = (
                self._dataset()
                .merge_insert("shard_id")
                .when_not_matched_insert_all()
                .execute(rows)
            )
            inserted = int(merged["num_inserted_rows"])
            _logger.info("populated_claims", inserted=inserted, requested=len(ids))
            return inserted
        _logger.info("created_claims_table", shards=len(ids), uri=self.uri)
        return len(ids)

    def claim(self) -> ClaimedShard | None:
        """Win one shard, or report the table drained.

        Picks a random claimable row (so a fleet doesn't race for one row;
        ``secrets.choice`` only because bandit flags ``random`` — security is
        irrelevant here),
        then applies the conditional update and re-reads the row to confirm
        ownership — a returned update alone is not proof against a lapsed
        lease being re-taken between commit and read.

        Raises ``RuntimeError`` (from the attempt) when the table breaks the
        one-row-per-shard invariant, and ``OSError`` on storage failures
        other than commit contention.

        :returns: The won claim, or ``None`` when nothing is claimable now
            (all rows done, or claimed under live leases).
        """
        while True:
            now_s = int(time.time())
            candidates = (
                self._dataset()
                .to_table(filter=_claimable_predicate(now_s), columns=["shard_id"])
                .column("shard_id")
                .to_pylist()
            )
            if not candidates:
                return None
            won = self._attempt_claim(_rng.choice(candidates), now_s)
            if won is not None:
                return won

    def _attempt_claim(self, shard_id: int, now_s: int) -> ClaimedShard | None:
        """Run one conditional-update attempt on ``shard_id`` and confirm ownership.

        :param shard_id: Candidate row to take.
        :param now_s: Epoch seconds shared with the candidate scan's predicate.
        :returns: The won claim, or ``None`` on any lost race (including a
            contention storm, absorbed with jittered backoff).
        :raises RuntimeError: The table breaks the one-row-per-shard invariant.
        :raises OSError: A storage failure other than commit contention.
        """
        try:
            updated = self._dataset().update(
                {
                    "status": f"'{_STATUS_CLAIMED}'",
                    "owner": f"'{self.owner}'",
                    "lease_expiry_s": str(now_s + int(self.lease.total_seconds())),
                    "attempts": "attempts + 1",
                    "claim_gen": "claim_gen + 1",
                },
                where=f"shard_id = {shard_id} AND ({_claimable_predicate(now_s)})",
            )
        except OSError as exc:
            if _CONTENTION_ERROR not in str(exc):
                raise
            # A whole fleet colliding can exhaust Lance's internal commit
            # retries; back off with jitter and re-scan instead of dying.
            _logger.info("claim_contention_backoff", shard_id=shard_id, owner=self.owner)
            time.sleep(_rng.uniform(*_CONTENTION_BACKOFF_RANGE_S))
            return None
        if updated["num_rows_updated"] == 0:
            return None
        rows = (
            self._dataset()
            .to_table(filter=f"shard_id = {shard_id}", columns=["owner", "claim_gen"])
            .to_pylist()
        )
        if updated["num_rows_updated"] > 1 or len(rows) != 1:
            raise RuntimeError(
                f"claims table holds {len(rows)} rows for shard_id {shard_id}; expected exactly 1"
            )
        if rows[0]["owner"] != self.owner:
            return None
        claim_gen = int(rows[0]["claim_gen"])
        _logger.info("claimed_shard", shard_id=shard_id, claim_gen=claim_gen, owner=self.owner)
        return ClaimedShard(shard_id=shard_id, claim_gen=claim_gen)

    def complete(self, claimed: ClaimedShard) -> bool:
        """Mark a won claim done, fenced against lapsed-lease takeovers.

        A lost fence is benign — the peer that re-claimed the shard also
        re-renders (or skip-probes) it and completes under its own
        generation — so callers only need the return value for logging. The
        same holds for a contention storm here: the row recovers via lease
        lapse and the durable output is skip-probed, so a rendered shard
        must never fail its run over this bookkeeping write.

        :param claimed: Claim previously won by this facade's ``owner``.
        :returns: ``False`` when a peer re-claimed the shard first, or the
            write was abandoned to contention.
        :raises OSError: A storage failure other than commit contention.
        """
        try:
            updated = self._dataset().update(
                {"status": f"'{_STATUS_DONE}'"},
                where=(
                    f"shard_id = {claimed.shard_id} AND owner = '{self.owner}' "
                    f"AND claim_gen = {claimed.claim_gen}"
                ),
            )
        except OSError as exc:
            if _CONTENTION_ERROR not in str(exc):
                raise
            _logger.warning(
                "abandoned_complete_to_contention",
                shard_id=claimed.shard_id,
                claim_gen=claimed.claim_gen,
                owner=self.owner,
            )
            return False
        if updated["num_rows_updated"] == 0:
            _logger.warning(
                "lost_claim_fence",
                shard_id=claimed.shard_id,
                claim_gen=claimed.claim_gen,
                owner=self.owner,
            )
            return False
        return True

    def status_counts(self) -> dict[str, int]:
        """Count rows per status — the run's whole progress story.

        :returns: Mapping of present statuses to row counts.
        """
        statuses = self._dataset().to_table(columns=["status"]).column("status").to_pylist()
        return dict(Counter(statuses))
