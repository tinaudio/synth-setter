"""Behavioral tests for the Lance-backed shard-claims table.

Every test drives the real ``ShardClaims`` API over a real Lance dataset on
the local filesystem (no mocks, no fakes): local commits use the same
conditional-commit protocol Lance applies on object storage, so the
claim/steal/fence semantics exercised here are the ones production sees on
R2. Multi-machine contention is exercised with real OS processes in
``test_concurrent_claims_hammer_every_generation_has_one_owner``.
"""

from __future__ import annotations

import multiprocessing
import traceback
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from synth_setter.pipeline.shard_claims import (
    CLAIM_LEASE,
    ClaimedShard,
    ShardClaims,
    default_owner,
)


def _claims(
    tmp_path: Path, owner: str = "worker-a", lease: timedelta = CLAIM_LEASE
) -> ShardClaims:
    """Build a ``ShardClaims`` over a Lance table under ``tmp_path``.

    :param tmp_path: Directory receiving the ``shard-claims.lance`` dataset.
    :param owner: Claimant identity recorded on won rows.
    :param lease: How long a won claim holds off reclaiming peers.
    :returns: Claims facade over the local-filesystem table.
    """
    return ShardClaims(
        uri=str(tmp_path / "shard-claims.lance"),
        storage_options=None,
        owner=owner,
        lease=lease,
    )


class TestPopulate:
    """Operator-side seeding of the claims table."""

    def test_populate_new_table_seeds_every_shard_available(self, tmp_path: Path) -> None:
        """A fresh table holds one available row per requested shard.

        :param tmp_path: Hosts the per-test claims table.
        """
        claims = _claims(tmp_path)

        inserted = claims.populate(range(3))

        assert inserted == 3
        assert claims.status_counts() == {"available": 3}

    def test_populate_existing_table_is_idempotent(self, tmp_path: Path) -> None:
        """Re-populating the same IDs inserts nothing and changes nothing.

        :param tmp_path: Hosts the per-test claims table.
        """
        claims = _claims(tmp_path)
        claims.populate(range(3))

        assert claims.populate(range(3)) == 0
        assert claims.status_counts() == {"available": 3}

    def test_populate_inserts_only_missing_shard_ids(self, tmp_path: Path) -> None:
        """A grown request tops up only the rows not already present.

        :param tmp_path: Hosts the per-test claims table.
        """
        claims = _claims(tmp_path)
        claims.populate(range(2))

        assert claims.populate(range(4)) == 2
        assert claims.status_counts() == {"available": 4}

    def test_populate_preserves_in_flight_claims_on_relaunch(self, tmp_path: Path) -> None:
        """Relaunch re-population never resets a live claim.

        :param tmp_path: Hosts the per-test claims table.
        """
        claims = _claims(tmp_path)
        claims.populate(range(2))
        claimed = claims.claim()
        assert claimed is not None

        assert claims.populate(range(2)) == 0
        assert claims.status_counts() == {"available": 1, "claimed": 1}

    def test_populate_creates_table_with_pinned_claim_schema(self, tmp_path: Path) -> None:
        """The created table carries exactly the claim columns and dtypes callers rely on.

        :param tmp_path: Hosts the per-test claims table.
        """
        import lance
        import pyarrow as pa

        claims = _claims(tmp_path)
        claims.populate(range(2))

        expected = pa.schema(
            [
                pa.field("shard_id", pa.int64()),
                pa.field("status", pa.string()),
                pa.field("owner", pa.string()),
                pa.field("lease_expiry_s", pa.int64()),
                pa.field("attempts", pa.int64()),
                pa.field("claim_gen", pa.int64()),
            ]
        )
        assert lance.dataset(claims.uri).schema.equals(expected)

    def test_claim_and_complete_preserve_the_claim_schema(self, tmp_path: Path) -> None:
        """Conditional updates never alter the table's columns or dtypes.

        :param tmp_path: Hosts the per-test claims table.
        """
        import lance

        claims = _claims(tmp_path)
        claims.populate(range(1))
        created_schema = lance.dataset(claims.uri).schema
        claimed = claims.claim()
        assert claimed is not None
        claims.complete(claimed)

        assert lance.dataset(claims.uri).schema.equals(created_schema)

    def test_populate_creation_failure_other_than_exists_propagates(self, tmp_path: Path) -> None:
        """Only the dataset-exists signal routes to the merge path; real IO errors surface.

        :param tmp_path: Hosts the blocking plain file standing in for the table path.
        """
        blocker = tmp_path / "blocker"
        blocker.write_text("not a dataset directory")
        claims = ShardClaims(
            uri=str(blocker / "shard-claims.lance"), storage_options=None, owner="worker-a"
        )

        with pytest.raises(OSError, match="(?i)io error|walk dir"):
            claims.populate(range(1))

    def test_populate_deduplicates_requested_ids(self, tmp_path: Path) -> None:
        """Duplicate requested IDs seed a single row each.

        :param tmp_path: Hosts the per-test claims table.
        """
        claims = _claims(tmp_path)

        assert claims.populate([0, 1, 1, 0]) == 2
        assert claims.status_counts() == {"available": 2}


class TestClaimLifecycle:
    """Worker-side claim → complete flow."""

    def test_claim_returns_each_shard_exactly_once_until_drained(self, tmp_path: Path) -> None:
        """Sequential claims partition the table without repeats.

        :param tmp_path: Hosts the per-test claims table.
        """
        claims = _claims(tmp_path)
        claims.populate(range(3))

        seen = []
        while (claimed := claims.claim()) is not None:
            seen.append(claimed.shard_id)

        assert sorted(seen) == [0, 1, 2]

    def test_claim_on_drained_table_returns_none(self, tmp_path: Path) -> None:
        """A fully completed table yields no further claims.

        :param tmp_path: Hosts the per-test claims table.
        """
        claims = _claims(tmp_path)
        claims.populate(range(1))
        claimed = claims.claim()
        assert claimed is not None
        claims.complete(claimed)

        assert claims.claim() is None

    def test_claim_records_owner_lease_and_first_generation(self, tmp_path: Path) -> None:
        """A first claim wins generation 1 and marks the row claimed.

        :param tmp_path: Hosts the per-test claims table.
        """
        claims = _claims(tmp_path, owner="worker-a")
        claims.populate(range(1))

        claimed = claims.claim()

        assert claimed == ClaimedShard(shard_id=0, claim_gen=1)
        assert claims.status_counts() == {"claimed": 1}

    def test_claim_on_missing_table_raises(self, tmp_path: Path) -> None:
        """Claiming before the operator populated fails loudly.

        :param tmp_path: Hosts the per-test claims table.
        """
        claims = _claims(tmp_path)

        with pytest.raises(ValueError, match="not found"):
            claims.claim()

    def test_claim_on_corrupt_table_with_duplicate_rows_raises(self, tmp_path: Path) -> None:
        """A broken one-row-per-shard invariant must fail loudly, never claim silently.

        :param tmp_path: Hosts the per-test claims table.
        """
        import lance

        claims = _claims(tmp_path)
        claims.populate(range(1))
        duplicate = lance.dataset(claims.uri).to_table()
        lance.write_dataset(duplicate, claims.uri, mode="append")

        with pytest.raises(RuntimeError, match="expected exactly 1"):
            claims.claim()

    def test_complete_marks_shard_done(self, tmp_path: Path) -> None:
        """Completing a won claim flips its row to done.

        :param tmp_path: Hosts the per-test claims table.
        """
        claims = _claims(tmp_path)
        claims.populate(range(2))
        claimed = claims.claim()
        assert claimed is not None

        assert claims.complete(claimed) is True
        assert claims.status_counts() == {"available": 1, "done": 1}

    def test_live_claim_is_not_reclaimable_by_peer(self, tmp_path: Path) -> None:
        """A peer cannot steal a claim whose lease is still live.

        :param tmp_path: Hosts the per-test claims table.
        """
        claims_a = _claims(tmp_path, owner="worker-a")
        claims_a.populate(range(1))
        assert claims_a.claim() is not None

        claims_b = _claims(tmp_path, owner="worker-b")

        assert claims_b.claim() is None


class TestLeaseExpiryAndFencing:
    """Crashed-worker recovery via lease lapse, and stale-owner fencing."""

    def test_expired_lease_is_reclaimed_with_advanced_generation(self, tmp_path: Path) -> None:
        """A lapsed lease is stolen at the next fencing generation.

        :param tmp_path: Hosts the per-test claims table.
        """
        crashed = _claims(tmp_path, owner="worker-a", lease=timedelta(seconds=-1))
        crashed.populate(range(1))
        assert crashed.claim() == ClaimedShard(shard_id=0, claim_gen=1)

        successor = _claims(tmp_path, owner="worker-b")

        assert successor.claim() == ClaimedShard(shard_id=0, claim_gen=2)

    def test_stale_owner_complete_after_takeover_is_fenced_out(self, tmp_path: Path) -> None:
        """A stale owner's complete matches nothing after a takeover.

        :param tmp_path: Hosts the per-test claims table.
        """
        crashed = _claims(tmp_path, owner="worker-a", lease=timedelta(seconds=-1))
        crashed.populate(range(1))
        stale = crashed.claim()
        assert stale is not None

        successor = _claims(tmp_path, owner="worker-b")
        assert successor.claim() is not None

        assert crashed.complete(stale) is False
        assert crashed.status_counts() == {"claimed": 1}

    def test_complete_after_own_lease_expired_unclaimed_is_fenced_by_gen(
        self, tmp_path: Path
    ) -> None:
        """Self re-claim advances the generation and fences the older claim.

        :param tmp_path: Hosts the per-test claims table.
        """
        worker = _claims(tmp_path, owner="worker-a", lease=timedelta(seconds=-1))
        worker.populate(range(1))
        first = worker.claim()
        assert first is not None
        second = worker.claim()
        assert second is not None
        assert second == ClaimedShard(shard_id=0, claim_gen=2)

        assert worker.complete(first) is False
        assert worker.complete(second) is True

    def test_default_lease_covers_a_long_render(self) -> None:
        """The default lease outlasts the slowest expected render."""
        assert CLAIM_LEASE == timedelta(hours=2)

    def test_claim_survives_exhausted_commit_retries_and_retries_the_scan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A CAS storm exhausting Lance's internal retries must not kill the worker.

        First conditional update raises the real retry-exhaustion error
        (``OSError: Too many concurrent writers``, reproduced from two
        same-base-version handles); the claim loop backs off and wins on the
        next scan. Real storms are exercised by the concurrent hammer tests.

        :param tmp_path: Hosts the per-test claims table.
        :param monkeypatch: Injects the failing Lance update.
        """
        import lance

        claims = _claims(tmp_path)
        claims.populate(range(1))
        real_update = lance.LanceDataset.update
        storms = iter([True])

        def _stormy_update(self: lance.LanceDataset, *args: Any, **kwargs: Any) -> Any:
            if next(storms, False):
                raise OSError("Too many concurrent writers. Attempted 10 retries.")
            return real_update(self, *args, **kwargs)

        monkeypatch.setattr(lance.LanceDataset, "update", _stormy_update)

        assert claims.claim() == ClaimedShard(shard_id=0, claim_gen=1)

    def test_claim_reraises_unrelated_storage_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the contention error is retried; real storage failures propagate.

        :param tmp_path: Hosts the per-test claims table.
        :param monkeypatch: Injects the failing Lance update.
        """
        import lance

        claims = _claims(tmp_path)
        claims.populate(range(1))

        def _broken_update(self: lance.LanceDataset, *args: Any, **kwargs: Any) -> Any:
            raise OSError("Permission denied")

        monkeypatch.setattr(lance.LanceDataset, "update", _broken_update)

        with pytest.raises(OSError, match="Permission denied"):
            claims.claim()

    def test_complete_under_exhausted_commit_retries_abandons_without_raising(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A CAS storm at completion never fails a run that already rendered.

        The row recovers via lease lapse — the re-claiming peer skip-probes
        the durable output and completes it — so ``complete`` reports the
        abandonment instead of raising.

        :param tmp_path: Hosts the per-test claims table.
        :param monkeypatch: Injects the failing Lance update.
        """
        import lance

        claims = _claims(tmp_path)
        claims.populate(range(1))
        claimed = claims.claim()
        assert claimed is not None

        def _stormy_update(self: lance.LanceDataset, *args: Any, **kwargs: Any) -> Any:
            raise OSError("Too many concurrent writers. Attempted 10 retries.")

        monkeypatch.setattr(lance.LanceDataset, "update", _stormy_update)

        assert claims.complete(claimed) is False


def test_constructing_claims_with_sql_unsafe_owner_raises() -> None:
    """The dataclass boundary rejects owners that could corrupt update predicates."""
    with pytest.raises(ValueError, match="owner"):
        ShardClaims(uri="unused", storage_options=None, owner="bad'owner")


def test_default_owner_is_unique_and_sql_literal_safe() -> None:
    """Two owners never collide, and neither can break out of a SQL string."""
    first, second = default_owner(), default_owner()

    assert first != second
    for owner in (first, second):
        assert owner == "".join(c for c in owner if c.isalnum() or c in "-._")


def _drain_worker(uri: str, worker_index: int, out: multiprocessing.Queue) -> None:
    """Claim and complete shards until drained, reporting wins.

    Module-level so ``spawn`` can pickle it.

    :param uri: Shared local-filesystem claims table.
    :param worker_index: Distinguishes this worker's owner identity.
    :param out: Receives one ``[(shard_id, claim_gen), ...]`` list.
    """
    claims = ShardClaims(uri=uri, storage_options=None, owner=f"proc-{worker_index}")
    try:
        wins = []
        while (claimed := claims.claim()) is not None:
            wins.append((claimed.shard_id, claimed.claim_gen))
            claims.complete(claimed)
        out.put(wins)
    except BaseException:  # noqa: BLE001 — surfaced as the test's failure detail
        out.put(traceback.format_exc())


def _steal_worker(uri: str, worker_index: int, out: multiprocessing.Queue) -> None:
    """Hammer claims under an always-expired lease, reporting every win.

    Module-level so ``spawn`` can pickle it.

    :param uri: Shared local-filesystem claims table.
    :param worker_index: Distinguishes this worker's owner identity.
    :param out: Receives one ``[(shard_id, claim_gen), ...]`` list.
    """
    claims = ShardClaims(
        uri=uri,
        storage_options=None,
        owner=f"proc-{worker_index}",
        lease=timedelta(seconds=-5),
    )
    try:
        wins = []
        for _ in range(12):
            claimed = claims.claim()
            if claimed is not None:
                wins.append((claimed.shard_id, claimed.claim_gen))
        out.put(wins)
    except BaseException:  # noqa: BLE001 — surfaced as the test's failure detail
        out.put(traceback.format_exc())


class TestConcurrentClaims:
    """Real multi-process contention over one shared table."""

    def _run_workers(
        self,
        target: Callable[[str, int, multiprocessing.Queue], None],
        uri: str,
        count: int,
    ) -> list[list[tuple[int, int]]]:
        """Spawn ``count`` worker processes and collect their win lists.

        :param target: Module-level worker function to spawn.
        :param uri: Shared claims-table location passed to every worker.
        :param count: Number of concurrent OS processes.
        :returns: One win list per worker.
        """
        ctx = multiprocessing.get_context("spawn")
        out: multiprocessing.Queue = ctx.Queue()
        procs = [ctx.Process(target=target, args=(uri, index, out)) for index in range(count)]
        for proc in procs:
            proc.start()
        results = [out.get(timeout=120) for _ in procs]
        for proc in procs:
            proc.join(timeout=60)
        crashed = [r for r in results if isinstance(r, str)]
        assert not crashed, f"worker died: {crashed[0]}"
        return results

    def test_concurrent_drain_renders_every_shard_exactly_once(self, tmp_path: Path) -> None:
        """Racing workers partition a real shared table with no double-grant.

        :param tmp_path: Hosts the per-test claims table.
        """
        uri = str(tmp_path / "shard-claims.lance")
        ShardClaims(uri=uri, storage_options=None, owner="operator").populate(range(6))

        results = self._run_workers(_drain_worker, uri, count=4)

        all_wins = [win for wins in results for win in wins]
        assert sorted(shard_id for shard_id, _ in all_wins) == [0, 1, 2, 3, 4, 5]
        assert all(claim_gen == 1 for _, claim_gen in all_wins)
        counts = ShardClaims(uri=uri, storage_options=None, owner="operator").status_counts()
        assert counts == {"done": 6}

    def test_concurrent_claims_hammer_every_generation_has_one_owner(self, tmp_path: Path) -> None:
        """The CAS invariant under maximum steal pressure.

        Every claim is instantly reclaimable (negative lease), so all workers
        fight over two rows continuously. If Lance re-applied a conflicting
        update without re-evaluating its predicate, two workers would win the
        same ``(shard_id, claim_gen)`` or generations would be lost.

        :param tmp_path: Hosts the per-test claims table.
        """
        uri = str(tmp_path / "shard-claims.lance")
        ShardClaims(uri=uri, storage_options=None, owner="operator").populate(range(2))

        results = self._run_workers(_steal_worker, uri, count=4)

        all_wins = [win for wins in results for win in wins]
        assert len(set(all_wins)) == len(all_wins), "two workers won the same generation"
        import lance

        rows = lance.dataset(uri).to_table(columns=["shard_id", "claim_gen"]).to_pylist()
        final_generation_total = sum(row["claim_gen"] for row in rows)
        assert final_generation_total == len(all_wins), "a won generation was lost"
