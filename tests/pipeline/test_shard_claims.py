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
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import pytest

from synth_setter.pipeline.shard_claims import (
    CLAIM_LEASE,
    ClaimedShard,
    ShardClaims,
    default_owner,
)


def _claims(tmp_path: Path, owner: str = "worker-a", **kwargs: object) -> ShardClaims:
    """Build a ``ShardClaims`` over a Lance table under ``tmp_path``.

    :param tmp_path: Directory receiving the ``shard-claims.lance`` dataset.
    :param owner: Claimant identity recorded on won rows.
    :param kwargs: Extra ``ShardClaims`` fields (e.g. ``lease``).
    :returns: Claims facade over the local-filesystem table.
    """
    return ShardClaims(
        uri=str(tmp_path / "shard-claims.lance"),
        storage_options=None,
        owner=owner,
        **kwargs,  # type: ignore[arg-type]
    )


class TestPopulate:
    """Operator-side seeding of the claims table."""

    def test_populate_new_table_seeds_every_shard_available(self, tmp_path: Path) -> None:
        claims = _claims(tmp_path)

        inserted = claims.populate(range(3))

        assert inserted == 3
        assert claims.status_counts() == {"available": 3}

    def test_populate_existing_table_is_idempotent(self, tmp_path: Path) -> None:
        claims = _claims(tmp_path)
        claims.populate(range(3))

        assert claims.populate(range(3)) == 0
        assert claims.status_counts() == {"available": 3}

    def test_populate_inserts_only_missing_shard_ids(self, tmp_path: Path) -> None:
        claims = _claims(tmp_path)
        claims.populate(range(2))

        assert claims.populate(range(4)) == 2
        assert claims.status_counts() == {"available": 4}

    def test_populate_preserves_in_flight_claims_on_relaunch(self, tmp_path: Path) -> None:
        claims = _claims(tmp_path)
        claims.populate(range(2))
        claimed = claims.claim()
        assert claimed is not None

        assert claims.populate(range(2)) == 0
        assert claims.status_counts() == {"available": 1, "claimed": 1}

    def test_populate_deduplicates_requested_ids(self, tmp_path: Path) -> None:
        claims = _claims(tmp_path)

        assert claims.populate([0, 1, 1, 0]) == 2
        assert claims.status_counts() == {"available": 2}


class TestClaimLifecycle:
    """Worker-side claim → complete flow."""

    def test_claim_returns_each_shard_exactly_once_until_drained(self, tmp_path: Path) -> None:
        claims = _claims(tmp_path)
        claims.populate(range(3))

        seen = []
        while (claimed := claims.claim()) is not None:
            seen.append(claimed.shard_id)

        assert sorted(seen) == [0, 1, 2]

    def test_claim_on_drained_table_returns_none(self, tmp_path: Path) -> None:
        claims = _claims(tmp_path)
        claims.populate(range(1))
        claimed = claims.claim()
        assert claimed is not None
        claims.complete(claimed)

        assert claims.claim() is None

    def test_claim_records_owner_lease_and_first_generation(self, tmp_path: Path) -> None:
        claims = _claims(tmp_path, owner="worker-a")
        claims.populate(range(1))

        claimed = claims.claim()

        assert claimed == ClaimedShard(shard_id=0, claim_gen=1)
        assert claims.status_counts() == {"claimed": 1}

    def test_claim_on_missing_table_raises(self, tmp_path: Path) -> None:
        claims = _claims(tmp_path)

        with pytest.raises(ValueError, match="not found"):
            claims.claim()

    def test_complete_marks_shard_done(self, tmp_path: Path) -> None:
        claims = _claims(tmp_path)
        claims.populate(range(2))
        claimed = claims.claim()
        assert claimed is not None

        assert claims.complete(claimed) is True
        assert claims.status_counts() == {"available": 1, "done": 1}

    def test_live_claim_is_not_reclaimable_by_peer(self, tmp_path: Path) -> None:
        claims_a = _claims(tmp_path, owner="worker-a")
        claims_a.populate(range(1))
        assert claims_a.claim() is not None

        claims_b = _claims(tmp_path, owner="worker-b")

        assert claims_b.claim() is None


class TestLeaseExpiryAndFencing:
    """Crashed-worker recovery via lease lapse, and stale-owner fencing."""

    def test_expired_lease_is_reclaimed_with_advanced_generation(self, tmp_path: Path) -> None:
        crashed = _claims(tmp_path, owner="worker-a", lease=timedelta(seconds=-1))
        crashed.populate(range(1))
        assert crashed.claim() == ClaimedShard(shard_id=0, claim_gen=1)

        successor = _claims(tmp_path, owner="worker-b")

        assert successor.claim() == ClaimedShard(shard_id=0, claim_gen=2)

    def test_stale_owner_complete_after_takeover_is_fenced_out(self, tmp_path: Path) -> None:
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
        assert CLAIM_LEASE == timedelta(hours=2)


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
    wins = []
    while (claimed := claims.claim()) is not None:
        wins.append((claimed.shard_id, claimed.claim_gen))
        claims.complete(claimed)
    out.put(wins)


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
    wins = []
    for _ in range(12):
        claimed = claims.claim()
        if claimed is not None:
            wins.append((claimed.shard_id, claimed.claim_gen))
    out.put(wins)


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
        return results

    def test_concurrent_drain_renders_every_shard_exactly_once(self, tmp_path: Path) -> None:
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
