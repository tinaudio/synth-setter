"""Tests for pipeline.partitioning.

``get_my_shards`` and ``validate_rank_world`` are pure — tests pass
``rank`` / ``world`` / ``total_shards`` as direct arguments. ``read_rank_world_from_env``
is the imperative shell that reads ``SYNTH_SETTER_WORKER_RANK`` /
``SYNTH_SETTER_NUM_WORKERS`` from the environment; tests inject env
via ``monkeypatch.setenv`` / ``monkeypatch.delenv``.
"""

from __future__ import annotations

import pytest

from src.pipeline.partitioning import (
    get_my_shards,
    read_rank_world_from_env,
    validate_rank_world,
)


class TestSingleWorker:
    """World=1 → the lone worker owns every shard."""

    def test_returns_full_range_for_any_total(self) -> None:
        """Rank=0, world=1 → owns range(0, total) regardless of total."""
        assert get_my_shards(10, rank=0, world=1) == range(0, 10)

    def test_zero_total_shards_returns_empty_range(self) -> None:
        """total_shards=0 → returns empty range, not an error."""
        assert get_my_shards(0, rank=0, world=1) == range(0, 0)


class TestEvenDivide:
    """total_shards % world == 0 → equal-sized contiguous blocks per rank."""

    @pytest.mark.parametrize(
        ("rank", "expected"),
        [
            (0, range(0, 2)),
            (1, range(2, 4)),
            (2, range(4, 6)),
            (3, range(6, 8)),
        ],
    )
    def test_each_rank_owns_two_consecutive_shards(self, rank: int, expected: range) -> None:
        """8 shards / 4 workers → each rank owns 2 consecutive shards."""
        assert get_my_shards(8, rank=rank, world=4) == expected


class TestUnevenDivide:
    """First (total % world) ranks get one extra shard so imbalance ≤ 1."""

    @pytest.mark.parametrize(
        ("rank", "expected"),
        [
            (0, range(0, 3)),
            (1, range(3, 6)),
            (2, range(6, 8)),
            (3, range(8, 10)),
        ],
    )
    def test_first_two_ranks_get_three_shards_remaining_get_two(
        self, rank: int, expected: range
    ) -> None:
        """10 shards / 4 workers → ranks 0,1 own 3 shards; ranks 2,3 own 2."""
        assert get_my_shards(10, rank=rank, world=4) == expected

    def test_partition_covers_all_shards_with_no_overlap(self) -> None:
        """Union of every rank's range == range(0, total); ranges are pairwise disjoint."""
        owned: list[int] = []
        for rank in range(4):
            owned.extend(get_my_shards(10, rank=rank, world=4))
        assert sorted(owned) == list(range(10))


class TestMoreWorkersThanShards:
    """World > total → first `total` ranks own one shard each, rest empty."""

    def test_first_two_ranks_own_one_shard_each(self) -> None:
        """2 shards / 4 workers → ranks 0,1 each get one shard."""
        assert get_my_shards(2, rank=0, world=4) == range(0, 1)
        assert get_my_shards(2, rank=1, world=4) == range(1, 2)

    def test_excess_ranks_own_empty_range(self) -> None:
        """2 shards / 4 workers → ranks 2,3 own empty ranges and exit cleanly."""
        assert len(get_my_shards(2, rank=2, world=4)) == 0
        assert len(get_my_shards(2, rank=3, world=4)) == 0


class TestValidation:
    """Misconfiguration must fail fast, not silently render the wrong slice."""

    @pytest.mark.parametrize(
        ("rank", "world"),
        [
            (-1, 2),  # negative rank
            (2, 2),  # rank == world (out of range)
            (3, 2),  # rank > world
            (0, 0),  # world < 1
        ],
    )
    def test_get_my_shards_rejects_invalid_rank_world(self, rank: int, world: int) -> None:
        """Out-of-bounds rank/world combinations raise ValueError naming the values."""
        with pytest.raises(ValueError, match=f"rank={rank}"):
            get_my_shards(10, rank=rank, world=world)

    def test_get_my_shards_rejects_negative_total_shards(self) -> None:
        """Negative total_shards raises ValueError before the partition math runs.

        Without this guard, divmod() produces nonsense (negative base, nonzero remainder) and
        ranges would be silently malformed.
        """
        with pytest.raises(ValueError, match="total_shards must be >= 0"):
            get_my_shards(-3, rank=0, world=4)

    def test_validate_rank_world_accepts_valid_inputs(self) -> None:
        """Valid rank/world pairs (rank=0/world=1, rank=3/world=4) pass without raising."""
        validate_rank_world(rank=0, world=1)
        validate_rank_world(rank=3, world=4)

    def test_validate_rank_world_rejects_negative_rank(self) -> None:
        """Negative rank raises ValueError with the offending rank in the message."""
        with pytest.raises(ValueError, match="rank=-1"):
            validate_rank_world(rank=-1, world=2)

    def test_validate_rank_world_rejects_rank_equal_to_world(self) -> None:
        """Rank == world is out-of-range and raises ValueError."""
        with pytest.raises(ValueError, match="rank=2"):
            validate_rank_world(rank=2, world=2)

    def test_validate_rank_world_rejects_zero_world(self) -> None:
        """World < 1 is invalid and raises ValueError naming the offending world."""
        with pytest.raises(ValueError, match="world=0"):
            validate_rank_world(rank=0, world=0)


class TestReadRankWorldFromEnv:
    """``read_rank_world_from_env`` reads SYNTH_SETTER_WORKER_RANK / SYNTH_SETTER_NUM_WORKERS, no
    defaults.

    The silent-default behavior is intentionally refused: a worker invoked
    with a multi-shard spec but no partition env would otherwise duplicate
    every shard across every node. Any missing/malformed/out-of-bounds env
    must raise — generate_dataset.run propagates the ValueError to the worker exit path.
    """

    @pytest.fixture(autouse=True)
    def _clear_partition_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Strip partition rank/world env vars from the test process for isolation."""
        monkeypatch.delenv("SYNTH_SETTER_WORKER_RANK", raising=False)
        monkeypatch.delenv("SYNTH_SETTER_NUM_WORKERS", raising=False)

    def test_returns_rank_world_tuple_when_env_valid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both vars set + valid → returns (rank, world) as ints."""
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "2")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "4")
        assert read_rank_world_from_env() == (2, 4)

    def test_both_missing_raises_with_both_names_in_message(self) -> None:
        """Both vars missing → ValueError naming both."""
        with pytest.raises(ValueError) as excinfo:
            read_rank_world_from_env()
        message = str(excinfo.value)
        assert "SYNTH_SETTER_WORKER_RANK" in message
        assert "SYNTH_SETTER_NUM_WORKERS" in message

    def test_only_rank_missing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only rank var missing → ValueError naming rank."""
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")
        with pytest.raises(ValueError, match="SYNTH_SETTER_WORKER_RANK"):
            read_rank_world_from_env()

    def test_only_world_missing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only world var missing → ValueError naming world."""
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
        with pytest.raises(ValueError, match="SYNTH_SETTER_NUM_WORKERS"):
            read_rank_world_from_env()

    def test_non_integer_rank_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-integer rank ('abc') → ValueError naming SYNTH_SETTER_WORKER_RANK."""
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "abc")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")
        with pytest.raises(ValueError, match="SYNTH_SETTER_WORKER_RANK"):
            read_rank_world_from_env()

    def test_non_integer_world_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-integer world ('xyz') → ValueError naming SYNTH_SETTER_NUM_WORKERS."""
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "xyz")
        with pytest.raises(ValueError, match="SYNTH_SETTER_NUM_WORKERS"):
            read_rank_world_from_env()

    def test_out_of_bounds_rank_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rank >= world is rejected by validate_rank_world."""
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "5")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "2")
        with pytest.raises(ValueError, match="rank=5"):
            read_rank_world_from_env()

    def test_world_zero_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """World=0 violates the world>=1 invariant."""
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "0")
        with pytest.raises(ValueError, match="world=0"):
            read_rank_world_from_env()
