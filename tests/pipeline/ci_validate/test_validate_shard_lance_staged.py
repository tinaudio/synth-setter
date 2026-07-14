"""Lance from-R2 validation over staged winner attempts (#1776).

Drives the public ``validate_all_shards_from_r2`` against real staged attempts
on the ``fake_r2_remote`` local remote — the same reconciliation the smoke
workflow runs between generate and finalize.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_setter.pipeline.ci.validate_shard import validate_all_shards_from_r2
from tests.pipeline.data.test_lance_finalize import stage_all_shards, staging_file
from tests.pipeline.data.test_lance_staging import tiny_lance_spec

pytestmark = pytest.mark.usefixtures("fake_r2_remote")


def test_validate_all_lance_shards_passes_when_every_shard_has_a_staged_winner(
    tmp_path: Path,
) -> None:
    """A fully staged run validates clean.

    :param tmp_path: Scratch dir for the local shard datasets.
    """
    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)

    assert validate_all_shards_from_r2(spec) == []


def test_validate_all_lance_shards_reports_shard_with_no_staged_attempt(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    """A shard whose ``.valid`` marker is gone reports as missing, prefixed by filename.

    :param fake_r2_remote: Root the ``r2:`` remote resolves to.
    :param tmp_path: Scratch dir for the local shard datasets.
    """
    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)
    staging_file(fake_r2_remote, spec, 1, "pod-a-u0001.valid").unlink()

    errors = validate_all_shards_from_r2(spec)

    assert len(errors) == 1
    assert errors[0].startswith("shard-000001.lance: no staged-valid attempt under ")


def test_validate_all_lance_shards_surfaces_structural_check_failures_per_shard(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    """One corrupt sidecar aggregates as that shard's error instead of crashing the run.

    :param fake_r2_remote: Root the ``r2:`` remote resolves to.
    :param tmp_path: Scratch dir for the local shard datasets.
    """
    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)
    sidecar = staging_file(fake_r2_remote, spec, 3, "pod-a-u0003.fragment.json")
    sidecar.write_text("{not json")

    errors = validate_all_shards_from_r2(spec)

    assert len(errors) == 1
    assert errors[0].startswith("shard-000003.lance: ")
    assert "invalid fragment sidecar" in errors[0]


def test_validate_all_lance_shards_aggregates_malformed_stats_sidecar(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    """A malformed stats archive reports against its shard instead of aborting validation.

    :param fake_r2_remote: Root the ``r2:`` remote resolves to.
    :param tmp_path: Scratch dir for the local shard datasets.
    """
    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)
    stats = staging_file(fake_r2_remote, spec, 2, "pod-a-u0002.shard-stats.npz")
    stats.write_bytes(b"")

    errors = validate_all_shards_from_r2(spec)

    assert len(errors) == 1
    assert errors[0].startswith("shard-000002.lance: shard 2 attempt pod-a-u0002: ")
    assert "invalid shard-stats.npz" in errors[0]
