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
    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)

    assert validate_all_shards_from_r2(spec) == []


def test_validate_all_lance_shards_reports_shard_with_no_staged_attempt(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)
    staging_file(fake_r2_remote, spec, 1, "pod-a-u0001.valid").unlink()

    errors = validate_all_shards_from_r2(spec)

    assert len(errors) == 1
    assert errors[0].startswith("shard-000001.lance: no staged-valid attempt under ")


def test_validate_all_lance_shards_surfaces_structural_check_failures_per_shard(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)
    sidecar = staging_file(fake_r2_remote, spec, 3, "pod-a-u0003.fragment.json")
    sidecar.write_text("{not json")

    errors = validate_all_shards_from_r2(spec)

    assert len(errors) == 1
    assert errors[0].startswith("shard-000003.lance: ")
    assert "invalid fragment sidecar" in errors[0]
