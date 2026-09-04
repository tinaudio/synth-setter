"""Public ``synth-setter-growing-lance`` CLI flow over the ``fake_r2_remote``.

E2e over real Lance and the real ``rclone`` binary against the local ``r2:``
remote — the same public entrypoint operators run, with no mocks on the
storage or Lance paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import lance
import pytest

from synth_setter.cli.finalize_dataset import finalize_from_spec
from synth_setter.cli.growing_lance import main
from synth_setter.pipeline.data.growing_lance import GrowingPlan
from synth_setter.pipeline.data.lance_staging import stage_lance_shard_attempt
from synth_setter.pipeline.schemas.spec import DatasetSpec
from synth_setter.pipeline.spec_io import upload_spec
from tests.pipeline.data.test_lance_finalize import stage_all_shards
from tests.pipeline.data.test_lance_staging import tiny_lance_spec, write_local_shard

pytestmark = pytest.mark.usefixtures("fake_r2_remote")


@pytest.fixture(autouse=True)
def _skip_auth_ping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass the credential pre-flight; the fake remote needs no auth.

    :param monkeypatch: Pytest fixture used to stub the pre-flight.
    """
    monkeypatch.setattr("synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda *a, **k: None)


@pytest.fixture
def finalized_spec(fake_r2_remote: Path, tmp_path: Path) -> DatasetSpec:
    """Generate, stage, and finalize the tiny baseline dataset in the fake remote.

    :param fake_r2_remote: Root the ``r2:`` remote resolves to.
    :param tmp_path: Scratch dir for local shard datasets.
    :returns: The finalized spec, with ``input_spec.json`` uploaded.
    """
    spec = tiny_lance_spec()
    upload_spec(spec)
    stage_all_shards(spec, tmp_path / "baseline-shards")
    finalize_from_spec(spec, tmp_path / "finalize-work")
    return spec


def _train_dataset(fake_r2_remote: Path, spec: DatasetSpec) -> lance.LanceDataset:
    return lance.dataset(str(fake_r2_remote / spec.r2.bucket / spec.r2.prefix / "train.lance"))


def _remote_metadata_dir(fake_r2_remote: Path, spec: DatasetSpec, branch: str) -> Path:
    return fake_r2_remote / spec.r2.bucket / spec.r2.prefix / "metadata" / "growing" / branch


def _init(spec: DatasetSpec, work_dir: Path, *extra: str) -> None:
    main(
        [
            "init",
            spec.r2.input_spec_uri(),
            "--branch",
            "g",
            "--max-train-shards",
            "3",
            "--num-extra-shards",
            "1",
            "--work-dir",
            str(work_dir),
            *extra,
        ]
    )


def test_init_without_baseline_version_pins_the_finalized_train_version(
    fake_r2_remote: Path, finalized_spec: DatasetSpec, tmp_path: Path
) -> None:
    """Omitting ``--baseline-version`` resolves the finalized train dataset version.

    :param fake_r2_remote: Root the ``r2:`` remote resolves to.
    :param finalized_spec: Finalized baseline spec.
    :param tmp_path: Operator work dir root.
    """
    finalized_version = _train_dataset(fake_r2_remote, finalized_spec).version

    _init(finalized_spec, tmp_path / "operator")

    train = _train_dataset(fake_r2_remote, finalized_spec)
    ready_version = train.tags.get_version("g-ready")
    snapshot_path = (
        _remote_metadata_dir(fake_r2_remote, finalized_spec, "g")
        / "versions"
        / str(ready_version)
        / "snapshot.json"
    )
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["baseline_version"] == finalized_version
    assert snapshot["high_watermark"] == 2


def _grow(spec: DatasetSpec, work_dir: Path) -> None:
    main(
        [
            "grow",
            spec.r2.input_spec_uri(),
            "--branch",
            "g",
            "--work-dir",
            str(work_dir),
            "--poll-seconds",
            "0",
        ]
    )


def test_grow_single_pass_enqueues_and_returns_before_shards_are_staged(
    fake_r2_remote: Path, finalized_spec: DatasetSpec, tmp_path: Path
) -> None:
    """A grow pass with nothing staged persists the pending range and exits cleanly.

    :param fake_r2_remote: Root the ``r2:`` remote resolves to.
    :param finalized_spec: Finalized baseline spec.
    :param tmp_path: Operator work dir root.
    """
    _init(finalized_spec, tmp_path / "operator")
    ready_before = _train_dataset(fake_r2_remote, finalized_spec).tags.get_version("g-ready")

    _grow(finalized_spec, tmp_path / "operator")

    metadata_dir = _remote_metadata_dir(fake_r2_remote, finalized_spec, "g")
    pending = json.loads((metadata_dir / "pending.json").read_text(encoding="utf-8"))
    assert pending["enqueue_shard_ids"] == [2]
    ready_after = _train_dataset(fake_r2_remote, finalized_spec).tags.get_version("g-ready")
    assert ready_after == ready_before


def _stage_growing_shard(
    fake_r2_remote: Path, spec: DatasetSpec, shard_id: int, work_dir: Path
) -> None:
    """Stage one growing shard attempt exactly as a generator worker does.

    :param fake_r2_remote: Root the ``r2:`` remote resolves to.
    :param spec: Finalized baseline spec.
    :param shard_id: Direct train position to stage.
    :param work_dir: Local scratch directory for the shard dataset.
    """
    ready_version = _train_dataset(fake_r2_remote, spec).tags.get_version("g-ready")
    snapshot = json.loads(
        (
            _remote_metadata_dir(fake_r2_remote, spec, "g")
            / "versions"
            / str(ready_version)
            / "snapshot.json"
        ).read_text(encoding="utf-8")
    )
    plan = GrowingPlan(
        snapshot["baseline_train_shards"],
        snapshot["max_train_shards"],
        snapshot["num_extra_shards"],
        snapshot["high_watermark"],
    )
    shard = plan.extra_shard(spec, shard_id)
    local = write_local_shard(spec, shard_id, work_dir, shard=shard)
    stage_lance_shard_attempt(
        spec,
        shard,
        local,
        worker_id="grow-worker",
        attempt_uuid=f"u{shard_id:04d}",
        target_lance_uri=snapshot["branch_uri"],
        attempt_staging_dir_uri=spec.r2.growing_shard_staging_dir_uri("g", shard_id),
    )


def test_grow_finalizes_staged_range_then_stops_cleanly_at_capacity(
    fake_r2_remote: Path, finalized_spec: DatasetSpec, tmp_path: Path
) -> None:
    """Grow appends the staged range, clears the pending request, and halts at capacity.

    :param fake_r2_remote: Root the ``r2:`` remote resolves to.
    :param finalized_spec: Finalized baseline spec.
    :param tmp_path: Operator work dir root.
    """
    operator = tmp_path / "operator"
    _init(finalized_spec, operator)
    _grow(finalized_spec, operator)
    _stage_growing_shard(fake_r2_remote, finalized_spec, 2, tmp_path / "grow-shards")

    _grow(finalized_spec, operator)

    train = _train_dataset(fake_r2_remote, finalized_spec)
    ready_version = train.tags.get_version("g-ready")
    grown = train.checkout_version(("g", ready_version))
    assert grown.count_rows() == 6
    assert len(grown.get_fragments()) == 3
    metadata_dir = _remote_metadata_dir(fake_r2_remote, finalized_spec, "g")
    assert not (metadata_dir / "pending.json").exists()
    assert (metadata_dir / "completed" / f"{ready_version}.json").is_file()

    _grow(finalized_spec, operator)

    assert _train_dataset(fake_r2_remote, finalized_spec).tags.get_version("g-ready") == (
        ready_version
    )


def test_generate_before_first_enqueue_exits_cleanly_without_rendering(
    fake_r2_remote: Path, finalized_spec: DatasetSpec, tmp_path: Path
) -> None:
    """A generator that starts before the first enqueue waits instead of failing.

    :param fake_r2_remote: Root the ``r2:`` remote resolves to.
    :param finalized_spec: Finalized baseline spec.
    :param tmp_path: Operator work dir root.
    """
    _init(finalized_spec, tmp_path / "operator")

    main(
        [
            "generate",
            finalized_spec.r2.input_spec_uri(),
            "--branch",
            "g",
            "--work-dir",
            str(tmp_path / "generate"),
            "--poll-seconds",
            "0",
        ]
    )

    metadata_dir = _remote_metadata_dir(fake_r2_remote, finalized_spec, "g")
    assert not (metadata_dir / "workers").exists()


def _materialize(spec: DatasetSpec, local_root: Path, work_dir: Path) -> None:
    main(
        [
            "materialize",
            spec.r2.input_spec_uri(),
            "--branch",
            "g",
            "--local-root",
            str(local_root),
            "--work-dir",
            str(work_dir),
        ]
    )


def test_materialize_tick_with_current_active_record_needs_no_version_metadata(
    fake_r2_remote: Path, finalized_spec: DatasetSpec, tmp_path: Path
) -> None:
    """An up-to-date active record short-circuits the tick before any download.

    Deleting the remote version metadata proves the repeat tick never fetches
    it — a tick that re-downloaded would fail on the missing objects.

    :param fake_r2_remote: Root the ``r2:`` remote resolves to.
    :param finalized_spec: Finalized baseline spec.
    :param tmp_path: Operator work dir root.
    """
    _init(finalized_spec, tmp_path / "operator")
    local_root = tmp_path / "growing-local"
    _materialize(finalized_spec, local_root, tmp_path / "materializer")
    active_before = (local_root / "active.json").read_bytes()
    ready_version = _train_dataset(fake_r2_remote, finalized_spec).tags.get_version("g-ready")
    version_dir = (
        _remote_metadata_dir(fake_r2_remote, finalized_spec, "g") / "versions" / str(ready_version)
    )
    for name in ("snapshot.json", "stats.npz", "welford.npz"):
        (version_dir / name).unlink()

    _materialize(finalized_spec, local_root, tmp_path / "materializer")

    assert (local_root / "active.json").read_bytes() == active_before
