"""Branch-level unit tests for ``synth_setter.cli.finalize_dataset``.

The entrypoint surface — ``finalize(cfg)`` / ``main()`` — is exercised in the
canonical ``tests/test_finalize_dataset.py`` module, which
``tests/_meta/test_entrypoint_test_modules.py`` guards to stay free of private
``synth_setter.cli`` references. This sibling module holds the per-branch tests
that drive ``finalize_lance`` / ``finalize_from_spec`` directly and legitimately
reference module internals to pin the marker-last ordering and the empty-train
guard at the branch altitude — so it is deliberately NOT on the entrypoint-only
rail.

``finalize_lance`` commits staged winner fragments and reads the staging prefix
over ``s3://`` object storage, which the local-typed ``fake_r2_remote`` can't
serve; the winner selection + fragment commit are stubbed via
``stub_finalize_lance_io`` so ``stats.npz`` + ``dataset.json`` +
``dataset.complete`` still route through the real ``r2_io.upload``. The full
fragment commit is covered against real R2 in
``tests/integration/test_finalize_dataset_r2.py``.

Seeders and smoke-spec builders shared with the entrypoint and real-R2 lanes
live in ``tests/helpers/finalize_shards.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import NoReturn
from unittest.mock import MagicMock

import pytest

from synth_setter.cli import finalize_dataset
from synth_setter.pipeline import r2_io
from tests.helpers.finalize_shards import (
    build_lance_smoke_spec,
    install_finalize_setup_stubs,
    stub_finalize_lance_io,
    uri_to_local_path,
)


@pytest.fixture()
def stub_finalize_setup(monkeypatch: pytest.MonkeyPatch) -> Callable[[int | None], None]:
    """Install the auth + marker-probe stubs so ``finalize_from_spec`` proceeds.

    Thin wrapper over :func:`tests.helpers.finalize_shards.install_finalize_setup_stubs`;
    the branch-level ``finalize_from_spec`` tests need the same stub set as the
    entrypoint lane because that path also probes the ``dataset.complete`` marker.

    :param monkeypatch: Pytest fixture used to install the stubs.
    :returns: A setter overriding the marker-probe response (``None`` proceeds;
        an ``int`` short-circuits).
    """
    return install_finalize_setup_stubs(monkeypatch)


def test_finalize_from_spec_uploads_stats_then_marker_at_canonical_uris(
    tmp_path: Path,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_finalize_setup: Callable[[int | None], None],  # noqa: ARG001 — installs stubs only
) -> None:
    """``finalize_from_spec`` honors the marker-last ordering without re-loading the spec.

    Mirrors ``test_finalize_uploads_stats_then_marker_at_canonical_uris`` but
    calls the in-memory entry point directly — no ``cfg`` synthesis, no
    ``load_spec_from_root`` round-trip — so the inline path
    (``generate_dataset.main`` reuses) is pinned independently of the URI-driven
    entry point. The Lance fragment commit is stubbed, so ``stats.npz``,
    ``dataset.json``, and the marker are the objects routed through ``r2_io.upload``.

    :param tmp_path: Hosts the Hydra-style work_dir.
    :param fake_r2_remote: Local-typed rclone remote; the artifacts land here.
    :param monkeypatch: Pytest fixture used to stub the Lance fragment I/O and
        wrap ``synth_setter.pipeline.r2_io.upload`` with an order-recording spy.
    :param stub_finalize_setup: Fixture-activation only — installs the
        ``ensure_r2_env_loaded`` / ``object_size`` stubs.
    """
    spec = build_lance_smoke_spec(task_name="finalize-from-spec-marker-last")
    stub_finalize_lance_io(monkeypatch)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    real_upload = r2_io.upload
    upload_order: list[str] = []

    def spy_upload(src: str | Path, dst: str) -> None:
        upload_order.append(dst)
        real_upload(src, dst)

    monkeypatch.setattr("synth_setter.pipeline.r2_io.upload", spy_upload)

    finalize_dataset.finalize_from_spec(spec, work_dir)

    stats_uri = spec.r2.stats_uri()
    marker_uri = spec.r2.dataset_complete_marker_uri()
    assert uri_to_local_path(fake_r2_remote, stats_uri).is_file()
    assert uri_to_local_path(fake_r2_remote, marker_uri).is_file()
    assert upload_order.count(stats_uri) == 1
    assert upload_order.count(marker_uri) == 1
    assert upload_order.index(marker_uri) == len(upload_order) - 1
    assert upload_order.index(stats_uri) < upload_order.index(marker_uri)


def test_finalize_from_spec_non_canonical_prefix_warns_and_proceeds(
    tmp_path: Path,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_finalize_setup: Callable[[int | None], None],  # noqa: ARG001 — installs stubs only
) -> None:
    """A custom (non-canonical) ``r2.prefix`` is finalized, not rejected.

    Specs may set ``r2.prefix`` independently of ``task_name``/``run_id`` — e.g.
    the oracle-eval e2e isolates its objects under ``test-runs/<test>/<uuid>/``.
    finalize reads the same prefix generate wrote to, so the spec is
    self-consistent; a prefix that diverges from ``make_r2_prefix`` is advisory
    (logged), never fatal. Pins both halves: finalize emits the warning and
    still lands its artifacts at the custom prefix.

    :param tmp_path: Hosts the scratch ``work_dir``.
    :param fake_r2_remote: Local-typed rclone remote; outputs land here.
    :param monkeypatch: Stubs the Lance fragment I/O and patches
        ``finalize_dataset.logger`` with a recording mock (loguru output does
        not reach pytest ``caplog``).
    :param stub_finalize_setup: Fixture-activation only — installs the
        ``ensure_r2_env_loaded`` / ``object_size`` stubs.
    """
    spec = build_lance_smoke_spec(task_name="finalize-custom-prefix")
    custom_r2 = spec.r2.model_copy(
        update={"prefix": "test-runs/finalize-custom-prefix/abc123def456/"}
    )
    custom_spec = spec.model_copy(update={"r2": custom_r2})
    stub_finalize_lance_io(monkeypatch)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    recording_logger = MagicMock(wraps=finalize_dataset.logger)
    monkeypatch.setattr(finalize_dataset, "logger", recording_logger)

    finalize_dataset.finalize_from_spec(custom_spec, work_dir)

    assert uri_to_local_path(fake_r2_remote, custom_spec.r2.stats_uri()).is_file()
    assert uri_to_local_path(
        fake_r2_remote, custom_spec.r2.dataset_complete_marker_uri()
    ).is_file()
    assert any(
        "non-canonical r2 prefix" in str(call.args[0])
        for call in recording_logger.warning.call_args_list
    ), recording_logger.warning.call_args_list


def test_finalize_lance_raises_on_empty_train_split(fake_r2_remote: Path, tmp_path: Path) -> None:
    """An empty train split surfaces as a clear ValueError before any storage work.

    Stats can't be computed without at least one train shard; the guard converts a would-be low-
    signal object-store failure into a contract violation the operator can fix.

    :param fake_r2_remote: Local-typed rclone remote — asserted untouched because the empty-train
        guard short-circuits before any I/O.
    :param tmp_path: Pytest tmp dir used as finalize's local work_dir.
    """
    spec = build_lance_smoke_spec(task_name="empty-train-lance", train_val_test_sizes=(0, 4, 0))

    with pytest.raises(ValueError, match="train split is empty"):
        finalize_dataset.finalize_lance(spec, tmp_path)

    assert [p for p in fake_r2_remote.rglob("*") if p.is_file()] == []


def test_finalize_from_spec_propagates_winner_failure_before_marker_upload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A winner-selection failure surfaces and no stats or marker upload runs.

    ``finalize_lance_fragments`` raises on missing/degenerate staged attempts
    (see ``_select_checked_winners``); finalize must propagate the error rather
    than swallow it and proceed to upload ``dataset.complete``. Drives
    ``finalize_from_spec`` directly so the assertion is local to the commit step.

    :param monkeypatch: Pytest fixture used to install transport + selection stubs.
    :param tmp_path: Pytest tmp dir; hosts the scratch work_dir.
    """
    uploaded: list[str] = []
    monkeypatch.setattr("synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda *a, **k: None)
    monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda _uri: None)
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.upload", lambda src, dst: uploaded.append(dst)
    )

    def boom(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise RuntimeError("no healthy staged-valid attempt")

    monkeypatch.setattr("synth_setter.pipeline.data.lance_finalize._select_checked_winners", boom)

    spec = build_lance_smoke_spec(task_name="winner-raises-lance")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    with pytest.raises(RuntimeError, match="no healthy staged-valid attempt"):
        finalize_dataset.finalize_from_spec(spec, work_dir)

    assert uploaded == []
