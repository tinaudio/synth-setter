"""Tests for ``synth_setter.cli.finalize_dataset`` — finalize entrypoint.

End-to-end test invokes ``main()`` against the real ``smoke-shard-wds``
experiment so the Hydra compose + ``DatasetSpec`` construction stay
exercised; rclone (download / upload / auth ping) is stubbed.
"""

from __future__ import annotations

import io
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from synth_setter.cli import finalize_dataset
from synth_setter.pipeline.schemas.spec import DatasetSpec

_FIXED_NOW = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)


def _write_minimal_wds_shard(dest: Path) -> None:
    """Write a tar at ``dest`` with one ``00000000.mel_spec.npy`` member.

    :param dest: Filesystem path where the tar is written.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    # 4 rows so Welford variance is non-degenerate.
    payload = np.arange(8, dtype=np.float32).reshape(4, 2)
    buf = io.BytesIO()
    np.save(buf, payload)
    member_bytes = buf.getvalue()
    with tarfile.open(dest, mode="w") as tar:
        info = tarfile.TarInfo(name="00000000.mel_spec.npy")
        info.size = len(member_bytes)
        tar.addfile(info, io.BytesIO(member_bytes))


def _build_wds_smoke_spec(
    task_name: str = "finalize-wds-unit",
    train_val_test_sizes: tuple[int, int, int] = (4, 0, 0),
) -> DatasetSpec:
    """Construct a wds ``DatasetSpec`` directly (no Hydra compose).

    :param task_name: Unique task name so each test gets a distinct r2.prefix.
    :param train_val_test_sizes: Three-tuple of sample counts; default is one
        4-sample shard.
    :returns: A frozen wds ``DatasetSpec`` whose shards are deterministic.
    """
    kwargs: dict[str, Any] = {
        "task_name": task_name,
        "output_format": "wds",
        "train_val_test_sizes": list(train_val_test_sizes),
        "base_seed": 42,
        "r2": {"bucket": "intermediate-data"},
        "render": {
            "plugin_path": "/fake/Plugin.vst3",
            "preset_path": "presets/surge-base.vstpreset",
            "param_spec_name": "surge_simple",
            "renderer_version": "1.0.0-test",
            "sample_rate": 16000,
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "samples_per_render_batch": 4,
            "samples_per_shard": 4,
            "gui_toggle_cadence": "never",
        },
    }
    return DatasetSpec(**kwargs)  # type: ignore[arg-type]


@pytest.fixture()
def patch_finalize_io(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], list[tuple[Path, str]]]:
    """Patch ``r2_io`` download/upload/auth so finalize never touches real R2.

    :param monkeypatch: Pytest fixture used to install the stubs.
    :returns:``(downloaded_uris, uploaded)`` — the two recording lists. Tests read these to assert
        behaviour.
    """
    downloaded_uris: list[str] = []
    uploaded: list[tuple[Path, str]] = []

    def fake_download(r2_uri: str, dest_path: Path) -> None:
        downloaded_uris.append(r2_uri)
        _write_minimal_wds_shard(dest_path)

    def record_upload(source: str | Path, destination_uri: str) -> None:
        uploaded.append((Path(source), destination_uri))

    monkeypatch.setattr("synth_setter.pipeline.r2_io.download_to_path", fake_download)
    monkeypatch.setattr("synth_setter.pipeline.r2_io.upload", record_upload)
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.ensure_r2_env_loaded",
        lambda *args, **kwargs: None,
    )
    # main() probes the marker URI for the idempotency gate; default is "absent".
    monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda uri: None)
    # Pin runtime fields so DatasetSpec.run_id is reproducible across the
    # main()-side compose and the assertion-side compose.
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._get_git_sha", lambda: "a" * 40)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._is_repo_dirty", lambda: False)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._utc_now", lambda: _FIXED_NOW)
    return downloaded_uris, uploaded


def test_main_uploads_stats_then_marker_at_canonical_uris(
    monkeypatch: pytest.MonkeyPatch,
    patch_finalize_io: tuple[list[str], list[tuple[Path, str]]],
) -> None:
    """``main()`` against smoke-shard-wds uploads exactly ``stats.npz`` then ``dataset.complete``.

    :param monkeypatch: Pytest fixture used to patch ``sys.argv``.
    :param patch_finalize_io: Recording stubs for download / upload / auth.
    """
    _downloaded, uploaded = patch_finalize_io
    monkeypatch.setattr(sys, "argv", ["finalize", "experiment=generate_dataset/smoke-shard-wds"])

    finalize_dataset.main()

    # Reconstruct the spec the same way main() does so we can assert exact URIs.
    spec = _compose_smoke_wds_spec()
    upload_destinations = [dst for _, dst in uploaded]
    assert upload_destinations == [
        spec.r2.stats_uri(),
        spec.r2.dataset_complete_marker_uri(),
    ]
    assert upload_destinations[-1] == spec.r2.dataset_complete_marker_uri()


def test_main_is_idempotent_when_marker_already_exists(
    monkeypatch: pytest.MonkeyPatch,
    patch_finalize_io: tuple[list[str], list[tuple[Path, str]]],
) -> None:
    """If ``dataset.complete`` already exists at the run prefix, ``main()`` does no work.

    :param monkeypatch: Pytest fixture used to patch ``sys.argv`` and ``object_size``.
    :param patch_finalize_io: Recording stubs; we override ``object_size`` after
        the fixture installs the default-absent stub.
    """
    downloaded, uploaded = patch_finalize_io
    monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda uri: 0)
    monkeypatch.setattr(sys, "argv", ["finalize", "experiment=generate_dataset/smoke-shard-wds"])

    finalize_dataset.main()

    assert downloaded == []
    assert uploaded == []


def test_main_hdf5_branch_raises_not_implemented_without_uploading_anything(
    monkeypatch: pytest.MonkeyPatch,
    patch_finalize_io: tuple[list[str], list[tuple[Path, str]]],
) -> None:
    """The hdf5 stub raises before any upload — never strands a spurious marker.

    :param monkeypatch: Pytest fixture used to patch ``sys.argv``.
    :param patch_finalize_io: Recording stubs; asserted to be empty after the raise.
    """
    _downloaded, uploaded = patch_finalize_io
    monkeypatch.setattr(sys, "argv", ["finalize", "experiment=generate_dataset/smoke-shard"])

    with pytest.raises(NotImplementedError, match="hdf5 finalize is not implemented"):
        finalize_dataset.main()
    assert uploaded == []


def test_finalize_wds_downloads_every_train_shard_uri(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Multi-shard train split: every ``spec.shards[train_lo:train_hi]`` URI is downloaded.

    :param monkeypatch: Pytest fixture used to install download/upload stubs.
    :param tmp_path: Pytest tmp dir used as the in-process scratch work_dir.
    """
    downloaded_uris: list[str] = []
    uploaded: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_to_path",
        lambda uri, dest: (downloaded_uris.append(uri), _write_minimal_wds_shard(dest)),
    )
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.upload",
        lambda source, destination_uri: uploaded.append((Path(source), destination_uri)),
    )

    spec = _build_wds_smoke_spec(task_name="multi-shard-train", train_val_test_sizes=(8, 0, 0))
    finalize_dataset.finalize_wds(spec, tmp_path)

    expected_uris = [spec.r2.shard_uri(shard) for shard in spec.shards]
    assert downloaded_uris == expected_uris
    assert len(uploaded) == 1
    assert uploaded[0][1] == spec.r2.stats_uri()


def test_finalize_wds_raises_on_empty_train_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty train split surfaces as a clear ValueError, not a misleading FileNotFoundError.

    :param monkeypatch: Pytest fixture used to install download/upload stubs.
    :param tmp_path: Pytest tmp dir used as the in-process scratch work_dir.
    """
    # download/upload must never be called — the empty-train guard short-circuits.
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_to_path",
        lambda *a, **kw: pytest.fail("download_to_path should not be reached"),
    )
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.upload",
        lambda *a, **kw: pytest.fail("upload should not be reached"),
    )

    spec = _build_wds_smoke_spec(task_name="empty-train", train_val_test_sizes=(0, 4, 0))
    with pytest.raises(ValueError, match="train split is empty"):
        finalize_dataset.finalize_wds(spec, tmp_path)


def test_finalize_wds_unlinks_each_shard_after_folding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Peak-disk invariant: at any moment at most one shard sits in ``work_dir``.

    :param monkeypatch: Pytest fixture used to install the download stub.
    :param tmp_path: Pytest tmp dir used as the in-process scratch work_dir.
    """
    concurrent_shards_seen: list[int] = []

    def fake_download(r2_uri: str, dest_path: Path) -> None:
        del r2_uri
        _write_minimal_wds_shard(dest_path)
        concurrent_shards_seen.append(len(list(tmp_path.glob("shard-*.tar"))))

    monkeypatch.setattr("synth_setter.pipeline.r2_io.download_to_path", fake_download)
    monkeypatch.setattr("synth_setter.pipeline.r2_io.upload", lambda src, dst: None)

    spec = _build_wds_smoke_spec(task_name="peak-disk", train_val_test_sizes=(8, 0, 0))
    finalize_dataset.finalize_wds(spec, tmp_path)

    assert concurrent_shards_seen == [1, 1]


def _compose_smoke_wds_spec() -> DatasetSpec:
    """Re-compose the wds smoke spec the same way ``main()`` does, for URI assertions.

    :returns: A ``DatasetSpec`` whose ``r2.prefix`` matches what ``main()``
        constructs from the ``smoke-shard-wds`` experiment in the same process.
    """
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(finalize_dataset._CONFIG_DIR),  # noqa: SLF001 — test mirrors main()
    ):
        cfg = compose(
            config_name="dataset",
            overrides=["experiment=generate_dataset/smoke-shard-wds"],
        )
    cfg.paths.root_dir = str(finalize_dataset._REPO_ROOT)  # noqa: SLF001
    cfg.paths.output_dir = str(finalize_dataset._REPO_ROOT)  # noqa: SLF001
    cfg.paths.work_dir = str(finalize_dataset._REPO_ROOT)  # noqa: SLF001
    return finalize_dataset.spec_from_cfg(cfg)
