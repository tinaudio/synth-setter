"""Tests for the ``synth-setter-browse-dataset`` CLI."""

from __future__ import annotations

from pathlib import Path

import lance
import numpy as np
import pytest
from click.testing import CliRunner

from synth_setter.cli import browse_dataset
from tests.helpers.lance_fixtures import write_lance_shard

_COLUMNS = {"param_array": np.arange(5, dtype=np.float32).reshape(1, 5)}


def _raise_launched(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("sense must not be launched under --no-launch")


def _raise_nonzero_exit(cmd: list[str], **_kwargs: object) -> None:
    raise browse_dataset.subprocess.CalledProcessError(returncode=3, cmd=cmd)


def _refuse_download(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("download must not run when sources collide on a name")


def test_main_no_launch_writes_datasets_and_skips_sense(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--no-launch`` writes the datasets and never shells out to ``sense``.

    :param tmp_path: Holds the source shard and the exported browse-db.
    :param monkeypatch: Stubs ``subprocess.run`` to fail if launch is attempted.
    """
    shard = tmp_path / "train.lance"
    write_lance_shard(shard, _COLUMNS)
    db_dir = tmp_path / "browse"

    monkeypatch.setattr(browse_dataset.subprocess, "run", _raise_launched)

    result = CliRunner().invoke(
        browse_dataset.main, [str(shard), "--db-dir", str(db_dir), "--no-launch"]
    )

    assert result.exit_code == 0, result.output
    assert lance.dataset(str(db_dir / "train.lance")).count_rows() == 1
    assert "sense db" in result.output


def test_main_launch_runs_sense_on_the_written_db_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful export launches ``sense db`` pointed at a populated export dir.

    :param tmp_path: Holds the source shard and the exported browse-db.
    :param monkeypatch: Stubs the ``sense`` binary lookup and captures the launch.
    """
    shard = tmp_path / "train.lance"
    write_lance_shard(shard, _COLUMNS)
    db_dir = tmp_path / "browse"
    calls: list[list[str]] = []

    monkeypatch.setattr(browse_dataset.shutil, "which", lambda _name: "/usr/bin/sense")
    monkeypatch.setattr(browse_dataset.subprocess, "run", lambda cmd, **_kw: calls.append(cmd))

    result = CliRunner().invoke(
        browse_dataset.main, [str(shard), "--db-dir", str(db_dir), "--launch"]
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    sense_argv = calls[0]
    assert sense_argv[:2] == ["/usr/bin/sense", "db"]
    # Behavior, not just argv: the directory handed to `sense` must already hold
    # the exported dataset, so a launch-before-export regression would fail here.
    launched_db = Path(sense_argv[2])
    assert lance.dataset(str(launched_db / "train.lance")).count_rows() == 1


def test_main_launch_nonzero_sense_exit_errors_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-zero ``sense`` exit becomes a clean UsageError naming the status.

    :param tmp_path: Holds the source shard and the export dir.
    :param monkeypatch: Stubs the ``sense`` binary and a failing subprocess run.
    """
    shard = tmp_path / "train.lance"
    write_lance_shard(shard, _COLUMNS)

    monkeypatch.setattr(browse_dataset.shutil, "which", lambda _name: "/usr/bin/sense")
    monkeypatch.setattr(browse_dataset.subprocess, "run", _raise_nonzero_exit)

    result = CliRunner().invoke(
        browse_dataset.main, [str(shard), "--db-dir", str(tmp_path / "browse")]
    )

    assert result.exit_code == 2
    assert "status 3" in result.output


def test_main_without_db_dir_writes_to_a_fresh_temp_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting ``--db-dir`` exports into a fresh temp directory that persists.

    :param tmp_path: Backs the stubbed temp directory and the source shard.
    :param monkeypatch: Pins ``tempfile.mkdtemp`` to a known path to inspect it.
    """
    shard = tmp_path / "val.lance"
    write_lance_shard(shard, _COLUMNS)
    auto_dir = tmp_path / "auto-db"
    real_mkdtemp = browse_dataset.tempfile.mkdtemp

    def _fake_mkdtemp(
        suffix: str | None = None, prefix: str | None = None, dir: str | None = None
    ) -> str:
        # TemporaryDirectory (the download scratch) also calls mkdtemp; only the
        # db-root call (no "-dl-" prefix) is redirected to the inspectable path.
        if prefix and prefix.startswith("synth-setter-browse-dl-"):
            return real_mkdtemp(suffix, prefix, dir)
        auto_dir.mkdir()
        return str(auto_dir)

    monkeypatch.setattr(browse_dataset.tempfile, "mkdtemp", _fake_mkdtemp)

    result = CliRunner().invoke(browse_dataset.main, [str(shard), "--no-launch"])

    assert result.exit_code == 0, result.output
    assert lance.dataset(str(auto_dir / "val.lance")).count_rows() == 1
    assert str(auto_dir) in result.output


def test_main_multiple_local_sources_writes_one_table_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Several local sources in one invocation each land as their own table.

    :param tmp_path: Holds the source shards and the exported browse-db.
    :param monkeypatch: Stubs ``subprocess.run`` to fail if launch is attempted.
    """
    train = tmp_path / "train.lance"
    val = tmp_path / "val.lance"
    write_lance_shard(train, _COLUMNS)
    write_lance_shard(val, _COLUMNS)
    db_dir = tmp_path / "browse"

    monkeypatch.setattr(browse_dataset.subprocess, "run", _raise_launched)

    result = CliRunner().invoke(
        browse_dataset.main, [str(train), str(val), "--db-dir", str(db_dir), "--no-launch"]
    )

    assert result.exit_code == 0, result.output
    assert lance.dataset(str(db_dir / "train.lance")).count_rows() == 1
    assert lance.dataset(str(db_dir / "val.lance")).count_rows() == 1


def test_main_launch_without_sense_binary_errors_with_install_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing ``sense`` binary fails with a UsageError pointing at the install command.

    :param tmp_path: Holds the source shard and the export dir.
    :param monkeypatch: Makes ``shutil.which`` report ``sense`` absent.
    """
    shard = tmp_path / "train.lance"
    write_lance_shard(shard, _COLUMNS)

    monkeypatch.setattr(browse_dataset.shutil, "which", lambda _name: None)

    result = CliRunner().invoke(
        browse_dataset.main, [str(shard), "--db-dir", str(tmp_path / "browse")]
    )

    assert result.exit_code == 2
    assert "uv tool install -U smoosense" in result.output


def test_main_missing_local_source_errors_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing source path surfaces a clean UsageError, not a raw traceback.

    :param tmp_path: Provides the nonexistent source path and export dir.
    :param monkeypatch: Stubs ``subprocess.run`` so no launch is attempted.
    """
    monkeypatch.setattr(browse_dataset.subprocess, "run", _raise_launched)

    result = CliRunner().invoke(
        browse_dataset.main,
        [str(tmp_path / "absent.lance"), "--db-dir", str(tmp_path / "browse"), "--no-launch"],
    )

    assert result.exit_code == 2
    assert "not found" in result.output


def test_main_r2_source_downloads_before_export_and_writes_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ``r2://`` source is fetched via r2_io before being exported to a dataset.

    :param tmp_path: Holds the exported browse-db.
    :param monkeypatch: Stubs r2_io to materialize a shard at the download path.
    """
    db_dir = tmp_path / "browse"
    events: list[str] = []
    downloaded: list[str] = []

    def _fake_download(uri: str, dest: Path) -> None:
        downloaded.append(uri)
        events.append("download")
        write_lance_shard(dest, _COLUMNS)

    monkeypatch.setattr(browse_dataset.r2_io, "ensure_r2_env_loaded", lambda: events.append("env"))
    monkeypatch.setattr(browse_dataset.r2_io, "download_to_path", _fake_download)

    result = CliRunner().invoke(
        browse_dataset.main,
        ["r2://bucket/run/val.lance", "--db-dir", str(db_dir), "--no-launch"],
    )

    assert result.exit_code == 0, result.output
    assert downloaded == ["r2://bucket/run/val.lance"]
    # Creds must load before the download is attempted.
    assert events == ["env", "download"]
    assert lance.dataset(str(db_dir / "val.lance")).count_rows() == 1


def test_main_duplicate_table_names_error_before_any_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two r2 sources sharing a stem fail fast, before r2_io is touched.

    :param tmp_path: Provides the export dir (never written on this path).
    :param monkeypatch: Asserts r2_io download is never reached.
    """
    # Both stubbed to refuse: no R2 I/O of any kind may precede the collision check.
    monkeypatch.setattr(browse_dataset.r2_io, "ensure_r2_env_loaded", _refuse_download)
    monkeypatch.setattr(browse_dataset.r2_io, "download_to_path", _refuse_download)

    result = CliRunner().invoke(
        browse_dataset.main,
        [
            "r2://bucket/a/train.lance",
            "r2://bucket/b/train.lance",
            "--db-dir",
            str(tmp_path / "browse"),
            "--no-launch",
        ],
    )

    assert result.exit_code == 2
    assert "collide on table name" in result.output


def test_main_r2_uri_without_lance_filename_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare-bucket ``r2://`` URI is rejected before any R2 I/O.

    :param tmp_path: Provides the export dir (never written on this path).
    :param monkeypatch: Asserts r2 creds loading is never reached.
    """
    monkeypatch.setattr(browse_dataset.r2_io, "ensure_r2_env_loaded", _refuse_download)

    result = CliRunner().invoke(
        browse_dataset.main, ["r2://bucket", "--db-dir", str(tmp_path / "browse"), "--no-launch"]
    )

    assert result.exit_code == 2
    assert "no .lance filename component" in result.output


def test_main_auto_db_dir_removed_when_source_resolution_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An auto-created temp db root is cleaned up when a source raises (no orphan dir).

    :param tmp_path: Backs the stubbed auto db root.
    :param monkeypatch: Pins ``tempfile.mkdtemp`` so the db root path is inspectable.
    """
    auto_dir = tmp_path / "auto-db"
    real_mkdtemp = browse_dataset.tempfile.mkdtemp

    def _fake_mkdtemp(
        suffix: str | None = None, prefix: str | None = None, dir: str | None = None
    ) -> str:
        if prefix and prefix.startswith("synth-setter-browse-dl-"):
            return real_mkdtemp(suffix, prefix, dir)
        auto_dir.mkdir()
        return str(auto_dir)

    monkeypatch.setattr(browse_dataset.tempfile, "mkdtemp", _fake_mkdtemp)

    # "r2://bucket" has no .lance filename, so _resolve_source raises UsageError.
    result = CliRunner().invoke(browse_dataset.main, ["r2://bucket", "--no-launch"])

    assert result.exit_code == 2
    assert not auto_dir.exists()
