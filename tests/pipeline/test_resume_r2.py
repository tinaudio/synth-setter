"""R2-tier tests for ``synth_setter.utils.resume`` against the fake rclone remote."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from synth_setter.pipeline import r2_io
from synth_setter.utils.resume import discover_r2_checkpoint, discover_resume_checkpoint

_UUID_A = "a" * 32
_UUID_B = "b" * 32


def _seed_mirror(root: Path, config_id: str, namespace: str, payload: bytes, mtime: float) -> Path:
    """Materialize one mid-run mirror object in the fake remote.

    :param root: Fake R2 root (``fake_r2_remote``).
    :param config_id: Checkpoint identity segment of the key.
    :param namespace: Recovery-namespace directory name.
    :param payload: Bytes written as the checkpoint object.
    :param mtime: File mtime; the local rclone backend reports it as ``LastModified``.
    :returns: The seeded object path.
    """
    obj = root / "test-bucket" / "checkpoints" / config_id / namespace / "last.ckpt"
    obj.parent.mkdir(parents=True)
    obj.write_bytes(payload)
    os.utime(obj, (mtime, mtime))
    return obj


def test_discover_r2_downloads_newest_namespace_last_ckpt(
    fake_r2_remote: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The newest recovery-namespace mirror is downloaded and its run id parsed.

    :param fake_r2_remote: Fake R2 root backing the ``r2:`` remote.
    :param tmp_path: Pytest tmp dir holding the test's fake directories.
    :param monkeypatch: Pytest fixture used to stub module attributes.
    """
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *args, **kwargs: None)
    _seed_mirror(
        fake_r2_remote,
        "ffn_simple",
        f"ffn_simple-20260714T000000000Z-{_UUID_A}",
        b"old",
        mtime=1_000,
    )
    _seed_mirror(
        fake_r2_remote,
        "ffn_simple",
        f"ffn_simple-20260715T225004231Z-{_UUID_B}",
        b"new",
        mtime=2_000,
    )
    dest_dir = tmp_path / "resume-dest"

    decision = discover_r2_checkpoint(
        prefix="r2://test-bucket/checkpoints/ffn_simple", config_id="ffn_simple", dest_dir=dest_dir
    )

    assert decision is not None
    assert decision.source == "r2"
    assert decision.wandb_run_id == "ffn_simple-20260715T225004231Z"
    assert decision.ckpt_path.read_bytes() == b"new"


def test_discover_r2_empty_prefix_returns_none(
    fake_r2_remote: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config_id with no mirrors yields no decision.

    :param fake_r2_remote: Fake R2 root backing the ``r2:`` remote.
    :param tmp_path: Pytest tmp dir holding the test's fake directories.
    :param monkeypatch: Pytest fixture used to stub module attributes.
    """
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *args, **kwargs: None)

    decision = discover_r2_checkpoint(
        prefix="r2://test-bucket/checkpoints/ffn_simple",
        config_id="ffn_simple",
        dest_dir=tmp_path / "dest",
    )

    assert decision is None


def test_discover_r2_unavailable_creds_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing R2 creds degrade to None instead of aborting the launch.

    :param tmp_path: Pytest tmp dir holding the test's fake directories.
    :param monkeypatch: Pytest fixture used to stub module attributes.
    """

    def _unavailable(*args: object, **kwargs: object) -> None:
        raise RuntimeError("no creds")

    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", _unavailable)

    decision = discover_r2_checkpoint(
        prefix="r2://test-bucket/checkpoints/ffn_simple",
        config_id="ffn_simple",
        dest_dir=tmp_path / "dest",
    )

    assert decision is None


def test_discover_resume_checkpoint_prefers_local_over_r2(
    fake_r2_remote: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local sibling checkpoint wins over a newer R2 mirror (tier order).

    :param fake_r2_remote: Fake R2 root backing the ``r2:`` remote.
    :param tmp_path: Pytest tmp dir holding the test's fake directories.
    :param monkeypatch: Pytest fixture used to stub module attributes.
    """
    from omegaconf import OmegaConf

    from synth_setter.utils.resume import discover_resume_checkpoint

    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *args, **kwargs: None)
    _seed_mirror(
        fake_r2_remote,
        "ffn_simple",
        f"ffn_simple-20260715T225004231Z-{_UUID_B}",
        b"from-r2",
        mtime=9_000,
    )
    runs_root = tmp_path / "runs"
    local_ckpt = runs_root / "ffn-prior" / "checkpoints" / "last.ckpt"
    local_ckpt.parent.mkdir(parents=True)
    local_ckpt.write_bytes(b"from-local")
    wandb_dir = (
        runs_root / "ffn-prior" / "wandb" / "run-20260715_185004-ffn_simple-20260715T225004231Z"
    )
    wandb_dir.mkdir(parents=True)
    current = runs_root / "ffn-current"
    current.mkdir()
    cfg = OmegaConf.create(
        {"paths": {"output_dir": str(current)}, "r2": {"bucket": "test-bucket"}}
    )

    decision = discover_resume_checkpoint(cfg, config_id="ffn_simple")

    assert decision is not None
    assert decision.source == "local"
    assert decision.ckpt_path == local_ckpt


def test_discover_r2_ignores_stray_objects_outside_namespace_mirrors(
    fake_r2_remote: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only ``<namespace>/last.ckpt`` objects count — noise under the prefix is skipped.

    :param fake_r2_remote: Fake R2 root backing the ``r2:`` remote.
    :param tmp_path: Holds the download destination.
    :param monkeypatch: Pytest fixture used to stub the R2 auth probe.
    """
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *args, **kwargs: None)
    prefix = fake_r2_remote / "test-bucket" / "checkpoints" / "ffn_simple"
    valid = _seed_mirror(
        fake_r2_remote,
        "ffn_simple",
        f"ffn_simple-20260714T000000000Z-{_UUID_A}",
        b"valid",
        mtime=1_000,
    )
    top_level = prefix / "last.ckpt"
    top_level.write_bytes(b"top-level-noise")
    os.utime(top_level, (9_000, 9_000))
    nested = prefix / f"ffn_simple-20260715T225004231Z-{_UUID_B}" / "sub" / "last.ckpt"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"nested-noise")
    os.utime(nested, (9_000, 9_000))
    epoch = prefix / f"ffn_simple-20260715T225004231Z-{_UUID_B}" / "epoch_001.ckpt"
    epoch.write_bytes(b"epoch-noise")
    os.utime(epoch, (9_000, 9_000))

    decision = discover_r2_checkpoint(
        prefix="r2://test-bucket/checkpoints/ffn_simple",
        config_id="ffn_simple",
        dest_dir=tmp_path / "dest",
    )

    assert decision is not None
    assert decision.ckpt_path.read_bytes() == valid.read_bytes()
    assert decision.wandb_run_id == "ffn_simple-20260714T000000000Z"


def test_discover_resume_checkpoint_honors_upload_uri_override(
    fake_r2_remote: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirrors under a custom ``training.upload_checkpoints_uri`` parent are found.

    :param fake_r2_remote: Fake R2 root backing the ``r2:`` remote.
    :param tmp_path: Holds the current run dir.
    :param monkeypatch: Pytest fixture used to stub the R2 auth probe.
    """
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *args, **kwargs: None)
    namespace = f"ffn_simple-20260715T225004231Z-{_UUID_A}"
    obj = fake_r2_remote / "custom-bucket" / "my" / "spot" / namespace / "last.ckpt"
    obj.parent.mkdir(parents=True)
    obj.write_bytes(b"custom")
    current = tmp_path / "runs" / "ffn-current"
    current.mkdir(parents=True)
    cfg = OmegaConf.create(
        {
            "paths": {"output_dir": str(current)},
            "r2": {"bucket": "other-bucket"},
            "training": {"upload_checkpoints_uri": "r2://custom-bucket/my/spot/model.ckpt"},
        }
    )

    decision = discover_resume_checkpoint(cfg, config_id="ffn_simple")

    assert decision is not None
    assert decision.source == "r2"
    assert decision.ckpt_path.read_bytes() == b"custom"
    assert decision.wandb_run_id == "ffn_simple-20260715T225004231Z"


def test_discover_r2_download_failure_degrades_to_none(
    fake_r2_remote: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed transfer (post-listing) degrades to None instead of aborting.

    :param fake_r2_remote: Fake R2 root backing the ``r2:`` remote.
    :param tmp_path: Holds the download destination.
    :param monkeypatch: Pytest fixture used to stub the auth probe and break the download.
    """
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *args, **kwargs: None)
    _seed_mirror(
        fake_r2_remote,
        "ffn_simple",
        f"ffn_simple-20260715T225004231Z-{_UUID_A}",
        b"new",
        mtime=1_000,
    )

    def _broken_download(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(r2_io, "download_to_path", _broken_download)

    decision = discover_r2_checkpoint(
        prefix="r2://test-bucket/checkpoints/ffn_simple",
        config_id="ffn_simple",
        dest_dir=tmp_path / "dest",
    )

    assert decision is None


def test_discover_r2_non_degradable_error_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A programming error (outside the degradable set) raises instead of degrading.

    :param tmp_path: Holds the download destination.
    :param monkeypatch: Pytest fixture used to make listing raise a TypeError.
    """
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *args, **kwargs: None)

    def _broken_listing(*args: object, **kwargs: object) -> None:
        raise TypeError("programming error")

    monkeypatch.setattr(r2_io, "list_entries", _broken_listing)

    with pytest.raises(TypeError, match="programming error"):
        discover_r2_checkpoint(
            prefix="r2://test-bucket/checkpoints/ffn_simple",
            config_id="ffn_simple",
            dest_dir=tmp_path / "dest",
        )


def test_discover_r2_newer_foreign_namespace_is_ignored(
    fake_r2_remote: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A newer mirror from another config cannot displace this config's checkpoint.

    :param fake_r2_remote: Fake R2 root backing the ``r2:`` remote.
    :param tmp_path: Holds the download destination.
    :param monkeypatch: Pytest fixture used to stub the R2 auth probe.
    """
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *args, **kwargs: None)
    _seed_mirror(
        fake_r2_remote,
        "ffn_simple",
        f"ffn_simple-20260714T000000000Z-{_UUID_A}",
        b"valid",
        mtime=1_000,
    )
    _seed_mirror(
        fake_r2_remote,
        "ffn_simple",
        f"other_config-20260715T225004231Z-{_UUID_B}",
        b"foreign",
        mtime=2_000,
    )

    decision = discover_r2_checkpoint(
        prefix="r2://test-bucket/checkpoints/ffn_simple",
        config_id="ffn_simple",
        dest_dir=tmp_path / "dest",
    )

    assert decision is not None
    assert decision.wandb_run_id == "ffn_simple-20260714T000000000Z"
    assert decision.ckpt_path.read_bytes() == b"valid"
