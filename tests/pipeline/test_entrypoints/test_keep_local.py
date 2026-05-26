"""Tests for ``synth_setter.cli._keep_local`` — shared --keep-local plumbing."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from synth_setter.cli._keep_local import (
    KEEP_LOCAL_FLAG,
    redirect_r2_to_local,
    split_keep_local,
)


class TestSplitKeepLocal:
    """``split_keep_local`` strips the flag from argv; boolean-only surface."""

    def test_absent_returns_false_and_original_argv(self) -> None:
        """No ``--keep-local`` → ``(False, argv)`` with argv unchanged."""
        argv = ["experiment=foo", "render.parallel=true"]
        keep_local, remaining = split_keep_local(argv)
        assert keep_local is False
        assert remaining == argv

    def test_present_returns_true_and_strips_flag(self) -> None:
        """``--keep-local`` is removed; other overrides survive in order."""
        argv = ["experiment=foo", KEEP_LOCAL_FLAG, "render.parallel=true"]
        keep_local, remaining = split_keep_local(argv)
        assert keep_local is True
        assert remaining == ["experiment=foo", "render.parallel=true"]

    def test_present_at_start_strips_correctly(self) -> None:
        """Flag at index 0 still strips cleanly."""
        argv = [KEEP_LOCAL_FLAG, "experiment=foo"]
        keep_local, remaining = split_keep_local(argv)
        assert keep_local is True
        assert remaining == ["experiment=foo"]

    def test_equals_form_rejected(self) -> None:
        """``--keep-local=true`` form raises — flag is boolean-only."""
        argv = ["experiment=foo", f"{KEEP_LOCAL_FLAG}=true"]
        with pytest.raises(ValueError, match="boolean flag"):
            split_keep_local(argv)

    def test_duplicate_rejected(self) -> None:
        """Passing the flag twice raises so silent dedup never masks a typo."""
        argv = [KEEP_LOCAL_FLAG, "experiment=foo", KEEP_LOCAL_FLAG]
        with pytest.raises(ValueError, match="more than once"):
            split_keep_local(argv)

    def test_empty_argv_returns_false_and_empty(self) -> None:
        """``[]`` is a no-op."""
        assert split_keep_local([]) == (False, [])

    def test_prefix_match_passes_through(self) -> None:
        """``--keep-localx`` (prefix without ``=``) is not the flag; survives in remaining.

        Pins the boundary of the ``startswith("--keep-local=")`` gate so a
        rename or loosening to ``startswith("--keep-local")`` would surface here.
        """
        argv = ["--keep-localx", "experiment=foo"]
        keep_local, remaining = split_keep_local(argv)
        assert keep_local is False
        assert remaining == argv

    def test_space_separated_value_falls_through_to_hydra(self) -> None:
        """``--keep-local true`` strips the flag but leaves ``true`` in remaining.

        Documented contract — Hydra then rejects the dangling token. Pins the
        behavior so a future "peek-ahead" implementation announces itself by
        breaking this test.
        """
        argv = [KEEP_LOCAL_FLAG, "true", "experiment=foo"]
        keep_local, remaining = split_keep_local(argv)
        assert keep_local is True
        assert remaining == ["true", "experiment=foo"]


class TestRedirectR2ToLocal:
    """``redirect_r2_to_local`` sets the two env vars rclone reads for ``r2:``."""

    @pytest.fixture(autouse=True)
    def _clear_r2_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Drop any ``RCLONE_CONFIG_R2_*`` keys so each test starts clean.

        :param monkeypatch: Pytest fixture used to remove env vars.
        """
        for key in list(os.environ):
            if key.startswith("RCLONE_CONFIG_R2_"):
                monkeypatch.delenv(key, raising=False)

    def test_sets_alias_type_and_remote_path(self, tmp_path: Path) -> None:
        """``r2:`` is reconfigured to an ``alias`` rooted at ``local_root``.

        :param tmp_path: Pytest tmp dir used as the local-R2 root.
        """
        redirect_r2_to_local(tmp_path)
        assert os.environ["RCLONE_CONFIG_R2_TYPE"] == "alias"
        assert os.environ["RCLONE_CONFIG_R2_REMOTE"] == str(tmp_path)

    def test_creates_root_directory_if_missing(self, tmp_path: Path) -> None:
        """A non-existent ``local_root`` is created so the first rclone copy lands.

        :param tmp_path: Pytest tmp dir under which the missing path lives.
        """
        local_root = tmp_path / "data" / "missing"
        assert not local_root.exists()
        redirect_r2_to_local(local_root)
        assert local_root.is_dir()

    def test_idempotent_with_same_root(self, tmp_path: Path) -> None:
        """Calling twice with the same root is a no-op on the second call.

        :param tmp_path: Pytest tmp dir used as the local-R2 root.
        """
        redirect_r2_to_local(tmp_path)
        redirect_r2_to_local(tmp_path)
        assert os.environ["RCLONE_CONFIG_R2_REMOTE"] == str(tmp_path)
