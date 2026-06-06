"""Tests for the ``${wandb:...}`` OmegaConf resolver in ``utils.utils``.

The W&B public API is faked (no network): ``wandb.Api().artifact(ref)`` returns
a stub whose ``download(root=...)`` writes a ``model.ckpt`` into the cache and
records every call, so a second resolution can assert the cache is reused.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from synth_setter.utils.utils import register_resolvers


class _FakeArtifact:
    """W&B artifact stub that writes ``model.ckpt`` on download and counts calls."""

    def __init__(self, calls: list[str]) -> None:
        """Capture the shared call log the download path appends to.

        :param calls: Shared list recording every ``download`` invocation.
        """
        self._calls = calls

    def download(self, root: str) -> str:
        """Write a placeholder checkpoint into ``root`` and record the call.

        :param root: Destination directory the resolver passes for caching.
        :returns: The ``root`` it was given, mirroring the real API.
        """
        self._calls.append(root)
        dest = Path(root)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "model.ckpt").write_bytes(b"weights")
        return root


def _fake_api(calls: list[str]) -> SimpleNamespace:
    """Build a fake ``wandb`` module whose ``Api().artifact(...)`` returns the stub.

    :param calls: Shared list that records every ``download`` invocation.
    :returns: A ``wandb``-shaped namespace with an ``Api`` factory.
    """
    artifact = _FakeArtifact(calls)
    api = SimpleNamespace(artifact=lambda ref: artifact)
    return SimpleNamespace(Api=lambda: api)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the workspace anchor at ``tmp_path`` so the cache lands under it.

    :param tmp_path: Per-test temp dir used as ``$PROJECT_ROOT``.
    :param monkeypatch: Sets ``SYNTH_SETTER_WORKSPACE`` and clears the lru_cache.
    :returns: The temp workspace root.
    """
    monkeypatch.setenv("SYNTH_SETTER_WORKSPACE", str(tmp_path))
    from synth_setter import workspace as workspace_mod

    workspace_mod.operator_workspace.cache_clear()
    return tmp_path


def test_register_resolvers_registers_wandb_resolver() -> None:
    """register_resolvers makes the ``wandb`` resolver available to OmegaConf."""
    register_resolvers()
    assert OmegaConf.has_resolver("wandb")


def test_wandb_resolver_returns_cached_checkpoint_path(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolving ``${wandb:ref}`` downloads the artifact and returns the ckpt path.

    :param workspace: Temp ``$PROJECT_ROOT`` the cache lands under.
    :param monkeypatch: Injects the fake ``wandb`` module into ``sys.modules``.
    """
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "wandb", _fake_api(calls))
    register_resolvers()

    cfg = OmegaConf.create({"ckpt": "${wandb:entity/project/model-x:latest}"})
    resolved = Path(cfg.ckpt)

    assert resolved.name == "model.ckpt"
    assert resolved.is_file()
    assert workspace / ".cache" / "checkpoints" in resolved.parents
    assert len(calls) == 1


def test_wandb_resolver_reuses_cache_without_redownload(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second resolution of the same ref returns the cache without downloading again.

    :param workspace: Temp ``$PROJECT_ROOT`` the cache lands under.
    :param monkeypatch: Injects the fake ``wandb`` module into ``sys.modules``.
    """
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "wandb", _fake_api(calls))
    register_resolvers()

    first = OmegaConf.create({"ckpt": "${wandb:model-x:latest}"})
    second = OmegaConf.create({"ckpt": "${wandb:model-x:latest}"})

    assert Path(first.ckpt) == Path(second.ckpt)
    assert len(calls) == 1
