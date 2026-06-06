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
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from synth_setter.utils import utils as utils_mod
from synth_setter.utils.utils import _resolve_wandb_checkpoint, register_resolvers


class _FakeArtifact:
    """W&B artifact stub that writes the configured files on download and counts calls."""

    def __init__(self, calls: list[str], filenames: tuple[str, ...]) -> None:
        """Capture the shared call log and the filenames each download materializes.

        :param calls: Shared list recording every ``download`` invocation.
        :param filenames: Files written into the download root (empty ⇒ no ckpt).
        """
        self._calls = calls
        self._filenames = filenames

    def download(self, root: str) -> str:
        """Write the configured files into ``root`` and record the call.

        :param root: Destination directory the resolver passes for caching.
        :returns: The ``root`` it was given, mirroring the real API.
        """
        self._calls.append(root)
        dest = Path(root)
        dest.mkdir(parents=True, exist_ok=True)
        for name in self._filenames:
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"weights")
        return root


def _fake_api(calls: list[str], filenames: tuple[str, ...] = ("model.ckpt",)) -> SimpleNamespace:
    """Build a fake ``wandb`` module whose ``Api().artifact(...)`` returns the stub.

    :param calls: Shared list that records every ``download`` invocation.
    :param filenames: Files the stub materializes on download.
    :returns: A ``wandb``-shaped namespace with an ``Api`` factory.
    """
    artifact = _FakeArtifact(calls, filenames)
    api = SimpleNamespace(artifact=lambda ref: artifact)
    # __spec__ must be present and non-None so the resolver's find_spec("wandb")
    # guard treats this injected stub as an installed package.
    return SimpleNamespace(Api=lambda: api, __spec__=SimpleNamespace())


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


def test_resolve_wandb_checkpoint_traversal_ref_stays_inside_cache(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``ref`` with ``..`` and ``:`` resolves inside the cache root, never above it.

    :param workspace: Temp ``$PROJECT_ROOT`` the cache lands under.
    :param monkeypatch: Injects the fake ``wandb`` module into ``sys.modules``.
    """
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "wandb", _fake_api(calls))

    resolved = Path(_resolve_wandb_checkpoint("../../etc/model:latest"))

    cache_root = (workspace / ".cache" / "checkpoints").resolve()
    assert cache_root in resolved.resolve().parents


def test_resolve_wandb_checkpoint_dot_dot_ref_stays_inside_cache(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare ``..`` ref resolves inside the cache root, never above it.

    :param workspace: Temp ``$PROJECT_ROOT`` the cache lands under.
    :param monkeypatch: Injects the fake ``wandb`` module into ``sys.modules``.
    """
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "wandb", _fake_api(calls))

    resolved = Path(_resolve_wandb_checkpoint(".."))

    cache_root = (workspace / ".cache" / "checkpoints").resolve()
    assert cache_root in resolved.resolve().parents


def test_resolve_wandb_checkpoint_slug_colliding_refs_get_distinct_dirs(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refs that slug identically (``a/b`` vs ``a:b``) cache to distinct dirs.

    :param workspace: Temp ``$PROJECT_ROOT`` the cache lands under.
    :param monkeypatch: Injects the fake ``wandb`` module into ``sys.modules``.
    """
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "wandb", _fake_api(calls))

    first = Path(_resolve_wandb_checkpoint("a/b:latest")).parent
    second = Path(_resolve_wandb_checkpoint("a:b:latest")).parent

    assert first != second


def test_resolve_wandb_checkpoint_missing_wandb_raises_module_not_found(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A minimal install without ``wandb`` raises a clear ``ModuleNotFoundError``.

    :param workspace: Temp ``$PROJECT_ROOT`` the cache lands under.
    :param monkeypatch: Forces ``find_spec`` to report ``wandb`` absent.
    """
    monkeypatch.setattr(utils_mod, "find_spec", lambda name: None)

    with pytest.raises(ModuleNotFoundError, match="wandb"):
        _resolve_wandb_checkpoint("model-x:latest")


def test_resolve_wandb_checkpoint_multiple_ckpts_raises(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An artifact with several non-``model.ckpt`` files errors instead of guessing.

    :param workspace: Temp ``$PROJECT_ROOT`` the cache lands under.
    :param monkeypatch: Injects the fake ``wandb`` module into ``sys.modules``.
    """
    calls: list[str] = []
    monkeypatch.setitem(
        sys.modules, "wandb", _fake_api(calls, filenames=("epoch=1.ckpt", "epoch=2.ckpt"))
    )

    with pytest.raises(ValueError, match="ambiguous"):
        _resolve_wandb_checkpoint("model-x:latest")


def test_resolve_wandb_checkpoint_multiple_model_ckpts_raises(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Several ``model.ckpt`` files across nested dirs error instead of guessing.

    :param workspace: Temp ``$PROJECT_ROOT`` the cache lands under.
    :param monkeypatch: Injects the fake ``wandb`` module into ``sys.modules``.
    """
    calls: list[str] = []
    monkeypatch.setitem(
        sys.modules, "wandb", _fake_api(calls, filenames=("a/model.ckpt", "b/model.ckpt"))
    )

    with pytest.raises(ValueError, match="ambiguous"):
        _resolve_wandb_checkpoint("model-x:latest")


def test_resolve_wandb_checkpoint_long_ref_cache_dir_within_name_limit(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A very long ref yields a cache-dir name within the 255-byte filesystem limit.

    :param workspace: Temp ``$PROJECT_ROOT`` the cache lands under.
    :param monkeypatch: Injects the fake ``wandb`` module into ``sys.modules``.
    """
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "wandb", _fake_api(calls))

    resolved = Path(_resolve_wandb_checkpoint("x" * 400 + ":latest"))

    cache_dir = resolved.parent
    assert len(cache_dir.name.encode()) <= 255
    assert resolved.is_file()


def test_resolve_wandb_checkpoint_partial_download_redownloads(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached dir with no ``.ckpt`` (partial download) triggers a fresh download.

    :param workspace: Temp ``$PROJECT_ROOT`` the cache lands under.
    :param monkeypatch: Injects the fake ``wandb`` module into ``sys.modules``.
    """
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "wandb", _fake_api(calls, filenames=()))
    with pytest.raises(FileNotFoundError):
        _resolve_wandb_checkpoint("model-x:latest")

    monkeypatch.setitem(sys.modules, "wandb", _fake_api(calls, filenames=("model.ckpt",)))
    resolved = Path(_resolve_wandb_checkpoint("model-x:latest"))

    assert resolved.is_file()
    assert len(calls) == 2


def test_eval_ckpt_path_wandb_override_resolves_to_cached_checkpoint(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Composing eval.yaml with ``ckpt_path=${wandb:...}`` resolves to the cached ckpt.

    Proves the resolver is registered when Hydra composes ``eval.yaml`` and that
    ``${wandb:...}`` interpolates through the real ``ckpt_path`` key ``evaluate()``
    consumes — the seam the unit tests, which bind the resolver to an ad-hoc key,
    do not cover.

    :param workspace: Temp ``$PROJECT_ROOT`` the cache lands under.
    :param monkeypatch: Injects the fake ``wandb`` module into ``sys.modules``.
    """
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "wandb", _fake_api(calls))
    register_resolvers()

    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            overrides=[
                "datamodule=ksin",
                "model=ffn",
                "trainer=cpu",
                "ckpt_path=${wandb:entity/project/model-x:latest}",
            ],
        )
        resolved = Path(cfg.ckpt_path)

    assert resolved.name == "model.ckpt"
    assert resolved.is_file()
    assert workspace / ".cache" / "checkpoints" in resolved.parents
    assert len(calls) == 1

    GlobalHydra.instance().clear()
