"""Tests for the ``${wandb:...}`` OmegaConf resolver in ``utils.utils``.

The W&B public API is faked (no network): ``wandb.Api().artifact(ref)`` returns
a stub whose ``download(root=...)`` writes a ``model.ckpt`` into the cache and
records every call, so a second resolution can assert the cache is reused. A
stub may instead carry ``s3://`` manifest references (a reference-only model
artifact); the resolver then rclone-downloads from R2 rather than calling
``download()`` — exercised end-to-end against a local-backed ``r2:`` remote.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from synth_setter.utils import utils as utils_mod
from synth_setter.utils.utils import _resolve_wandb_checkpoint, register_resolvers


class _FakeArtifact:
    """W&B artifact stub: writes files on ``download`` (counting calls) or carries ``s3://`` refs.

    A ``manifest.entries`` of ``s3://`` reference entries models a reference-only
    model artifact; an empty manifest (the default) models a legacy file-upload
    artifact whose bytes materialize via ``download``.
    """

    def __init__(
        self,
        calls: list[str],
        filenames: tuple[str, ...],
        refs: tuple[str, ...] = (),
    ) -> None:
        """Capture the shared call log, the download filenames, and any ``s3://`` references.

        :param calls: Shared list recording every ``download`` invocation.
        :param filenames: Files written into the download root (empty ⇒ no ckpt).
        :param refs: ``s3://`` reference URIs exposed on ``manifest.entries``.
        """
        self._calls = calls
        self._filenames = filenames
        entries = {ref: SimpleNamespace(ref=ref) for ref in refs}
        self.manifest = SimpleNamespace(entries=entries)

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


def _fake_api(
    calls: list[str],
    filenames: tuple[str, ...] = ("model.ckpt",),
    refs: tuple[str, ...] = (),
) -> SimpleNamespace:
    """Build a fake ``wandb`` module whose ``Api().artifact(...)`` returns the stub.

    :param calls: Shared list that records every ``download`` invocation.
    :param filenames: Files the stub materializes on download.
    :param refs: ``s3://`` reference URIs the stub's manifest exposes.
    :returns: A ``wandb``-shaped namespace with an ``Api`` factory.
    """
    artifact = _FakeArtifact(calls, filenames, refs)
    api = SimpleNamespace(artifact=lambda ref: artifact)
    # __spec__ must be present and non-None so the resolver's find_spec("wandb")
    # guard treats this injected stub as an installed package.
    return SimpleNamespace(Api=lambda: api, __spec__=SimpleNamespace())


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the workspace anchor at ``tmp_path`` so the cache lands under it.

    :param tmp_path: Per-test temp dir used as ``$PROJECT_ROOT``.
    :param monkeypatch: Sets ``SYNTH_SETTER_WORKSPACE`` and ``PROJECT_ROOT``, and
        clears ``operator_workspace``'s ``@cache``.
    :returns: The temp workspace root.
    """
    monkeypatch.setenv("SYNTH_SETTER_WORKSPACE", str(tmp_path))
    # operator_workspace() publishes PROJECT_ROOT via os.environ.setdefault. Pin
    # it through monkeypatch (not delenv, which is a no-op when PROJECT_ROOT is
    # unset and so registers no undo) so teardown always restores it rather than
    # leaking tmp_path into later (order-dependent) tests.
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
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


def test_wandb_resolver_reference_artifact_downloads_checkpoint_from_r2(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reference-only artifact's ``s3://`` ckpt is rclone-pulled from R2, not ``download()``-ed.

    Drives the real ``rclone`` binary against a local-backed ``r2:`` remote: the
    checkpoint is staged at the referenced R2 location, the artifact exposes it as
    an ``s3://`` manifest reference, and the resolver must convert the scheme and
    materialize the bytes into its cache — never touching the native ``download()``.

    :param workspace: Temp ``$PROJECT_ROOT`` the cache lands under.
    :param monkeypatch: Injects the fake ``wandb`` module and points rclone at the local fs.
    """
    if shutil.which("rclone") is None:
        pytest.skip("rclone binary not available on PATH")
    monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "local")
    monkeypatch.setattr("synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda *a, **k: None)
    monkeypatch.chdir(workspace)
    staged = workspace / "intermediate-data" / "checkpoints" / "flow-simple" / "model.ckpt"
    staged.parent.mkdir(parents=True, exist_ok=True)
    # Stage a real torch checkpoint so the assertion proves the resolved path is a
    # loadable checkpoint, not just byte-identical bytes the reference branch moved.
    ckpt_state = {"state_dict": {"w": torch.tensor([1.0, 2.0])}, "epoch": 3}
    torch.save(ckpt_state, staged)

    calls: list[str] = []
    ref = "s3://intermediate-data/checkpoints/flow-simple/model.ckpt"
    monkeypatch.setitem(sys.modules, "wandb", _fake_api(calls, filenames=(), refs=(ref,)))

    resolved = Path(_resolve_wandb_checkpoint("entity/project/model-flow-simple:latest"))

    assert resolved.name == "model.ckpt"
    assert workspace / ".cache" / "checkpoints" in resolved.parents
    assert calls == [], "native download() must not run for a reference artifact"
    loaded = torch.load(resolved, weights_only=False)
    assert loaded["epoch"] == 3
    assert torch.equal(loaded["state_dict"]["w"], torch.tensor([1.0, 2.0]))


def test_wandb_resolver_file_artifact_uses_native_download(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An artifact with no ``s3://`` references falls back to native ``download()`` (legacy path).

    :param workspace: Temp ``$PROJECT_ROOT`` the cache lands under.
    :param monkeypatch: Injects the fake ``wandb`` module into ``sys.modules``.
    """
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "wandb", _fake_api(calls, filenames=("model.ckpt",)))

    resolved = Path(_resolve_wandb_checkpoint("model-x:latest"))

    assert resolved.name == "model.ckpt"
    assert resolved.is_file()
    assert len(calls) == 1


def test_wandb_resolver_reference_unsafe_basename_raises(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reference whose basename is ``..`` is rejected, never written above the cache.

    :param workspace: Temp ``$PROJECT_ROOT`` the cache lands under.
    :param monkeypatch: Injects the fake ``wandb`` module and stubs the R2 env load.
    """
    monkeypatch.setattr(utils_mod.shutil, "which", lambda _name: "/usr/bin/rclone")
    monkeypatch.setattr("synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda *a, **k: None)
    calls: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        _fake_api(
            calls, filenames=(), refs=("s3://intermediate-data/checkpoints/flow-simple/..",)
        ),
    )

    with pytest.raises(ValueError, match="unsafe checkpoint basename"):
        _resolve_wandb_checkpoint("entity/project/model-flow-simple:latest")


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

    # Pin the guidance to the real PEP 735 group ('util'), not a non-existent 'wandb' group.
    with pytest.raises(ModuleNotFoundError, match="util"):
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

    # finally clears the global Hydra state even if an assertion fails, so a
    # leaked HydraConfig singleton can't flake later xdist-sibling tests.
    try:
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
    finally:
        GlobalHydra.instance().clear()


# Surge wandb_checkpoint overlays and the ``model-{config_id}`` artifact each pins, where
# config_id is the experiment basename (see ``resolve_run_config_id``). Pins the
# ckpt-wiring contract: a launcher composing ``experiment=surge/wandb_checkpoint/<name>``
# inherits a ``${wandb:tinaudio/synth-setter/model-<name>:latest}`` ckpt_path with no CLI
# override, while the train-side ``surge/<name>`` config carries no ckpt_path.
_WIRED_PREDICT_EXPERIMENTS: tuple[str, ...] = (
    "ffn_full",
    "ffn_simple",
    "flow_full",
    "flow_simple",
    "flow_mlp_full",
    "flow_mlp_simple",
    "vae_full",
    "vae_simple",
)


@pytest.mark.parametrize("experiment", _WIRED_PREDICT_EXPERIMENTS)
def test_surge_experiment_pins_wandb_model_artifact_ckpt(
    experiment: str, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each surge wandb_checkpoint overlay resolves ``ckpt_path`` to its ``model-<id>`` artifact.

    Composes ``experiment=surge/wandb_checkpoint/<name>`` with no ``ckpt_path`` CLI override,
    proving the overlay alone pins ``${wandb:tinaudio/synth-setter/model-<name>:latest}`` —
    the config-pinned replacement for ``get-ckpt-from-wandb.sh``. The fake artifact's download
    dir name encodes the ref slug, so the assertion confirms the per-experiment artifact id
    reached the resolver.

    :param experiment: Surge experiment basename, also the ``model-<id>`` artifact id.
    :param workspace: Temp ``$PROJECT_ROOT`` the cache lands under.
    :param monkeypatch: Injects the fake ``wandb`` module into ``sys.modules``.
    """
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "wandb", _fake_api(calls))
    register_resolvers()

    try:
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            cfg = compose(
                config_name="eval.yaml",
                overrides=[f"experiment=surge/wandb_checkpoint/{experiment}", "trainer=cpu"],
            )
            raw_container = cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=False))
            raw_ckpt = raw_container["ckpt_path"]
            resolved = Path(cfg.ckpt_path)

        assert raw_ckpt == f"${{wandb:tinaudio/synth-setter/model-{experiment}:latest}}"
        assert resolved.name == "model.ckpt"
        assert resolved.is_file()
        assert workspace / ".cache" / "checkpoints" in resolved.parents
        assert len(calls) == 1
    finally:
        GlobalHydra.instance().clear()


@pytest.mark.parametrize("experiment", _WIRED_PREDICT_EXPERIMENTS)
def test_train_surge_experiment_composes_null_ckpt_without_wandb_resolution(
    experiment: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared surge experiment composes a null ckpt under train.yaml and never resolves W&B.

    Regression guard for the wandb_checkpoint overlay split (#128): the ``${wandb:...}`` pin
    lives only in the predict-side ``surge/wandb_checkpoint/<id>`` overlay, never in the shared
    ``surge/<id>`` experiment that ``train.yaml`` composes. Were it to leak back, ``train.py``'s
    ``trainer.fit(ckpt_path=cfg.get("ckpt_path"))`` would resolve the artifact (needing a W&B
    key) and silently resume from the published model. Fast + key-free so regular CI catches the
    regression at PR time, not just the MPS smoke leg.

    :param experiment: Surge experiment basename composed under ``train.yaml``.
    :param monkeypatch: Injects a call-recording fake ``wandb`` so a stray resolution is observable.
    """
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "wandb", _fake_api(calls))
    register_resolvers()

    try:
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            cfg = compose(
                config_name="train.yaml",
                overrides=[f"experiment=surge/{experiment}", "trainer=cpu"],
            )
            # train.py reads exactly cfg.get("ckpt_path"); it must stay None and fire no resolver.
            ckpt_path = cfg.get("ckpt_path")
    finally:
        GlobalHydra.instance().clear()

    assert ckpt_path is None
    assert calls == []
