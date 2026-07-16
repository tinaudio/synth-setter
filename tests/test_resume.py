"""Tests for ``synth_setter.utils.resume`` — auto-resume checkpoint discovery."""

import importlib.machinery
import os
import sys
import types
from pathlib import Path

import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from synth_setter.utils import resume
from synth_setter.utils.resume import (
    discover_local_checkpoint,
    resolve_resume_mode,
    run_id_from_recovery_namespace,
)


def _cfg(resume_value: object = None, ckpt_path: str | None = None) -> DictConfig:
    """Build the minimal cfg slice ``resolve_resume_mode`` reads.

    :param resume_value: Value for ``training.resume``.
    :param ckpt_path: Value for ``ckpt_path``.
    :returns: An OmegaConf config with just those keys.
    """
    return OmegaConf.create({"training": {"resume": resume_value}, "ckpt_path": ckpt_path})


@pytest.mark.parametrize(
    "disabled", [None, "off", False], ids=["null", "off-string", "yaml-false"]
)
def test_resolve_resume_mode_disabled_values_return_none(disabled: object) -> None:
    """``null``, ``"off"``, and YAML-1.1 ``off``-as-``False`` all disable resume.

    :param disabled: Parametrized disabled ``training.resume`` value.
    """
    assert resolve_resume_mode(_cfg(resume_value=disabled)) is None


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_resolve_resume_mode_active_modes_pass_through(mode: str) -> None:
    """``auto`` and ``require`` are returned verbatim.

    :param mode: Parametrized active resume mode.
    """
    assert resolve_resume_mode(_cfg(resume_value=mode)) == mode


def test_resolve_resume_mode_unknown_value_raises() -> None:
    """An unrecognized mode is a config error, not a silent fresh start."""
    with pytest.raises(ValueError, match="training.resume"):
        resolve_resume_mode(_cfg(resume_value="best"))


def test_resolve_resume_mode_auto_with_explicit_ckpt_path_raises() -> None:
    """Discovery plus a hand-picked checkpoint is ambiguous intent."""
    cfg = _cfg(resume_value="auto", ckpt_path="/some/last.ckpt")
    with pytest.raises(ValueError, match="ckpt_path"):
        resolve_resume_mode(cfg)


def test_resolve_resume_mode_off_with_explicit_ckpt_path_is_allowed() -> None:
    """Manual resume (today's flow) stays untouched when resume is disabled."""
    assert resolve_resume_mode(_cfg(resume_value=None, ckpt_path="/some/last.ckpt")) is None


def test_resolve_resume_mode_missing_training_block_returns_none() -> None:
    """A cfg without a ``training`` block (e.g. eval) means resume is off."""
    assert resolve_resume_mode(OmegaConf.create({"ckpt_path": None})) is None


def _make_run_dir(
    root: Path,
    name: str,
    *,
    ckpt_mtime: float | None = None,
    wandb_run_id: str | None = None,
) -> Path:
    """Create one fake Hydra run dir with a ``checkpoints/last.ckpt``.

    :param root: Parent of all sibling run dirs.
    :param name: Run dir basename.
    :param ckpt_mtime: Explicit mtime for ``last.ckpt``; newer wins discovery.
    :param wandb_run_id: When set, adds a ``wandb/run-<ts>-<id>`` dir so the id
        is recoverable.
    :returns: The created run dir.
    """
    run_dir = root / name
    ckpt = run_dir / "checkpoints" / "last.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"ckpt")
    if ckpt_mtime is not None:
        os.utime(ckpt, (ckpt_mtime, ckpt_mtime))
    if wandb_run_id is not None:
        (run_dir / "wandb" / f"run-20260715_185004-{wandb_run_id}").mkdir(parents=True)
    return run_dir


def test_discover_local_picks_newest_sibling_last_ckpt(tmp_path: Path) -> None:
    """The newest sibling ``last.ckpt`` by mtime wins.

    :param tmp_path: Pytest tmp dir holding the test's fake directories.
    """
    _make_run_dir(tmp_path, "ffn-old", ckpt_mtime=1_000)
    newest = _make_run_dir(tmp_path, "ffn-new", ckpt_mtime=2_000)
    current = tmp_path / "ffn-current"
    current.mkdir()

    decision = discover_local_checkpoint(current, config_id="ffn_simple")

    assert decision is not None
    assert decision.ckpt_path == newest / "checkpoints" / "last.ckpt"
    assert decision.source == "local"


def test_discover_local_excludes_current_output_dir(tmp_path: Path) -> None:
    """The launching run's own dir never resumes itself.

    :param tmp_path: Pytest tmp dir holding the test's fake directories.
    """
    current = _make_run_dir(tmp_path, "ffn-current", ckpt_mtime=9_000)

    assert discover_local_checkpoint(current, config_id="ffn_simple") is None


def test_discover_local_recovers_wandb_run_id_from_run_dir(tmp_path: Path) -> None:
    """The prior launch's W&B run id is read from its ``wandb/run-*`` dirname.

    :param tmp_path: Pytest tmp dir holding the test's fake directories.
    """
    _make_run_dir(tmp_path, "ffn-prior", wandb_run_id="ffn_simple-20260715T225004231Z")
    current = tmp_path / "ffn-current"
    current.mkdir()

    decision = discover_local_checkpoint(current, config_id="ffn_simple")

    assert decision is not None
    assert decision.wandb_run_id == "ffn_simple-20260715T225004231Z"


def test_discover_local_skips_sibling_of_a_different_config_id(tmp_path: Path) -> None:
    """A newer checkpoint from a different experiment must not hijack the resume.

    :param tmp_path: Pytest tmp dir holding the test's fake directories.
    """
    _make_run_dir(
        tmp_path,
        "flow-prior",
        ckpt_mtime=9_000,
        wandb_run_id="flow_simple-20260715T225004231Z",
    )
    older_same_config = _make_run_dir(
        tmp_path,
        "ffn-prior",
        ckpt_mtime=1_000,
        wandb_run_id="ffn_simple-20260714T000000000Z",
    )
    current = tmp_path / "ffn-current"
    current.mkdir()

    decision = discover_local_checkpoint(current, config_id="ffn_simple")

    assert decision is not None
    assert decision.ckpt_path == older_same_config / "checkpoints" / "last.ckpt"


def test_discover_local_sibling_without_wandb_dir_is_accepted_with_no_run_id(
    tmp_path: Path,
) -> None:
    """A wandb-less run dir (logger disabled) still resumes, minting a fresh id.

    :param tmp_path: Pytest tmp dir holding the test's fake directories.
    """
    _make_run_dir(tmp_path, "ffn-prior")
    current = tmp_path / "ffn-current"
    current.mkdir()

    decision = discover_local_checkpoint(current, config_id="ffn_simple")

    assert decision is not None
    assert decision.wandb_run_id is None


def test_discover_local_no_siblings_returns_none(tmp_path: Path) -> None:
    """An empty run-dir family yields no decision.

    :param tmp_path: Pytest tmp dir holding the test's fake directories.
    """
    current = tmp_path / "ffn-current"
    current.mkdir()

    assert discover_local_checkpoint(current, config_id="ffn_simple") is None


def test_namespace_run_id_strips_uuid_suffix() -> None:
    """A ``{run_id}-{32-hex}`` recovery namespace yields the embedded run id."""
    namespace = "ffn_simple-20260715T225004231Z-" + "a" * 32
    assert run_id_from_recovery_namespace(namespace) == "ffn_simple-20260715T225004231Z"


@pytest.mark.parametrize(
    "namespace",
    ["ffn_simple-20260715T225004231Z", "run-abc123"],
    ids=["no-suffix", "short-hex"],
)
def test_namespace_run_id_without_uuid_suffix_returns_none(namespace: str) -> None:
    """Names without the fixed-width uuid4 suffix carry no recoverable id.

    :param namespace: Parametrized namespace without a valid uuid suffix.
    """
    assert run_id_from_recovery_namespace(namespace) is None


def test_discover_wandb_artifact_without_wandb_installed_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The artifact tier is a silent no-op on wandb-free installs.

    :param monkeypatch: Pytest fixture used to stub module attributes.
    """
    monkeypatch.setattr(resume, "find_spec", lambda name: None)

    assert resume.discover_wandb_artifact_checkpoint("ffn_simple") is None


def test_apply_wandb_resume_continuity_sets_allow_on_wandb_logger() -> None:
    """A cfg with a wandb logger group gets ``resume: allow`` pinned."""
    cfg = OmegaConf.create({"logger": {"wandb": {"id": None, "resume": None}}})

    resume.apply_wandb_resume_continuity(cfg)

    assert cfg.logger.wandb.resume == "allow"


def test_apply_wandb_resume_continuity_without_wandb_logger_is_noop() -> None:
    """A logger-free cfg is left untouched."""
    cfg = OmegaConf.create({"logger": None})

    resume.apply_wandb_resume_continuity(cfg)

    assert cfg.logger is None


def _install_fake_wandb(
    monkeypatch: pytest.MonkeyPatch,
    artifact_factory: object,
) -> type[Exception]:
    """Install a fake ``wandb`` module whose ``Api().artifact(ref)`` delegates.

    :param monkeypatch: Pytest fixture used to patch ``sys.modules``.
    :param artifact_factory: Callable invoked as ``artifact_factory(ref)``.
    :returns: The fake ``wandb.errors.Error`` class, for raising in tests.
    """

    class _FakeWandbError(Exception):
        pass

    class _FakeApi:
        def __init__(self, timeout: int | None = None) -> None:
            self.timeout = timeout

        def artifact(self, ref: str) -> object:
            return artifact_factory(ref)  # type: ignore[operator]

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
    setattr(fake_wandb, "Api", _FakeApi)
    fake_errors = types.ModuleType("wandb.errors")
    setattr(fake_errors, "Error", _FakeWandbError)
    setattr(fake_wandb, "errors", fake_errors)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setitem(sys.modules, "wandb.errors", fake_errors)
    return _FakeWandbError


def test_discover_wandb_artifact_success_returns_producing_run_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The artifact tier returns the cached checkpoint plus the producing run's id.

    :param monkeypatch: Pytest fixture used to stub wandb and the resolver.
    :param tmp_path: Holds the fake resolved checkpoint.
    """
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"weights")
    run = types.SimpleNamespace(id="ffn_simple-20260715T225004231Z")
    artifact = types.SimpleNamespace(logged_by=lambda: run)
    _install_fake_wandb(monkeypatch, lambda ref: artifact)
    import synth_setter.utils.utils as utils_module

    monkeypatch.setattr(utils_module, "resolve_wandb_checkpoint", lambda ref: str(ckpt))

    decision = resume.discover_wandb_artifact_checkpoint("ffn_simple")

    assert decision is not None
    assert decision.source == "wandb-artifact"
    assert decision.ckpt_path == ckpt
    assert decision.wandb_run_id == "ffn_simple-20260715T225004231Z"


def test_discover_wandb_artifact_unlogged_artifact_has_no_run_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``logged_by() is None`` (deleted producing run) degrades to a fresh id.

    :param monkeypatch: Pytest fixture used to stub wandb and the resolver.
    :param tmp_path: Holds the fake resolved checkpoint.
    """
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"weights")
    artifact = types.SimpleNamespace(logged_by=lambda: None)
    _install_fake_wandb(monkeypatch, lambda ref: artifact)
    import synth_setter.utils.utils as utils_module

    monkeypatch.setattr(utils_module, "resolve_wandb_checkpoint", lambda ref: str(ckpt))

    decision = resume.discover_wandb_artifact_checkpoint("ffn_simple")

    assert decision is not None
    assert decision.wandb_run_id is None


def test_discover_wandb_artifact_api_error_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wandb API failure (unknown artifact, auth) degrades to None.

    :param monkeypatch: Pytest fixture used to stub wandb.
    """
    holder: dict[str, type[Exception]] = {}

    def _raise(ref: str) -> object:
        raise holder["error"]("no such artifact")

    holder["error"] = _install_fake_wandb(monkeypatch, _raise)

    assert resume.discover_wandb_artifact_checkpoint("ffn_simple") is None


def test_discover_resume_checkpoint_no_bucket_no_wandb_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no local sibling, no r2.bucket, and no wandb, discovery falls through to None.

    :param monkeypatch: Pytest fixture used to disable the wandb tier.
    :param tmp_path: Empty run-dir family.
    """
    monkeypatch.setattr(resume, "find_spec", lambda name: None)
    current = tmp_path / "ffn-current"
    current.mkdir()
    cfg = OmegaConf.create({"paths": {"output_dir": str(current)}, "r2": {"bucket": None}})

    assert resume.discover_resume_checkpoint(cfg, config_id="ffn_simple") is None


def test_discover_resume_checkpoint_falls_through_r2_to_wandb_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When local misses and R2 creds are unavailable, the wandb-artifact tier wins.

    :param monkeypatch: Pytest fixture used to break R2 and stub wandb.
    :param tmp_path: Empty run-dir family plus the fake resolved checkpoint.
    """
    from synth_setter.pipeline import r2_io

    def _no_creds(*args: object, **kwargs: object) -> None:
        raise RuntimeError("no creds")

    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", _no_creds)
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"weights")
    artifact = types.SimpleNamespace(logged_by=lambda: None)
    _install_fake_wandb(monkeypatch, lambda ref: artifact)
    import synth_setter.utils.utils as utils_module

    monkeypatch.setattr(utils_module, "resolve_wandb_checkpoint", lambda ref: str(ckpt))
    current = tmp_path / "ffn-current"
    current.mkdir()
    cfg = OmegaConf.create(
        {"paths": {"output_dir": str(current)}, "r2": {"bucket": "test-bucket"}}
    )

    decision = resume.discover_resume_checkpoint(cfg, config_id="ffn_simple")

    assert decision is not None
    assert decision.source == "wandb-artifact"


def test_composed_train_config_ships_resume_keys() -> None:
    """The real Hydra-composed train config carries both new resume keys, defaulted off."""
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["datamodule=ksin", "model=ffn", "trainer=cpu"],
        )

    assert OmegaConf.select(cfg, "training.resume") is None
    assert "resume" in cfg.logger.wandb
    assert cfg.logger.wandb.resume is None
