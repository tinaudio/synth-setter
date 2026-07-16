"""Tests for ``synth_setter.utils.resume`` — auto-resume checkpoint discovery."""

import os
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
    offline_wandb: bool = False,
    hydra_experiment: str | None = None,
) -> Path:
    """Create one fake Hydra run dir with a ``checkpoints/last.ckpt``.

    :param root: Parent of all sibling run dirs.
    :param name: Run dir basename.
    :param ckpt_mtime: Explicit mtime for ``last.ckpt``; newer wins discovery.
    :param wandb_run_id: When set, adds a ``wandb/run-<ts>-<id>`` dir so the id
        is recoverable.
    :param offline_wandb: Name the wandb dir ``offline-run-*`` instead of ``run-*``.
    :param hydra_experiment: When set, records this experiment choice in a
        ``.hydra/hydra.yaml`` so identity is provable without a wandb dir.
    :returns: The created run dir.
    """
    run_dir = root / name
    ckpt = run_dir / "checkpoints" / "last.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"ckpt")
    if ckpt_mtime is not None:
        os.utime(ckpt, (ckpt_mtime, ckpt_mtime))
    if wandb_run_id is not None:
        prefix = "offline-run" if offline_wandb else "run"
        (run_dir / "wandb" / f"{prefix}-20260715_185004-{wandb_run_id}").mkdir(parents=True)
    if hydra_experiment is not None:
        hydra_dir = run_dir / ".hydra"
        hydra_dir.mkdir()
        (hydra_dir / "hydra.yaml").write_text(
            f"hydra:\n  runtime:\n    choices:\n      experiment: {hydra_experiment}\n"
        )
    return run_dir


def test_discover_local_picks_newest_sibling_last_ckpt(tmp_path: Path) -> None:
    """The newest sibling ``last.ckpt`` by mtime wins.

    :param tmp_path: Pytest tmp dir holding the test's fake directories.
    """
    _make_run_dir(tmp_path, "ffn-old", ckpt_mtime=1_000, hydra_experiment="surge/ffn_simple")
    newest = _make_run_dir(
        tmp_path, "ffn-new", ckpt_mtime=2_000, hydra_experiment="surge/ffn_simple"
    )
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


def test_discover_local_wandb_less_sibling_with_matching_hydra_state_is_accepted(
    tmp_path: Path,
) -> None:
    """A wandb-less run dir (logger disabled) resumes when its Hydra state matches.

    :param tmp_path: Pytest tmp dir holding the test's fake directories.
    """
    _make_run_dir(tmp_path, "ffn-prior", hydra_experiment="surge/ffn_simple")
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


def test_discover_resume_checkpoint_no_local_and_no_bucket_returns_none(
    tmp_path: Path,
) -> None:
    """With no local sibling and no r2.bucket, discovery falls through to None.

    :param tmp_path: Empty run-dir family.
    """
    current = tmp_path / "ffn-current"
    current.mkdir()
    cfg = OmegaConf.create({"paths": {"output_dir": str(current)}, "r2": {"bucket": None}})

    assert resume.discover_resume_checkpoint(cfg, config_id="ffn_simple") is None


def test_discover_resume_checkpoint_unreachable_r2_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When local misses and R2 creds are unavailable, discovery degrades to None.

    :param monkeypatch: Pytest fixture used to break the R2 tier.
    :param tmp_path: Empty run-dir family.
    """
    from synth_setter.pipeline import r2_io

    def _no_creds(*args: object, **kwargs: object) -> None:
        raise RuntimeError("no creds")

    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", _no_creds)
    current = tmp_path / "ffn-current"
    current.mkdir()
    cfg = OmegaConf.create(
        {"paths": {"output_dir": str(current)}, "r2": {"bucket": "test-bucket"}}
    )

    assert resume.discover_resume_checkpoint(cfg, config_id="ffn_simple") is None


def test_run_id_from_run_dir_malformed_wandb_dirname_returns_none(tmp_path: Path) -> None:
    """A wandb dir whose name lacks the ``run-<ts>-<id>`` shape yields no run id.

    :param tmp_path: Pytest tmp dir holding the test's fake directories.
    """
    run_dir = tmp_path / "ffn-prior"
    (run_dir / "wandb" / "run-onlyoneseg").mkdir(parents=True)
    ckpt = run_dir / "checkpoints" / "last.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"ckpt")
    current = tmp_path / "ffn-current"
    current.mkdir()

    decision = discover_local_checkpoint(current, config_id="ffn_simple")

    # A malformed name yields no run id, so identity falls to (absent) Hydra state.
    assert decision is None


def test_config_id_from_hydra_dir_malformed_yaml_returns_none(tmp_path: Path) -> None:
    """Malformed ``.hydra/hydra.yaml`` state cannot prove identity.

    :param tmp_path: Pytest tmp dir holding the test's fake directories.
    """
    run_dir = _make_run_dir(tmp_path, "ffn-prior")
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir()
    (hydra_dir / "hydra.yaml").write_text("hydra: [unclosed")
    current = tmp_path / "ffn-current"
    current.mkdir()

    assert discover_local_checkpoint(current, config_id="ffn_simple") is None


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


def test_discover_local_sibling_without_any_identity_evidence_is_skipped(
    tmp_path: Path,
) -> None:
    """No wandb dir and no ``.hydra`` state means the sibling cannot be trusted.

    :param tmp_path: Pytest tmp dir holding the test's fake directories.
    """
    _make_run_dir(tmp_path, "ffn-prior")
    current = tmp_path / "ffn-current"
    current.mkdir()

    assert discover_local_checkpoint(current, config_id="ffn_simple") is None


def test_discover_local_wandb_less_sibling_with_mismatched_hydra_state_is_skipped(
    tmp_path: Path,
) -> None:
    """A wandb-less sibling recorded for another experiment must not be resumed.

    :param tmp_path: Pytest tmp dir holding the test's fake directories.
    """
    _make_run_dir(tmp_path, "flow-prior", hydra_experiment="surge/flow_simple")
    current = tmp_path / "ffn-current"
    current.mkdir()

    assert discover_local_checkpoint(current, config_id="ffn_simple") is None


def test_discover_local_recovers_run_id_from_offline_wandb_dir(tmp_path: Path) -> None:
    """Offline-mode ``wandb/offline-run-*`` dirs carry a recoverable run id too.

    :param tmp_path: Pytest tmp dir holding the test's fake directories.
    """
    _make_run_dir(
        tmp_path,
        "ffn-prior",
        wandb_run_id="ffn_simple-20260715T225004231Z",
        offline_wandb=True,
    )
    current = tmp_path / "ffn-current"
    current.mkdir()

    decision = discover_local_checkpoint(current, config_id="ffn_simple")

    assert decision is not None
    assert decision.wandb_run_id == "ffn_simple-20260715T225004231Z"


def test_discover_local_prefix_related_config_id_cannot_cross_match(tmp_path: Path) -> None:
    """A ``flow-x`` sibling never matches config_id ``flow`` despite the shared prefix.

    :param tmp_path: Pytest tmp dir holding the test's fake directories.
    """
    _make_run_dir(tmp_path, "flowx-prior", wandb_run_id="flow-x-20260715T225004231Z")
    current = tmp_path / "flow-current"
    current.mkdir()

    assert discover_local_checkpoint(current, config_id="flow") is None


def test_discover_local_equal_mtime_tie_breaks_on_path_string(tmp_path: Path) -> None:
    """Two candidates stamped identically resolve deterministically by path.

    :param tmp_path: Pytest tmp dir holding the test's fake directories.
    """
    _make_run_dir(tmp_path, "ffn-a", ckpt_mtime=1_000, hydra_experiment="surge/ffn_simple")
    later_path = _make_run_dir(
        tmp_path, "ffn-b", ckpt_mtime=1_000, hydra_experiment="surge/ffn_simple"
    )
    current = tmp_path / "ffn-current"
    current.mkdir()

    decision = discover_local_checkpoint(current, config_id="ffn_simple")

    assert decision is not None
    assert decision.ckpt_path == later_path / "checkpoints" / "last.ckpt"


def test_checkpoint_mirror_prefix_prefers_upload_uri_override() -> None:
    """The override's parent wins over the auto-derived bucket prefix."""
    cfg = OmegaConf.create(
        {
            "r2": {"bucket": "other-bucket"},
            "training": {"upload_checkpoints_uri": "r2://custom/my/spot/model.ckpt"},
        }
    )

    assert resume.checkpoint_mirror_prefix(cfg, "ffn_simple") == "r2://custom/my/spot"


def test_checkpoint_mirror_prefix_auto_derives_from_bucket() -> None:
    """Without an override the canonical checkpoints/{config_id} prefix is derived."""
    cfg = OmegaConf.create({"r2": {"bucket": "bkt"}, "training": {}})

    assert resume.checkpoint_mirror_prefix(cfg, "ffn_simple") == "r2://bkt/checkpoints/ffn_simple"


@pytest.mark.parametrize("override", ["r2://", "r2://bucket/", "not-a-uri"])
def test_checkpoint_mirror_prefix_malformed_override_returns_none(override: str) -> None:
    """A malformed override degrades to no-prefix instead of aborting the launch.

    :param override: Parametrized malformed ``training.upload_checkpoints_uri``.
    """
    cfg = OmegaConf.create(
        {"r2": {"bucket": "bkt"}, "training": {"upload_checkpoints_uri": override}}
    )

    assert resume.checkpoint_mirror_prefix(cfg, "ffn_simple") is None
