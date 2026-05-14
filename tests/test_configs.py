"""Tests that Hydra config groups compose without errors."""

from typing import Any

import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from tests.conftest import _build_surge_xt_smoke_cfg


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig) -> None:
    """Tests the evaluation configuration provided by the `cfg_eval` pytest fixture.

    :param cfg_train: A DictConfig containing a valid evaluation configuration.
    """
    assert cfg_eval
    assert cfg_eval.data
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.data)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)


def test_cfg_train_trainer_keys_coherent_with_test_mode(cfg_train: DictConfig) -> None:
    """Guard: ``cfg_train`` fixture produces a coherent epoch-based trainer config.

    Regression guard for #625: the original mismatch was that
    ``configs/trainer/default.yaml`` shipped step-based keys (``min_steps``,
    ``max_steps``) which the fixture never unset, silently suppressing
    validation under ``limit_train_batches=0.01`` (#47, #619, #620, #624).

    The fix on this branch removes those keys from ``trainer/default.yaml``
    entirely and pins dataset shape via ``train_val_test_sizes`` instead of
    fractional ``limit_*_batches``. The guard now asserts the structural
    invariant: step-based keys must not be present in the composed trainer.
    """
    assert cfg_train.trainer.max_epochs == 1
    assert cfg_train.trainer.check_val_every_n_epoch == 1
    assert cfg_train.trainer.val_check_interval == 1
    assert "min_steps" not in cfg_train.trainer
    assert "max_steps" not in cfg_train.trainer


class TestWandbConfigResolvesFromEnv:
    """Verify wandb entity/project resolve from env vars (#265)."""

    def test_wandb_entity_resolves_from_env(self, monkeypatch):
        """OmegaConf resolves WANDB_ENTITY from environment."""
        monkeypatch.setenv("WANDB_ENTITY", "test-entity")
        cfg = OmegaConf.load("configs/logger/wandb.yaml")
        assert OmegaConf.select(cfg, "wandb.entity") == "test-entity"

    def test_wandb_project_resolves_from_env(self, monkeypatch):
        """OmegaConf resolves WANDB_PROJECT from environment."""
        monkeypatch.setenv("WANDB_PROJECT", "test-project")
        cfg = OmegaConf.load("configs/logger/wandb.yaml")
        assert OmegaConf.select(cfg, "wandb.project") == "test-project"

    def test_wandb_entity_defaults_to_none_when_env_unset(self, monkeypatch):
        """Entity falls back to None (user's default W&B entity) when env var unset."""
        monkeypatch.delenv("WANDB_ENTITY", raising=False)
        cfg = OmegaConf.load("configs/logger/wandb.yaml")
        assert OmegaConf.select(cfg, "wandb.entity") is None

    def test_wandb_project_defaults_to_synth_setter_when_env_unset(self, monkeypatch):
        """Project falls back to synth-setter when env var unset."""
        monkeypatch.delenv("WANDB_PROJECT", raising=False)
        cfg = OmegaConf.load("configs/logger/wandb.yaml")
        assert OmegaConf.select(cfg, "wandb.project") == "synth-setter"


# Volatile cfg branches that legitimately differ between the fixture builder and the YAML
# compose: ``paths`` is filesystem-anchored at ``rootutils.find_root`` time, ``hydra``
# carries the per-invocation runtime metadata, and ``task_name`` is interpolated from the
# entry-point script name (test runner vs. ``train.py``).
_VOLATILE_TOP_KEYS = ("paths", "hydra", "task_name")


def _strip_volatile(cfg_dict: dict[Any, Any]) -> dict[Any, Any]:
    """Drop top-level keys whose values legitimately differ across compose contexts."""
    return {k: v for k, v in cfg_dict.items() if k not in _VOLATILE_TOP_KEYS}


def _diff_dicts(a: dict[Any, Any], b: dict[Any, Any], prefix: str = "") -> list[str]:
    """Recursively diff two dicts and return human-readable difference lines."""
    diffs: list[str] = []
    for key in sorted(set(a) | set(b)):
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in a:
            diffs.append(f"  + {path} (only in test-mps): {b[key]!r}")
            continue
        if key not in b:
            diffs.append(f"  - {path} (only in fixture): {a[key]!r}")
            continue
        va, vb = a[key], b[key]
        if isinstance(va, dict) and isinstance(vb, dict):
            diffs.extend(_diff_dicts(va, vb, path))
        elif va != vb:
            diffs.append(f"  ~ {path}: fixture={va!r}  test-mps={vb!r}")
    return diffs


def test_test_mps_yaml_matches_cfg_surge_xt_global() -> None:
    """``configs/experiment/surge/test-mps.yaml`` must resolve to the same cfg the surge smoke.

    fixture builds for ``accelerator="mps"`` and ``param_spec_name="surge_4"``.

    Guard against silent drift: the fixture's open_dict bake-ins and the YAML's defaults
    list / overrides must stay in lockstep, otherwise a test that uses one and an
    ``experiment=surge/test-mps`` invocation that uses the other will produce different
    runs without anyone noticing. Builds both configs in-process (no MPS hardware
    needed — only the cfg shape is compared, not runtime behavior).
    """
    fixture_cfg = _build_surge_xt_smoke_cfg(accelerator="mps", param_spec_name="surge_4")
    fixture_d_out = fixture_cfg.model.net.d_out
    GlobalHydra.instance().clear()

    with initialize(version_base="1.3", config_path="../configs"):
        experiment_cfg = compose(
            config_name="train.yaml",
            return_hydra_config=False,
            overrides=[
                "experiment=surge/test-mps",
                f"model.net.d_out={fixture_d_out}",
            ],
        )
    GlobalHydra.instance().clear()

    # ``resolve=False`` keeps interpolation strings (``${paths.output_dir}``,
    # ``${hydra:...}``) verbatim on both sides so the comparison doesn't require an
    # active HydraConfig and doesn't trip on runtime-only resolvers.
    fixture_dict = _strip_volatile(OmegaConf.to_container(fixture_cfg, resolve=False))  # type: ignore[arg-type]
    experiment_dict = _strip_volatile(OmegaConf.to_container(experiment_cfg, resolve=False))  # type: ignore[arg-type]

    diffs = _diff_dicts(fixture_dict, experiment_dict)
    assert not diffs, (
        "test-mps.yaml drifted from cfg_surge_xt_global(mps, surge_4):\n" + "\n".join(diffs)
    )
