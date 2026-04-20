import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict


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

    Regression guard for #625: the conftest fixture used to override only
    ``max_epochs = 1`` while leaving ``configs/trainer/default.yaml`` step-based
    keys (``min_steps``, ``max_steps``, ``val_check_interval``,
    ``check_val_every_n_epoch=null``) untouched. That mismatch silently
    suppressed validation under ``limit_train_batches=0.01``, breaking every
    test that depends on ``val/loss`` (#47, #619, #620, #624).
    """
    assert cfg_train.trainer.max_epochs == 1
    assert cfg_train.trainer.min_steps is None
    # ``-1`` is Lightning's "unbounded" sentinel; ``None`` trips an internal
    # ``None < int`` comparison on Trainer init.
    assert cfg_train.trainer.max_steps == -1
    assert cfg_train.trainer.val_check_interval is None
    assert cfg_train.trainer.check_val_every_n_epoch == 1


def test_cfg_train_t_max_interpolation_resolves() -> None:
    """Guard: ``${trainer.max_steps}`` interpolation used by surge model configs
    still resolves when the test-mode fixture sets ``trainer.max_steps = -1``
    (Lightning's unbounded sentinel).

    Several ``configs/model/surge_*.yaml`` files interpolate
    ``T_max: ${trainer.max_steps}``. If a future test forgets to guard this,
    composing those configs crashes. This test composes the train config with a
    surge model override and selects ``model.scheduler.T_max`` to force that
    specific interpolation; unrelated interpolations (e.g. env-var resolvers)
    are not exercised here.
    """
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=["model=surge_ffn"],
        )
        with open_dict(cfg):
            cfg.trainer.max_epochs = 1
            cfg.trainer.min_steps = None
            cfg.trainer.max_steps = -1
            cfg.trainer.val_check_interval = None
            cfg.trainer.check_val_every_n_epoch = 1

        # Resolving only T_max exercises the ${trainer.max_steps} interpolation
        # without pulling in unrelated env-var resolvers.
        assert OmegaConf.select(cfg, "model.scheduler.T_max") == -1
    GlobalHydra.instance().clear()


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

    def test_wandb_defaults_when_env_unset(self, monkeypatch):
        """Falls back to tinaudio/synth-setter when env vars unset."""
        monkeypatch.delenv("WANDB_ENTITY", raising=False)
        monkeypatch.delenv("WANDB_PROJECT", raising=False)
        cfg = OmegaConf.load("configs/logger/wandb.yaml")
        assert OmegaConf.select(cfg, "wandb.entity") == "tinaudio"
        assert OmegaConf.select(cfg, "wandb.project") == "synth-setter"
