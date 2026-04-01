import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


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


class TestWandbConfigResolvesFromEnv:
    """Verify wandb entity/project resolve from env vars (#265)."""

    def test_wandb_entity_resolves_from_env(self, monkeypatch):
        # plumb:req-e12fa149
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
        # plumb:req-1755adf6
        """Falls back to tinaudio/synth-setter when env vars unset."""
        monkeypatch.delenv("WANDB_ENTITY", raising=False)
        monkeypatch.delenv("WANDB_PROJECT", raising=False)
        cfg = OmegaConf.load("configs/logger/wandb.yaml")
        assert OmegaConf.select(cfg, "wandb.entity") == "tinaudio"
        assert OmegaConf.select(cfg, "wandb.project") == "synth-setter"
