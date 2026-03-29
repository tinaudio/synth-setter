"""Integration tests for W&B config wiring.

These tests verify that wandb entity/project resolve from environment variables via OmegaConf's
oc.env resolver. We use OmegaConf.select() to resolve individual keys without requiring the full
Hydra config (other fields like save_dir reference ${paths.output_dir} which is only available in a
composed Hydra config).
"""

from omegaconf import OmegaConf


class TestWandbConfigResolvesFromEnv:
    """Task 1.2: entity/project must resolve from env vars, not hardcoded."""

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
