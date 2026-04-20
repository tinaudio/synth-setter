"""Tests for src/utils/instantiators.py — focus on the WANDB_API_KEY gate (#598)."""

from omegaconf import DictConfig, OmegaConf

from src.utils.instantiators import instantiate_loggers


def _make_wandb_cfg() -> DictConfig:
    cfg = OmegaConf.create(
        {
            "wandb": {
                "_target_": "lightning.pytorch.loggers.wandb.WandbLogger",
                "save_dir": ".",
                "project": "test",
            }
        }
    )
    assert isinstance(cfg, DictConfig)
    return cfg


def _make_csv_cfg() -> DictConfig:
    """CSVLogger needs no extra deps and ships with Lightning, safe for unit tests."""
    cfg = OmegaConf.create(
        {
            "csv": {
                "_target_": "lightning.pytorch.loggers.csv_logs.CSVLogger",
                "save_dir": ".",
                "name": "test",
            }
        }
    )
    assert isinstance(cfg, DictConfig)
    return cfg


def _make_wandb_plus_csv_cfg() -> DictConfig:
    cfg = OmegaConf.create(
        {
            "wandb": {
                "_target_": "lightning.pytorch.loggers.wandb.WandbLogger",
                "save_dir": ".",
                "project": "test",
            },
            "csv": {
                "_target_": "lightning.pytorch.loggers.csv_logs.CSVLogger",
                "save_dir": ".",
                "name": "test",
            },
        }
    )
    assert isinstance(cfg, DictConfig)
    return cfg


class TestInstantiateLoggersWandbGate:
    """WandbLogger is skipped when WANDB_API_KEY is unset (#598)."""

    def test_skips_wandb_when_api_key_unset(self, monkeypatch) -> None:
        """Wandb config with no API key produces an empty logger list."""
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        loggers = instantiate_loggers(_make_wandb_cfg())
        assert loggers == []

    def test_instantiates_wandb_when_api_key_set(self, monkeypatch) -> None:
        """Wandb config with API key set produces a WandbLogger instance."""
        monkeypatch.setenv("WANDB_API_KEY", "fake-key-for-test")
        loggers = instantiate_loggers(_make_wandb_cfg())
        assert len(loggers) == 1
        assert type(loggers[0]).__name__ == "WandbLogger"

    def test_non_wandb_logger_unaffected_by_api_key(self, monkeypatch) -> None:
        """Non-W&B loggers (e.g. CSVLogger) always instantiate regardless of API key."""
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        loggers = instantiate_loggers(_make_csv_cfg())
        assert len(loggers) == 1
        assert type(loggers[0]).__name__ == "CSVLogger"

    def test_mixed_config_skips_only_wandb_when_api_key_unset(self, monkeypatch) -> None:
        """Mixed logger config skips only the W&B entry; others still instantiate."""
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        loggers = instantiate_loggers(_make_wandb_plus_csv_cfg())
        assert len(loggers) == 1
        assert type(loggers[0]).__name__ == "CSVLogger"
