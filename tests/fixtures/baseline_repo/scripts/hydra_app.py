from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


@dataclass
class Config:
    """Tiny Hydra config schema used by the python-shim test harness."""

    host: str = "localhost"
    port: int = 5432
    url: str = "${host}:${port}"
    task_id: int = 0


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: DictConfig) -> None:
    """No-op entrypoint; the harness invokes via ``--cfg job --resolve``."""
    pass


if __name__ == "__main__":
    main()
