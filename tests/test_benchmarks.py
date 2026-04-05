"""Performance benchmarks for critical code paths."""

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict


@pytest.mark.benchmark
@pytest.mark.slow
def test_config_resolution_speed(benchmark, cfg_train: DictConfig) -> None:
    """Benchmark Hydra config resolution speed.

    Mirrors the production resolution path: Hydra strips its own `hydra` section
    from the config before handing it to the user task, so the benchmark resolves
    the user-facing subtree. `HydraConfig` is still set so that `${hydra:runtime.*}`
    resolver references inside the user tree (e.g. `paths.work_dir`) work.
    """
    HydraConfig().set_config(cfg_train)
    cfg = cfg_train.copy()
    with open_dict(cfg):
        cfg.pop("hydra", None)

    def resolve_config():
        return OmegaConf.to_container(cfg, resolve=True)

    result = benchmark(resolve_config)
    assert result is not None
