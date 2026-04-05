"""Performance benchmarks for critical code paths."""

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


@pytest.mark.benchmark
@pytest.mark.slow
def test_config_resolution_speed(benchmark, cfg_train: DictConfig) -> None:
    """Benchmark Hydra config resolution speed."""
    HydraConfig().set_config(cfg_train)

    def resolve_config():
        return OmegaConf.to_container(cfg_train, resolve=True)

    result = benchmark(resolve_config)
    assert result is not None
