"""Performance benchmarks for critical code paths."""

import pytest


@pytest.mark.benchmark
@pytest.mark.slow
def test_config_resolution_speed(benchmark, cfg_train):
    """Benchmark Hydra config resolution speed."""
    from omegaconf import OmegaConf

    def resolve_config():
        return OmegaConf.to_container(cfg_train, resolve=True)

    result = benchmark(resolve_config)
    assert result is not None
