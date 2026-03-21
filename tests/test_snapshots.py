"""Snapshot tests for config resolution and data structures."""

import pytest


@pytest.mark.snapshot
def test_train_config_snapshot(cfg_train, snapshot):
    """Train config resolves to expected structure."""
    # Convert to dict for snapshot comparison
    from omegaconf import OmegaConf

    cfg_dict = OmegaConf.to_container(cfg_train, resolve=True)
    assert isinstance(cfg_dict, dict)
    # Snapshot the top-level keys and their types
    structure = {k: type(v).__name__ for k, v in cfg_dict.items()}
    assert structure == snapshot
