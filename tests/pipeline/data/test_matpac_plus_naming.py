"""Public MATPAC++ embedding naming contract."""

from __future__ import annotations

import importlib.util

from synth_setter.pipeline.data.add_embeddings import EMBEDDING_REGISTRY


def test_matpac_plus_registry_replaces_tinymu_name() -> None:
    """Expose the encoder identity without retaining the wrapper-package alias."""
    assert "matpac_plus" in EMBEDDING_REGISTRY
    assert "tinymu" not in EMBEDDING_REGISTRY


def test_matpac_plus_module_replaces_tinymu_module() -> None:
    """Name the integration module for its stored representation."""
    assert importlib.util.find_spec("synth_setter.pipeline.data.matpac_plus") is not None
    assert importlib.util.find_spec("synth_setter.pipeline.data.tinymu") is None
