"""Shared fixtures and helpers for ``tests/schemas/``.

Must NOT chain into ``tests/conftest.py``'s ``lightning``/``torch``/``h5py``
imports — the schemas suite stays importable on a minimal install. Run as
``pytest tests/schemas/ --confcutdir=tests/schemas``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

__all__ = ["_to_dict", "compose_subtree", "compose_train_cfg"]

_DEFAULT_OVERRIDES = ["data=ksin", "model=ffn", "trainer=cpu"]


@pytest.fixture(autouse=True)
def clean_global_hydra() -> Iterator[None]:
    """Clear Hydra's global singleton before and after every test in this package."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    yield
    assert not GlobalHydra.instance().is_initialized(), (
        "Hydra leaked from a test in tests/schemas/"
    )


def _to_dict(node: Any) -> dict[str, Any]:  # noqa: DOC101,DOC103,DOC201,DOC203
    """Resolve an OmegaConf node to ``dict[str, Any]`` for ``model_validate``."""
    return cast("dict[str, Any]", OmegaConf.to_container(node, resolve=False))


def compose_train_cfg(  # noqa: DOC101,DOC103,DOC201,DOC203
    *,
    overrides: list[str] | None = None,
    return_hydra_config: bool = False,
) -> dict[str, Any]:
    """Compose ``configs/train.yaml`` and return it as a plain dict.

    Default overrides pin ``data=ksin model=ffn trainer=cpu`` so the suite
    doesn't depend on root-config ``???`` sentinels; caller overrides are
    appended after.
    """
    selected_overrides = list(_DEFAULT_OVERRIDES)
    if overrides is not None:
        selected_overrides.extend(overrides)
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=return_hydra_config,
            overrides=selected_overrides,
        )
    return _to_dict(cfg)


def compose_subtree(group: str, name: str) -> dict[str, Any]:  # noqa: DOC101,DOC103,DOC201,DOC203
    """Compose ``train.yaml`` with ``<group>=<name>`` selected and return that subtree.

    The subtree must be a dict; groups that compose to ``None`` (e.g.
    ``callbacks/none.yaml``) are unsupported and surfaced via assertion.
    """
    cfg_dict = compose_train_cfg(overrides=[f"{group}={name}"])
    subtree = cfg_dict[group]
    assert isinstance(subtree, dict), (
        f"compose_subtree({group}={name}) produced {type(subtree).__name__}, not dict"
    )
    return cast("dict[str, Any]", subtree)
