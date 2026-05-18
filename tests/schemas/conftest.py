"""Shared fixtures and helpers for the ``tests/schemas/`` module.

The Hydra global-state dance (``GlobalHydra.instance().clear()`` before each
``initialize`` block) belongs in a fixture rather than copy-pasted into every
helper — that way an xdist-parallelised run can't race on the singleton, and
a mid-compose exception in one test can't leak initialised state into the
next.

This conftest must NOT chain into ``tests/conftest.py``'s heavy imports
(``lightning``, ``torch``, ``h5py``) — composing Hydra configs is a
documentation-layer concern and we want the suite to stay importable on a
minimal install. Run as ``pytest tests/schemas/ --confcutdir=tests/schemas``
(matches ``.github/workflows/docs.yml``).
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
    # Defense against a test that forgot to use ``initialize()`` as a context
    # manager and leaked initialised state into the next test in the package.
    assert not GlobalHydra.instance().is_initialized(), (
        "Hydra leaked from a test in tests/schemas/"
    )


def _to_dict(node: Any) -> dict[str, Any]:  # noqa: DOC101,DOC103,DOC201,DOC203
    """Resolve an OmegaConf node to a typed ``dict[str, Any]`` for pydantic.

    Centralises the ``Any`` boundary so individual test modules don't need
    to import ``cast`` just to convert an OmegaConf container to a plain
    dict that ``model_validate`` accepts.
    """
    return cast("dict[str, Any]", OmegaConf.to_container(node, resolve=False))


def compose_train_cfg(  # noqa: DOC101,DOC103,DOC201,DOC203
    *,
    overrides: list[str] | None = None,
    return_hydra_config: bool = False,
) -> dict[str, Any]:
    """Compose ``configs/train.yaml`` and return it as a plain dict.

    The default overrides pin ``data``, ``model``, and ``trainer`` to
    composable leaves so the suite doesn't depend on the ``???``
    mandatory-override sentinels in the root config. Caller-supplied
    overrides are appended after the defaults so the caller can switch
    one composition group without re-spelling the others.

    The ``clean_global_hydra`` autouse fixture owns the
    ``GlobalHydra.instance().clear()`` lifecycle, so this helper just
    calls ``initialize`` directly inside its own context manager.
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
    """Compose one Hydra group at a specific name and return its subtree.

    Example: ``compose_subtree("data", "ksin")`` returns the ``data``
    subtree from ``train.yaml`` with ``data=ksin`` selected. The subtree
    must be a dict (groups that compose to ``None``, e.g. an empty
    ``callbacks/none.yaml``, are not supported here — callers test those
    via the parametrized discovery helper).
    """
    cfg_dict = compose_train_cfg(overrides=[f"{group}={name}"])
    subtree = cfg_dict[group]
    assert isinstance(subtree, dict), (
        f"compose_subtree({group}={name}) produced {type(subtree).__name__}, not dict"
    )
    return cast("dict[str, Any]", subtree)
