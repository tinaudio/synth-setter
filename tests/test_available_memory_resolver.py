"""Behavior tests for the available-memory OmegaConf resolver."""

from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from synth_setter.utils.utils import register_resolvers

_GIB = 1024**3
_VST_CONFIG = Path(__file__).parents[1] / "src/synth_setter/configs/datamodule/vst.yaml"


@pytest.mark.parametrize(
    ("available_bytes", "expected"),
    [
        (31 * _GIB, False),
        (32 * _GIB, False),
        (33 * _GIB, True),
    ],
    ids=["below", "equal", "above"],
)
def test_available_memory_exceeds_gib_boundary_resolves_strictly_above(
    available_bytes: int,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resolver enables only when currently available RAM exceeds the threshold.

    :param available_bytes: Available memory reported by the operating system.
    :param expected: Expected resolver result.
    :param monkeypatch: Replaces the operating-system memory reading.
    """
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=available_bytes),
    )
    register_resolvers()
    cfg = OmegaConf.load(_VST_CONFIG)

    assert cfg.high_memory_materialization is expected


@pytest.mark.parametrize("override", [False, True], ids=["false", "true"])
def test_high_memory_materialization_explicit_bool_override_is_preserved(
    override: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit boolean replaces the automatic memory-based default.

    :param override: Explicit value supplied by an operator.
    :param monkeypatch: Makes resolver evaluation fail if an override does not replace it.
    """
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: pytest.fail("explicit override unexpectedly read available memory"),
    )
    register_resolvers()
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(version_base="1.3", config_dir=str(_VST_CONFIG.parent)):
            cfg = compose(
                config_name="vst",
                overrides=[f"high_memory_materialization={str(override).lower()}"],
            )
    finally:
        GlobalHydra.instance().clear()

    assert cfg.high_memory_materialization is override
