"""Behavior tests for the available-memory OmegaConf resolver."""

from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from synth_setter.utils import utils
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
    monkeypatch.setattr(utils, "_cgroup_available_memory_bytes", lambda: None)
    register_resolvers()
    cfg = OmegaConf.load(_VST_CONFIG)

    assert cfg.high_memory_materialization is expected


def test_cgroup_available_memory_bytes_subtracts_current_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finite cgroup accounting reports the unused portion of its limit.

    :param tmp_path: Holds synthetic cgroup accounting files.
    :param monkeypatch: Redirects cgroup discovery to the synthetic files.
    """
    limit_path = tmp_path / "memory.max"
    usage_path = tmp_path / "memory.current"
    limit_path.write_text(str(48 * _GIB))
    usage_path.write_text(str(16 * _GIB))
    monkeypatch.setattr(utils, "_CGROUP_MEMORY_FILES", ((limit_path, usage_path),))

    assert utils._cgroup_available_memory_bytes() == 32 * _GIB


def test_cgroup_available_memory_bytes_unlimited_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unlimited cgroup leaves host availability as the effective limit.

    :param tmp_path: Holds synthetic cgroup accounting files.
    :param monkeypatch: Redirects cgroup discovery to the synthetic files.
    """
    limit_path = tmp_path / "memory.max"
    usage_path = tmp_path / "memory.current"
    limit_path.write_text("max")
    monkeypatch.setattr(utils, "_CGROUP_MEMORY_FILES", ((limit_path, usage_path),))

    assert utils._cgroup_available_memory_bytes() is None


def test_available_memory_exceeds_gib_uses_lower_cgroup_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A container memory cap prevents host RAM from enabling high-memory mode.

    :param monkeypatch: Supplies distinct host and container memory availability.
    """
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=64 * _GIB),
    )
    monkeypatch.setattr(utils, "_cgroup_available_memory_bytes", lambda: 16 * _GIB)
    register_resolvers()
    cfg = OmegaConf.load(_VST_CONFIG)

    assert cfg.high_memory_materialization is False


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
    monkeypatch.setattr(
        utils,
        "_cgroup_available_memory_bytes",
        lambda: pytest.fail("explicit override unexpectedly read cgroup memory"),
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
