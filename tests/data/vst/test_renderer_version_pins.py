"""Pin each render group's ``renderer_version`` against the artifact it describes.

``renderer_version`` is a plugin-binary fact that the launcher must be able to
declare with no plugin present, so it is hand-written into
``configs/render/<synth>.yaml`` and only cross-checked at worker startup
(``cli.generate_dataset``). These tests move that check earlier: a stale pin fails
here instead of after a shard has been dispatched to a worker.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_module

from synth_setter.data.vst.core import extract_renderer_version
from synth_setter.renderer_backend import TORCHSYNTH_PLUGIN_NAME
from tests._vst import TEST_RENDERER_VERSION, TEST_SYNTH

_VST_RENDER_GROUPS = ["obxf", "surge_4", "surge_simple", "surge_xt"]
_TORCHSYNTH_RENDER_GROUPS = ["torchsynth_adsr", "torchsynth_full", "torchsynth_simple"]


def _render_group(group: str) -> tuple[str, str]:
    """Compose one render group and read the identity fields this module checks.

    :param group: Render group name below ``configs/render``.
    :returns: ``(plugin_path, renderer_version)`` as the group declares them.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name=f"render/{group}").render
    return cfg.plugin_path, cfg.renderer_version


def test_test_renderer_version_matches_the_selected_synths_render_group() -> None:
    """``tests/_vst.py`` reports the pin its own render group declares.

    Guards the shared test constant against re-acquiring a hardcoded table that could disagree with
    the shipped configs.
    """
    _, renderer_version = _render_group(TEST_SYNTH)

    assert TEST_RENDERER_VERSION == renderer_version


@pytest.mark.parametrize("group", _VST_RENDER_GROUPS + _TORCHSYNTH_RENDER_GROUPS)
def test_every_render_group_resolves_a_renderer_version(group: str) -> None:
    """Every registered synth composes a non-blank pin, torchsynth backends included.

    The previous hardcoded table covered only the four VST synths, so selecting a
    torchsynth backend via ``SYNTH_SETTER_TEST_SYNTH`` raised ``KeyError`` at import.

    :param group: Render group name under test.
    """
    _, renderer_version = _render_group(group)

    assert renderer_version.strip()


@pytest.mark.requires_vst
@pytest.mark.parametrize("group", _VST_RENDER_GROUPS)
def test_vst_render_group_pins_the_installed_plugin_version(group: str) -> None:
    """Each VST group's pin matches the version read off the plugin bundle.

    :param group: Render group name under test.
    """
    plugin_path, renderer_version = _render_group(group)

    assert extract_renderer_version(Path(plugin_path)) == renderer_version


@pytest.mark.parametrize("group", _TORCHSYNTH_RENDER_GROUPS)
def test_torchsynth_render_group_pins_the_installed_package_version(group: str) -> None:
    """Each torchsynth group's pin matches the installed package version.

    No plugin bundle is involved, so this runs everywhere the package is importable.

    :param group: Render group name under test.
    """
    plugin_path, renderer_version = _render_group(group)

    assert plugin_path == TORCHSYNTH_PLUGIN_NAME
    assert extract_renderer_version(Path(plugin_path)) == renderer_version
