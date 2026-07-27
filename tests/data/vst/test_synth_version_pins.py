"""Pin each synth group's ``synth_version`` against its runtime artifact.

The version belongs to synth identity and is composed from
``configs/render/synth/<synth>.yaml``. Workers still inspect the installed
artifact through ``extract_renderer_version`` before rendering.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_module

from synth_setter.data.vst.core import extract_renderer_version
from synth_setter.renderer_backend import TORCHSYNTH_PLUGIN_NAME
from synth_setter.synth_spec import SYNTHS, SynthName
from tests._vst import TEST_SYNTH, TEST_SYNTH_VERSION

_VST_SYNTHS = sorted(
    str(name) for name, synth in SYNTHS.items() if Path(synth.plugin_path).suffix == ".vst3"
)
_TORCHSYNTH_SYNTHS = sorted(
    str(name) for name, synth in SYNTHS.items() if synth.plugin_path == TORCHSYNTH_PLUGIN_NAME
)


def _composed_synth(group: str) -> tuple[str, str]:
    """Compose one render group and read its artifact identity.

    :param group: Render group name below ``configs/render``.
    :returns: ``(plugin_path, synth_version)`` declared by the synth group.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        synth = compose(config_name=f"render/{group}").render.synth
    return synth.plugin_path, synth.synth_version


def test_test_synth_version_matches_the_selected_synth_group() -> None:
    """The shared VST fixture reports the selected synth group's version."""
    _, synth_version = _composed_synth(TEST_SYNTH)

    assert TEST_SYNTH_VERSION == synth_version


@pytest.mark.parametrize("group", sorted(SYNTHS))
def test_every_synth_group_resolves_a_synth_version(group: str) -> None:
    """Every registered synth composes the non-blank registry version.

    :param group: Synth group name under test.
    """
    _, synth_version = _composed_synth(group)

    assert synth_version == SYNTHS[SynthName(group)].synth_version
    assert synth_version.strip()


@pytest.mark.requires_vst
@pytest.mark.parametrize("group", _VST_SYNTHS)
def test_vst_synth_group_pins_the_installed_plugin_version(group: str) -> None:
    """Each VST synth's pin matches the version read from its plugin bundle.

    :param group: Synth group name under test.
    """
    plugin_path, synth_version = _composed_synth(group)

    assert extract_renderer_version(Path(plugin_path)) == synth_version


@pytest.mark.parametrize("group", _TORCHSYNTH_SYNTHS)
def test_torchsynth_group_pins_the_installed_package_version(group: str) -> None:
    """Each torchsynth pin matches the installed package version.

    :param group: Synth group name under test.
    """
    plugin_path, synth_version = _composed_synth(group)

    assert plugin_path == TORCHSYNTH_PLUGIN_NAME
    assert extract_renderer_version(Path(plugin_path)) == synth_version
