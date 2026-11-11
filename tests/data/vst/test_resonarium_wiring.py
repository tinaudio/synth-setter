"""Integration contracts for the native Resonarium renderer wiring."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_module
from pydantic import ValidationError

from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.renderer_backend import missing_render_artifacts
from synth_setter.synth_spec import SYNTHS, SynthName, SynthSpec


def _resonarium_render_config(**overrides: object) -> RenderConfig:
    values: dict[str, Any] = {
        "synth": SYNTHS[SynthName("resonarium_py")],
        "renderer_backend": "resonarium",
        "sample_rate": 44_100,
        "channels": 2,
        "velocity": 100,
        "signal_duration_seconds": 4.0,
        "min_loudness": -55.0,
        "audio_dtype": "float16",
        "mel_spec_dtype": "float32",
        "samples_per_shard": 4,
        "plugin_reload_cadence": "render",
        "gui_toggle_cadence": "never",
    }
    values.update(overrides)
    return RenderConfig.model_validate(values)


def test_resonarium_registry_identity_and_param_spec_are_wired() -> None:
    """The native identity resolves its host-normalized parameter specification."""
    from synth_setter.data.vst.param_spec_registry import param_specs

    synth = SYNTHS[SynthName("resonarium_py")]

    assert synth.model_dump(exclude_none=True) == {
        "name": "resonarium_py",
        "param_spec_name": "resonarium_py",
        "format": "resonarium",
        "plugin_path": "resonarium",
        "plugin_state_path": "",
        "synth_version": "0.1.1",
    }
    assert len(param_specs["resonarium_py"].synth_params) == 692
    assert missing_render_artifacts("resonarium", "") == ()


def test_resonarium_format_is_inferred_from_native_backend_name() -> None:
    """Identities recorded without ``format`` infer it from the Resonarium sentinel."""
    values = SYNTHS[SynthName("resonarium_py")].model_dump(exclude={"format"})

    assert SynthSpec.model_validate(values).format == "resonarium"


def test_resonarium_hydra_groups_compose_to_valid_render_config() -> None:
    """The shipped render and synth groups form the required lifecycle."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        render = compose(config_name="render/resonarium").render
        synth = compose(config_name="synth/resonarium_py").synth

    config = RenderConfig.from_cfg_nodes(render, synth)

    assert config.renderer_backend == "resonarium"
    assert config.plugin_reload_cadence == "render"
    assert config.gui_toggle_cadence == "never"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"channels": 3}, "one or two channels"),
        ({"plugin_reload_cadence": "once"}, "plugin_reload_cadence"),
        ({"gui_toggle_cadence": "once"}, "gui_toggle_cadence"),
        ({"signal_duration_seconds": 1.0}, "signal_duration_seconds>=4.0"),
    ],
)
def test_resonarium_render_contract_rejects_incompatible_settings(
    overrides: dict[str, object], message: str
) -> None:
    """Invalid geometry, lifecycle, or a clip shorter than the note window fail at config time.

    :param overrides: Invalid fields replacing the valid Resonarium defaults.
    :param message: Expected validation-error fragment.
    """
    with pytest.raises(ValidationError, match=message):
        _resonarium_render_config(**overrides)


def test_resonarium_render_contract_rejects_preset() -> None:
    """The native backend rejects plugin-host preset paths."""
    synth = SYNTHS[SynthName("resonarium_py")].model_copy(
        update={"plugin_state_path": "presets/resonarium.xml"}
    )

    with pytest.raises(ValidationError, match="no preset"):
        _resonarium_render_config(synth=synth)


@pytest.mark.parametrize(
    ("synth_name", "backend"),
    [("resonarium_py", "pedalboard"), ("surge_xt", "resonarium")],
)
def test_resonarium_backend_and_format_must_be_paired(synth_name: str, backend: str) -> None:
    """Resonarium format and backend cannot be selected independently.

    :param synth_name: Registered synth identity under test.
    :param backend: Incompatible backend paired with that identity.
    """
    with pytest.raises(ValidationError, match="renderer_backend"):
        _resonarium_render_config(synth=SYNTHS[SynthName(synth_name)], renderer_backend=backend)


def test_resonarium_backend_requires_registered_identity() -> None:
    """The native backend accepts only the registered resonarium_py identity."""
    synth = SYNTHS[SynthName("resonarium_py")].model_copy(update={"name": "other"})

    with pytest.raises(ValidationError, match="registered resonarium_py synth identity"):
        _resonarium_render_config(synth=synth)


@pytest.mark.requires_resonarium
def test_resonarium_factory_constructs_native_renderer() -> None:
    """The public factory forwards the full native renderer configuration."""
    from synth_setter.data.vst.resonarium_renderer import ResonariumRenderer
    from synth_setter.renderer_factory import make_audio_renderer

    renderer = make_audio_renderer(_resonarium_render_config(sample_rate=48_000, channels=1))

    assert isinstance(renderer, ResonariumRenderer)
    assert renderer.plugin_path == "resonarium"
    assert renderer.sample_rate == 48_000
    assert renderer.channels == 1
    assert renderer.signal_duration_seconds == 4.0
    assert renderer.plugin_state_path == ""
    assert renderer.synth_version == "0.1.1"


@pytest.mark.requires_resonarium
def test_resonarium_renderer_version_reads_native_package_metadata() -> None:
    """The sentinel reports the installed resonarium-native distribution version."""
    from synth_setter.data.vst.core import extract_renderer_version

    assert extract_renderer_version(Path("resonarium")) == importlib.metadata.version(
        "resonarium-native"
    )
