"""Integration contracts for the native dexed-py renderer wiring."""

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


def _dexed_render_config(**overrides: object) -> RenderConfig:
    values: dict[str, Any] = {
        "synth": SYNTHS[SynthName("dexed_py")],
        "renderer_backend": "dexed",
        "sample_rate": 44_100,
        "channels": 2,
        "velocity": 100,
        "signal_duration_seconds": 4.0,
        "min_loudness": -55.0,
        "audio_dtype": "float32",
        "mel_spec_dtype": "float32",
        "samples_per_shard": 4,
        "plugin_reload_cadence": "render",
        "gui_toggle_cadence": "never",
    }
    values.update(overrides)
    return RenderConfig.model_validate(values)


def test_dexed_registry_identity_and_param_spec_are_wired() -> None:
    """The native identity resolves its scalar parameter specification."""
    from synth_setter.data.vst.param_spec_registry import param_specs

    synth = SYNTHS[SynthName("dexed_py")]

    assert synth.model_dump(exclude_none=True) == {
        "name": "dexed_py",
        "param_spec_name": "dexed_py",
        "format": "dexed",
        "plugin_path": "dexed",
        "plugin_state_path": "",
        "synth_version": "0.3.0",
    }
    assert len(param_specs["dexed_py"].synth_params) == 145
    assert missing_render_artifacts("dexed", "") == ()


def test_dexed_format_is_inferred_from_native_backend_name() -> None:
    """Legacy identities infer the native format from the Dexed sentinel."""
    values = SYNTHS[SynthName("dexed_py")].model_dump(exclude={"format"})

    assert SynthSpec.model_validate(values).format == "dexed"


def test_dexed_hydra_groups_compose_to_valid_render_config() -> None:
    """The shipped render and synth groups form the required lifecycle."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        render = compose(config_name="render/dexed").render
        synth = compose(config_name="synth/dexed_py").synth

    config = RenderConfig.from_cfg_nodes(render, synth)

    assert config.renderer_backend == "dexed"
    assert config.audio_dtype == "float32"
    assert config.plugin_reload_cadence == "render"
    assert config.gui_toggle_cadence == "never"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"channels": 3}, "one or two channels"),
        ({"plugin_reload_cadence": "once"}, "plugin_reload_cadence"),
        ({"gui_toggle_cadence": "once"}, "gui_toggle_cadence"),
    ],
)
def test_dexed_render_contract_rejects_incompatible_lifecycle(
    overrides: dict[str, object], message: str
) -> None:
    """Invalid native geometry and lifecycle settings fail schema validation.

    :param overrides: Invalid fields replacing the valid Dexed defaults.
    :param message: Expected validation-error fragment.
    """
    with pytest.raises(ValidationError, match=message):
        _dexed_render_config(**overrides)


def test_dexed_render_contract_rejects_preset() -> None:
    """The native backend rejects plugin-host preset paths."""
    synth = SYNTHS[SynthName("dexed_py")].model_copy(
        update={"plugin_state_path": "presets/dexed.syx"}
    )

    with pytest.raises(ValidationError, match="no preset"):
        _dexed_render_config(synth=synth)


@pytest.mark.parametrize(
    ("synth_name", "backend"),
    [("dexed_py", "pedalboard"), ("surge_xt", "dexed")],
)
def test_dexed_backend_and_format_must_be_paired(synth_name: str, backend: str) -> None:
    """Dexed format and backend cannot be selected independently.

    :param synth_name: Registered synth identity under test.
    :param backend: Incompatible backend paired with that identity.
    """
    with pytest.raises(ValidationError, match="renderer_backend"):
        _dexed_render_config(synth=SYNTHS[SynthName(synth_name)], renderer_backend=backend)


def test_dexed_backend_requires_registered_identity() -> None:
    """The native backend accepts only the registered dexed_py identity."""
    synth = SYNTHS[SynthName("dexed_py")].model_copy(update={"name": "other"})

    with pytest.raises(ValidationError, match="registered dexed_py synth identity"):
        _dexed_render_config(synth=synth)


def test_dexed_factory_constructs_native_renderer() -> None:
    """The public factory forwards the full native renderer configuration."""
    from synth_setter.data.vst.dexed_renderer import DexedRenderer
    from synth_setter.renderer_factory import make_audio_renderer

    renderer = make_audio_renderer(_dexed_render_config(sample_rate=48_000, channels=1))

    assert isinstance(renderer, DexedRenderer)
    assert renderer.plugin_path == "dexed"
    assert renderer.sample_rate == 48_000
    assert renderer.channels == 1
    assert renderer.signal_duration_seconds == 4.0
    assert renderer.plugin_state_path == ""
    assert renderer.synth_version == "0.3.0"


def test_dexed_renderer_version_reads_native_package_metadata() -> None:
    """The sentinel reports the installed dexed-py distribution version."""
    from synth_setter.data.vst.core import extract_renderer_version

    assert extract_renderer_version(Path("dexed")) == importlib.metadata.version("dexed-py")
