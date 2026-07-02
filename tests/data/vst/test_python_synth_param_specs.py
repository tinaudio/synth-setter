"""Param-spec + registry + render-config coverage for the Python synth backends.

The checked-in ``TORCHSYNTH_PARAM_SPEC`` / ``SYNTHAX_PARAM_SPEC`` modules are
generated from live library introspection; the sync tests here fail if either
library changes its parameter surface, signalling the spec module must be
regenerated (``python -m synth_setter.data.vst.python_synth``).
"""

from importlib.metadata import version as dist_version

import numpy as np
import pytest
from hydra import compose, initialize_config_module

from synth_setter.data.vst.param_spec import (
    ContinuousParameter,
    DiscreteLiteralParameter,
    NoteDurationParameter,
)
from synth_setter.data.vst.param_spec_registry import param_specs, preset_paths
from synth_setter.data.vst.python_synth import SynthaxPlugin, TorchSynthPlugin
from synth_setter.pipeline.schemas.spec import RenderConfig


@pytest.mark.parametrize("name", ["torchsynth", "synthax"])
class TestRegistry:
    """Registry entries for the Python synth backends are complete and coherent."""

    def test_spec_registered(self, name: str) -> None:
        """The backend has a spec in ``param_specs``.

        :param name: Python synth backend under test.
        """
        assert name in param_specs

    def test_preset_path_registered_as_empty_no_preset_sentinel(self, name: str) -> None:
        """The backend's preset entry is the ``""`` no-preset sentinel.

        :param name: Python synth backend under test.
        """
        assert preset_paths[name] == ""

    def test_spec_sample_encode_decode_round_trips(self, name: str) -> None:
        """A sampled patch survives ``encode`` → ``decode`` unchanged.

        :param name: Python synth backend under test.
        """
        spec = param_specs[name]
        synth_params, note_params = spec.sample(np.random.default_rng(7))
        decoded_synth, decoded_note = spec.decode(spec.encode(synth_params, note_params))
        assert decoded_synth == pytest.approx(synth_params)
        assert decoded_note["pitch"] == note_params["pitch"]
        assert decoded_note["note_start_and_end"] == pytest.approx(
            note_params["note_start_and_end"]
        )

    def test_note_params_match_repo_convention(self, name: str) -> None:
        """Note params follow the repo's pitch + note-duration convention.

        :param name: Python synth backend under test.
        """
        spec = param_specs[name]
        pitch, duration = spec.note_params
        assert isinstance(pitch, DiscreteLiteralParameter)
        assert isinstance(duration, NoteDurationParameter)


@pytest.mark.parametrize(
    ("name", "plugin_cls"), [("torchsynth", TorchSynthPlugin), ("synthax", SynthaxPlugin)]
)
class TestSpecMatchesLiveLibrary:
    """The checked-in generated specs stay in sync with live library introspection."""

    def test_spec_names_match_plugin_parameters(self, name: str, plugin_cls: type) -> None:
        """Spec parameter names equal the live plugin's parameter names, in order.

        :param name: Python synth backend under test.
        :param plugin_cls: The backend's adapter class.
        """
        spec = param_specs[name]
        assert [p.name for p in spec.synth_params] == list(plugin_cls().parameters)

    def test_spec_synth_params_are_full_range_continuous(
        self, name: str, plugin_cls: type
    ) -> None:
        """Every generated synth param is a full-range ``ContinuousParameter``.

        :param name: Python synth backend under test.
        :param plugin_cls: The backend's adapter class (unused; keeps the matrix).
        """
        spec = param_specs[name]
        assert all(isinstance(p, ContinuousParameter) for p in spec.synth_params)


@pytest.mark.parametrize("name", ["torchsynth", "synthax"])
class TestRenderConfigCompose:
    """``render=<backend>`` composes into a valid ``RenderConfig``."""

    def test_render_config_composes_with_python_synth_identity(self, name: str) -> None:
        """Hydra ``render=<name>`` yields the backend's identity and cadences.

        :param name: Python synth backend under test.
        """
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            cfg = compose(
                config_name="dataset",
                overrides=[
                    "experiment=generate_dataset/ci-materialize-test",
                    f"render={name}",
                ],
            )
        render = RenderConfig(**{k: cfg.render[k] for k in cfg.render})
        assert render.plugin_path == name
        assert render.preset_path == ""
        assert render.param_spec_name == name
        assert render.renderer_version == dist_version(name)
        assert render.plugin_reload_cadence == "once"
        assert render.gui_toggle_cadence == "never"
