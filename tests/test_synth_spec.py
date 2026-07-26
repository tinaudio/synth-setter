"""Tests for the ``SYNTHS`` identity table and its Hydra bridge.

``SynthSpec`` is the single authoring point for a synth's identity — which param
spec, which plugin, which baseline preset. These tests pin the table's invariants
against the registries and render groups that previously restated the same facts.
"""

from __future__ import annotations

from importlib.resources import files

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from pydantic import ValidationError

from synth_setter.data.vst.param_spec_registry import param_specs, plugin_state_paths
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.renderer_backend import TORCHSYNTH_PLUGIN_NAME
from synth_setter.synth_spec import SYNTHS, SynthName, SynthSpec, resolve_synth

_ALL_SYNTHS = sorted(SYNTHS)


class TestSynthSpecValidation:
    """Construction-time invariants a registered identity must satisfy."""

    def test_torchsynth_backend_rejects_a_preset_path(self) -> None:
        """The in-process backend has no preset file, so naming one is a config error."""
        with pytest.raises(ValidationError, match="preset"):
            SynthSpec(
                name=SynthName("bogus"),
                param_spec_name=ParamSpecName("torchsynth_full"),
                plugin_path=TORCHSYNTH_PLUGIN_NAME,
                plugin_state_path="presets/nope.vstpreset",
            )

    def test_a_vst_plugin_accepts_a_preset_path(self) -> None:
        """A plugin-hosted synth carries the baseline preset it was mapped against."""
        spec = SynthSpec(
            name=SynthName("obxf"),
            param_spec_name=ParamSpecName("obxf"),
            plugin_path="plugins/OB-Xf.vst3",
            plugin_state_path="presets/obxf-base.vstpreset",
        )

        assert spec.plugin_state_path == "presets/obxf-base.vstpreset"

    def test_identity_is_frozen(self) -> None:
        """Identity cannot be mutated after construction."""
        spec = resolve_synth(SynthName("obxf"))

        with pytest.raises(ValidationError):
            spec.plugin_path = "plugins/Other.vst3"  # type: ignore[misc]

    def test_unknown_name_raises_key_error(self) -> None:
        """Resolving an unregistered synth fails loudly rather than returning a default."""
        with pytest.raises(KeyError):
            resolve_synth(SynthName("not_a_synth"))


class TestSynthsTable:
    """Cross-registry invariants that previously had no enforcement."""

    @pytest.mark.parametrize("name", _ALL_SYNTHS)
    def test_every_entry_names_a_registered_param_spec(self, name: str) -> None:
        """Each identity points at a ParamSpec the registry can resolve.

        :param name: Registry key under test.
        """
        assert SYNTHS[SynthName(name)].param_spec_name in param_specs

    @pytest.mark.parametrize("name", _ALL_SYNTHS)
    def test_entry_key_matches_its_own_name_field(self, name: str) -> None:
        """The mapping key and the entry's ``name`` cannot disagree.

        :param name: Registry key under test.
        """
        assert SYNTHS[SynthName(name)].name == name

    def test_plugin_state_paths_agrees_with_the_table(self) -> None:
        """The legacy preset mapping states the same presets the table declares.

        ``param_spec_registry.plugin_state_paths`` stays a literal dict because
        ``registration.registry_with_spec`` rewrites it by line anchor; this pins the
        two against each other until that transform learns to write ``SYNTHS``.
        """
        assert plugin_state_paths == {
            synth.name: synth.plugin_state_path for synth in SYNTHS.values()
        }

    def test_param_specs_covers_every_registered_synth(self) -> None:
        """No identity names a spec the ParamSpec registry cannot resolve."""
        assert {synth.param_spec_name for synth in SYNTHS.values()} <= set(param_specs)

    @pytest.mark.parametrize("name", _ALL_SYNTHS)
    def test_dawdreamer_capable_synths_ship_a_parameter_map(self, name: str) -> None:
        """A synth is DawDreamer-renderable exactly when its param map is packaged.

        ``DawDreamerRenderer`` requires the map, so this pins which synths that
        backend actually supports instead of leaving it to a mid-shard failure.

        :param name: Registry key under test.
        """
        from importlib.resources import files

        synth = SYNTHS[SynthName(name)]
        packaged = (
            files("synth_setter") / "data" / "vst" / f"{synth.param_spec_name}_param_map.json"
        ).is_file()

        assert packaged == (name in {"surge_4", "surge_simple", "surge_xt"})


class TestFromRenderCfg:
    """The Hydra bridge every raw-config identity read goes through."""

    @pytest.mark.parametrize("name", _ALL_SYNTHS)
    def test_composed_render_group_yields_the_registered_identity(self, name: str) -> None:
        """Each shipped render group resolves to the identity the table declares.

        This is the check that catches a render YAML drifting from ``SYNTHS``.

        :param name: Render group / registry key under test.
        """
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            render = compose(config_name=f"render/{name}").render

        assert SynthSpec.from_render_cfg(render) == SYNTHS[SynthName(name)]

    def test_absent_render_group_yields_none(self) -> None:
        """A missing render node reports absence rather than raising."""
        assert SynthSpec.from_render_cfg(None) is None

    def test_generic_vst_scaffold_yields_none(self) -> None:
        """``render=vst`` carries knobs but no synth identity, so it resolves to nothing."""
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            render = compose(config_name="render/vst").render

        assert SynthSpec.from_render_cfg(render) is None

    def test_per_run_plugin_path_override_survives(self) -> None:
        """An overridden plugin path reaches the identity, not the registry default.

        Dataset tests swap in a stub bundle so the renderer-version gate passes without a real
        install; reading the registry instead would discard that.
        """
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            render = compose(
                config_name="render/obxf",
                overrides=["render.synth.plugin_path=plugins/TestPlugin.vst3"],
            ).render

        synth = SynthSpec.from_render_cfg(render)

        assert synth is not None
        assert synth.plugin_path == "plugins/TestPlugin.vst3"


class TestSynthConfigGroup:
    """``configs/render/synth`` is a generated artifact of ``SYNTHS``, pinned here."""

    @pytest.mark.parametrize("name", _ALL_SYNTHS)
    def test_synth_group_matches_the_table(self, name: str) -> None:
        """Each shipped group states exactly what the table declares.

        The YAML is checked in so ``--register`` and the fake-synth end-to-end test
        can compose from a temp checkout without importing its Python. This test is
        what keeps the two from drifting.

        :param name: Registry key / synth group under test.
        """
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            group = compose(config_name=f"render/synth/{name}").render.synth

        assert OmegaConf.to_container(group) == SYNTHS[SynthName(name)].model_dump()

    def test_group_covers_every_registered_synth(self) -> None:
        """No table entry lacks a config group, and no group lacks a table entry."""
        group_dir = files("synth_setter") / "configs" / "render" / "synth"
        shipped = {p.name.removesuffix(".yaml") for p in group_dir.iterdir()}

        assert shipped == set(SYNTHS)
