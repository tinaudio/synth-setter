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

from synth_setter.data.vst.param_spec_registry import (
    param_specs,
    plugin_state_paths,
    resolve_param_spec_width,
)
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.renderer_backend import TORCHSYNTH_PLUGIN_NAME
from synth_setter.synth_spec import (
    SYNTHS,
    SynthName,
    SynthSpec,
    resolve_synth,
    validate_faust_registry_reference,
    validate_synth_identity,
)

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
                synth_version="1.0.2",
            )

    def test_legacy_faust_sentinel_without_legacy_backend_is_rejected(self) -> None:
        """The Faust sentinel is migrated only as part of the exact legacy render pair."""
        with pytest.raises(ValidationError, match="format"):
            SynthSpec(
                name=SynthName("faust_bubble"),
                param_spec_name=ParamSpecName("faust_bubble"),
                plugin_path="faust",
                plugin_state_path="",
                synth_version="0.8.3",
            )

    def test_a_vst_plugin_accepts_a_preset_path(self) -> None:
        """A plugin-hosted synth carries the baseline preset it was mapped against."""
        spec = SynthSpec(
            name=SynthName("obxf"),
            param_spec_name=ParamSpecName("obxf"),
            plugin_path="plugins/OB-Xf.vst3",
            plugin_state_path="presets/obxf-base.vstpreset",
            synth_version="1.0.3",
        )

        assert spec.plugin_state_path == "presets/obxf-base.vstpreset"

    def test_missing_synth_version_is_rejected(self) -> None:
        """A synth identity without an artifact version is incomplete."""
        with pytest.raises(ValidationError, match="synth_version"):
            SynthSpec(  # type: ignore[call-arg]
                name=SynthName("obxf"),
                param_spec_name=ParamSpecName("obxf"),
                plugin_path="plugins/OB-Xf.vst3",
                plugin_state_path="presets/obxf-base.vstpreset",
            )

    def test_blank_synth_version_is_rejected(self) -> None:
        """Whitespace cannot stand in for an artifact version."""
        with pytest.raises(ValidationError, match="synth_version must not be blank"):
            SynthSpec(
                name=SynthName("obxf"),
                param_spec_name=ParamSpecName("obxf"),
                plugin_path="plugins/OB-Xf.vst3",
                plugin_state_path="presets/obxf-base.vstpreset",
                synth_version="  ",
            )

    def test_misspelled_synth_version_is_rejected(self) -> None:
        """A misspelled field cannot satisfy the required canonical field."""
        with pytest.raises(ValidationError) as exc_info:
            SynthSpec(
                name=SynthName("obxf"),
                param_spec_name=ParamSpecName("obxf"),
                plugin_path="plugins/OB-Xf.vst3",
                plugin_state_path="presets/obxf-base.vstpreset",
                synth_verion="1.0.3",  # type: ignore[call-arg]
            )

        error_types = {error["type"] for error in exc_info.value.errors()}
        assert error_types == {"extra_forbidden", "missing"}

    def test_identity_is_frozen(self) -> None:
        """Identity cannot be mutated after construction."""
        spec = resolve_synth(SynthName("obxf"))

        with pytest.raises(ValidationError):
            spec.plugin_path = "plugins/Other.vst3"  # type: ignore[misc]

    def test_unknown_name_raises_key_error(self) -> None:
        """Resolving an unregistered synth fails loudly rather than returning a default."""
        with pytest.raises(KeyError):
            resolve_synth(SynthName("not_a_synth"))

    def test_faust_registry_reference_returns_registered_identity(self) -> None:
        """A canonical reference resolves to its checked-in source identity."""
        identity = validate_faust_registry_reference(
            "registry://faust/faust_bright_organ", "faust_bright_organ"
        )

        assert identity == "faust_bright_organ"

    @pytest.mark.parametrize(
        "reference",
        [
            "registry:/faust/faust_bright_organ",
            "registry://other/faust_bright_organ",
            "registry://faust/faust_bright_organ/extra",
            "registry://faust/faust_bright_organ?version=1",
        ],
    )
    def test_malformed_faust_registry_reference_raises(self, reference: str) -> None:
        """Only the exact registry scheme, namespace, and one-part identity are accepted.

        :param reference: Malformed registry reference under test.
        """
        with pytest.raises(ValueError, match="registry://faust/<registered-source-name>"):
            validate_faust_registry_reference(reference, "faust_bright_organ")

    def test_unknown_faust_registry_reference_raises(self) -> None:
        """A canonical-looking reference cannot select an unregistered source."""
        with pytest.raises(ValueError, match="not registered"):
            validate_faust_registry_reference("registry://faust/faust_unknown", "faust_unknown")

    def test_mismatched_faust_registry_reference_raises(self) -> None:
        """The reference identity must agree with the selected parameter specification."""
        with pytest.raises(ValueError, match="faust_bubble.*faust_bright_organ"):
            validate_faust_registry_reference(
                "registry://faust/faust_bubble", "faust_bright_organ"
            )

    def test_registry_reference_without_format_derives_faust(self) -> None:
        """A recognized registry URI supplies its non-filesystem representation format."""
        values = SYNTHS[SynthName("faust_bright_organ")].model_dump(exclude={"format"})

        spec = SynthSpec.model_validate(values)

        assert spec.format == "faust"

    def test_registry_reference_with_mismatched_explicit_format_raises(self) -> None:
        """An explicitly authored format cannot contradict a Faust registry URI."""
        values = SYNTHS[SynthName("faust_bright_organ")].model_dump()
        values["format"] = "vst3"
        values["source_sha256"] = None

        with pytest.raises(ValidationError, match="requires format='faust'"):
            SynthSpec.model_validate(values)


class TestSynthsTable:
    """Cross-registry invariants that previously had no enforcement."""

    def test_faust_identity_declares_registered_source_reference(self) -> None:
        """Faust source identity uses the in-process registry rather than a file path."""
        synth = SYNTHS[SynthName("faust_bright_organ")]

        assert synth.format == "faust"
        assert synth.plugin_path == "registry://faust/faust_bright_organ"
        assert synth.synth_version == "1"
        assert synth.source_sha256 == (
            "a1bf9f6e45ebbf78dd11fc18603cda048a91a778af1ad79683339b1951813465"
        )

    @pytest.mark.parametrize("name", _ALL_SYNTHS)
    def test_every_identity_declares_its_supported_format(self, name: str) -> None:
        """Every registered synth identifies the representation consumed by its backend.

        :param name: Registry key under test.
        """
        assert SYNTHS[SynthName(name)].format in {
            "faust",
            "pyfdn",
            "surgepy",
            "torchsynth",
            "vst3",
        }

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

        ``DawDreamerRenderer`` requires the map, so this pins which specs that
        backend actually supports instead of leaving it to a mid-shard failure.

        :param name: Registry key under test.
        """
        from importlib.resources import files

        synth = SYNTHS[SynthName(name)]
        packaged = (
            files("synth_setter") / "data" / "vst" / f"{synth.param_spec_name}_param_map.json"
        ).is_file()

        assert packaged == (
            synth.param_spec_name
            in {
                "cardinal",
                "surge_4",
                "surge_simple",
                "surge_xt",
                "ultramaster_kr106",
                "ultramaster_kr106_onehot",
                "ultramaster_kr106_single_note",
            }
        )

    @pytest.mark.parametrize(
        "variant", ["surge_xt_surgepy", "surge_simple_surgepy", "surge_4_surgepy"]
    )
    def test_surgepy_variant_shares_its_base_param_spec(self, variant: str) -> None:
        """Each surgepy rendering variant reuses its base synth's spec identity.

        :param variant: surgepy registry key under test.
        """
        base = SYNTHS[SynthName(variant.removesuffix("_surgepy"))]
        surgepy = SYNTHS[SynthName(variant)]

        assert surgepy.param_spec_name == base.param_spec_name
        assert surgepy.plugin_path == "surgepy"
        assert surgepy.plugin_state_path.endswith(".fxp")


class TestSynthConfigGroup:
    """``configs/synth`` is a generated artifact of ``SYNTHS``, pinned here."""

    @pytest.mark.parametrize("name", _ALL_SYNTHS)
    def test_synth_group_matches_the_table(self, name: str) -> None:
        """Each shipped group states exactly what the table declares.

        The YAML is checked in so ``--register`` and the fake-synth end-to-end test
        can compose from a temp checkout without importing its Python. This test is
        what keeps the two from drifting.

        :param name: Registry key / synth group under test.
        """
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            group = compose(config_name=f"synth/{name}").synth

        expected = SYNTHS[SynthName(name)].model_dump(exclude_none=True)
        if expected["format"] == "faust":
            expected.pop("format")
        assert OmegaConf.to_container(group) == expected

    def test_ultramaster_onehot_selector_resolves_configured_width(self) -> None:
        """The opt-in Hydra selector and width resolver agree on 250 columns."""
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            group = compose(config_name="synth/ultramaster_kr106_onehot").synth

        assert group.name == "ultramaster_kr106_onehot"
        assert group.param_spec_name == "ultramaster_kr106_onehot"
        assert resolve_param_spec_width(group.param_spec_name) == 250

    def test_group_covers_every_registered_synth(self) -> None:
        """No table entry lacks a config group, and no group lacks a table entry."""
        group_dir = files("synth_setter") / "configs" / "synth"
        shipped = {p.name.removesuffix(".yaml") for p in group_dir.iterdir()}

        assert shipped == set(SYNTHS)

    def test_per_run_plugin_path_override_survives(self) -> None:
        """An overridden plugin path reaches the identity, not the registry default.

        Dataset tests swap in a stub bundle so the renderer-version gate passes without a real
        install; pinning binding fields to the registry would discard that.
        """
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            cfg = compose(
                config_name="synth/obxf",
                overrides=["synth.plugin_path=plugins/TestPlugin.vst3"],
            )
        synth = validate_synth_identity(cfg)

        assert synth is not None
        assert synth.plugin_path == "plugins/TestPlugin.vst3"


def _synth_node(row: str, **overrides: str) -> dict[str, str]:
    """Return the ``SYNTHS`` row for ``row`` as a plain composed-node dict.

    :param row: Registry key.
    :param **overrides: Field values replacing the registry row's.
    :returns: Five-field mapping mirroring ``configs/synth/<row>.yaml``.
    """
    return {**SYNTHS[SynthName(row)].model_dump(), **overrides}


class TestValidateSynthIdentity:
    """Compose-time guard for the root ``synth`` identity group (#2565)."""

    def test_registered_identity_returns_composed_spec(self) -> None:
        """A well-formed selection validates and returns the composed identity."""
        cfg = OmegaConf.create({"synth": _synth_node("surge_xt")})
        assert validate_synth_identity(cfg) == SYNTHS[SynthName("surge_xt")]

    def test_absent_synth_node_is_a_noop(self) -> None:
        """Audio/legacy configs compose no synth group and must pass untouched."""
        assert validate_synth_identity(OmegaConf.create({"datamodule": {"k": 8}})) is None

    def test_unregistered_name_raises(self) -> None:
        """A name with no ``SYNTHS`` row is rejected at validation time."""
        cfg = OmegaConf.create({"synth": _synth_node("surge_xt", name="nope")})
        with pytest.raises(ValueError, match="nope"):
            validate_synth_identity(cfg)

    def test_param_spec_mismatching_registry_row_raises(self) -> None:
        """A spec that contradicts the registry row is rejected."""
        cfg = OmegaConf.create({"synth": _synth_node("surge_xt", param_spec_name="surge_4")})
        with pytest.raises(ValueError, match="surge_4"):
            validate_synth_identity(cfg)

    def test_surge_xt_overridden_to_surgepy_format_requires_variant_identity(self) -> None:
        """A backend-format override cannot retain the VST3 registry identity."""
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            cfg = compose(
                config_name="synth/surge_xt",
                overrides=[
                    "synth.format=surgepy",
                    "synth.plugin_path=surgepy",
                    "synth.plugin_state_path=presets/surge-base.fxp",
                ],
            )

        with pytest.raises(ValueError, match="surge_xt_surgepy"):
            validate_synth_identity(cfg)

    def test_overridden_binding_fields_pass_and_survive(self) -> None:
        """Plugin binding stays per-run overridable; only identity is registry-pinned."""
        cfg = OmegaConf.create(
            {"synth": _synth_node("surge_xt", plugin_path="plugins/TestPlugin.vst3")}
        )
        spec = validate_synth_identity(cfg)

        assert spec is not None
        assert spec.plugin_path == "plugins/TestPlugin.vst3"

    def test_extra_key_in_synth_node_raises(self) -> None:
        """The group mirrors ``SynthSpec`` exactly; stray keys are a config error."""
        cfg = OmegaConf.create({"synth": {**_synth_node("surge_xt"), "width": "300"}})
        with pytest.raises(ValidationError, match="width"):
            validate_synth_identity(cfg)

    def test_datamodule_literal_spec_mismatch_raises(self) -> None:
        """A CLI-forced datamodule spec that skews from the synth fails loudly."""
        cfg = OmegaConf.create(
            {
                "synth": _synth_node("surge_xt"),
                "datamodule": {"param_spec_name": "surge_4"},
            }
        )
        with pytest.raises(ValueError, match="surge_4"):
            validate_synth_identity(cfg)

    def test_datamodule_interpolated_spec_matches_and_passes(self) -> None:
        """The shipped rootward interpolation always agrees with the synth node."""
        cfg = OmegaConf.create(
            {
                "synth": _synth_node("surge_4"),
                "datamodule": {"param_spec_name": "${synth.param_spec_name}"},
            }
        )
        assert validate_synth_identity(cfg) == SYNTHS[SynthName("surge_4")]
