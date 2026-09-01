"""Tests for ``RenderConfig``'s nested synth identity.

Identity fields are nested under ``SynthSpec`` while flat read properties preserve
access to the plugin and parameter-spec fields used throughout rendering.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.schemas.spec import PyFDNEffectConfig, RenderConfig
from synth_setter.synth_spec import SYNTHS, SynthName, SynthSpec

_KNOBS: dict[str, Any] = {
    "sample_rate": 44100,
    "channels": 2,
    "velocity": 100,
    "signal_duration_seconds": 4.0,
    "min_loudness": -55.0,
    "samples_per_shard": 32,
}


class TestNestedIdentity:
    """Construction through the nested field and the properties that read it."""

    def test_renderer_version_is_not_a_render_config_field(self) -> None:
        """Artifact versioning belongs to synth identity, not render mechanics."""
        assert "renderer_version" not in RenderConfig.model_fields

    def test_nested_synth_is_readable_through_the_flat_properties(self) -> None:
        """The three former fields still read, now delegating to the nested identity."""
        cfg = RenderConfig(synth=SYNTHS[SynthName("obxf")], **_KNOBS)

        assert cfg.param_spec_name == "obxf"
        assert cfg.plugin_path == "plugins/OB-Xf.vst3"
        assert cfg.plugin_state_path == "presets/obxf-base.vstpreset"

    def test_param_spec_name_delegates_to_param_spec_name_not_the_entry_name(self) -> None:
        """A preset variant reports the spec it shares, not its own registry key."""
        variant = SynthSpec(
            name=SynthName("obxf_bright"),
            param_spec_name=ParamSpecName("obxf"),
            plugin_path="plugins/OB-Xf.vst3",
            plugin_state_path="presets/obxf-bright.vstpreset",
            synth_version="1.0.3",
        )

        cfg = RenderConfig(synth=variant, **_KNOBS)

        assert cfg.param_spec_name == "obxf"

    def test_torchsynth_backend_still_requires_the_bare_plugin_name(self) -> None:
        """The backend cross-check reads through the nested identity."""
        with pytest.raises(ValidationError, match="torchsynth"):
            RenderConfig(
                synth=SYNTHS[SynthName("surge_xt")],
                renderer_backend="torchsynth",
                gui_toggle_cadence="never",
                **_KNOBS,
            )


class TestPyFDNEffectConfig:
    """Validation and serialization for the optional live render effect."""

    def test_render_config_defaults_to_dry_audio(self) -> None:
        """Existing render configs remain dry unless the effect is explicitly enabled."""
        cfg = RenderConfig(synth=SYNTHS[SynthName("obxf")], **_KNOBS)

        assert cfg.pyfdn_effect is None

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("decay_seconds", 0.0),
            ("decay_seconds", -1.0),
            ("decay_seconds", 1e-20),
            ("decay_seconds", float("inf")),
            ("wet_mix", 0.0),
            ("wet_mix", 1.01),
        ],
    )
    def test_effect_rejects_out_of_range_values(self, field: str, value: float) -> None:
        """Decay must remain finite in pyFDN and wet mix stays in the enabled interval.

        :param field: Effect field under test.
        :param value: Invalid boundary value.
        """
        values = {
            "package_version": "0.4.2",
            "preset_name": "colorless_N8_d1",
            "decay_seconds": 1.5,
            "wet_mix": 0.1,
            field: value,
        }

        with pytest.raises(ValidationError):
            PyFDNEffectConfig(**values)  # type: ignore[arg-type]

    def test_effect_rejects_incompatible_render_sample_rate(self) -> None:
        """The bundled preset cannot silently run against non-48 kHz audio."""
        with pytest.raises(ValidationError, match="sample_rate=48000"):
            RenderConfig(
                synth=SYNTHS[SynthName("obxf")],
                pyfdn_effect=PyFDNEffectConfig(
                    package_version="0.4.2",
                    preset_name="colorless_N8_d1",
                    decay_seconds=1.5,
                    wet_mix=0.1,
                ),
                **_KNOBS,
            )

    def test_effect_round_trip_preserves_provenance(self) -> None:
        """Effect identity and controls survive persisted spec JSON."""
        cfg = RenderConfig(
            synth=SYNTHS[SynthName("obxf")],
            pyfdn_effect=PyFDNEffectConfig(
                package_version="0.4.2",
                preset_name="colorless_N8_d1",
                decay_seconds=1.5,
                wet_mix=0.1,
            ),
            **(_KNOBS | {"sample_rate": 48000}),
        )

        restored = RenderConfig.model_validate_json(cfg.model_dump_json())

        assert restored.pyfdn_effect == cfg.pyfdn_effect


class TestBreakingVersionMigration:
    """The removed render-level version cannot satisfy synth identity."""

    def test_old_renderer_version_cannot_supply_missing_synth_version(self) -> None:
        """The removed render-level pin is ignored rather than promoted."""
        synth = SYNTHS[SynthName("surge_xt")].model_dump(exclude={"synth_version"})

        with pytest.raises(ValidationError, match="synth_version"):
            RenderConfig(
                synth=synth,  # type: ignore[arg-type]
                renderer_version="1.3.4",  # type: ignore[call-arg]
                **_KNOBS,
            )

    def test_old_renderer_version_is_ignored_when_synth_version_is_present(self) -> None:
        """A stale render-level pin cannot override the canonical synth identity."""
        cfg = RenderConfig(
            synth=SYNTHS[SynthName("obxf")],
            renderer_version="9.9.9",  # type: ignore[call-arg]
            **_KNOBS,
        )

        assert cfg.synth.synth_version == "1.0.3"
        assert "renderer_version" not in cfg.model_dump()

    def test_round_trip_through_json_preserves_identity(self) -> None:
        """A dumped config reparses, so spec upload and re-read stay lossless."""
        cfg = RenderConfig(synth=SYNTHS[SynthName("surge_4")], **_KNOBS)

        restored = RenderConfig.model_validate(json.loads(cfg.model_dump_json()))

        assert restored.synth == cfg.synth


class TestSerializedShape:
    """What ``model_dump`` emits, which the worker argv and spec JSON depend on."""

    def test_identity_serializes_only_under_the_nested_key(self) -> None:
        """Flat keys are gone from the dump, so the CLI cannot receive duplicates.

        ``_GenerateCliArgs`` is ``extra="forbid"``; re-emitting the flat keys
        alongside ``synth`` would hard-fail the renderer subprocess at parse.
        """
        dumped = RenderConfig(synth=SYNTHS[SynthName("obxf")], **_KNOBS).model_dump()

        assert "synth" in dumped
        assert not {"param_spec_name", "plugin_path", "plugin_state_path"} & dumped.keys()

    def test_model_copy_over_a_flat_property_does_not_change_identity(self) -> None:
        """Updating a read-only property through ``model_copy`` is inert, by design.

        Properties are data descriptors, so they shadow anything ``model_copy``
        injects into the instance dict. Callers must copy ``synth`` itself; this
        pins the trap rather than leaving it to be rediscovered.
        """
        cfg = RenderConfig(synth=SYNTHS[SynthName("obxf")], **_KNOBS)

        copied = cfg.model_copy(update={"param_spec_name": "surge_xt"})

        assert copied.param_spec_name == "obxf"

    def test_model_copy_over_synth_replaces_identity(self) -> None:
        """Copying the nested field is the supported way to retarget a render."""
        cfg = RenderConfig(synth=SYNTHS[SynthName("obxf")], **_KNOBS)

        copied = cfg.model_copy(update={"synth": SYNTHS[SynthName("surge_xt")]})

        assert copied.param_spec_name == "surge_xt"
        assert copied.plugin_state_path == "presets/surge-base.vstpreset"

    def test_a_non_mapping_payload_passes_through_to_pydantic(self) -> None:
        """The lift only touches mappings, leaving other shapes to normal validation."""
        with pytest.raises(ValidationError):
            RenderConfig.model_validate("not-a-mapping")
