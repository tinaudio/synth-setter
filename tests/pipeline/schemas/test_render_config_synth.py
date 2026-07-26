"""Tests for ``RenderConfig``'s nested synth identity and its legacy-shape bridge.

Identity moved from three flat fields onto a nested ``SynthSpec``. Read access is
preserved by properties, and already-serialized specs (which carry the flat keys)
must still parse, so both shapes are exercised here.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.synth_spec import SYNTHS, SynthName, SynthSpec

_KNOBS: dict[str, Any] = {
    "renderer_version": "1.3.4",
    "sample_rate": 44100,
    "channels": 2,
    "velocity": 100,
    "signal_duration_seconds": 4.0,
    "min_loudness": -55.0,
    "samples_per_shard": 32,
}


def _legacy_payload(**overrides: Any) -> dict[str, Any]:
    r"""Build a render payload in the pre-nesting flat shape.

    :param \*\*overrides: Fields replacing the Surge XT defaults.
    :returns: A render mapping using flat identity keys.
    """
    return {
        **_KNOBS,
        "param_spec_name": "surge_xt",
        "plugin_path": "plugins/Surge XT.vst3",
        "plugin_state_path": "presets/surge-base.vstpreset",
        **overrides,
    }


class TestNestedIdentity:
    """Construction through the nested field and the properties that read it."""

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


class TestLegacySpecCompatibility:
    """Already-serialized specs on R2 carry flat identity keys and must still parse."""

    def test_flat_identity_keys_lift_into_the_nested_field(self) -> None:
        """A pre-nesting payload parses, with identity hoisted onto ``synth``."""
        cfg = RenderConfig(**_legacy_payload())

        assert cfg.synth.param_spec_name == "surge_xt"
        assert cfg.synth.plugin_state_path == "presets/surge-base.vstpreset"

    def test_lifted_entry_name_mirrors_the_param_spec_name(self) -> None:
        """Flat payloads name no synth separately, so the key mirrors the spec."""
        cfg = RenderConfig(**_legacy_payload())

        assert cfg.synth.name == "surge_xt"

    def test_a_stub_plugin_path_survives_the_lift(self) -> None:
        """Per-run plugin overrides in old specs are preserved, not replaced by defaults."""
        cfg = RenderConfig(**_legacy_payload(plugin_path="plugins/TestPlugin.vst3"))

        assert cfg.plugin_path == "plugins/TestPlugin.vst3"

    def test_mixing_both_shapes_is_rejected(self) -> None:
        """A payload carrying both shapes is ambiguous rather than silently preferring one."""
        with pytest.raises(ValidationError):
            RenderConfig(synth=SYNTHS[SynthName("obxf")], **_legacy_payload())

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
