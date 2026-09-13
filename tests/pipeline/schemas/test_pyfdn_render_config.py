"""Offline pyFDN render configuration contracts."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import synth_setter.renderer_backend as renderer_backend_contract
from synth_setter.data.pyfdn_param_spec import (
    PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC,
    PYFDN_N8_MONO_KRONECKER_PARAM_SPEC,
    pyfdn_param_spec_sha256,
)
from synth_setter.pipeline.schemas.spec import InputAudioSource, RenderConfig


def _pyfdn_render_kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "synth": {
            "name": "pyfdn_n8_mono_householder",
            "param_spec_name": "pyfdn_n8_mono_householder",
            "format": "pyfdn",
            "plugin_path": "pyfdn",
            "plugin_state_path": "",
            "synth_version": "0.4.2",
            "source_sha256": pyfdn_param_spec_sha256(PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC),
        },
        "renderer_backend": "pyfdn",
        "pyfdn_excitation": "impulse",
        "sample_rate": 44_100,
        "channels": 1,
        "velocity": 0,
        "signal_duration_seconds": 4.0,
        "min_loudness": -55.0,
        "audio_dtype": "float32",
        "mel_spec_dtype": "float32",
        "samples_per_render_batch": 1,
        "samples_per_shard": 1,
        "param_sample_cadence": "sample",
        "plugin_reload_cadence": "render",
        "gui_toggle_cadence": "never",
    }
    values.update(overrides)
    return values


def _input_audio_source(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "dataset_uri": "source-dataset",
        "split": "train",
        "snapshot_txid": "txid-123",
        "sampling_seed": 17,
    }
    values.update(overrides)
    return values


def test_input_audio_source_valid_values_are_frozen() -> None:
    """Source identity cannot drift after config validation."""
    source = InputAudioSource.model_validate(_input_audio_source())

    with pytest.raises(ValidationError, match="frozen"):
        source.sampling_seed = 18


@pytest.mark.parametrize("field", ["dataset_uri", "snapshot_txid"])
def test_input_audio_source_blank_identity_raises(field: str) -> None:
    """Blank dataset and transaction identities fail closed.

    :param field: Identity field replaced with whitespace.
    """
    with pytest.raises(ValidationError, match=field):
        InputAudioSource.model_validate(_input_audio_source(**{field: "  "}))


def test_input_audio_source_unsupported_uri_scheme_raises() -> None:
    """Network schemes outside R2 cannot enter worker materialization."""
    with pytest.raises(ValidationError, match="dataset_uri"):
        InputAudioSource.model_validate(
            _input_audio_source(dataset_uri="https://example.test/data")
        )


def test_pyfdn_render_config_uses_existing_renderer_stubs() -> None:
    """PyFDN supplies fixed values required by the MIDI-shaped contract."""
    render = RenderConfig.model_validate(_pyfdn_render_kwargs())

    assert render.velocity == 0
    assert render.pyfdn_excitation == "impulse"
    assert render.plugin_reload_cadence == "render"
    assert render.gui_toggle_cadence == "never"


def test_pyfdn_legacy_json_without_source_digest_restores_registered_identity() -> None:
    """Persisted pre-digest specs gain the registered canonical source identity."""
    values = _pyfdn_render_kwargs(render_contract_version=2)
    synth = values["synth"]
    assert isinstance(synth, dict)
    del synth["source_sha256"]

    restored = RenderConfig.model_validate_json(json.dumps(values))

    assert restored.synth.source_sha256 == pyfdn_param_spec_sha256(
        PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC
    )


def test_pyfdn_authored_config_without_source_digest_is_rejected() -> None:
    """New Python-side configurations must state the canonical source identity."""
    values = _pyfdn_render_kwargs()
    synth = values["synth"]
    assert isinstance(synth, dict)
    del synth["source_sha256"]

    with pytest.raises(ValidationError, match="registered source_sha256"):
        RenderConfig.model_validate(values)


def test_pyfdn_render_config_omitted_excitation_defaults_digest_to_impulse() -> None:
    """Specs predating the field retain the new impulse-response default."""
    explicit = RenderConfig.model_validate(_pyfdn_render_kwargs())
    legacy_kwargs = _pyfdn_render_kwargs()
    del legacy_kwargs["pyfdn_excitation"]
    omitted = RenderConfig.model_validate(legacy_kwargs)

    assert omitted.pyfdn_excitation is None
    assert (
        omitted.shard_metadata().render_contract_digest
        == explicit.shard_metadata().render_contract_digest
    )


def test_pyfdn_render_contract_digest_includes_source_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The source byte identity participates in each shard contract.

    :param monkeypatch: Temporary canonical source digest replacement.
    """
    render = RenderConfig.model_validate(_pyfdn_render_kwargs(pyfdn_excitation="chirp"))
    original = render.shard_metadata().render_contract_digest
    monkeypatch.setattr(renderer_backend_contract, "PYFDN_CANONICAL_SOURCE_SHA256", "0" * 64)

    assert render.shard_metadata().render_contract_digest != original


def test_pyfdn_dataset_input_requires_pyfdn_backend() -> None:
    """Hosted renderers reject the pyFDN-only external input contract."""
    with pytest.raises(
        ValidationError, match="input_audio_source requires renderer_backend='pyfdn'"
    ):
        RenderConfig.model_validate(
            _pyfdn_render_kwargs(
                renderer_backend="pedalboard",
                pyfdn_excitation=None,
                input_audio_source=_input_audio_source(),
            )
        )


def test_pyfdn_dataset_input_rejects_chirp_excitation() -> None:
    """A dataset input cannot compete with the explicit chirp source."""
    with pytest.raises(ValidationError, match="pyfdn_excitation='chirp'"):
        RenderConfig.model_validate(
            _pyfdn_render_kwargs(
                pyfdn_excitation="chirp",
                input_audio_source=_input_audio_source(),
            )
        )


def test_pyfdn_dataset_input_rejects_legacy_contract() -> None:
    """The v1 digest projection cannot silently omit dataset input identity."""
    with pytest.raises(ValidationError, match="render_contract_version=1"):
        RenderConfig.model_validate(
            _pyfdn_render_kwargs(
                render_contract_version=1,
                input_audio_source=_input_audio_source(),
            )
        )


def test_pyfdn_dataset_input_contract_digest_includes_source_identity() -> None:
    """Distinct pinned snapshots cannot finalize into one output dataset."""
    first = RenderConfig.model_validate(
        _pyfdn_render_kwargs(render_contract_version=2, input_audio_source=_input_audio_source())
    )
    second = RenderConfig.model_validate(
        _pyfdn_render_kwargs(
            render_contract_version=2,
            input_audio_source=_input_audio_source(snapshot_txid="txid-456"),
        )
    )

    assert (
        first.shard_metadata().render_contract_digest
        != second.shard_metadata().render_contract_digest
    )


def test_pyfdn_dataset_input_projects_excitation_as_dataset() -> None:
    """Dataset input and built-in impulse runs carry distinct contracts."""
    dataset_input = RenderConfig.model_validate(
        _pyfdn_render_kwargs(render_contract_version=2, input_audio_source=_input_audio_source())
    )
    impulse = RenderConfig.model_validate(_pyfdn_render_kwargs())

    assert (
        dataset_input.shard_metadata().render_contract_digest
        != impulse.shard_metadata().render_contract_digest
    )


def test_pyfdn_render_contract_digest_distinguishes_excitation() -> None:
    """Impulse-response and chirp datasets cannot finalize into one dataset."""
    impulse = RenderConfig.model_validate(_pyfdn_render_kwargs())
    chirp = RenderConfig.model_validate(_pyfdn_render_kwargs(pyfdn_excitation="chirp"))

    assert (
        chirp.shard_metadata().render_contract_digest
        != impulse.shard_metadata().render_contract_digest
    )


def test_render_contract_digest_ignores_shard_seed_position() -> None:
    """Shard placement does not alter the shared render identity."""
    render = RenderConfig.model_validate(_pyfdn_render_kwargs())
    relocated = render.model_copy(
        update={"base_seed": 123, "retain_local_shards": False, "sample_offset": 456}
    )

    assert (
        relocated.shard_metadata().render_contract_digest
        == render.shard_metadata().render_contract_digest
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("sample_rate", 48_000, "sample_rate"),
        ("channels", 2, "channels"),
        ("velocity", 1, "velocity=0"),
        ("audio_dtype", "float16", "audio_dtype"),
        ("mel_spec_dtype", "float16", "mel_spec_dtype"),
        ("param_sample_cadence", "shard", "param_sample_cadence"),
        ("plugin_reload_cadence", "once", "plugin_reload_cadence"),
        ("gui_toggle_cadence", "render", "gui_toggle_cadence"),
    ],
)
def test_pyfdn_render_config_noncanonical_contract_raises(
    field: str,
    value: object,
    match: str,
) -> None:
    """PyFDN rejects settings that its renderer would otherwise ignore.

    :param field: Render field changed from its canonical value.
    :param value: Invalid field value.
    :param match: Expected validation diagnostic.
    """
    with pytest.raises(ValidationError, match=match):
        RenderConfig.model_validate(_pyfdn_render_kwargs(**{field: value}))


def test_pyfdn_param_spec_with_hosted_synth_rejects_hosted_backend() -> None:
    """A registered pyFDN spec cannot route through a hosted synth identity."""
    synth = {
        "name": "surge_4",
        "param_spec_name": "pyfdn_n8_mono_householder",
        "plugin_path": "plugins/Surge XT.vst3",
        "plugin_state_path": "presets/surge-mini.vstpreset",
        "synth_version": "1.3.4",
    }

    with pytest.raises(
        ValidationError, match="all pyFDN identities require renderer_backend='pyfdn'"
    ):
        RenderConfig.model_validate(
            _pyfdn_render_kwargs(
                synth=synth,
                renderer_backend="pedalboard",
                pyfdn_excitation=None,
            )
        )


def test_pyfdn_plugin_path_with_unregistered_name_rejects_hosted_backend() -> None:
    """The native package sentinel cannot route through a hosted backend."""
    synth = {
        "name": "unregistered_pyfdn",
        "param_spec_name": "pyfdn_n8_mono_householder",
        "plugin_path": "pyfdn",
        "plugin_state_path": "",
        "synth_version": "0.4.2",
        "source_sha256": pyfdn_param_spec_sha256(PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC),
    }

    with pytest.raises(
        ValidationError, match="all pyFDN identities require renderer_backend='pyfdn'"
    ):
        RenderConfig.model_validate(
            _pyfdn_render_kwargs(
                synth=synth,
                renderer_backend="pedalboard",
                pyfdn_excitation=None,
            )
        )


def test_pyfdn_name_with_mismatched_spec_rejects_native_backend() -> None:
    """A pyFDN synth and unrelated parameter spec are not a registered identity."""
    synth = {
        "name": "pyfdn_n8_mono_householder",
        "param_spec_name": "pyfdn_n8_mono_kronecker",
        "plugin_path": "pyfdn",
        "plugin_state_path": "",
        "synth_version": "0.4.2",
        "source_sha256": pyfdn_param_spec_sha256(PYFDN_N8_MONO_KRONECKER_PARAM_SPEC),
    }

    with pytest.raises(ValidationError, match="registered pyfdn synth identity"):
        RenderConfig.model_validate(_pyfdn_render_kwargs(synth=synth))


def test_pyfdn_name_with_mismatched_spec_rejects_hosted_backend() -> None:
    """A malformed pyFDN identity cannot evade native-backend validation."""
    synth = {
        "name": "pyfdn_n8_mono_householder",
        "param_spec_name": "surge_4",
        "plugin_path": "plugins/Surge XT.vst3",
        "plugin_state_path": "presets/surge-mini.vstpreset",
        "synth_version": "1.3.4",
    }

    with pytest.raises(
        ValidationError, match="all pyFDN identities require renderer_backend='pyfdn'"
    ):
        RenderConfig.model_validate(
            _pyfdn_render_kwargs(
                synth=synth,
                renderer_backend="pedalboard",
                pyfdn_excitation=None,
            )
        )


def test_pyfdn_identity_rejects_hosted_backend() -> None:
    """The pyFDN identity cannot route through an external plugin host."""
    with pytest.raises(
        ValidationError,
        match="all pyFDN identities require renderer_backend='pyfdn'",
    ):
        RenderConfig.model_validate(_pyfdn_render_kwargs(renderer_backend="pedalboard"))


def test_pyfdn_mono_identity_rejects_stereo_output() -> None:
    """Mono pyFDN identities keep requiring the single source channel."""
    with pytest.raises(ValidationError, match="channels=1"):
        RenderConfig.model_validate(_pyfdn_render_kwargs(channels=2))
