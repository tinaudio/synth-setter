"""KR-106 native render-contract validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.renderer_backend import NO_FLUSH_BLOCKS
from synth_setter.synth_spec import SYNTHS, SynthName


def _native_render_kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "synth": SYNTHS[SynthName("ultramaster_kr106_native")],
        "renderer_backend": "kr106_native",
        "backend_version": "0.1.0",
        "block_size": 512,
        "render_contract_version": 2,
        "sample_rate": 44_100,
        "channels": 2,
        "velocity": 100,
        "signal_duration_seconds": 4.0,
        "min_loudness": -55.0,
        "audio_dtype": "float16",
        "mel_spec_dtype": "float32",
        "samples_per_render_batch": 32,
        "samples_per_shard": 1,
        "plugin_reload_cadence": "render",
        "gui_toggle_cadence": "never",
        "post_load_flush_blocks": 0,
        "post_param_flush_blocks": 0,
        "post_render_flush_blocks": 0,
    }
    values.update(overrides)
    return values


def test_kr106_native_render_config_accepts_pinned_in_process_contract() -> None:
    """The registered native identity needs no VST bundle and has zero flushes."""
    config = RenderConfig.model_validate(_native_render_kwargs())

    assert config.plugin_path == "kr106_native"
    assert config.backend_version == "0.1.0"
    assert config.block_size == 512
    assert config.flush_blocks == NO_FLUSH_BLOCKS


@pytest.mark.parametrize("channels", [1, 2])
def test_kr106_native_render_config_accepts_mono_and_stereo(channels: int) -> None:
    """The native renderer supports one and two output channels.

    :param channels: Native output geometry under test.
    """
    config = RenderConfig.model_validate(_native_render_kwargs(channels=channels))

    assert config.channels == channels


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("backend_version", "0.1.1", "backend_version"),
        ("block_size", None, "block_size"),
        ("render_contract_version", 1, "render_contract_version"),
        ("plugin_reload_cadence", "once", "plugin_reload_cadence"),
        ("gui_toggle_cadence", "once", "gui_toggle_cadence"),
        ("channels", 3, "one or two channels"),
        ("post_load_flush_blocks", 1, "zero flush"),
    ],
)
def test_kr106_native_render_config_rejects_noncanonical_contract(
    field: str, value: object, match: str
) -> None:
    """The renderer rejects lifecycle settings it cannot faithfully execute.

    :param field: Native render field changed from its required value.
    :param value: Invalid replacement value.
    :param match: Expected validation diagnostic.
    """
    with pytest.raises(ValidationError, match=match):
        RenderConfig.model_validate(_native_render_kwargs(**{field: value}))


def test_kr106_native_render_contract_digest_changes_with_identity() -> None:
    """Distinct KR-106 parameter/preset identities cannot share a dataset digest."""
    native = RenderConfig.model_validate(_native_render_kwargs())
    onehot = RenderConfig.model_validate(
        _native_render_kwargs(synth=SYNTHS[SynthName("ultramaster_kr106_onehot_native")])
    )

    assert (
        native.shard_metadata().render_contract_digest
        != onehot.shard_metadata().render_contract_digest
    )
