"""Tests for the render config the interactive tool writes captured patches with.

Importing ``vst_interactive`` pulls pedalboard, so these stay CPU-only and touch
no audio device: the helper is pure config construction.
"""

from __future__ import annotations

import pytest

from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.tools.vst_interactive import (
    MAKE_DATASET_VELOCITY,
    SAMPLE_RATE,
    make_dataset_render_cfg,
)


def _cfg(**overrides: object) -> RenderConfig:
    """Build a captured-patch render config with Surge defaults.

    :param \\*\\*overrides: Fields replacing the defaults.
    :returns: The constructed render config.
    """
    kwargs: dict[str, object] = {
        "param_spec_name": "surge_simple",
        "plugin_path": "plugins/Surge XT.vst3",
        "plugin_state_path": "presets/surge-simple.vstpreset",
        "synth_version": "1.3.4",
        "samples_per_shard": 3,
    }
    kwargs.update(overrides)
    return make_dataset_render_cfg(**kwargs)  # type: ignore[arg-type]


def test_render_cfg_carries_the_auditioned_synth_identity() -> None:
    """The dataset records the spec, plugin, and preset the patches were captured on."""
    cfg = _cfg()

    assert cfg.synth.param_spec_name == "surge_simple"
    assert cfg.synth.plugin_path == "plugins/Surge XT.vst3"
    assert cfg.synth.plugin_state_path == "presets/surge-simple.vstpreset"


def test_render_cfg_shard_size_is_the_captured_patch_count() -> None:
    """One shard holds exactly the patches the session recorded."""
    assert _cfg(samples_per_shard=7).samples_per_shard == 7


def test_render_cfg_uses_the_session_audio_settings() -> None:
    """Render settings match the audition session rather than the CLI defaults."""
    cfg = _cfg()

    assert cfg.sample_rate == SAMPLE_RATE
    assert cfg.velocity == MAKE_DATASET_VELOCITY


def test_render_cfg_rejects_a_preset_on_the_in_process_backend() -> None:
    """Identity validation still runs, so an incoherent pairing cannot be written."""
    with pytest.raises(ValueError, match="preset"):
        _cfg(plugin_path="torchsynth", plugin_state_path="presets/surge-simple.vstpreset")
