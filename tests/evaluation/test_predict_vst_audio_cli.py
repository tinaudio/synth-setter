"""CLI transport tests for prediction-audio rendering."""

from pathlib import Path

from pydantic_settings import CliApp

from synth_setter.evaluation.predict_vst_audio import _PredictAudioCliArgs
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.synth_spec import SynthName, SynthSpec


def _render_config() -> RenderConfig:
    """Return a complete DawDreamer render config for CLI transport.

    :returns: Validated configuration for serialization.
    """
    return RenderConfig(
        synth=SynthSpec(
            name=SynthName("surge_simple"),
            param_spec_name=ParamSpecName("surge_simple"),
            plugin_path="plugins/Surge XT.vst3",
            plugin_state_path="presets/surge-base.vstpreset",
        ),
        renderer_version="1.3.4",
        renderer_backend="dawdreamer",
        sample_rate=44100,
        channels=2,
        velocity=100,
        signal_duration_seconds=4.0,
        min_loudness=-55.0,
        samples_per_render_batch=2,
        samples_per_shard=4,
        plugin_reload_cadence="render",
        gui_toggle_cadence="never",
    )


def test_predict_audio_cli_round_trips_serialized_render_config() -> None:
    """CliApp transports every validated render field without a manual flag list."""
    config = _render_config()
    argv = [
        "predictions",
        "audio",
        *CliApp.serialize(config),
        "--rerender-target",
        "True",
    ]

    parsed = CliApp.run(_PredictAudioCliArgs, cli_args=argv)

    assert parsed.pred_dir == Path("predictions")
    assert parsed.output_dir == Path("audio")
    assert parsed.rerender_target is True
    assert parsed.renderer_backend == "dawdreamer"
    assert parsed.plugin_reload_cadence == "render"
    assert (
        RenderConfig.model_validate(parsed.model_dump(include=set(RenderConfig.model_fields)))
        == config
    )
