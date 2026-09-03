"""Contracts for the learnable pyFDN pitch-shift shimmer instrument."""

from typing import cast

import numpy as np
import pytest

from synth_setter.data.pyfdn_instrument import PyFDNRenderer, params_to_pitchshift_fdn_build
from synth_setter.data.pyfdn_param_spec import (
    PYFDN_PITCHSHIFT_ACTIVE_CHANNELS_NAME,
    PYFDN_PITCHSHIFT_N8_MONO_PARAM_SPEC,
    PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_MAX,
    PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_NAME,
    PYFDN_PITCHSHIFT_WINDOW_SIZE_MAX,
    PYFDN_PITCHSHIFT_WINDOW_SIZE_NAME,
    PYFDN_RT_GEQ_SECONDS_NAME,
)
from synth_setter.data.vst.param_spec import ParameterValues
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.renderer_factory import make_audio_renderer

_REFERENCE_RT = np.array(
    [4.4, 4.4, 4.3, 4.1, 3.8, 3.4, 3.0, 1.7, 1.5, 0.5], dtype=np.float64
)


def _reference_params() -> ParameterValues:
    params, _ = PYFDN_PITCHSHIFT_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(7))
    params[PYFDN_RT_GEQ_SECONDS_NAME] = _REFERENCE_RT.copy()
    params[PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_NAME] = -700.0
    params[PYFDN_PITCHSHIFT_WINDOW_SIZE_NAME] = 2048
    params[PYFDN_PITCHSHIFT_ACTIVE_CHANNELS_NAME] = np.array(
        [0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int64
    )
    return params


def test_pitchshift_spec_reference_controls_round_trip() -> None:
    """The exact upstream shimmer controls survive the model codec."""
    params = _reference_params()

    encoded = PYFDN_PITCHSHIFT_N8_MONO_PARAM_SPEC.encode(
        params, {"pitch": 0, "note_start_and_end": (0.0, 0.0)}
    )
    decoded, _ = PYFDN_PITCHSHIFT_N8_MONO_PARAM_SPEC.decode(encoded)

    assert encoded.shape == (109,)
    np.testing.assert_allclose(decoded[PYFDN_RT_GEQ_SECONDS_NAME], _REFERENCE_RT, atol=1e-6)
    assert decoded[PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_NAME] == pytest.approx(-700.0, abs=2e-5)
    assert decoded[PYFDN_PITCHSHIFT_WINDOW_SIZE_NAME] == 2048
    np.testing.assert_array_equal(
        decoded[PYFDN_PITCHSHIFT_ACTIVE_CHANNELS_NAME],
        np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int64),
    )


def test_pitchshift_build_uses_ten_band_geq() -> None:
    """Ten predicted RT values produce pyFDN's eleven-section GEQ bank."""
    build = params_to_pitchshift_fdn_build(_reference_params(), sample_rate=44_100.0)

    assert isinstance(build.post_delay, np.ndarray)
    assert build.post_delay.shape == (11, 6, 8)
    assert build.post_delay.dtype == np.float64


@pytest.mark.parametrize("rt_index", range(10))
def test_pitchshift_build_each_rt_coordinate_changes_geq(rt_index: int) -> None:
    """Every predicted RT coordinate participates in native GEQ design.

    :param rt_index: Ten-band RT coordinate changed from the reference profile.
    """
    baseline = _reference_params()
    changed = dict(baseline)
    changed_rt = _REFERENCE_RT.copy()
    changed_rt[rt_index] = 4.8
    changed[PYFDN_RT_GEQ_SECONDS_NAME] = changed_rt

    baseline_build = params_to_pitchshift_fdn_build(baseline, sample_rate=44_100.0)
    changed_build = params_to_pitchshift_fdn_build(changed, sample_rate=44_100.0)

    assert changed_build.post_delay is not None
    assert baseline_build.post_delay is not None
    assert not np.array_equal(changed_build.post_delay, baseline_build.post_delay)


def test_pitchshift_renderer_reference_patch_is_repeatable() -> None:
    """Fresh GEQ and pitch-shifter state make repeated renders identical."""
    renderer = PyFDNRenderer(param_spec_name=ParamSpecName("pyfdn_pitchshift_n8_mono"))
    params = _reference_params()

    first = renderer.render(params)
    second = renderer.render(params)

    assert first.shape == (1, 176_400)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(second, first)


def test_pitchshift_renderer_impulse_provenance_names_process_fdn() -> None:
    """Impulse provenance identifies the processing path used by shimmer."""
    renderer = PyFDNRenderer(param_spec_name=ParamSpecName("pyfdn_pitchshift_n8_mono"))

    assert renderer.source_provenance["implementation"] == "pyFDN.process_fdn"


def test_pitchshift_renderer_chirp_returns_finite_audio() -> None:
    """The canonical chirp traverses the native pitch-shift topology."""
    renderer = PyFDNRenderer(
        excitation="chirp",
        param_spec_name=ParamSpecName("pyfdn_pitchshift_n8_mono"),
    )

    audio = renderer.render(_reference_params())

    assert audio.shape == (1, 176_400)
    assert audio.dtype == np.float32
    assert np.isfinite(audio).all()


def test_pitchshift_renderer_rejects_transpose_above_spec_bound() -> None:
    """Native rendering rejects transpose values outside the learnable domain."""
    params = _reference_params()
    params[PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_NAME] = (
        PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_MAX + 1.0
    )
    renderer = PyFDNRenderer(param_spec_name=ParamSpecName("pyfdn_pitchshift_n8_mono"))

    with pytest.raises(ValueError, match="transpose_cents must be between"):
        renderer.render(params)


def test_pitchshift_renderer_rejects_window_above_spec_bound() -> None:
    """Native rendering rejects window sizes outside the learnable domain."""
    params = _reference_params()
    params[PYFDN_PITCHSHIFT_WINDOW_SIZE_NAME] = PYFDN_PITCHSHIFT_WINDOW_SIZE_MAX + 1
    renderer = PyFDNRenderer(param_spec_name=ParamSpecName("pyfdn_pitchshift_n8_mono"))

    with pytest.raises(ValueError, match="window_size must be between"):
        renderer.render(params)


def test_pitchshift_renderer_transpose_changes_real_audio() -> None:
    """The predicted transpose control reaches the native pitch shifter."""
    baseline = _reference_params()
    changed = dict(baseline)
    changed[PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_NAME] = 700.0
    renderer = PyFDNRenderer(param_spec_name=ParamSpecName("pyfdn_pitchshift_n8_mono"))

    baseline_audio = renderer.render(baseline)
    changed_audio = renderer.render(changed)

    assert not np.array_equal(changed_audio, baseline_audio)


def test_pitchshift_renderer_window_changes_real_audio() -> None:
    """The predicted window size reaches the native pitch shifter."""
    baseline = _reference_params()
    changed = dict(baseline)
    changed[PYFDN_PITCHSHIFT_WINDOW_SIZE_NAME] = 1024
    renderer = PyFDNRenderer(param_spec_name=ParamSpecName("pyfdn_pitchshift_n8_mono"))

    baseline_audio = renderer.render(baseline)
    changed_audio = renderer.render(changed)

    assert not np.array_equal(changed_audio, baseline_audio)


def test_pitchshift_renderer_active_mask_changes_real_audio() -> None:
    """The predicted delay-line mask reaches the native pitch shifter."""
    baseline = _reference_params()
    changed = dict(baseline)
    changed[PYFDN_PITCHSHIFT_ACTIVE_CHANNELS_NAME] = np.ones(8, dtype=np.int64)
    renderer = PyFDNRenderer(param_spec_name=ParamSpecName("pyfdn_pitchshift_n8_mono"))

    baseline_audio = renderer.render(baseline)
    changed_audio = renderer.render(changed)

    assert not np.array_equal(changed_audio, baseline_audio)


def test_pitchshift_render_config_constructs_matching_renderer() -> None:
    """The registered identity reaches the pitch-shift renderer topology."""
    render = RenderConfig.model_validate(
        {
            "synth": {
                "name": "pyfdn_pitchshift_n8_mono",
                "param_spec_name": "pyfdn_pitchshift_n8_mono",
                "plugin_path": "pyfdn",
                "plugin_state_path": "",
                "synth_version": "0.4.2",
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
    )

    renderer = make_audio_renderer(render)

    assert isinstance(renderer, PyFDNRenderer)
    assert renderer._param_spec_name == ParamSpecName("pyfdn_pitchshift_n8_mono")


def test_pitchshift_spec_samples_native_control_types() -> None:
    """Sampling emits native values accepted directly by pyFDN."""
    params, _ = PYFDN_PITCHSHIFT_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(11))

    assert cast(np.ndarray, params[PYFDN_RT_GEQ_SECONDS_NAME]).shape == (10,)
    assert isinstance(params[PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_NAME], float)
    assert isinstance(params[PYFDN_PITCHSHIFT_WINDOW_SIZE_NAME], int)
    mask = cast(np.ndarray, params[PYFDN_PITCHSHIFT_ACTIVE_CHANNELS_NAME])
    assert (mask.shape, mask.dtype) == ((8,), np.dtype(np.int64))
