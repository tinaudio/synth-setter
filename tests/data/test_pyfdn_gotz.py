"""Contracts for the Götz et al.

(arXiv:2510.23158) pyFDN reverb instruments.
"""

from typing import cast

import numpy as np
import pytest

from synth_setter.data.pyfdn_instrument import PyFDNRenderer, params_to_gotz_fdn_build
from synth_setter.data.pyfdn_param_spec import (
    PYFDN_FEEDBACK_GIVENS_ANGLES_NAME,
    PYFDN_FEEDBACK_SKEW_NAME,
    PYFDN_GEQ_BAND_GAIN_DB_NAME,
    PYFDN_GEQ_GAIN_DB_NAME,
    PYFDN_GOTZ_DELAYS,
    PYFDN_GOTZ_N8_MONO_FIXED_DELAYS_GIVENS_PARAM_SPEC,
    PYFDN_GOTZ_N8_MONO_FIXED_DELAYS_PARAM_SPEC,
    PYFDN_GOTZ_N8_MONO_LEARNED_DELAYS_GIVENS_PARAM_SPEC,
    PYFDN_GOTZ_N8_MONO_LEARNED_DELAYS_PARAM_SPEC,
    PYFDN_TONE_GEQ_GAIN_DB_NAME,
    PyFDNGotzParamSpec,
    givens_to_orthogonal,
    skew_to_orthogonal,
)
from synth_setter.data.vst.param_spec import (
    AngleArrayParameter,
    ContinuousArrayParameter,
    DiscreteArrayParameter,
    ParameterValues,
)
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.renderer_factory import make_audio_renderer
from synth_setter.synth_spec import SYNTHS, SynthName

_NOTES: ParameterValues = {"pitch": 0, "note_start_and_end": (0.0, 0.0)}
_FIXED = PYFDN_GOTZ_N8_MONO_FIXED_DELAYS_PARAM_SPEC
_LEARNED = PYFDN_GOTZ_N8_MONO_LEARNED_DELAYS_PARAM_SPEC
_FIXED_GIVENS = PYFDN_GOTZ_N8_MONO_FIXED_DELAYS_GIVENS_PARAM_SPEC
_LEARNED_GIVENS = PYFDN_GOTZ_N8_MONO_LEARNED_DELAYS_GIVENS_PARAM_SPEC
_PAPER_DELAYS = np.array([809, 877, 937, 1049, 1151, 1249, 1373, 1499], dtype=np.int64)


def _reference_params() -> ParameterValues:
    params, _ = _FIXED.sample(np.random.default_rng(7))
    params["input_matrix"] = np.full((8, 1), 0.5, dtype=np.float64)
    params["output_matrix"] = np.full((1, 8), 0.5, dtype=np.float64)
    params["direct_matrix"] = np.array([[0.25]], dtype=np.float64)
    params[PYFDN_GEQ_GAIN_DB_NAME] = np.full((8,), -1.0, dtype=np.float64)
    params[PYFDN_GEQ_BAND_GAIN_DB_NAME] = np.zeros((10, 8), dtype=np.float64)
    params[PYFDN_TONE_GEQ_GAIN_DB_NAME] = np.zeros((11,), dtype=np.float64)
    return params


def test_gotz_fixed_delays_spec_layout_excludes_delays_from_model_coordinates() -> None:
    """The fixed-delay variant learns 144 coordinates and no delay line."""
    layout = [
        (parameter.name, span.start, span.stop) for parameter, span in _FIXED.encoded_slices()
    ]

    assert _FIXED.encoded_width == 144
    assert layout == [
        ("feedback_skew", 0, 28),
        ("input_matrix", 28, 36),
        ("output_matrix", 36, 44),
        ("direct_matrix", 44, 45),
        ("post_delay.geq.gain_db", 45, 53),
        ("post_delay.geq.band_gain_db", 53, 133),
        ("input.tone_geq.command_gain_db", 133, 144),
    ]
    assert _FIXED.note_params == []


def test_gotz_learned_delays_spec_layout_prepends_delays() -> None:
    """The learned-delay variant adds eight delay coordinates ahead of the shared ones."""
    layout = [
        (parameter.name, span.start, span.stop)
        for parameter, span in _LEARNED.encoded_slices()
    ]

    assert _LEARNED.encoded_width == 152
    assert layout[:2] == [("delays", 0, 8), ("feedback_skew", 8, 36)]
    assert layout[-1] == ("input.tone_geq.command_gain_db", 141, 152)


def test_givens_single_plane_quarter_turn_uses_documented_sign_convention() -> None:
    """The first angle rotates the (0, 1) plane with block [cos,-sin;sin,cos]."""
    angles = np.zeros((28,), dtype=np.float64)
    angles[0] = np.pi / 2

    feedback = givens_to_orthogonal(angles)

    expected = np.eye(8)
    expected[:2, :2] = [[0.0, -1.0], [1.0, 0.0]]
    np.testing.assert_allclose(feedback, expected, atol=1e-15)


def test_givens_noncommuting_planes_multiply_in_lexicographic_order() -> None:
    """Angles (0,1) then (0,2) produce G01 @ G02, not the reversed product."""
    angles = np.zeros((28,), dtype=np.float64)
    angles[:2] = [np.pi / 3, np.pi / 4]
    g01 = np.eye(8)
    g01[:2, :2] = [
        [np.cos(angles[0]), -np.sin(angles[0])],
        [np.sin(angles[0]), np.cos(angles[0])],
    ]
    g02 = np.eye(8)
    indices = np.ix_([0, 2], [0, 2])
    g02[indices] = [
        [np.cos(angles[1]), -np.sin(angles[1])],
        [np.sin(angles[1]), np.cos(angles[1])],
    ]

    feedback = givens_to_orthogonal(angles)

    np.testing.assert_allclose(feedback, g01 @ g02, atol=1e-15)
    assert not np.allclose(feedback, g02 @ g01)


def test_givens_angles_are_periodic_by_two_pi() -> None:
    """Adding full turns to native angles leaves the feedback matrix unchanged."""
    angles = np.linspace(-np.pi, np.pi, 28, endpoint=False)
    turns = 2 * np.pi * np.arange(28)

    np.testing.assert_allclose(
        givens_to_orthogonal(angles + turns), givens_to_orthogonal(angles), atol=2e-14
    )


def test_gotz_givens_sampled_matrices_are_special_orthogonal() -> None:
    """Every sampled Givens product lies in SO(8)."""
    for seed in range(5):
        params, _ = _FIXED_GIVENS.sample(np.random.default_rng(seed))
        feedback = cast(np.ndarray, params["feedback_matrix"])
        np.testing.assert_allclose(feedback.T @ feedback, np.eye(8), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(feedback), 1.0, atol=1e-12)


def test_gotz_givens_angle_codec_round_trips_and_projects_model_pairs() -> None:
    """The real Givens codec preserves angles and normalizes model-space pairs."""
    angles = cast(AngleArrayParameter, _FIXED_GIVENS.synth_params[0])
    native = np.linspace(-np.pi, np.pi, 28, endpoint=False)
    encoded = angles.encode(native)
    decoded = angles.decode(encoded)
    model = (encoded * 2.0 - 1.0) * 3.0

    np.testing.assert_allclose(np.exp(1j * decoded), np.exp(1j * native), atol=1e-7)
    np.testing.assert_allclose(angles.model_to_encoded(model), encoded, atol=1e-7)


def test_gotz_learned_delays_bounds_cover_the_paper_delays() -> None:
    """Every paper delay length is a reachable learned value."""
    delays = cast(DiscreteArrayParameter, _LEARNED.synth_params[0])

    assert (delays.min, delays.max) == (809, 1499)


def test_gotz_fixed_delays_spec_samples_paper_delay_lengths() -> None:
    """Every fixed-delay patch uses the paper's coprime delay lengths verbatim."""
    params, _ = _FIXED.sample(np.random.default_rng(3))
    delays = cast(np.ndarray, params["delays"])

    np.testing.assert_array_equal(delays, _PAPER_DELAYS)
    assert delays.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(PYFDN_GOTZ_DELAYS, _PAPER_DELAYS)


def test_gotz_param_spec_composes_callable_feedback_with_fixed_delays() -> None:
    """The shared derived-feedback codec and Götz fixed-value restoration compose."""
    spec = PyFDNGotzParamSpec(
        synth_params=[],
        feedback_matrix=lambda _: np.eye(8, dtype=np.float64),
        fixed_delays=_PAPER_DELAYS,
    )

    sampled, _ = spec.sample(np.random.default_rng(3))
    decoded, _ = spec.decode(np.empty((0,), dtype=np.float32))

    np.testing.assert_array_equal(sampled["feedback_matrix"], np.eye(8))
    np.testing.assert_array_equal(decoded["feedback_matrix"], np.eye(8))
    np.testing.assert_array_equal(sampled["delays"], _PAPER_DELAYS)
    np.testing.assert_array_equal(decoded["delays"], _PAPER_DELAYS)


def test_gotz_spec_samples_orthogonal_feedback_that_varies_by_seed() -> None:
    """Sampled feedback is orthogonal by construction and not a single fixed matrix."""
    first, _ = _FIXED.sample(np.random.default_rng(1))
    second, _ = _FIXED.sample(np.random.default_rng(2))
    first_feedback = cast(np.ndarray, first["feedback_matrix"])
    second_feedback = cast(np.ndarray, second["feedback_matrix"])

    np.testing.assert_allclose(first_feedback.T @ first_feedback, np.eye(8), atol=1e-12)
    assert first_feedback.shape == (8, 8)
    assert first_feedback.dtype == np.dtype(np.float64)
    assert not np.allclose(first_feedback, second_feedback)


def test_gotz_spec_zero_skew_decodes_to_identity_feedback() -> None:
    """The matrix exponential of a zero skew-symmetric matrix is the identity."""
    params = _reference_params()
    params[PYFDN_FEEDBACK_SKEW_NAME] = np.zeros((28,), dtype=np.float64)

    decoded, _ = _FIXED.decode(_FIXED.encode(params, _NOTES))

    np.testing.assert_allclose(decoded["feedback_matrix"], np.eye(8), atol=1e-6)


def test_gotz_spec_encoding_round_trips_learned_fields_and_derived_feedback() -> None:
    """Skew coordinates survive the codec and rebuild the same feedback matrix."""
    params, notes = _LEARNED.sample(np.random.default_rng(11))

    encoded = _LEARNED.encode(params, notes)
    decoded, decoded_notes = _LEARNED.decode(encoded)

    assert (encoded.shape, encoded.dtype) == ((152,), np.dtype(np.float32))
    assert decoded_notes == _NOTES
    np.testing.assert_array_equal(decoded["delays"], params["delays"])
    np.testing.assert_allclose(decoded["feedback_matrix"], params["feedback_matrix"], atol=1e-5)
    for name in (
        PYFDN_FEEDBACK_SKEW_NAME,
        "input_matrix",
        "output_matrix",
        "direct_matrix",
        PYFDN_GEQ_GAIN_DB_NAME,
        PYFDN_GEQ_BAND_GAIN_DB_NAME,
        PYFDN_TONE_GEQ_GAIN_DB_NAME,
    ):
        np.testing.assert_allclose(decoded[name], params[name], atol=2e-6)


def test_gotz_spec_attenuation_gains_never_exceed_unity() -> None:
    """Attenuation command gains stay at or below 0 dB so the loop cannot grow."""
    gain = cast(ContinuousArrayParameter, _FIXED.synth_params[4])
    band_gain = cast(ContinuousArrayParameter, _FIXED.synth_params[5])

    assert gain.max < 0.0
    assert band_gain.max == 0.0


def test_gotz_build_carries_eleven_section_per_line_attenuation() -> None:
    """Each delay line gets its own eleven-section GEQ cascade."""
    build = params_to_gotz_fdn_build(_reference_params(), sample_rate=44_100.0)

    post_delay = cast(np.ndarray, build.post_delay)
    assert post_delay.shape == (11, 6, 8)
    assert post_delay.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(build.delays, _PAPER_DELAYS)
    assert build.post_matrix is None and build.post_output is None


def test_gotz_build_zero_band_gains_make_shaped_sections_transparent() -> None:
    """0 dB band command gains reduce each shaped section to a pass-through."""
    build = params_to_gotz_fdn_build(_reference_params(), sample_rate=44_100.0)

    shaped = cast(np.ndarray, build.post_delay)[1:]
    np.testing.assert_allclose(shaped[:, :3, :], shaped[:, 3:, :], atol=1e-12)


def test_gotz_build_flat_gain_scales_first_section_only() -> None:
    """The flat command gain lands in section zero as a linear amplitude."""
    params = _reference_params()
    params[PYFDN_GEQ_GAIN_DB_NAME] = np.full((8,), -6.0, dtype=np.float64)

    build = params_to_gotz_fdn_build(params, sample_rate=44_100.0)

    flat = cast(np.ndarray, build.post_delay)[0]
    np.testing.assert_allclose(flat[0], np.full(8, 0.5011872336272722))
    np.testing.assert_array_equal(flat[3], np.ones(8))
    np.testing.assert_array_equal(flat[[1, 2, 4, 5]], np.zeros((4, 8)))


def test_gotz_build_each_line_gets_its_own_band_gain() -> None:
    """Attenuation is per line: changing one line's band leaves the others untouched."""
    params = _reference_params()
    band = cast(np.ndarray, params[PYFDN_GEQ_BAND_GAIN_DB_NAME]).copy()
    band[3, 5] = -6.0
    params[PYFDN_GEQ_BAND_GAIN_DB_NAME] = band

    reference = cast(np.ndarray, params_to_gotz_fdn_build(_reference_params(), sample_rate=44_100.0).post_delay)
    changed = cast(np.ndarray, params_to_gotz_fdn_build(params, sample_rate=44_100.0).post_delay)

    differs = np.any(reference != changed, axis=(0, 1))
    np.testing.assert_array_equal(differs, [False, False, False, False, False, True, False, False])


def test_gotz_build_rejects_missing_tone_key() -> None:
    """The native mapping must contain exactly the Götz topology keys."""
    params = _reference_params()
    del params[PYFDN_TONE_GEQ_GAIN_DB_NAME]

    with pytest.raises(ValueError, match="gotz params must contain exactly"):
        params_to_gotz_fdn_build(params, sample_rate=44_100.0)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        (PYFDN_GEQ_GAIN_DB_NAME, np.full((8,), -20.5), "gain_db must be at least -20 dB"),
        (PYFDN_GEQ_BAND_GAIN_DB_NAME, np.full((10, 8), -6.5), "band_gain_db must be at least -6 dB"),
        (PYFDN_TONE_GEQ_GAIN_DB_NAME, np.full((11,), -12.5), "command_gain_db must be at least -12 dB"),
        (PYFDN_TONE_GEQ_GAIN_DB_NAME, np.full((11,), 12.5), "command_gain_db must be at most 12 dB"),
    ],
    ids=["flat_below", "band_below", "tone_below", "tone_above"],
)
def test_gotz_build_rejects_gain_outside_spec_bounds(
    name: str, value: np.ndarray, message: str
) -> None:
    """Every dB control is rejected just past its ParamSpec bound.

    :param name: Native control to violate.
    :param value: Out-of-bounds float64 array for that control.
    :param message: Expected diagnostic fragment.
    """
    params = _reference_params()
    params[name] = value

    with pytest.raises(ValueError, match=message):
        params_to_gotz_fdn_build(params, sample_rate=44_100.0)


def test_gotz_build_rejects_feedback_matrix_that_disagrees_with_skew() -> None:
    """A tampered feedback matrix cannot bypass the orthogonality encoded by the skew."""
    params = _reference_params()
    params["feedback_matrix"] = 2.0 * np.eye(8, dtype=np.float64)

    with pytest.raises(ValueError, match="orthogonal matrix encoded by feedback_skew"):
        params_to_gotz_fdn_build(params, sample_rate=44_100.0)


def test_gotz_build_uses_feedback_regenerated_from_skew_not_the_supplied_copy() -> None:
    """Within tolerance, the rendered matrix is still the exact skew image, not the input."""
    params = _reference_params()
    skew = cast(np.ndarray, params[PYFDN_FEEDBACK_SKEW_NAME])
    params["feedback_matrix"] = skew_to_orthogonal(skew) + 5e-7

    build = params_to_gotz_fdn_build(params, sample_rate=44_100.0)

    np.testing.assert_array_equal(build.A, skew_to_orthogonal(skew))


def test_gotz_build_fixed_delays_rejects_other_delay_lengths() -> None:
    """The fixed-delay identity cannot render with delays that differ from the paper's."""
    params = _reference_params()
    params["delays"] = np.ones((8,), dtype=np.int64)

    with pytest.raises(ValueError, match="delays must equal the fixed lengths"):
        params_to_gotz_fdn_build(params, sample_rate=44_100.0, fixed_delays=PYFDN_GOTZ_DELAYS)


def test_gotz_renderer_fixed_identity_rejects_other_delay_lengths() -> None:
    """The fixed-delay renderer enforces the paper's delays on every patch it renders."""
    renderer = PyFDNRenderer(param_spec_name=ParamSpecName("pyfdn_gotz_n8_mono_fixed_delays"))
    params = _reference_params()
    params["delays"] = np.array([900, 1000, 1100, 1200, 1300, 1400, 1450, 1499], dtype=np.int64)

    with pytest.raises(ValueError, match="delays must equal the fixed lengths"):
        renderer.render(params)


def test_gotz_build_accepts_feedback_matrix_rebuilt_from_float32_codec() -> None:
    """A row decoded from float32 coordinates still passes the skew consistency check."""
    params, notes = _LEARNED.sample(np.random.default_rng(29))
    decoded, _ = _LEARNED.decode(_LEARNED.encode(params, notes))

    build = params_to_gotz_fdn_build(decoded, sample_rate=44_100.0)

    np.testing.assert_allclose(build.A.T @ build.A, np.eye(8), atol=1e-12)


def test_gotz_build_rejects_positive_attenuation_gain() -> None:
    """An attenuation command gain above 0 dB breaks the stability guarantee."""
    params = _reference_params()
    params[PYFDN_GEQ_BAND_GAIN_DB_NAME] = np.full((10, 8), 0.5, dtype=np.float64)

    with pytest.raises(ValueError, match="band_gain_db must be at most 0 dB"):
        params_to_gotz_fdn_build(params, sample_rate=44_100.0)


@pytest.mark.parametrize(
    ("name", "width"),
    [
        ("pyfdn_gotz_n8_mono_fixed_delays", 144),
        ("pyfdn_gotz_n8_mono_learned_delays", 152),
        ("pyfdn_gotz_n8_mono_fixed_delays_givens", 172),
        ("pyfdn_gotz_n8_mono_learned_delays_givens", 180),
    ],
)
def test_gotz_identities_are_registered(name: str, width: int) -> None:
    """Both variants resolve through the ParamSpec registry and the synth table.

    :param name: Registered synth and ParamSpec name.
    :param width: Encoded width the registry must serve for it.
    """
    synth = SYNTHS[SynthName(name)]

    assert (synth.plugin_path, synth.param_spec_name) == ("pyfdn", name)
    assert param_specs[name].encoded_width == width


@pytest.mark.parametrize(
    ("renderer_name", "patch_spec"),
    [
        ("pyfdn_gotz_n8_mono_fixed_delays_givens", _FIXED),
        ("pyfdn_gotz_n8_mono_fixed_delays", _FIXED_GIVENS),
    ],
)
def test_gotz_renderer_rejects_patch_from_other_feedback_codec(
    renderer_name: str, patch_spec: PyFDNGotzParamSpec
) -> None:
    """Distinct synth identities cannot silently reuse incompatible codec rows.

    :param renderer_name: Renderer identity selecting one feedback coordinate system.
    :param patch_spec: Spec producing a patch from the other coordinate system.
    """
    params, _ = patch_spec.sample(np.random.default_rng(13))

    with pytest.raises(ValueError, match="gotz params must contain exactly"):
        PyFDNRenderer(param_spec_name=ParamSpecName(renderer_name)).render(params)


def test_gotz_givens_build_rejects_inconsistent_feedback_matrix() -> None:
    """A native Givens patch cannot carry a matrix from different angles."""
    params, _ = _FIXED_GIVENS.sample(np.random.default_rng(17))
    params["feedback_matrix"] = np.eye(8, dtype=np.float64)

    with pytest.raises(ValueError, match="encoded by feedback_givens_angles"):
        params_to_gotz_fdn_build(
            params,
            sample_rate=44_100.0,
            fixed_delays=PYFDN_GOTZ_DELAYS,
            feedback_parameter=PYFDN_FEEDBACK_GIVENS_ANGLES_NAME,
        )


@pytest.mark.parametrize(
    ("name", "spec"),
    [
        ("pyfdn_gotz_n8_mono_fixed_delays_givens", _FIXED_GIVENS),
        ("pyfdn_gotz_n8_mono_learned_delays_givens", _LEARNED_GIVENS),
    ],
)
def test_gotz_givens_delay_variants_render_real_audio(name: str, spec: PyFDNGotzParamSpec) -> None:
    """Both Givens delay identities dispatch through the distinct Götz renderer.

    :param name: Registered fixed- or learned-delay Givens identity.
    :param spec: Matching Götz ParamSpec used to sample a native patch.
    """
    params, _ = spec.sample(np.random.default_rng(23))

    audio = PyFDNRenderer(param_spec_name=ParamSpecName(name)).render(params)

    assert audio.shape == (1, 176_400)
    assert audio.dtype == np.dtype(np.float32)
    assert np.isfinite(audio).all()


def test_gotz_renderer_sampled_fixed_patch_returns_finite_decaying_response() -> None:
    """A sampled patch renders a finite impulse response whose tail is quieter than its head."""
    renderer = PyFDNRenderer(param_spec_name=ParamSpecName("pyfdn_gotz_n8_mono_fixed_delays"))
    params, _ = _FIXED.sample(np.random.default_rng(5))

    audio = renderer.render(params)

    assert audio.shape == (1, 176_400)
    assert audio.dtype == np.dtype(np.float32)
    assert np.isfinite(audio).all()
    assert np.abs(audio[0, -44_100:]).max() < np.abs(audio[0, :44_100]).max()


def test_gotz_renderer_direct_path_arrives_two_samples_late() -> None:
    """With output gains muted, the impulse response is the direct gain delayed by two samples."""
    renderer = PyFDNRenderer(param_spec_name=ParamSpecName("pyfdn_gotz_n8_mono_fixed_delays"))
    params = _reference_params()
    params["output_matrix"] = np.zeros((1, 8), dtype=np.float64)

    audio = renderer.render(params)

    np.testing.assert_allclose(audio[0, :3], [0.0, 0.0, 0.25], atol=1e-7)
    np.testing.assert_allclose(audio[0, 3:], 0.0, atol=1e-7)


def test_gotz_renderer_tone_geq_at_minus_12_db_attenuates_real_audio() -> None:
    """Negative tone-correction commands lower the rendered energy, so polarity is right."""
    renderer = PyFDNRenderer(param_spec_name=ParamSpecName("pyfdn_gotz_n8_mono_fixed_delays"))
    params = _reference_params()
    reference = renderer.render(params)
    params[PYFDN_TONE_GEQ_GAIN_DB_NAME] = np.full((11,), -12.0, dtype=np.float64)

    changed = renderer.render(params)

    assert np.sqrt(np.mean(changed**2)) < 0.5 * np.sqrt(np.mean(reference**2))


def test_gotz_renderer_flat_attenuation_controls_the_wet_tail() -> None:
    """Per-line attenuation reaches the render: -20 dB per pass leaves a far quieter tail."""
    renderer = PyFDNRenderer(param_spec_name=ParamSpecName("pyfdn_gotz_n8_mono_fixed_delays"))
    params = _reference_params()
    params["direct_matrix"] = np.zeros((1, 1), dtype=np.float64)
    params[PYFDN_GEQ_GAIN_DB_NAME] = np.full((8,), -0.3, dtype=np.float64)
    slow_decay = renderer.render(params)
    params[PYFDN_GEQ_GAIN_DB_NAME] = np.full((8,), -20.0, dtype=np.float64)

    fast_decay = renderer.render(params)

    late = slice(44_100, 88_200)
    assert np.sum(fast_decay[0, late] ** 2) < 1e-6 * np.sum(slow_decay[0, late] ** 2)


def test_gotz_renderer_learned_delays_change_real_audio() -> None:
    """The learned-delay variant renders the delays it is given."""
    renderer = PyFDNRenderer(param_spec_name=ParamSpecName("pyfdn_gotz_n8_mono_learned_delays"))
    params = _reference_params()
    reference = renderer.render(params)
    params["delays"] = np.array([900, 1000, 1100, 1200, 1300, 1400, 1450, 1499], dtype=np.int64)

    changed = renderer.render(params)

    assert not np.allclose(reference, changed)


def test_gotz_renderer_repeated_renders_are_identical() -> None:
    """Filter and delay state never leaks between renders."""
    renderer = PyFDNRenderer(param_spec_name=ParamSpecName("pyfdn_gotz_n8_mono_fixed_delays"))
    params = _reference_params()

    np.testing.assert_array_equal(renderer.render(params), renderer.render(params))


def test_gotz_render_config_constructs_matching_renderer() -> None:
    """The pipeline render config reaches the Götz renderer with process_fdn provenance."""
    config = RenderConfig.model_validate(
        {
            "renderer_backend": "pyfdn",
            "pyfdn_excitation": "impulse",
            "sample_rate": 44_100,
            "channels": 1,
            "velocity": 0,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "audio_dtype": "float32",
            "mel_spec_dtype": "float32",
            "samples_per_shard": 100,
            "samples_per_render_batch": 8,
            "max_retries": 0,
            "parallel": False,
            "retain_local_shards": True,
            "param_sample_cadence": "sample",
            "plugin_reload_cadence": "render",
            "gui_toggle_cadence": "never",
            "synth": {
                "name": "pyfdn_gotz_n8_mono_learned_delays",
                "param_spec_name": "pyfdn_gotz_n8_mono_learned_delays",
                "plugin_path": "pyfdn",
                "plugin_state_path": "",
                "synth_version": "0.4.2",
            },
        }
    )

    renderer = make_audio_renderer(config)

    assert isinstance(renderer, PyFDNRenderer)
    assert renderer.source_provenance["implementation"] == "pyFDN.process_fdn"
