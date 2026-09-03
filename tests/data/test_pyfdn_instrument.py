"""Native build and fixed-source rendering contracts for the pyFDN instrument."""

import inspect
from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from pyFDN import FDNBuild
from scipy.signal import sosfreqz

import synth_setter.data.pyfdn_instrument as pyfdn_instrument
from synth_setter.data.pyfdn_instrument import PyFDNRenderer, params_to_fdn_build
from synth_setter.data.pyfdn_param_spec import PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC
from synth_setter.data.pyfdn_source import canonical_pyfdn_source_provenance
from synth_setter.data.vst.param_spec import ParameterValues


@pytest.fixture
def fdn_params() -> ParameterValues:
    """Return one valid native order-8 mono patch.

    :returns: Exact-shape and exact-dtype arrays accepted by the build codec.
    """
    return {
        "delays": np.arange(400, 408, dtype=np.int64),
        "feedback_matrix": np.eye(8, dtype=np.float64),
        "input_matrix": np.full((8, 1), 0.25, dtype=np.float64),
        "output_matrix": np.full((1, 8), 0.125, dtype=np.float64),
        "direct_matrix": np.array([[0.5]], dtype=np.float64),
        "post_delay.rt_dc_seconds": 1.0,
        "post_delay.rt_nyquist_seconds": 0.5,
    }


def test_params_to_fdn_build_reconstructs_exact_native_build(
    fdn_params: ParameterValues,
) -> None:
    """The codec preserves the base FDN and derives only native delay filters.

    :param fdn_params: Valid native patch.
    """
    build = params_to_fdn_build(fdn_params, sample_rate=44_100.0)

    assert isinstance(build, FDNBuild)
    assert build.A is fdn_params["feedback_matrix"]
    assert build.B is fdn_params["input_matrix"]
    assert build.C is fdn_params["output_matrix"]
    assert build.D is fdn_params["direct_matrix"]
    assert build.delays is fdn_params["delays"]
    assert build.fs == 44_100.0
    post_delay = build.post_delay
    assert post_delay is not None
    assert post_delay.shape == (1, 6, 8)
    assert post_delay.dtype == np.float64
    assert np.isfinite(post_delay).all()
    assert build.post_matrix is None
    assert build.post_output is None


def test_params_to_fdn_build_same_controls_reproduce_exact_decay_sos(
    fdn_params: ParameterValues,
) -> None:
    """The fixed crossover and RT tuple deterministically define the native SOS.

    :param fdn_params: Valid native patch with unequal endpoint RTs.
    """
    first = params_to_fdn_build(fdn_params, sample_rate=44_100.0)
    second = params_to_fdn_build(fdn_params, sample_rate=44_100.0)
    expected_end_lines = np.array(
        [
            [
                [0.9122841131033455, 0.9108196387460381],
                [0.07767066820536879, 0.07779384062478191],
                [0.0, 0.0],
                [1.0, 1.0],
                [0.05396512816322539, 0.05369180514782246],
                [0.0, 0.0],
            ]
        ],
        dtype=np.float64,
    )

    first_post_delay = first.post_delay
    second_post_delay = second.post_delay
    assert first_post_delay is not None
    assert second_post_delay is not None
    np.testing.assert_array_equal(second_post_delay, first_post_delay)
    np.testing.assert_allclose(
        first_post_delay[:, :, [0, 7]], expected_end_lines, rtol=0.0, atol=1e-12
    )


@pytest.mark.parametrize("delay_line", range(8))
def test_params_to_fdn_build_unequal_rt_extremes_attenuate_every_delay_line(
    fdn_params: ParameterValues,
    delay_line: int,
) -> None:
    """The derived shelf has positive sub-unity gain across every delay line.

    :param fdn_params: Valid native patch to move to opposite RT bounds.
    :param delay_line: Delay-filter index under test.
    """
    params = dict(fdn_params)
    params["post_delay.rt_dc_seconds"] = 0.1
    params["post_delay.rt_nyquist_seconds"] = 4.0

    build = params_to_fdn_build(params, sample_rate=44_100.0)

    assert build.post_delay is not None
    _, response = sosfreqz(build.post_delay[:, :, delay_line], worN=512)
    magnitude = np.abs(response)
    assert np.all((0.0 < magnitude) & (magnitude < 1.0))


def test_params_to_fdn_build_preserves_nonorthogonal_prediction_without_projection(
    fdn_params: ParameterValues,
) -> None:
    """Decoded feedback is passed through unchanged rather than rejected or repaired.

    :param fdn_params: Valid patch to perturb.
    """
    params = dict(fdn_params)
    feedback = np.ones((8, 8), dtype=np.float64)
    params["feedback_matrix"] = feedback

    build = params_to_fdn_build(params, sample_rate=44_100.0)

    assert build.A is feedback
    assert build.post_delay is not None
    assert np.isfinite(build.post_delay).all()


def test_pyfdn_renderer_fixed_householder_spec_returns_finite_impulse_response() -> None:
    """The fixed-feedback spec produces complete patches consumed by pyFDN."""
    params, _ = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(np.random.default_rng(123))

    audio = PyFDNRenderer().render(params)

    assert audio.shape == (1, 176_400)
    assert audio.dtype == np.float32
    assert np.isfinite(audio).all()


def test_pyfdn_renderer_nonfinite_extreme_prediction_raises_without_repair(
    fdn_params: ParameterValues,
) -> None:
    """An unstable model-range prediction fails only at the finite-output boundary.

    :param fdn_params: Valid patch to move to the model-domain extrema.
    """
    params = dict(fdn_params)
    params["feedback_matrix"] = np.ones((8, 8), dtype=np.float64)
    params["post_delay.rt_dc_seconds"] = 4.0
    params["post_delay.rt_nyquist_seconds"] = 4.0

    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite"):
            PyFDNRenderer().render(params)


@pytest.mark.parametrize(
    ("post_delay", "post_matrix", "post_output", "expected_error", "match"),
    [
        (
            np.zeros((1, 6, 7), dtype=np.float64),
            None,
            None,
            ValueError,
            "shape",
        ),
        (
            np.zeros((1, 6, 8), dtype=np.float32),
            None,
            None,
            TypeError,
            "dtype",
        ),
        (
            np.full((1, 6, 8), np.nan, dtype=np.float64),
            None,
            None,
            ValueError,
            "finite",
        ),
        (
            np.zeros((1, 6, 8), dtype=np.float64),
            np.eye(8),
            None,
            ValueError,
            "remain disabled",
        ),
        (
            np.zeros((1, 6, 8), dtype=np.float64),
            None,
            np.eye(1),
            ValueError,
            "remain disabled",
        ),
    ],
)
def test_params_to_fdn_build_malformed_derived_hooks_raise(
    fdn_params: ParameterValues,
    monkeypatch: pytest.MonkeyPatch,
    post_delay: np.ndarray,
    post_matrix: np.ndarray | None,
    post_output: np.ndarray | None,
    expected_error: type[Exception],
    match: str,
) -> None:
    """Malformed pyFDN-derived hooks fail at the build boundary.

    :param fdn_params: Valid native patch.
    :param monkeypatch: Scoped malformed pyFDN result injection.
    :param post_delay: Candidate derived delay SOS.
    :param post_matrix: Candidate unsupported matrix hook.
    :param post_output: Candidate unsupported output hook.
    :param expected_error: Contract error type for this violation.
    :param match: Diagnostic fragment for this violation.
    """

    def malformed_decay_build(base: FDNBuild, *args: object, **kwargs: object) -> FDNBuild:
        del args, kwargs
        return replace(
            base,
            post_delay=post_delay,
            post_matrix=post_matrix,
            post_output=post_output,
        )

    monkeypatch.setattr(pyfdn_instrument, "build_set_decay", malformed_decay_build)

    with pytest.raises(expected_error, match=match):
        params_to_fdn_build(fdn_params, sample_rate=44_100.0)


@pytest.mark.parametrize(
    "control",
    ["post_delay.rt_dc_seconds", "post_delay.rt_nyquist_seconds"],
)
@pytest.mark.parametrize("bad_rt", [-1.0, 0.0, 0.099, 4.001])
def test_params_to_fdn_build_out_of_bounds_rt_raises(
    fdn_params: ParameterValues,
    control: str,
    bad_rt: float,
) -> None:
    """RT controls outside the predicted semantic domain fail before filter design.

    :param fdn_params: Valid native patch.
    :param control: RT endpoint receiving the invalid value.
    :param bad_rt: Value outside the inclusive RT bounds.
    """
    params = dict(fdn_params)
    params[control] = bad_rt

    with pytest.raises(ValueError, match="between 0.1 and 4.0"):
        params_to_fdn_build(params, sample_rate=44_100.0)


@pytest.mark.parametrize(
    "control",
    ["post_delay.rt_dc_seconds", "post_delay.rt_nyquist_seconds"],
)
@pytest.mark.parametrize("bad_rt", [np.nan, np.inf, -np.inf])
def test_params_to_fdn_build_nonfinite_rt_raises(
    fdn_params: ParameterValues,
    control: str,
    bad_rt: float,
) -> None:
    """Non-finite RT controls never reach native filter design.

    :param fdn_params: Valid native patch.
    :param control: RT endpoint receiving the invalid value.
    :param bad_rt: NaN or infinity injected into the patch.
    """
    params = dict(fdn_params)
    params[control] = bad_rt

    with pytest.raises(ValueError, match="finite"):
        params_to_fdn_build(params, sample_rate=44_100.0)


@pytest.mark.parametrize(
    "control",
    ["post_delay.rt_dc_seconds", "post_delay.rt_nyquist_seconds"],
)
@pytest.mark.parametrize("bad_rt", [True, "1.0"])
def test_params_to_fdn_build_nonreal_rt_raises(
    fdn_params: ParameterValues,
    control: str,
    bad_rt: object,
) -> None:
    """Non-real RT controls fail instead of being coerced.

    :param fdn_params: Valid native patch.
    :param control: RT endpoint receiving the invalid value.
    :param bad_rt: Boolean or string supplied as an RT control.
    """
    params = dict(fdn_params)
    params[control] = cast(float, bad_rt)

    with pytest.raises(TypeError, match="real scalar"):
        params_to_fdn_build(params, sample_rate=44_100.0)


@pytest.mark.parametrize("bad_keys", ["missing", "extra"])
def test_params_to_fdn_build_non_exact_keys_raise(
    fdn_params: ParameterValues, bad_keys: str
) -> None:
    """Missing and extra patch fields fail before native construction.

    :param fdn_params: Valid patch to perturb.
    :param bad_keys: Key-set violation to inject.
    """
    params = dict(fdn_params)
    if bad_keys == "missing":
        del params["direct_matrix"]
    else:
        params["unsupported"] = 0.0

    with pytest.raises(ValueError, match="exactly"):
        params_to_fdn_build(params, sample_rate=44_100.0)


@pytest.mark.parametrize(
    ("name", "bad_shape"),
    [
        ("delays", (1, 8)),
        ("feedback_matrix", (64,)),
        ("input_matrix", (1, 8)),
        ("output_matrix", (8, 1)),
        ("direct_matrix", (1,)),
    ],
)
def test_params_to_fdn_build_wrong_shape_raises(
    fdn_params: ParameterValues, name: str, bad_shape: tuple[int, ...]
) -> None:
    """Every native field requires its exact order-8 mono shape.

    :param fdn_params: Valid patch to perturb.
    :param name: Field receiving the invalid shape.
    :param bad_shape: Shape that preserves element count but violates layout.
    """
    params = dict(fdn_params)
    params[name] = np.asarray(params[name]).reshape(bad_shape)

    with pytest.raises(ValueError, match="shape"):
        params_to_fdn_build(params, sample_rate=44_100.0)


@pytest.mark.parametrize(
    ("name", "bad_dtype"),
    [
        ("delays", np.int32),
        ("feedback_matrix", np.float32),
        ("input_matrix", np.float32),
        ("output_matrix", np.float32),
        ("direct_matrix", np.float32),
    ],
)
def test_params_to_fdn_build_wrong_dtype_raises(
    fdn_params: ParameterValues, name: str, bad_dtype: np.dtype[np.generic]
) -> None:
    """The codec rejects coercible arrays instead of silently converting them.

    :param fdn_params: Valid patch to perturb.
    :param name: Field receiving the invalid dtype.
    :param bad_dtype: Coercible but contract-invalid NumPy dtype.
    """
    params = dict(fdn_params)
    params[name] = np.asarray(params[name]).astype(bad_dtype)

    with pytest.raises(TypeError, match="dtype"):
        params_to_fdn_build(params, sample_rate=44_100.0)


@pytest.mark.parametrize(
    "name",
    ["feedback_matrix", "input_matrix", "output_matrix", "direct_matrix"],
)
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_params_to_fdn_build_non_finite_matrix_value_raises(
    fdn_params: ParameterValues, name: str, bad_value: float
) -> None:
    """Non-finite matrix coefficients never reach pyFDN processing.

    :param fdn_params: Valid patch to perturb.
    :param name: Matrix receiving the non-finite coefficient.
    :param bad_value: NaN or infinity injected into one element.
    """
    params = dict(fdn_params)
    matrix = np.asarray(params[name]).copy()
    matrix.flat[0] = bad_value
    params[name] = matrix

    with pytest.raises(ValueError, match="finite"):
        params_to_fdn_build(params, sample_rate=44_100.0)


@pytest.mark.parametrize("bad_delay", [0, -1])
def test_params_to_fdn_build_nonpositive_delay_raises(
    fdn_params: ParameterValues, bad_delay: int
) -> None:
    """Zero and negative recursion delays are invalid native builds.

    :param fdn_params: Valid patch to perturb.
    :param bad_delay: Nonpositive delay injected into the first line.
    """
    params = dict(fdn_params)
    delays = np.asarray(params["delays"]).copy()
    delays[0] = bad_delay
    params["delays"] = delays

    with pytest.raises(ValueError, match="positive"):
        params_to_fdn_build(params, sample_rate=44_100.0)


@pytest.mark.parametrize("sample_rate", [48_000.0, 44_100.1, np.nan, np.inf])
def test_params_to_fdn_build_non_44100_sample_rate_raises(
    fdn_params: ParameterValues, sample_rate: float
) -> None:
    """The fixed instrument rejects every rate other than exact 44.1 kHz.

    :param fdn_params: Valid native patch.
    :param sample_rate: Unsupported or non-finite sample rate.
    """
    with pytest.raises(ValueError, match="44100"):
        params_to_fdn_build(fdn_params, sample_rate=sample_rate)


def test_pyfdn_renderer_rejects_installed_version_mismatch() -> None:
    """The renderer refuses a requested synth version other than the installed pin."""
    with pytest.raises(ValueError, match="installed pyFDN version"):
        PyFDNRenderer(synth_version="0.0.0")


def test_pyfdn_renderer_instances_hold_independent_immutable_canonical_bytes() -> None:
    """Chirp renderer instances do not share mutable source array state."""
    first = PyFDNRenderer(excitation="chirp")
    second = PyFDNRenderer(excitation="chirp")

    assert first._source_audio is not None
    assert second._source_audio is not None
    np.testing.assert_array_equal(first._source_audio, second._source_audio)
    assert not np.shares_memory(first._source_audio, second._source_audio)
    assert not first._source_audio.flags.writeable
    assert not second._source_audio.flags.writeable


def test_pyfdn_renderer_exposes_provenance_for_custom_chirp() -> None:
    """Chirp provenance identifies the immutable bytes used for processing."""
    renderer = PyFDNRenderer(excitation="chirp")

    assert renderer.source_provenance == canonical_pyfdn_source_provenance()


def test_pyfdn_renderer_source_provenance_caller_mutation_is_isolated() -> None:
    """Caller annotations cannot alter provenance retained by the renderer."""
    renderer = PyFDNRenderer(excitation="chirp")
    provenance = renderer.source_provenance

    provenance["identity"] = "caller-annotation"

    assert renderer.source_provenance["identity"] == "librosa_log_chirp_v1"


def test_pyfdn_renderer_generates_source_once_for_repeated_renders(
    fdn_params: ParameterValues,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One renderer reuses its immutable source across independent FDN runs.

    :param fdn_params: Valid native patch.
    :param monkeypatch: Scoped source-generation counter.
    """
    generated = 0
    generate = pyfdn_instrument.generate_canonical_pyfdn_source

    def count_generation() -> np.ndarray:
        nonlocal generated
        generated += 1
        return generate()

    monkeypatch.setattr(
        pyfdn_instrument,
        "generate_canonical_pyfdn_source",
        count_generation,
    )
    renderer = PyFDNRenderer(excitation="chirp")

    renderer.render(fdn_params)
    renderer.render(fdn_params)

    assert generated == 1


@pytest.mark.parametrize(
    ("control", "changed_rt", "response_index"),
    [
        ("post_delay.rt_dc_seconds", 2.0, 0),
        ("post_delay.rt_nyquist_seconds", 2.0, -1),
    ],
)
def test_pyfdn_renderer_rt_control_change_changes_sos_and_audio(
    fdn_params: ParameterValues,
    control: str,
    changed_rt: float,
    response_index: int,
) -> None:
    """Changing either predicted RT endpoint changes derived filters and real output.

    :param fdn_params: Valid native patch with unequal endpoint RTs.
    :param control: RT endpoint changed for this case.
    :param changed_rt: Replacement reverberation time in seconds.
    :param response_index: DC or near-Nyquist response index.
    """
    changed_params = dict(fdn_params)
    changed_params[control] = changed_rt

    baseline_build = params_to_fdn_build(fdn_params, sample_rate=44_100.0)
    changed_build = params_to_fdn_build(changed_params, sample_rate=44_100.0)
    renderer = PyFDNRenderer()
    baseline_audio = renderer.render(fdn_params)
    changed_audio = renderer.render(changed_params)

    assert baseline_build.post_delay is not None
    assert changed_build.post_delay is not None
    _, baseline_response_raw = sosfreqz(
        baseline_build.post_delay[:, :, 0], worN=512
    )
    _, changed_response_raw = sosfreqz(changed_build.post_delay[:, :, 0], worN=512)
    baseline_response = cast(np.ndarray, baseline_response_raw)
    changed_response = cast(np.ndarray, changed_response_raw)
    assert np.abs(changed_response[response_index]) > np.abs(
        baseline_response[response_index]
    )
    assert not np.array_equal(changed_build.post_delay, baseline_build.post_delay)
    assert not np.array_equal(changed_audio, baseline_audio)


def test_pyfdn_renderer_default_returns_impulse_response(
    fdn_params: ParameterValues,
) -> None:
    """Default rendering exposes the FDN response to a unit impulse.

    :param fdn_params: Valid native patch whose first delayed output starts at frame 400.
    """
    audio = PyFDNRenderer().render(fdn_params)

    assert audio[0, 0] == pytest.approx(0.5)
    np.testing.assert_array_equal(audio[0, 1:400], np.zeros(399, dtype=np.float32))


def test_pyfdn_renderer_custom_chirp_changes_excitation(
    fdn_params: ParameterValues,
) -> None:
    """The historical chirp render remains available only by explicit selection.

    :param fdn_params: Valid native patch.
    """
    impulse_response = PyFDNRenderer().render(fdn_params)
    chirp_response = PyFDNRenderer(excitation="chirp").render(fdn_params)

    assert not np.array_equal(chirp_response, impulse_response)


def test_pyfdn_renderer_real_process_has_exact_output_contract(
    fdn_params: ParameterValues,
) -> None:
    """Real pyFDN processing returns finite contiguous channel-first float32 audio.

    :param fdn_params: Valid native patch.
    """
    audio = PyFDNRenderer().render(fdn_params)

    assert audio.shape == (1, 176_400)
    assert audio.dtype == np.float32
    assert audio.flags.c_contiguous
    assert np.isfinite(audio).all()


def test_pyfdn_renderer_repeated_renders_reset_recursion_state(
    fdn_params: ParameterValues,
) -> None:
    """Rendering the same patch twice starts from fresh zero recursion state.

    :param fdn_params: Valid native patch.
    """
    renderer = PyFDNRenderer()

    first = renderer.render(fdn_params)
    second = renderer.render(fdn_params)

    np.testing.assert_array_equal(second, first)


def test_pyfdn_renderer_does_not_clip_native_output(
    fdn_params: ParameterValues,
) -> None:
    """The renderer preserves finite native amplitudes outside the audio unit interval.

    :param fdn_params: Valid native patch.
    """
    params = dict(fdn_params)
    params["direct_matrix"] = np.array([[20.0]], dtype=np.float64)

    audio = PyFDNRenderer().render(params)

    assert np.max(np.abs(audio)) > 1.0


def test_pyfdn_renderer_implements_common_audio_renderer_signature() -> None:
    """The fixed-source backend accepts the existing MIDI compatibility inputs."""
    render_names = set(inspect.signature(PyFDNRenderer.render).parameters)

    assert render_names == {
        "self",
        "params",
        "midi_note",
        "velocity",
        "note_start_and_end",
        "warmup",
    }
