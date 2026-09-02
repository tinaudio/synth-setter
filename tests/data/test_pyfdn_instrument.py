"""Native build and fixed-source rendering contracts for the pyFDN instrument."""

import hashlib
import inspect
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import soundfile as sf
from pyFDN import FDNBuild
from scipy.signal import sosfreqz

import synth_setter.data.pyfdn_instrument as pyfdn_instrument
from synth_setter.data.pyfdn_instrument import PyFDNRenderer, params_to_fdn_build
from synth_setter.data.vst.param_spec import ParameterValues


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    build = params_to_fdn_build(fdn_params, sample_rate=48_000.0)

    assert isinstance(build, FDNBuild)
    assert build.A is fdn_params["feedback_matrix"]
    assert build.B is fdn_params["input_matrix"]
    assert build.C is fdn_params["output_matrix"]
    assert build.D is fdn_params["direct_matrix"]
    assert build.delays is fdn_params["delays"]
    assert build.fs == 48_000.0
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
    first = params_to_fdn_build(fdn_params, sample_rate=48_000.0)
    second = params_to_fdn_build(fdn_params, sample_rate=48_000.0)
    expected_end_lines = np.array(
        [
            [
                [0.9172759353897796, 0.9158909125939455],
                [0.0131997506097520, 0.0134104340706896],
                [0.0, 0.0],
                [1.0, 1.0],
                [-0.0143901634181029, -0.0146419555934989],
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

    build = params_to_fdn_build(params, sample_rate=48_000.0)

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
    feedback = np.eye(8, dtype=np.float64)
    feedback[0, 0] = 0.9
    params["feedback_matrix"] = feedback

    build = params_to_fdn_build(params, sample_rate=48_000.0)

    assert build.A is feedback
    assert build.post_delay is not None
    assert np.isfinite(build.post_delay).all()


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
        params_to_fdn_build(fdn_params, sample_rate=48_000.0)


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
        params_to_fdn_build(params, sample_rate=48_000.0)


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
        params_to_fdn_build(params, sample_rate=48_000.0)


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
        params_to_fdn_build(params, sample_rate=48_000.0)


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
        params_to_fdn_build(params, sample_rate=48_000.0)


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
        params_to_fdn_build(params, sample_rate=48_000.0)


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
        params_to_fdn_build(params, sample_rate=48_000.0)


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
        params_to_fdn_build(params, sample_rate=48_000.0)


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
        params_to_fdn_build(params, sample_rate=48_000.0)


@pytest.mark.parametrize("sample_rate", [44_100.0, 48_000.1, np.nan, np.inf])
def test_params_to_fdn_build_non_48000_sample_rate_raises(
    fdn_params: ParameterValues, sample_rate: float
) -> None:
    """The fixed instrument rejects every rate other than exact 48 kHz.

    :param fdn_params: Valid native patch.
    :param sample_rate: Unsupported or non-finite sample rate.
    """
    with pytest.raises(ValueError, match="48000"):
        params_to_fdn_build(fdn_params, sample_rate=sample_rate)


def test_pyfdn_renderer_rejects_installed_version_mismatch(
    source_file: tuple[Path, str],
) -> None:
    """The renderer refuses a requested synth version other than the installed pin.

    :param source_file: Valid fixed source and checksum.
    """
    path, checksum = source_file

    with pytest.raises(ValueError, match="installed pyFDN version"):
        PyFDNRenderer(path, checksum, synth_version="0.0.0")


def test_pyfdn_renderer_rejects_source_checksum_mismatch(
    source_file: tuple[Path, str],
) -> None:
    """Source identity is bound to exact file bytes before decoding.

    :param source_file: Valid fixed source and checksum.
    """
    path, _ = source_file

    with pytest.raises(ValueError, match="SHA-256"):
        PyFDNRenderer(path, "0" * 64)


def test_pyfdn_renderer_rejects_lossy_source_subtype(tmp_path: Path) -> None:
    """A companded WAV cannot satisfy the fixed lossless source contract.

    :param tmp_path: Temporary directory owned by pytest.
    """
    path = tmp_path / "lossy.wav"
    sf.write(path, np.zeros(192_000), 48_000, subtype="ULAW")

    with pytest.raises(ValueError, match="lossless WAV"):
        PyFDNRenderer(path, _sha256(path))


def test_pyfdn_renderer_accepts_lossless_pcm_u8_source(
    tmp_path: Path,
    fdn_params: ParameterValues,
) -> None:
    """A checksum-pinned PCM_U8 WAV renders through the production processor.

    :param tmp_path: Temporary directory owned by pytest.
    :param fdn_params: Valid native patch.
    """
    path = tmp_path / "pcm-u8.wav"
    sf.write(path, np.zeros(192_000), 48_000, subtype="PCM_U8")

    audio = PyFDNRenderer(path, _sha256(path)).render(fdn_params)

    assert audio.shape == (1, 192_000)
    assert audio.dtype == np.float32
    assert np.isfinite(audio).all()


def test_pyfdn_renderer_post_delay_attenuates_feedback_impulse(tmp_path: Path) -> None:
    """Positive RT controls produce a finite, progressively attenuated feedback response.

    :param tmp_path: Temporary directory owned by pytest.
    """
    path = tmp_path / "impulse.wav"
    source = np.zeros(192_000, dtype=np.float32)
    source[0] = 1.0
    sf.write(path, source, 48_000, subtype="FLOAT")
    input_matrix = np.zeros((8, 1), dtype=np.float64)
    input_matrix[0, 0] = 1.0
    output_matrix = np.zeros((1, 8), dtype=np.float64)
    output_matrix[0, 0] = 1.0
    params: ParameterValues = {
        "delays": np.full(8, 8, dtype=np.int64),
        "feedback_matrix": np.diag([-1.0, *([1.0] * 7)]).astype(np.float64),
        "input_matrix": input_matrix,
        "output_matrix": output_matrix,
        "direct_matrix": np.zeros((1, 1), dtype=np.float64),
        "post_delay.rt_dc_seconds": 0.1,
        "post_delay.rt_nyquist_seconds": 0.1,
    }

    response = PyFDNRenderer(path, _sha256(path)).render(params)

    first, second, third = np.abs(response[0, [8, 16, 24]])
    assert np.isfinite(response).all()
    assert 0.0 < third < second < first < 1.0


@pytest.mark.parametrize(
    ("control", "changed_rt", "response_index"),
    [
        ("post_delay.rt_dc_seconds", 2.0, 0),
        ("post_delay.rt_nyquist_seconds", 2.0, -1),
    ],
)
def test_pyfdn_renderer_rt_control_change_changes_sos_and_audio(
    source_file: tuple[Path, str],
    fdn_params: ParameterValues,
    control: str,
    changed_rt: float,
    response_index: int,
) -> None:
    """Changing either predicted RT endpoint changes derived filters and real output.

    :param source_file: Valid fixed source and checksum.
    :param fdn_params: Valid native patch with unequal endpoint RTs.
    :param control: RT endpoint changed for this case.
    :param changed_rt: Replacement reverberation time in seconds.
    :param response_index: DC or near-Nyquist response index.
    """
    path, checksum = source_file
    changed_params = dict(fdn_params)
    changed_params[control] = changed_rt

    baseline_build = params_to_fdn_build(fdn_params, sample_rate=48_000.0)
    changed_build = params_to_fdn_build(changed_params, sample_rate=48_000.0)
    renderer = PyFDNRenderer(path, checksum)
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


def test_pyfdn_renderer_rejects_source_sample_rate(tmp_path: Path) -> None:
    """A lossless mono source at any rate other than 48 kHz is invalid.

    :param tmp_path: Temporary directory owned by pytest.
    """
    path = tmp_path / "wrong-rate.wav"
    sf.write(path, np.zeros(192_000), 44_100, subtype="PCM_16")

    with pytest.raises(ValueError, match="sample rate"):
        PyFDNRenderer(path, _sha256(path))


def test_pyfdn_renderer_rejects_source_channels(tmp_path: Path) -> None:
    """A 48 kHz source with more than one channel is invalid.

    :param tmp_path: Temporary directory owned by pytest.
    """
    path = tmp_path / "stereo.wav"
    sf.write(path, np.zeros((192_000, 2)), 48_000, subtype="PCM_16")

    with pytest.raises(ValueError, match="channels"):
        PyFDNRenderer(path, _sha256(path))


def test_pyfdn_renderer_rejects_source_frame_count(tmp_path: Path) -> None:
    """The renderer never crops or pads a source that is not exactly four seconds.

    :param tmp_path: Temporary directory owned by pytest.
    """
    path = tmp_path / "short.wav"
    sf.write(path, np.zeros(191_999), 48_000, subtype="PCM_16")

    with pytest.raises(ValueError, match="frames"):
        PyFDNRenderer(path, _sha256(path))


def test_pyfdn_renderer_rejects_non_finite_decoded_source(tmp_path: Path) -> None:
    """NaN source samples fail instead of entering the recursion.

    :param tmp_path: Temporary directory owned by pytest.
    """
    path = tmp_path / "non-finite.wav"
    audio = np.zeros(192_000, dtype=np.float32)
    audio[100] = np.nan
    sf.write(path, audio, 48_000, subtype="FLOAT")

    with pytest.raises(ValueError, match="finite"):
        PyFDNRenderer(path, _sha256(path))


@pytest.mark.parametrize(
    ("sample_rate", "channels", "signal_duration_seconds"),
    [(44_100, 1, 4.0), (48_000, 2, 4.0), (48_000, 1, 3.0)],
)
def test_pyfdn_renderer_rejects_non_fixed_audio_geometry(
    source_file: tuple[Path, str],
    sample_rate: int,
    channels: int,
    signal_duration_seconds: float,
) -> None:
    """Public geometry inputs cannot select an unsupported FDN mode.

    :param source_file: Valid fixed source and checksum.
    :param sample_rate: Candidate fixed source rate.
    :param channels: Candidate fixed source channel count.
    :param signal_duration_seconds: Candidate fixed source duration.
    """
    path, checksum = source_file

    with pytest.raises(ValueError, match="fixed"):
        PyFDNRenderer(
            path,
            checksum,
            sample_rate=sample_rate,
            channels=channels,
            signal_duration_seconds=signal_duration_seconds,
        )


def test_pyfdn_renderer_source_replacement_after_checksum_uses_checked_bytes(
    source_file: tuple[Path, str],
    fdn_params: ParameterValues,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decoding consumes the exact bytes whose SHA-256 passed validation.

    :param source_file: Valid fixed source and checksum.
    :param fdn_params: Valid native patch.
    :param tmp_path: Temporary directory owned by pytest.
    :param monkeypatch: Scoped replacement-race injection.
    """
    path, checksum = source_file
    checked_source, _ = sf.read(path, dtype="float64")
    replacement_path = tmp_path / "replacement.wav"
    sf.write(replacement_path, np.zeros(192_000), 48_000, subtype="PCM_16")
    original_read_bytes = Path.read_bytes

    def read_then_replace(candidate: Path) -> bytes:
        checked_bytes = original_read_bytes(candidate)
        candidate.write_bytes(original_read_bytes(replacement_path))
        return checked_bytes

    monkeypatch.setattr(Path, "read_bytes", read_then_replace)
    params = dict(fdn_params)
    params["input_matrix"] = np.zeros((8, 1), dtype=np.float64)
    params["output_matrix"] = np.zeros((1, 8), dtype=np.float64)
    params["direct_matrix"] = np.ones((1, 1), dtype=np.float64)

    audio = PyFDNRenderer(path, checksum).render(params)

    np.testing.assert_array_equal(audio[0], checked_source.astype(np.float32))


def test_pyfdn_renderer_real_process_has_exact_output_contract(
    source_file: tuple[Path, str], fdn_params: ParameterValues
) -> None:
    """Real pyFDN processing returns finite contiguous channel-first float32 audio.

    :param source_file: Valid fixed source and checksum.
    :param fdn_params: Valid native patch.
    """
    path, checksum = source_file
    renderer = PyFDNRenderer(path, checksum)

    audio = renderer.render(fdn_params)

    assert audio.shape == (1, 192_000)
    assert audio.dtype == np.float32
    assert audio.flags.c_contiguous
    assert np.isfinite(audio).all()


def test_pyfdn_renderer_repeated_renders_reset_recursion_state(
    source_file: tuple[Path, str], fdn_params: ParameterValues
) -> None:
    """Rendering the same patch twice starts from fresh zero recursion state.

    :param source_file: Valid fixed source and checksum.
    :param fdn_params: Valid native patch.
    """
    path, checksum = source_file
    renderer = PyFDNRenderer(path, checksum)

    first = renderer.render(fdn_params)
    second = renderer.render(fdn_params)

    np.testing.assert_array_equal(second, first)


def test_pyfdn_renderer_does_not_clip_native_output(
    source_file: tuple[Path, str], fdn_params: ParameterValues
) -> None:
    """The renderer preserves finite native amplitudes outside the audio unit interval.

    :param source_file: Valid fixed source and checksum.
    :param fdn_params: Valid native patch.
    """
    path, checksum = source_file
    params = dict(fdn_params)
    params["direct_matrix"] = np.array([[20.0]], dtype=np.float64)

    audio = PyFDNRenderer(path, checksum).render(params)

    assert np.max(np.abs(audio)) > 1.0


def test_pyfdn_renderer_public_signature_has_no_midi_or_effect_options() -> None:
    """The fixed-source domain API exposes only native patch rendering."""
    constructor_names = set(inspect.signature(PyFDNRenderer).parameters)
    render_names = set(inspect.signature(PyFDNRenderer.render).parameters)

    assert constructor_names == {
        "source_audio_path",
        "source_audio_sha256",
        "synth_version",
        "sample_rate",
        "channels",
        "signal_duration_seconds",
    }
    assert render_names == {"self", "params"}
