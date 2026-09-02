"""Native build and fixed-source rendering contracts for the pyFDN instrument."""

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pyFDN import FDNBuild

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
    }


def test_params_to_fdn_build_reconstructs_exact_native_build(
    fdn_params: ParameterValues,
) -> None:
    """The codec maps each native field without projection, repair, or optional hooks.

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
    assert build.post_delay is None
    assert build.post_matrix is None
    assert build.post_output is None


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
