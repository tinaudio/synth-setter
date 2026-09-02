"""Native pyFDN build conversion and fixed-source instrument rendering."""

import hashlib
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf
from jaxtyping import Float32
from pyFDN import FDNBuild, process_fdn

from synth_setter.data.vst.param_spec import ParameterValue, ParameterValues

_PYFDN_VERSION = "0.4.2"
_SAMPLE_RATE = 48_000.0
_CHANNELS = 1
_SIGNAL_DURATION_SECONDS = 4.0
_SIGNAL_LENGTH = 192_000
_REQUIRED_KEYS = frozenset({
    "feedback_matrix",
    "input_matrix",
    "output_matrix",
    "direct_matrix",
    "delays",
})


def _require_array(
    name: str,
    value: ParameterValue,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[np.generic],
) -> np.ndarray:
    """Validate one native array without coercing or copying it.

    :param name: Patch field name used in validation errors.
    :param value: Native patch value to validate.
    :param shape: Required array shape.
    :param dtype: Required NumPy dtype.
    :returns: The original validated array.
    :raises TypeError: The value is not an array or has the wrong dtype.
    :raises ValueError: The array has the wrong shape or non-finite values.
    """
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def params_to_fdn_build(
    params: ParameterValues,
    *,
    sample_rate: float,
) -> FDNBuild:
    """Convert one exact native patch into an order-8 mono pyFDN build.

    :param params: Exact feedback, input, output, direct, and delay arrays.
    :param sample_rate: Fixed processing rate; only ``48000.0`` is valid.
    :returns: Native build with every optional post-processing hook disabled.
    :raises ValueError: Keys, shapes, values, delays, or sample rate violate the fixed contract.
    """
    if set(params) != _REQUIRED_KEYS:
        raise ValueError(f"params must contain exactly {sorted(_REQUIRED_KEYS)}")
    if sample_rate != _SAMPLE_RATE:
        raise ValueError("sample_rate must be exactly 48000.0")

    feedback_matrix = _require_array(
        "feedback_matrix",
        params["feedback_matrix"],
        shape=(8, 8),
        dtype=np.dtype(np.float64),
    )
    input_matrix = _require_array(
        "input_matrix",
        params["input_matrix"],
        shape=(8, 1),
        dtype=np.dtype(np.float64),
    )
    output_matrix = _require_array(
        "output_matrix",
        params["output_matrix"],
        shape=(1, 8),
        dtype=np.dtype(np.float64),
    )
    direct_matrix = _require_array(
        "direct_matrix",
        params["direct_matrix"],
        shape=(1, 1),
        dtype=np.dtype(np.float64),
    )
    delays = _require_array(
        "delays", params["delays"], shape=(8,), dtype=np.dtype(np.int64)
    )
    if np.any(delays <= 0):
        raise ValueError("delays must be positive")

    return FDNBuild(
        A=feedback_matrix,
        B=input_matrix,
        C=output_matrix,
        D=direct_matrix,
        delays=delays,
        fs=_SAMPLE_RATE,
        post_delay=None,
        post_matrix=None,
        post_output=None,
    )


class PyFDNRenderer:
    """Render one checksum-pinned mono source through native pyFDN patches."""

    def __init__(
        self,
        source_audio_path: str | Path,
        source_audio_sha256: str,
        synth_version: str = _PYFDN_VERSION,
        sample_rate: int = 48_000,
        channels: int = _CHANNELS,
        signal_duration_seconds: float = _SIGNAL_DURATION_SECONDS,
    ) -> None:
        """Load and validate the fixed source for this process-local renderer.

        :param source_audio_path: Path to the exact lossless mono source.
        :param source_audio_sha256: Expected SHA-256 of the stored source bytes.
        :param synth_version: Required installed pyFDN version.
        :param sample_rate: Fixed source and build sample rate in Hz.
        :param channels: Fixed source channel count.
        :param signal_duration_seconds: Fixed source duration in seconds.
        :raises ValueError: Version, geometry, checksum, metadata, or samples violate the contract.
        """
        installed_version = version("pyFDN")
        if installed_version != _PYFDN_VERSION:
            raise ValueError(
                f"installed pyFDN version must be {_PYFDN_VERSION}, got {installed_version}"
            )
        if synth_version != installed_version:
            raise ValueError(
                f"installed pyFDN version {installed_version} does not match "
                f"requested version {synth_version}"
            )
        if (
            sample_rate != _SAMPLE_RATE
            or channels != _CHANNELS
            or signal_duration_seconds != _SIGNAL_DURATION_SECONDS
        ):
            raise ValueError("pyFDN renderer geometry is fixed at 48 kHz mono for 4 seconds")

        path = Path(source_audio_path)
        actual_checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_checksum != source_audio_sha256.lower():
            raise ValueError("source audio SHA-256 does not match source_audio_sha256")

        info = sf.info(path)
        if info.samplerate != sample_rate:
            raise ValueError(
                f"source sample rate must be {sample_rate}, got {info.samplerate}"
            )
        if info.channels != channels:
            raise ValueError(f"source channels must be {channels}, got {info.channels}")
        if info.frames != _SIGNAL_LENGTH:
            raise ValueError(f"source frames must be {_SIGNAL_LENGTH}, got {info.frames}")

        decoded, decoded_rate = sf.read(path, dtype="float64", always_2d=True)
        if decoded_rate != sample_rate:
            raise ValueError(f"decoded sample rate must be {sample_rate}, got {decoded_rate}")
        if decoded.shape != (_SIGNAL_LENGTH, _CHANNELS):
            raise ValueError(
                "decoded source must have shape "
                f"{(_SIGNAL_LENGTH, _CHANNELS)}, got {decoded.shape}"
            )
        if not np.isfinite(decoded).all():
            raise ValueError("decoded source must contain only finite values")
        self._source_audio = np.ascontiguousarray(decoded[:, 0], dtype=np.float64)

    def render(
        self, params: ParameterValues
    ) -> Float32[np.ndarray, "1 192000"]:
        """Process the fixed source through one exact patch with fresh recursion state.

        :param params: Native order-8 mono pyFDN arrays.
        :returns: Contiguous finite channel-first float32 audio shaped ``(1, 192000)``.
        :raises ValueError: The patch or rendered audio violates the fixed contract.
        """
        build = params_to_fdn_build(params, sample_rate=_SAMPLE_RATE)
        output = process_fdn(
            self._source_audio,
            build.delays,
            build.A,
            build.B,
            build.C,
            build.D,
            post_delay=build.post_delay,
            post_matrix=build.post_matrix,
            post_output=build.post_output,
        )
        output_array = np.asarray(output)
        if output_array.shape != (_SIGNAL_LENGTH,):
            raise ValueError(
                f"pyFDN output must have shape {(_SIGNAL_LENGTH,)}, got {output_array.shape}"
            )
        if not np.isfinite(output_array).all():
            raise ValueError("pyFDN output must contain only finite values")
        with np.errstate(over="ignore"):
            audio = np.ascontiguousarray(output_array, dtype=np.float32).reshape(
                _CHANNELS, _SIGNAL_LENGTH
            )
        if not np.isfinite(audio).all():
            raise ValueError("float32 pyFDN output must contain only finite values")
        return audio
