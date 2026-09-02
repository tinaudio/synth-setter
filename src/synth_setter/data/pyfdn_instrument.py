"""Native pyFDN build conversion and fixed-source instrument rendering.

Example:
    ``PyFDNRenderer(source_path, sha256).render(native_params)`` returns channel-first audio.
"""

import hashlib
import io
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf
from jaxtyping import Float32
from pyFDN import FDNBuild, process_fdn

from synth_setter.data.pyfdn_param_spec import PYFDN_ORDER
from synth_setter.data.vst.param_spec import ParameterValue, ParameterValues

_PYFDN_VERSION = "0.4.2"
_SAMPLE_RATE = 48_000.0
_CHANNELS = 1
_SIGNAL_DURATION_SECONDS = 4.0
_SIGNAL_LENGTH = 192_000
_ARRAY_CONTRACTS = (
    ("feedback_matrix", (PYFDN_ORDER, PYFDN_ORDER), np.dtype(np.float64)),
    ("input_matrix", (PYFDN_ORDER, _CHANNELS), np.dtype(np.float64)),
    ("output_matrix", (_CHANNELS, PYFDN_ORDER), np.dtype(np.float64)),
    ("direct_matrix", (_CHANNELS, _CHANNELS), np.dtype(np.float64)),
    ("delays", (PYFDN_ORDER,), np.dtype(np.int64)),
)
_REQUIRED_KEYS = frozenset(name for name, _, _ in _ARRAY_CONTRACTS)


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

    arrays = {
        name: _require_array(name, params[name], shape=shape, dtype=dtype)
        for name, shape, dtype in _ARRAY_CONTRACTS
    }
    if np.any(arrays["delays"] <= 0):
        raise ValueError("delays must be positive")

    return FDNBuild(
        A=arrays["feedback_matrix"],
        B=arrays["input_matrix"],
        C=arrays["output_matrix"],
        D=arrays["direct_matrix"],
        delays=arrays["delays"],
        fs=_SAMPLE_RATE,
        post_delay=None,
        post_matrix=None,
        post_output=None,
    )


def _validate_version(synth_version: str) -> None:
    """Require the exact installed and requested pyFDN version.

    :param synth_version: Version requested by the renderer caller.
    :raises ValueError: Installed or requested pyFDN does not equal the runtime pin.
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


def _validate_geometry(
    sample_rate: int, channels: int, signal_duration_seconds: float
) -> None:
    """Require the fixed F1 source geometry.

    :param sample_rate: Candidate source rate in Hz.
    :param channels: Candidate source channel count.
    :param signal_duration_seconds: Candidate source duration in seconds.
    :raises ValueError: Any geometry field differs from the fixed contract.
    """
    if (
        sample_rate != _SAMPLE_RATE
        or channels != _CHANNELS
        or signal_duration_seconds != _SIGNAL_DURATION_SECONDS
    ):
        raise ValueError("pyFDN renderer geometry is fixed at 48 kHz mono for 4 seconds")


def _load_source(
    path: Path,
    source_audio_sha256: str,
    *,
    sample_rate: int,
    channels: int,
) -> np.ndarray:
    """Validate and decode the exact checksum-verified source bytes.

    :param path: Path read once to obtain the immutable source bytes.
    :param source_audio_sha256: Expected SHA-256 of those bytes.
    :param sample_rate: Required decoded sample rate in Hz.
    :param channels: Required decoded channel count.
    :returns: Contiguous mono float64 samples shaped ``(192000,)``.
    :raises ValueError: Checksum, metadata, shape, or samples violate the fixed contract.
    """
    source_bytes = path.read_bytes()
    actual_checksum = hashlib.sha256(source_bytes).hexdigest()
    if actual_checksum != source_audio_sha256.lower():
        raise ValueError("source audio SHA-256 does not match source_audio_sha256")

    info = sf.info(io.BytesIO(source_bytes))
    if info.samplerate != sample_rate:
        raise ValueError(f"source sample rate must be {sample_rate}, got {info.samplerate}")
    if info.channels != channels:
        raise ValueError(f"source channels must be {channels}, got {info.channels}")
    if info.frames != _SIGNAL_LENGTH:
        raise ValueError(f"source frames must be {_SIGNAL_LENGTH}, got {info.frames}")

    decoded, decoded_rate = sf.read(
        io.BytesIO(source_bytes), dtype="float64", always_2d=True
    )
    if decoded_rate != sample_rate:
        raise ValueError(f"decoded sample rate must be {sample_rate}, got {decoded_rate}")
    if decoded.shape != (_SIGNAL_LENGTH, _CHANNELS):
        raise ValueError(
            f"decoded source must have shape {(_SIGNAL_LENGTH, _CHANNELS)}, got {decoded.shape}"
        )
    if not np.isfinite(decoded).all():
        raise ValueError("decoded source must contain only finite values")
    return np.ascontiguousarray(decoded[:, 0], dtype=np.float64)


class PyFDNRenderer:
    """Render one checksum-pinned mono source through native pyFDN patches."""

    def __init__(
        self,
        source_audio_path: str | Path,
        source_audio_sha256: str,
        *,
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
        """
        _validate_version(synth_version)
        _validate_geometry(sample_rate, channels, signal_duration_seconds)
        self._source_audio = _load_source(
            Path(source_audio_path),
            source_audio_sha256,
            sample_rate=sample_rate,
            channels=channels,
        )

    def render(
        self, params: ParameterValues
    ) -> Float32[np.ndarray, "1 192000"]:
        """Process the fixed source through one exact patch with fresh recursion state.

        :param params: Native order-8 mono pyFDN arrays.
        :returns: Contiguous finite channel-first float32 audio shaped ``(1, 192000)``; native
            amplitude is preserved without clipping or normalization.
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
