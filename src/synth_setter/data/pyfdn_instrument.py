"""Native pyFDN build conversion and canonical-source instrument rendering.

Example:
    ``PyFDNRenderer().render(native_params)`` returns channel-first audio.
"""

from collections.abc import Mapping
from importlib.metadata import version
from numbers import Real
from typing import cast

import numpy as np
from jaxtyping import Float32
from pyFDN import FDNBuild, build_set_decay, process_fdn
from pyFDN.td import SOSBank

from synth_setter.data.pyfdn_param_spec import (
    PYFDN_ORDER,
    PYFDN_RT_CROSSOVER_HZ,
    PYFDN_RT_DC_NAME,
    PYFDN_RT_MAX_SECONDS,
    PYFDN_RT_MIN_SECONDS,
    PYFDN_RT_NYQUIST_NAME,
)
from synth_setter.data.pyfdn_source import (
    PYFDN_SOURCE_CHANNELS,
    PYFDN_SOURCE_SAMPLE_RATE_HZ,
    PYFDN_SOURCE_TOTAL_FRAMES,
    PyFDNSourceProvenance,
    _canonical_pyfdn_source_provenance,
    generate_canonical_pyfdn_source,
)
from synth_setter.data.vst.param_spec import ParameterValue, ParameterValues

_PYFDN_VERSION = "0.4.2"
_SAMPLE_RATE = float(PYFDN_SOURCE_SAMPLE_RATE_HZ)
_CHANNELS = PYFDN_SOURCE_CHANNELS
_SIGNAL_LENGTH = PYFDN_SOURCE_TOTAL_FRAMES
_POST_DELAY_SOS_SHAPE = (1, 6, PYFDN_ORDER)
_ARRAY_CONTRACTS = (
    ("feedback_matrix", (PYFDN_ORDER, PYFDN_ORDER), np.dtype(np.float64)),
    ("input_matrix", (PYFDN_ORDER, _CHANNELS), np.dtype(np.float64)),
    ("output_matrix", (_CHANNELS, PYFDN_ORDER), np.dtype(np.float64)),
    ("direct_matrix", (_CHANNELS, _CHANNELS), np.dtype(np.float64)),
    ("delays", (PYFDN_ORDER,), np.dtype(np.int64)),
)
_REQUIRED_KEYS = frozenset(name for name, _, _ in _ARRAY_CONTRACTS).union(
    {PYFDN_RT_DC_NAME, PYFDN_RT_NYQUIST_NAME}
)


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


def _require_rt_seconds(name: str, value: ParameterValue) -> float:
    """Validate one RT control against the predicted semantic domain.

    :param name: Patch field name used in validation errors.
    :param value: Native scalar to validate.
    :returns: The validated reverberation time in seconds.
    :raises TypeError: The value is not a real scalar.
    :raises ValueError: The value is non-finite or outside the RT bounds.
    """
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if not PYFDN_RT_MIN_SECONDS <= result <= PYFDN_RT_MAX_SECONDS:
        raise ValueError(
            f"{name} must be between {PYFDN_RT_MIN_SECONDS} and "
            f"{PYFDN_RT_MAX_SECONDS} seconds"
        )
    return result


def _validate_decay_hooks(build: FDNBuild) -> None:
    """Validate pyFDN's derived delay-filter build contract.

    :param build: Result returned by ``build_set_decay``.
    :raises TypeError: ``post_delay`` is not a float64 NumPy array.
    :raises ValueError: Hook shape, values, or unsupported hooks violate the contract.
    """
    post_delay = build.post_delay
    if not isinstance(post_delay, np.ndarray):
        raise TypeError("post_delay must be a NumPy array")
    if post_delay.shape != _POST_DELAY_SOS_SHAPE:
        raise ValueError(
            f"post_delay must have shape {_POST_DELAY_SOS_SHAPE}, got {post_delay.shape}"
        )
    if post_delay.dtype != np.float64:
        raise TypeError(f"post_delay must have dtype float64, got {post_delay.dtype}")
    if not np.isfinite(post_delay).all():
        raise ValueError("post_delay must contain only finite values")
    if build.post_matrix is not None or build.post_output is not None:
        raise ValueError("post_matrix and post_output must remain disabled")


def _build_decay_fdn(
    arrays: Mapping[str, np.ndarray],
    rt_seconds: tuple[float, float],
) -> FDNBuild:
    """Derive native delay filters from an exact base FDN.

    :param arrays: Float64 feedback ``(8, 8)``, input ``(8, 1)``, output ``(1, 8)``,
        and direct ``(1, 1)`` matrices plus int64 delays ``(8,)``.
    :param rt_seconds: DC and Nyquist reverberation times in seconds.
    :returns: Build carrying the validated delay-filter SOS.
    """
    base_build = FDNBuild(
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
    decay_build = build_set_decay(
        base_build,
        rt=rt_seconds,
        rt_crossover=PYFDN_RT_CROSSOVER_HZ,
    )
    _validate_decay_hooks(decay_build)
    return decay_build


def params_to_fdn_build(
    params: ParameterValues,
    *,
    sample_rate: float,
) -> FDNBuild:
    """Build an order-8 mono FDN from an exact native parameter mapping.

    :param params: Mapping containing ``feedback_matrix`` float64 ``(8, 8)``,
        ``input_matrix`` float64 ``(8, 1)``, ``output_matrix`` float64 ``(1, 8)``,
        ``direct_matrix`` float64 ``(1, 1)``, positive ``delays`` int64 ``(8,)``,
        and finite scalar DC and Nyquist reverberation times in seconds; every array
        must contain only finite values.
    :param sample_rate: Processing rate in Hz; exactly ``48000.0``.
    :returns: Native build with derived ``post_delay`` SOS and no other post hooks.
    :raises ValueError: Keys, shapes, values, delays, or sample rate violate the contract.
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
    rt_seconds = (
        _require_rt_seconds(PYFDN_RT_DC_NAME, params[PYFDN_RT_DC_NAME]),
        _require_rt_seconds(PYFDN_RT_NYQUIST_NAME, params[PYFDN_RT_NYQUIST_NAME]),
    )
    return _build_decay_fdn(arrays, rt_seconds)


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


class PyFDNRenderer:
    """Render the canonical procedural source through native pyFDN patches."""

    def __init__(self, *, synth_version: str = _PYFDN_VERSION) -> None:
        """Generate the immutable source for this process-local renderer.

        :param synth_version: Required installed pyFDN version.
        """
        _validate_version(synth_version)
        self._source_audio = generate_canonical_pyfdn_source()
        self._source_provenance = _canonical_pyfdn_source_provenance(self._source_audio)

    @property
    def source_provenance(self) -> PyFDNSourceProvenance:
        """Return provenance for the source bytes used by this renderer.

        :returns: Independent provenance metadata safe for caller mutation.
        """
        return self._source_provenance.copy()

    def render(
        self, params: ParameterValues
    ) -> Float32[np.ndarray, "1 192000"]:
        """Process the fixed source through one exact patch with fresh recursion state.

        Invalid patch types and values propagate ``TypeError`` or ``ValueError`` from the build
        boundary; invalid rendered audio propagates ``ValueError`` from :meth:`render_build`.

        :param params: Native order-8 mono pyFDN arrays.
        :returns: Contiguous finite channel-first float32 audio shaped ``(1, 192000)``; native
            amplitude is preserved without clipping or normalization.
        """
        return self.render_build(params_to_fdn_build(params, sample_rate=_SAMPLE_RATE))

    def render_build(
        self, build: FDNBuild
    ) -> Float32[np.ndarray, "1 192000"]:
        """Process the fixed source through one validated build with fresh recursion state.

        :param build: Exact order-8 mono build produced by :func:`params_to_fdn_build`.
        :returns: Contiguous finite channel-first float32 audio shaped ``(1, 192000)``; native
            amplitude is preserved without clipping or normalization.
        :raises ValueError: The rendered audio violates the fixed contract.
        """
        post_delay = cast(np.ndarray, build.post_delay)
        output = process_fdn(
            self._source_audio[0],
            build.delays,
            build.A,
            build.B,
            build.C,
            build.D,
            post_delay=SOSBank(post_delay),
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
