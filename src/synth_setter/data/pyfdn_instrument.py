"""Native pyFDN build conversion and instrument rendering.

Example:
    ``PyFDNRenderer().render(native_params)`` returns a channel-first impulse response.
"""

from collections.abc import Mapping
from importlib.metadata import version
from numbers import Integral, Real
from typing import cast

import numpy as np
from jaxtyping import Float32
from pyFDN import FDNBuild, build_set_decay, build_to_impz, decay_to_geq, process_fdn
from pyFDN.td import PitchShift, SOSBank, Series

from synth_setter.data.pyfdn_param_spec import (
    PYFDN_GEQ_RT_MAX_SECONDS,
    PYFDN_ORDER,
    PYFDN_PITCHSHIFT_ACTIVE_CHANNELS_NAME,
    PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_MAX,
    PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_MIN,
    PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_NAME,
    PYFDN_PITCHSHIFT_WINDOW_SIZE_MAX,
    PYFDN_PITCHSHIFT_WINDOW_SIZE_MIN,
    PYFDN_PITCHSHIFT_WINDOW_SIZE_NAME,
    PYFDN_RT_CROSSOVER_HZ,
    PYFDN_RT_DC_NAME,
    PYFDN_RT_GEQ_SECONDS_NAME,
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
from synth_setter.data.vst.param_spec import ParameterValue
from synth_setter.data.vst.renderers import AudioRenderer
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.renderer_backend import PyFDNExcitation

_PYFDN_VERSION = "0.4.2"
_SAMPLE_RATE = float(PYFDN_SOURCE_SAMPLE_RATE_HZ)
_CHANNELS = PYFDN_SOURCE_CHANNELS
_SIGNAL_LENGTH = PYFDN_SOURCE_TOTAL_FRAMES
_POST_DELAY_SOS_SHAPE = (1, 6, PYFDN_ORDER)
_PITCHSHIFT_GEQ_SOS_SHAPE = (11, 6, PYFDN_ORDER)
PYFDN_PITCHSHIFT_MIN_DELAY_SAMPLES = 3
PYFDN_PITCHSHIFT_MAX_DELAY_WINDOW_MULTIPLIER = 2
_PLAIN_PARAM_SPEC = ParamSpecName("pyfdn_n8_mono_householder")
_PITCHSHIFT_PARAM_SPEC = ParamSpecName("pyfdn_pitchshift_n8_mono_householder")
_ARRAY_CONTRACTS = (
    ("feedback_matrix", (PYFDN_ORDER, PYFDN_ORDER), np.dtype(np.float64)),
    ("input_matrix", (PYFDN_ORDER, _CHANNELS), np.dtype(np.float64)),
    ("output_matrix", (_CHANNELS, PYFDN_ORDER), np.dtype(np.float64)),
    ("direct_matrix", (_CHANNELS, _CHANNELS), np.dtype(np.float64)),
    ("delays", (PYFDN_ORDER,), np.dtype(np.int64)),
)
_BASE_KEYS = frozenset(name for name, _, _ in _ARRAY_CONTRACTS)
_REQUIRED_KEYS = _BASE_KEYS.union({PYFDN_RT_DC_NAME, PYFDN_RT_NYQUIST_NAME})
_PITCHSHIFT_REQUIRED_KEYS = _BASE_KEYS.union(
    {
        PYFDN_RT_GEQ_SECONDS_NAME,
        PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_NAME,
        PYFDN_PITCHSHIFT_WINDOW_SIZE_NAME,
        PYFDN_PITCHSHIFT_ACTIVE_CHANNELS_NAME,
    }
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


def _validate_base_params(
    params: Mapping[str, ParameterValue],
    *,
    required_keys: frozenset[str],
    sample_rate: float,
    topology: str,
) -> dict[str, np.ndarray]:
    """Validate and return the shared native FDN arrays.

    :param params: Native parameter mapping for one pyFDN topology.
    :param required_keys: Exact keys required by that topology.
    :param sample_rate: Processing rate in Hz; exactly ``44100.0``.
    :param topology: Name included in invalid-key diagnostics.
    :returns: Validated float64 A/B/C/D arrays and positive int64 delays.
    :raises ValueError: Keys, sample rate, shapes, values, or delays violate the contract.
    """
    if set(params) != required_keys:
        raise ValueError(f"{topology} params must contain exactly {sorted(required_keys)}")
    if sample_rate != _SAMPLE_RATE:
        raise ValueError("sample_rate must be exactly 44100.0")
    arrays = {
        name: _require_array(name, params[name], shape=shape, dtype=dtype)
        for name, shape, dtype in _ARRAY_CONTRACTS
    }
    if np.any(arrays["delays"] <= 0):
        raise ValueError("delays must be positive")
    return arrays


def params_to_fdn_build(
    params: Mapping[str, ParameterValue],
    *,
    sample_rate: float,
) -> FDNBuild:
    """Build an order-8 mono FDN from an exact native parameter mapping.

    :param params: Mapping containing ``feedback_matrix`` float64 ``(8, 8)``,
        ``input_matrix`` float64 ``(8, 1)``, ``output_matrix`` float64 ``(1, 8)``,
        ``direct_matrix`` float64 ``(1, 1)``, positive ``delays`` int64 ``(8,)``,
        and finite scalar DC and Nyquist reverberation times in seconds; every array
        must contain only finite values.
    :param sample_rate: Processing rate in Hz; exactly ``44100.0``.
    :returns: Native build with derived ``post_delay`` SOS and no other post hooks.
    """
    arrays = _validate_base_params(
        params,
        required_keys=_REQUIRED_KEYS,
        sample_rate=sample_rate,
        topology="plain",
    )
    rt_seconds = (
        _require_rt_seconds(PYFDN_RT_DC_NAME, params[PYFDN_RT_DC_NAME]),
        _require_rt_seconds(PYFDN_RT_NYQUIST_NAME, params[PYFDN_RT_NYQUIST_NAME]),
    )
    return _build_decay_fdn(arrays, rt_seconds)


def params_to_pitchshift_fdn_build(
    params: Mapping[str, ParameterValue],
    *,
    sample_rate: float,
) -> FDNBuild:
    """Build an order-8 FDN with the reference ten-band decay profile.

    :param params: Mapping containing ``feedback_matrix`` float64 ``(8, 8)``,
        ``input_matrix`` float64 ``(8, 1)``, ``output_matrix`` float64 ``(1, 8)``,
        ``direct_matrix`` float64 ``(1, 1)``, positive ``delays`` int64 ``(8,)``,
        ten GEQ reverberation times float64 ``(10,)`` bounded to 0.1–5.0 seconds,
        transpose bounded to -1200–1200 cents, window size bounded to 256–4096
        samples, and an active-channel int64 mask ``(8,)`` containing zero or one.
    :param sample_rate: Processing rate in Hz; exactly ``44100.0``.
    :returns: Native build carrying an eleven-section GEQ SOS bank.
    :raises TypeError: A native control has the wrong type or dtype.
    :raises ValueError: Keys, shapes, values, or sample rate violate the contract.
    """
    arrays = _validate_base_params(
        params,
        required_keys=_PITCHSHIFT_REQUIRED_KEYS,
        sample_rate=sample_rate,
        topology="pitch-shift",
    )
    _pitchshift_controls(params)
    rt_seconds = _require_array(
        PYFDN_RT_GEQ_SECONDS_NAME,
        params[PYFDN_RT_GEQ_SECONDS_NAME],
        shape=(10,),
        dtype=np.dtype(np.float64),
    )
    if np.any(
        (rt_seconds < PYFDN_RT_MIN_SECONDS)
        | (rt_seconds > PYFDN_GEQ_RT_MAX_SECONDS)
    ):
        raise ValueError("GEQ reverberation times must be between 0.1 and 5.0 seconds")
    post_delay = np.asarray(decay_to_geq(rt_seconds, arrays["delays"], sample_rate))
    build = FDNBuild(
        A=arrays["feedback_matrix"],
        B=arrays["input_matrix"],
        C=arrays["output_matrix"],
        D=arrays["direct_matrix"],
        delays=arrays["delays"],
        fs=sample_rate,
        post_delay=post_delay,
        post_matrix=None,
        post_output=None,
    )
    if post_delay.shape != _PITCHSHIFT_GEQ_SOS_SHAPE:
        raise ValueError(
            f"pitch-shift post_delay must have shape {_PITCHSHIFT_GEQ_SOS_SHAPE}, "
            f"got {post_delay.shape}"
        )
    if post_delay.dtype != np.float64:
        raise TypeError(f"pitch-shift post_delay must have dtype float64, got {post_delay.dtype}")
    if not np.isfinite(post_delay).all():
        raise ValueError("pitch-shift post_delay must contain only finite values")
    return build


def _pitchshift_controls(
    params: Mapping[str, ParameterValue],
) -> tuple[float, int, np.ndarray]:
    """Validate native pitch-shifter controls.

    :param params: Mapping containing transpose, window, and active-channel controls.
    :returns: Validated transpose cents, window samples, and int64 active mask ``(8,)``.
    :raises TypeError: Scalar or active-channel controls have invalid native types.
    :raises ValueError: Scalar bounds or active-channel values violate the ParamSpec.
    """
    transpose = params[PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_NAME]
    if not isinstance(transpose, Real) or isinstance(transpose, bool):
        raise TypeError("transpose_cents must be a real scalar")
    transpose_value = float(transpose)
    if not (
        PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_MIN
        <= transpose_value
        <= PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_MAX
    ):
        raise ValueError(
            "transpose_cents must be between "
            f"{PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_MIN} and "
            f"{PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_MAX}"
        )
    window_size = params[PYFDN_PITCHSHIFT_WINDOW_SIZE_NAME]
    if not isinstance(window_size, Integral) or isinstance(window_size, bool):
        raise TypeError("window_size must be an integer")
    window_value = int(window_size)
    if not PYFDN_PITCHSHIFT_WINDOW_SIZE_MIN <= window_value <= PYFDN_PITCHSHIFT_WINDOW_SIZE_MAX:
        raise ValueError(
            "window_size must be between "
            f"{PYFDN_PITCHSHIFT_WINDOW_SIZE_MIN} and "
            f"{PYFDN_PITCHSHIFT_WINDOW_SIZE_MAX}"
        )
    active_mask = _require_array(
        PYFDN_PITCHSHIFT_ACTIVE_CHANNELS_NAME,
        params[PYFDN_PITCHSHIFT_ACTIVE_CHANNELS_NAME],
        shape=(PYFDN_ORDER,),
        dtype=np.dtype(np.int64),
    )
    if np.any((active_mask != 0) & (active_mask != 1)):
        raise ValueError("active_channels must contain only zero or one")
    return transpose_value, window_value, active_mask


def _pitchshift_post_delay(
    build: FDNBuild,
    params: Mapping[str, ParameterValue],
) -> Series:
    """Construct fresh reference-ordered GEQ and pitch-shifter state.

    :param build: Pitch-shift FDN build carrying the GEQ SOS bank.
    :param params: Native pitch-shift controls.
    :returns: Stateful post-delay chain for one render only.
    """
    transpose, window_size, active_mask = _pitchshift_controls(params)
    geq = cast(np.ndarray, build.post_delay)
    return Series(
        [
            SOSBank(geq),
            PitchShift(
                PYFDN_ORDER,
                max_delay_samps=window_size * PYFDN_PITCHSHIFT_MAX_DELAY_WINDOW_MULTIPLIER,
                window_size=window_size,
                transpose_cents=transpose,
                fs=_SAMPLE_RATE,
                active_channels=np.flatnonzero(active_mask),
                min_delay_samps=PYFDN_PITCHSHIFT_MIN_DELAY_SAMPLES,
            ),
        ]
    )


def _process_source(
    build: FDNBuild,
    source: np.ndarray,
    post_delay: SOSBank | Series,
) -> np.ndarray:
    """Process one mono source through a configured native FDN.

    :param build: Native FDN matrices, delays, and optional post hooks.
    :param source: Mono source waveform shaped ``(176400,)``.
    :param post_delay: Fresh stateful delay-line processor for this render.
    :returns: Native pyFDN output as a NumPy array.
    """
    return np.asarray(
        process_fdn(
            source,
            build.delays,
            build.A,
            build.B,
            build.C,
            build.D,
            post_delay=post_delay,
            post_matrix=build.post_matrix,
            post_output=build.post_output,
        )
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


class PyFDNRenderer(AudioRenderer):
    """Render an FDN impulse response or an explicitly selected custom source."""

    def __init__(
        self,
        *,
        excitation: PyFDNExcitation = "impulse",
        param_spec_name: ParamSpecName = _PLAIN_PARAM_SPEC,
        synth_version: str = _PYFDN_VERSION,
        plugin_path: str = "pyfdn",
        sample_rate: float = _SAMPLE_RATE,
        channels: int = _CHANNELS,
        signal_duration_seconds: float = _SIGNAL_LENGTH / _SAMPLE_RATE,
        plugin_state_path: str | None = None,
    ) -> None:
        """Configure impulse-response rendering or the optional canonical chirp.

        :param excitation: ``"impulse"`` for the native IR or ``"chirp"`` for the custom source.
        :param param_spec_name: Registered plain or pitch-shift pyFDN topology.
        :param synth_version: Required installed pyFDN version.
        :param plugin_path: Required in-process backend sentinel.
        :param sample_rate: Required sample rate.
        :param channels: Required mono output channel count.
        :param signal_duration_seconds: Required render duration.
        :param plugin_state_path: Required empty preset path.
        :raises ValueError: The excitation, geometry, or artifact identity drifts.
        """
        _validate_version(synth_version)
        if excitation not in ("chirp", "impulse"):
            raise ValueError("pyFDN excitation must be 'impulse' or 'chirp'")
        if param_spec_name not in (_PLAIN_PARAM_SPEC, _PITCHSHIFT_PARAM_SPEC):
            raise ValueError(f"unsupported pyFDN param spec {param_spec_name!r}")
        if (
            plugin_path != "pyfdn"
            or sample_rate != _SAMPLE_RATE
            or channels != _CHANNELS
            or signal_duration_seconds != _SIGNAL_LENGTH / _SAMPLE_RATE
            or plugin_state_path not in (None, "")
        ):
            raise ValueError("pyFDN renderer requires its fixed render contract")
        super().__init__(
            plugin_path=plugin_path,
            sample_rate=sample_rate,
            channels=channels,
            signal_duration_seconds=signal_duration_seconds,
            plugin_state_path=plugin_state_path,
        )
        self._excitation = excitation
        self._param_spec_name = param_spec_name
        self._source_audio = (
            generate_canonical_pyfdn_source() if excitation == "chirp" else None
        )
        self._source_provenance = (
            _canonical_pyfdn_source_provenance(self._source_audio)
            if self._source_audio is not None
            else {
                "identity": "unit_impulse_v1",
                "implementation": (
                    "pyFDN.process_fdn"
                    if param_spec_name == _PITCHSHIFT_PARAM_SPEC
                    else "pyFDN.build_to_impz"
                ),
                "sample_rate_hz": PYFDN_SOURCE_SAMPLE_RATE_HZ,
                "total_frames": PYFDN_SOURCE_TOTAL_FRAMES,
                "channels": PYFDN_SOURCE_CHANNELS,
                "dtype": "float32",
                "layout": "channel_first",
            }
        )

    @property
    def source_provenance(self) -> PyFDNSourceProvenance:
        """Return provenance for the configured excitation.

        :returns: Independent provenance metadata safe for caller mutation.
        """
        return self._source_provenance.copy()

    def render(
        self,
        params: Mapping[str, ParameterValue],
        midi_note: int = 0,
        velocity: int = 0,
        note_start_and_end: tuple[float, float] = (0.0, 0.0),
        *,
        warmup: bool = False,
    ) -> Float32[np.ndarray, "1 176400"]:
        """Render the configured excitation through one patch with fresh recursion state.

        :param params: Native order-8 mono pyFDN arrays.
        :param midi_note: Ignored compatibility stub.
        :param velocity: Ignored compatibility stub.
        :param note_start_and_end: Ignored compatibility stub.
        :param warmup: Ignored compatibility stub.
        :returns: Contiguous finite channel-first float32 audio shaped ``(1, 176400)``; native
            amplitude is preserved without clipping or normalization.
        :raises ValueError: The patch or rendered audio violates the fixed contract.
        """
        del midi_note, velocity, note_start_and_end, warmup
        if self._param_spec_name == _PITCHSHIFT_PARAM_SPEC:
            build = params_to_pitchshift_fdn_build(params, sample_rate=_SAMPLE_RATE)
            if self._excitation == "impulse":
                source = np.zeros(_SIGNAL_LENGTH, dtype=np.float32)
                source[0] = 1.0
            else:
                source = cast(np.ndarray, self._source_audio)[0]
            output_array = _process_source(
                build,
                source,
                _pitchshift_post_delay(build, params),
            )
        else:
            build = params_to_fdn_build(params, sample_rate=_SAMPLE_RATE)
            if self._excitation == "impulse":
                impulse_response = np.asarray(build_to_impz(build, ir_len=_SIGNAL_LENGTH))
                expected_shape = (_SIGNAL_LENGTH, _CHANNELS, _CHANNELS)
                if impulse_response.shape != expected_shape:
                    raise ValueError(
                        f"pyFDN impulse response must have shape {expected_shape}, "
                        f"got {impulse_response.shape}"
                    )
                output_array = impulse_response[:, 0, 0]
            else:
                post_delay = cast(np.ndarray, build.post_delay)
                source = cast(np.ndarray, self._source_audio)[0]
                output_array = _process_source(build, source, SOSBank(post_delay))
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
