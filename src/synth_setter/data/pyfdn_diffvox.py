"""DiffVox vocal effects chain (arXiv:2504.14735) rendered with pyFDN block processing.

Example:
    ``render_diffvox_chain(params, impulse, sample_rate=44_100.0)`` returns the
    stereo ``(176400, 2)`` response of one sampled chain.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import numpy as np
from pyFDN import decay_to_geq, process_fdn
from pyFDN.td import SOSBank
from scipy.linalg import expm

from synth_setter.data.pyfdn_param_spec import (
    PYFDN_DIFFVOX_DELAY_EVEN_PAN_NAME,
    PYFDN_DIFFVOX_DELAY_FEEDBACK_NAME,
    PYFDN_DIFFVOX_DELAY_GAIN_NAME,
    PYFDN_DIFFVOX_DELAY_LP_BAND,
    PYFDN_DIFFVOX_DELAY_ODD_PAN_NAME,
    PYFDN_DIFFVOX_DELAY_TIME_NAME,
    PYFDN_DIFFVOX_DIRECT_PAN_NAME,
    PYFDN_DIFFVOX_ORDER,
    PYFDN_DIFFVOX_PARAM_SPEC,
    PYFDN_DIFFVOX_PEQ_BANDS,
    PYFDN_DIFFVOX_REVERB_DELAYS,
    PYFDN_DIFFVOX_REVERB_INPUT_NAME,
    PYFDN_DIFFVOX_REVERB_OUTPUT_NAME,
    PYFDN_DIFFVOX_REVERB_RT_NAME,
    PYFDN_DIFFVOX_REVERB_SKEW_NAME,
    PYFDN_DIFFVOX_SEND_NAME,
    PYFDN_DIFFVOX_TONE_BANDS,
    EqBand,
    EqBandKind,
    require_array,
)
from synth_setter.data.vst.param_spec import (
    ContinuousArrayParameter,
    ContinuousParameter,
    ParameterValue,
)

# Equal-power panning law with a sqrt(2) makeup so a centre pan passes unity gain.
_PAN_NORM = float(np.sqrt(2.0))
_DIFFVOX_REQUIRED_KEYS = frozenset(PYFDN_DIFFVOX_PARAM_SPEC.synth_param_names)


@dataclass(frozen=True)
class _Controls:
    """Validated native DiffVox patch split by value kind.

    .. attribute :: scalars

       Real controls keyed by ParamSpec name.

    .. attribute :: arrays

       Float64 array controls keyed by ParamSpec name.
    """

    scalars: Mapping[str, float]
    arrays: Mapping[str, np.ndarray]


def _validated_array(parameter: ContinuousArrayParameter, value: ParameterValue) -> np.ndarray:
    """Check one native array against its ParamSpec shape, dtype, and bounds.

    :param parameter: Array control definition.
    :param value: Native value supplied for it.
    :returns: The same array, unchanged.
    :raises ValueError: The bounds violate the ParamSpec.
    """
    value = require_array(
        parameter.name, value, shape=parameter.shape, dtype=np.dtype(np.float64)
    )
    if np.any((value < parameter.min) | (value > parameter.max)):
        raise ValueError(f"{parameter.name} must be within [{parameter.min}, {parameter.max}]")
    return value


def _validated_scalar(parameter: ContinuousParameter, value: ParameterValue) -> float:
    """Check one native scalar against its ParamSpec bounds.

    :param parameter: Scalar control definition.
    :param value: Native value supplied for it.
    :returns: The value as a Python float.
    :raises TypeError: The value is not a real scalar.
    :raises ValueError: The value is non-finite or outside the ParamSpec bounds.
    """
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{parameter.name} must be a real scalar")
    scalar = float(value)
    if not np.isfinite(scalar) or not parameter.min <= scalar <= parameter.max:
        raise ValueError(f"{parameter.name} must be within [{parameter.min}, {parameter.max}]")
    return scalar


def _validate_params(params: Mapping[str, ParameterValue]) -> _Controls:
    """Validate a native DiffVox patch against its ParamSpec.

    :param params: Native chain controls keyed by ParamSpec name.
    :returns: Validated scalars and arrays.
    :raises TypeError: The ParamSpec holds a control kind the chain cannot validate.
    :raises ValueError: The key set differs from the ParamSpec.
    """
    if set(params) != _DIFFVOX_REQUIRED_KEYS:
        raise ValueError(
            f"diffvox params must contain exactly {sorted(_DIFFVOX_REQUIRED_KEYS)}"
        )
    scalars: dict[str, float] = {}
    arrays: dict[str, np.ndarray] = {}
    for parameter in PYFDN_DIFFVOX_PARAM_SPEC.synth_params:
        if isinstance(parameter, ContinuousArrayParameter):
            arrays[parameter.name] = _validated_array(parameter, params[parameter.name])
        elif isinstance(parameter, ContinuousParameter):
            scalars[parameter.name] = _validated_scalar(parameter, params[parameter.name])
        else:  # pragma: no cover - the spec only holds the two kinds above
            raise TypeError(f"unsupported DiffVox parameter {parameter!r}")
    return _Controls(scalars=scalars, arrays=arrays)


def _biquad_section(
    kind: EqBandKind,
    *,
    sample_rate: float,
    freq_hz: float,
    gain_db: float,
    q: float,
) -> np.ndarray:
    """Design one Audio EQ Cookbook section as a normalised SOS row.

    :param kind: Cookbook section type.
    :param sample_rate: Processing rate in Hz.
    :param freq_hz: Centre or cutoff frequency in Hz.
    :param gain_db: Peak or shelf gain in dB; ignored by pass filters.
    :param q: Quality factor; shelves pass the cookbook ``S = 1`` slope of 0.707.
    :returns: ``[b0, b1, b2, 1, a1, a2]`` float64 coefficients.
    :raises ValueError: ``kind`` is not a cookbook section.
    """
    omega = 2.0 * np.pi * freq_hz / sample_rate
    cos_w, sin_w = np.cos(omega), np.sin(omega)
    alpha = sin_w / (2.0 * q)
    amp = 10.0 ** (gain_db / 40.0)
    match kind:
        case "peak":
            b = [1.0 + alpha * amp, -2.0 * cos_w, 1.0 - alpha * amp]
            a = [1.0 + alpha / amp, -2.0 * cos_w, 1.0 - alpha / amp]
        case "lowshelf":
            root = 2.0 * np.sqrt(amp) * alpha
            b = [
                amp * ((amp + 1.0) - (amp - 1.0) * cos_w + root),
                2.0 * amp * ((amp - 1.0) - (amp + 1.0) * cos_w),
                amp * ((amp + 1.0) - (amp - 1.0) * cos_w - root),
            ]
            a = [
                (amp + 1.0) + (amp - 1.0) * cos_w + root,
                -2.0 * ((amp - 1.0) + (amp + 1.0) * cos_w),
                (amp + 1.0) + (amp - 1.0) * cos_w - root,
            ]
        case "highshelf":
            root = 2.0 * np.sqrt(amp) * alpha
            b = [
                amp * ((amp + 1.0) + (amp - 1.0) * cos_w + root),
                -2.0 * amp * ((amp - 1.0) + (amp + 1.0) * cos_w),
                amp * ((amp + 1.0) + (amp - 1.0) * cos_w - root),
            ]
            a = [
                (amp + 1.0) - (amp - 1.0) * cos_w + root,
                2.0 * ((amp - 1.0) - (amp + 1.0) * cos_w),
                (amp + 1.0) - (amp - 1.0) * cos_w - root,
            ]
        case "lowpass":
            b = [(1.0 - cos_w) / 2.0, 1.0 - cos_w, (1.0 - cos_w) / 2.0]
            a = [1.0 + alpha, -2.0 * cos_w, 1.0 - alpha]
        case "highpass":
            b = [(1.0 + cos_w) / 2.0, -(1.0 + cos_w), (1.0 + cos_w) / 2.0]
            a = [1.0 + alpha, -2.0 * cos_w, 1.0 - alpha]
        case _:
            raise ValueError(f"unknown biquad kind {kind!r}")
    return np.asarray([*b, *a], dtype=np.float64) / a[0]


def _band_sections(
    bands: tuple[EqBand, ...],
    scalars: Mapping[str, float],
    *,
    sample_rate: float,
    channels: int,
) -> np.ndarray:
    """Design one cascade per channel from the learned band controls.

    :param bands: Bands in cascade order.
    :param scalars: Validated scalar controls.
    :param sample_rate: Processing rate in Hz.
    :param channels: Channel count that shares the same cascade.
    :returns: Bank shaped ``(len(bands), 6, channels)`` for :class:`SOSBank`.
    """
    sections = np.stack(
        [
            _biquad_section(
                band.kind,
                sample_rate=sample_rate,
                freq_hz=scalars[band.freq_name],
                gain_db=band.gain_db(scalars),
                q=band.q(scalars),
            )
            for band in bands
        ]
    )
    return np.repeat(sections[:, :, None], channels, axis=2)


def _pan_column(pan: float) -> np.ndarray:
    """Map a unit pan (0 = left, 1 = right) to left/right gains.

    :param pan: Pan position in ``[0, 1]``.
    :returns: ``(2,)`` gains following the sqrt(2)-normalised equal-power law.
    """
    angle = pan * np.pi / 2.0
    return _PAN_NORM * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)


def diffvox_feedback_matrix(skew: np.ndarray) -> np.ndarray:
    """Build the unilossless feedback matrix ``exp(T - T^T)`` from its upper triangle.

    :param skew: Strict upper-triangle entries in row-major order, shaped ``(15,)``.
    :returns: Orthogonal float64 matrix shaped ``(6, 6)``.
    """
    upper = np.zeros((PYFDN_DIFFVOX_ORDER, PYFDN_DIFFVOX_ORDER), dtype=np.float64)
    upper[np.triu_indices(PYFDN_DIFFVOX_ORDER, k=1)] = np.asarray(skew, dtype=np.float64)
    return np.asarray(expm(upper - upper.T), dtype=np.float64)


def _ping_pong_delay(mono: np.ndarray, controls: _Controls, *, sample_rate: float) -> np.ndarray:
    """Run the cross-fed two-line delay whose taps alternate between two panners.

    Both lines share one low-pass inside the loop, so the k-th echo is filtered k
    times and scaled by ``feedback ** (k - 1)`` exactly as in the reference model.

    :param mono: Equalised source shaped ``(frames, 1)``.
    :param controls: Validated native controls.
    :param sample_rate: Processing rate in Hz.
    :returns: Stereo delay output shaped ``(frames, 2)`` including the wet gain.
    """
    scalars = controls.scalars
    delay_samples = int(round(scalars[PYFDN_DIFFVOX_DELAY_TIME_NAME] * sample_rate))
    feedback = scalars[PYFDN_DIFFVOX_DELAY_FEEDBACK_NAME]
    panners = np.stack(
        [
            _pan_column(scalars[PYFDN_DIFFVOX_DELAY_ODD_PAN_NAME]),
            _pan_column(scalars[PYFDN_DIFFVOX_DELAY_EVEN_PAN_NAME]),
        ],
        axis=1,
    )
    output = process_fdn(
        mono,
        np.array([delay_samples, delay_samples], dtype=np.int64),
        A=np.array([[0.0, feedback], [feedback, 0.0]]),
        B=np.array([[1.0], [0.0]]),
        C=scalars[PYFDN_DIFFVOX_DELAY_GAIN_NAME] * panners,
        D=np.zeros((2, 1)),
        post_delay=SOSBank(
            _band_sections(
                (PYFDN_DIFFVOX_DELAY_LP_BAND,), scalars, sample_rate=sample_rate, channels=2
            )
        ),
    )
    return np.asarray(output, dtype=np.float64)


def _fdn_reverb(stereo_in: np.ndarray, controls: _Controls, *, sample_rate: float) -> np.ndarray:
    """Run the six-line FDN with delay-proportional GEQ decay and a tone PEQ.

    :param stereo_in: Reverb input shaped ``(frames, 2)``.
    :param controls: Validated native controls.
    :param sample_rate: Processing rate in Hz.
    :returns: Stereo reverb output shaped ``(frames, 2)``.
    """
    arrays = controls.arrays
    delays = np.asarray(PYFDN_DIFFVOX_REVERB_DELAYS, dtype=np.int64)
    output = process_fdn(
        stereo_in,
        delays,
        A=diffvox_feedback_matrix(arrays[PYFDN_DIFFVOX_REVERB_SKEW_NAME]),
        B=arrays[PYFDN_DIFFVOX_REVERB_INPUT_NAME],
        C=arrays[PYFDN_DIFFVOX_REVERB_OUTPUT_NAME],
        D=np.zeros((2, 2)),
        post_delay=SOSBank(decay_to_geq(arrays[PYFDN_DIFFVOX_REVERB_RT_NAME], delays, sample_rate)),
        post_output=SOSBank(
            _band_sections(
                PYFDN_DIFFVOX_TONE_BANDS, controls.scalars, sample_rate=sample_rate, channels=2
            )
        ),
    )
    return np.asarray(output, dtype=np.float64)


def render_diffvox_chain(
    params: Mapping[str, ParameterValue],
    source: np.ndarray,
    *,
    sample_rate: float,
) -> np.ndarray:
    """Process a mono source through PEQ, direct panner, delay send, and reverb send.

    The compressor/expander stage of the reference chain is omitted; every other
    stage is linear and time-invariant, so an impulse source yields the chain IR.

    :param params: Native DiffVox controls keyed by ParamSpec name.
    :param source: Mono source waveform shaped ``(frames,)``.
    :param sample_rate: Processing rate in Hz.
    :returns: Float64 stereo output shaped ``(frames, 2)``.
    """
    controls = _validate_params(params)
    equalised = SOSBank(
        _band_sections(
            PYFDN_DIFFVOX_PEQ_BANDS, controls.scalars, sample_rate=sample_rate, channels=1
        )
    ).process(np.asarray(source, dtype=np.float64))
    direct = equalised * _pan_column(controls.scalars[PYFDN_DIFFVOX_DIRECT_PAN_NAME])
    delayed = _ping_pong_delay(equalised, controls, sample_rate=sample_rate)
    reverb_in = equalised + controls.scalars[PYFDN_DIFFVOX_SEND_NAME] * delayed
    reverberated = _fdn_reverb(reverb_in, controls, sample_rate=sample_rate)
    return direct + delayed + reverberated
