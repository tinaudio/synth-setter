"""Parameter distributions for pyFDN instruments.

Example:
    ``PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(rng)`` draws one native patch.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pyFDN import householder_matrix

from synth_setter.data.vst.param_spec import (
    AngleArrayParameter,
    ContinuousArrayParameter,
    ContinuousParameter,
    DirectionArrayParameter,
    DiscreteArrayParameter,
    DiscreteLiteralParameter,
    Parameter,
    ParameterValue,
    ParameterValues,
    ParamSpec,
)


PYFDN_ORDER = 8
PYFDN_KRONECKER_LEVELS = PYFDN_ORDER.bit_length() - 1
PYFDN_KRONECKER_ANGLES_NAME = "kronecker_angles"
PYFDN_KRONECKER_REFLECT_NAME = "kronecker_reflect"
PYFDN_HOUSEHOLDER_VECTOR_NAME = "householder_vector"
# Below this norm the reflection direction is numerically meaningless.
PYFDN_HOUSEHOLDER_MIN_NORM = 1e-6
PYFDN_RT_CROSSOVER_HZ = 6_000.0
PYFDN_RT_DC_NAME = "post_delay.rt_dc_seconds"
PYFDN_RT_MAX_SECONDS = 4.0
PYFDN_RT_MIN_SECONDS = 0.1
PYFDN_RT_NYQUIST_NAME = "post_delay.rt_nyquist_seconds"
PYFDN_RT_GEQ_SECONDS_NAME = "post_delay.geq.rt_seconds"
PYFDN_GEQ_RT_MAX_SECONDS = 5.0
PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_NAME = "post_delay.pitch_shift.transpose_cents"
PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_MIN = -1200.0
PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_MAX = 1200.0
PYFDN_PITCHSHIFT_WINDOW_SIZE_NAME = "post_delay.pitch_shift.window_size"
PYFDN_PITCHSHIFT_WINDOW_SIZE_MIN = 256
PYFDN_PITCHSHIFT_WINDOW_SIZE_MAX = 4096
PYFDN_PITCHSHIFT_ACTIVE_CHANNELS_NAME = "post_delay.pitch_shift.active_channels"

# DiffVox vocal chain (arXiv:2504.14735): PEQ -> direct panner + ping-pong delay send
# -> six-line FDN send with a tone PEQ. Bounds follow the reference ``fx_config.yaml``.
PYFDN_DIFFVOX_ORDER = 6
PYFDN_DIFFVOX_REVERB_DELAYS = (997, 1153, 1327, 1559, 1801, 2099)
PYFDN_DIFFVOX_EQ_GAIN_DB_MAX = 20.0
PYFDN_FIXED_SHELF_Q = 0.707
PYFDN_DIFFVOX_DIRECT_PAN_NAME = "direct.pan"
PYFDN_DIFFVOX_DELAY_TIME_NAME = "delay.time_seconds"
PYFDN_DIFFVOX_DELAY_TIME_MIN_SECONDS = 0.1
PYFDN_DIFFVOX_DELAY_TIME_MAX_SECONDS = 1.0
PYFDN_DIFFVOX_DELAY_FEEDBACK_NAME = "delay.feedback"
PYFDN_DIFFVOX_DELAY_FEEDBACK_MAX = 0.95
PYFDN_DIFFVOX_DELAY_GAIN_NAME = "delay.gain"
PYFDN_DIFFVOX_DELAY_LP_FREQ_NAME = "delay.lp.freq_hz"
PYFDN_DIFFVOX_DELAY_ODD_PAN_NAME = "delay.odd_pan"
PYFDN_DIFFVOX_DELAY_EVEN_PAN_NAME = "delay.even_pan"
PYFDN_DIFFVOX_SEND_NAME = "send.delay_to_reverb"
PYFDN_DIFFVOX_REVERB_INPUT_NAME = "reverb.input_matrix"
PYFDN_DIFFVOX_REVERB_OUTPUT_NAME = "reverb.output_matrix"
PYFDN_DIFFVOX_REVERB_SKEW_NAME = "reverb.feedback_skew"
PYFDN_DIFFVOX_REVERB_SKEW_SIZE = PYFDN_DIFFVOX_ORDER * (PYFDN_DIFFVOX_ORDER - 1) // 2
PYFDN_DIFFVOX_REVERB_RT_NAME = "reverb.geq.rt_seconds"

type EqBandKind = Literal["peak", "lowshelf", "highshelf", "lowpass", "highpass"]


def require_array(
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


@dataclass(frozen=True)
class EqBand:
    """One Audio-EQ-Cookbook biquad band and the learned controls it exposes.

    .. attribute :: prefix

       Control-name prefix, e.g. ``peq.pk1``.

    .. attribute :: kind

       Cookbook section type.

    .. attribute :: freq_min_hz

       Inclusive lower bound of the frequency control in Hz.

    .. attribute :: freq_max_hz

       Inclusive upper bound of the frequency control in Hz.

    .. attribute :: q_range

       Inclusive Q bounds, or ``None`` for a fixed 0.707 shelf slope.
    """

    prefix: str
    kind: EqBandKind
    freq_min_hz: float
    freq_max_hz: float
    q_range: tuple[float, float] | None

    @property
    def freq_name(self) -> str:
        """Return the cutoff / centre frequency control name in Hz."""
        return f"{self.prefix}.freq_hz"

    @property
    def gain_name(self) -> str:
        """Return the gain control name in dB; shelves and peaks only."""
        return f"{self.prefix}.gain_db"

    @property
    def q_name(self) -> str:
        """Return the Q control name; fixed at 0.707 when ``q_range`` is ``None``."""
        return f"{self.prefix}.q"

    @property
    def has_gain(self) -> bool:
        """Return whether the band kind carries a dB gain."""
        return self.kind in ("peak", "lowshelf", "highshelf")

    def gain_db(self, scalars: Mapping[str, float]) -> float:
        """Read the band gain from validated controls.

        :param scalars: Validated scalar controls keyed by name.
        :returns: Gain in dB; ``0.0`` for pass filters, which carry none.
        """
        return scalars[self.gain_name] if self.has_gain else 0.0

    def q(self, scalars: Mapping[str, float]) -> float:
        """Read the band Q from validated controls.

        :param scalars: Validated scalar controls keyed by name.
        :returns: Learned Q, or the fixed 0.707 shelf slope when the band exposes none.
        """
        return scalars[self.q_name] if self.q_range is not None else PYFDN_FIXED_SHELF_Q

    def parameters(self) -> list[Parameter]:
        """Build fresh learned controls for this band.

        :returns: Frequency, then gain and Q when the band exposes them.
        """
        controls: list[Parameter] = [
            ContinuousParameter(
                name=self.freq_name, min=self.freq_min_hz, max=self.freq_max_hz
            )
        ]
        if self.has_gain:
            controls.append(
                ContinuousParameter(
                    name=self.gain_name,
                    min=-PYFDN_DIFFVOX_EQ_GAIN_DB_MAX,
                    max=PYFDN_DIFFVOX_EQ_GAIN_DB_MAX,
                )
            )
        if self.q_range is not None:
            controls.append(
                ContinuousParameter(name=self.q_name, min=self.q_range[0], max=self.q_range[1])
            )
        return controls


PYFDN_DIFFVOX_PEQ_BANDS = (
    EqBand("peq.pk1", "peak", 33.0, 5_400.0, (0.2, 20.0)),
    EqBand("peq.pk2", "peak", 200.0, 17_500.0, (0.2, 20.0)),
    EqBand("peq.ls", "lowshelf", 30.0, 200.0, None),
    EqBand("peq.hs", "highshelf", 750.0, 8_300.0, None),
    EqBand("peq.lp", "lowpass", 200.0, 18_000.0, (0.5, 10.0)),
    EqBand("peq.hp", "highpass", 16.0, 5_300.0, (0.5, 10.0)),
)
PYFDN_DIFFVOX_DELAY_LP_BAND = EqBand("delay.lp", "lowpass", 200.0, 16_000.0, (0.5, 2.0))
PYFDN_DIFFVOX_TONE_BANDS = (
    EqBand("reverb.eq.pk1", "peak", 200.0, 2_500.0, (0.1, 3.0)),
    EqBand("reverb.eq.pk2", "peak", 600.0, 7_000.0, (0.1, 3.0)),
    EqBand("reverb.eq.ls", "lowshelf", 30.0, 450.0, None),
    EqBand("reverb.eq.hs", "highshelf", 1_500.0, 16_000.0, None),
)


_PYFDN_MIDI_STUBS: ParameterValues = {
    "pitch": 0,
    "note_start_and_end": (0.0, 0.0),
}


def kronecker_feedback_matrix(angles: np.ndarray, reflect: np.ndarray) -> np.ndarray:
    """Build the Kronecker feedback matrix of Coppola (DAFx26) from per-level kernels.

    ``Psi = K_M kron ... kron K_1``: level 0 is the innermost kernel mixing adjacent
    (even/odd) lines, the last level mixes the two contiguous halves. Every kernel is
    orthogonal, so the product is orthogonal for any angles.

    :param angles: One kernel angle in radians per level, shaped ``(PYFDN_KRONECKER_LEVELS,)``.
    :param reflect: Per-level flags shaped like ``angles``; 1 selects ``Ref(theta)``
        over ``Rot(theta)``.
    :returns: Float64 orthogonal matrix shaped ``(PYFDN_ORDER, PYFDN_ORDER)``.
    :raises ValueError: Either control array does not hold exactly one entry per level.
    """
    expected_shape = (PYFDN_KRONECKER_LEVELS,)
    if angles.shape != expected_shape or reflect.shape != expected_shape:
        raise ValueError(
            f"kronecker controls must be shaped {expected_shape}, "
            f"got angles {angles.shape} and reflect {reflect.shape}"
        )
    feedback = np.eye(1, dtype=np.float64)
    for theta, flag in zip(angles, reflect, strict=True):
        cos, sin = np.cos(theta), np.sin(theta)
        kernel = (
            np.array([[cos, sin], [sin, -cos]]) if flag else np.array([[cos, -sin], [sin, cos]])
        )
        feedback = np.kron(kernel, feedback)
    return feedback


def householder_feedback_matrix(vector: np.ndarray) -> np.ndarray:
    """Build the Householder reflection ``I - 2uu^T/(u^T u)`` for one delay-line vector.

    Only the direction matters: ``u``, ``-u``, and any rescaling give the same matrix.

    :param vector: Reflection direction shaped ``(PYFDN_ORDER,)``; need not be unit length.
    :returns: Float64 orthogonal symmetric matrix shaped ``(PYFDN_ORDER, PYFDN_ORDER)``.
    :raises ValueError: The vector has the wrong length or a vanishing norm.
    """
    if vector.shape != (PYFDN_ORDER,):
        raise ValueError(f"householder vector must be shaped ({PYFDN_ORDER},), got {vector.shape}")
    if np.linalg.norm(vector) < PYFDN_HOUSEHOLDER_MIN_NORM:
        raise ValueError(f"householder vector norm must be at least {PYFDN_HOUSEHOLDER_MIN_NORM}")
    return householder_matrix(np.asarray(vector, dtype=np.float64))


class PyFDNParamSpec(ParamSpec):
    """Derive the feedback matrix and MIDI stubs outside the learned coordinates."""

    def __init__(
        self,
        synth_params: list[Parameter],
        feedback_matrix: Callable[[ParameterValues], np.ndarray] | None,
    ) -> None:
        """Bind learned parameters to an optional renderer-native feedback-matrix rule.

        :param synth_params: Parameters represented in each encoded row.
        :param feedback_matrix: Maps decoded fields to a fresh feedback matrix, or
            ``None`` when the topology represents feedback through other controls.
        """
        super().__init__(synth_params=synth_params, note_params=[])
        self._feedback_matrix = feedback_matrix

    def _derive_feedback(self, synth_params: ParameterValues) -> ParameterValues:
        """Add the derived feedback matrix when the topology requires one.

        :param synth_params: Learned native values to extend in place.
        :returns: The same mapping.
        """
        if self._feedback_matrix is not None:
            synth_params["feedback_matrix"] = self._feedback_matrix(synth_params)
        return synth_params

    def sample(
        self, rng: np.random.Generator | None = None
    ) -> tuple[ParameterValues, ParameterValues]:
        """Sample learned fields and derive the remaining renderer values.

        :param rng: Optional caller-owned random generator.
        :returns: Complete native FDN values and fixed MIDI compatibility values.
        """
        synth_params, _ = super().sample(rng)
        return self._derive_feedback(synth_params), _PYFDN_MIDI_STUBS.copy()

    def decode(self, params: np.ndarray) -> tuple[ParameterValues, ParameterValues]:
        """Decode learned fields and derive the remaining renderer values.

        :param params: Encoded learned FDN parameter row shaped ``(self.encoded_width,)``.
        :returns: Complete native FDN values and fixed MIDI compatibility values.
        """
        synth_params, _ = super().decode(params)
        return self._derive_feedback(synth_params), _PYFDN_MIDI_STUBS.copy()


def _kronecker_feedback_from_params(synth_params: ParameterValues) -> np.ndarray:
    """Read the learned kernel controls off one decoded patch.

    :param synth_params: Decoded fields carrying the angle and reflect arrays.
    :returns: The orthogonal feedback matrix those controls describe.
    """
    return kronecker_feedback_matrix(
        np.asarray(synth_params[PYFDN_KRONECKER_ANGLES_NAME]),
        np.asarray(synth_params[PYFDN_KRONECKER_REFLECT_NAME]),
    )


def _householder_vector_feedback_from_params(synth_params: ParameterValues) -> np.ndarray:
    """Read the learned reflection vector off one decoded patch.

    :param synth_params: Decoded fields carrying the reflection vector.
    :returns: The Householder feedback matrix that vector describes.
    """
    return householder_feedback_matrix(np.asarray(synth_params[PYFDN_HOUSEHOLDER_VECTOR_NAME]))


def _fdn_matrix_parameters(*, delay_min: int, delay_max: int) -> list[Parameter]:
    """Build fresh common FDN parameters in renderer encoding order.

    :param delay_min: Inclusive delay-line lower bound in samples.
    :param delay_max: Inclusive delay-line upper bound in samples.
    :returns: Delay and A/B/C/D parameter definitions excluding fixed feedback.
    """
    return [
        DiscreteArrayParameter(
            name="delays", shape=(PYFDN_ORDER,), min=delay_min, max=delay_max
        ),
        ContinuousArrayParameter(
            name="input_matrix", shape=(PYFDN_ORDER, 1), min=-1.0, max=1.0
        ),
        ContinuousArrayParameter(
            name="output_matrix", shape=(1, PYFDN_ORDER), min=-1.0, max=1.0
        ),
        ContinuousArrayParameter(
            name="direct_matrix", shape=(1, 1), min=-1.0, max=1.0
        ),
    ]


_PYFDN_N8_HOUSEHOLDER_FEEDBACK = householder_feedback_matrix(np.ones(PYFDN_ORDER))


def _householder_feedback(synth_params: ParameterValues) -> np.ndarray:
    """Return a fresh copy of the fixed all-ones Householder reflection.

    :param synth_params: Decoded fields, unused because the matrix is constant.
    :returns: The order-8 Householder feedback matrix.
    """
    del synth_params
    return _PYFDN_N8_HOUSEHOLDER_FEEDBACK.copy()


PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC = PyFDNParamSpec(
    synth_params=[
        *_fdn_matrix_parameters(delay_min=400, delay_max=1200),
        ContinuousParameter(
            name=PYFDN_RT_DC_NAME,
            min=PYFDN_RT_MIN_SECONDS,
            max=PYFDN_RT_MAX_SECONDS,
        ),
        ContinuousParameter(
            name=PYFDN_RT_NYQUIST_NAME,
            min=PYFDN_RT_MIN_SECONDS,
            max=PYFDN_RT_MAX_SECONDS,
        ),
    ],
    feedback_matrix=_householder_feedback,
)

PYFDN_N8_MONO_KRONECKER_PARAM_SPEC = PyFDNParamSpec(
    synth_params=[
        *_fdn_matrix_parameters(delay_min=400, delay_max=1200),
        ContinuousParameter(
            name=PYFDN_RT_DC_NAME,
            min=PYFDN_RT_MIN_SECONDS,
            max=PYFDN_RT_MAX_SECONDS,
        ),
        ContinuousParameter(
            name=PYFDN_RT_NYQUIST_NAME,
            min=PYFDN_RT_MIN_SECONDS,
            max=PYFDN_RT_MAX_SECONDS,
        ),
        AngleArrayParameter(
            name=PYFDN_KRONECKER_ANGLES_NAME,
            shape=(PYFDN_KRONECKER_LEVELS,),
        ),
        DiscreteArrayParameter(
            name=PYFDN_KRONECKER_REFLECT_NAME,
            shape=(PYFDN_KRONECKER_LEVELS,),
            min=0,
            max=1,
        ),
    ],
    feedback_matrix=_kronecker_feedback_from_params,
)

PYFDN_N8_MONO_HOUSEHOLDER_VECTOR_PARAM_SPEC = PyFDNParamSpec(
    synth_params=[
        *_fdn_matrix_parameters(delay_min=400, delay_max=1200),
        ContinuousParameter(
            name=PYFDN_RT_DC_NAME,
            min=PYFDN_RT_MIN_SECONDS,
            max=PYFDN_RT_MAX_SECONDS,
        ),
        ContinuousParameter(
            name=PYFDN_RT_NYQUIST_NAME,
            min=PYFDN_RT_MIN_SECONDS,
            max=PYFDN_RT_MAX_SECONDS,
        ),
        DirectionArrayParameter(name=PYFDN_HOUSEHOLDER_VECTOR_NAME, shape=(PYFDN_ORDER,)),
    ],
    feedback_matrix=_householder_vector_feedback_from_params,
)

PYFDN_PITCHSHIFT_N8_MONO_HOUSEHOLDER_PARAM_SPEC = PyFDNParamSpec(
    synth_params=[
        *_fdn_matrix_parameters(delay_min=1000, delay_max=6000),
        ContinuousArrayParameter(
            name=PYFDN_RT_GEQ_SECONDS_NAME,
            shape=(10,),
            min=PYFDN_RT_MIN_SECONDS,
            max=PYFDN_GEQ_RT_MAX_SECONDS,
        ),
        ContinuousParameter(
            name=PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_NAME,
            min=PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_MIN,
            max=PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_MAX,
        ),
        DiscreteLiteralParameter(
            name=PYFDN_PITCHSHIFT_WINDOW_SIZE_NAME,
            min=PYFDN_PITCHSHIFT_WINDOW_SIZE_MIN,
            max=PYFDN_PITCHSHIFT_WINDOW_SIZE_MAX,
        ),
        DiscreteArrayParameter(
            name=PYFDN_PITCHSHIFT_ACTIVE_CHANNELS_NAME,
            shape=(PYFDN_ORDER,),
            min=0,
            max=1,
        ),
    ],
    feedback_matrix=_householder_feedback,
)


def _band_controls(bands: tuple[EqBand, ...]) -> list[Parameter]:
    """Expand EQ bands into their learned controls in cascade order.

    :param bands: Bands in cascade order.
    :returns: Frequency, gain, and Q controls band by band.
    """
    controls: list[Parameter] = []
    for band in bands:
        controls.extend(band.parameters())
    return controls


def _diffvox_parameters() -> list[Parameter]:
    """Build the DiffVox chain controls in renderer encoding order.

    :returns: PEQ, direct panner, ping-pong delay, send, and FDN reverb controls.
    """
    unit = {"min": 0.0, "max": 1.0}
    return [
        *_band_controls(PYFDN_DIFFVOX_PEQ_BANDS),
        ContinuousParameter(name=PYFDN_DIFFVOX_DIRECT_PAN_NAME, **unit),
        ContinuousParameter(
            name=PYFDN_DIFFVOX_DELAY_TIME_NAME,
            min=PYFDN_DIFFVOX_DELAY_TIME_MIN_SECONDS,
            max=PYFDN_DIFFVOX_DELAY_TIME_MAX_SECONDS,
        ),
        ContinuousParameter(
            name=PYFDN_DIFFVOX_DELAY_FEEDBACK_NAME, min=0.0, max=PYFDN_DIFFVOX_DELAY_FEEDBACK_MAX
        ),
        ContinuousParameter(name=PYFDN_DIFFVOX_DELAY_GAIN_NAME, **unit),
        *PYFDN_DIFFVOX_DELAY_LP_BAND.parameters(),
        ContinuousParameter(name=PYFDN_DIFFVOX_DELAY_ODD_PAN_NAME, **unit),
        ContinuousParameter(name=PYFDN_DIFFVOX_DELAY_EVEN_PAN_NAME, **unit),
        ContinuousParameter(name=PYFDN_DIFFVOX_SEND_NAME, **unit),
        ContinuousArrayParameter(
            name=PYFDN_DIFFVOX_REVERB_INPUT_NAME,
            shape=(PYFDN_DIFFVOX_ORDER, 2),
            min=-1.0,
            max=1.0,
        ),
        ContinuousArrayParameter(
            name=PYFDN_DIFFVOX_REVERB_OUTPUT_NAME,
            shape=(2, PYFDN_DIFFVOX_ORDER),
            min=-1.0,
            max=1.0,
        ),
        ContinuousArrayParameter(
            name=PYFDN_DIFFVOX_REVERB_SKEW_NAME,
            shape=(PYFDN_DIFFVOX_REVERB_SKEW_SIZE,),
            min=-math.pi,
            max=math.pi,
        ),
        ContinuousArrayParameter(
            name=PYFDN_DIFFVOX_REVERB_RT_NAME,
            shape=(10,),
            min=PYFDN_RT_MIN_SECONDS,
            max=PYFDN_GEQ_RT_MAX_SECONDS,
        ),
        *_band_controls(PYFDN_DIFFVOX_TONE_BANDS),
    ]


PYFDN_DIFFVOX_PARAM_SPEC = PyFDNParamSpec(
    synth_params=_diffvox_parameters(),
    feedback_matrix=None,
)
