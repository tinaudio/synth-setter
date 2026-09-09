"""Parameter distributions for order-8 mono pyFDN instruments.

Example:
    ``PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(rng)`` draws one native patch.
"""

from collections.abc import Callable

import numpy as np
from pyFDN import householder_matrix
from scipy.linalg import expm

from synth_setter.data.vst.param_spec import (
    AngleArrayParameter,
    ContinuousArrayParameter,
    ContinuousParameter,
    DirectionArrayParameter,
    DiscreteArrayParameter,
    DiscreteLiteralParameter,
    Parameter,
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
# Götz et al., arXiv:2510.23158 §2.2.2: coprime, log-spaced lengths kept verbatim in samples.
PYFDN_GOTZ_DELAYS = np.array([809, 877, 937, 1049, 1151, 1249, 1373, 1499], dtype=np.int64)
PYFDN_FEEDBACK_SKEW_NAME = "feedback_skew"
PYFDN_FEEDBACK_SKEW_SIZE = PYFDN_ORDER * (PYFDN_ORDER - 1) // 2
PYFDN_FEEDBACK_SKEW_MAX = np.pi
PYFDN_GEQ_SECTIONS = 11
PYFDN_GEQ_GAIN_DB_NAME = "post_delay.geq.gain_db"
# Flat per-pass loss: -0.3 dB on the longest line is RT60 ~6.8 s, -20 dB is ~0.1 s.
PYFDN_GEQ_GAIN_DB_MIN = -20.0
PYFDN_GEQ_GAIN_DB_MAX = -0.3
PYFDN_GEQ_BAND_GAIN_DB_NAME = "post_delay.geq.band_gain_db"
PYFDN_GEQ_BAND_GAIN_DB_MIN = -6.0
PYFDN_GEQ_BAND_GAIN_DB_MAX = 0.0
PYFDN_TONE_GEQ_GAIN_DB_NAME = "input.tone_geq.command_gain_db"
PYFDN_TONE_GEQ_GAIN_DB_MAX = 12.0
PYFDN_DIRECT_DELAY_SAMPLES = 2


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
        feedback_matrix: Callable[[ParameterValues], np.ndarray],
    ) -> None:
        """Bind learned parameters to a renderer-native feedback-matrix rule.

        :param synth_params: Parameters represented in each encoded row.
        :param feedback_matrix: Maps decoded learned fields to a fresh order-8 matrix.
        """
        super().__init__(synth_params=synth_params, note_params=[])
        self._feedback_matrix = feedback_matrix

    def sample(
        self, rng: np.random.Generator | None = None
    ) -> tuple[ParameterValues, ParameterValues]:
        """Sample learned fields and derive the remaining renderer values.

        :param rng: Optional caller-owned random generator.
        :returns: Complete native FDN values and fixed MIDI compatibility values.
        """
        synth_params, _ = super().sample(rng)
        synth_params["feedback_matrix"] = self._feedback_matrix(synth_params)
        return synth_params, _PYFDN_MIDI_STUBS.copy()

    def decode(self, params: np.ndarray) -> tuple[ParameterValues, ParameterValues]:
        """Decode learned fields and derive the remaining renderer values.

        :param params: Encoded learned FDN parameter row shaped ``(self.encoded_width,)``.
        :returns: Complete native FDN values and fixed MIDI compatibility values.
        """
        synth_params, _ = super().decode(params)
        synth_params["feedback_matrix"] = self._feedback_matrix(synth_params)
        return synth_params, _PYFDN_MIDI_STUBS.copy()


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


def _fdn_io_parameters() -> list[Parameter]:
    """Build fresh input, output, and direct gain parameters shared by every topology.

    :returns: B, C, and D parameter definitions in renderer encoding order.
    """
    return [
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


def _fdn_matrix_parameters(*, delay_min: int, delay_max: int) -> list[Parameter]:
    """Build fresh common FDN parameters in renderer encoding order.

    :param delay_min: Inclusive delay-line lower bound in samples.
    :param delay_max: Inclusive delay-line upper bound in samples.
    :returns: Delay and B/C/D parameter definitions excluding fixed feedback.
    """
    return [
        DiscreteArrayParameter(
            name="delays", shape=(PYFDN_ORDER,), min=delay_min, max=delay_max
        ),
        *_fdn_io_parameters(),
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


def skew_to_orthogonal(skew: np.ndarray) -> np.ndarray:
    """Map strictly-upper-triangular coordinates to an orthogonal matrix.

    Implements Götz et al. §2.2.2: ``U = exp(Tr(P) - Tr(P)^T)``, so every
    decoded feedback matrix is orthogonal without projection.

    :param skew: Upper-triangle entries shaped ``(PYFDN_FEEDBACK_SKEW_SIZE,)`` in row order.
    :returns: Float64 orthogonal matrix shaped ``(PYFDN_ORDER, PYFDN_ORDER)``.
    """
    upper = np.zeros((PYFDN_ORDER, PYFDN_ORDER), dtype=np.float64)
    upper[np.triu_indices(PYFDN_ORDER, k=1)] = np.asarray(skew, dtype=np.float64)
    return np.asarray(expm(upper - upper.T), dtype=np.float64)


def _skew_feedback_from_params(synth_params: ParameterValues) -> np.ndarray:
    """Read the Götz §2.2.2 skew coordinates off one decoded patch.

    :param synth_params: Decoded fields carrying the upper-triangle coordinates.
    :returns: The special orthogonal feedback matrix those coordinates describe.
    """
    return skew_to_orthogonal(np.asarray(synth_params[PYFDN_FEEDBACK_SKEW_NAME]))


class PyFDNGotzParamSpec(PyFDNParamSpec):
    """Restore optional fixed delays after the shared feedback-matrix derivation."""

    def __init__(
        self,
        synth_params: list[Parameter],
        feedback_matrix: Callable[[ParameterValues], np.ndarray],
        fixed_delays: np.ndarray | None,
    ) -> None:
        """Bind learned parameters to derived feedback and optional fixed delays.

        :param synth_params: Parameters represented in each encoded row.
        :param feedback_matrix: Maps decoded learned fields to a fresh order-8 matrix.
        :param fixed_delays: Int64 delays ``(PYFDN_ORDER,)`` restored after decoding, or
            ``None`` when ``delays`` is itself a learned parameter.
        """
        super().__init__(synth_params=synth_params, feedback_matrix=feedback_matrix)
        self._fixed_delays = None if fixed_delays is None else fixed_delays.copy()

    def _restore_delays(self, synth_params: ParameterValues) -> ParameterValues:
        if self._fixed_delays is not None:
            synth_params["delays"] = self._fixed_delays.copy()
        return synth_params

    def sample(
        self, rng: np.random.Generator | None = None
    ) -> tuple[ParameterValues, ParameterValues]:
        """Sample learned fields and restore optional fixed delays.

        :param rng: Optional caller-owned random generator.
        :returns: Complete native FDN values and fixed MIDI compatibility values.
        """
        synth_params, note_params = super().sample(rng)
        return self._restore_delays(synth_params), note_params

    def decode(self, params: np.ndarray) -> tuple[ParameterValues, ParameterValues]:
        """Decode learned fields and restore optional fixed delays.

        :param params: Encoded learned FDN parameter row shaped ``(self.encoded_width,)``.
        :returns: Complete native FDN values and fixed MIDI compatibility values.
        """
        synth_params, note_params = super().decode(params)
        return self._restore_delays(synth_params), note_params


def _gotz_parameters() -> list[Parameter]:
    """Build fresh Götz-topology parameters shared by both delay variants.

    :returns: Skew feedback coordinates, B/C/D gains, per-line attenuation, and tone GEQ.
    """
    return [
        ContinuousArrayParameter(
            name=PYFDN_FEEDBACK_SKEW_NAME,
            shape=(PYFDN_FEEDBACK_SKEW_SIZE,),
            min=-PYFDN_FEEDBACK_SKEW_MAX,
            max=PYFDN_FEEDBACK_SKEW_MAX,
        ),
        *_fdn_io_parameters(),
        ContinuousArrayParameter(
            name=PYFDN_GEQ_GAIN_DB_NAME,
            shape=(PYFDN_ORDER,),
            min=PYFDN_GEQ_GAIN_DB_MIN,
            max=PYFDN_GEQ_GAIN_DB_MAX,
        ),
        ContinuousArrayParameter(
            name=PYFDN_GEQ_BAND_GAIN_DB_NAME,
            shape=(PYFDN_GEQ_SECTIONS - 1, PYFDN_ORDER),
            min=PYFDN_GEQ_BAND_GAIN_DB_MIN,
            max=PYFDN_GEQ_BAND_GAIN_DB_MAX,
        ),
        ContinuousArrayParameter(
            name=PYFDN_TONE_GEQ_GAIN_DB_NAME,
            shape=(PYFDN_GEQ_SECTIONS,),
            min=-PYFDN_TONE_GEQ_GAIN_DB_MAX,
            max=PYFDN_TONE_GEQ_GAIN_DB_MAX,
        ),
    ]


PYFDN_GOTZ_N8_MONO_FIXED_DELAYS_PARAM_SPEC = PyFDNGotzParamSpec(
    synth_params=_gotz_parameters(),
    feedback_matrix=_skew_feedback_from_params,
    fixed_delays=PYFDN_GOTZ_DELAYS,
)

PYFDN_GOTZ_N8_MONO_LEARNED_DELAYS_PARAM_SPEC = PyFDNGotzParamSpec(
    synth_params=[
        DiscreteArrayParameter(
            name="delays",
            shape=(PYFDN_ORDER,),
            min=int(PYFDN_GOTZ_DELAYS.min()),
            max=int(PYFDN_GOTZ_DELAYS.max()),
        ),
        *_gotz_parameters(),
    ],
    feedback_matrix=_skew_feedback_from_params,
    fixed_delays=None,
)
