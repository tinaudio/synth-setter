"""Parameter distributions for order-8 mono pyFDN instruments.

Example:
    ``PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(rng)`` draws one native patch.
"""

from collections.abc import Callable

import numpy as np
from pyFDN import householder_matrix

from synth_setter.data.vst.param_spec import (
    ContinuousArrayParameter,
    ContinuousParameter,
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


_PYFDN_N8_HOUSEHOLDER_FEEDBACK = householder_matrix(np.ones(PYFDN_ORDER, dtype=np.float64))


def _householder_feedback(synth_params: ParameterValues) -> np.ndarray:
    """Return a fresh copy of the fixed all-ones Householder reflection.

    :param synth_params: Decoded fields, unused because the matrix is constant.
    :returns: The order-8 Householder feedback matrix.
    """
    del synth_params
    return _PYFDN_N8_HOUSEHOLDER_FEEDBACK.copy()


def _plain_rt_parameters() -> list[Parameter]:
    """Build fresh DC and Nyquist reverberation-time parameters.

    :returns: The two-band decay controls shared by the plain topologies.
    """
    return [
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
    ]


PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC = PyFDNParamSpec(
    synth_params=[
        *_fdn_matrix_parameters(delay_min=400, delay_max=1200),
        *_plain_rt_parameters(),
    ],
    feedback_matrix=_householder_feedback,
)

# A full turn keeps both signs of every kernel reachable; the wrap at +-pi is the
# price paid for regressing a periodic angle with an MSE loss.
PYFDN_N8_MONO_KRONECKER_PARAM_SPEC = PyFDNParamSpec(
    synth_params=[
        *_fdn_matrix_parameters(delay_min=400, delay_max=1200),
        *_plain_rt_parameters(),
        ContinuousArrayParameter(
            name=PYFDN_KRONECKER_ANGLES_NAME,
            shape=(PYFDN_KRONECKER_LEVELS,),
            min=-np.pi,
            max=np.pi,
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
