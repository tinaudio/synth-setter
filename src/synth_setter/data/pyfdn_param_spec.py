"""Parameter distributions for pyFDN instruments.

Example:
    ``PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(rng)`` draws one native patch.
"""

import hashlib
import json
from collections.abc import Callable, Mapping

import numpy as np
from pyFDN import householder_matrix

from synth_setter.data.flamo_fdn import FlamoFDN
from synth_setter.data.vst.param_spec import (
    AngleArrayParameter,
    ContinuousArrayParameter,
    ContinuousParameter,
    DirectionArrayParameter,
    DiscreteArrayParameter,
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


_PYFDN_MIDI_STUBS: ParameterValues = {
    "pitch": 0,
    "note_start_and_end": (0.0, 0.0),
}


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


def pyfdn_param_spec_json(spec: PyFDNParamSpec) -> str:
    """Serialize a pyFDN parameter schema in one canonical JSON form.

    :param spec: pyFDN parameter specification to serialize.
    :returns: Compact, key-sorted JSON preserving parameter order and feedback identity.
    """
    feedback_matrix = spec._feedback_matrix
    payload = {
        "feedback_matrix": None if feedback_matrix is None else feedback_matrix.__name__,
        "note_params": [
            {"type": type(parameter).__name__, **vars(parameter)}
            for parameter in spec.note_params
        ],
        "synth_params": [
            {"type": type(parameter).__name__, **vars(parameter)}
            for parameter in spec.synth_params
        ],
        "type": type(spec).__name__,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def pyfdn_param_spec_sha256(spec: PyFDNParamSpec) -> str:
    """Hash the canonical JSON representation of a pyFDN parameter schema.

    :param spec: pyFDN parameter specification to hash.
    :returns: Lowercase SHA-256 digest.
    """
    return hashlib.sha256(pyfdn_param_spec_json(spec).encode()).hexdigest()


class FlamoFDNParamSpec(PyFDNParamSpec):
    """A parameterization whose complete impulse response is represented by one FlamoFDN."""

    def to_flamo_fdn(self, params: Mapping[str, ParameterValue]) -> FlamoFDN:
        """Validate decoded controls and build the complete effect without outer processing.

        :param params: Full native synth mapping, including the derived feedback matrix.
        :returns: Canonical build shared by offline and FLAMO rendering.
        :raises ValueError: Fields or feedback disagree with this parameterization.
        """
        # The instrument imports these specs; load its existing native validator lazily.
        from synth_setter.data.pyfdn_instrument import params_to_fdn_build
        from synth_setter.data.pyfdn_source import PYFDN_SOURCE_SAMPLE_RATE_HZ

        expected = {parameter.name for parameter in self.synth_params} | {"feedback_matrix"}
        if set(params) != expected:
            raise ValueError(f"FLAMO FDN params must contain exactly {sorted(expected)}")
        for parameter in self.synth_params:
            encoded = np.asarray(parameter.encode(params[parameter.name]))
            if not np.isfinite(encoded).all() or np.any((encoded < 0.0) | (encoded > 1.0)):
                raise ValueError(f"{parameter.name} is outside its declared parameter domain")
        native_fields = (
            "feedback_matrix", "input_matrix", "output_matrix", "direct_matrix", "delays",
            PYFDN_RT_DC_NAME, PYFDN_RT_NYQUIST_NAME,
        )
        build = params_to_fdn_build(
            {name: params[name] for name in native_fields},
            sample_rate=float(PYFDN_SOURCE_SAMPLE_RATE_HZ),
        )
        if self._feedback_matrix is None or not np.allclose(
            build.A, self._feedback_matrix(dict(params)), rtol=0.0, atol=1e-6
        ):
            raise ValueError("feedback_matrix does not match the FLAMO FDN controls")
        return FlamoFDN(build)


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


PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC = FlamoFDNParamSpec(
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

PYFDN_N8_MONO_KRONECKER_PARAM_SPEC = FlamoFDNParamSpec(
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

PYFDN_N8_MONO_HOUSEHOLDER_VECTOR_PARAM_SPEC = FlamoFDNParamSpec(
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
