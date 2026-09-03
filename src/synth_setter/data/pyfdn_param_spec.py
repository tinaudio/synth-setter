"""Fixed parameter distribution for the order-8 mono pyFDN instrument.

Example:
    ``PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(seed))`` draws one native patch.
"""

import numpy as np

from synth_setter.data.vst.param_spec import (
    ContinuousArrayParameter,
    ContinuousParameter,
    DiscreteArrayParameter,
    Parameter,
    ParameterValues,
    ParamSpec,
)


PYFDN_ORDER = 8
PYFDN_RT_CROSSOVER_HZ = 6_000.0
PYFDN_RT_DC_NAME = "post_delay.rt_dc_seconds"
PYFDN_RT_MAX_SECONDS = 4.0
PYFDN_RT_MIN_SECONDS = 0.1
PYFDN_RT_NYQUIST_NAME = "post_delay.rt_nyquist_seconds"


class OrthogonalMatrixParameter(ContinuousArrayParameter):
    """Sample Haar-random orthogonal matrices from a caller-owned RNG."""

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        min: float,
        max: float,
    ) -> None:
        """Bind a square matrix to its elementwise encoding range.

        :param name: Logical parameter name.
        :param shape: Square two-dimensional native matrix shape.
        :param min: Inclusive encoded native lower bound.
        :param max: Inclusive encoded native upper bound.
        :raises ValueError: The shape is not square or the bounds are not ``[-1, 1]``.
        """
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("orthogonal matrix shape must be square")
        if min != -1.0 or max != 1.0:
            raise ValueError("orthogonal matrix bounds must be [-1, 1]")
        super().__init__(name=name, shape=shape, min=min, max=max)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one Haar-random orthogonal float64 matrix.

        :param rng: Generator that owns the deterministic sample stream.
        :returns: Float64 orthogonal matrix shaped ``self.shape``.
        """
        matrix = rng.standard_normal(self.shape)
        orthogonal, triangular = np.linalg.qr(matrix)
        signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
        return orthogonal * signs


_PYFDN_MIDI_STUBS: ParameterValues = {
    "pitch": 0,
    "note_start_and_end": (0.0, 0.0),
}


class PyFDNParamSpec(ParamSpec):
    """Expose fixed MIDI stubs without adding model coordinates."""

    def sample(
        self, rng: np.random.Generator | None = None
    ) -> tuple[ParameterValues, ParameterValues]:
        """Sample an FDN patch and return the common renderer's MIDI stubs.

        :param rng: Optional caller-owned random generator.
        :returns: Native FDN values and fixed MIDI compatibility values.
        """
        synth_params, _ = super().sample(rng)
        return synth_params, _PYFDN_MIDI_STUBS.copy()

    def decode(self, params: np.ndarray) -> tuple[ParameterValues, ParameterValues]:
        """Decode an FDN patch and restore the unencoded MIDI stubs.

        :param params: Encoded FDN parameter row.
        :returns: Native FDN values and fixed MIDI compatibility values.
        """
        synth_params, _ = super().decode(params)
        return synth_params, _PYFDN_MIDI_STUBS.copy()


class _FixedFeedbackPyFDNParamSpec(PyFDNParamSpec):
    """Restore a fixed feedback matrix outside the learned coordinates."""

    def __init__(
        self,
        synth_params: list[Parameter],
        feedback_matrix: np.ndarray,
    ) -> None:
        """Bind the learned parameters to one renderer-native feedback matrix.

        :param synth_params: Parameters represented in each encoded row.
        :param feedback_matrix: Fixed order-8 feedback matrix restored after decoding.
        """
        super().__init__(synth_params=synth_params, note_params=[])
        self._feedback_matrix = feedback_matrix.copy()

    def sample(
        self, rng: np.random.Generator | None = None
    ) -> tuple[ParameterValues, ParameterValues]:
        """Sample learned fields and add an independent fixed feedback array.

        :param rng: Optional caller-owned random generator.
        :returns: Complete native FDN values and fixed MIDI compatibility values.
        """
        synth_params, note_params = super().sample(rng)
        synth_params["feedback_matrix"] = self._feedback_matrix.copy()
        return synth_params, note_params

    def decode(self, params: np.ndarray) -> tuple[ParameterValues, ParameterValues]:
        """Decode learned fields and restore an independent fixed feedback array.

        :param params: Encoded learned FDN parameter row.
        :returns: Complete native FDN values and fixed MIDI compatibility values.
        """
        synth_params, note_params = super().decode(params)
        synth_params["feedback_matrix"] = self._feedback_matrix.copy()
        return synth_params, note_params


def _pyfdn_n8_mono_synth_params(
    feedback_parameter: Parameter | None,
) -> list[Parameter]:
    """Build the shared order-8 mono parameter sequence.

    :param feedback_parameter: Learned feedback parameter, or ``None`` when fixed.
    :returns: Fresh parameter definitions in renderer encoding order.
    """
    params: list[Parameter] = [
        DiscreteArrayParameter(
            name="delays", shape=(PYFDN_ORDER,), min=400, max=1200
        )
    ]
    if feedback_parameter is not None:
        params.append(feedback_parameter)
    params.extend(
        [
            ContinuousArrayParameter(
                name="input_matrix", shape=(PYFDN_ORDER, 1), min=-1.0, max=1.0
            ),
            ContinuousArrayParameter(
                name="output_matrix", shape=(1, PYFDN_ORDER), min=-1.0, max=1.0
            ),
            ContinuousArrayParameter(
                name="direct_matrix", shape=(1, 1), min=-1.0, max=1.0
            ),
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
    )
    return params


PYFDN_N8_MONO_PARAM_SPEC = PyFDNParamSpec(
    synth_params=_pyfdn_n8_mono_synth_params(
        OrthogonalMatrixParameter(
            name="feedback_matrix",
            shape=(PYFDN_ORDER, PYFDN_ORDER),
            min=-1.0,
            max=1.0,
        )
    ),
    note_params=[],
)

_PYFDN_N8_HADAMARD_FEEDBACK = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1],
        [1, 1, 1, 1, -1, -1, -1, -1],
        [1, -1, 1, -1, -1, 1, -1, 1],
        [1, 1, -1, -1, -1, -1, 1, 1],
        [1, -1, -1, 1, -1, 1, 1, -1],
    ],
    dtype=np.float64,
) / np.sqrt(PYFDN_ORDER)

PYFDN_N8_MONO_HADAMARD_PARAM_SPEC = _FixedFeedbackPyFDNParamSpec(
    synth_params=_pyfdn_n8_mono_synth_params(feedback_parameter=None),
    feedback_matrix=_PYFDN_N8_HADAMARD_FEEDBACK,
)
