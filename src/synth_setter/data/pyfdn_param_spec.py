"""Fixed parameter distribution for the order-8 mono pyFDN instrument."""

import numpy as np

from synth_setter.data.vst.param_spec import (
    ContinuousArrayParameter,
    DiscreteArrayParameter,
    ParamSpec,
)


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


PYFDN_N8_MONO_PARAM_SPEC = ParamSpec(
    synth_params=[
        DiscreteArrayParameter(name="delays", shape=(8,), min=400, max=1200),
        OrthogonalMatrixParameter(
            name="feedback_matrix", shape=(8, 8), min=-1.0, max=1.0
        ),
        ContinuousArrayParameter(
            name="input_matrix", shape=(8, 1), min=-1.0, max=1.0
        ),
        ContinuousArrayParameter(
            name="output_matrix", shape=(1, 8), min=-1.0, max=1.0
        ),
        ContinuousArrayParameter(
            name="direct_matrix", shape=(1, 1), min=-1.0, max=1.0
        ),
    ],
    note_params=[],
)
