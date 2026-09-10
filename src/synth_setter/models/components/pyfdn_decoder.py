"""Tensor-native decoding of model-space pyFDN parameters."""

from __future__ import annotations

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor, nn

from synth_setter.data.pyfdn_param_spec import (
    PYFDN_FEEDBACK_GIVENS_ANGLES_NAME,
    PYFDN_FEEDBACK_SKEW_NAME,
    PYFDN_GOTZ_DELAYS,
    PYFDN_HOUSEHOLDER_VECTOR_NAME,
    PYFDN_KRONECKER_ANGLES_NAME,
    PYFDN_KRONECKER_REFLECT_NAME,
    PyFDNParamSpec,
)
from synth_setter.data.vst.param_spec import (
    _ANGLE_PAIR_MIN_NORM,
    _DIRECTION_MIN_NORM,
    AngleArrayParameter,
    ContinuousArrayParameter,
    ContinuousParameter,
    DirectionArrayParameter,
    DiscreteArrayParameter,
    DiscreteLiteralParameter,
    Parameter,
    ParamSpec,
)
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.param_spec_name import ParamSpecName

_PARAM_AXIS = "params"
_ANY_SHAPE = "..."
_ORDER = "order"
_LEVELS = "levels"
_COORDINATES = "coordinates"


class PyFDNParameterDecoder(nn.Module):
    """Decode one model-space row into differentiable pyFDN-native tensors."""

    @jaxtyped(typechecker=beartype)
    def __init__(self, param_spec: str) -> None:
        """Resolve the pyFDN schema used by every subsequent row.

        :param param_spec: Registered pyFDN parameter-spec name.
        :raises ValueError: The registered name does not identify a pyFDN spec.
        """
        super().__init__()
        spec = resolve_param_spec(ParamSpecName(param_spec))
        if not isinstance(spec, PyFDNParamSpec):
            raise ValueError(f"expected a pyFDN param spec, got {param_spec}")
        self.spec: ParamSpec = spec
        self._gotz_delays: Int[Tensor, _ORDER]
        self.register_buffer(
            "_gotz_delays", torch.from_numpy(PYFDN_GOTZ_DELAYS.copy()), persistent=False
        )

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def _unit(values: Float[Tensor, _ANY_SHAPE]) -> Float[Tensor, _ANY_SHAPE]:
        """Map ordinary model coordinates to their saturated unit domain.

        :param values: Model-space coordinates.
        :returns: Coordinates affinely mapped and clipped to ``[0, 1]``.
        """
        return ((values + 1.0) / 2.0).clamp(0.0, 1.0)

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def _affine(
        values: Float[Tensor, _ANY_SHAPE],
        parameter: ContinuousParameter | ContinuousArrayParameter,
    ) -> Float[Tensor, _ANY_SHAPE]:
        """Decode saturated continuous coordinates through declared bounds.

        :param values: Model-space coordinates.
        :param parameter: Scalar or array parameter carrying native bounds.
        :returns: Native values with the input tensor's dtype and device.
        """
        unit = PyFDNParameterDecoder._unit(values)
        return parameter.min + unit * (parameter.max - parameter.min)

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def _decode_discrete(
        values: Float[Tensor, _ANY_SHAPE], parameter: DiscreteArrayParameter
    ) -> Float[Tensor, _ANY_SHAPE]:
        """Round an integer array with NumPy's promoted, ties-to-even semantics.

        :param values: Model-space coordinates.
        :param parameter: Integer array metadata.
        :returns: Rounded native values represented in the input floating dtype.
        """
        promoted = values.to(torch.float64)
        decoded = parameter.min + PyFDNParameterDecoder._unit(promoted) * (
            parameter.max - parameter.min
        )
        return torch.round(decoded).to(values.dtype)

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def _decode_literal(
        values: Float[Tensor, _ANY_SHAPE], parameter: DiscreteLiteralParameter
    ) -> Float[Tensor, _ANY_SHAPE]:
        """Round a scalar integer with the offline decoder's half-up rule.

        :param values: One model-space coordinate.
        :param parameter: Integer-literal metadata.
        :returns: Rounded native value represented in the input floating dtype.
        """
        promoted = values.to(torch.float64)
        offset = PyFDNParameterDecoder._unit(promoted) * (parameter.max - parameter.min)
        lower = torch.floor(offset)
        rounded = parameter.min + lower + (offset - lower >= 0.5).to(offset.dtype)
        return rounded.to(values.dtype)

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def _decode_angles(
        values: Float[Tensor, _ANY_SHAPE], parameter: AngleArrayParameter
    ) -> Float[Tensor, _ANY_SHAPE]:
        """Decode un-clipped direction pairs to periodic radians.

        :param values: Flat model-space ``(cos, sin)`` pairs.
        :param parameter: Angle-array shape metadata.
        :returns: Native radians with deterministic zero-direction values.
        """
        pairs = values.reshape(-1, 2)
        zero = torch.linalg.vector_norm(pairs, dim=-1, keepdim=True) < _ANGLE_PAIR_MIN_NORM
        basis = (torch.arange(2, device=values.device) == 0).to(values.dtype)
        safe_pairs = torch.where(zero, basis.expand_as(pairs), pairs)
        angles = torch.atan2(safe_pairs[:, 1], safe_pairs[:, 0])
        return angles.reshape(parameter.shape)

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def _decode_direction(
        values: Float[Tensor, _ANY_SHAPE], parameter: DirectionArrayParameter
    ) -> Float[Tensor, _ANY_SHAPE]:
        """Normalize an un-clipped direction with a zero-safe basis fallback.

        :param values: Flat model-space direction coordinates.
        :param parameter: Direction-array shape metadata.
        :returns: Unit direction in its native shape.
        """
        norm = torch.linalg.vector_norm(values)
        basis = (torch.arange(values.numel(), device=values.device) == 0).to(values.dtype)
        safe_values = torch.where(norm < _DIRECTION_MIN_NORM, basis, values)
        return (safe_values / torch.linalg.vector_norm(safe_values)).reshape(parameter.shape)

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def _decode_parameter(
        values: Float[Tensor, _ANY_SHAPE], parameter: Parameter
    ) -> Float[Tensor, _ANY_SHAPE]:
        """Dispatch one declared encoded span to its tensor-native inverse.

        :param values: Model-space coordinates owned by ``parameter``.
        :param parameter: Registered parameter metadata.
        :returns: Renderer-native tensor value.
        :raises TypeError: The pyFDN spec contains an unsupported parameter type.
        """
        if isinstance(parameter, AngleArrayParameter):
            return PyFDNParameterDecoder._decode_angles(values, parameter)
        if isinstance(parameter, DirectionArrayParameter):
            return PyFDNParameterDecoder._decode_direction(values, parameter)
        if isinstance(parameter, DiscreteArrayParameter):
            return PyFDNParameterDecoder._decode_discrete(values, parameter).reshape(
                parameter.shape
            )
        if isinstance(parameter, DiscreteLiteralParameter):
            return PyFDNParameterDecoder._decode_literal(values, parameter).squeeze(0)
        if isinstance(parameter, ContinuousArrayParameter):
            return PyFDNParameterDecoder._affine(values, parameter).reshape(parameter.shape)
        if isinstance(parameter, ContinuousParameter):
            return PyFDNParameterDecoder._affine(values, parameter).squeeze(0)
        raise TypeError(f"unsupported pyFDN parameter type: {type(parameter).__name__}")

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def _householder(
        vector: Float[Tensor, _ORDER],
    ) -> Float[Tensor, "order order"]:
        """Build a Householder reflection from a normalized direction.

        :param vector: Unit reflection direction.
        :returns: Orthogonal reflection matrix.
        """
        identity = torch.eye(vector.numel(), dtype=vector.dtype, device=vector.device)
        return identity - 2.0 * torch.outer(vector, vector)

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def _kronecker(
        angles: Float[Tensor, _LEVELS], reflect: Float[Tensor, _LEVELS]
    ) -> Float[Tensor, "order order"]:
        """Build level-ordered Kronecker rotation/reflection feedback.

        :param angles: One periodic angle per level.
        :param reflect: Rounded reflection flags per level.
        :returns: Orthogonal feedback matrix.
        """
        feedback = angles.new_ones((1, 1))
        for angle, flag in zip(angles, reflect, strict=True):
            cosine = torch.cos(angle)
            sine = torch.sin(angle)
            rotation = torch.stack((torch.stack((cosine, -sine)), torch.stack((sine, cosine))))
            reflection = torch.stack((torch.stack((cosine, sine)), torch.stack((sine, -cosine))))
            kernel = torch.where(flag.to(torch.bool), reflection, rotation)
            feedback = torch.kron(kernel, feedback)
        return feedback

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def _skew_feedback(skew: Float[Tensor, _COORDINATES]) -> Float[Tensor, "order order"]:
        """Exponentiate lexicographic upper-triangle skew coordinates.

        :param skew: Strictly upper-triangular entries in row order.
        :returns: Special orthogonal feedback matrix.
        """
        order = round((1.0 + (1.0 + 8.0 * skew.numel()) ** 0.5) / 2.0)
        rows, columns = torch.triu_indices(order, order, offset=1, device=skew.device)
        upper = skew.new_zeros((order, order)).index_put((rows, columns), skew)
        return torch.matrix_exp(upper - upper.mT)

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def _givens_feedback(
        angles: Float[Tensor, _COORDINATES],
    ) -> Float[Tensor, "order order"]:
        """Multiply lexicographic Givens rotations on the right.

        :param angles: One periodic angle per upper-triangle plane.
        :returns: Special orthogonal feedback matrix.
        """
        order = round((1.0 + (1.0 + 8.0 * angles.numel()) ** 0.5) / 2.0)
        identity = torch.eye(order, dtype=angles.dtype, device=angles.device)
        rows, columns = torch.triu_indices(order, order, offset=1, device=angles.device)
        feedback = identity
        for angle, row, column in zip(angles, rows, columns, strict=True):
            first = identity[row]
            second = identity[column]
            plane = torch.outer(first, first) + torch.outer(second, second)
            orientation = torch.outer(second, first) - torch.outer(first, second)
            rotation = identity + (torch.cos(angle) - 1.0) * plane + torch.sin(angle) * orientation
            feedback = feedback @ rotation
        return feedback

    @jaxtyped(typechecker=beartype)
    def forward(self, params: Float[Tensor, _PARAM_AXIS]) -> dict[str, Float[Tensor, _ANY_SHAPE]]:
        """Decode one finite model-space row and derive renderer-native feedback.

        :param params: One model prediction row shaped ``(spec.encoded_width,)``.
        :returns: Renderer-native pyFDN values keyed by declared field name.
        :raises ValueError: The row has the wrong width or contains non-finite values.
        """
        if params.shape[0] != self.spec.encoded_width:
            raise ValueError(
                f"pyFDN row must have width {self.spec.encoded_width}, got {params.shape[0]}"
            )
        if not torch.isfinite(params).all():
            raise ValueError("pyFDN row must contain only finite values")

        decoded = {
            parameter.name: self._decode_parameter(params[span], parameter)
            for parameter, span in self.spec.encoded_slices()
        }
        if PYFDN_HOUSEHOLDER_VECTOR_NAME in decoded:
            decoded["feedback_matrix"] = self._householder(decoded[PYFDN_HOUSEHOLDER_VECTOR_NAME])
        elif PYFDN_KRONECKER_ANGLES_NAME in decoded:
            decoded["feedback_matrix"] = self._kronecker(
                decoded[PYFDN_KRONECKER_ANGLES_NAME], decoded[PYFDN_KRONECKER_REFLECT_NAME]
            )
        elif PYFDN_FEEDBACK_SKEW_NAME in decoded:
            decoded["feedback_matrix"] = self._skew_feedback(decoded[PYFDN_FEEDBACK_SKEW_NAME])
        elif PYFDN_FEEDBACK_GIVENS_ANGLES_NAME in decoded:
            decoded["feedback_matrix"] = self._givens_feedback(
                decoded[PYFDN_FEEDBACK_GIVENS_ANGLES_NAME]
            )
        elif "input_matrix" in decoded:
            direction = params.new_ones(decoded["input_matrix"].shape[0])
            direction = direction / torch.linalg.vector_norm(direction)
            decoded["feedback_matrix"] = self._householder(direction)

        if "feedback_matrix" in decoded and "delays" not in decoded:
            decoded["delays"] = self._gotz_delays.to(params)
        return decoded
