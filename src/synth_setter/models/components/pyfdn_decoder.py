"""Tensor-native decoding of model-space pyFDN parameters."""

from __future__ import annotations

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from pyFDN import decay_to_first_order_shelf
from torch import Tensor, nn

from synth_setter.data.pyfdn_param_spec import (
    PYFDN_HOUSEHOLDER_VECTOR_NAME,
    PYFDN_KRONECKER_ANGLES_NAME,
    PYFDN_KRONECKER_REFLECT_NAME,
    PYFDN_RT_CROSSOVER_HZ,
    PYFDN_RT_DC_NAME,
    PYFDN_RT_NYQUIST_NAME,
    FlamoFDNParamSpec,
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
    Parameter,
    ParamSpec,
)
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.param_spec_name import ParamSpecName

_PARAM_AXIS = "params"
_ANY_SHAPE = "..."
_ORDER = "order"
_LEVELS = "levels"


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

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def _unit(values: Float[Tensor, _ANY_SHAPE]) -> Float[Tensor, _ANY_SHAPE]:
        """Clip ordinary coordinates in the forward pass with an affine surrogate gradient.

        :param values: Unconstrained model-space coordinates.
        :returns: Exact unit-domain values with straight-through clipping gradients.
        """
        unit = (values + 1.0) / 2.0
        return unit.clamp(0.0, 1.0).detach() + (unit - unit.detach())

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

    @jaxtyped(typechecker=beartype)
    def decode_build_fields(
        self, params: Float[Tensor, _PARAM_AXIS], *, sample_rate: float
    ) -> dict[str, Float[Tensor, _ANY_SHAPE]]:
        """Translate a basic model parameterization into the graph's varying build fields.

        :param params: One model-space row for a FlamoFDNParamSpec.
        :param sample_rate: Canonical build's processing rate in Hz.
        :returns: Tensor counterparts of A/B/C/D, delays and the post-delay SOS bank.
        :raises ValueError: This parameterization does not represent a complete FlamoFDN.
        """
        if not isinstance(self.spec, FlamoFDNParamSpec):
            raise ValueError("parameterization does not represent a complete FlamoFDN")
        native = self(params)
        return {
            "A": native["feedback_matrix"],
            "B": native["input_matrix"],
            "C": native["output_matrix"],
            "D": native["direct_matrix"],
            "delays": native["delays"],
            "post_delay": decay_to_first_order_shelf(
                native[PYFDN_RT_DC_NAME],
                native[PYFDN_RT_NYQUIST_NAME],
                PYFDN_RT_CROSSOVER_HZ,
                native["delays"],
                sample_rate,
            ),
        }

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
        elif "input_matrix" in decoded:
            direction = params.new_ones(decoded["input_matrix"].shape[0])
            direction = direction / torch.linalg.vector_norm(direction)
            decoded["feedback_matrix"] = self._householder(direction)
        return decoded
