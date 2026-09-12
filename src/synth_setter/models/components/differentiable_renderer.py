"""Tensor-native differentiable rendering backends for model training."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from beartype import beartype
from flamo.processor import system
from jaxtyping import Float, Shaped, jaxtyped
from torch import Tensor, nn

from synth_setter.data.flamo_fdn import FlamoFDN
from synth_setter.data.pyfdn_param_spec import FlamoFDNParamSpec
from synth_setter.data.pyfdn_source import PYFDN_SOURCE_SAMPLE_RATE_HZ
from synth_setter.data.torchsynth_grad_render import (
    differentiable_decode,
    render_torchsynth_grad,
    validate_torchsynth_params,
)
from synth_setter.data.vst.param_spec import decode_model_output
from synth_setter.models.components.pyfdn_decoder import PyFDNParameterDecoder

_BATCH_PARAMS = "batch params"
_BATCH_AUDIO = "batch *channels samples"
_ROW_PARAMS = "params"
_ROW_AUDIO = "channels samples"
_BUILD_FIELD = "..."


@runtime_checkable
class FDNBuildDecoder(Protocol):
    """Translate model coordinates into tensor counterparts of a complete build's fields."""

    @jaxtyped(typechecker=beartype)
    def decode_build_fields(
        self, params: Float[Tensor, _ROW_PARAMS], *, sample_rate: float
    ) -> dict[str, Float[Tensor, _BUILD_FIELD]]:
        """Decode matrices, delays and every declared SOS bank without crossing NumPy.

        :param params: One model-space parameter row.
        :param sample_rate: Canonical build's fixed processing rate.
        :returns: Field-name mapping with the build's native array shapes.
        """
        ...


@runtime_checkable
class DifferentiableRenderer(Protocol):
    """Render model-space parameter rows without breaking autograd."""

    @jaxtyped(typechecker=beartype)
    def validate(self, params: Float[Tensor, _BATCH_PARAMS]) -> None:
        """Reject malformed rows even when the caller skips rendering.

        :param params: Model-space parameter rows.
        """

    @jaxtyped(typechecker=beartype)
    def __call__(self, params: Float[Tensor, _BATCH_PARAMS]) -> Float[Tensor, _BATCH_AUDIO]:
        """Render audio in the backend's dataset amplitude convention.

        :param params: Model-space parameter rows.
        :returns: Batched audio retaining the backend's channels and supported gradients.
        """
        ...


class TorchSynthDifferentiableRenderer(nn.Module):
    """Adapt the existing TorchSynth gradient renderer to the backend-neutral boundary."""

    @jaxtyped(typechecker=beartype)
    def __init__(self, *, sample_rate: int, signal_length: int, render_batch_size: int) -> None:
        """Configure TorchSynth's fixed-size voice batch.

        :param sample_rate: Output sample rate in Hz.
        :param signal_length: Output samples per row.
        :param render_batch_size: Voice batch capacity; shorter batches are padded.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.signal_length = signal_length
        self.render_batch_size = render_batch_size

    @jaxtyped(typechecker=beartype)
    def validate(self, params: Float[Tensor, _BATCH_PARAMS]) -> None:
        """Validate TorchSynth rows before an optional render skip.

        :param params: Finite TorchSynth-full model-space rows.
        """
        validate_torchsynth_params(differentiable_decode(params))

    @jaxtyped(typechecker=beartype)
    def forward(self, params: Float[Tensor, _BATCH_PARAMS]) -> Float[Tensor, _BATCH_AUDIO]:
        """Decode and render TorchSynth model-space rows.

        :param params: TorchSynth-full rows in model space ``[-1, 1]``.
        :returns: Rendered audio shaped ``(batch, signal_length)``.
        """
        decoded = differentiable_decode(params)
        validate_torchsynth_params(decoded)
        rendered = render_torchsynth_grad(
            decoded,
            sample_rate=self.sample_rate,
            signal_length=self.signal_length,
            render_batch_size=self.render_batch_size,
        )
        # Match stored TorchSynth audio without discarding gradients at clipped samples.
        return rendered + (rendered.clamp(-1.0, 1.0) - rendered).detach()


class FlamoFDNDifferentiableRenderer(nn.Module):
    """Bind model predictions to a complete FlamoFDN's upstream FLAMO graph."""

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        *,
        fdn: FlamoFDN,
        decoder: nn.Module,
        parameter_width: int,
        signal_length: int,
        fft_size: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Construct the complete graph with channel geometry taken from its build.

        :param fdn: Complete template whose delays reserve maximum prediction capacity.
        :param decoder: Module implementing tensor-native ``decode_build_fields`` for every
            matrix, delay and present SOS hook; sample rate remains fixed by the build.
        :param parameter_width: Number of model-space coordinates consumed by the decoder.
        :param signal_length: Positive output length in samples.
        :param fft_size: FFT period, at least twice the output length; defaults to the next
            power of two. Longer periods reduce circular tail aliasing.
        :param dtype: Construction precision for parameters and frequency-grid buffers.
        :raises ValueError: Render geometry or parameter width is invalid.
        :raises TypeError: The decoder does not implement the tensor build contract.
        """
        super().__init__()
        if signal_length <= 0 or parameter_width <= 0:
            raise ValueError("signal_length and parameter_width must be positive")
        minimum_fft_size = 2 * signal_length
        self.fft_size = (
            fft_size if fft_size is not None else 1 << (minimum_fft_size - 1).bit_length()
        )
        if self.fft_size < minimum_fft_size:
            raise ValueError("fft_size must be at least twice signal_length")
        if not isinstance(decoder, FDNBuildDecoder):
            raise TypeError("decoder must implement decode_build_fields")
        self.fdn = fdn
        self.decoder = decoder
        self._decode_build_fields = decoder.decode_build_fields
        self.parameter_width = parameter_width
        self.input_channels = fdn.build.B.shape[1]
        self.output_channels = fdn.build.C.shape[0]
        self._field_shapes = {
            name: tuple(value.shape)
            for name in ("A", "B", "C", "D", "delays", "post_delay", "post_matrix", "post_output")
            if (value := getattr(fdn.build, name)) is not None
        }
        self._graph = fdn.to_flamo(nfft=self.fft_size, device="cpu", dtype=dtype)
        self.sample_rate = fdn.build.fs
        self.signal_length = signal_length

    @jaxtyped(typechecker=beartype)
    def _apply(
        self,
        fn: Callable[[Shaped[Tensor, ...]], Shaped[Tensor, ...]],
        recurse: bool = True,
    ) -> FlamoFDNDifferentiableRenderer:
        """Apply a tensor conversion while preserving FLAMO graph invariants.

        :param fn: PyTorch tensor conversion applied recursively.
        :param recurse: Whether to convert child modules.
        :returns: This renderer after conversion.
        """
        super()._apply(fn, recurse)
        # FLAMO 0.2.18 skips same-dtype complex-buffer repair; remove with #3380.
        for module in self._graph.modules():
            if isinstance(module, system.Recursion):
                module._rebuild_buffers()
        return self

    @classmethod
    @jaxtyped(typechecker=beartype)
    def from_param_spec(
        cls,
        *,
        param_spec: str,
        sample_rate: int,
        signal_length: int,
        fft_size: int | None = None,
    ) -> FlamoFDNDifferentiableRenderer:
        """Adapt a registered model encoding without constraining the renderer geometry.

        :param param_spec: Registered FlamoFDNParamSpec identity.
        :param sample_rate: Must match the registered pyFDN dataset sample rate.
        :param signal_length: Positive output length in samples.
        :param fft_size: Optional FFT period forwarded to the renderer.
        :returns: Renderer with a canonical build and its model-specific tensor decoder.
        :raises ValueError: The spec is advanced or the dataset rate is incompatible.
        """
        decoder = PyFDNParameterDecoder(param_spec)
        spec = decoder.spec
        if not isinstance(spec, FlamoFDNParamSpec):
            raise ValueError(f"unsupported FLAMO topology: {param_spec} is not a FlamoFDN spec")
        if sample_rate != PYFDN_SOURCE_SAMPLE_RATE_HZ:
            raise ValueError(
                f"FLAMO pyFDN parity requires sample_rate={PYFDN_SOURCE_SAMPLE_RATE_HZ}"
            )
        # The upper endpoint reserves enough delay capacity for every decoded prediction.
        template, _ = decode_model_output(np.ones(spec.encoded_width), spec)
        return cls(
            fdn=spec.to_flamo_fdn(template),
            decoder=decoder,
            parameter_width=spec.encoded_width,
            signal_length=signal_length,
            fft_size=fft_size,
        )

    @jaxtyped(typechecker=beartype)
    def validate(self, params: Float[Tensor, _BATCH_PARAMS]) -> None:
        """Validate every row before an optional render skip.

        :param params: Model-space rows for the selected ParamSpec.
        :raises ValueError: A row has the wrong width or a non-finite value.
        """
        width = self.parameter_width
        if params.shape[-1] != width:
            raise ValueError(f"pyFDN rows must have width {width}, got {params.shape[-1]}")
        if not torch.isfinite(params).all():
            raise ValueError("pyFDN rows must contain only finite values")

    @jaxtyped(typechecker=beartype)
    def _render_row(self, params: Float[Tensor, _ROW_PARAMS]) -> Float[Tensor, _ROW_AUDIO]:
        """Bind one prediction because FLAMO shares DSP parameters across its batch.

        :param params: Validated model-space row.
        :returns: All transfer paths, channel index ``output * input_channels + input``.
        :raises ValueError: The decoder changes field geometry or exceeds delay capacity.
        """
        fields = self._decode_build_fields(params, sample_rate=self.sample_rate)
        if fields.keys() != self._field_shapes.keys():
            raise ValueError(
                "decoded fields must cover the complete build, including every SOS hook"
            )
        for name, shape in self._field_shapes.items():
            if tuple(fields[name].shape) != shape or not torch.isfinite(fields[name]).all():
                raise ValueError(f"decoded {name} must be finite with shape {shape}")
        if (fields["delays"] <= 0).any() or (fields["delays"] > self.fdn.build.delays.max()).any():
            raise ValueError("decoded delays must be positive and within template capacity")
        delays = fields["delays"] / self.sample_rate
        feedforward = (
            {"delay": delays, "post_delay": fields["post_delay"]}
            if "post_delay" in fields
            else delays
        )
        feedback = (
            {"mixing_matrix": fields["A"], "post_matrix": fields["post_matrix"]}
            if "post_matrix" in fields
            else fields["A"]
        )
        branch = {
            "input_gain": fields["B"],
            "feedback_loop": {"feedforward": feedforward, "feedback": feedback},
            "output_gain": fields["C"],
        }
        if "post_output" in fields:
            branch["post_output"] = fields["post_output"]
        impulse = params.new_zeros((self.input_channels, self.fft_size, self.input_channels))
        impulse[:, 0, :] = torch.eye(self.input_channels, device=params.device, dtype=params.dtype)
        response = self._graph(impulse, {"branchA": branch, "branchB": fields["D"]})
        return (
            response[:, : self.signal_length, :]
            .permute(2, 0, 1)
            .reshape(self.output_channels * self.input_channels, self.signal_length)
        )

    @jaxtyped(typechecker=beartype)
    def forward(self, params: Float[Tensor, _BATCH_PARAMS]) -> Float[Tensor, _BATCH_AUDIO]:
        """Decode and render model predictions with gradients to continuous controls.

        :param params: Selected pyFDN rows in model space; discrete controls are rounded.
        :returns: Unnormalized ``(batch, outputs * inputs, samples)`` transfer responses.
        :raises ValueError: A row is malformed or produces non-finite audio.
        """
        self.validate(params)
        if params.shape[0] == 0:
            return params.new_empty(
                (0, self.output_channels * self.input_channels, self.signal_length)
            )
        rendered = torch.stack([self._render_row(row) for row in params])
        if not torch.isfinite(rendered).all():
            raise ValueError("FLAMO rendered non-finite audio")
        return rendered
