"""Tensor-native differentiable rendering backends for model training."""

from __future__ import annotations

from collections import OrderedDict
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from beartype import beartype
from flamo.processor import system
from jaxtyping import Float, jaxtyped
from pyFDN import dss_to_flamo
from pyFDN.auxiliary.flamo import delay_module, sos_filter_module
from torch import Tensor, nn

from synth_setter.data.pyfdn_param_spec import (
    PYFDN_DIRECT_DELAY_SAMPLES,
    PYFDN_GEQ_BAND_GAIN_DB_NAME,
    PYFDN_GEQ_GAIN_DB_NAME,
    PYFDN_GEQ_SECTIONS,
    PYFDN_RT_DC_NAME,
    PYFDN_RT_NYQUIST_NAME,
    PYFDN_TONE_GEQ_GAIN_DB_NAME,
)
from synth_setter.data.pyfdn_source import PYFDN_SOURCE_SAMPLE_RATE_HZ
from synth_setter.data.torchsynth_grad_render import (
    differentiable_decode,
    render_torchsynth_grad,
    validate_torchsynth_params,
)
from synth_setter.data.vst.param_spec import DiscreteArrayParameter, decode_model_output
from synth_setter.models.components.pyfdn_decoder import PyFDNParameterDecoder
from synth_setter.models.components.pyfdn_filters import attenuation_sos, command_geq_sos

_BATCH_PARAMS = "batch params"
_BATCH_AUDIO = "batch samples"
_ROW_PARAMS = "params"
_ROW_AUDIO = "samples"
_SUPPORTED_FLAMO_SPECS = frozenset(
    {
        "pyfdn_n8_mono_householder",
        "pyfdn_n8_mono_householder_vector",
        "pyfdn_n8_mono_kronecker",
        "pyfdn_gotz_n8_mono_fixed_delays",
        "pyfdn_gotz_n8_mono_learned_delays",
        "pyfdn_gotz_n8_mono_fixed_delays_givens",
        "pyfdn_gotz_n8_mono_learned_delays_givens",
    }
)


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
        :returns: Batched mono audio with gradients to supported controls.
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
    """Bind decoded predictions to an upstream FLAMO mono, time-invariant FDN graph."""

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        *,
        param_spec: str,
        sample_rate: int,
        signal_length: int,
        fft_size: int | None = None,
    ) -> None:
        """Construct the selected FDN topology once, independently of model predictions.

        :param param_spec: Registered plain or Götz mono FDN parameterization.
        :param sample_rate: Must match the pyFDN source sample rate.
        :param signal_length: Positive number of output samples.
        :param fft_size: FFT period, at least twice the output length; defaults to the next power
            of two. Longer periods reduce circular tail aliasing.
        :raises ValueError: Unsupported topology, sample rate, or render geometry.
        """
        super().__init__()
        if param_spec not in _SUPPORTED_FLAMO_SPECS:
            raise ValueError(f"unsupported FLAMO topology: {param_spec}")
        if sample_rate != PYFDN_SOURCE_SAMPLE_RATE_HZ:
            raise ValueError(
                f"FLAMO pyFDN parity requires sample_rate={PYFDN_SOURCE_SAMPLE_RATE_HZ}"
            )
        if signal_length <= 0:
            raise ValueError("signal_length must be positive")
        minimum_fft_size = 2 * signal_length
        self.fft_size = (
            fft_size if fft_size is not None else 1 << (minimum_fft_size - 1).bit_length()
        )
        if self.fft_size < minimum_fft_size:
            raise ValueError("fft_size must be at least twice signal_length")
        self.sample_rate = sample_rate
        self.signal_length = signal_length
        self.decoder = PyFDNParameterDecoder(param_spec)
        self._is_gotz = PYFDN_GEQ_GAIN_DB_NAME in self.decoder.spec.synth_param_names
        self._core = self._build_flamo_core()

    @jaxtyped(typechecker=beartype)
    def _build_flamo_core(self) -> nn.Module:
        """Use a native template only to establish graph shapes and delay capacity.

        :returns: Frequency-domain graph accepting FLAMO's external parameter mapping.
        """
        spec = self.decoder.spec
        native, _ = decode_model_output(np.zeros(spec.encoded_width), spec)
        delays = np.asarray(native["delays"])
        for parameter in spec.synth_params:
            if parameter.name == "delays" and isinstance(parameter, DiscreteArrayParameter):
                delays = np.full(parameter.shape, parameter.max)
        sections = PYFDN_GEQ_SECTIONS if self._is_gotz else 1
        sos = np.zeros((sections, 6, len(delays)))
        sos[:, 0, :] = sos[:, 3, :] = 1.0
        core = dss_to_flamo(
            np.asarray(native["feedback_matrix"]),
            np.asarray(native["input_matrix"]),
            np.asarray(native["output_matrix"]),
            np.asarray(native["direct_matrix"]),
            delays,
            self.sample_rate,
            nfft=self.fft_size,
            device="cpu",
            shell=False,
            post_delay=sos,
        )
        if not self._is_gotz:
            return core
        # Götz tone correction precedes both branches; only its direct branch is delayed.
        core.branchB = system.Series(
            OrderedDict(
                delay=delay_module(
                    np.array([PYFDN_DIRECT_DELAY_SAMPLES / self.sample_rate]),
                    self.fft_size,
                    fs=self.sample_rate,
                    device="cpu",
                ),
                gain=core.branchB,
            )
        )
        tone = sos_filter_module(sos[:, :, :1], self.fft_size, device="cpu")
        return system.Series(OrderedDict(tone=tone, fdn=core))

    @jaxtyped(typechecker=beartype)
    def validate(self, params: Float[Tensor, _BATCH_PARAMS]) -> None:
        """Validate every row before an optional render skip.

        :param params: Model-space rows for the selected ParamSpec.
        :raises ValueError: A row has the wrong width or a non-finite value.
        """
        width = self.decoder.spec.encoded_width
        if params.shape[-1] != width:
            raise ValueError(f"pyFDN rows must have width {width}, got {params.shape[-1]}")
        if not torch.isfinite(params).all():
            raise ValueError("pyFDN rows must contain only finite values")

    @jaxtyped(typechecker=beartype)
    def _render_row(self, params: Float[Tensor, _ROW_PARAMS]) -> Float[Tensor, _ROW_AUDIO]:
        """Bind one prediction because FLAMO shares DSP parameters across its batch.

        :param params: Validated model-space row.
        :returns: Truncated mono impulse response including the direct path.
        """
        native = self.decoder(params)
        delays = native["delays"]
        if self._is_gotz:
            sos = command_geq_sos(
                torch.cat(
                    (native[PYFDN_GEQ_GAIN_DB_NAME][None, :], native[PYFDN_GEQ_BAND_GAIN_DB_NAME])
                ),
                sample_rate=self.sample_rate,
            )
        else:
            sos = attenuation_sos(
                delays,
                native[PYFDN_RT_DC_NAME],
                native[PYFDN_RT_NYQUIST_NAME],
                sample_rate=self.sample_rate,
            )
        external = {
            "branchA": {
                "input_gain": native["input_matrix"],
                "feedback_loop": {
                    "feedforward": {"delay": delays / self.sample_rate, "post_delay": sos},
                    "feedback": native["feedback_matrix"],
                },
                "output_gain": native["output_matrix"],
            },
            "branchB": native["direct_matrix"],
        }
        if self._is_gotz:
            external["branchB"] = {"gain": native["direct_matrix"]}
            external = {
                "tone": command_geq_sos(
                    native[PYFDN_TONE_GEQ_GAIN_DB_NAME][:, None], sample_rate=self.sample_rate
                ),
                "fdn": external,
            }
        impulse = params.new_zeros((1, self.fft_size, 1))
        impulse[:, 0, 0] = 1.0
        spectrum = torch.fft.rfft(impulse, n=self.fft_size, dim=1)
        rendered = torch.fft.irfft(self._core(spectrum, external), n=self.fft_size, dim=1)
        return rendered[0, : self.signal_length, 0]

    @jaxtyped(typechecker=beartype)
    def forward(self, params: Float[Tensor, _BATCH_PARAMS]) -> Float[Tensor, _BATCH_AUDIO]:
        """Decode and render model predictions with gradients to continuous controls.

        :param params: Selected pyFDN rows in model space; discrete controls are rounded.
        :returns: Mono impulse responses in pyFDN's unnormalized amplitude convention.
        :raises ValueError: A row is malformed or produces non-finite audio.
        """
        self.validate(params)
        if params.shape[0] == 0:
            return params.new_empty((0, self.signal_length))
        rendered = torch.stack([self._render_row(row) for row in params])
        if not torch.isfinite(rendered).all():
            raise ValueError("FLAMO rendered non-finite audio")
        return rendered
