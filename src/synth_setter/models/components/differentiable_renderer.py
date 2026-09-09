"""Tensor-native differentiable rendering backends for model training."""

from __future__ import annotations

from collections import OrderedDict
from typing import Protocol, runtime_checkable

import torch
from beartype import beartype
from flamo.functional import prop_shelving_filter
from flamo.processor import dsp, system
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from synth_setter.data.pyfdn_param_spec import (
    PYFDN_ORDER,
    PYFDN_RT_CROSSOVER_HZ,
    PYFDN_RT_MAX_SECONDS,
    PYFDN_RT_MIN_SECONDS,
)
from synth_setter.data.torchsynth_grad_render import (
    differentiable_decode,
    render_torchsynth_grad,
    validate_torchsynth_params,
)

_BATCH_PARAMS = "batch params"
_BATCH_AUDIO = "batch samples"
_SUPPORTED_FLAMO_SPEC = "pyfdn_n8_mono_householder"
_ROW_PARAMS = "params"
_ROW_AUDIO = "samples"
_SCALAR = ""
_ORDER = "order"


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
    """Render the plain order-8 pyFDN topology through FLAMO's frequency-domain DSP."""

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        *,
        param_spec: str,
        sample_rate: int,
        signal_length: int,
        fft_size: int | None = None,
    ) -> None:
        """Configure a fixed-Householder order-8 impulse-response renderer.

        :param param_spec: Must be ``pyfdn_n8_mono_householder``.
        :param sample_rate: Must match pyFDN's 44100 Hz source.
        :param signal_length: Positive number of output samples.
        :param fft_size: FFT period, at least twice the output length; defaults to
            the next power of two. Longer periods reduce circular tail aliasing.
        :raises ValueError: Unsupported topology, sample rate, or render geometry.
        """
        super().__init__()
        if param_spec != _SUPPORTED_FLAMO_SPEC:
            raise ValueError(f"FLAMO supports only {_SUPPORTED_FLAMO_SPEC}, got {param_spec}")
        if sample_rate != 44_100:
            raise ValueError("FLAMO pyFDN parity requires sample_rate=44100")
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
        self._core = self._build_flamo_core()
        feedback = torch.eye(PYFDN_ORDER) - 2.0 / PYFDN_ORDER
        self.register_buffer("feedback", feedback, persistent=False)

    @jaxtyped(typechecker=beartype)
    def _build_flamo_core(self) -> system.Series:
        """Place the shelf after each delay, matching pyFDN's post-delay hook.

        :returns: B → recursive delay/shelf/feedback → C network.
        """
        geometry = {"nfft": self.fft_size, "alias_decay_db": 0.0, "device": "cpu"}
        delay = dsp.parallelDelay(
            **geometry,
            size=(PYFDN_ORDER,),
            max_len=1200,
            isint=True,
            fs=self.sample_rate,
            unit=1,
        )
        attenuation = dsp.parallelSOSFilter(
            **geometry,
            size=(PYFDN_ORDER,),
            n_sections=1,
            fs=self.sample_rate,
            normalize_a0=False,
        )
        feedback_path = system.Series(OrderedDict(delay=delay, attenuation=attenuation))
        recursion = system.Recursion(
            feedback_path,
            dsp.Matrix(**geometry, size=(PYFDN_ORDER, PYFDN_ORDER)),
        )
        return system.Series(
            OrderedDict(
                input=dsp.Gain(**geometry, size=(PYFDN_ORDER, 1)),
                recursion=recursion,
                output=dsp.Gain(**geometry, size=(1, PYFDN_ORDER)),
            )
        )

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def _decode(
        params: Float[Tensor, _ROW_PARAMS],
    ) -> tuple[
        Float[Tensor, _ORDER],
        Float[Tensor, _ORDER],
        Float[Tensor, _ORDER],
        Float[Tensor, _SCALAR],
        Float[Tensor, _SCALAR],
        Float[Tensor, _SCALAR],
    ]:
        """Decode the fixed-Householder layout without crossing the NumPy boundary.

        :param params: Validated model-space row.
        :returns: Delays in samples, B/C/D gains, and DC/Nyquist RT60 in seconds.
        """
        encoded = ((params + 1.0) / 2.0).clamp(0.0, 1.0)
        delays = torch.round(400.0 + 800.0 * encoded[:8])
        input_gain = -1.0 + 2.0 * encoded[8:16]
        output_gain = -1.0 + 2.0 * encoded[16:24]
        direct_gain = -1.0 + 2.0 * encoded[24]
        rt_span = PYFDN_RT_MAX_SECONDS - PYFDN_RT_MIN_SECONDS
        rt_dc = PYFDN_RT_MIN_SECONDS + rt_span * encoded[25]
        rt_nyquist = PYFDN_RT_MIN_SECONDS + rt_span * encoded[26]
        return delays, input_gain, output_gain, direct_gain, rt_dc, rt_nyquist

    @jaxtyped(typechecker=beartype)
    def validate(self, params: Float[Tensor, _BATCH_PARAMS]) -> None:
        """Validate every row before an optional render skip.

        :param params: Fixed-Householder model-space rows.
        :raises ValueError: A row has the wrong width or a non-finite value.
        """
        if params.shape[-1] != 27:
            raise ValueError(f"pyFDN Householder rows must have width 27, got {params.shape[-1]}")
        if not torch.isfinite(params).all():
            raise ValueError("pyFDN Householder rows must contain only finite values")

    @jaxtyped(typechecker=beartype)
    def _attenuation_sos(
        self,
        delays: Float[Tensor, _ORDER],
        rt_dc: Float[Tensor, _SCALAR],
        rt_nyquist: Float[Tensor, _SCALAR],
    ) -> Float[Tensor, "one coefficients order"]:
        """Convert decay times into normalized proportional-shelf SOS.

        :param delays: Delay-line lengths in samples.
        :param rt_dc: Low-frequency RT60 in seconds.
        :param rt_nyquist: Nyquist RT60 in seconds.
        :returns: One six-coefficient SOS per delay line.
        """
        gain_dc_db = -60.0 * delays / (rt_dc * self.sample_rate)
        gain_nyquist_db = -60.0 * delays / (rt_nyquist * self.sample_rate)
        numerator, denominator = prop_shelving_filter(
            torch.full_like(delays, PYFDN_RT_CROSSOVER_HZ),
            gain_dc_db - gain_nyquist_db,
            fs=self.sample_rate,
            device=str(delays.device),
            dtype=delays.dtype,
        )
        numerator = numerator * torch.pow(10.0, gain_nyquist_db / 20.0)
        # FLAMO's SOS interface needs three coefficients for each first-order polynomial.
        zeros = torch.zeros_like(delays).unsqueeze(0)
        sos = torch.cat((numerator, zeros, denominator, zeros), dim=0)
        return (sos / denominator[0]).unsqueeze(0)

    @jaxtyped(typechecker=beartype)
    def _render_row(self, params: Float[Tensor, _ROW_PARAMS]) -> Float[Tensor, _ROW_AUDIO]:
        """Pass one patch externally because FLAMO shares parameters across its batch.

        :param params: Validated model-space row.
        :returns: Truncated mono impulse response, including the direct path.
        """
        delays, input_gain, output_gain, direct_gain, rt_dc, rt_nyquist = self._decode(params)
        impulse = torch.zeros(1, self.fft_size, 1, dtype=params.dtype, device=params.device)
        impulse[:, 0, 0] = 1.0
        spectrum = torch.fft.rfft(impulse, n=self.fft_size, dim=1)
        external = {
            "input": input_gain[:, None],
            "recursion": {
                "feedforward": {
                    "delay": delays / self.sample_rate,
                    "attenuation": self._attenuation_sos(delays, rt_dc, rt_nyquist),
                },
                "feedback": self.feedback,
            },
            "output": output_gain[None, :],
        }
        wet = self._core(spectrum, external)
        rendered = torch.fft.irfft(wet + direct_gain * spectrum, n=self.fft_size, dim=1)
        return rendered[0, : self.signal_length, 0]

    @jaxtyped(typechecker=beartype)
    def forward(self, params: Float[Tensor, _BATCH_PARAMS]) -> Float[Tensor, _BATCH_AUDIO]:
        """Decode and render model-space pyFDN rows independently.

        Integer delays are rounded exactly as the registered ParamSpec and intentionally
        carry zero gradient. B/C/D and both RT controls remain tensor-native.

        :param params: Fixed-Householder pyFDN rows in model space ``[-1, 1]``.
        :returns: Finite mono impulse responses shaped ``(batch, signal_length)``.
        :raises ValueError: A row has the wrong width or produces non-finite audio.
        """
        self.validate(params)
        if params.shape[0] == 0:
            return params.new_empty((0, self.signal_length))
        rendered = torch.stack([self._render_row(row) for row in params])
        if not torch.isfinite(rendered).all():
            raise ValueError("FLAMO rendered non-finite audio")
        return rendered
