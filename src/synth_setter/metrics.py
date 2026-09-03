"""TorchMetrics-based audio and parameter-space distance metrics."""

import re
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from scipy.optimize import linear_sum_assignment
from torchmetrics import Metric

if TYPE_CHECKING:
    from synth_setter.data.vst.param_spec import DiscreteLiteralParameter, ParamSpec

_NUMBER_VALUE_PATTERN = re.compile(r"[\s_.-]*\d+[\s_.-]*")


def _scalar_midi_pitch_field(
    param_spec: "ParamSpec",
) -> tuple["DiscreteLiteralParameter", slice] | None:
    from synth_setter.data.vst.param_spec import DiscreteLiteralParameter

    pitch_fields = [
        (parameter, span)
        for parameter, span in param_spec.encoded_slices()
        if parameter.name == "pitch"
        and isinstance(parameter, DiscreteLiteralParameter)
        and parameter.encoding == "scalar"
    ]
    return pitch_fields[0] if len(pitch_fields) == 1 else None


def supports_midi_pitch_residuals(param_spec: "ParamSpec") -> bool:
    """Return whether a spec has one scalar discrete MIDI pitch coordinate.

    :param param_spec: Parameter schema to inspect.
    :returns: True when signed MIDI residuals are defined for the schema.
    """
    return _scalar_midi_pitch_field(param_spec) is not None


def midi_pitch_residuals(
    predicted: torch.Tensor,
    target: torch.Tensor,
    param_spec: "ParamSpec",
) -> dict[str, torch.Tensor]:
    """Return signed continuous and quantized MIDI residuals in semitones.

    :param predicted: Model-space parameter vectors shaped ``(batch, num_params)``.
    :param target: Ground-truth model-space vectors with the same shape.
    :param param_spec: Spec containing one scalar discrete parameter named ``pitch``.
    :returns: Per-row predicted-minus-target residuals for each decoding policy.
    :raises ValueError: Tensor shapes mismatch or the spec lacks one scalar discrete pitch.
    """
    if predicted.ndim != 2 or predicted.shape != target.shape:
        raise ValueError(
            f"expected matching 2-D shapes, got {tuple(predicted.shape)} and {tuple(target.shape)}"
        )
    if predicted.shape[1] != param_spec.encoded_width:
        raise ValueError(
            f"expected ParamSpec width {param_spec.encoded_width}, got {predicted.shape[1]}"
        )

    pitch_field = _scalar_midi_pitch_field(param_spec)
    if pitch_field is None:
        raise ValueError("expected a unique scalar discrete pitch parameter")

    pitch, span = pitch_field
    predicted_encoded = ((predicted[:, span].squeeze(1) + 1) / 2).clamp(0, 1)
    target_encoded = (target[:, span].squeeze(1) + 1) / 2
    pitch_span = pitch.max - pitch.min
    predicted_midi = pitch.min + predicted_encoded * pitch_span
    target_midi = torch.floor(pitch.min + target_encoded * pitch_span + 0.5)
    return {
        "continuous": predicted_midi - target_midi,
        "floor": torch.floor(predicted_midi) - target_midi,
        "nearest": torch.floor(predicted_midi + 0.5) - target_midi,
    }


def spec_quantized_per_param_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    param_spec: "ParamSpec",
) -> torch.Tensor:
    """Return MSE after predictions snap to values used by the renderer.

    :param predicted: Model-space parameter vectors shaped ``(batch, num_params)``.
    :param target: Ground-truth model-space vectors with the same shape.
    :param param_spec: Spec defining clipping and discrete parameter values.
    :returns: Per-encoded-column mean squared error shaped ``(num_params,)``.
    :raises ValueError: Shapes mismatch or either tensor contains a non-finite value.
    """
    if predicted.ndim != 2 or predicted.shape != target.shape:
        raise ValueError(
            f"expected matching 2-D shapes, got {tuple(predicted.shape)} and {tuple(target.shape)}"
        )
    if predicted.shape[1] != param_spec.encoded_width:
        raise ValueError(
            f"expected ParamSpec width {param_spec.encoded_width}, got {predicted.shape[1]}"
        )
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError("predicted and target parameters must contain only finite values")

    from synth_setter.data.vst.param_spec import (
        CategoricalParameter,
        DiscreteArrayParameter,
        DiscreteLiteralParameter,
    )

    effective_encoded = ((predicted.float() + 1) / 2).clamp(0, 1)
    for parameter, span in param_spec.encoded_slices():
        values = effective_encoded[:, span]
        is_onehot = isinstance(parameter, (CategoricalParameter, DiscreteLiteralParameter)) and (
            parameter.encoding == "onehot"
        )
        if is_onehot:
            quantized = torch.zeros_like(values)
            quantized.scatter_(1, values.argmax(dim=1, keepdim=True), 1)
            effective_encoded[:, span] = quantized
        elif isinstance(parameter, CategoricalParameter):
            levels = values.new_tensor(parameter.raw_values)
            nearest = (values - levels).abs().argmin(dim=1)
            effective_encoded[:, span] = levels[nearest].unsqueeze(1)
        elif isinstance(parameter, DiscreteLiteralParameter):
            native = values * (parameter.max - parameter.min) + parameter.min
            effective_encoded[:, span] = (native.trunc() - parameter.min) / (
                parameter.max - parameter.min
            )
        elif isinstance(parameter, DiscreteArrayParameter):
            native = values * (parameter.max - parameter.min) + parameter.min
            effective_encoded[:, span] = (native.round() - parameter.min) / (
                parameter.max - parameter.min
            )

    effective_model = effective_encoded * 2 - 1
    return (effective_model - target.float()).square().mean(dim=0)


def complex_to_dbfs(z: torch.Tensor, eps: float = 1e-8):
    squared_modulus = z.real.square() + z.imag.square()
    clamped = torch.clamp(squared_modulus, min=eps)
    return 10 * torch.log10(clamped)


class LogSpectralDistance(Metric):
    """Mean log-spectral distance between predicted and target signals (dBFS magnitude spectra)."""

    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.add_state("lsd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.eps = eps

    def update(
        self,
        predicted_params: torch.Tensor,
        target_signal: torch.Tensor,
        synth_fn: Callable,
    ):
        pred_signal = synth_fn(predicted_params)

        pred_fft = torch.fft.rfft(pred_signal, norm="forward")
        target_fft = torch.fft.rfft(target_signal, norm="forward")

        pred_power = complex_to_dbfs(pred_fft, self.eps)
        target_power = complex_to_dbfs(target_fft, self.eps)

        self.lsd += (pred_power - target_power).square().mean(dim=-1).sqrt().mean()
        self.count += 1

    def compute(self):
        lsd = self.lsd / self.count
        return lsd


class SpectralDistance(Metric):
    """Mean L1 distance between predicted- and target-signal magnitude spectra."""

    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.eps = eps

    def update(
        self,
        predicted_params: torch.Tensor,
        target_signal: torch.Tensor,
        synth_fn: Callable,
    ):
        pred_signal = synth_fn(predicted_params)

        pred_fft = torch.fft.rfft(pred_signal, norm="forward")
        target_fft = torch.fft.rfft(target_signal, norm="forward")

        pred_mag = pred_fft.abs()
        target_mag = target_fft.abs()

        self.sd += torch.nn.functional.l1_loss(pred_mag, target_mag)
        self.count += 1

    def compute(self):
        return self.sd / self.count


def _number_group_indices(
    param_spec: "ParamSpec",
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    grouped_spans: defaultdict[tuple[str, int], list[tuple[int, ...]]] = defaultdict(list)
    for param, span in param_spec.encoded_slices():
        number_group_name = _NUMBER_VALUE_PATTERN.sub("#", param.name)
        grouped_spans[(number_group_name, len(param))].append(tuple(range(span.start, span.stop)))
    return tuple(tuple(spans) for spans in grouped_spans.values())


def number_group_swap_per_param_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    param_spec: "ParamSpec",
) -> torch.Tensor:
    """Return per-parameter MSE after swaps within number-collapsed name groups.

    :param predicted: Parameter vectors, shape ``(batch, num_params)``.
    :param target: Ground-truth vectors, same shape as ``predicted``.
    :param param_spec: Parameter names and encoded spans defining eligible swaps.
    :returns: Per-target-dimension mean squared error, shape ``(num_params,)``.
    :raises ValueError: If tensor shapes or the ParamSpec width do not match.
    """
    if predicted.ndim != 2 or predicted.shape != target.shape:
        raise ValueError(
            f"expected matching 2-D shapes, got {tuple(predicted.shape)} and {tuple(target.shape)}"
        )
    if predicted.shape[1] != param_spec.encoded_width:
        raise ValueError(
            f"expected ParamSpec width {param_spec.encoded_width}, got {predicted.shape[1]}"
        )

    per_target_errors = torch.empty_like(predicted, dtype=torch.float32)
    for group in _number_group_indices(param_spec):
        block_indices = [torch.tensor(block, device=predicted.device) for block in group]
        predicted_blocks = torch.stack([predicted[:, block] for block in block_indices], dim=1)
        target_blocks = torch.stack([target[:, block] for block in block_indices], dim=1)
        if predicted_blocks.shape[-1] == 1:
            sorted_predicted = predicted_blocks.squeeze(-1).sort(dim=1, stable=True).values.float()
            sorted_target, target_indices = target_blocks.squeeze(-1).sort(dim=1, stable=True)
            sorted_errors = (sorted_predicted - sorted_target.float()).square()
            target_errors = torch.empty_like(sorted_errors).scatter(
                1, target_indices, sorted_errors
            )
            per_target_errors[:, torch.cat(block_indices)] = target_errors
            continue

        costs = (
            (predicted_blocks.unsqueeze(2) - target_blocks.unsqueeze(1))
            .float()
            .square()
            .sum(dim=-1)
        )
        for sample_index, sample_costs in enumerate(costs.detach().cpu()):
            predicted_indices, target_indices = linear_sum_assignment(sample_costs)
            for predicted_index, target_index in zip(
                predicted_indices, target_indices, strict=True
            ):
                errors = (
                    predicted_blocks[sample_index, predicted_index].float()
                    - target_blocks[sample_index, target_index].float()
                ).square()
                per_target_errors[sample_index, block_indices[target_index]] = errors
    return per_target_errors.mean(dim=0)


def best_swap_per_param_mse(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return best-swap MSE attributed to each target parameter dimension.

    :param predicted: Parameter vectors, shape ``(batch, num_params)``.
    :param target: Ground-truth vectors, same shape as ``predicted``.
    :returns: Per-target-dimension mean squared error, shape ``(num_params,)``.
    :raises ValueError: If shapes differ or inputs are not 2-D.
    """
    if predicted.ndim != 2 or predicted.shape != target.shape:
        raise ValueError(
            f"expected matching 2-D shapes, got {tuple(predicted.shape)} and {tuple(target.shape)}"
        )

    sorted_predicted = predicted.sort(dim=1, stable=True).values.float()
    sorted_target, target_indices = target.sort(dim=1, stable=True)
    sorted_errors = (sorted_predicted - sorted_target.float()).square()
    per_target_errors = torch.empty_like(sorted_errors).scatter(1, target_indices, sorted_errors)
    return per_target_errors.mean(dim=0)


class NumberGroupSwapParamMSE(Metric):
    """MSE after optimal swaps within number-collapsed parameter-name groups."""

    def __init__(self, param_spec: "ParamSpec") -> None:
        """Register accumulators and the ParamSpec defining eligible swaps.

        :param param_spec: Parameter names and encoded spans defining eligible swaps.
        """
        super().__init__()
        self.param_spec = param_spec
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("element_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate number-group-constrained squared errors.

        :param predicted: Parameter vectors, shape ``(batch, num_params)``.
        :param target: Ground-truth vectors, same shape as ``predicted``.
        """
        per_param_mse = number_group_swap_per_param_mse(predicted, target, self.param_spec)
        self.sum_squared_error = self.sum_squared_error + per_param_mse.sum() * predicted.shape[0]
        self.element_count = self.element_count + predicted.numel()

    def compute(self) -> torch.Tensor:
        """Return the accumulated mean constrained-swap squared error.

        :returns: Scalar mean over every accumulated element.
        """
        return self.sum_squared_error / self.element_count


class BestSwapParamMSE(Metric):
    """MSE after the error-minimizing one-to-one swap of predicted and target scalars.

    The optimistic bracket to plain ``param_mse``: invariant to every permutation
    of parameter values — including sound-changing ones — so it is a floor, never
    a quality verdict. Read the pair as bounds: ``param_mse`` is pessimistic
    (penalizes sound-equivalent reorderings), this metric is optimistic (credits
    non-equivalent ones); a widening gap over training tracks the model producing
    right values in different arrangements. For squared error the optimal scalar
    matching is sort-both-and-compare (rearrangement inequality), so no explicit
    assignment is solved.
    """

    def __init__(self) -> None:
        """Register the squared-error accumulator states."""
        super().__init__()
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("element_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate per-sample sorted-match squared errors.

        :param predicted: Parameter vectors, shape ``(batch, num_params)``.
        :param target: Ground-truth vectors, same shape as ``predicted``.
        """
        per_param_mse = best_swap_per_param_mse(predicted, target)
        self.sum_squared_error = self.sum_squared_error + per_param_mse.sum() * predicted.shape[0]
        self.element_count = self.element_count + predicted.numel()

    def compute(self) -> torch.Tensor:
        """Return the accumulated mean squared error under optimal swapping.

        :returns: Scalar mean over every accumulated element.
        """
        return self.sum_squared_error / self.element_count
