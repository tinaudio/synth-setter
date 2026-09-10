"""TorchMetrics-based audio and parameter-space distance metrics."""

import re
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchmetrics import Metric

if TYPE_CHECKING:
    from synth_setter.data.vst.param_spec import (
        CategoricalParameter,
        DiscreteLiteralParameter,
        ParamSpec,
    )

_NUMBER_GROUP_PATTERN = re.compile(r"[\s_.-]*\d+[\s_.-]*")
_DIGIT_RUN_PATTERN = re.compile(r"\d+")


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
    calculation_device = (
        torch.device("cpu") if predicted.device.type == "mps" else predicted.device
    )
    predicted_model = predicted[:, span].squeeze(1).to(calculation_device).to(torch.float64)
    target_model = target[:, span].squeeze(1).to(calculation_device).to(torch.float64)
    predicted_encoded = ((predicted_model + 1) / 2).clamp(0, 1)
    target_encoded = (target_model + 1) / 2
    pitch_span = pitch.max - pitch.min
    predicted_midi = pitch.min + predicted_encoded * pitch_span
    target_midi = torch.floor(pitch.min + target_encoded * pitch_span + 0.5)
    return {
        "continuous": (predicted_midi - target_midi).to(predicted),
        "floor": (torch.floor(predicted_midi) - target_midi).to(predicted),
        "nearest": (torch.floor(predicted_midi + 0.5) - target_midi).to(predicted),
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
    """
    _validate_semantic_metric_inputs(predicted, target, param_spec)

    from synth_setter.data.vst.param_spec import spec_quantize_model_output

    effective_rows = np.stack(
        [
            spec_quantize_model_output(row, param_spec)
            for row in predicted.detach().float().cpu().numpy()
        ]
    )
    effective = torch.as_tensor(effective_rows, device=predicted.device, dtype=torch.float32)
    return (effective - target.float()).square().mean(dim=0)


def spec_per_param_abs_cosine_distance(
    predicted: torch.Tensor,
    target: torch.Tensor,
    param_spec: "ParamSpec",
) -> dict[str, torch.Tensor]:
    """Return mean sign-invariant cosine distances for model-space array parameters.

    :param predicted: Model-space vectors shaped ``(batch, num_params)``.
    :param target: Ground-truth model-space vectors with the same shape.
    :param param_spec: Spec defining logical array spans.
    :returns: Array names mapped to scalar batch-mean distances in ``[0, 1]``.
        Zero vectors score one; norms are stabilized at ``1e-8``. Angles are
        compared per ``(cos, sin)`` pair before averaging, not as one flat vector.
    :raises ValueError: Inputs have incompatible shapes, an empty batch, or non-finite values.
    """
    if predicted.ndim != 2 or predicted.shape != target.shape:
        raise ValueError(
            f"expected matching 2-D shapes, got {tuple(predicted.shape)} and {tuple(target.shape)}"
        )
    if predicted.shape[1] != param_spec.encoded_width:
        raise ValueError(
            f"expected ParamSpec width {param_spec.encoded_width}, got {predicted.shape[1]}"
        )
    if predicted.shape[0] == 0:
        raise ValueError("expected a non-empty batch")
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError("predicted and target parameters must contain only finite values")

    from synth_setter.data.vst.param_spec import (
        AngleArrayParameter,
        ContinuousArrayParameter,
        DirectionArrayParameter,
    )

    distances = {}
    for parameter, span in param_spec.encoded_slices():
        if not isinstance(
            parameter, (ContinuousArrayParameter, DirectionArrayParameter, AngleArrayParameter)
        ):
            continue
        predicted_array = predicted[:, span].float()
        target_array = target[:, span].float()
        if isinstance(parameter, AngleArrayParameter):
            predicted_array = predicted_array.reshape(predicted.shape[0], -1, 2)
            target_array = target_array.reshape(target.shape[0], -1, 2)
        similarity = torch.nn.functional.cosine_similarity(predicted_array, target_array, dim=-1)
        distances[parameter.name] = (1 - similarity.abs().clamp(max=1)).mean()
    return distances


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


def _number_groups(
    param_spec: "ParamSpec",
) -> tuple[tuple[str, tuple[tuple[int, ...], ...]], ...]:
    groups: defaultdict[tuple[str, int], list[tuple[str, tuple[int, ...]]]] = defaultdict(list)
    for param, span in param_spec.encoded_slices():
        group_key = (_NUMBER_GROUP_PATTERN.sub("#", param.name), len(param))
        groups[group_key].append((param.name, tuple(range(span.start, span.stop))))

    labelled_groups = [
        (
            names_and_spans[0][0]
            if len(names_and_spans) == 1
            else _DIGIT_RUN_PATTERN.sub("N", names_and_spans[0][0]),
            tuple(name for name, _ in names_and_spans),
            tuple(span for _, span in names_and_spans),
        )
        for names_and_spans in groups.values()
    ]
    label_counts = defaultdict(int)
    for label, _, _ in labelled_groups:
        label_counts[label] += 1
    return tuple(
        (
            label if label_counts[label] == 1 else f"{label}__members_{'-'.join(names)}",
            spans,
        )
        for label, names, spans in labelled_groups
    )


def _validate_semantic_metric_inputs(
    predicted: torch.Tensor,
    target: torch.Tensor,
    param_spec: "ParamSpec",
) -> None:
    if predicted.ndim != 2 or predicted.shape != target.shape:
        raise ValueError(
            f"expected matching 2-D shapes, got {tuple(predicted.shape)} and {tuple(target.shape)}"
        )
    if predicted.shape[1] != param_spec.encoded_width:
        raise ValueError(
            f"expected ParamSpec width {param_spec.encoded_width}, got {predicted.shape[1]}"
        )
    if predicted.shape[0] == 0:
        raise ValueError("expected a non-empty batch")
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError("predicted and target parameters must contain only finite values")


def _categorical_identities(
    values: torch.Tensor,
    param_spec: "ParamSpec",
) -> dict[str, np.ndarray]:
    from synth_setter.data.vst.param_spec import CategoricalParameter

    cpu_values = values.detach().cpu()
    if cpu_values.dtype == torch.bfloat16:
        cpu_values = cpu_values.float()
    model_rows = cpu_values.numpy()
    identities = {}
    for parameter, span in param_spec.encoded_slices():
        if not isinstance(parameter, CategoricalParameter):
            continue
        encoded_field = parameter.model_to_encoded(model_rows[:, span])
        if parameter.encoding == "onehot":
            identities[parameter.name] = encoded_field.argmax(axis=1)
        else:
            raw_values = np.asarray(parameter.raw_values)
            identities[parameter.name] = np.abs(encoded_field - raw_values).argmin(axis=1)
    return identities


def categorical_mismatch_rates(
    predicted: torch.Tensor,
    target: torch.Tensor,
    param_spec: "ParamSpec",
) -> dict[str, torch.Tensor]:
    """Return decoded mismatch rate for each logical categorical field.

    :param predicted: Model-space parameter vectors shaped ``(batch, num_params)``.
    :param target: Ground-truth model-space vectors with the same shape.
    :param param_spec: Spec defining categorical encodings and category domains.
    :returns: Parameter names mapped to batch mismatch rates.
    """
    _validate_semantic_metric_inputs(predicted, target, param_spec)
    predicted_ids = _categorical_identities(predicted, param_spec)
    target_ids = _categorical_identities(target, param_spec)
    return {
        name: torch.tensor(
            np.not_equal(identities, target_ids[name]).mean(),
            device=predicted.device,
            dtype=torch.float32,
        )
        for name, identities in predicted_ids.items()
    }


def _categorical_number_groups(
    param_spec: "ParamSpec",
) -> tuple[tuple[str, tuple["CategoricalParameter", ...]], ...]:
    from synth_setter.data.vst.param_spec import CategoricalParameter

    groups: defaultdict[tuple[object, ...], list[CategoricalParameter]] = defaultdict(list)
    for parameter, _ in param_spec.encoded_slices():
        if not isinstance(parameter, CategoricalParameter):
            continue
        compatibility = (
            _NUMBER_GROUP_PATTERN.sub("#", parameter.name),
            parameter.encoding,
            tuple(map(repr, parameter.values)),
            tuple(parameter.raw_values),
        )
        groups[compatibility].append(parameter)
    labelled_groups = [
        (
            parameters[0].name
            if len(parameters) == 1
            else _DIGIT_RUN_PATTERN.sub("N", parameters[0].name),
            tuple(parameters),
        )
        for parameters in groups.values()
    ]
    label_counts = defaultdict(int)
    for label, _ in labelled_groups:
        label_counts[label] += 1
    return tuple(
        (
            label
            if label_counts[label] == 1
            else f"{label}__members_{'-'.join(param.name for param in parameters)}",
            parameters,
        )
        for label, parameters in labelled_groups
    )


def number_group_optimal_assignment_categorical_mismatch_rates(
    predicted: torch.Tensor,
    target: torch.Tensor,
    param_spec: "ParamSpec",
) -> dict[str, torch.Tensor]:
    """Return mismatch-minimizing rates within compatible numbered families.

    :param predicted: Model-space parameter vectors shaped ``(batch, num_params)``.
    :param target: Ground-truth model-space vectors with the same shape.
    :param param_spec: Spec defining compatible categorical numbered families.
    :returns: Collapsed family labels mapped to decoded mismatch rates.
    """
    _validate_semantic_metric_inputs(predicted, target, param_spec)
    predicted_ids = _categorical_identities(predicted, param_spec)
    target_ids = _categorical_identities(target, param_spec)
    rates = {}
    for label, parameters in _categorical_number_groups(param_spec):
        predicted_group = np.stack([predicted_ids[param.name] for param in parameters], axis=1)
        target_group = np.stack([target_ids[param.name] for param in parameters], axis=1)
        categories = np.arange(len(parameters[0].raw_values))
        predicted_counts = (predicted_group[:, :, None] == categories).sum(axis=1)
        target_counts = (target_group[:, :, None] == categories).sum(axis=1)
        match_count = np.minimum(predicted_counts, target_counts).sum()
        decision_count = predicted.shape[0] * len(parameters)
        rates[label] = torch.tensor(
            1 - match_count / decision_count,
            device=predicted.device,
            dtype=torch.float32,
        )
    return rates


def number_group_optimal_assignment_mse_groups(
    per_param_mse: torch.Tensor,
    param_spec: "ParamSpec",
) -> dict[str, torch.Tensor]:
    """Collapse assigned coordinate errors into one mean per numbered family.

    :param per_param_mse: Assigned MSE for each encoded coordinate.
    :param param_spec: Parameter names and encoded spans defining numbered families.
    :returns: Display label to mean assigned squared error.
    :raises ValueError: If the metric width does not match the ParamSpec.
    """
    if per_param_mse.ndim != 1 or per_param_mse.shape[0] != param_spec.encoded_width:
        raise ValueError(
            f"expected {param_spec.encoded_width} per-parameter errors, "
            f"got shape {tuple(per_param_mse.shape)}"
        )

    grouped_mse = {}
    for label, spans in _number_groups(param_spec):
        indices = []
        for span in spans:
            indices.extend(span)
        grouped_mse[label] = per_param_mse[
            torch.tensor(indices, device=per_param_mse.device)
        ].mean()
    return grouped_mse


def number_group_optimal_assignment_per_param_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    param_spec: "ParamSpec",
) -> torch.Tensor:
    """Return per-coordinate MSE after optimal assignment within numbered families.

    :param predicted: Parameter vectors, shape ``(batch, num_params)``.
    :param target: Ground-truth vectors, same shape as ``predicted``.
    :param param_spec: Parameter names and encoded spans defining eligible assignments.
    :returns: Per-coordinate mean squared error, shape ``(num_params,)``.
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
    for _, group in _number_groups(param_spec):
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


class NumberGroupOptimalAssignmentParamMSE(Metric):
    """MSE after optimal assignment within number-collapsed parameter-name groups."""

    def __init__(self, param_spec: "ParamSpec") -> None:
        """Register accumulators and the ParamSpec defining eligible assignments.

        :param param_spec: Parameter names and encoded spans defining eligible assignments.
        """
        super().__init__()
        self.param_spec = param_spec
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("element_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate number-group optimal-assignment squared errors.

        :param predicted: Parameter vectors, shape ``(batch, num_params)``.
        :param target: Ground-truth vectors, same shape as ``predicted``.
        """
        per_param_mse = number_group_optimal_assignment_per_param_mse(
            predicted, target, self.param_spec
        )
        self.sum_squared_error = self.sum_squared_error + per_param_mse.sum() * predicted.shape[0]
        self.element_count = self.element_count + predicted.numel()

    def compute(self) -> torch.Tensor:
        """Return the accumulated mean constrained-assignment squared error.

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
