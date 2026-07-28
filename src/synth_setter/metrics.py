"""TorchMetrics-based audio and parameter-space distance metrics."""

from collections.abc import Callable

import torch
from torchmetrics import Metric


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
