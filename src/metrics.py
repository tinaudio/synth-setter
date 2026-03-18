from collections.abc import Callable

import torch
from scipy.optimize import linear_sum_assignment
from torchmetrics import Metric

from src.models.components.loss import chamfer_loss, params_to_tokens


def complex_to_dbfs(z: torch.Tensor, eps: float = 1e-8):
    squared_modulus = z.real.square() + z.imag.square()
    clamped = torch.clamp(squared_modulus, min=eps)
    return 10 * torch.log10(clamped)


class LogSpectralDistance(Metric):
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


class ChamferDistance(Metric):
    def __init__(self, params_per_token: int, **kwargs):
        super().__init__(**kwargs)
        self.add_state("chamfer_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.params_per_token = params_per_token

    def update(self, predicted: torch.Tensor, target: torch.Tensor):
        self.chamfer_distance += chamfer_loss(predicted, target, self.params_per_token)
        self.count += 1

    def compute(self):
        return self.chamfer_distance / self.count


class LinearAssignmentDistance(Metric):
    def __init__(self, params_per_token: int, **kwargs):
        super().__init__(**kwargs)
        self.add_state(
            "linear_assignment_distance",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.params_per_token = params_per_token

    def update(self, predicted: torch.Tensor, target: torch.Tensor):
        predicted_tokens = params_to_tokens(predicted, self.params_per_token)
        target_tokens = params_to_tokens(target, self.params_per_token)

        dist = torch.cdist(predicted_tokens, target_tokens)
        dist_c = dist.detach().cpu()

        cost = 0.0
        for b in range(dist_c.shape[0]):
            row_ind, col_ind = linear_sum_assignment(dist_c[b])
            cost = cost + dist[b, row_ind, col_ind].mean()

        self.count += dist.shape[0]
        self.linear_assignment_distance += cost

    def compute(self):
        return self.linear_assignment_distance / self.count
