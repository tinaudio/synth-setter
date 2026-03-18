import torch
import torch.nn as nn


def params_to_tokens(params: torch.Tensor, params_per_token: int = 2):
    """Assuming our model outputs a tensor of shape (batch, 2k), we stack it into a tensor of shape
    (batch, k, 2) to allow for metric computation."""
    units = params.chunk(params_per_token, dim=-1)
    return torch.stack(units, dim=-1)


def chamfer_loss(predicted: torch.Tensor, target: torch.Tensor, params_per_token: int = 2):
    predicted_tokens = params_to_tokens(predicted, params_per_token)
    target_tokens = params_to_tokens(target, params_per_token)

    costs = torch.cdist(predicted_tokens, target_tokens).square()
    min1 = costs.min(dim=1)[0].mean(dim=-1)
    min2 = costs.min(dim=2)[0].mean(dim=-1)

    chamfer_distance = torch.mean(min1 + min2)
    return chamfer_distance


class ChamferLoss(nn.Module):
    def __init__(self, params_per_token: int = 2):
        super().__init__()
        self.params_per_token = params_per_token

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return chamfer_loss(predicted, target, self.params_per_token)


class MSESortLoss(nn.Module):
    def __init__(self, params_per_token: int = 2):
        super().__init__()
        self.params_per_token = params_per_token

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_tokens = params_to_tokens(target, self.params_per_token)
        pred_tokens = params_to_tokens(predicted, self.params_per_token)

        target_freqs = target_tokens[..., 0]
        sort_idx = torch.argsort(target_freqs, dim=-1)
        sort_idx = sort_idx.unsqueeze(-1).expand(-1, -1, self.params_per_token)
        target_tokens = torch.gather(target_tokens, 1, sort_idx)

        return nn.functional.mse_loss(pred_tokens, target_tokens)
