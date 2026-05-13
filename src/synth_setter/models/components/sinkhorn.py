import math

import torch
import torch.nn as nn


def sinkhorn_C(
    costs: torch.Tensor,
    iters: int = 5,
    reg: float = 1.0,
    # eps: float = 1e-6,
) -> torch.Tensor:
    # costs has shape (b k n)
    K = costs / reg

    # we initialize our scaling vectors to ones
    num_tokens, num_params = costs.shape
    u = torch.zeros(1, num_params, device=K.device, dtype=K.dtype)
    v = torch.zeros(num_tokens, 1, device=K.device, dtype=K.dtype)

    # we initialize our marginals proportionally to the number of tokens
    # a = num_tokens / num_params  # row marginal
    a = 1.0
    b = 1.0  # column marginal

    log_a = math.log(a)
    log_b = math.log(b)

    for _ in range(iters):
        u = log_a - torch.logsumexp(K + v, dim=0, keepdim=True)
        v = log_b - torch.logsumexp(K + u, dim=1, keepdim=True)

    logPi = K + u + v
    return torch.exp(logPi)


def sinkhorn(
    query: torch.Tensor,
    key: torch.Tensor,
    iters: int = 5,
    reg: float = 0.1,
    # eps: float = 1e-6,
) -> torch.Tensor:
    # q has shape (k d)
    # k has shape (b n d)
    costs = torch.cdist(query[None], key, p=2.0)  # (b k n)
    K = -costs / reg

    # we initialize our scaling vectors to ones
    batch_size, num_params, _ = key.shape
    num_tokens, _ = query.shape
    u = torch.zeros(batch_size, 1, num_params, device=K.device, dtype=K.dtype)
    v = torch.zeros(batch_size, num_tokens, 1, device=K.device, dtype=K.dtype)

    # we initialize our marginals proportionally to the number of tokens
    a = 1 / num_params  # row marginal
    b = 1  # column marginal
    log_a = math.log(a)
    log_b = math.log(b)

    for _ in range(iters):
        u = log_a - torch.logsumexp(K + v, dim=1, keepdim=True)
        v = log_b - torch.logsumexp(K + u, dim=2, keepdim=True)

    logPi = K + u + v
    return torch.exp(logPi)


class SinkhornAttention(nn.Module):
    """Applies attention using a fixed number of Sinkhorn iterations in lieu of softmax.

    This allows our attention matrix to approximate a doubly stochastic matrix, or a transportation
    polytope in the case of cross-attention.
    """

    def __init__(
        self,
        d_model: int,
        kdim: int,
        vdim: int,
        sinkhorn_iters: int = 5,
        sinkhorn_reg: float = 1.0,
    ):
        super().__init__()

        self.Q = nn.Parameter(torch.empty(d_model, d_model))
        self.K = nn.Parameter(torch.empty(kdim, d_model))
        self.V = nn.Parameter(torch.empty(vdim, d_model))

        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.K)
        nn.init.xavier_uniform_(self.V)

        self.sinkhorn_iters = sinkhorn_iters

        self.sinkhorn_reg = sinkhorn_reg

    def forward(self, q, k, v):
        q = q @ self.Q
        k = k @ self.K
        v = v @ self.V

        pi = sinkhorn(q, k, iters=self.sinkhorn_iters, reg=self.sinkhorn_reg)
        x = torch.einsum("bkn,bnd->bkd", pi, v)

        return x
