"""TorchMetrics-based audio and parameter-space distance metrics."""

import itertools
import re
from collections.abc import Callable, Sequence

import torch
from scipy.optimize import linear_sum_assignment
from torchmetrics import Metric

from synth_setter.models.components.loss import chamfer_loss, params_to_tokens


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


class ChamferDistance(Metric):
    """Mean Chamfer distance between predicted and target parameter token sets."""

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
    """Mean linear-assignment (Hungarian-matched) distance between predicted and target tokens."""

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


def derive_interchangeable_groups(
    names: Sequence[str], widths: Sequence[int] | None = None
) -> list[list[list[int]]]:
    """Derive interchangeable parameter blocks from numbered param-name prefixes.

    A name matching ``<base>_<number>_<suffix>`` belongs to block ``number`` of
    family ``base``; blocks of one family with an identical suffix-to-width layout
    form one interchangeable group (>=2 blocks required). Block index lists are
    suffix-sorted, so position ``i`` addresses the same suffix in every block.

    :param names: Param names in encoded-vector order.
    :param widths: Encoded width per name (one-hot params span several indices);
        ``None`` means every name is one scalar index.
    :returns: Groups of aligned encoded-index blocks; ``[]`` when no family has
        two structurally identical blocks.
    """
    if widths is None:
        widths = [1] * len(names)

    offsets = list(itertools.accumulate(widths, initial=0))[:-1]

    pattern = re.compile(r"^(?P<base>.+)_(?P<block>\d+)_(?P<suffix>.+)$")
    families: dict[str, dict[str, dict[str, tuple[int, int]]]] = {}
    for name, offset, width in zip(names, offsets, widths, strict=True):
        match = pattern.match(name)
        if match is None:
            continue
        blocks = families.setdefault(match["base"], {})
        blocks.setdefault(match["block"], {})[match["suffix"]] = (offset, width)

    groups: list[list[list[int]]] = []
    for blocks in families.values():
        layouts: dict[tuple[tuple[str, int], ...], list[list[int]]] = {}
        for block in blocks.values():
            signature = tuple(sorted((suffix, width) for suffix, (_, width) in block.items()))
            indices: list[int] = []
            for suffix in sorted(block):
                offset, width = block[suffix]
                indices.extend(range(offset, offset + width))
            layouts.setdefault(signature, []).append(indices)
        groups.extend(group for group in layouts.values() if len(group) >= 2)

    return groups


class PartitionedLinearAssignmentDistance(Metric):
    """Permutation-optimal MSE: Hungarian-matched over interchangeable blocks, plain elsewhere.

    Directly comparable to a plain elementwise MSE over the same tensors: equal
    when the identity matching is optimal, lower when permuting a group's blocks
    fits the target better.
    """

    def __init__(self, groups: list[list[list[int]]], num_params: int, **kwargs):
        """Validate the partition and register the squared-error accumulator states.

        :param groups: Interchangeable groups of aligned encoded-index blocks
            (see :func:`derive_interchangeable_groups`); indices must be disjoint
            and within ``num_params``.
        :param num_params: Width of the parameter vectors passed to ``update``.
        :param **kwargs: Forwarded to :class:`torchmetrics.Metric`.
        :raises ValueError: If group indices overlap, fall outside ``num_params``,
            or blocks within a group have unequal sizes.
        """
        super().__init__(**kwargs)
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("element_count", default=torch.tensor(0), dist_reduce_fx="sum")

        seen: set[int] = set()
        for group in groups:
            if len({len(block) for block in group}) > 1:
                raise ValueError(f"blocks within a group must share one size, got {group}")
            for block in group:
                indices = set(block)
                if indices & seen:
                    raise ValueError(f"group indices overlap: {sorted(indices & seen)}")
                if not indices <= set(range(num_params)):
                    raise ValueError(f"group indices out of range [0, {num_params}): {block}")
                seen |= indices

        self.groups = [
            [torch.tensor(block, dtype=torch.long) for block in group] for group in groups
        ]
        self.fixed_indices = torch.tensor(sorted(set(range(num_params)) - seen), dtype=torch.long)
        self.num_params = num_params

    def update(self, predicted: torch.Tensor, target: torch.Tensor):
        """Accumulate the batch's permutation-optimal squared error.

        :param predicted: ``(batch, num_params)`` predictions.
        :param target: ``(batch, num_params)`` targets.
        :raises ValueError: If either tensor's width differs from ``num_params``.
        """
        if predicted.shape[-1] != self.num_params or target.shape[-1] != self.num_params:
            raise ValueError(
                f"expected width {self.num_params}, got predicted {predicted.shape[-1]} "
                f"/ target {target.shape[-1]}"
            )
        squared_error = (predicted - target).square()
        total = squared_error[:, self.fixed_indices.to(predicted.device)].sum()

        for group in self.groups:
            block_indices = [block.to(predicted.device) for block in group]
            predicted_blocks = torch.stack([predicted[:, b] for b in block_indices], dim=1)
            target_blocks = torch.stack([target[:, b] for b in block_indices], dim=1)
            cost = (
                (predicted_blocks.unsqueeze(2) - target_blocks.unsqueeze(1)).square().sum(dim=-1)
            )
            cost_cpu = cost.detach().cpu()
            for sample in range(cost_cpu.shape[0]):
                row_ind, col_ind = linear_sum_assignment(cost_cpu[sample])
                total = total + cost[sample, row_ind, col_ind].sum()

        self.sum_squared_error += total
        self.element_count += predicted.shape[0] * predicted.shape[1]

    def compute(self):
        """Mean permutation-optimal squared error over all accumulated elements.

        :returns: Scalar tensor, comparable to a plain elementwise MSE.
        """
        return self.sum_squared_error / self.element_count
