import torch


def divmod(a, b):  # noqa: ANN001
    return torch.div(a, b, rounding_mode="floor"), torch.remainder(a, b)
