"""Math utility helpers used across the project."""

import torch


def divmod(a: torch.Tensor, b: int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the floor quotient and remainder of ``a`` divided by ``b``.

    :param a: Dividend tensor.
    :param b: Divisor. Either a tensor broadcastable against ``a`` or a Python
        ``int`` scalar (``torch.div``/``torch.remainder`` accept either).
    :returns: A ``(floor_quotient, remainder)`` tuple, where ``floor_quotient``
        is ``torch.div(a, b, rounding_mode="floor")`` and ``remainder`` is
        ``torch.remainder(a, b)``.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    return torch.div(a, b, rounding_mode="floor"), torch.remainder(a, b)
