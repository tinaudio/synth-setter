"""Shared inference-batch policy for offline embedding encoders."""


def resolve_encode_batch_size(max_batch_size: int, rows: int) -> int:
    """Resolve ``-1`` to the complete non-empty encoder input.

    :param max_batch_size: Positive row cap or ``-1``.
    :param rows: Rows in the current encoder input.
    :returns: Positive rows per encoder call.
    :raises ValueError: The cap is invalid or the encoder input is empty.
    """
    if rows < 1:
        raise ValueError("embedding encoders require a non-empty batch")
    if max_batch_size == -1:
        return rows
    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be positive or -1, got {max_batch_size}")
    return max_batch_size
