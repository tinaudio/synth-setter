"""Shared inference-batch policy for offline embedding encoders."""


def resolve_encode_batch_size(batch_size: int, rows: int) -> int:
    """Resolve ``-1`` to the complete non-empty encoder input.

    :param batch_size: Positive row count or ``-1``.
    :param rows: Rows in the current encoder input.
    :returns: Positive rows per encoder call.
    :raises ValueError: The batch size is invalid or the encoder input is empty.
    """
    if rows < 1:
        raise ValueError("embedding encoders require a non-empty batch")
    if batch_size == -1:
        return rows
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive or -1, got {batch_size}")
    return batch_size
