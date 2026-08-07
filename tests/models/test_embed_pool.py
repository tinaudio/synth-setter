"""Sinusoidal positional-encoding construction for pooled sequence heads."""

import pytest
import torch

from synth_setter.models.components.embed_pool import make_sin_pos_enc


@pytest.mark.parametrize("d_enc", [8, 7], ids=["even-width", "odd-width"])
def test_make_sin_pos_enc_at_any_width_pairs_each_cosine_column_with_its_sine(
    d_enc: int,
) -> None:
    """Adjacent sin/cos columns share a frequency, so each pair is a unit vector.

    An odd width leaves the final sine column without a cosine partner rather than shifting the
    pairing, which would retune every band below it.

    :param d_enc: Encoding width under test.
    """
    pe = make_sin_pos_enc(6, d_enc)

    assert pe.shape == (1, 6, d_enc)
    paired = 2 * (d_enc // 2)
    pairs = pe[0, :, :paired].reshape(6, d_enc // 2, 2)
    torch.testing.assert_close(pairs.square().sum(dim=-1), torch.ones(6, d_enc // 2))


def test_make_sin_pos_enc_at_odd_width_still_populates_the_unpaired_final_column() -> None:
    """The top band keeps its sine rather than being dropped for lacking a cosine."""
    pe = make_sin_pos_enc(6, 7)

    assert not torch.allclose(pe[0, 1:, 6], torch.zeros(5))
