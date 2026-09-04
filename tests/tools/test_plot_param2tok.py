"""Tests for ``plot_param2tok.get_labels``' encoded-column interval layout.

The returned ``(label, width)`` intervals annotate axes over an encoded parameter
row, so their widths must tile that row exactly — a short or long total silently
misaligns every label drawn after the gap.
"""

from __future__ import annotations

import numpy as np
import pytest

from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.tools.plot_param2tok import cosine_self_sim, get_labels


@pytest.mark.parametrize("spec", ["surge_4", "surge_simple", "surge_xt", "obxf"])
def test_get_labels_intervals_tile_the_encoded_row_exactly(spec: str) -> None:
    """Interval widths sum to the spec's encoded width, covering every column.

    :param spec: Registered ParamSpec name under test.
    """
    intervals = get_labels(spec)

    assert sum(width for _, width in intervals) == param_specs[spec].encoded_width


def test_get_labels_orders_note_parameters_after_synth_parameters() -> None:
    """Note parameters land at the end, matching the encoding order."""
    labels = [label for label, _ in get_labels("surge_4")]

    assert labels[-2:] == ["Note Pitch", "Note On/Off"]


def test_get_labels_widths_are_positive() -> None:
    """No interval is empty, so every label annotates at least one column."""
    intervals = get_labels("surge_simple")

    assert all(width > 0 for _, width in intervals)


def test_cosine_self_sim_with_different_norms_returns_symmetric_similarity() -> None:
    """Pairwise cosine similarity normalizes both vectors independently."""
    vectors = np.array([[2.0, 0.0], [1.0, 1.0]])

    similarity = cosine_self_sim(vectors)

    np.testing.assert_allclose(
        similarity,
        np.array([[1.0, 1 / np.sqrt(2)], [1 / np.sqrt(2), 1.0]]),
    )


def test_cosine_self_sim_with_zero_vector_returns_zero_similarity() -> None:
    """A zero projection has finite zero similarity with every vector."""
    vectors = np.array([[0.0, 0.0], [1.0, 0.0]])

    similarity = cosine_self_sim(vectors)

    np.testing.assert_array_equal(similarity, np.array([[0.0, 0.0], [0.0, 1.0]]))
