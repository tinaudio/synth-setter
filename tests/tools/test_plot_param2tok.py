"""Tests for ``plot_param2tok.get_labels``' encoded-column interval layout.

The returned ``(label, width)`` intervals annotate axes over an encoded parameter
row, so their widths must tile that row exactly — a short or long total silently
misaligns every label drawn after the gap.
"""

from __future__ import annotations

import pytest

from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.tools.plot_param2tok import get_labels


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


def test_get_labels_for_a_kosc_spec_uses_the_oscillator_layout() -> None:
    """A ``k_<n>`` spec bypasses the registry and reports three per-oscillator bands."""
    assert get_labels("k_4") == [("Frequency", 4), ("Amplitude", 4), ("Waveform", 4)]
