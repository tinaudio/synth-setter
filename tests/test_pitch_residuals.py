"""Tests for signed MIDI-pitch residual diagnostics."""

import pytest
import torch

from synth_setter.data.vst.param_spec import (
    ContinuousParameter,
    DiscreteLiteralParameter,
    ParamSpec,
)
from synth_setter.metrics import midi_pitch_residuals


def test_midi_pitch_residuals_fractional_predictions_reports_decode_counterfactuals() -> None:
    """Residuals distinguish continuous, floor-decoded, and nearest-note predictions."""
    spec = ParamSpec(
        synth_params=[],
        note_params=[DiscreteLiteralParameter(name="pitch", min=48, max=72)],
    )
    predicted = torch.tensor([[-0.0208333333333333], [0.0625]])
    target = torch.tensor([[0.0], [0.0]])

    residuals = midi_pitch_residuals(predicted, target, spec)

    torch.testing.assert_close(residuals["continuous"], torch.tensor([-0.25, 0.75]))
    torch.testing.assert_close(residuals["floor"], torch.tensor([-1.0, 0.0]))
    torch.testing.assert_close(residuals["nearest"], torch.tensor([0.0, 1.0]))


def test_midi_pitch_residuals_pitch_between_other_coordinates_uses_spec_slice() -> None:
    """Residuals read the ParamSpec pitch slice rather than a fixed coordinate."""
    spec = ParamSpec(
        synth_params=[ContinuousParameter(name="before_pitch")],
        note_params=[
            DiscreteLiteralParameter(name="pitch", min=48, max=72),
            ContinuousParameter(name="after_pitch"),
        ],
    )
    predicted = torch.tensor([[1.0, -0.0208333333333333, -1.0]])
    target = torch.tensor([[-1.0, 0.0, 1.0]])

    residuals = midi_pitch_residuals(predicted, target, spec)

    torch.testing.assert_close(residuals["continuous"], torch.tensor([-0.25]))
    torch.testing.assert_close(residuals["floor"], torch.tensor([-1.0]))
    torch.testing.assert_close(residuals["nearest"], torch.tensor([0.0]))


def test_midi_pitch_residuals_out_of_range_predictions_clips_to_pitch_bounds() -> None:
    """Residuals use the same clipping boundary as production model decoding."""
    spec = ParamSpec(
        synth_params=[],
        note_params=[DiscreteLiteralParameter(name="pitch", min=48, max=72)],
    )
    predicted = torch.tensor([[-2.0], [2.0]])
    target = torch.tensor([[0.0], [0.0]])

    residuals = midi_pitch_residuals(predicted, target, spec)

    torch.testing.assert_close(residuals["continuous"], torch.tensor([-12.0, 12.0]))
    torch.testing.assert_close(residuals["floor"], torch.tensor([-12.0, 12.0]))
    torch.testing.assert_close(residuals["nearest"], torch.tensor([-12.0, 12.0]))


def test_midi_pitch_residuals_mismatched_batches_raises_value_error() -> None:
    """Residuals reject tensors whose row counts differ instead of broadcasting."""
    spec = ParamSpec(
        synth_params=[],
        note_params=[DiscreteLiteralParameter(name="pitch", min=48, max=72)],
    )

    with pytest.raises(ValueError, match="expected matching 2-D shapes"):
        midi_pitch_residuals(torch.zeros(2, 1), torch.zeros(1, 1), spec)


def test_midi_pitch_residuals_wrong_width_raises_value_error() -> None:
    """Residuals reject tensors that do not cover the selected ParamSpec."""
    spec = ParamSpec(
        synth_params=[DiscreteLiteralParameter(name="velocity", min=0, max=127)],
        note_params=[DiscreteLiteralParameter(name="pitch", min=48, max=72)],
    )
    values = torch.zeros(1, 1)

    with pytest.raises(ValueError, match="expected ParamSpec width 2"):
        midi_pitch_residuals(values, values, spec)


def test_midi_pitch_residuals_onehot_pitch_raises_value_error() -> None:
    """The scalar diagnostic rejects a one-hot pitch representation."""
    spec = ParamSpec(
        synth_params=[],
        note_params=[DiscreteLiteralParameter(name="pitch", min=48, max=72, encoding="onehot")],
    )
    predicted = torch.zeros(1, len(spec))

    with pytest.raises(ValueError, match="unique scalar discrete pitch"):
        midi_pitch_residuals(predicted, predicted, spec)


def test_midi_pitch_residuals_missing_pitch_raises_value_error() -> None:
    """The diagnostic fails clearly when the selected synth has no MIDI pitch coordinate."""
    spec = ParamSpec(
        synth_params=[],
        note_params=[DiscreteLiteralParameter(name="velocity", min=0, max=127)],
    )
    predicted = torch.zeros(1, len(spec))

    with pytest.raises(ValueError, match="unique scalar discrete pitch"):
        midi_pitch_residuals(predicted, predicted, spec)
