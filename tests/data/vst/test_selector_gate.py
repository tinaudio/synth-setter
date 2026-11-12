"""Pin the ParamSpec selector gate and its static selector-value to active-column table."""

from __future__ import annotations

import numpy as np
import pytest

from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    ParamSpec,
    SelectorGate,
)


def _synth_params() -> list:
    """Return a selector (cols 0-2), gated ``slot1``/``slot2``/``mode`` and ungated ``shared``.

    :returns: Synth parameters encoding to eight columns.
    """
    return [
        CategoricalParameter(name="fx_type", values=["A", "B", "C"], encoding="onehot"),
        ContinuousParameter(name="slot1", min=0.0, max=1.0),
        ContinuousParameter(name="slot2", min=0.0, max=1.0),
        CategoricalParameter(name="mode", values=["x", "y"], encoding="onehot"),
        ContinuousParameter(name="shared", min=0.0, max=1.0),
    ]


_GATE = SelectorGate(
    selector="fx_type",
    blocks={"A": ("slot1", "slot2"), "B": ("slot1",), "C": ("mode",)},
)


def test_active_coordinate_table_gated_spec_marks_selected_block_and_ungated_columns() -> None:
    """Each selector value activates its own block, the selector, and every ungated column."""
    spec = ParamSpec(_synth_params(), [], gate=_GATE)

    selector_columns, table = spec.active_coordinate_table()

    assert selector_columns == slice(0, 3)
    # Columns: fx_type x3, slot1, slot2, mode x2, shared.
    expected = np.array(
        [
            [1, 1, 1, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 1, 1, 1],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(table, expected)


def test_active_coordinate_table_without_gate_raises() -> None:
    """A spec that declares no gate has no routing to build a table from."""
    spec = ParamSpec(_synth_params(), [])

    with pytest.raises(ValueError, match="no selector gate"):
        spec.active_coordinate_table()


@pytest.mark.parametrize(
    ("gate", "message"),
    [
        (SelectorGate(selector="missing", blocks={}), "one-hot CategoricalParameter"),
        (SelectorGate(selector="slot1", blocks={}), "one-hot CategoricalParameter"),
        (
            SelectorGate(selector="fx_type", blocks={"A": (), "B": ()}),
            "every value of 'fx_type'",
        ),
        (
            SelectorGate(selector="fx_type", blocks={"A": (), "B": (), "C": (), "D": ()}),
            "every value of 'fx_type'",
        ),
        (
            SelectorGate(selector="fx_type", blocks={"A": ("nope",), "B": (), "C": ()}),
            "'nope'",
        ),
        (
            SelectorGate(selector="fx_type", blocks={"A": ("fx_type",), "B": (), "C": ()}),
            "'fx_type'",
        ),
    ],
)
def test_param_spec_invalid_gate_raises(gate: SelectorGate, message: str) -> None:
    """A gate that does not resolve against the synth params fails at construction.

    :param gate: Malformed gate.
    :param message: Fragment the error must name.
    """
    with pytest.raises(ValueError, match=message):
        ParamSpec(_synth_params(), [], gate=gate)
