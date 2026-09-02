"""Behavior tests for sketch-conditioned rendering."""

import pytest

from synth_setter.cli.sketch_render import cfg_grid


def test_cfg_grid_repeated_strengths_returns_argument_order_product() -> None:
    """Repeated strengths expand content-major into every requested arm."""
    assert cfg_grid([0.0, 2.0], [1.0, 3.0]) == (
        (0.0, 1.0),
        (0.0, 3.0),
        (2.0, 1.0),
        (2.0, 3.0),
    )


def test_cfg_grid_nonfinite_strength_raises() -> None:
    """A non-finite guidance strength cannot identify a valid arm."""
    with pytest.raises(ValueError, match="finite and non-negative"):
        cfg_grid([float("nan")], [1.0])
