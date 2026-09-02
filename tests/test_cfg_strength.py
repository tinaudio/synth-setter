"""Behavior tests for model-owned CFG strength resolution."""

import pytest

from synth_setter.models.cfg import CfgStrengths, resolve_cfg_strengths


def test_cfg_strengths_explicit_zero_preserves_independent_values() -> None:
    """Zero is an explicit value for both guidance branches."""
    resolved = resolve_cfg_strengths(
        CfgStrengths(content=0.0, sketch=0.0),
        default_content=4.0,
        default_sketch=6.0,
    )

    assert resolved == CfgStrengths(content=0.0, sketch=0.0)


@pytest.mark.parametrize("default_sketch", [None, pytest.param("missing", id="missing")])
def test_cfg_strengths_legacy_sketch_default_uses_effective_content(
    default_sketch: float | str | None,
) -> None:
    """A missing or null legacy sketch scale follows overridden content.

    :param default_sketch: Legacy missing or null checkpoint representation.
    """
    kwargs = {} if default_sketch == "missing" else {"default_sketch": default_sketch}

    resolved = resolve_cfg_strengths(
        CfgStrengths(content=1.5, sketch=None),
        default_content=4.0,
        **kwargs,
    )

    assert resolved == CfgStrengths(content=1.5, sketch=1.5)


@pytest.mark.parametrize(
    ("requested", "default_content", "default_sketch", "message"),
    [
        (CfgStrengths(content=None, sketch=None), "bad", None, "content"),
        (CfgStrengths(content=None, sketch=None), float("nan"), None, "content"),
        (CfgStrengths(content=None, sketch=None), 4.0, "bad", "sketch"),
        (CfgStrengths(content=None, sketch=None), 4.0, -1.0, "sketch"),
        (CfgStrengths(content=-1.0, sketch=None), 4.0, None, "content"),
        (CfgStrengths(content=None, sketch=float("inf")), 4.0, None, "sketch"),
    ],
)
def test_cfg_strengths_invalid_value_raises(
    requested: CfgStrengths[float | None],
    default_content: object,
    default_sketch: object,
    message: str,
) -> None:
    """Invalid requested or checkpoint values fail at the model boundary.

    :param requested: Runtime guidance overrides.
    :param default_content: Saved content guidance value.
    :param default_sketch: Saved sketch guidance value.
    :param message: Invalid branch named by the error.
    """
    with pytest.raises(ValueError, match=message):
        resolve_cfg_strengths(
            requested,
            default_content=default_content,
            default_sketch=default_sketch,
        )
