"""Behavior tests for named CFG strength resolution."""

from synth_setter.cli._cfg_strength import CfgStrengths, apply_cfg_strength_overrides


def test_cfg_strengths_omitted_content_and_zero_sketch_preserve_independent_values() -> None:
    """An omitted content override and explicit sketch zero resolve independently."""
    hparams: dict[str, object] = {
        "test_cfg_strength": 4.0,
        "test_sketch_cfg_strength": 6.0,
    }

    effective = apply_cfg_strength_overrides(
        hparams,
        CfgStrengths(content=None, sketch=0.0),
    )

    assert effective == CfgStrengths(content=4.0, sketch=0.0)
    assert hparams == {
        "test_cfg_strength": 4.0,
        "test_sketch_cfg_strength": 0.0,
    }
