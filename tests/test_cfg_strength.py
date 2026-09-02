"""Behavior tests for named CFG strength resolution."""

import pytest

from synth_setter.cli._cfg_strength import CfgStrengths, temporary_cfg_strength_overrides


def test_cfg_strengths_sequential_override_restores_checkpoint_defaults() -> None:
    """One explicit override cannot change a later omitted request."""
    hparams: dict[str, object] = {
        "test_cfg_strength": 4.0,
        "test_sketch_cfg_strength": 6.0,
    }

    with temporary_cfg_strength_overrides(
        hparams,
        CfgStrengths(content=0.0, sketch=0.0),
    ) as explicit:
        assert explicit == CfgStrengths(content=0.0, sketch=0.0)
        assert hparams == {
            "test_cfg_strength": 0.0,
            "test_sketch_cfg_strength": 0.0,
        }

    with temporary_cfg_strength_overrides(
        hparams,
        CfgStrengths(content=None, sketch=None),
    ) as omitted:
        assert omitted == CfgStrengths(content=4.0, sketch=6.0)


def test_cfg_strengths_exception_restores_missing_and_null_checkpoint_values() -> None:
    """Exceptional prediction restores both CFG fields exactly.

    :raises RuntimeError: Always, to exercise context-manager restoration.
    """
    hparams: dict[str, object] = {
        "test_cfg_strength": 4.0,
        "test_sketch_cfg_strength": None,
    }

    with pytest.raises(RuntimeError, match="prediction failed"):
        with temporary_cfg_strength_overrides(
            hparams,
            CfgStrengths(content=0.0, sketch=0.0),
        ):
            del hparams["test_sketch_cfg_strength"]
            raise RuntimeError("prediction failed")

    assert hparams == {
        "test_cfg_strength": 4.0,
        "test_sketch_cfg_strength": None,
    }
