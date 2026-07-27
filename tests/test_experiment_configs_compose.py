"""Every shipped experiment config must compose against its entrypoint.

A config that no longer composes is unreachable: nothing can select it, and the
failure only surfaces when someone tries to launch it. Eleven configs went that
way unnoticed when their model group was dropped during a directory restructure,
so the whole ``experiment/`` tree is swept here rather than spot-checked.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra

import synth_setter.configs

if TYPE_CHECKING:
    from _pytest.mark import ParameterSet

_CONFIG_MODULE = "synth_setter.configs"
_EXPERIMENT_ROOT = Path(synth_setter.configs.__file__).parent / "experiment"

# Dataset-generation experiments target the generate entrypoint, whose top-level
# config carries no model or trainer group.
_DATASET_ENTRYPOINT_PREFIX = "generate_dataset/"

# Bases carry a synth's shared trainer/callback wiring and deliberately leave the
# model to the concrete experiments that inherit them; supplying one proves the
# base itself is intact.
_REQUIRES_MODEL = frozenset(
    {
        "fm/base",
        "kosc/base",
        "ksin/base",
        "ksin_ood/base",
        "surge/base",
    }
)

# Algorithm overlays select a training scheme only, so they need both groups.
_REQUIRES_MODEL_AND_DATAMODULE = frozenset(
    {
        "fm/algorithm/conditional",
        "fm/algorithm/hierarchical",
        "fm/algorithm/mixed",
    }
)

# Refs https://github.com/tinaudio/synth-setter/issues/2572: these reference
# model/ksin_flow and model/ksin_ff, deleted in a4dfabbe7. Strict xfail, so
# restoring those configs fails this test until the entries are removed.
_MISSING_KSIN_MODEL = frozenset(
    {
        "flow_size/base",
        "flow_size/bigenc",
        "flow_size/medenc",
        "flow_size/smallenc",
        "flow_size/tinyenc",
        "flow_size/vbigenc",
        "ksin_ood/flow",
        "ksin_ood/mlp_chamfer",
        "ksin_ood/mlp_mse",
        "ksin_ood/mlp_sort",
        "time_weighting",
    }
)


def _experiment_names() -> list[str]:
    """Collect every shipped experiment config by its Hydra selector.

    :returns: Sorted ``experiment=`` values, e.g. ``surge/ffn_smoke``.
    """
    return sorted(
        str(path.relative_to(_EXPERIMENT_ROOT).with_suffix("")).replace("\\", "/")
        for path in _EXPERIMENT_ROOT.rglob("*.yaml")
    )


def _overrides_for(experiment: str) -> tuple[str, list[str]]:
    """Pair one experiment with the entrypoint and groups it needs.

    :param experiment: Hydra ``experiment=`` selector.
    :returns: Top-level config name and the overrides to compose it with.
    """
    overrides = [f"experiment={experiment}"]
    if experiment.startswith(_DATASET_ENTRYPOINT_PREFIX):
        return "dataset.yaml", overrides
    if experiment in _REQUIRES_MODEL_AND_DATAMODULE:
        return "train.yaml", [*overrides, "model=ffn", "datamodule=fm"]
    if experiment in _REQUIRES_MODEL:
        return "train.yaml", [*overrides, "model=ffn"]
    return "train.yaml", overrides


def _experiment_params() -> list[ParameterSet]:
    """Build one parameter per experiment, xfailing the known-unreachable set.

    :returns: Parameter sets covering every shipped experiment config.
    """
    params = []
    for name in _experiment_names():
        marks = (
            [pytest.mark.xfail(strict=True, reason="model/ksin_* deleted in a4dfabbe7 (#2572)")]
            if name in _MISSING_KSIN_MODEL
            else []
        )
        params.append(pytest.param(name, marks=marks, id=name))
    return params


@pytest.mark.parametrize("experiment", _experiment_params())
def test_shipped_experiment_config_composes(experiment: str) -> None:
    """A shipped experiment config resolves against its entrypoint.

    :param experiment: Hydra ``experiment=`` selector under test.
    """
    config_name, overrides = _overrides_for(experiment)
    GlobalHydra.instance().clear()
    try:
        with initialize_config_module(version_base="1.3", config_module=_CONFIG_MODULE):
            cfg = compose(config_name=config_name, return_hydra_config=False, overrides=overrides)
    finally:
        GlobalHydra.instance().clear()

    assert cfg is not None


def test_experiment_sweep_covers_every_shipped_config() -> None:
    """The sweep reads the config tree, so a new experiment is covered on sight."""
    discovered = _experiment_names()

    assert discovered, "no experiment configs discovered — the sweep would pass vacuously"
    assert len(discovered) == len(set(discovered))


def test_special_case_sets_name_only_shipped_configs() -> None:
    """Every hand-listed exception still exists, so stale entries cannot hide rot."""
    shipped = set(_experiment_names())
    listed = _REQUIRES_MODEL | _REQUIRES_MODEL_AND_DATAMODULE | _MISSING_KSIN_MODEL

    assert listed <= shipped, f"listed configs no longer shipped: {sorted(listed - shipped)}"
