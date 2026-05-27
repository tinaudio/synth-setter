"""End-to-end test for the ``generate_dataset`` W&B grid sweep over render cadences.

Exercises the ``sweeps/generate_dataset_cadence.yaml`` plumbing on the live
W&B backend: registers the sweep, then drives an in-process agent that
threads each swept ``render.plugin_reload_cadence`` / ``render.gui_toggle_cadence``
cell through the same Hydra compose path the CLI uses, and confirms the
constructed ``DatasetSpec`` carries the swept values.

The trial body deliberately stops at ``spec_from_cfg`` — the cadence
validator (``RenderConfig._always_on_requires_plugin_reload_once``) fires
inside compose, before any rendering side effect, so this test costs nothing
on R2 / SkyPilot / VST3 while still catching regressions in the
``${args_no_hyphens}`` → Hydra → ``RenderConfig`` round-trip.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict

from tests.helpers.run_if import RunIf

if TYPE_CHECKING:
    from omegaconf import DictConfig

_SWEEP_YAML = Path(__file__).resolve().parent.parent / "sweeps" / "generate_dataset_cadence.yaml"

# 2x2 grid — every cell satisfies the RenderConfig cross-field validator on
# Linux (where CI runs); the "always_on + render" rejection cell is covered
# by the render-matrix workflow's contract job, not by this test.
_EXPECTED_CELLS: frozenset[tuple[str, str]] = frozenset(
    {
        ("once", "never"),
        ("once", "once"),
        ("render", "never"),
        ("render", "once"),
    }
)


def _compose_dataset_cfg(
    plugin_reload_cadence: str, gui_toggle_cadence: str, paths_root: Path
) -> DictConfig:
    """Compose the smoke-shard dataset cfg with the swept cadence overrides.

    :param plugin_reload_cadence: Value the sweep agent selected for
        ``render.plugin_reload_cadence``.
    :param gui_toggle_cadence: Value the sweep agent selected for
        ``render.gui_toggle_cadence``.
    :param paths_root: Per-test scratch dir; pinned into ``cfg.paths.*`` so
        ``spec_from_cfg``'s ``resolve=True`` does not trip on the unresolved
        ``${hydra:runtime.output_dir}`` interpolation.

    :returns: Composed DictConfig ready for ``spec_from_cfg``.
    """
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="dataset",
            overrides=[
                "experiment=generate_dataset/smoke-shard",
                f"render.plugin_reload_cadence={plugin_reload_cadence}",
                f"render.gui_toggle_cadence={gui_toggle_cadence}",
            ],
        )
    with open_dict(cfg):
        cfg.paths.root_dir = str(paths_root)
        cfg.paths.output_dir = str(paths_root)
        cfg.paths.work_dir = str(paths_root)
    return cfg


@pytest.mark.slow
@pytest.mark.network
@RunIf(wandb=True)
@pytest.mark.skipif(
    not os.environ.get("WANDB_API_KEY"),
    reason="WANDB_API_KEY not set — wandb sweep requires the live backend (offline mode is unsupported for sweeps).",
)
def test_wandb_grid_sweep_threads_cadence_overrides(tmp_path: Path) -> None:
    """Live W&B grid sweep over the two render cadence params on smoke-shard.

    Registers a grid sweep on the configured entity/project (overridable via
    ``WANDB_ENTITY`` / ``WANDB_PROJECT``), runs the agent in-process via
    ``function=trial`` so each cell stays within this interpreter, and
    asserts every (plugin_reload_cadence, gui_toggle_cadence) cell threads
    cleanly through Hydra into the constructed ``DatasetSpec``.

    :param tmp_path: Per-test scratch dir handed to ``_compose_dataset_cfg``
        so ``cfg.paths.*`` resolves without leaking into the operator
        workspace.
    """
    import wandb

    from synth_setter.cli.generate_dataset import spec_from_cfg

    sweep_cfg = yaml.safe_load(_SWEEP_YAML.read_text())
    entity = os.environ.get("WANDB_ENTITY", sweep_cfg.get("entity"))
    project = os.environ.get("WANDB_PROJECT", sweep_cfg.get("project", "synth-setter"))

    observed: set[tuple[str, str]] = set()

    def trial() -> None:
        # Force online mode — sweeps require the live backend, so silently
        # honoring a leaked WANDB_MODE=offline from another test would mask
        # a real failure as a green run.
        with wandb.init(mode="online") as run:
            plugin_reload_cadence = run.config["render.plugin_reload_cadence"]
            gui_toggle_cadence = run.config["render.gui_toggle_cadence"]
            cfg = _compose_dataset_cfg(plugin_reload_cadence, gui_toggle_cadence, tmp_path)
            spec = spec_from_cfg(cfg)
            assert spec.render.plugin_reload_cadence == plugin_reload_cadence
            assert spec.render.gui_toggle_cadence == gui_toggle_cadence
            run.log(
                {
                    "plugin_reload_cadence_observed": spec.render.plugin_reload_cadence,
                    "gui_toggle_cadence_observed": spec.render.gui_toggle_cadence,
                    "render.samples_per_shard": spec.render.samples_per_shard,
                }
            )
            observed.add((spec.render.plugin_reload_cadence, spec.render.gui_toggle_cadence))

    sweep_id = wandb.sweep(sweep_cfg, entity=entity, project=project)
    wandb.agent(
        sweep_id, function=trial, entity=entity, project=project, count=len(_EXPECTED_CELLS)
    )

    assert observed == _EXPECTED_CELLS, (
        f"Sweep agent did not observe the full grid: missing "
        f"{_EXPECTED_CELLS - observed}, extra {observed - _EXPECTED_CELLS}"
    )
