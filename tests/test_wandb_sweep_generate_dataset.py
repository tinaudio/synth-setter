"""End-to-end test for the ``generate_dataset`` W&B grid sweep over render cadences.

Stops at ``spec_from_cfg`` so the cross-field validator
(``RenderConfig._always_on_requires_plugin_reload_once``) fires inside Hydra
compose — no R2 / SkyPilot / VST3 cost — while still catching regressions
in the ``${args_no_hyphens}`` → Hydra → ``RenderConfig`` round-trip.
"""

from __future__ import annotations

import itertools
import os
import warnings
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


def _expected_cells_from_yaml() -> frozenset[tuple[str, str]]:
    """Cartesian product of the cadence param values declared in the sweep YAML.

    Reading the grid out of the YAML rather than hardcoding it keeps this test from drifting
    silently when an operator edits the sweep config.

    :returns: Every (plugin_reload_cadence, gui_toggle_cadence) cell the sweep advertises to W&B.
    """
    cfg = yaml.safe_load(_SWEEP_YAML.read_text())
    params = cfg["parameters"]
    plug = params["render.plugin_reload_cadence"]["values"]
    gui = params["render.gui_toggle_cadence"]["values"]
    return frozenset(itertools.product(plug, gui))


_EXPECTED_CELLS = _expected_cells_from_yaml()


def _compose_dataset_cfg(
    plugin_reload_cadence: str, gui_toggle_cadence: str, paths_root: Path
) -> DictConfig:
    """Compose the smoke-shard dataset cfg with the swept cadence overrides.

    :param plugin_reload_cadence: One of ``{"once", "render"}`` (matches
        ``_PluginReloadCadence`` in ``pipeline/schemas/spec.py``).
    :param gui_toggle_cadence: One of ``{"never", "once", "render", "always_on"}``;
        ``"always_on"`` requires ``plugin_reload_cadence == "once"`` per the
        ``RenderConfig`` cross-field validator.
    :param paths_root: Pinned into ``cfg.paths.*`` so ``spec_from_cfg``'s
        ``resolve=True`` does not trip on ``${hydra:runtime.output_dir}``.

    :returns: Composed DictConfig ready for ``spec_from_cfg``.
    """
    GlobalHydra.instance().clear()
    try:
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
    finally:
        # Leave GlobalHydra clean so later tests that call initialize_*
        # without their own pre-clear do not hit "already initialized".
        GlobalHydra.instance().clear()


@pytest.mark.slow
@pytest.mark.network
@RunIf(wandb=True)
@pytest.mark.skipif(
    not os.environ.get("WANDB_API_KEY"),
    reason="WANDB_API_KEY not set — sweeps need the live W&B backend.",
)
def test_wandb_grid_sweep_threads_cadence_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live W&B grid sweep over the two render cadence params on smoke-shard.

    Registers a grid sweep on the configured entity/project (overridable via
    ``WANDB_ENTITY`` / ``WANDB_PROJECT``), runs the agent in-process via
    ``function=trial`` so each cell stays within this interpreter, and
    asserts every (plugin_reload_cadence, gui_toggle_cadence) cell declared
    in the sweep YAML threads cleanly through Hydra into the constructed
    ``DatasetSpec``.

    :param tmp_path: Per-test scratch dir handed to ``_compose_dataset_cfg``
        so ``cfg.paths.*`` resolves without leaking into the operator
        workspace.
    :param monkeypatch: Used to scrub ``WANDB_MODE`` from the environment so
        a leaked ``offline`` value cannot silently neuter ``wandb.sweep`` /
        ``wandb.agent`` (sweeps require the live backend).
    """
    # Local import: wandb is optional at install time, gated by RunIf(wandb=True).
    import wandb

    from synth_setter.cli.generate_dataset import spec_from_cfg

    monkeypatch.delenv("WANDB_MODE", raising=False)
    # Pin W&B's per-run output dir under tmp_path so wandb.init() does not
    # drop a `./wandb/` directory into the repo checkout for local runs.
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))

    sweep_cfg = yaml.safe_load(_SWEEP_YAML.read_text())
    # `or` (not the dict default) — CI templating sometimes exports an
    # empty WANDB_ENTITY / WANDB_PROJECT, and "" would silently shadow the
    # YAML's hardcoded values and crash inside wandb.sweep().
    entity = os.environ.get("WANDB_ENTITY") or sweep_cfg.get("entity")
    project = os.environ.get("WANDB_PROJECT") or sweep_cfg.get("project", "synth-setter")

    observed: set[tuple[str, str]] = set()

    def trial() -> None:
        with wandb.init(mode="online") as run:
            plug = run.config["render.plugin_reload_cadence"]
            gui = run.config["render.gui_toggle_cadence"]
            cfg = _compose_dataset_cfg(plug, gui, tmp_path)
            spec = spec_from_cfg(cfg)
            cell = (spec.render.plugin_reload_cadence, spec.render.gui_toggle_cadence)
            assert cell in _EXPECTED_CELLS, f"unexpected sweep cell {cell!r}"
            run.log(
                {
                    "plugin_reload_cadence_observed": cell[0],
                    "gui_toggle_cadence_observed": cell[1],
                }
            )
            observed.add(cell)

    sweep_id = wandb.sweep(sweep_cfg, entity=entity, project=project)
    try:
        wandb.agent(
            sweep_id, function=trial, entity=entity, project=project, count=len(_EXPECTED_CELLS)
        )
    finally:
        # Stop the sweep on the server so repeated CI runs do not leak
        # half-grids into the W&B project history.
        try:
            wandb.Api().sweep(f"{entity}/{project}/{sweep_id}").stop()
        except Exception as cleanup_exc:  # noqa: BLE001 — best-effort cleanup
            warnings.warn(f"wandb sweep cleanup failed: {cleanup_exc!r}", stacklevel=1)

    missing = sorted(_EXPECTED_CELLS - observed)
    extra = sorted(observed - _EXPECTED_CELLS)
    assert observed == _EXPECTED_CELLS, (
        f"Sweep agent did not observe the full grid: missing={missing}, extra={extra}"
    )
