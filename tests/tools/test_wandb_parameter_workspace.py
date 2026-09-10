"""Contract tests for the shared synth-parameter W&B workspace."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import wandb_workspaces.workspaces as ws
from click.testing import CliRunner

from synth_setter.tools import wandb_parameter_workspace
from synth_setter.tools.wandb_parameter_workspace import build_parameter_workspace


def test_parameter_workspace_includes_every_synth_metric_family() -> None:
    """The shared view discovers parameter panels without filtering synth runs."""
    workspace = build_parameter_workspace("team", "project")

    assert workspace.runset_settings.query == ""
    assert workspace.runset_settings.regex_query is False
    assert workspace.auto_generate_panels is False
    assert [section.name for section in workspace.sections] == [
        "Training parameter objectives",
        "Validation parameter errors",
    ]


def test_parameter_workspace_serializes_dynamic_metric_regexes() -> None:
    """The W&B payload keeps regex discovery enabled instead of fixed Surge keys."""
    workspace = build_parameter_workspace("team", "project")

    spec = workspace._spec["spec"]["section"]
    sections = spec["panelBankConfig"]["sections"]
    panel_configs = [panel["config"] for section in sections for panel in section["panels"]]

    assert [config["metricRegex"] for config in panel_configs] == [
        "^train/per_param_(flow|endpoint)_mse/.+$",
        "^val/per_param_mse/.+$",
        "^val[/_]per_param_mse_best_swap/.+$",
        "^val/number_group_optimal_assignment_mse/.+$",
        "^val/per_param_mse_spec_quantized/.+$",
        "^val/categorical_mismatch_rate/.+$",
        "^val/number_group_optimal_assignment_categorical_mismatch_rate/.+$",
        "^val/angular_mae_radians/.+$",
        "^val/axis_angular_error_radians/.+$",
        "^val/discrete_(mae|mismatch_rate)/.+$",
        "^val/note_timing_mae_seconds/.+$",
    ]
    assert all(config["useMetricRegex"] is True for config in panel_configs)
    assert spec["runSets"][0]["search"]["query"] == ""


def test_main_with_explicit_entity_builds_saves_workspace_and_prints_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI builds the real workspace before crossing the W&B save boundary.

    :param monkeypatch: Pytest fixture that isolates the W&B save boundary.
    """
    saved_workspaces: list[ws.Workspace] = []

    def save(workspace: ws.Workspace) -> SimpleNamespace:
        saved_workspaces.append(workspace)
        return SimpleNamespace(url="https://wandb.ai/team/project?nw=view")

    monkeypatch.setattr(wandb_parameter_workspace.ws.Workspace, "save", save)

    result = CliRunner().invoke(
        wandb_parameter_workspace.main,
        ["--entity", "team", "--project", "project"],
    )

    assert result.exit_code == 0
    assert result.output == "https://wandb.ai/team/project?nw=view\n"
    assert saved_workspaces[0].name == "Synth parameter metrics"


def test_main_without_configured_entity_reports_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing account identity fails before any workspace is created.

    :param monkeypatch: Pytest fixture that removes the default W&B entity.
    """
    api = MagicMock(default_entity=None)
    monkeypatch.setattr(wandb_parameter_workspace.wandb, "Api", MagicMock(return_value=api))

    result = CliRunner().invoke(wandb_parameter_workspace.main, ["--project", "project"])

    assert result.exit_code == 1
    assert "No W&B entity configured; pass --entity." in result.output
