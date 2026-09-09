"""Contract tests for the shared synth-parameter W&B workspace."""

from __future__ import annotations

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
        "^val/per_param_mse_best_swap/.+$",
        "^val/per_param_mse_number_group_swap/.+$",
        "^val/per_param_mse_spec_quantized/.+$",
    ]
    assert all(config["useMetricRegex"] is True for config in panel_configs)
    assert spec["runSets"][0]["search"]["query"] == ""
