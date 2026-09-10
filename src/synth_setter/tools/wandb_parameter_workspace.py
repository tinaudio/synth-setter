"""Provision the shared synth-parameter W&B workspace."""

from __future__ import annotations

import click
import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

_WORKSPACE_NAME = "Synth parameter metrics"


def build_parameter_workspace(entity: str, project: str) -> ws.Workspace:
    """Build a synth-neutral workspace that discovers parameter metrics dynamically.

    :param entity: W&B entity that owns the project.
    :param project: W&B project containing training runs.
    :returns: Unsaved workspace definition.
    """
    return ws.Workspace(
        entity=entity,
        project=project,
        name=_WORKSPACE_NAME,
        auto_generate_panels=False,
        sections=[
            ws.Section(
                name="Training parameter objectives",
                is_open=True,
                panels=[
                    wr.LinePlot(
                        title="Flow and endpoint objective by parameter",
                        metric_regex=r"^train/per_param_(flow|endpoint)_mse/.+$",
                    )
                ],
            ),
            ws.Section(
                name="Validation parameter errors",
                is_open=True,
                panels=[
                    wr.LinePlot(
                        title="Sampled endpoint MSE by parameter",
                        metric_regex=r"^val/per_param_mse/.+$",
                    ),
                    wr.LinePlot(
                        title="Best-swap MSE by parameter",
                        metric_regex=r"^val[/_]per_param_mse_best_swap/.+$",
                    ),
                    wr.LinePlot(
                        title="Number-group optimal-assignment MSE",
                        metric_regex=r"^val/number_group_optimal_assignment_mse/.+$",
                    ),
                    wr.LinePlot(
                        title="Spec-quantized MSE by parameter",
                        metric_regex=r"^val/per_param_mse_spec_quantized/.+$",
                    ),
                    wr.LinePlot(
                        title="Categorical mismatch rate by parameter",
                        metric_regex=r"^val/categorical_mismatch_rate/.+$",
                    ),
                    wr.LinePlot(
                        title="Number-group categorical mismatch rate",
                        metric_regex=(
                            r"^val/number_group_optimal_assignment_"
                            r"categorical_mismatch_rate/.+$"
                        ),
                    ),
                ],
            ),
        ],
        runset_settings=ws.RunsetSettings(query="", regex_query=False),
    )


@click.command()
@click.option("--entity", help="W&B entity; defaults to the authenticated user's entity.")
@click.option("--project", default="synth-setter", show_default=True)
def main(entity: str | None, project: str) -> None:
    """Create the shared synth-parameter workspace and print its URL.

    :param entity: W&B entity override.
    :param project: W&B project containing training runs.
    :raises click.ClickException: If no entity is passed or configured for the W&B account.
    """
    resolved_entity = entity or wandb.Api().default_entity
    if resolved_entity is None:
        raise click.ClickException("No W&B entity configured; pass --entity.")
    workspace = build_parameter_workspace(resolved_entity, project).save()
    click.echo(workspace.url)


if __name__ == "__main__":
    main()
