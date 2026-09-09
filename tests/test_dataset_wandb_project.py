"""Dataset command loggers keep production and explicit project routing separate."""

from pathlib import Path

import pytest
import wandb
from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from tests.helpers.wandb_offline import read_run_project


@pytest.mark.parametrize(
    ("config_name", "overrides"),
    [
        ("dataset", ["experiment=generate_dataset/smoke-shard"]),
        ("add_embeddings", []),
        ("finalize_dataset", []),
    ],
)
@pytest.mark.parametrize(
    ("project_override", "expected_project"),
    [(None, "synth-setter-generate-dataset"), ("custom-project", "custom-project")],
)
def test_dataset_logger_project_offline_run_uses_selected_project(
    config_name: str,
    overrides: list[str],
    project_override: str | None,
    expected_project: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real offline runs record the command default or explicit environment project.

    :param config_name: Dataset command configuration to compose.
    :param overrides: Required command-specific configuration selections.
    :param project_override: Optional environment override.
    :param expected_project: Project expected in the persisted W&B run.
    :param tmp_path: Isolated W&B output directory.
    :param monkeypatch: Isolates project selection from the operator environment.
    """
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    if project_override is not None:
        monkeypatch.setenv("WANDB_PROJECT", project_override)
    wandb.teardown()
    with initialize_config_module(config_module="synth_setter.configs", version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides)
    logger = instantiate(
        cfg.logger.wandb,
        save_dir=str(tmp_path),
        name="project-routing",
        offline=True,
    )
    try:
        logger.log_metrics({"routing_check": 1})
    finally:
        wandb.finish()
        wandb.teardown()
    (run_binary,) = tmp_path.glob("wandb/offline-run-*/run-*.wandb")
    assert read_run_project(run_binary) == expected_project
