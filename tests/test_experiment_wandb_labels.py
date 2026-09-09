"""Experiment metadata reaches real W&B runs without changing their identities."""

from pathlib import Path

import pytest
import wandb
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from wandb.sdk.lib.service.service_token import WandbServiceConnectionError


@pytest.mark.xfail(
    raises=WandbServiceConnectionError,
    reason="#2564: shared offline W&B service sockets can disappear during the full suite",
    strict=False,
)
@pytest.mark.parametrize(
    ("config_name", "experiment", "expected_name", "expected_tags"),
    [
        ("train", "pyfdn/flow", "pyfdn-flow", {"pyfdn", "flow"}),
        (
            "train",
            "pyfdn/flow_ast_online_sketch",
            "pyfdn-flow-ast-online-sketch",
            {"pyfdn", "flow", "ast-online", "sketch"},
        ),
        ("train", "torchsynth/ffn", "torchsynth-ffn", {"torchsynth", "ffn"}),
        (
            "train",
            "torchsynth/flow_audio_same",
            "torchsynth-flow_audio_same",
            {"torchsynth", "flow", "audio_loss", "same"},
        ),
        (
            "train",
            "surge/flow_simple",
            "surge-simple-onehot_flow",
            {"surge", "surge-simple-onehot", "flow"},
        ),
        (
            "train",
            "surge/flow_full_nofinalffn",
            "surge-full-onehot_flow-nofinalffn",
            {"surge", "surge-full-onehot", "flow-nofinalffn"},
        ),
        (
            "dataset",
            "generate_dataset/smoke-shard",
            "generate-dataset-smoke-shard",
            {"generate_dataset", "smoke-shard"},
        ),
    ],
)
def test_experiment_labels_offline_run_identifies_synth_and_variant(
    config_name: str,
    experiment: str,
    expected_name: str,
    expected_tags: set[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W&B receives descriptive names and filterable experiment tags.

    :param config_name: Command whose configuration is composed.
    :param experiment: Shipped experiment selected by the operator.
    :param expected_name: Human-readable run display name.
    :param expected_tags: Tags required to distinguish the selected experiment.
    :param tmp_path: Isolated offline run output.
    :param monkeypatch: Forces offline W&B without operator run identity settings.
    """
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    wandb.teardown()
    with initialize_config_module(config_module="synth_setter.configs", version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=[f"experiment={experiment}"])
    logger = instantiate(cfg.logger.wandb, save_dir=str(tmp_path), offline=True)
    try:
        logger.log_metrics({"labels_check": 1})
        run = logger.experiment
        assert run.name == expected_name
        assert expected_tags <= set(run.tags)
    finally:
        wandb.finish()
        wandb.teardown()


@pytest.mark.parametrize("logger_selection", ["[]", "csv"])
def test_experiment_labels_other_logger_selection_does_not_enable_wandb(
    logger_selection: str,
) -> None:
    """Metadata defaults do not create a W&B logger when another logger is selected.

    :param logger_selection: Non-W&B logger selection.
    """
    with initialize_config_module(config_module="synth_setter.configs", version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=["experiment=pyfdn/flow", f"logger={logger_selection}"],
        )
    assert "wandb" not in cfg.get("logger", {})


def test_experiment_labels_explicit_overrides_reach_offline_run(tmp_path: Path) -> None:
    """Operator-selected labels take precedence over the descriptive defaults.

    :param tmp_path: Isolated offline run output.
    """
    wandb.teardown()
    with initialize_config_module(config_module="synth_setter.configs", version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=pyfdn/flow",
                "logger.wandb.name=custom-name",
                "tags=[custom-tag]",
            ],
        )
    logger = instantiate(cfg.logger.wandb, save_dir=str(tmp_path), offline=True)
    try:
        logger.log_metrics({"labels_check": 1})
        assert logger.experiment.name == "custom-name"
        assert logger.experiment.tags == ("custom-tag",)
    finally:
        wandb.finish()
        wandb.teardown()
