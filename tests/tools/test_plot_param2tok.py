"""Tests for ``plot_param2tok.get_labels``' encoded-column interval layout.

The returned ``(label, width)`` intervals annotate axes over an encoded parameter
row, so their widths must tile that row exactly — a short or long total silently
misaligns every label drawn after the gap.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pytest
import torch
from click.testing import CliRunner
from omegaconf import DictConfig, OmegaConf

from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.tools.plot_param2tok import get_labels, instantiate_model, main


def _tiny_model_config(projection: dict[str, object], num_params: int) -> DictConfig:
    return OmegaConf.create(
        {
            "model": {
                "_target_": "synth_setter.models.vst_flow_matching_module.VSTFlowMatchingModule",
                "encoder": {"_target_": "torch.nn.Identity"},
                "vector_field": {
                    "_target_": "synth_setter.models.components.transformer.ApproxEquivTransformer",
                    "projection": projection,
                    "num_layers": 1,
                    "d_model": 8,
                    "conditioning_dim": 4,
                    "num_heads": 1,
                    "d_ff": 8,
                    "learn_projection": True,
                    "d_enc": 4,
                },
                "optimizer": {
                    "_target_": "torch.optim.Adam",
                    "_partial_": True,
                    "lr": 0.001,
                },
                "scheduler": None,
                "num_params": num_params,
            }
        }
    )


@pytest.mark.parametrize("spec", ["surge_4", "surge_simple", "surge_xt", "obxf"])
def test_get_labels_intervals_tile_the_encoded_row_exactly(spec: str) -> None:
    """Interval widths sum to the spec's encoded width, covering every column.

    :param spec: Registered ParamSpec name under test.
    """
    intervals = get_labels(spec)

    assert sum(width for _, width in intervals) == param_specs[spec].encoded_width


def test_get_labels_orders_note_parameters_after_synth_parameters() -> None:
    """Note parameters land at the end, matching the encoding order."""
    labels = [label for label, _ in get_labels("surge_4")]

    assert labels[-2:] == ["Note Pitch", "Note On/Off"]


def test_get_labels_widths_are_positive() -> None:
    """No interval is empty, so every label annotates at least one column."""
    intervals = get_labels("surge_simple")

    assert all(width > 0 for _, width in intervals)


def test_main_with_incompatible_projection_rejects_before_plotting(tmp_path: Path) -> None:
    """The real Hydra/checkpoint entrypoint clearly rejects a non-learnt projection.

    :param tmp_path: Isolated log tree, checkpoint, and prospective plot directory.
    """
    config = _tiny_model_config(
        {
            "_target_": "synth_setter.models.components.transformer.GroupedParameterProjection",
            "d_model": 8,
            "param_spec_name": "surge_simple",
        },
        num_params=92,
    )
    model = hydra.utils.instantiate(config.model)
    run_dir = tmp_path / "logs" / "run"
    wandb_dir = run_dir / "wandb" / "run-test-id"
    wandb_dir.mkdir(parents=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    torch.save({"state_dict": model.state_dict()}, checkpoint_dir / "last.ckpt")
    hparams_dir = run_dir / "csv" / "version_0"
    hparams_dir.mkdir(parents=True)
    OmegaConf.save(config, hparams_dir / "hparams.yaml")
    output_dir = tmp_path / "plots"

    result = CliRunner().invoke(
        main,
        ["test-id", str(output_dir), "--log-dir", str(tmp_path / "logs"), "--device", "cpu"],
    )

    assert isinstance(result.exception, TypeError)
    assert "requires LearntProjection" in str(result.exception)
    assert not output_dir.exists()


def test_instantiate_model_with_incompatible_state_rejects_checkpoint(tmp_path: Path) -> None:
    """Strict loading rejects a checkpoint missing a learnt projection matrix.

    :param tmp_path: Isolated checkpoint location.
    """
    config = _tiny_model_config(
        {
            "_target_": "synth_setter.models.components.transformer.LearntProjection",
            "d_model": 8,
            "d_token": 8,
            "num_params": 3,
            "num_tokens": 2,
            "initial_ffn": False,
            "final_ffn": False,
        },
        num_params=3,
    )
    model = hydra.utils.instantiate(config.model)
    state_dict = model.state_dict()
    del state_dict["vector_field.projection._assignment"]
    checkpoint = tmp_path / "incompatible.ckpt"
    torch.save({"state_dict": state_dict}, checkpoint)

    with pytest.raises(RuntimeError, match="Missing key.*_assignment"):
        instantiate_model(config.model, checkpoint, map_location="cpu")
