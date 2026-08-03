"""Behavior tests for shared text-conditioned Surge rendering."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import click
import numpy as np
import pytest
import torch

from synth_setter.cli import clap_render, surge_render
from synth_setter.conditioning import EmbeddingConditioningSpec
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule


def test_resolve_inverse_checkpoint_missing_local_path_raises(tmp_path: Path) -> None:
    """A misspelled local checkpoint fails instead of being treated as remote.

    :param tmp_path: Temporary absent checkpoint path.
    """
    with pytest.raises(FileNotFoundError, match="does not exist"):
        surge_render.resolve_inverse_checkpoint(str(tmp_path / "absent.ckpt"))


def test_resolve_device_unavailable_cuda_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit unavailable CUDA request fails rather than falling back.

    :param monkeypatch: CUDA availability override fixture.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(click.ClickException, match="CUDA was requested"):
        surge_render.resolve_device("cuda")


def test_resolve_device_unavailable_mps_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit unavailable MPS request fails rather than falling back.

    :param monkeypatch: Accelerator availability override fixture.
    """
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(click.ClickException, match="MPS was requested"):
        surge_render.resolve_device("mps")


def test_resolve_device_cpu_returns_cpu() -> None:
    """An explicit CPU request is preserved on accelerator hosts."""
    assert surge_render.resolve_device("cpu") == torch.device("cpu")


def test_validate_inverse_model_wrong_conditioning_raises() -> None:
    """A SAME latent cannot be consumed by a CLAP-conditioned checkpoint."""
    render = clap_render._load_settings().render
    model = cast(
        VSTFlowMatchingModule,
        SimpleNamespace(
            hparams={
                "conditioning": {"column": "clap", "input_shape": [512]},
                "sketch_controls": None,
                "num_params": len(param_specs[render.param_spec_name]),
            }
        ),
    )

    with pytest.raises(ValueError, match="same_s"):
        surge_render.validate_inverse_model(
            model,
            render,
            EmbeddingConditioningSpec(column="same_s", input_shape=(256, 44)),
        )


def test_validate_inverse_model_sketch_checkpoint_raises() -> None:
    """A text-only command rejects checkpoints requiring sketch controls."""
    render = clap_render._load_settings().render
    conditioning = EmbeddingConditioningSpec(column="clap", input_shape=(512,))
    model = cast(
        VSTFlowMatchingModule,
        SimpleNamespace(
            hparams={
                "conditioning": conditioning,
                "sketch_controls": {"num_frames": 32},
                "num_params": len(param_specs[render.param_spec_name]),
            }
        ),
    )

    with pytest.raises(ValueError, match="sketch-conditioned"):
        surge_render.validate_inverse_model(model, render, conditioning)


def test_validate_inverse_model_wrong_output_width_raises() -> None:
    """A checkpoint trained for another parameter spec cannot render silently."""
    render = clap_render._load_settings().render
    conditioning = EmbeddingConditioningSpec(column="clap", input_shape=(512,))
    model = cast(
        VSTFlowMatchingModule,
        SimpleNamespace(
            hparams={
                "conditioning": conditioning,
                "sketch_controls": None,
                "num_params": 1,
            }
        ),
    )

    with pytest.raises(ValueError, match="output width"):
        surge_render.validate_inverse_model(model, render, conditioning)


def test_workspace_render_config_relative_preset_becomes_absolute() -> None:
    """A packaged relative preset resolves against the operator workspace."""
    render = clap_render._load_settings().render

    resolved = surge_render.workspace_render_config(render)

    assert Path(resolved.plugin_state_path).is_absolute()


def test_validate_rendered_audio_nonfinite_raises() -> None:
    """A NaN waveform is rejected before persistence."""
    audio = np.zeros((2, 16), dtype=np.float32)
    audio[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        surge_render.validate_rendered_audio(audio)
