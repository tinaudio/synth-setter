"""ONNX graphs must preserve the production flow field and conditioning."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from hydra.utils import instantiate

from synth_setter.models.components.transformer import (
    ApproxEquivTransformer,
    LearntProjection,
)
from synth_setter.models.flow_onnx import export_flow_onnx
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule


@pytest.fixture
def flow_model() -> VSTFlowMatchingModule:
    """Return a small instance of the production mel/sketch architecture.

    :returns: Evaluation model with nonzero sketch-control projections.
    """
    torch.manual_seed(17)
    model = instantiate(
        {"_target_": "synth_setter.models.vst_flow_matching_module.VSTFlowMatchingModule"},
        encoder=instantiate(
            {"_target_": "synth_setter.models.components.transformer.AudioSpectrogramTransformer"},
            d_model=16,
            n_heads=2,
            n_layers=1,
            n_conditioning_outputs=1,
            patch_size=4,
            patch_stride=2,
            spec_shape=(8, 8),
        ),
        vector_field=ApproxEquivTransformer(
            projection=LearntProjection(d_model=16, d_token=16, num_params=92, num_tokens=4),
            d_model=16,
            conditioning_dim=16,
            num_heads=2,
            num_layers=1,
            d_ff=16,
            zero_init=False,
        ),
        optimizer=torch.optim.Adam,
        scheduler=None,
        num_params=92,
        param_spec="surge_simple",
        sketch_controls={"num_frames": 4, "num_control_tokens": 4},
    ).eval()
    assert isinstance(model, VSTFlowMatchingModule)
    assert model.sketch_tokens is not None
    for projection in model.sketch_tokens.projections.values():
        assert isinstance(projection, torch.nn.Linear)
        torch.nn.init.normal_(projection.weight, std=0.01)
    return model


@pytest.mark.parametrize("strengths", [(0.0, 0.0), (2.0, 3.0)])
def test_exported_velocity_same_inputs_matches_production(
    tmp_path: Path, flow_model: VSTFlowMatchingModule, strengths: tuple[float, float]
) -> None:
    """Exported conditioning and guidance preserve native inference values.

    :param tmp_path: Export directory.
    :param flow_model: Small production architecture with nonzero sketch projections.
    :param strengths: Independent content and sketch guidance strengths.
    """
    assert flow_model.sketch_tokens is not None
    batch = {
        "mel": torch.randn(1, 2, 8, 8),
        "sketch_ctrl": torch.rand(1, flow_model.sketch_tokens.layout.num_controls, 4),
    }
    export_flow_onnx(flow_model, batch, tmp_path)
    encoder = ort.InferenceSession(str(tmp_path / "conditioning.onnx"))
    encoded = encoder.run(None, {key: value.numpy() for key, value in batch.items()})
    x = torch.randn(1, 92)
    t = torch.tensor([[0.3]])
    field = ort.InferenceSession(str(tmp_path / "velocity.onnx"))
    actual = field.run(
        None,
        {
            "x": x.numpy(),
            "t": t.numpy(),
            "conditioning": encoded[0],
            "controls": encoded[1],
            "null_controls": encoded[2],
            "guidance": np.array(strengths, dtype=np.float32),
        },
    )[0]
    with torch.no_grad():
        expected = flow_model._velocity_field(
            flow_model.encoder(batch["mel"]),
            strengths[0],
            flow_model._control_token_branches_from_batch({"sketch_ctrl": batch["sketch_ctrl"]}),
            sketch_cfg_strength=strengths[1],
        )(x, t)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == expected.shape
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_export_nonfinite_input_rejected_before_writing(
    tmp_path: Path, flow_model: VSTFlowMatchingModule, value: float
) -> None:
    """Malformed inputs must not publish a usable-looking artifact.

    :param tmp_path: Export destination.
    :param flow_model: Supported production architecture.
    :param value: Nonfinite input value.
    """
    assert flow_model.sketch_tokens is not None
    batch = {
        "mel": torch.full((1, 2, 8, 8), value),
        "sketch_ctrl": torch.zeros(1, flow_model.sketch_tokens.layout.num_controls, 4),
    }
    with pytest.raises(ValueError, match="finite CPU float32"):
        export_flow_onnx(flow_model, batch, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("missing_key", ["mel", "sketch_ctrl"])
def test_export_missing_conditioning_input_rejected_before_writing(
    tmp_path: Path, flow_model: VSTFlowMatchingModule, missing_key: str
) -> None:
    """Incomplete conditioning must fail without creating an export destination.

    :param tmp_path: Parent of the absent export destination.
    :param flow_model: CPU evaluation architecture requiring both conditioning inputs.
    :param missing_key: Required conditioning input removed from the batch.
    """
    assert flow_model.sketch_tokens is not None
    batch = {
        "mel": torch.zeros(1, 2, 8, 8),
        "sketch_ctrl": torch.zeros(1, flow_model.sketch_tokens.layout.num_controls, 4),
    }
    batch.pop(missing_key)
    destination = tmp_path / "bundle"
    with pytest.raises(ValueError, match="mel and sketch_ctrl"):
        export_flow_onnx(flow_model, batch, destination)
    assert not destination.exists()


def test_export_nonempty_destination_preserves_existing_artifacts(
    tmp_path: Path, flow_model: VSTFlowMatchingModule
) -> None:
    """Refuse overwriting an existing bundle rather than mixing graph generations.

    :param tmp_path: Existing export destination.
    :param flow_model: Supported model.
    """
    existing = tmp_path / "conditioning.onnx"
    existing.write_bytes(b"existing bundle")
    with pytest.raises(FileExistsError):
        export_flow_onnx(flow_model, {}, tmp_path)
    assert existing.read_bytes() == b"existing bundle"


def test_export_endpoint_checkpoint_rejected_before_writing(
    tmp_path: Path, flow_model: VSTFlowMatchingModule
) -> None:
    """Velocity-only export must not silently change endpoint sampling semantics.

    :param tmp_path: Export destination.
    :param flow_model: Model configured as an unsupported endpoint field.
    """
    flow_model.hparams["parameterization"] = "endpoint"
    with pytest.raises(ValueError, match="velocity-parameterized"):
        export_flow_onnx(flow_model, {}, tmp_path)
    assert not list(tmp_path.iterdir())
