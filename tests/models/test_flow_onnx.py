"""ONNX graphs must preserve the production flow field and conditioning."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from hydra.utils import instantiate

from synth_setter.evaluation import browser_flow
from synth_setter.models.components.transformer import (
    ApproxEquivTransformer,
    LearntProjection,
)
from synth_setter.models.flow_onnx import branch_weights, export_flow_onnx
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
            "branch_weights": np.array(branch_weights("both", *strengths), dtype=np.float32),
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


@pytest.fixture
def browser_batch(
    flow_model: VSTFlowMatchingModule, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, torch.Tensor]:
    """Prepare a supported pair in an uninstalled browser environment.

    :param flow_model: Architecture defining the sketch-control width.
    :param tmp_path: Absent runtime location for invalid-request tests.
    :param monkeypatch: Prevents a missing guard from starting a browser server.
    :returns: One mel/sketch pair on the model's fixed temporal grid.
    """
    monkeypatch.setattr(browser_flow, "_WEB_ROOT", tmp_path / "uninstalled-web")
    assert flow_model.sketch_tokens is not None
    return {
        "mel": torch.zeros(1, 2, 8, 8),
        "sketch_ctrl": torch.zeros(1, flow_model.sketch_tokens.layout.num_controls, 4),
    }


@pytest.mark.parametrize(
    ("rows", "width_offset", "dtype", "value"),
    [
        (2, 0, torch.float32, 0.0),
        (1, -1, torch.float32, 0.0),
        (1, 0, torch.float64, 0.0),
        (1, 0, torch.float32, float("nan")),
        (1, 0, torch.float32, float("inf")),
    ],
)
def test_browser_export_invalid_noise_rejected_before_writing(
    tmp_path: Path,
    flow_model: VSTFlowMatchingModule,
    browser_batch: dict[str, torch.Tensor],
    rows: int,
    width_offset: int,
    dtype: torch.dtype,
    value: float,
) -> None:
    """Malformed initial states cannot create browser artifacts.

    :param tmp_path: Absent bundle's parent directory.
    :param flow_model: Real model defining the required noise width.
    :param browser_batch: Valid conditioning independent of the malformed noise.
    :param rows: Requested noise batch size.
    :param width_offset: Deliberate deviation from the checkpoint's parameter width.
    :param dtype: Noise precision, including unsupported float64.
    :param value: Noise coordinate, including nonfinite values.
    """
    noise = torch.full((rows, flow_model.hparams["num_params"] + width_offset), value, dtype=dtype)
    with pytest.raises(ValueError, match="finite float32 noise row"):
        browser_flow.sample_in_browser(
            flow_model,
            browser_batch,
            noise,
            content_cfg_strength=2.0,
            sketch_cfg_strength=3.0,
            sample_steps=8,
            output_dir=tmp_path / "bundle",
        )
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("steps", [0, 1001, True])
def test_browser_export_invalid_step_count_rejected_before_writing(
    tmp_path: Path,
    flow_model: VSTFlowMatchingModule,
    browser_batch: dict[str, torch.Tensor],
    steps: int,
) -> None:
    """Invalid integration counts cannot start an export.

    :param tmp_path: Absent bundle's parent directory.
    :param flow_model: Supported real export architecture.
    :param browser_batch: Valid conditioning pair.
    :param steps: Nonpositive, excessive, or boolean integration count.
    """
    with pytest.raises(ValueError, match="integer between 1 and 1000"):
        browser_flow.sample_in_browser(
            flow_model,
            browser_batch,
            torch.zeros(1, flow_model.hparams["num_params"]),
            content_cfg_strength=2.0,
            sketch_cfg_strength=3.0,
            sample_steps=steps,
            output_dir=tmp_path / "bundle",
        )
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    ("content", "sketch"),
    [(-1.0, 0.0), (0.0, -1.0), (float("nan"), 0.0), (0.0, float("inf")), (1e40, 0.0)],
)
def test_browser_export_invalid_guidance_rejected_before_writing(
    tmp_path: Path,
    flow_model: VSTFlowMatchingModule,
    browser_batch: dict[str, torch.Tensor],
    content: float,
    sketch: float,
) -> None:
    """Guidance must remain finite and nonnegative after float32 conversion.

    :param tmp_path: Absent bundle's parent directory.
    :param flow_model: Supported real export architecture.
    :param browser_batch: Valid conditioning pair.
    :param content: Content guidance, including invalid values.
    :param sketch: Sketch guidance, including invalid values.
    """
    with pytest.raises(ValueError, match="finite and non-negative"):
        browser_flow.sample_in_browser(
            flow_model,
            browser_batch,
            torch.zeros(1, flow_model.hparams["num_params"]),
            content_cfg_strength=content,
            sketch_cfg_strength=sketch,
            sample_steps=8,
            output_dir=tmp_path / "bundle",
        )
    assert not (tmp_path / "bundle").exists()


def test_browser_export_multiple_pairs_rejected_before_writing(
    tmp_path: Path,
    flow_model: VSTFlowMatchingModule,
    browser_batch: dict[str, torch.Tensor],
) -> None:
    """A browser session cannot silently discard additional input pairs.

    :param tmp_path: Absent bundle's parent directory.
    :param flow_model: Supported real export architecture.
    :param browser_batch: Conditioning expanded to an unsupported two-pair batch.
    """
    batch = {
        key: value.repeat(2, *([1] * (value.ndim - 1))) for key, value in browser_batch.items()
    }
    with pytest.raises(ValueError, match="one content/sketch pair"):
        browser_flow.sample_in_browser(
            flow_model,
            batch,
            torch.zeros(1, flow_model.hparams["num_params"]),
            content_cfg_strength=2.0,
            sketch_cfg_strength=3.0,
            sample_steps=8,
            output_dir=tmp_path / "bundle",
        )
    assert not (tmp_path / "bundle").exists()


def test_browser_export_missing_runtime_rejected_before_writing(
    tmp_path: Path,
    flow_model: VSTFlowMatchingModule,
    browser_batch: dict[str, torch.Tensor],
) -> None:
    """Missing npm assets fail before publishing any ONNX bundle.

    :param tmp_path: Absent export destination's parent directory.
    :param flow_model: Supported real export architecture.
    :param browser_batch: Valid pair with no npm runtime installed.
    """
    with pytest.raises(FileNotFoundError, match="npm ci"):
        browser_flow.sample_in_browser(
            flow_model,
            browser_batch,
            torch.zeros(1, flow_model.hparams["num_params"]),
            content_cfg_strength=2.0,
            sketch_cfg_strength=3.0,
            sample_steps=8,
            output_dir=tmp_path / "bundle",
        )
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("both", (-2.0, 1.0, 0.0, 2.0)),
        ("mel_only", (-1.0, 0.0, 2.0, 0.0)),
        ("sketch_only", (-2.0, 3.0, 0.0, 0.0)),
        ("unconditional", (1.0, 0.0, 0.0, 0.0)),
    ],
)
def test_branch_weights_mode_selects_branches_and_sums_to_one(
    mode: str, expected: tuple[float, float, float, float]
) -> None:
    """Every mode is a convex-style combination over the four velocity branches.

    :param mode: Conditioning mode selected in the browser.
    :param expected: Weights over unconditional, sketch-only, content-only, and full branches.
    """
    assert branch_weights(mode, 2.0, 3.0) == expected


@pytest.mark.parametrize("mode", ["", "content", "MEL_ONLY"])
def test_branch_weights_unknown_mode_rejected(mode: str) -> None:
    """Unknown modes fail loudly rather than silently sampling unconditionally.

    :param mode: Unsupported conditioning mode.
    """
    with pytest.raises(ValueError, match="mode"):
        branch_weights(mode, 2.0, 3.0)


def test_exported_velocity_mel_only_weights_ignore_sketch_tokens(
    tmp_path: Path, flow_model: VSTFlowMatchingModule
) -> None:
    """Mel-only weights evaluate content against PE-only controls, so sketch content is inert.

    :param tmp_path: Export directory.
    :param flow_model: Small production architecture with nonzero sketch projections.
    """
    assert flow_model.sketch_tokens is not None
    controls = flow_model.sketch_tokens.layout.num_controls
    batch = {"mel": torch.randn(1, 2, 8, 8), "sketch_ctrl": torch.rand(1, controls, 4)}
    export_flow_onnx(flow_model, batch, tmp_path)
    encoder = ort.InferenceSession(str(tmp_path / "conditioning.onnx"))
    field = ort.InferenceSession(str(tmp_path / "velocity.onnx"))
    x, t = torch.randn(1, 92), torch.tensor([[0.3]])

    def guided(sketch_ctrl: torch.Tensor) -> np.ndarray:
        """Run the exported graphs for one sketch input under mel-only weights.

        :param sketch_ctrl: Sketch controls fed to the conditioning graph.
        :returns: Guided velocity row.
        """
        encoded = encoder.run(
            None, {"mel": batch["mel"].numpy(), "sketch_ctrl": sketch_ctrl.numpy()}
        )
        (velocity,) = field.run(
            None,
            {
                "x": x.numpy(),
                "t": t.numpy(),
                "conditioning": encoded[0],
                "controls": encoded[1],
                "null_controls": encoded[2],
                "branch_weights": np.array(branch_weights("mel_only", 2.0, 3.0), dtype=np.float32),
            },
        )
        assert isinstance(velocity, np.ndarray)
        return velocity

    with torch.no_grad():
        expected = flow_model._velocity_field(
            flow_model.encoder(batch["mel"]),
            2.0,
            flow_model._control_token_branches_from_batch(
                {"sketch_ctrl": torch.zeros_like(batch["sketch_ctrl"])}
            ),
            sketch_cfg_strength=0.0,
        )(x, t)
    np.testing.assert_allclose(
        guided(batch["sketch_ctrl"]), guided(torch.rand(1, controls, 4)), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        guided(batch["sketch_ctrl"]), expected.numpy(), rtol=2e-5, atol=2e-5
    )
