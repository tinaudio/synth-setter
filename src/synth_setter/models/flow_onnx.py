"""Export mel flow conditioning and guided velocity for host-side RK4.

Typical usage::

    export_flow_onnx(model.cpu().eval(), batch, Path("flow-bundle"))
"""

import os
import tempfile
from pathlib import Path

import torch
from beartype import beartype
from jaxtyping import Float, Shaped, jaxtyped
from torch import nn

from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule

_OPSET_VERSION = 18
# Velocity-graph branch order: unconditional, sketch-only, content-only, full.
_MODES: frozenset[str] = frozenset({"both", "mel_only", "sketch_only", "unconditional"})


@jaxtyped(typechecker=beartype)
def branch_weights(
    mode: str, content_cfg_strength: float, sketch_cfg_strength: float
) -> tuple[float, float, float, float]:
    """Return velocity-branch weights implementing classifier-free guidance for ``mode``.

    ``both`` reproduces the three-branch content/sketch guidance of training-time sampling;
    the other modes drop the unused branch so its tokens never influence the sample.

    :param mode: ``both``, ``mel_only``, ``sketch_only``, or ``unconditional``.
    :param content_cfg_strength: Content guidance scale.
    :param sketch_cfg_strength: Sketch guidance scale.
    :returns: Weights over the unconditional, sketch-only, content-only, and full branches.
    :raises ValueError: The mode is unknown.
    """
    if mode not in _MODES:
        raise ValueError(f"unknown conditioning mode {mode!r}; expected one of {sorted(_MODES)}")
    content, sketch = float(content_cfg_strength), float(sketch_cfg_strength)
    if mode == "both":
        return (1.0 - sketch, sketch - content, 0.0, content)
    if mode == "mel_only":
        return (1.0 - content, 0.0, content, 0.0)
    if mode == "sketch_only":
        return (1.0 - sketch, sketch, 0.0, 0.0)
    return (1.0, 0.0, 0.0, 0.0)


class FlowConditioning(nn.Module):
    """Encode content and both sketch branches once before integration."""

    @jaxtyped(typechecker=beartype)
    def __init__(self, model: VSTFlowMatchingModule) -> None:
        """Bind trained conditioning modules.

        :param model: Mel/sketch flow checkpoint loaded in evaluation mode.
        """
        super().__init__()
        self.model = model

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        mel: Float[torch.Tensor, "batch channels bins frames"],
        sketch_ctrl: Float[torch.Tensor, "batch controls sketch_frames"],
    ) -> tuple[
        Float[torch.Tensor, "batch slots dim"],
        Float[torch.Tensor, "batch tokens dim"],
        Float[torch.Tensor, "batch tokens dim"],
    ]:
        """Return encoded content, full-sketch tokens, and PE-only tokens.

        :param mel: Normalized spectrogram using the training dataset statistics.
        :param sketch_ctrl: Pooled controls using the checkpoint sketch contract.
        :returns: Content and both control-token branches for the velocity graph.
        :raises ValueError: The checkpoint has no sketch controls.
        """
        branches = self.model._control_token_branches_from_batch({"sketch_ctrl": sketch_ctrl})
        if branches is None:
            raise ValueError("ONNX conditioning requires sketch controls")
        return self.model.encoder(mel), branches.conditional, branches.unconditional


class BaselineFlowConditioning(nn.Module):
    """Encode mel content once before baseline integration."""

    @jaxtyped(typechecker=beartype)
    def __init__(self, model: VSTFlowMatchingModule) -> None:
        """Bind the trained content encoder.

        :param model: Mel-only flow checkpoint loaded in evaluation mode.
        """
        super().__init__()
        self.encoder = model.encoder

    @jaxtyped(typechecker=beartype)
    def forward(
        self, mel: Float[torch.Tensor, "batch channels bins frames"]
    ) -> Float[torch.Tensor, "batch slots dim"]:
        """Return encoded mel conditioning.

        :param mel: Normalized spectrogram using the training dataset statistics.
        :returns: Content tokens for the baseline velocity graph.
        """
        return self.encoder(mel)


class BaselineFlowVelocity(nn.Module):
    """Expose unconditional and content-conditioned baseline branches."""

    @jaxtyped(typechecker=beartype)
    def __init__(self, model: VSTFlowMatchingModule) -> None:
        """Bind the checkpoint's learned vector field.

        :param model: Mel-only velocity-parameterized checkpoint.
        """
        super().__init__()
        self.field = model.vector_field

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "batch params"],
        t: Float[torch.Tensor, "batch 1"],
        *,
        conditioning: Float[torch.Tensor, "batch slots dim"],
        branch_weights: Float[torch.Tensor, "2"],
    ) -> Float[torch.Tensor, "batch params"]:
        """Combine unconditional and content-conditioned fields.

        :param x: Current model-space parameter state.
        :param t: Flow time in the closed unit interval.
        :param conditioning: Encoded content from the conditioning graph.
        :param branch_weights: Host-supplied ``(1 - c, c)`` guidance weights.
        :returns: Guided velocity at the supplied state and time.
        """
        unconditional = self.field(x, t, None)
        conditional = self.field(x, t, conditioning)
        return branch_weights[0] * unconditional + branch_weights[1] * conditional


class FlowVelocity(nn.Module):
    """Keep branch weights and trajectory state as graph inputs."""

    @jaxtyped(typechecker=beartype)
    def __init__(self, model: VSTFlowMatchingModule) -> None:
        """Bind the checkpoint's learned vector field.

        :param model: Mel/sketch velocity-parameterized checkpoint.
        """
        super().__init__()
        self.field = model.vector_field

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "batch params"],
        t: Float[torch.Tensor, "batch 1"],
        *,
        conditioning: Float[torch.Tensor, "batch slots dim"],
        controls: Float[torch.Tensor, "batch tokens dim"],
        null_controls: Float[torch.Tensor, "batch tokens dim"],
        branch_weights: Float[torch.Tensor, "4"],
    ) -> Float[torch.Tensor, "batch params"]:
        """Combine the four conditioning branches with host-supplied weights.

        :param x: Current model-space parameter state.
        :param t: Flow time in the closed unit interval.
        :param conditioning: Encoded content from the conditioning graph.
        :param controls: Conditional sketch tokens.
        :param null_controls: PE-only unconditional sketch tokens.
        :param branch_weights: Weights over the unconditional, sketch-only, content-only,
            and full branches, from :func:`branch_weights`.
        :returns: Guided velocity at the supplied state and time.
        """
        branches = torch.stack(
            (
                self.field(x, t, None, control_tokens=null_controls),
                self.field(x, t, None, control_tokens=controls),
                self.field(x, t, conditioning, control_tokens=null_controls),
                self.field(x, t, conditioning, control_tokens=controls),
            )
        )
        return torch.einsum("b,b...->...", branch_weights, branches)


@jaxtyped(typechecker=beartype)
def export_flow_onnx(
    model: VSTFlowMatchingModule,
    batch: dict[str, Shaped[torch.Tensor, "batch ..."]],
    output_dir: Path,
) -> None:
    """Export fixed-shape float32 graphs without unrolling the integration loop.

    :param model: CPU evaluation model using mel velocity parameterization, optionally with sketch
        controls.
    :param batch: CPU float32 ``mel`` shaped ``(batch, channels, bins, frames)`` and, for sketch
        checkpoints, ``sketch_ctrl`` shaped ``(batch, controls, control_tokens)``.
    :param output_dir: Empty or absent destination, atomically populated with both graphs.
    :raises FileExistsError: The destination already contains artifacts.
    :raises ValueError: Model type, conditioning, parameterization, or inputs are unsupported.
    """
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"ONNX destination is not empty: {output_dir}")
    if type(model) is not VSTFlowMatchingModule or model.hparams["parameterization"] != "velocity":
        raise ValueError("ONNX export requires the standard velocity-parameterized flow model")
    if model.hparams["conditioning"] != "mel":
        raise ValueError("ONNX export requires mel conditioning")
    if model.training or next(model.parameters()).device.type != "cpu":
        raise ValueError("ONNX export requires a CPU model in evaluation mode")
    has_sketch_controls = model.sketch_tokens is not None
    if has_sketch_controls and ("mel" not in batch or "sketch_ctrl" not in batch):
        raise ValueError("ONNX inputs require mel and sketch_ctrl conditioning")
    if "mel" not in batch:
        raise ValueError("ONNX inputs require mel conditioning")

    mel = batch["mel"]
    if mel.device.type != "cpu" or mel.dtype != torch.float32 or not torch.isfinite(mel).all():
        raise ValueError("ONNX inputs must be finite CPU float32 tensors")
    if mel.ndim != 4 or mel.shape[0] == 0:
        raise ValueError("ONNX export requires nonempty rank-4 mel input")
    if has_sketch_controls:
        sketch_ctrl = batch["sketch_ctrl"]
        if (
            sketch_ctrl.device.type != "cpu"
            or sketch_ctrl.dtype != torch.float32
            or not torch.isfinite(sketch_ctrl).all()
        ):
            raise ValueError("ONNX inputs must be finite CPU float32 tensors")
        if sketch_ctrl.ndim != 3:
            raise ValueError("ONNX export requires nonempty rank-4 mel and rank-3 sketch inputs")
        if mel.shape[0] != sketch_ctrl.shape[0]:
            raise ValueError("ONNX mel and sketch batch sizes must match")
        assert model.sketch_tokens is not None
        if sketch_ctrl.shape[-1] != model.sketch_tokens.positional_encoding.shape[1]:
            raise ValueError("ONNX sketch inputs must already use the control-token temporal grid")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
        staged = Path(temporary)
        if has_sketch_controls:
            _export_graphs(model, (mel, batch["sketch_ctrl"]), staged)
        else:
            _export_baseline_graphs(model, mel, staged)
        os.replace(staged, output_dir)


@jaxtyped(typechecker=beartype)
def _export_baseline_graphs(
    model: VSTFlowMatchingModule,
    mel: Float[torch.Tensor, "batch channels bins frames"],
    output_dir: Path,
) -> None:
    """Write mel-only graphs to an unpublished staging directory.

    :param model: Validated CPU evaluation checkpoint without sketch controls.
    :param mel: Validated normalized mel input.
    :param output_dir: Temporary export directory.
    """
    conditioning = BaselineFlowConditioning(model).eval()
    velocity = BaselineFlowVelocity(model).eval()
    with torch.no_grad():
        encoded = conditioning(mel)
        torch.onnx.export(
            conditioning,
            (mel,),
            output_dir / "conditioning.onnx",
            input_names=["mel"],
            output_names=["conditioning"],
            opset_version=_OPSET_VERSION,
            dynamo=True,
            external_data=False,
        )
        state = mel.new_zeros(mel.shape[0], model.hparams["num_params"])
        time = mel.new_zeros(mel.shape[0], 1)
        torch.onnx.export(
            velocity,
            (state, time),
            output_dir / "velocity.onnx",
            kwargs={"conditioning": encoded, "branch_weights": mel.new_ones(2)},
            input_names=["x", "t", "conditioning", "branch_weights"],
            output_names=["velocity"],
            opset_version=_OPSET_VERSION,
            dynamo=True,
            external_data=False,
        )


@jaxtyped(typechecker=beartype)
def _export_graphs(
    model: VSTFlowMatchingModule,
    inputs: tuple[Shaped[torch.Tensor, "batch ..."], Shaped[torch.Tensor, "batch ..."]],
    output_dir: Path,
) -> None:
    """Write both graphs to an unpublished staging directory.

    :param model: Validated CPU evaluation checkpoint.
    :param inputs: Validated normalized mel and pooled sketch inputs.
    :param output_dir: Temporary export directory.
    """
    conditioning = FlowConditioning(model).eval()
    velocity = FlowVelocity(model).eval()
    with torch.no_grad():
        encoded = conditioning(*inputs)
        torch.onnx.export(
            conditioning,
            inputs,
            output_dir / "conditioning.onnx",
            input_names=["mel", "sketch_ctrl"],
            output_names=["conditioning", "controls", "null_controls"],
            opset_version=_OPSET_VERSION,
            dynamo=True,
            external_data=False,
        )
        state = inputs[0].new_zeros(inputs[0].shape[0], model.hparams["num_params"])
        time = inputs[0].new_zeros(inputs[0].shape[0], 1)
        torch.onnx.export(
            velocity,
            (state, time),
            output_dir / "velocity.onnx",
            kwargs={
                "conditioning": encoded[0],
                "controls": encoded[1],
                "null_controls": encoded[2],
                "branch_weights": inputs[0].new_ones(4),
            },
            input_names=["x", "t", "conditioning", "controls", "null_controls", "branch_weights"],
            output_names=["velocity"],
            opset_version=_OPSET_VERSION,
            dynamo=True,
            external_data=False,
        )
