"""Export mel/sketch flow conditioning and guided velocity for host-side RK4."""

from pathlib import Path

import torch
from beartype import beartype
from jaxtyping import Float, Shaped, jaxtyped
from torch import nn

from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule


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


class FlowVelocity(nn.Module):
    """Keep guidance strengths and trajectory state as graph inputs."""

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
        conditioning: Float[torch.Tensor, "batch slots dim"],
        controls: Float[torch.Tensor, "batch tokens dim"],
        null_controls: Float[torch.Tensor, "batch tokens dim"],
        guidance: Float[torch.Tensor, "2"],
    ) -> Float[torch.Tensor, "batch params"]:
        """Evaluate independent content and sketch classifier-free guidance.

        :param x: Current model-space parameter state.
        :param t: Flow time in the closed unit interval.
        :param conditioning: Encoded content from the conditioning graph.
        :param controls: Conditional sketch tokens.
        :param null_controls: PE-only unconditional sketch tokens.
        :param guidance: Content and sketch strengths, in that order.
        :returns: Guided velocity at the supplied state and time.
        """
        unconditional = self.field(x, t, None, control_tokens=null_controls)
        sketch = self.field(x, t, None, control_tokens=controls)
        full = self.field(x, t, conditioning, control_tokens=controls)
        return (
            unconditional + guidance[1] * (sketch - unconditional) + guidance[0] * (full - sketch)
        )


@jaxtyped(typechecker=beartype)
def export_flow_onnx(
    model: VSTFlowMatchingModule,
    batch: dict[str, Shaped[torch.Tensor, "batch ..."]],
    output_dir: Path,
) -> None:
    """Export fixed-shape float32 graphs without unrolling the integration loop.

    :param model: CPU evaluation model using mel/sketch velocity parameterization.
    :param batch: Representative normalized mel and pooled sketch inputs.
    :param output_dir: Destination for conditioning.onnx and velocity.onnx.
    :raises ValueError: Model type, conditioning, parameterization, or inputs are unsupported.
    """
    if type(model) is not VSTFlowMatchingModule or model.hparams["parameterization"] != "velocity":
        raise ValueError("ONNX export requires the standard velocity-parameterized flow model")
    if model.hparams["conditioning"] != "mel" or model.sketch_tokens is None:
        raise ValueError("ONNX export requires mel conditioning and sketch controls")
    if model.training or next(model.parameters()).device.type != "cpu":
        raise ValueError("ONNX export requires a CPU model in evaluation mode")
    inputs = (batch["mel"], batch["sketch_ctrl"])
    if any(
        value.device.type != "cpu"
        or value.dtype != torch.float32
        or not torch.isfinite(value).all()
        for value in inputs
    ):
        raise ValueError("ONNX inputs must be finite CPU float32 tensors")
    if inputs[1].shape[-1] != model.sketch_tokens.positional_encoding.shape[1]:
        raise ValueError("ONNX sketch inputs must already use the control-token temporal grid")
    output_dir.mkdir(parents=True, exist_ok=True)
    conditioning = FlowConditioning(model).eval()
    velocity = FlowVelocity(model).eval()
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        # Fused native attention has no portable ONNX Runtime Web kernel.
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.no_grad():
            encoded = conditioning(*inputs)
            torch.onnx.export(
                conditioning,
                inputs,
                output_dir / "conditioning.onnx",
                input_names=["mel", "sketch_ctrl"],
                output_names=["conditioning", "controls", "null_controls"],
                opset_version=18,
                dynamo=True,
                external_data=False,
            )
            state = torch.zeros(inputs[0].shape[0], model.hparams["num_params"])
            time = torch.zeros(inputs[0].shape[0], 1)
            torch.onnx.export(
                velocity,
                (state, time, *encoded, torch.ones(2)),
                output_dir / "velocity.onnx",
                input_names=["x", "t", "conditioning", "controls", "null_controls", "guidance"],
                output_names=["velocity"],
                opset_version=18,
                dynamo=True,
                external_data=False,
            )
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
