"""Behavioral tests for layerwise adaLN fusion of m2l and sketch conditioning."""

from functools import partial

import pytest
import torch

from synth_setter.conditioning import (
    EmbeddingConditioningSpec,
    SketchControlSpec,
    resolve_sketch_controls,
)
from synth_setter.models.components.embed_pool import EmbeddingPool
from synth_setter.models.components.fused_conditioning import (
    FusedConditioningEncoder,
    SketchLayerwiseEncoder,
)
from synth_setter.models.components.transformer import (
    ApproxEquivTransformer,
    LearntProjection,
)
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule

_BATCH = 2
_NUM_LAYERS = 3
_D_M2L = 8
_D_SKETCH = 4
_NUM_FRAMES = 101
_M2L_SHAPE = (6, 5)


def _fused_encoder(
    modality_dropout_rate: float = 0.2, joint_dropout_rate: float = 0.2
) -> FusedConditioningEncoder:
    """Build a tiny fused encoder for shape and dropout tests.

    :param modality_dropout_rate: Per-modality independent drop probability.
    :param joint_dropout_rate: Probability of dropping both modalities together.
    :returns: Fused encoder producing ``(_BATCH, _NUM_LAYERS, _D_M2L + _D_SKETCH)``.
    """
    return FusedConditioningEncoder(
        m2l_encoder=EmbeddingPool(
            embed_dim=_M2L_SHAPE[0], d_model=_D_M2L, num_heads=2, max_seq_len=_M2L_SHAPE[1]
        ),
        sketch_encoder=SketchLayerwiseEncoder(
            num_controls=3,
            num_frames=_NUM_FRAMES,
            d_model=_D_SKETCH,
            num_layers=_NUM_LAYERS,
            num_heads=2,
        ),
        d_m2l=_D_M2L,
        d_sketch=_D_SKETCH,
        num_layers=_NUM_LAYERS,
        modality_dropout_rate=modality_dropout_rate,
        joint_dropout_rate=joint_dropout_rate,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Draw one m2l conditioning matrix and one sketch control matrix.

    :returns: ``(conditioning, sketch_ctrl)`` tensors.
    """
    return torch.randn(_BATCH, *_M2L_SHAPE), torch.randn(_BATCH, 3, _NUM_FRAMES)


def test_sketch_layerwise_encoder_outputs_one_token_per_layer() -> None:
    """The sketch encoder maps (B, 3, N) controls to one token per vector-field layer."""
    encoder = SketchLayerwiseEncoder(
        num_controls=3,
        num_frames=_NUM_FRAMES,
        d_model=_D_SKETCH,
        num_layers=_NUM_LAYERS,
        num_heads=2,
    )

    tokens = encoder(torch.randn(_BATCH, 3, _NUM_FRAMES))

    assert tokens.shape == (_BATCH, _NUM_LAYERS, _D_SKETCH)
    assert torch.isfinite(tokens).all()


def test_fused_encoder_concatenates_m2l_broadcast_with_sketch_tokens() -> None:
    """Eval-mode fusion broadcasts the m2l vector to every layer slot."""
    encoder = _fused_encoder().eval()
    conditioning, sketch = _inputs()

    fused = encoder(conditioning, sketch)

    assert fused.shape == (_BATCH, _NUM_LAYERS, _D_M2L + _D_SKETCH)
    m2l = encoder.m2l_encoder(conditioning)
    torch.testing.assert_close(fused[..., :_D_M2L], m2l.unsqueeze(1).expand(-1, _NUM_LAYERS, -1))


def test_fused_encoder_joint_dropout_one_reproduces_null_conditioning() -> None:
    """With certain joint dropout, training-mode output is the all-null conditioning."""
    encoder = _fused_encoder(modality_dropout_rate=0.0, joint_dropout_rate=1.0).train()
    conditioning, sketch = _inputs()

    fused = encoder(conditioning, sketch)

    torch.testing.assert_close(fused, encoder.null_conditioning(_BATCH, fused.device))


def test_fused_encoder_zero_dropout_keeps_both_modalities() -> None:
    """Zero dropout in training mode matches the eval-mode fusion exactly."""
    encoder = _fused_encoder(modality_dropout_rate=0.0, joint_dropout_rate=0.0)
    conditioning, sketch = _inputs()

    trained = encoder.train()(conditioning, sketch)
    evaled = encoder.eval()(conditioning, sketch)

    torch.testing.assert_close(trained, evaled)


def test_fused_encoder_sketch_only_dropout_keeps_m2l_slice() -> None:
    """Dropping only the sketch modality preserves the encoded m2l slice."""
    torch.manual_seed(0)
    encoder = _fused_encoder(modality_dropout_rate=0.0, joint_dropout_rate=0.0)
    conditioning, sketch = _inputs()
    fused = encoder.eval()(conditioning, sketch)

    dropped = encoder.drop_sketch(fused)

    torch.testing.assert_close(dropped[..., :_D_M2L], fused[..., :_D_M2L])
    expected_sketch = encoder.sketch_null.expand(_BATCH, _NUM_LAYERS, -1)
    torch.testing.assert_close(dropped[..., _D_M2L:], expected_sketch)


def _vector_field(num_params: int = 4) -> ApproxEquivTransformer:
    """Build a tiny rank-3-conditioned vector field.

    :param num_params: Parameter width the field operates on.
    :returns: Transformer with ``_NUM_LAYERS`` layers.
    """
    return ApproxEquivTransformer(
        projection=LearntProjection(
            d_model=8,
            d_token=8,
            num_params=num_params,
            num_tokens=4,
            initial_ffn=True,
            final_ffn=False,
        ),
        num_layers=_NUM_LAYERS,
        d_model=8,
        conditioning_dim=_D_M2L + _D_SKETCH,
        num_heads=2,
        d_ff=8,
        num_tokens=4,
        pe_type="initial",
        time_encoding="scalar",
        learn_projection=True,
    )


def test_rank3_conditioning_flows_through_transformer_forward() -> None:
    """Layerwise (B, num_layers, d) conditioning drives the field without error."""
    field = _vector_field()
    x = torch.randn(_BATCH, 4)
    t = torch.rand(_BATCH, 1)
    conditioning = torch.randn(_BATCH, _NUM_LAYERS, _D_M2L + _D_SKETCH)

    out = field(x, t, conditioning)

    assert out.shape == (_BATCH, 4)
    assert torch.isfinite(out).all()


def _fused_module() -> VSTFlowMatchingModule:
    """Build a flow module wired for fused m2l + sketch conditioning.

    :returns: Module reading the canonical conditioning and sketch batch keys.
    """
    return VSTFlowMatchingModule(
        encoder=_fused_encoder(),
        vector_field=_vector_field(),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=4,
        conditioning=EmbeddingConditioningSpec(column="m2l", input_shape=_M2L_SHAPE),
        sketch=SketchControlSpec(num_frames=_NUM_FRAMES),
    )


def _fused_batch() -> dict[str, torch.Tensor]:
    """Draw one synthetic fused-conditioning training batch.

    :returns: Batch with conditioning, sketch controls, params, and noise.
    """
    return {
        "conditioning": torch.randn(_BATCH, *_M2L_SHAPE),
        "sketch_ctrl": torch.randn(_BATCH, 3, _NUM_FRAMES),
        "params": torch.randn(_BATCH, 4),
        "noise": torch.randn(_BATCH, 4),
    }


def test_fused_module_training_step_returns_finite_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One training step over random fused batches produces a finite scalar loss.

    :param monkeypatch: Detaches Lightning logging from a Trainer.
    """
    module = _fused_module()
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)

    loss = module.training_step(_fused_batch(), batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_fused_module_cfg_sampling_smoke_returns_param_shape() -> None:
    """CFG sampling with the fused encoder integrates to the parameter width."""
    module = _fused_module().eval()
    batch = _fused_batch()

    with torch.no_grad():
        sample = module._sample(  # noqa: SLF001
            (batch["conditioning"], batch["sketch_ctrl"]),
            torch.randn(_BATCH, 4),
            steps=2,
            cfg_strength=2.0,
        )

    assert sample.shape == (_BATCH, 4)
    assert torch.isfinite(sample).all()


def test_fused_module_unconditional_sampling_still_works() -> None:
    """The conditioning-free branch samples through the transformer null token."""
    module = _fused_module().eval()

    with torch.no_grad():
        sample = module._sample(  # noqa: SLF001
            None, torch.randn(_BATCH, 4), steps=2, cfg_strength=1.0
        )

    assert sample.shape == (_BATCH, 4)
    assert torch.isfinite(sample).all()


def test_resolve_sketch_controls_mapping_and_none_round_trip() -> None:
    """Hydra mappings resolve to the strict spec and None stays None."""
    assert resolve_sketch_controls(None) is None
    spec = resolve_sketch_controls({"column": "sketch_ctrl", "num_controls": 3, "num_frames": 401})
    assert spec == SketchControlSpec(num_frames=401)
