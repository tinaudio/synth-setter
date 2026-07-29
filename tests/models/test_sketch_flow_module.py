"""Tests for sketch-control conditioning in the flow-matching module."""

from functools import partial
from typing import Literal, cast

import torch

from synth_setter.conditioning import SketchControlSpec
from synth_setter.data.vst.shapes import NUM_SKETCH_CONTROLS
from synth_setter.models.components.transformer import (
    ApproxEquivTransformer,
    LearntProjection,
)
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule

_BATCH = 2
_D_MODEL = 16
_NUM_PARAMS = 6
_NUM_FRAMES = 9
_MEL_SHAPE = (2, 8, 5)


class _FlattenEncoder(torch.nn.Module):
    """Minimal mel encoder projecting the flattened spectrogram to one vector."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(_MEL_SHAPE[0] * _MEL_SHAPE[1] * _MEL_SHAPE[2], 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


def _module(
    sketch: SketchControlSpec | None,
    *,
    sketch_dropout_rate: float = 0.2,
    sketch_all_dropout_rate: float = 0.2,
) -> VSTFlowMatchingModule:
    """Build a tiny CPU flow module.

    :param sketch: Sketch-control spec, or ``None`` for the baseline model.
    :param sketch_dropout_rate: Independent per-control drop probability.
    :param sketch_all_dropout_rate: Probability of dropping every control.
    :returns: Module ready for ``_train_step``.
    """
    torch.manual_seed(0)
    field = ApproxEquivTransformer(
        projection=LearntProjection(
            d_model=_D_MODEL,
            d_token=_D_MODEL,
            num_params=_NUM_PARAMS,
            num_tokens=5,
        ),
        num_layers=2,
        d_model=_D_MODEL,
        conditioning_dim=8,
        num_heads=2,
        d_ff=_D_MODEL,
        num_tokens=5,
        # "none" is a valid runtime value model/vst_flow.yaml ships; the
        # constructor's Literal annotation omits it.
        pe_type=cast(Literal["initial", "layerwise"], "none"),
        time_encoding="scalar",
        learn_projection=True,
    )
    return VSTFlowMatchingModule(
        encoder=_FlattenEncoder(),
        vector_field=field,
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=_NUM_PARAMS,
        sketch_controls=sketch,
        sketch_dropout_rate=sketch_dropout_rate,
        sketch_all_dropout_rate=sketch_all_dropout_rate,
        validation_sample_steps=2,
    )


def _batch(*, with_sketch: bool) -> dict[str, torch.Tensor]:
    """Draw a deterministic synthetic mel batch.

    :param with_sketch: Whether the batch carries a ``sketch_ctrl`` tensor.
    :returns: Model batch.
    """
    generator = torch.Generator().manual_seed(7)
    batch = {
        "mel_spec": torch.randn((_BATCH, *_MEL_SHAPE), generator=generator),
        "params": torch.rand((_BATCH, _NUM_PARAMS), generator=generator) * 2 - 1,
        "noise": torch.randn((_BATCH, _NUM_PARAMS), generator=generator),
    }
    if with_sketch:
        batch["sketch_ctrl"] = torch.rand(
            (_BATCH, NUM_SKETCH_CONTROLS, _NUM_FRAMES), generator=generator
        )
    return batch


def test_train_step_with_sketch_batch_produces_finite_loss() -> None:
    """Sketch-configured training consumes ``sketch_ctrl`` and stays finite."""
    module = _module(SketchControlSpec(num_frames=_NUM_FRAMES))

    loss, _penalty = module._train_step(_batch(with_sketch=True))  # noqa: SLF001

    assert torch.isfinite(loss)


def test_train_step_none_spec_ignores_sketch_free_batch() -> None:
    """The default configuration trains on batches without ``sketch_ctrl``."""
    module = _module(None)

    loss, _penalty = module._train_step(_batch(with_sketch=False))  # noqa: SLF001

    assert module.sketch_tokens is None
    assert torch.isfinite(loss)


def test_train_step_none_spec_matches_loss_before_sketch_support() -> None:
    """``sketch_controls=None`` reproduces the pre-sketch loss computation.

    The reference below is the documented pre-change ``_train_step`` recipe;
    equality on a fixed seed pins that the ``None`` path adds no RNG draws and
    no extra field inputs.
    """
    module = _module(None)
    batch = _batch(with_sketch=False)

    torch.manual_seed(11)
    loss, _penalty = module._train_step(batch)  # noqa: SLF001

    torch.manual_seed(11)
    field = cast(ApproxEquivTransformer, module.vector_field)
    z = field.apply_dropout(module.encoder(batch["mel_spec"]), module.hparams["cfg_dropout_rate"])
    t = torch.rand(_BATCH, 1)
    x_t = batch["noise"] * (1 - t) + batch["params"] * t
    target = batch["params"] - batch["noise"]
    prediction = field(x_t, t, z)
    expected = (prediction - target).square().mean(dim=-1).mean()

    torch.testing.assert_close(loss, expected)


def test_sketch_drop_mask_zero_rates_keep_every_control() -> None:
    """Zero drop rates yield an all-false mask."""
    module = _module(
        SketchControlSpec(num_frames=_NUM_FRAMES),
        sketch_dropout_rate=0.0,
        sketch_all_dropout_rate=0.0,
    )
    mask = module._sketch_drop_mask(4, torch.device("cpu"))  # noqa: SLF001
    assert mask.shape == (4, 3)
    assert not mask.any()


def test_sketch_drop_mask_full_joint_rate_drops_every_control() -> None:
    """A unit all-drop rate masks every control for every sample."""
    module = _module(
        SketchControlSpec(num_frames=_NUM_FRAMES),
        sketch_dropout_rate=0.0,
        sketch_all_dropout_rate=1.0,
    )
    assert module._sketch_drop_mask(4, torch.device("cpu")).all()  # noqa: SLF001


def test_validation_step_with_sketch_runs_cfg_sampling() -> None:
    """CFG sampling consumes undropped sketch tokens on the conditional branch."""
    module = _module(SketchControlSpec(num_frames=_NUM_FRAMES))

    out = module.validation_step(_batch(with_sketch=True), 0)

    assert torch.isfinite(out["param_mse"])
    assert out["preds"].shape == (_BATCH, _NUM_PARAMS)
