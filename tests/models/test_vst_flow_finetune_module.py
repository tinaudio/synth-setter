"""Smoke tests for the simulator-feedback finetune module (spike #2556)."""

from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch

from synth_setter.models.components.embed_pool import EmbeddingPool
from synth_setter.models.components.vector_field import VectorField
from synth_setter.models.vst_flow_matching_finetune_module import (
    VSTFlowMatchingFinetuneModule,
)
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule

_NUM_PARAMS = 5
_EMBED_DIM = 8
_SEQ_LEN = 6
_BATCH = 3
_AUDIO_SAMPLES = 32


@pytest.fixture()
def base_ckpt(tmp_path: Path) -> str:
    """Write a tiny embedding-conditioned base checkpoint.

    :param tmp_path: pytest-managed output directory.
    :returns: Checkpoint path loadable via ``load_from_checkpoint``.
    """
    base = VSTFlowMatchingModule(
        encoder=EmbeddingPool(_EMBED_DIM, 16, 2, max_seq_len=_SEQ_LEN),
        vector_field=VectorField(_NUM_PARAMS, 16, 16, 2),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=_NUM_PARAMS,
        conditioning={"column": "m2l", "input_shape": [_EMBED_DIM, _SEQ_LEN]},
    )
    path = tmp_path / "base.ckpt"
    torch.save(
        {
            "state_dict": base.state_dict(),
            "hyper_parameters": dict(base.hparams),
            "pytorch-lightning_version": "2.0.0",
        },
        path,
    )
    return str(path)


def _finetune_module(base_ckpt: str, *, feedback_enabled: bool) -> VSTFlowMatchingFinetuneModule:
    """Build a tiny finetune module with fake simulator callables.

    :param base_ckpt: Tiny base checkpoint path.
    :param feedback_enabled: Whether simulator feedback is on.
    :returns: Module ready for offline smoke testing.
    """
    module = VSTFlowMatchingFinetuneModule(
        base_ckpt,
        partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        None,
        feedback_enabled=feedback_enabled,
        control_dim=4,
        control_hidden_dim=8,
        control_num_blocks=1,
        control_encoder_hidden_dim=8,
    )

    def fake_render(rows: np.ndarray) -> tuple[np.ndarray, int]:
        return np.zeros((rows.shape[0], 2, _AUDIO_SAMPLES), np.float32), 0

    def fake_m2l(audio: np.ndarray) -> torch.Tensor:
        return torch.zeros(audio.shape[0], _EMBED_DIM, _SEQ_LEN)

    module.render_batch_fn = fake_render
    module.m2l_encode_fn = fake_m2l
    return module


def _batch() -> dict[str, torch.Tensor]:
    """Return one fake model batch.

    :returns: Batch with conditioning, params, and noise tensors.
    """
    return {
        "conditioning": torch.randn(_BATCH, _EMBED_DIM, _SEQ_LEN),
        "params": torch.rand(_BATCH, _NUM_PARAMS) * 2 - 1,
        "noise": torch.randn(_BATCH, _NUM_PARAMS),
    }


def test_construct_from_checkpoint_freezes_base(base_ckpt: str) -> None:
    """Construction freezes every base parameter and trains the control nets.

    :param base_ckpt: Tiny base checkpoint path.
    """
    module = _finetune_module(base_ckpt, feedback_enabled=True)

    assert all(not p.requires_grad for p in module.base.parameters())
    assert all(p.requires_grad for p in module.control_field.parameters())
    assert all(p.requires_grad for p in module.control_encoder.parameters())


def test_training_step_with_feedback_backprops_only_control(base_ckpt: str) -> None:
    """A feedback training step produces grads on control nets only.

    :param base_ckpt: Tiny base checkpoint path.
    """
    module = _finetune_module(base_ckpt, feedback_enabled=True)
    loss = module.training_step(_batch(), 0)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(p.grad is None for p in module.base.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in module.control_field.parameters()
    )


def test_training_step_ablation_gives_no_encoder_grads(base_ckpt: str) -> None:
    """The c == 0 ablation leaves the control encoder without gradients.

    :param base_ckpt: Tiny base checkpoint path.
    """
    module = _finetune_module(base_ckpt, feedback_enabled=False)
    loss = module.training_step(_batch(), 0)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        p.grad is None or p.grad.abs().sum() == 0 for p in module.control_encoder.parameters()
    )


def test_zero_init_control_matches_base_field(base_ckpt: str) -> None:
    """The zero-init head makes the initial correction exactly zero.

    :param base_ckpt: Tiny base checkpoint path.
    """
    module = _finetune_module(base_ckpt, feedback_enabled=True)
    batch = _batch()
    x_t = batch["noise"]
    t = torch.full((_BATCH, 1), 0.9)
    v = torch.randn(_BATCH, _NUM_PARAMS)
    c = torch.randn(_BATCH, 4)

    correction = module.control_field(x_t, t, v, c)

    assert torch.allclose(correction, torch.zeros_like(correction))


def test_sample_returns_param_shape(base_ckpt: str) -> None:
    """Sampling integrates to finite parameter rows of the right shape.

    :param base_ckpt: Tiny base checkpoint path.
    """
    module = _finetune_module(base_ckpt, feedback_enabled=True)
    batch = _batch()

    sample = module.sample(batch["conditioning"], batch["noise"], steps=5)

    assert sample.shape == (_BATCH, _NUM_PARAMS)
    assert torch.isfinite(sample).all()


def test_configure_optimizers_excludes_base_params(base_ckpt: str) -> None:
    """The optimizer covers control parameters and never the frozen base.

    :param base_ckpt: Tiny base checkpoint path.
    """
    module = _finetune_module(base_ckpt, feedback_enabled=True)

    config = module.configure_optimizers()

    optimized = {id(p) for group in config["optimizer"].param_groups for p in group["params"]}
    assert all(id(p) not in optimized for p in module.base.parameters())
    assert all(id(p) in optimized for p in module.control_field.parameters())
