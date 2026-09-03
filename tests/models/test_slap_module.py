"""Behavior tests for the SLAP Lightning model."""

from functools import partial

import torch
from torch import nn

from synth_setter.models.components.slap import BYOLLoss, SiameseArm
from synth_setter.models.components.slap_ema import MovingAverageWeightUpdate
from synth_setter.models.slap_module import SLAPModule


def _arm(input_dim: int) -> SiameseArm:
    return SiameseArm(
        encoder=nn.Linear(input_dim, 4),
        projector=nn.Linear(4, 3),
        transform=nn.Linear(3, 3),
        normalize_projections=True,
    )


def _model() -> SLAPModule:
    return SLAPModule(
        audio_encoder=_arm(5),
        text_encoder=_arm(2),
        loss_fn=BYOLLoss(),
        optimizer=partial(torch.optim.SGD, lr=0.1),
        scheduler=None,
        ma_callback=MovingAverageWeightUpdate(
            initial_tau=0.5,
            final_tau=0.5,
            update_method="lin",
        ),
    )


def test_siamese_arm_returns_normalized_projection_and_prediction() -> None:
    """A projected arm returns the three expected representations."""
    arm = _arm(input_dim=2)

    representation, projection, prediction = arm(torch.ones(3, 2))

    assert representation.shape == (3, 4)
    assert projection is not None
    assert prediction is not None
    assert projection.shape == prediction.shape == (3, 3)
    torch.testing.assert_close(projection.norm(dim=-1), torch.ones(3))
    torch.testing.assert_close(prediction.norm(dim=-1), torch.ones(3))


def test_byol_loss_reports_cross_and_intra_modal_terms() -> None:
    """Orthogonal modalities separate cross-modal from within-modal loss."""
    loss = BYOLLoss(ssl_weight=0.5)
    audio = torch.tensor([[1.0, 0.0]])
    text = torch.tensor([[0.0, 1.0]])

    result = loss(audio, text, audio, text)

    assert result["multimodal_loss"].item() == 2.0
    assert result["unimodal_loss"].item() == 0.0
    assert result["total_loss"].item() == 1.0


def test_training_step_updates_only_online_arm_gradients() -> None:
    """Backpropagation reaches online arms without entering target arms."""
    model = _model()
    for online, target in (
        (model.audio_encoder, model.audio_ema),
        (model.text_encoder, model.text_ema),
    ):
        for online_parameter, target_parameter in zip(
            online.parameters(), target.parameters(), strict=True
        ):
            torch.testing.assert_close(online_parameter, target_parameter)

    batch = {
        "audio": torch.randn(2, 5),
        "params": torch.randn(2, 2),
    }

    loss = model.training_step(batch, batch_idx=0)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.audio_encoder.parameters())
    assert all(parameter.grad is not None for parameter in model.text_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.audio_ema.parameters())
    assert all(parameter.grad is None for parameter in model.text_ema.parameters())


def test_moving_average_moves_target_toward_online_weights() -> None:
    """A half-retention update places target weights at the midpoint."""
    online = nn.Linear(1, 1, bias=False)
    target = nn.Linear(1, 1, bias=False)
    online.weight.data.fill_(2.0)
    target.weight.data.zero_()
    update = MovingAverageWeightUpdate(
        initial_tau=0.5,
        final_tau=0.5,
        update_method="lin",
    )

    update.update_weights(online, target)

    torch.testing.assert_close(target.weight, torch.ones_like(target.weight))
