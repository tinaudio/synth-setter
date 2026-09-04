"""Behavior tests for the SLAP Lightning model."""

import inspect
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Literal, cast

import pytest
import torch
from lightning.pytorch import Callback, Trainer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from synth_setter.models.components.slap import BYOLLoss, SiameseArm
from synth_setter.models.components.slap_ema import MovingAverageWeightUpdate
from synth_setter.models.slap_module import SLAPModule
from tests.helpers.run_if import RunIf


def _arm(input_dim: int, *, batch_norm: bool = False) -> SiameseArm:
    encoder_layers: list[nn.Module] = [nn.Linear(input_dim, 4)]
    if batch_norm:
        encoder_layers.append(nn.BatchNorm1d(4))
    return SiameseArm(
        encoder=nn.Sequential(*encoder_layers),
        projector=nn.Linear(4, 3),
        transform=nn.Linear(3, 3),
        normalize_projections=True,
    )


def _arm_with_named_transform(input_dim: int) -> SiameseArm:
    encoder = nn.Sequential(
        OrderedDict([("transform", nn.Linear(input_dim, 4))]),
    )
    return SiameseArm(encoder, nn.Linear(4, 3), nn.Linear(3, 3))


def _model(
    *,
    moving_average: MovingAverageWeightUpdate | None = None,
    batch_norm: bool = False,
    loss_fn: BYOLLoss | None = None,
) -> SLAPModule:
    return SLAPModule(
        audio_encoder=_arm(5, batch_norm=batch_norm),
        text_encoder=_arm(2, batch_norm=batch_norm),
        loss_fn=loss_fn or BYOLLoss(),
        optimizer=partial(torch.optim.SGD, lr=0.1),
        scheduler=None,
        ma_callback=moving_average
        or MovingAverageWeightUpdate(
            initial_tau=0.5,
            final_tau=0.5,
            update_method="lin",
        ),
    )


def _loader(num_batches: int) -> DataLoader[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(19)
    rows = [
        {
            "audio": torch.randn(5, generator=generator),
            "params": torch.randn(2, generator=generator),
        }
        for _ in range(num_batches * 2)
    ]
    return DataLoader(cast(Dataset[dict[str, torch.Tensor]], rows), batch_size=2)


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


def test_siamese_arm_without_projector_normalizes_representation() -> None:
    """Representation normalization does not depend on a projector being present."""
    arm = SiameseArm(encoder=nn.Identity(), normalize_representations=True)

    representation, projection, prediction = arm(torch.tensor([[3.0, 4.0]]))

    torch.testing.assert_close(representation, torch.tensor([[0.6, 0.8]]))
    assert projection is None
    assert prediction is None


def test_byol_loss_reports_cross_and_intra_modal_terms() -> None:
    """Orthogonal modalities separate cross-modal from within-modal loss."""
    loss = BYOLLoss(ssl_weight=0.5)
    audio = torch.tensor([[1.0, 0.0]])
    text = torch.tensor([[0.0, 1.0]])

    result = loss(audio, text, audio, text)

    assert result["multimodal_loss"].item() == 2.0
    assert result["unimodal_loss"].item() == 0.0
    assert result["total_loss"].item() == 1.0


def test_slap_loss_keeps_audio_prediction_in_first_byol_position() -> None:
    """The named audio-to-text loss uses the audio online prediction."""
    audio_transform = nn.Linear(2, 2, bias=False)
    text_transform = nn.Linear(2, 2, bias=False)
    audio_arm = SiameseArm(
        encoder=nn.Identity(),
        projector=nn.Identity(),
        transform=audio_transform,
    )
    text_arm = SiameseArm(
        encoder=nn.Identity(),
        projector=nn.Identity(),
        transform=text_transform,
    )
    with torch.no_grad():
        audio_transform.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        text_transform.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    model = SLAPModule(
        audio_encoder=audio_arm,
        text_encoder=text_arm,
        loss_fn=BYOLLoss(out_key="a_t_loss", unimodal=False),
        optimizer=partial(torch.optim.SGD, lr=0.1),
    )
    batch = {
        "audio": torch.tensor([[1.0, 0.0]]),
        "params": torch.tensor([[0.0, 1.0]]),
    }

    loss = model.training_step(batch, batch_idx=0)

    assert loss.item() == pytest.approx(0.0)


def test_slap_configured_audio_input_uses_stored_mel() -> None:
    """Selecting mel must leave the incompatible waveform tensor unused."""
    model = SLAPModule(
        audio_encoder=_arm(5),
        text_encoder=_arm(2),
        loss_fn=BYOLLoss(),
        optimizer=partial(torch.optim.SGD, lr=0.1),
        audio_input_key="mel",
    )
    batch = {
        "audio": torch.randn(2, 7),
        "mel": torch.randn(2, 5),
        "params": torch.randn(2, 2),
    }

    loss = model.training_step(batch, batch_idx=0)

    assert torch.isfinite(loss)


def test_slap_optional_dependencies_are_keyword_only() -> None:
    """Optional factories cannot be positionally misbound."""
    parameters = inspect.signature(SLAPModule).parameters

    assert parameters["scheduler"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["ma_callback"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["compile"].kind is inspect.Parameter.KEYWORD_ONLY


def test_training_step_updates_only_online_arm_gradients() -> None:
    """Backpropagation reaches online arms without entering target arms."""
    model = _model()
    batch = {
        "audio": torch.randn(2, 5),
        "params": torch.randn(2, 2),
    }

    loss = model.training_step(batch, batch_idx=0)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.audio_encoder.parameters())
    assert all(parameter.grad is not None for parameter in model.text_encoder.parameters())
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.audio_encoder.parameters()
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.text_encoder.parameters()
    )
    assert all(parameter.grad is None for parameter in model.audio_ema.parameters())
    assert all(parameter.grad is None for parameter in model.text_ema.parameters())


def test_target_arms_do_not_retain_unused_predictors() -> None:
    """Moving-average arms stop at projection and carry no predictor state."""
    model = _model()

    assert isinstance(model.audio_ema.transform, nn.Identity)
    assert isinstance(model.text_ema.transform, nn.Identity)


@pytest.mark.parametrize("compiled", [False, True])
def test_legacy_target_predictor_keys_load_strictly(compiled: bool) -> None:
    """Checkpoint migration removes target-predictor tensors.

    :param compiled: Whether checkpoint keys carry Torch's compiled-module prefix.
    """
    model = _model()
    state_dict = model.state_dict()
    compiled_prefix = "_orig_mod." if compiled else ""
    for name, value in model.audio_encoder.transform.state_dict().items():
        state_dict[f"audio_ema.{compiled_prefix}transform.{name}"] = value.clone()
    for name, value in model.text_encoder.transform.state_dict().items():
        state_dict[f"text_ema.{compiled_prefix}transform.{name}"] = value.clone()
    checkpoint: dict[str, object] = {"state_dict": state_dict}

    model.on_load_checkpoint(checkpoint)

    model.load_state_dict(cast(dict[str, torch.Tensor], checkpoint["state_dict"]), strict=True)


def test_checkpoint_migration_preserves_nested_transform_modules() -> None:
    """Only the target arm's predictor is removed from checkpoint state."""
    model = SLAPModule(
        audio_encoder=_arm_with_named_transform(5),
        text_encoder=_arm_with_named_transform(2),
        loss_fn=BYOLLoss(),
        optimizer=partial(torch.optim.SGD, lr=0.1),
    )
    checkpoint: dict[str, object] = {"state_dict": model.state_dict()}

    model.on_load_checkpoint(checkpoint)

    state_dict = cast(dict[str, torch.Tensor], checkpoint["state_dict"])
    assert "audio_ema.encoder.transform.weight" in state_dict
    assert "text_ema.encoder.transform.weight" in state_dict
    model.load_state_dict(state_dict, strict=True)


def test_target_arms_remain_frozen_and_eval_when_parent_trains() -> None:
    """Lightning train-mode propagation cannot enable target gradients or BatchNorm updates."""
    model = _model(batch_norm=True)

    model.train()

    assert model.audio_encoder.training
    assert model.text_encoder.training
    assert not model.audio_ema.training
    assert not model.text_ema.training
    assert all(not parameter.requires_grad for parameter in model.audio_ema.parameters())
    assert all(not parameter.requires_grad for parameter in model.text_ema.parameters())


def test_moving_average_updates_floating_buffers_and_copies_integral_buffers() -> None:
    """EMA gives BatchNorm running statistics and counters explicit update semantics."""
    online = nn.BatchNorm1d(2)
    target = nn.BatchNorm1d(2)
    assert online.running_mean is not None
    assert target.running_mean is not None
    assert online.num_batches_tracked is not None
    assert target.num_batches_tracked is not None
    online.running_mean.copy_(torch.tensor([2.0, 4.0]))
    target.running_mean.zero_()
    online.num_batches_tracked.fill_(7)
    target.num_batches_tracked.zero_()
    update = MovingAverageWeightUpdate(initial_tau=0.5, final_tau=0.5)

    update.update_weights(online, target, tau=0.5)

    torch.testing.assert_close(target.running_mean, torch.tensor([1.0, 2.0]))
    assert target.num_batches_tracked.item() == 7


@pytest.mark.parametrize("every_n_steps", [0, -1])
def test_moving_average_rejects_nonpositive_cadence(every_n_steps: int) -> None:
    """A cadence must identify a positive optimizer-step interval.

    :param every_n_steps: Invalid cadence under test.
    """
    with pytest.raises(ValueError, match="every_n_steps must be positive"):
        MovingAverageWeightUpdate(every_n_steps=every_n_steps)


@pytest.mark.parametrize(
    ("method", "completed_step", "expected"),
    [
        ("lin", 5, 0.9),
        ("cos", 5, 0.9),
        ("exp", 5, 0.8 + 0.4 * (1 - 2**-0.5)),
        ("lin", 11, 1.0),
        ("cos", 11, 1.0),
        ("exp", 11, 1.0),
    ],
)
def test_moving_average_tau_uses_completed_step_and_clamps_at_final(
    method: Literal["cos", "exp", "lin"],
    completed_step: int,
    expected: float,
) -> None:
    """Every schedule derives tau from completed optimizer-step progress.

    :param method: Interpolation schedule under test.
    :param completed_step: Completed optimizer-step count.
    :param expected: Expected retention at that step.
    """
    update = MovingAverageWeightUpdate(
        initial_tau=0.8,
        final_tau=1.0,
        update_method=method,
    )

    assert update.tau_at_step(completed_step, total_steps=10) == pytest.approx(expected)
    assert not hasattr(update, "current_tau")


def test_moving_average_updates_shared_state_once() -> None:
    """Aliased module state receives one interpolation per optimizer step."""
    online_weight = nn.Linear(1, 1, bias=False)
    target_weight = nn.Linear(1, 1, bias=False)
    online = nn.ModuleDict({"first": online_weight, "second": online_weight})
    target = nn.ModuleDict({"first": target_weight, "second": target_weight})
    online_weight.weight.data.fill_(2.0)
    target_weight.weight.data.zero_()
    update = MovingAverageWeightUpdate(initial_tau=0.5, final_tau=0.5)

    update.update_weights(online, target, tau=0.5)

    torch.testing.assert_close(target_weight.weight, torch.ones_like(target_weight.weight))


def test_moving_average_moves_target_toward_online_weights() -> None:
    """A half-retention update places target weights at the midpoint."""
    online = nn.Linear(1, 1, bias=False)
    target = nn.Linear(1, 1, bias=False)
    online.weight.data.fill_(2.0)
    target.weight.data.zero_()
    update = MovingAverageWeightUpdate(initial_tau=0.5, final_tau=0.5)

    update.update_weights(online, target, tau=0.5)

    torch.testing.assert_close(target.weight, torch.ones_like(target.weight))


def test_trainer_updates_targets_on_effective_steps_across_epoch_boundaries() -> None:
    """Accumulated batches trigger EMA only after Lightning optimizer steps."""
    moving_average = MovingAverageWeightUpdate(
        initial_tau=0.5,
        final_tau=1.0,
        every_n_steps=2,
        update_method="lin",
    )
    model = _model(moving_average=moving_average, batch_norm=True)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        accumulate_grad_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_dataloaders=_loader(num_batches=3))

    assert trainer.global_step == 4
    assert trainer.callback_metrics["MA rate"].item() == 1.0
    audio_batch_norm = cast(nn.BatchNorm1d, cast(nn.Sequential, model.audio_ema.encoder)[1])
    text_batch_norm = cast(nn.BatchNorm1d, cast(nn.Sequential, model.text_ema.encoder)[1])
    assert audio_batch_norm.num_batches_tracked is not None
    assert text_batch_norm.num_batches_tracked is not None
    assert audio_batch_norm.num_batches_tracked.item() == 6
    assert text_batch_norm.num_batches_tracked.item() == 6


class _StopAfterStep(Callback):
    """Stop a fit after a requested effective optimizer step."""

    def __init__(self, completed_step: int) -> None:
        """Configure the effective optimizer step that stops fitting.

        :param completed_step: Optimizer-step count at which to stop.
        """
        self.completed_step = completed_step

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: SLAPModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Request a clean stop once Lightning reaches the selected step.

        :param trainer: Active trainer whose loop may be stopped.
        :param pl_module: Unused SLAP model.
        :param outputs: Unused training-step outputs.
        :param batch: Unused training batch.
        :param batch_idx: Unused epoch-local batch index.
        """
        del pl_module, outputs, batch, batch_idx
        if trainer.global_step >= self.completed_step:
            trainer.should_stop = True


def test_checkpoint_resume_preserves_ema_cadence_and_schedule(tmp_path: Path) -> None:
    """The first resumed EMA update derives cadence and tau from restored steps.

    :param tmp_path: Isolated checkpoint directory.
    """
    moving_average = MovingAverageWeightUpdate(
        initial_tau=0.8,
        final_tau=1.0,
        every_n_steps=3,
        update_method="lin",
    )
    first_model = _model(moving_average=moving_average)
    first_trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=5,
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    first_trainer.fit(first_model, train_dataloaders=_loader(num_batches=4))
    checkpoint_path = tmp_path / "resume.ckpt"
    first_trainer.save_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    target_before = checkpoint["state_dict"]["audio_ema.encoder.0.weight"]

    resumed_model = _model(moving_average=moving_average)
    resumed_trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=5,
        max_steps=4,
        callbacks=[_StopAfterStep(completed_step=3)],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    resumed_trainer.fit(
        resumed_model,
        train_dataloaders=_loader(num_batches=4),
        ckpt_path=checkpoint_path,
    )

    assert resumed_trainer.global_step == 3
    expected_tau = 0.95
    online_encoder = cast(nn.Sequential, resumed_model.audio_encoder.encoder)
    target_encoder = cast(nn.Sequential, resumed_model.audio_ema.encoder)
    online_after = cast(nn.Linear, online_encoder[0]).weight.detach()
    expected_target = target_before * expected_tau + online_after * (1 - expected_tau)
    torch.testing.assert_close(cast(nn.Linear, target_encoder[0]).weight, expected_target)
    assert resumed_trainer.callback_metrics["MA rate"].item() == pytest.approx(expected_tau)


def test_validation_loss_is_epoch_aggregated_for_checkpoint_monitoring(tmp_path: Path) -> None:
    """Validation exposes an epoch metric after more than one batch.

    :param tmp_path: Isolated Lightning output directory.
    """
    model = _model()
    loader = _loader(num_batches=2)
    batches = list(loader)
    model.eval()
    expected = torch.stack([model._losses(batch)["total_loss"] for batch in batches]).mean()
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.validate(model, dataloaders=loader)

    torch.testing.assert_close(trainer.callback_metrics["loss/val/total_loss"], expected)


@pytest.mark.slow
def test_slap_overfits_deterministic_fixed_batch() -> None:
    """The online arms can reduce SLAP loss on one deterministic batch."""
    torch.manual_seed(7)
    model = _model(
        moving_average=MovingAverageWeightUpdate(initial_tau=1.0, final_tau=1.0),
        loss_fn=BYOLLoss(out_key="multimodal_loss", unimodal=False),
    )
    batch = {
        "audio": torch.randn(4, 5),
        "params": torch.randn(4, 2),
    }
    optimizer = torch.optim.Adam(
        [
            *model.audio_encoder.parameters(),
            *model.text_encoder.parameters(),
            *model.loss_fn.parameters(),
        ],
        lr=0.01,
    )
    initial_loss = model._losses(batch)["total_loss"].item()

    for _ in range(100):
        optimizer.zero_grad()
        loss = model._losses(batch)["total_loss"]
        loss.backward()
        optimizer.step()

    final_loss = model._losses(batch)["total_loss"]
    shuffled_batch = {
        "audio": batch["audio"],
        "params": batch["params"].roll(1, dims=0),
    }
    shuffled_loss = model._losses(shuffled_batch)["total_loss"]
    assert final_loss.item() < initial_loss * 0.25
    assert final_loss.item() < 0.05
    assert final_loss.item() < shuffled_loss.item() * 0.5


@pytest.mark.gpu
@RunIf(min_gpus=1)
def test_mixed_precision_overflow_does_not_advance_ema_schedule() -> None:
    """A skipped update does not count toward EMA cadence or tau progress."""
    moving_average = MovingAverageWeightUpdate(
        initial_tau=0.5,
        final_tau=1.0,
        update_method="lin",
    )
    model = _model(moving_average=moving_average)
    for parameter in model.audio_ema.parameters():
        parameter.zero_()
    rows = [
        {
            "audio": torch.full((5,), float("inf")),
            "params": torch.ones(2),
        },
        {
            "audio": torch.ones(5),
            "params": torch.ones(2),
        },
    ]
    loader = DataLoader(cast(Dataset[dict[str, torch.Tensor]], rows), batch_size=1)
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(model, train_dataloaders=loader)

    online_weight = next(model.audio_encoder.parameters()).detach()
    expected_target = online_weight * 0.25
    torch.testing.assert_close(next(model.audio_ema.parameters()), expected_target)
    assert trainer.callback_metrics["MA rate"].item() == pytest.approx(0.75)
