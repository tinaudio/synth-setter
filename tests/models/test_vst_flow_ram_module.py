"""Behaviour tests for Reinforce Adjoint Matching post-training of a pretrained flow.

Rewards come from the production render and spectral distance; only the constant-reward cases stand
in a fixed scorer, to isolate the loss algebra from the reward's variance.
"""

from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import lightning
import pytest
import torch
from lightning import Trainer

from synth_setter.models.components.audio_distance import MultiScaleSpectralDistance
from synth_setter.models.components.audio_feedback import AudioFeedbackLoss
from synth_setter.models.components.rendered_reward import RenderedAudioReward
from synth_setter.models.components.vector_field import VectorField
from synth_setter.models.vst_flow_ram_module import VSTFlowRAMModule
from tests.models.test_vst_flow_finetune_module import (
    _BATCH,
    _CONDITIONING_DIM,
    _SAMPLE_RATE,
    _SIGNAL_LENGTH,
    _WIDTH,
    _base_checkpoint,
    _base_module,
    _batch,
    _data,
    _WaveformEncoder,
)

_EMA_DECAY = 0.9


class _NormReward(torch.nn.Module):
    """Deterministic scorer preferring small-norm rows, so sampled endpoints always differ in reward.

    Rows a tiny untrained flow samples render silence, and silence scores identically under the
    spectral reward, which would zero every advantage and leave the fit tests vacuous.

    .. attribute :: target_key

       Batch column the module hands over as the (ignored) target.
    """

    target_key = "audio"

    def forward(self, theta: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        """Return the negative row norm.

        :param theta: Sampled rows.
        :param target_audio: Ignored.
        :returns: Rewards shaped ``(batch,)``.
        """
        return -theta.norm(dim=-1)


class _ConstantReward(torch.nn.Module):
    """Scorer that rates every row identically, so no advantage survives normalisation.

    .. attribute :: target_key

       Batch column the module hands over as the (ignored) target.
    """

    target_key = "audio"

    def forward(self, theta: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        """Return one identical reward per row.

        :param theta: Sampled rows.
        :param target_audio: Ignored.
        :returns: Constant rewards shaped ``(batch,)``.
        """
        return torch.full((theta.shape[0],), 0.5)


def _ram(checkpoint: Path, overrides: dict[str, object] | None = None) -> VSTFlowRAMModule:
    """Build a RAM module of the pinned tiny base shape.

    :param checkpoint: Path to the base checkpoint to post-train.
    :param overrides: Constructor arguments replacing the test defaults.
    :returns: Configured module.
    """
    kwargs = {
        "encoder": _WaveformEncoder(),
        "vector_field": VectorField(
            field_dim=_WIDTH,
            hidden_dim=32,
            conditioning_dim=_CONDITIONING_DIM,
            num_blocks=2,
        ),
        "optimizer": torch.optim.Adam,
        "scheduler": None,
        "num_params": _WIDTH,
        "conditioning": "audio",
        "base_checkpoint": checkpoint,
        "reward": RenderedAudioReward(
            distance=MultiScaleSpectralDistance(sample_rate=_SAMPLE_RATE),
            sample_rate=_SAMPLE_RATE,
            signal_length=_SIGNAL_LENGTH,
            render_batch_size=_BATCH,
        ),
        "num_samples_per_row": 2,
        "num_targets_per_sample": 2,
        "sampling_steps": 2,
        "ema_decay": _EMA_DECAY,
        "ema_warmup_rate": None,
    }
    kwargs.update(overrides or {})
    return VSTFlowRAMModule(**kwargs)  # pyright: ignore[reportArgumentType]


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _assert_same(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value, msg=name)


def _trainer(max_steps: int = 1) -> Trainer:
    return Trainer(
        max_steps=max_steps,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        limit_val_batches=0,
        enable_progress_bar=False,
    )


def test_ram_module_from_base_checkpoint_seeds_all_policy_copies(
    tmp_path: Path,
) -> None:
    """Policy, reference, sampling EMA, and eval EMA all start from pretrained weights.

    :param tmp_path: Directory for the base checkpoint.
    """
    torch.manual_seed(7)
    base = _base_module()
    module = _ram(_base_checkpoint(tmp_path, base))

    _assert_same(_state(module.encoder), _state(base.encoder))
    for field in (
        module.vector_field,
        module.reference_field,
        module.old_field,
        module.eval_field,
    ):
        _assert_same(_state(field), _state(base.vector_field))


def test_ram_module_with_endpoint_parameterization_raises(tmp_path: Path) -> None:
    """RAM rejects endpoint prediction even when its pretrained base is valid velocity.

    :param tmp_path: Directory for the velocity-parameterized base checkpoint.
    """
    with pytest.raises(ValueError, match="RAM requires parameterization='velocity'"):
        _ram(_base_checkpoint(tmp_path), {"parameterization": "endpoint"})


def test_ram_module_with_mismatched_checkpoint_raises(tmp_path: Path) -> None:
    """A checkpoint carrying a key this model has no slot for is refused, not silently dropped.

    :param tmp_path: Directory for the base checkpoint.
    """
    state = _base_module().state_dict()
    state["vector_field.not_a_real_parameter"] = torch.zeros(1)
    path = tmp_path / "stale.ckpt"
    torch.save({"state_dict": state}, path)

    with pytest.raises(ValueError, match="unexpected"):
        _ram(path)


def test_ram_module_trains_only_the_policy_field(tmp_path: Path) -> None:
    """Encoder, reference, and old copies carry no gradient; only the policy does.

    :param tmp_path: Directory for the base checkpoint.
    """
    module = _ram(_base_checkpoint(tmp_path))

    trainable = {name for name, p in module.named_parameters() if p.requires_grad}

    assert trainable
    assert all(name.startswith("vector_field.") for name in trainable)


def test_ram_train_mode_keeps_eval_field_frozen_in_eval_mode(tmp_path: Path) -> None:
    """Training mode applies only to the policy while evaluation weights remain deterministic.

    :param tmp_path: Directory for the base checkpoint.
    """
    module = _ram(_base_checkpoint(tmp_path))

    module.train()

    assert module.vector_field.training
    assert not module.eval_field.training
    assert all(not parameter.requires_grad for parameter in module.eval_field.parameters())


def test_ram_evaluation_sampling_uses_eval_field_not_policy_or_old(tmp_path: Path) -> None:
    """Public inference integrates the eval EMA even when policy and old fields disagree.

    :param tmp_path: Directory for the base checkpoint.
    """
    module = _ram(_base_checkpoint(tmp_path))
    with torch.no_grad():
        for parameter in module.eval_field.parameters():
            parameter.zero_()
        for parameter in module.vector_field.parameters():
            parameter.fill_(0.25)
        for parameter in module.old_field.parameters():
            parameter.fill_(-0.25)
    noise = torch.zeros(2, _WIDTH)

    sampled = module.sample_batch(
        _batch(2),
        noise=noise,
        content_cfg_strength=1.0,
        sketch_cfg_strength=0.0,
        sample_steps=1,
    )

    torch.testing.assert_close(sampled, noise)


def test_ram_training_step_with_constant_reward_has_zero_loss_at_initialisation(
    tmp_path: Path,
) -> None:
    """With no advantage the target is the reference, which the fresh policy equals.

    :param tmp_path: Directory for the base checkpoint.
    """
    module = _ram(_base_checkpoint(tmp_path), overrides={"reward": _ConstantReward()})
    module.log = lambda *args, **kwargs: None  # pyright: ignore[reportAttributeAccessIssue]

    loss = module.training_step(_batch(), 0)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_ram_training_step_logs_the_mean_reward(tmp_path: Path) -> None:
    """The step exposes what the sampler earned, not just the regression loss.

    :param tmp_path: Directory for the base checkpoint.
    """
    module = _ram(_base_checkpoint(tmp_path), overrides={"reward": _ConstantReward()})
    logged: dict[str, torch.Tensor] = {}
    module.log = lambda name, value, **kwargs: logged.__setitem__(name, torch.as_tensor(value))  # pyright: ignore[reportAttributeAccessIssue]

    module.training_step(_batch(), 0)

    torch.testing.assert_close(logged["train/reward"], torch.tensor(0.5))


def test_ram_fit_moves_the_policy_and_leaves_reference_and_encoder_fixed(
    tmp_path: Path,
) -> None:
    """One rewarded step updates the policy alone; the anchors it regresses toward stay put.

    :param tmp_path: Directory for the base checkpoint.
    """
    torch.manual_seed(11)
    base = _base_module()
    module = _ram(_base_checkpoint(tmp_path, base), overrides={"reward": _NormReward()})

    _trainer().fit(module, datamodule=_data())

    assert any(
        not torch.equal(after, before)
        for after, before in zip(
            _state(module.vector_field).values(), _state(base.vector_field).values(), strict=True
        )
    )
    _assert_same(_state(module.reference_field), _state(base.vector_field))
    _assert_same(_state(module.encoder), _state(base.encoder))


def test_ram_optimizer_step_moves_the_old_field_toward_the_policy_by_ema_decay(
    tmp_path: Path,
) -> None:
    """After one step ``old = decay * old + (1 - decay) * policy`` for every parameter.

    :param tmp_path: Directory for the base checkpoint.
    """
    torch.manual_seed(13)
    base = _base_module()
    module = _ram(_base_checkpoint(tmp_path, base), overrides={"reward": _NormReward()})
    before = {name: p.detach().clone() for name, p in base.vector_field.named_parameters()}

    _trainer().fit(module, datamodule=_data())

    policy = dict(module.vector_field.named_parameters())
    assert any(
        not torch.equal(old, policy[name]) for name, old in module.old_field.named_parameters()
    ), "policy never moved, so the lag is unobservable"
    for name, old in module.old_field.named_parameters():
        expected = _EMA_DECAY * before[name] + (1 - _EMA_DECAY) * policy[name].detach()
        torch.testing.assert_close(old, expected, msg=name)


@pytest.mark.parametrize(
    ("steps", "eval_base_weight"),
    [pytest.param(1, 0.9, id="first-step"), pytest.param(3, 0.729, id="third-step")],
)
def test_ram_eval_ema_uses_fixed_decay_independently_of_old_warmup(
    tmp_path: Path, steps: int, eval_base_weight: float
) -> None:
    """The eval copy compounds fixed decay 0.9 while the warmed sampler copies policy.

    :param tmp_path: Directory for the base checkpoint.
    :param steps: Number of optimizer steps.
    :param eval_base_weight: Expected contribution from the initial evaluation weights.
    """
    torch.manual_seed(17)
    base = _base_module()
    module = _ram(
        _base_checkpoint(tmp_path, base),
        overrides={
            "ema_decay": 0.5,
            "ema_warmup_rate": 0.01,
            "optimizer": partial(torch.optim.SGD, lr=0.0),
            "reward": _NormReward(),
        },
    )
    before = {name: p.detach().clone() for name, p in base.vector_field.named_parameters()}
    with torch.no_grad():
        for parameter in module.vector_field.parameters():
            parameter.add_(0.01)

    _trainer(max_steps=steps).fit(module, datamodule=_data())

    policy = dict(module.vector_field.named_parameters())
    _assert_same(_state(module.old_field), _state(module.vector_field))
    for name, evaluation in module.eval_field.named_parameters():
        expected = eval_base_weight * before[name] + (1 - eval_base_weight) * policy[name].detach()
        torch.testing.assert_close(evaluation, expected, rtol=1e-6, atol=1e-8, msg=name)


def test_ram_ema_warmup_copies_the_policy_on_the_first_step(tmp_path: Path) -> None:
    """A warmup rate makes the very first lag ``min(rate * step, decay)``, i.e. a full copy.

    :param tmp_path: Directory for the base checkpoint.
    """
    torch.manual_seed(17)
    module = _ram(
        _base_checkpoint(tmp_path), overrides={"ema_warmup_rate": 0.01, "reward": _NormReward()}
    )

    _trainer().fit(module, datamodule=_data())

    _assert_same(_state(module.old_field), _state(module.vector_field))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param(
            {
                "audio_loss": AudioFeedbackLoss(
                    lambda_audio=0.1,
                    t_min=0.0,
                    sample_rate=_SAMPLE_RATE,
                    signal_length=_SIGNAL_LENGTH,
                    render_batch_size=_BATCH,
                    distance=MultiScaleSpectralDistance(sample_rate=_SAMPLE_RATE),
                )
            },
            "audio_loss",
            id="audio-loss",
        ),
        pytest.param({"compile": True}, "compile", id="compile"),
        pytest.param({"rectified_sigma_min": 0.1}, "rectified_sigma_min", id="sigma"),
        pytest.param({"num_samples_per_row": 1}, "num_samples_per_row", id="single-sample"),
        pytest.param({"reward_multiplier": 0.0}, "reward_multiplier", id="zero-multiplier"),
        pytest.param({"ema_decay": 1.0}, "ema_decay", id="frozen-ema"),
        pytest.param({"ema_warmup_rate": float("nan")}, "ema_warmup_rate", id="nan-warmup"),
        pytest.param(
            {"time_power_law_alpha": float("nan")}, "time_power_law_alpha", id="nan-alpha"
        ),
        pytest.param(
            {"sampling_cfg_strength": float("nan")}, "sampling_cfg_strength", id="nan-cfg"
        ),
        pytest.param({"reward": torch.nn.Identity()}, "target_key", id="reward-without-target"),
    ],
)
def test_ram_module_rejects_configurations_it_cannot_serve(
    tmp_path: Path, overrides: dict[str, object], match: str
) -> None:
    """Each unsupported or degenerate setting fails at construction, naming the setting.

    :param tmp_path: Directory for the base checkpoint.
    :param overrides: Constructor arguments under test.
    :param match: Setting name the error must mention.
    """
    with pytest.raises(ValueError, match=match):
        _ram(_base_checkpoint(tmp_path), overrides=overrides)


def test_ram_on_train_start_rejects_multi_device_runs(tmp_path: Path) -> None:
    """The render reward mutates one shared voice, so a multi-rank fit is refused up front.

    :param tmp_path: Directory for the base checkpoint.
    """
    module = _ram(_base_checkpoint(tmp_path))
    two_ranks = SimpleNamespace(world_size=2)

    with (
        patch.object(
            VSTFlowRAMModule, "trainer", new_callable=PropertyMock, return_value=two_ranks
        ),
        pytest.raises(ValueError, match="single-device"),
    ):
        module.on_train_start()


def test_ram_loss_reaches_every_policy_parameter(tmp_path: Path) -> None:
    """A rewarded step sends finite, non-zero gradient to every trainable parameter.

    :param tmp_path: Directory for the base checkpoint.
    """
    torch.manual_seed(19)
    module = _ram(_base_checkpoint(tmp_path), overrides={"reward": _NormReward()})
    module.log = lambda *args, **kwargs: None  # pyright: ignore[reportAttributeAccessIssue]

    module.training_step(_batch(), 0).backward()

    for name, parameter in module.named_parameters():
        # RAM never drops conditioning, so the CFG token is exercised only by the sampler.
        if not parameter.requires_grad or name.endswith("cfg_dropout_token"):
            continue
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


@pytest.mark.slow
def test_ram_overfits_a_fixed_sampled_batch(tmp_path: Path) -> None:
    """Repeated steps on one frozen draw of endpoints, times, and noise drive the loss toward zero.

    Reseeding before every step replays the same endpoints, rewards, flow times, and noise, and the
    lagged sampler is left un-updated, so the target is a fixed regression the policy must fit.

    :param tmp_path: Directory for the base checkpoint.
    """
    torch.manual_seed(29)
    module = _ram(_base_checkpoint(tmp_path), overrides={"reward": _NormReward()})
    module.log = lambda *args, **kwargs: None  # pyright: ignore[reportAttributeAccessIssue]
    optimizer = torch.optim.Adam(module.vector_field.parameters(), lr=1e-2)
    batch = _batch()

    def step() -> float:
        torch.manual_seed(23)
        loss = module.training_step(batch, 0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    initial = step()
    for _ in range(299):
        final = step()

    assert initial > 0
    assert final < initial * 0.05


def test_ram_training_step_depends_on_the_target_audio(tmp_path: Path) -> None:
    """With every random draw replayed, changing only the target audio changes the RAM loss.

    The scorer here ignores audio, so any difference can come only from the conditioning the
    encoder feeds to the sampler and the policy.

    :param tmp_path: Directory for the base checkpoint.
    """
    module = _ram(_base_checkpoint(tmp_path), overrides={"reward": _NormReward()})
    module.log = lambda *args, **kwargs: None  # pyright: ignore[reportAttributeAccessIssue]
    batch = _batch(2)
    swapped = {**batch, "audio": batch["audio"].flip(0)}

    torch.manual_seed(31)
    original = module.training_step(batch, 0)
    torch.manual_seed(31)
    with_other_audio = module.training_step(swapped, 0)

    assert not torch.isclose(original, with_other_audio)


def test_ram_fit_checkpoint_load_preserves_eval_prediction(tmp_path: Path) -> None:
    """A fit checkpoint restores the eval EMA used by public prediction.

    :param tmp_path: Directory for the base and RAM checkpoints.
    """
    torch.manual_seed(37)
    module = _ram(_base_checkpoint(tmp_path), overrides={"reward": _NormReward()})
    trainer = _trainer()
    trainer.fit(module, datamodule=_data())
    noise = torch.randn(2, _WIDTH)
    expected = module.sample_batch(
        _batch(2),
        noise=noise,
        content_cfg_strength=1.0,
        sketch_cfg_strength=0.0,
        sample_steps=2,
    )
    checkpoint = tmp_path / "ram.ckpt"
    trainer.save_checkpoint(checkpoint)

    loaded = VSTFlowRAMModule.load_from_checkpoint(
        checkpoint,
        encoder=_WaveformEncoder(),
        reward=_NormReward(),
        base_checkpoint=None,
        weights_only=False,
    )
    actual = loaded.sample_batch(
        _batch(2),
        noise=noise,
        content_cfg_strength=1.0,
        sketch_cfg_strength=0.0,
        sample_steps=2,
    )

    _assert_same(_state(loaded.eval_field), _state(module.eval_field))
    torch.testing.assert_close(actual, expected)


def test_ram_legacy_checkpoint_without_eval_ema_raises(tmp_path: Path) -> None:
    """A checkpoint without the EMA trajectory is refused instead of inventing eval weights.

    :param tmp_path: Directory for the legacy checkpoint.
    """
    module = _ram(_base_checkpoint(tmp_path))
    checkpoint: dict[str, object] = {
        "state_dict": {
            name: value
            for name, value in module.state_dict().items()
            if not name.startswith("eval_field.")
        },
        "hyper_parameters": dict(module.hparams),
        "pytorch-lightning_version": lightning.__version__,
    }
    module.on_save_checkpoint(checkpoint)
    path = tmp_path / "legacy-ram.ckpt"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="eval EMA trajectory"):
        VSTFlowRAMModule.load_from_checkpoint(
            path,
            encoder=_WaveformEncoder(),
            reward=_NormReward(),
            base_checkpoint=None,
            weights_only=False,
        )


def test_ram_fit_without_base_or_resume_checkpoint_raises(tmp_path: Path) -> None:
    """A fresh post-training fit with neither weight source would tilt a random field.

    :param tmp_path: Unused output directory.
    """
    module = _ram(None, overrides={"reward": _NormReward()})  # pyright: ignore[reportArgumentType]

    with pytest.raises(ValueError, match="base_checkpoint"):
        _trainer().fit(module, datamodule=_data())
