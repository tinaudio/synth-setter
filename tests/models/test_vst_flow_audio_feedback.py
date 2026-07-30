"""The audio-feedback loss wired into the production flow module.

Exercises the integrated path: an online torchsynth dict batch through
``VSTFlowMatchingModule`` with an attached :class:`AudioFeedbackLoss`, plus the
runtime guards that refuse configurations the differentiable renderer cannot
serve.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.plugins.precision import Precision

from synth_setter.data.torchsynth_datamodule import (
    TorchSynthBatch,
    TorchSynthDataModule,
    TorchSynthDataset,
    _make_renderer,
    collate_audio_dict,
)
from synth_setter.data.vst.torchsynth_param_spec import TORCHSYNTH_FULL_PARAM_SPEC
from synth_setter.models.components.audio_distance import (
    LatentMseDistance,
    MultiScaleSpectralDistance,
)
from synth_setter.models.components.audio_feedback import AudioFeedbackLoss
from synth_setter.models.components.same_encoder import SameAudioEncoder
from synth_setter.models.components.vector_field import VectorField
from synth_setter.models.vst_flow_matching_module import (
    TrainStepOutputs,
    VSTFlowMatchingModule,
)

_SAMPLE_RATE = 44_100
_SIGNAL_LENGTH = 4_410
_BATCH = 4
_ENCODED_WIDTH = TORCHSYNTH_FULL_PARAM_SPEC.encoded_width
_BUFFER_SECONDS = _SIGNAL_LENGTH / _SAMPLE_RATE
_CONDITIONING_DIM = 32
# Peak above which a rendered row counts as audible rather than an all-zero buffer.
_AUDIBLE_PEAK = 1e-4
_AUDIBLE_ROW_POOL = 256
_OVERFIT_STEPS = 300
_OVERFIT_TOTAL_THRESHOLD = 0.1


class _WaveformEncoder(torch.nn.Module):
    """Minimal raw-audio conditioning encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(_SIGNAL_LENGTH, _CONDITIONING_DIM)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Map a waveform batch to a flat conditioning vector.

        :param audio: Audio shaped ``(batch, _SIGNAL_LENGTH)``.
        :returns: Conditioning shaped ``(batch, _CONDITIONING_DIM)``.
        """
        return self.linear(audio)


def _audio_loss() -> AudioFeedbackLoss:
    """Build an audio-feedback loss with test-sized render settings.

    :returns: Configured loss module at the shipped weight.
    """
    return AudioFeedbackLoss(
        lambda_audio=0.03,
        t_min=0.0,
        sample_rate=_SAMPLE_RATE,
        signal_length=_SIGNAL_LENGTH,
        render_batch_size=_BATCH,
        distance=MultiScaleSpectralDistance(sample_rate=_SAMPLE_RATE),
    )


def _module(
    audio_loss: AudioFeedbackLoss | None = None,
    compile: bool = False,
    cfg_dropout_rate: float = 0.1,
) -> VSTFlowMatchingModule:
    """Build a tiny flow module conditioned on raw audio.

    :param audio_loss: Optional audio-feedback term to attach.
    :param compile: Value forwarded to the module's ``compile`` flag.
    :param cfg_dropout_rate: Probability of replacing conditioning with the CFG token.
    :returns: Configured module.
    """
    return VSTFlowMatchingModule(
        encoder=_WaveformEncoder(),
        vector_field=VectorField(
            field_dim=_ENCODED_WIDTH,
            hidden_dim=32,
            conditioning_dim=_CONDITIONING_DIM,
            num_blocks=2,
        ),
        optimizer=torch.optim.Adam,  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=_ENCODED_WIDTH,
        conditioning="audio",
        audio_loss=audio_loss,
        cfg_dropout_rate=cfg_dropout_rate,
        compile=compile,
    )


def _datamodule(batch_size: int = _BATCH, train_size: int = 8) -> TorchSynthDataModule:
    """Build a tiny online datamodule already set up for fitting.

    :param batch_size: Number of online examples per batch.
    :param train_size: Number of online training rows per epoch.
    :returns: The datamodule.
    """
    datamodule = TorchSynthDataModule(
        sample_rate=_SAMPLE_RATE,
        signal_length=_SIGNAL_LENGTH,
        train_val_test_sizes=(train_size, 4, 4),
        train_val_test_seeds=(1, 2, 3),
        batch_size=batch_size,
        num_workers=0,
    )
    datamodule.setup("fit")
    return datamodule


def _audible_online_batch(rows: int = _BATCH) -> TorchSynthBatch:
    """Collate a real online batch from rows whose note sounds inside the render buffer.

    The spec draws each row's note window across a multi-second range, so on this short
    test buffer most online rows start their note past the end and render silence. An
    audio term measured on those rows compares silence against silence.

    :param rows: Number of audible rows to collate.
    :returns: The online batch contract: ``params``, ``noise``, and ``audio``.
    :raises AssertionError: If the scanned pool holds fewer audible rows than requested.
    """
    dataset = TorchSynthDataset(
        _AUDIBLE_ROW_POOL, seed=1, sample_rate=_SAMPLE_RATE, signal_length=_SIGNAL_LENGTH
    )
    audible = []
    for index in range(_AUDIBLE_ROW_POOL):
        item = dataset[index]
        if item[0].abs().max() > _AUDIBLE_PEAK:
            audible.append(item)
            if len(audible) == rows:
                return collate_audio_dict(audible)
    raise AssertionError(f"only {len(audible)} of {rows} audible rows in {_AUDIBLE_ROW_POOL}")


def _audible_rows(rows: int, seed: int) -> torch.Tensor:
    """Draw encoded rows whose note sounds across the whole render buffer.

    Uniform-random note columns decode to a note starting anywhere in the spec's
    multi-second range — past the end of this short test buffer, so both sides of a
    render comparison would be silence and agree for the wrong reason.

    :param rows: Number of rows to draw.
    :param seed: Seed for the synth columns.
    :returns: Encoded rows shaped ``(rows, encoded_width)`` in ``[0, 1]``.
    """
    import numpy as np

    synth_values, _ = TORCHSYNTH_FULL_PARAM_SPEC.sample(np.random.default_rng(0))
    reference = TORCHSYNTH_FULL_PARAM_SPEC.encode(
        synth_values, {"pitch": 60, "note_start_and_end": (0.0, _BUFFER_SECONDS)}
    )
    note_tail = torch.from_numpy(reference)[TORCHSYNTH_FULL_PARAM_SPEC.synth_columns.stop :]
    synth = torch.rand(
        rows,
        TORCHSYNTH_FULL_PARAM_SPEC.synth_param_length,
        generator=torch.Generator().manual_seed(seed),
    )
    return torch.cat([synth, note_tail.expand(rows, -1)], dim=1)


def _synthetic_batch() -> dict[str, torch.Tensor]:
    """Build a render-free batch matching the online dict contract.

    :returns: Batch with random params, noise, and audio.
    """
    generator = torch.Generator().manual_seed(0)
    return {
        "params": torch.rand(_BATCH, _ENCODED_WIDTH, generator=generator) * 2 - 1,
        "noise": torch.randn(_BATCH, _ENCODED_WIDTH, generator=generator),
        "audio": torch.randn(_BATCH, _SIGNAL_LENGTH, generator=generator),
    }


def test_train_step_with_audio_loss_backprops_a_finite_nonzero_audio_term() -> None:
    """A real online batch produces a finite audio term with gradients in the flow."""
    torch.manual_seed(0)
    module = _module(audio_loss=_audio_loss())
    batch = _audible_online_batch()

    outputs = module._train_step(batch)
    assert outputs.audio_term is not None
    encoder_gradients = torch.autograd.grad(
        outputs.loss, tuple(module.encoder.parameters()), retain_graph=True, allow_unused=True
    )
    total = outputs.loss + outputs.audio_term
    total.backward()

    assert torch.isfinite(total)
    assert outputs.audio_term.item() > 0.0
    assert any(
        gradient is not None
        and torch.isfinite(gradient).all()
        and torch.count_nonzero(gradient).item() > 0
        for gradient in encoder_gradients
    )
    # Every trainable field parameter must join the graph; only the CFG dropout token
    # may legitimately sit out of a step where no row was dropped.
    for name, parameter in module.vector_field.named_parameters():
        if "dropout" in name:
            continue
        assert parameter.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} gradient is non-finite"
        assert (parameter.grad != 0).any(), f"{name} gradient is identically zero"


def _train_step_under_probe(probe: bool) -> tuple[TrainStepOutputs, torch.Tensor, torch.Tensor]:
    """Run one train step with the gradient probe forced on or off.

    :param probe: Whether ``_should_probe_gradient_balance`` reports True.
    :returns: The step outputs, the RNG state left behind, and the flow's input-layer grad.
    """
    torch.manual_seed(0)
    module = _module(audio_loss=_audio_loss())
    module._should_probe_gradient_balance = lambda: probe  # pyright: ignore[reportAttributeAccessIssue]
    batch = _synthetic_batch()
    # Warm the cached renderer first: building its voice draws from the global RNG, which
    # would otherwise read as a probe-induced difference whenever the cache starts cold.
    module._train_step(batch)

    torch.manual_seed(123)
    outputs = module._train_step(batch)
    rng_state = torch.get_rng_state()
    assert outputs.audio_term is not None
    (outputs.loss + outputs.audio_term).backward()

    input_gradient = dict(module.vector_field.named_parameters())["input.weight"].grad
    assert input_gradient is not None
    return outputs, rng_state, input_gradient.clone()


def test_gradient_probe_leaves_the_rng_stream_where_an_unprobed_step_leaves_it() -> None:
    """The diagnostic must not consume RNG, or enabling it reshuffles every later draw."""
    _, probed_rng, _ = _train_step_under_probe(probe=True)
    _, unprobed_rng, _ = _train_step_under_probe(probe=False)

    assert torch.equal(probed_rng, unprobed_rng)


def test_gradient_probe_does_not_change_the_losses_the_step_returns() -> None:
    """Measuring the gradient split must not move the objective it measures."""
    probed, _, _ = _train_step_under_probe(probe=True)
    unprobed, _, _ = _train_step_under_probe(probe=False)

    assert probed.audio_term is not None and unprobed.audio_term is not None
    assert torch.equal(probed.loss, unprobed.loss)
    assert torch.equal(probed.audio_term, unprobed.audio_term)


def test_gradient_probe_does_not_change_the_gradients_the_step_accumulates() -> None:
    """The probe's extra backward passes must not add into ``.grad``."""
    _, _, probed_gradient = _train_step_under_probe(probe=True)
    _, _, unprobed_gradient = _train_step_under_probe(probe=False)

    assert torch.equal(probed_gradient, unprobed_gradient)


def test_gradient_probe_reports_one_audio_gradient_norm_per_batch_row() -> None:
    """The t-bucketed diagnostic needs un-reduced per-row norms to bucket."""
    outputs, _, _ = _train_step_under_probe(probe=True)

    assert outputs.grad_balance is not None
    assert outputs.grad_balance.audio_row_norms.shape == (_BATCH,)


def test_training_step_logs_an_audio_gradient_norm_for_every_populated_time_bucket() -> None:
    """The profile reaches the logger, so the envelope is measurable from a run."""
    torch.manual_seed(0)
    module = _module(audio_loss=_audio_loss())
    module._sample_time = lambda n, device: torch.linspace(  # pyright: ignore[reportAttributeAccessIssue]
        0.1, 0.9, n, device=device
    ).unsqueeze(-1)
    trainer = Trainer(
        fast_dev_run=True, accelerator="cpu", logger=False, enable_checkpointing=False
    )

    trainer.fit(module, datamodule=_datamodule())

    logged = {key for key in trainer.logged_metrics if "audio_grad_norm_t_bucket" in key}
    assert logged == {f"train/audio_grad_norm_t_bucket_{index}" for index in range(4)}


def _overfit_one_fixed_example() -> tuple[float, float]:
    """Overfit one deterministic online example.

    :returns: Initial and final combined objectives.
    """
    _make_renderer.cache_clear()
    torch.manual_seed(0)
    module = _module(audio_loss=_audio_loss(), cfg_dropout_rate=0.0)
    module._sample_time = lambda n, device: torch.full((n, 1), 0.9, device=device)
    batch = _audible_online_batch(rows=1)
    optimizer = torch.optim.Adam(module.parameters(), lr=3e-4)
    initial_total: float | None = None

    for _ in range(_OVERFIT_STEPS):
        outputs = module._train_step(batch)
        assert outputs.audio_term is not None
        total = outputs.loss + outputs.audio_term
        if initial_total is None:
            initial_total = total.item()
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

    final = module._train_step(batch)
    assert initial_total is not None
    assert final.audio_term is not None
    return initial_total, (final.loss + final.audio_term).item()


@pytest.mark.slow
def test_combined_audio_objective_overfits_one_fixed_online_example() -> None:
    """The integrated objective can fit one fixed example to near zero."""
    initial_total, final_total = _overfit_one_fixed_example()

    assert final_total < initial_total * 0.1
    assert final_total < _OVERFIT_TOTAL_THRESHOLD


def test_train_step_without_audio_loss_returns_no_audio_term() -> None:
    """The default module pays for no render and reports no audio term."""
    torch.manual_seed(0)
    module = _module()

    outputs = module._train_step(_synthetic_batch())

    assert outputs.audio_term is None
    assert torch.isfinite(outputs.loss)


def test_module_with_audio_loss_and_nonzero_sigma_min_raises() -> None:
    """A sigma-bearing path makes the one-step estimate inexact, so refuse it."""
    with pytest.raises(ValueError, match="rectified_sigma_min"):
        VSTFlowMatchingModule(
            encoder=_WaveformEncoder(),
            vector_field=VectorField(
                field_dim=_ENCODED_WIDTH,
                hidden_dim=32,
                conditioning_dim=_CONDITIONING_DIM,
                num_blocks=2,
            ),
            optimizer=torch.optim.Adam,  # pyright: ignore[reportArgumentType]
            scheduler=None,  # pyright: ignore[reportArgumentType]
            num_params=_ENCODED_WIDTH,
            conditioning="audio",
            audio_loss=_audio_loss(),
            rectified_sigma_min=0.01,
            compile=False,
        )


def test_audio_loss_keep_mask_zeroes_cfg_dropped_rows() -> None:
    """Rows dropped by CFG must contribute nothing to the audio term."""
    torch.manual_seed(0)
    loss = _audio_loss()
    batch = _audible_online_batch()
    theta_hat = torch.rand(_BATCH, _ENCODED_WIDTH) * 2 - 1
    t = torch.full((_BATCH, 1), 0.9)

    all_dropped = loss(theta_hat, t, batch["audio"], keep=torch.zeros(_BATCH))
    all_kept = loss(theta_hat, t, batch["audio"], keep=torch.ones(_BATCH))

    assert all_dropped.item() == 0.0
    assert all_kept.item() > 0.0


def test_grad_render_matches_the_row_at_a_time_production_render() -> None:
    """Batch grad renders align noise with the per-row target renderer.

    Without the chunk-0 noise alignment, every row past the first compares against a target
    rendered with a different noise realization.
    """
    from synth_setter.data.torchsynth_datamodule import render_torchsynth
    from synth_setter.data.torchsynth_grad_render import render_torchsynth_grad

    params01 = _audible_rows(_BATCH, 7)
    row_targets = torch.cat(
        [
            render_torchsynth(
                row.unsqueeze(0),
                sample_rate=_SAMPLE_RATE,
                signal_length=_SIGNAL_LENGTH,
            )
            for row in params01
        ]
    )
    with torch.no_grad():
        batched = render_torchsynth_grad(
            params01,
            sample_rate=_SAMPLE_RATE,
            signal_length=_SIGNAL_LENGTH,
            render_batch_size=_BATCH,
        ).clamp(-1, 1)

    assert torch.allclose(batched, row_targets, atol=1e-5)


def test_module_with_audio_loss_and_torch_compile_raises_at_construction() -> None:
    """Compiling over the functional_call render miscompiles, so refuse up front."""
    with pytest.raises(ValueError, match="compile"):
        _module(audio_loss=_audio_loss(), compile=True)


def test_fit_with_audio_loss_over_a_trailing_partial_batch_completes() -> None:
    """A split that does not divide the batch size trains: the render pads the remainder."""
    torch.manual_seed(0)
    module = _module(audio_loss=_audio_loss())
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        limit_val_batches=0,
    )

    trainer.fit(module, datamodule=_datamodule(train_size=_BATCH + 1))

    assert trainer.state.finished


def test_fit_with_audio_loss_on_the_online_datamodule_completes_one_step() -> None:
    """The supported configuration trains end to end through the real datamodule."""
    torch.manual_seed(0)
    module = _module(audio_loss=_audio_loss())
    trainer = Trainer(
        fast_dev_run=True, accelerator="cpu", logger=False, enable_checkpointing=False
    )

    trainer.fit(module, datamodule=_datamodule())

    assert trainer.state.finished


def test_non_finite_gradient_raises_before_clipping_scales_every_parameter() -> None:
    """A NaN gradient must fail at its source, not silently poison every weight."""
    module = _module()
    optimizer = torch.optim.Adam(module.parameters())
    for parameter in module.parameters():
        parameter.grad = torch.zeros_like(parameter)
    poisoned = next(iter(module.vector_field.parameters()))
    non_finite_grad = torch.zeros_like(poisoned)
    non_finite_grad[0] = float("nan")
    poisoned.grad = non_finite_grad

    with pytest.raises(ValueError, match="non-finite gradient"):
        module.configure_gradient_clipping(optimizer, gradient_clip_val=1.0)


def test_finite_gradients_are_clipped_to_the_configured_norm() -> None:
    """Ordinary large-but-finite gradients still clip instead of raising."""
    module = _module()
    optimizer = torch.optim.Adam(module.parameters())
    for parameter in module.parameters():
        parameter.grad = torch.full_like(parameter, 10.0)
    module._trainer = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        precision_plugin=Precision(),
        model=module,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    module.configure_gradient_clipping(optimizer, gradient_clip_val=1.0)

    total = torch.linalg.vector_norm(
        torch.stack([p.grad.norm() for p in module.parameters() if p.grad is not None])
    )
    assert total.item() == pytest.approx(1.0, abs=1e-3)


def test_non_finite_gradient_error_names_every_affected_parameter() -> None:
    """One name cannot show whether the corruption is encoder-local or global."""
    module = _module()
    optimizer = torch.optim.Adam(module.parameters())
    for parameter in module.parameters():
        parameter.grad = torch.zeros_like(parameter)
    encoder_parameter = next(iter(module.encoder.parameters()))
    field_parameter = next(iter(module.vector_field.parameters()))
    for parameter in (encoder_parameter, field_parameter):
        corrupted = torch.zeros_like(parameter)
        corrupted[0] = float("nan")
        parameter.grad = corrupted

    with pytest.raises(ValueError) as failure:
        module.configure_gradient_clipping(optimizer, gradient_clip_val=1.0)

    assert "encoder." in str(failure.value)
    assert "vector_field." in str(failure.value)


def test_module_accepts_a_weight_normalized_audio_loss_encoder(
    tiny_same_checkpoint: Path,
) -> None:
    """Lightning deep-copies its saved hyperparameters, which weight norm cannot survive.

    :param tiny_same_checkpoint: Loadable SAME checkpoint.
    """
    same_loss = AudioFeedbackLoss(
        lambda_audio=0.03,
        t_min=0.0,
        sample_rate=_SAMPLE_RATE,
        signal_length=_SIGNAL_LENGTH,
        render_batch_size=_BATCH,
        distance=LatentMseDistance(
            encoder=SameAudioEncoder.from_pretrained(
                sample_rate=_SAMPLE_RATE, checkpoint=str(tiny_same_checkpoint)
            )
        ),
    )

    module = _module(audio_loss=same_loss)

    assert module.audio_loss is same_loss
