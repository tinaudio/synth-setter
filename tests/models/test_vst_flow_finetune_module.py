"""Behaviour tests for simulator-feedback finetuning of a pretrained flow.

Every arm renders real torchsynth audio through the production differentiable render and scores it
with the production spectral distance; nothing here stands in for the simulator.
"""

from pathlib import Path

import pytest
import torch

from synth_setter.data.vst.torchsynth_param_spec import TORCHSYNTH_FULL_PARAM_SPEC
from synth_setter.models.components.audio_distance import MultiScaleSpectralDistance
from synth_setter.models.components.audio_feedback import AudioFeedbackLoss
from synth_setter.models.components.vector_field import VectorField
from synth_setter.models.vst_flow_finetune_module import VSTFlowFinetuneModule
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule

_SAMPLE_RATE = 16_000
_SIGNAL_LENGTH = 8_192
_BATCH = 2
_WIDTH = TORCHSYNTH_FULL_PARAM_SPEC.encoded_width
_CONDITIONING_DIM = 8
_FLOW_PREFIX = "vector_field.flow."


class _WaveformEncoder(torch.nn.Module):
    """Trainable conditioning encoder over a raw waveform batch."""

    def __init__(self, out_dim: int = _CONDITIONING_DIM) -> None:
        """Build the projection from waveform to conditioning.

        :param out_dim: Width of the produced conditioning vector.
        """
        super().__init__()
        self.linear = torch.nn.Linear(_SIGNAL_LENGTH, out_dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Map a waveform batch to a flat vector.

        :param audio: Audio shaped ``(batch, _SIGNAL_LENGTH)``.
        :returns: Vector shaped ``(batch, out_dim)``.
        """
        return self.linear(audio)


class _NormalisingEncoder(torch.nn.Module):
    """Conditioning encoder carrying batch-norm running statistics, as the shipped ones do."""

    def __init__(self) -> None:
        """Build the normalised projection from waveform to conditioning."""
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(_SIGNAL_LENGTH)
        self.linear = torch.nn.Linear(_SIGNAL_LENGTH, _CONDITIONING_DIM)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Map a waveform batch to a flat vector through the normalisation.

        :param audio: Audio shaped ``(batch, _SIGNAL_LENGTH)``.
        :returns: Vector shaped ``(batch, _CONDITIONING_DIM)``.
        """
        return self.linear(self.norm(audio))


def _base_module(encoder: torch.nn.Module | None = None) -> VSTFlowMatchingModule:
    """Build the tiny pretrained flow a finetune starts from.

    :param encoder: Conditioning encoder to train; a plain waveform encoder when omitted.
    :returns: Configured base module conditioned on raw audio.
    """
    torch.manual_seed(0)
    return VSTFlowMatchingModule(
        encoder=encoder or _WaveformEncoder(),
        vector_field=VectorField(
            field_dim=_WIDTH,
            hidden_dim=32,
            conditioning_dim=_CONDITIONING_DIM,
            num_blocks=2,
        ),
        optimizer=torch.optim.Adam,  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=_WIDTH,
        conditioning="audio",
    )


def _base_checkpoint(tmp_path: Path, module: VSTFlowMatchingModule | None = None) -> Path:
    """Persist a base module's weights in Lightning's checkpoint layout.

    :param tmp_path: Directory the checkpoint is written into.
    :param module: Module to persist; a fresh base module when omitted.
    :returns: Path to the written checkpoint.
    """
    path = tmp_path / "base.ckpt"
    torch.save({"state_dict": (module or _base_module()).state_dict()}, path)
    return path


def _finetune(
    checkpoint: Path,
    control_mode: str = "gradient_spectral",
    overrides: dict[str, object] | None = None,
) -> VSTFlowFinetuneModule:
    """Build a finetune module over a freshly built base of the pinned shape.

    :param checkpoint: Path to the base checkpoint to start from.
    :param control_mode: Which control arm to configure.
    :param overrides: Constructor arguments replacing the test defaults.
    :returns: Configured finetune module.
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
        "control_mode": control_mode,
        "control_hidden_dim": 16,
        "control_t_min": 0.0,
        "sample_rate": _SAMPLE_RATE,
        "signal_length": _SIGNAL_LENGTH,
        "render_batch_size": _BATCH,
        "cost": MultiScaleSpectralDistance(sample_rate=_SAMPLE_RATE),
    }
    kwargs.update(overrides or {})
    return VSTFlowFinetuneModule(**kwargs)  # pyright: ignore[reportArgumentType]


def _batch(rows: int = _BATCH) -> dict[str, torch.Tensor]:
    """Build one training batch of params, noise, and audible target audio.

    :param rows: Number of rows in the batch.
    :returns: Batch keyed as the flow modules consume it.
    """
    from synth_setter.data.torchsynth_grad_render import (
        differentiable_decode,
        render_torchsynth_grad,
    )

    torch.manual_seed(1)
    params = torch.rand(rows, _WIDTH) * 2 - 1
    with torch.no_grad():
        audio = render_torchsynth_grad(
            differentiable_decode(params),
            sample_rate=_SAMPLE_RATE,
            signal_length=_SIGNAL_LENGTH,
            render_batch_size=rows,
        )
    return {"params": params, "noise": torch.randn(rows, _WIDTH), "audio": audio}


def test_finetune_module_from_base_checkpoint_restores_every_pretrained_weight(
    tmp_path: Path,
) -> None:
    """Every pretrained tensor survives the wrap, reachable under the control's flow prefix.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    base = _base_module()
    module = _finetune(_base_checkpoint(tmp_path, base))

    restored = module.state_dict()
    for name, value in base.state_dict().items():
        key = name.replace("vector_field.", _FLOW_PREFIX, 1) if "vector_field." in name else name
        assert torch.equal(restored[key], value), name


def test_finetune_module_with_mismatched_checkpoint_raises(tmp_path: Path) -> None:
    """A checkpoint carrying a key this model has no slot for is refused, not silently dropped.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    base = _base_module().state_dict()
    base["vector_field.not_a_real_parameter"] = torch.zeros(1)
    path = tmp_path / "stale.ckpt"
    torch.save({"state_dict": base}, path)

    with pytest.raises(ValueError, match="unexpected"):
        _finetune(path)


def test_finetune_module_with_partial_checkpoint_raises(tmp_path: Path) -> None:
    """A checkpoint missing a pretrained weight is refused rather than trained from noise.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    base = _base_module().state_dict()
    del base[next(name for name in base if name.startswith("vector_field."))]
    path = tmp_path / "partial.ckpt"
    torch.save({"state_dict": base}, path)

    with pytest.raises(ValueError, match="missing"):
        _finetune(path)


def test_finetune_module_at_initialisation_matches_the_pretrained_velocity(tmp_path: Path) -> None:
    """The zero-initialised control makes the first step an exact identity on the base field.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    module = _finetune(_base_checkpoint(tmp_path))
    batch = _batch()

    with torch.no_grad():
        t = torch.full((_BATCH, 1), 0.9)
        z = module.encoder(batch["audio"])
        pretrained = module.vector_field.flow(batch["params"], t, z)
        controlled = module.vector_field(
            batch["params"], t, z, control_input=torch.zeros(_BATCH, 1 + _WIDTH)
        )

    assert torch.equal(controlled, pretrained)


@pytest.mark.parametrize("control_mode", ["gradient_spectral", "learned_audio", "null"])
def test_finetune_train_step_moves_only_the_control(tmp_path: Path, control_mode: str) -> None:
    """Every arm trains its control while the pretrained flow and encoder stay bit-identical.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    :param control_mode: Control arm under test.
    """
    module = _finetune(
        _base_checkpoint(tmp_path),
        control_mode=control_mode,
        overrides={
            "control_encoder": _WaveformEncoder(out_dim=12)
            if control_mode == "learned_audio"
            else None
        },
    )
    frozen = {
        name: value.clone()
        for name, value in module.state_dict().items()
        if name.startswith(_FLOW_PREFIX) or name.startswith("encoder.")
    }
    # Named, so the learned arm's assertion can single out its own encoder rather than pass
    # on the control network the other arms also train.
    trainable = {n: p for n, p in module.named_parameters() if p.requires_grad}
    optimizer = torch.optim.Adam(trainable.values(), lr=1e-2)
    before = {name: p.clone() for name, p in trainable.items()}

    for _ in range(2):
        optimizer.zero_grad()
        module._train_step(_batch()).loss.backward()
        optimizer.step()

    after = module.state_dict()
    for name, value in frozen.items():
        assert torch.equal(after[name], value), name

    def _moved(prefix: str) -> bool:
        return any(
            not torch.equal(p, before[name])
            for name, p in trainable.items()
            if name.startswith(prefix)
        )

    assert _moved("vector_field.control.")
    if control_mode == "learned_audio":
        assert _moved("control_encoder.")


def test_finetune_training_leaves_frozen_normalisation_statistics_untouched(
    tmp_path: Path,
) -> None:
    """A pretrained encoder's running statistics do not drift while the control trains.

    Clearing ``requires_grad`` does not stop them; only eval mode does. Drift here changes
    the conditioning the frozen field sees, confounding the arms this module compares.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    encoder = _NormalisingEncoder()
    base = _base_module(encoder=_NormalisingEncoder())
    # Deliberately not calling train(): Lightning does not either before the first steps,
    # which is exactly when the drift this guards against occurred.
    module = _finetune(_base_checkpoint(tmp_path, base), overrides={"encoder": encoder})
    before = {
        name: buffer.clone()
        for name, buffer in module.named_buffers()
        if name.startswith("encoder.")
    }
    assert before, "the pretrained encoder must carry running statistics for this to test"

    for _ in range(2):
        module._train_step(_batch()).loss.backward()

    for name, buffer in module.named_buffers():
        if name.startswith("encoder."):
            assert torch.equal(buffer, before[name]), name


def test_finetune_validation_step_samples_through_the_controlled_flow(tmp_path: Path) -> None:
    """Guided sampling reaches the wrapped flow through both CFG branches.

    The unconditional branch passes no conditioning at all, which the wrapper must accept.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    module = _finetune(_base_checkpoint(tmp_path))
    batch = _batch()

    with torch.no_grad():
        sampled = module._sample(batch["audio"], torch.randn(_BATCH, _WIDTH), 2, 2.0)

    assert sampled.shape == (_BATCH, _WIDTH)
    assert torch.isfinite(sampled).all()


def test_finetune_render_decodes_model_space_before_rendering(tmp_path: Path) -> None:
    """The render decodes model space first; the clamped raw estimate is a different sound.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    from synth_setter.data.torchsynth_grad_render import (
        differentiable_decode,
        render_torchsynth_grad,
    )

    module = _finetune(_base_checkpoint(tmp_path))
    theta = torch.full((_BATCH, _WIDTH), -0.5)

    with torch.no_grad():
        rendered = module._render(theta)
        decoded = render_torchsynth_grad(
            differentiable_decode(theta),
            sample_rate=_SAMPLE_RATE,
            signal_length=_SIGNAL_LENGTH,
            render_batch_size=_BATCH,
        )
        # The renderer clamps, so skipping the decode still yields audio — just the audio of
        # different parameters. Pinning both halves is what catches a dropped decode.
        undecoded = render_torchsynth_grad(
            theta.clamp(0, 1),
            sample_rate=_SAMPLE_RATE,
            signal_length=_SIGNAL_LENGTH,
            render_batch_size=_BATCH,
        )

    assert torch.equal(rendered, decoded)
    assert not torch.equal(rendered, undecoded)


def test_finetune_train_step_loss_carries_no_audio_term(tmp_path: Path) -> None:
    """The cost reaches the run only as control input, so the objective stays pure flow matching.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    outputs = _finetune(_base_checkpoint(tmp_path))._train_step(_batch())

    assert outputs.audio_term is None


@pytest.mark.slow
def test_finetune_fit_from_a_trained_base_moves_only_the_control(tmp_path: Path) -> None:
    """A real fit over a real trained base moves the control and leaves that base untouched.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    from lightning import Trainer

    from synth_setter.data.torchsynth_datamodule import TorchSynthDataModule

    def _trainer() -> Trainer:
        return Trainer(
            max_epochs=1,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            limit_val_batches=0,
        )

    def _data() -> TorchSynthDataModule:
        datamodule = TorchSynthDataModule(
            sample_rate=_SAMPLE_RATE,
            signal_length=_SIGNAL_LENGTH,
            train_val_test_sizes=(_BATCH * 2, 2, 2),
            train_val_test_seeds=(1, 2, 3),
            batch_size=_BATCH,
            num_workers=0,
        )
        datamodule.setup("fit")
        return datamodule

    torch.manual_seed(0)
    base = _base_module()
    _trainer().fit(base, datamodule=_data())
    checkpoint = _base_checkpoint(tmp_path, base)

    module = _finetune(checkpoint)
    control_before = [p.clone() for p in module.vector_field.control.parameters()]
    _trainer().fit(module, datamodule=_data())

    trained = module.state_dict()
    for name, value in base.state_dict().items():
        assert torch.equal(trained[name.replace("vector_field.", _FLOW_PREFIX, 1)], value), name
    assert any(
        not torch.equal(p, q)
        for p, q in zip(module.vector_field.control.parameters(), control_before, strict=True)
    )


def test_finetune_module_with_audio_loss_raises(tmp_path: Path) -> None:
    """The audio-loss term and the control cannot both charge the run for the same cost.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    with pytest.raises(ValueError, match="audio_loss"):
        _finetune(
            _base_checkpoint(tmp_path),
            overrides={
                "audio_loss": AudioFeedbackLoss(
                    lambda_audio=0.03,
                    t_min=0.0,
                    sample_rate=_SAMPLE_RATE,
                    signal_length=_SIGNAL_LENGTH,
                    render_batch_size=_BATCH,
                    distance=MultiScaleSpectralDistance(sample_rate=_SAMPLE_RATE),
                )
            },
        )


def test_finetune_module_learned_arm_without_encoder_raises(tmp_path: Path) -> None:
    """The equation-10 arm refuses to start without the encoder that produces its signal.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    with pytest.raises(ValueError, match="control_encoder"):
        _finetune(_base_checkpoint(tmp_path), control_mode="learned_audio")


def test_finetune_module_gradient_arm_without_cost_raises(tmp_path: Path) -> None:
    """The equation-9 arm refuses to start without the differentiable cost it differentiates.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    with pytest.raises(ValueError, match="cost"):
        _finetune(_base_checkpoint(tmp_path), overrides={"cost": None})
