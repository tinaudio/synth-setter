"""Behaviour tests for simulator-feedback finetuning of a pretrained flow.

Every arm renders real torchsynth audio through the production differentiable render and scores it
with the production spectral distance; nothing here stands in for the simulator.
"""

import hashlib
from pathlib import Path

import numpy as np
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
_BUFFER_SECONDS = _SIGNAL_LENGTH / _SAMPLE_RATE
# Below this peak the render is silence, and scoring silence against silence leaves the
# control signal identically zero — every simulator assertion then holds for free.
_AUDIBLE_PEAK = 1e-4


def _audible_model_rows(rows: int, seed: int) -> torch.Tensor:
    """Draw model-space rows whose note sounds across the whole render buffer.

    The spec draws ``note_start_and_end`` over a multi-second range, so uniform-random rows
    start their note past the end of this short buffer and render silence. Note columns are
    mapped back through ``differentiable_decode``'s ``(theta + 1) / 2`` so the decoded row
    carries the sounding note.

    :param rows: Number of rows to draw.
    :param seed: Seed for the synth columns.
    :returns: Rows shaped ``(rows, encoded_width)`` in model space ``[-1, 1]``.
    """
    synth_values, _ = TORCHSYNTH_FULL_PARAM_SPEC.sample(np.random.default_rng(0))
    reference = TORCHSYNTH_FULL_PARAM_SPEC.encode(
        synth_values, {"pitch": 60, "note_start_and_end": (0.0, _BUFFER_SECONDS)}
    )
    note_tail = torch.from_numpy(reference)[TORCHSYNTH_FULL_PARAM_SPEC.synth_columns.stop :]
    synth = (
        torch.rand(
            rows,
            TORCHSYNTH_FULL_PARAM_SPEC.synth_param_length,
            generator=torch.Generator().manual_seed(seed),
        )
        * 2
        - 1
    )
    return torch.cat([synth, (note_tail * 2 - 1).expand(rows, -1)], dim=1)


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
        "cfg_dropout_rate": 0.0,
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
    params = _audible_model_rows(rows, seed=1)
    with torch.no_grad():
        audio = render_torchsynth_grad(
            differentiable_decode(params),
            sample_rate=_SAMPLE_RATE,
            signal_length=_SIGNAL_LENGTH,
            render_batch_size=rows,
        )
    assert audio.abs().max() > _AUDIBLE_PEAK, "batch is silent; simulator assertions are vacuous"
    return {"params": params, "noise": torch.randn(rows, _WIDTH), "audio": audio}


def _trainer():
    """Build a one-epoch CPU trainer with validation off.

    :returns: Configured Lightning trainer.
    """
    from lightning import Trainer

    return Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        limit_val_batches=0,
    )


def _data():
    """Build a tiny online torchsynth datamodule already set up for fitting.

    :returns: The datamodule.
    """
    from synth_setter.data.torchsynth_datamodule import TorchSynthDataModule

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


def test_finetune_checkpoint_identity_without_source_uses_canonical_local_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An omitted source identifies the materialized local file and exact bytes.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    :param monkeypatch: Changes the working directory to exercise a relative path.
    """
    checkpoint = _base_checkpoint(tmp_path)
    expected_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.chdir(tmp_path)

    module = _finetune(Path("base.ckpt"))

    assert module.hparams["model/base_checkpoint/resolved_source"] == checkpoint.as_uri()
    assert module.hparams["model/base_checkpoint/materialized_path"] == str(checkpoint)
    assert module.hparams["model/base_checkpoint/sha256"] == expected_sha256
    assert type(module.hparams["model/base_checkpoint/resolved_source"]) is str
    assert type(module.hparams["model/base_checkpoint/materialized_path"]) is str
    assert type(module.hparams["model/base_checkpoint/sha256"]) is str


def test_finetune_checkpoint_identity_preserves_remote_source(tmp_path: Path) -> None:
    """A remote source remains exactly as supplied while the loaded path stays local.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    checkpoint = _base_checkpoint(tmp_path)
    source = "r2:training-checkpoints/flow/base.ckpt"

    module = _finetune(checkpoint, overrides={"base_checkpoint_source": source})

    assert module.hparams["base_checkpoint"] == checkpoint
    assert module.hparams["model/base_checkpoint/resolved_source"] == source
    assert module.hparams["model/base_checkpoint/materialized_path"] == str(checkpoint)


def test_finetune_checkpoint_identity_redacts_remote_credentials(tmp_path: Path) -> None:
    """Run metadata retains the object path without publishing URI credentials.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    source = (
        "https://operator:password@example.com/checkpoints/base.ckpt"
        "?X-Amz-Signature=secret#fragment"
    )

    module = _finetune(_base_checkpoint(tmp_path), overrides={"base_checkpoint_source": source})

    safe_source = "https://example.com/checkpoints/base.ckpt"
    assert module.hparams["model/base_checkpoint/resolved_source"] == safe_source
    assert module.hparams["base_checkpoint_source"] == safe_source


def test_finetune_checkpoint_identity_reads_sanitized_launcher_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Launcher provenance reaches the model without persisting URI credentials.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    :param monkeypatch: Sets the launcher-to-model checkpoint source environment value.
    """
    monkeypatch.setenv(
        "SYNTH_SETTER_BASE_CHECKPOINT_SOURCE",
        "https://operator:password@example.com/base.ckpt?signature=secret",
    )

    module = _finetune(_base_checkpoint(tmp_path))

    assert (
        module.hparams["model/base_checkpoint/resolved_source"] == "https://example.com/base.ckpt"
    )
    assert module.hparams["base_checkpoint_source"] == "https://example.com/base.ckpt"


def test_finetune_checkpoint_identity_canonicalizes_explicit_local_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit local source becomes a strict absolute file URI.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    :param monkeypatch: Changes the working directory to exercise a relative source.
    """
    checkpoint = _base_checkpoint(tmp_path)
    monkeypatch.chdir(tmp_path)

    module = _finetune(checkpoint, overrides={"base_checkpoint_source": "base.ckpt"})

    assert module.hparams["model/base_checkpoint/resolved_source"] == checkpoint.as_uri()


def test_finetune_checkpoint_identity_redacts_local_file_uri_credentials(
    tmp_path: Path,
) -> None:
    """Credentialed local file URIs persist only the canonical local identity.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    checkpoint = _base_checkpoint(tmp_path)
    source = f"file://operator:password@localhost{checkpoint}?token=secret#fragment"

    module = _finetune(checkpoint, overrides={"base_checkpoint_source": source})

    assert module.hparams["model/base_checkpoint/resolved_source"] == checkpoint.as_uri()
    assert module.hparams["base_checkpoint_source"] == checkpoint.as_uri()


def test_finetune_checkpoint_identity_nonlocal_file_error_redacts_credentials(
    tmp_path: Path,
) -> None:
    """Invalid file-host errors do not echo URI credentials or query secrets.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    source = "file://operator:password@remote.example/base.ckpt?token=secret"

    with pytest.raises(ValueError, match="got host 'remote.example'") as exc_info:
        _finetune(_base_checkpoint(tmp_path), overrides={"base_checkpoint_source": source})

    assert "password" not in str(exc_info.value)
    assert "secret" not in str(exc_info.value)


def test_finetune_checkpoint_identity_rejects_blank_explicit_source(tmp_path: Path) -> None:
    """Whitespace cannot stand in for a retrievable checkpoint identity.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    with pytest.raises(ValueError, match="base_checkpoint_source cannot be blank"):
        _finetune(_base_checkpoint(tmp_path), overrides={"base_checkpoint_source": "   "})


def test_finetune_resume_matching_base_checkpoint_digest_is_accepted(tmp_path: Path) -> None:
    """Resume accepts a finetune checkpoint tied to the currently loaded base bytes.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    module = _finetune(_base_checkpoint(tmp_path))
    digest = module.hparams["model/base_checkpoint/sha256"]

    module.on_load_checkpoint({"hyper_parameters": {"model/base_checkpoint/sha256": digest}})


def test_finetune_resume_changed_base_checkpoint_digest_raises(tmp_path: Path) -> None:
    """Resume fails before combining control weights with a different frozen base.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    module = _finetune(_base_checkpoint(tmp_path))

    with pytest.raises(ValueError, match="base checkpoint SHA-256 mismatch"):
        module.on_load_checkpoint({"hyper_parameters": {"model/base_checkpoint/sha256": "0" * 64}})


def test_finetune_resume_without_base_checkpoint_digest_raises(tmp_path: Path) -> None:
    """Legacy resume cannot silently bypass frozen-base identity validation.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    module = _finetune(_base_checkpoint(tmp_path))

    with pytest.raises(ValueError, match="has no base checkpoint SHA-256"):
        module.on_load_checkpoint({"hyper_parameters": {}})


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

    def _unmoved(prefix: str) -> list[str]:
        """Report trainables under a prefix that no step reached.

        :param prefix: State-dict prefix to check.
        :returns: Names that are still bit-identical to their pre-training value.
        """
        return [
            name
            for name, p in trainable.items()
            if name.startswith(prefix) and torch.equal(p, before[name])
        ]

    # Every intended trainable, not any: the zero-initialised output layer means a single
    # moving bias would otherwise satisfy this while the rest of the network sits dead.
    assert not _unmoved("vector_field.control.")
    if control_mode == "learned_audio":
        assert not _unmoved("control_encoder.")


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
        module.on_validation_batch_start(batch, 0)
        sampled = module._sample(batch["audio"], torch.randn(_BATCH, _WIDTH), 2, 2.0)
        module.on_validation_batch_end(None, batch, 0)

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


def _logged_control_metrics(
    module: VSTFlowFinetuneModule, batch: dict[str, torch.Tensor]
) -> dict[str, tuple[float, dict[str, object]]]:
    """Run an attached training step and capture its control telemetry.

    :param module: Finetune module to attach to a trainer.
    :param batch: Training batch to process.
    :returns: Metric names mapped to scalar values and logging options.
    """
    module.trainer = _trainer()  # pyright: ignore[reportAttributeAccessIssue]
    logged: dict[str, tuple[float, dict[str, object]]] = {}

    def capture(
        values: dict[str, object],
        *,
        on_step: bool,
        on_epoch: bool,
        sync_dist: bool,
        batch_size: int,
    ) -> None:
        """Capture one Lightning metric group.

        :param values: Metric keys and scalar values.
        :param on_step: Whether Lightning emits the per-step values.
        :param on_epoch: Whether Lightning aggregates metrics over the epoch.
        :param sync_dist: Whether Lightning synchronizes values across ranks.
        :param batch_size: Number of rows represented by each scalar.
        """
        options = {
            "on_step": on_step,
            "on_epoch": on_epoch,
            "sync_dist": sync_dist,
            "batch_size": batch_size,
        }
        logged.update(
            (name, (float(torch.as_tensor(value).detach()), options))
            for name, value in values.items()
        )

    module.log_dict = capture  # pyright: ignore[reportAttributeAccessIssue]
    module._train_step(batch)
    return logged


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


@pytest.mark.parametrize("control_mode", ["gradient_spectral", "learned_audio"])
def test_finetune_feedback_arms_depend_on_the_simulator_signal(
    tmp_path: Path, control_mode: str
) -> None:
    """A control reading only ``(t, velocity)`` would pass every movement test; this fails it.

    Parameter movement proves the control trains, not that it *uses* the simulator. After training,
    replacing the signal with zeros must change the correction — otherwise the arm is
    indistinguishable from the null ablation it is meant to be measured against.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    :param control_mode: Feedback arm under test.
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
    optimizer = torch.optim.Adam([p for p in module.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(3):
        optimizer.zero_grad()
        module._train_step(_batch()).loss.backward()
        optimizer.step()

    batch = _batch()
    t = torch.full((_BATCH, 1), 0.9)
    with torch.no_grad():
        z = module.encoder(batch["audio"])
        velocity = module.vector_field.flow(batch["params"], t, z)
        active = torch.ones(_BATCH, dtype=torch.bool)
        signal = module._control_signal(
            module._one_step_estimate(batch["params"], t, velocity), batch["audio"], active
        )
        with_feedback = module.vector_field.combine(velocity, t, signal)
        without = module.vector_field.combine(velocity, t, torch.zeros_like(signal))

    assert not torch.equal(with_feedback, without)


def test_finetune_overfits_a_single_fixed_batch(tmp_path: Path) -> None:
    """The control must be able to drive the flow-matching loss down on one fixed batch.

    A broken objective or an under-capacity control that never reduces loss would still pass a test
    that only asserts parameters moved.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    torch.manual_seed(0)
    module = _finetune(_base_checkpoint(tmp_path), overrides={"control_hidden_dim": 64})
    batch = _batch()
    optimizer = torch.optim.Adam([p for p in module.parameters() if p.requires_grad], lr=1e-2)

    initial = module._train_step(batch).loss.item()
    loss = initial
    for _ in range(60):
        optimizer.zero_grad()
        step = module._train_step(batch)
        step.loss.backward()
        optimizer.step()
        loss = step.loss.item()

    # Relative, not a bare absolute bound: the achievable floor depends on the frozen field's
    # own error, which differs per machine.
    assert loss < initial * 0.5


def test_finetune_train_step_skips_rendering_unusable_rows(tmp_path: Path) -> None:
    """Only rows the correction can reach are scored; the rest come back zeroed.

    At the shipped ``control_t_min`` most rows are disengaged, and the render dominates the
    step, so scoring them would spend the budget on a signal ``combine`` discards.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    module = _finetune(_base_checkpoint(tmp_path))
    batch = _batch()
    # A distinct estimate: scoring a row against its own render gives a zero cost and a zero
    # gradient, which would make the "scored rows are non-zero" half of this test vacuous.
    theta = _audible_model_rows(_BATCH, seed=7)
    rendered: list[int] = []
    original = module._render

    def _counting_render(estimate: torch.Tensor) -> torch.Tensor:
        rendered.append(len(estimate))
        return original(estimate)

    module._render = _counting_render  # pyright: ignore[reportAttributeAccessIssue]
    active = torch.tensor([True, False])
    signal = module._control_signal(theta, batch["audio"], active)

    assert rendered == [1]
    assert torch.count_nonzero(signal[1]) == 0
    assert torch.count_nonzero(signal[0]) > 0


def test_finetune_train_step_does_not_score_cfg_dropped_rows(tmp_path: Path) -> None:
    """A fully unconditional row estimates the marginal, so its residual is noise.

    Its render/target residual is unrelated to that row's own audio, and the amount of noise
    differs per arm — which would confound the very comparison this module exists to run.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    module = _finetune(_base_checkpoint(tmp_path), overrides={"cfg_dropout_rate": 1.0})
    rendered: list[int] = []
    original = module._render

    def _counting_render(estimate: torch.Tensor) -> torch.Tensor:
        rendered.append(len(estimate))
        return original(estimate)

    module._render = _counting_render  # pyright: ignore[reportAttributeAccessIssue]
    outputs = module._train_step(_batch())

    assert not outputs.conditioning_keep.identity_keep.any()
    assert rendered == []


def test_controlled_sampling_differs_from_the_frozen_base(tmp_path: Path) -> None:
    """A trained control must change the sample; otherwise the finetune is unmeasurable.

    Until the control reaches sampling, every arm reports the base model's metrics and a working
    finetune is indistinguishable from a broken one.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    # The base module carries the same encoder and field weights, so it is exactly what
    # this arm would report if the control never reached sampling.
    base = _base_module()
    module = _finetune(_base_checkpoint(tmp_path, base))
    batch = _batch()
    optimizer = torch.optim.Adam([p for p in module.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(3):
        optimizer.zero_grad()
        module._train_step(batch).loss.backward()
        optimizer.step()

    noise = torch.randn(_BATCH, _WIDTH)
    with torch.no_grad():
        module.on_validation_batch_start(batch, 0)
        controlled = module._sample(batch["audio"], noise, 2, 2.0)
        module.on_validation_batch_end(None, batch, 0)
        uncontrolled = base._sample(batch["audio"], noise, 2, 2.0)

    assert torch.isfinite(controlled).all()
    assert not torch.equal(controlled, uncontrolled)


def test_controlled_sampling_engages_only_above_t_min(tmp_path: Path) -> None:
    """Raising the threshold must shrink the set of evaluations that score.

    RK4's final evaluation always lands on t=1, so some evaluation engages for any valid
    threshold; what the gate controls is how many. The bitwise-passthrough property for a
    disengaged row is pinned on ``ControlledFlow.combine`` itself.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    batch = _batch()
    checkpoint = _base_checkpoint(tmp_path)

    def _render_count(t_min: float) -> int:
        """Count renderer calls over one fixed sampling run.

        :param t_min: Flow time above which the control engages.
        :returns: Number of renderer invocations.
        """
        module = _finetune(checkpoint, overrides={"control_t_min": t_min})
        calls: list[int] = []
        original = module._render
        module._render = lambda estimate: (  # pyright: ignore[reportAttributeAccessIssue]
            calls.append(len(estimate)),
            original(estimate),
        )[1]
        with torch.no_grad():
            module.on_validation_batch_start(batch, 0)
            module._sample(batch["audio"], torch.randn(_BATCH, _WIDTH), 4, 2.0)
            module.on_validation_batch_end(None, batch, 0)
        return len(calls)

    # Only the final evaluation reaches t=1.
    assert _render_count(0.99) == 1
    assert _render_count(0.0) == 16


def test_controlled_sampling_renders_only_engaged_evaluations(tmp_path: Path) -> None:
    """Renders are spent only where the correction can use them.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    module = _finetune(_base_checkpoint(tmp_path), overrides={"control_t_min": 0.5})
    batch = _batch()
    rendered: list[int] = []
    original = module._render

    def _counting_render(estimate: torch.Tensor) -> torch.Tensor:
        rendered.append(len(estimate))
        return original(estimate)

    module._render = _counting_render  # pyright: ignore[reportAttributeAccessIssue]
    with torch.no_grad():
        module.on_validation_batch_start(batch, 0)
        module._sample(batch["audio"], torch.randn(_BATCH, _WIDTH), 4, 2.0)
        module.on_validation_batch_end(None, batch, 0)

    # Only engaged evaluations may render.
    assert 0 < len(rendered) < 16


def test_sampling_without_a_bound_observation_raises(tmp_path: Path) -> None:
    """Sampling outside a validation batch must fail loudly, not silently use the base.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    module = _finetune(_base_checkpoint(tmp_path))

    with pytest.raises(RuntimeError, match="no observation bound"):
        module._sample(_batch()["audio"], torch.randn(_BATCH, _WIDTH), 2, 2.0)


def test_validation_batch_end_releases_the_bound_observation(tmp_path: Path) -> None:
    """A stale target must not leak into the next batch's sampling.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    module = _finetune(_base_checkpoint(tmp_path))
    batch = _batch()

    module.on_validation_batch_start(batch, 0)
    bound = module._sampling_target
    module.on_validation_batch_end(None, batch, 0)

    assert bound is batch["audio"]
    assert module._sampling_target is None


@pytest.mark.slow
@pytest.mark.parametrize("stage", ["validate", "test"])
def test_finetune_sampling_runs_through_the_lightning_loop(tmp_path: Path, stage: str) -> None:
    """Every evaluation entrypoint must sample without raising.

    Bracketing the hooks by hand in a test hides two whole classes of failure: a hook that
    Lightning never calls, and the standalone loops' ``inference_mode``, which
    ``torch.enable_grad`` cannot reopen. Mid-fit validation is the one path built with
    ``inference_mode=False``, so testing only that is testing the case that works.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    :param stage: Lightning entrypoint under test.
    """
    from lightning import Trainer

    module = _finetune(
        _base_checkpoint(tmp_path),
        overrides={"validation_sample_steps": 2, "test_sample_steps": 2},
    )
    trainer = Trainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        limit_val_batches=1,
        limit_test_batches=1,
    )

    getattr(trainer, stage)(module, datamodule=_data())

    assert module._sampling_target is None


def test_controlled_sampling_depends_on_the_bound_observation(tmp_path: Path) -> None:
    """The sample must change when the observation does, holding everything else fixed.

    A trained control can differ from the base using only its velocity and time features, so
    "differs from base" does not prove the simulator signal is read at all. Swapping the target
    between two otherwise identical runs isolates exactly that.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    module = _finetune(_base_checkpoint(tmp_path))
    batch = _batch()
    optimizer = torch.optim.Adam([p for p in module.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(3):
        optimizer.zero_grad()
        module._train_step(batch).loss.backward()
        optimizer.step()

    noise = torch.randn(_BATCH, _WIDTH)
    with torch.no_grad():
        module.on_validation_batch_start(batch, 0)
        as_bound = module._sample(batch["audio"], noise, 2, 2.0)
        module.on_validation_batch_end(None, batch, 0)

        swapped = dict(batch, audio=batch["audio"].flip(0))
        module.on_validation_batch_start(swapped, 0)
        as_swapped = module._sample(batch["audio"], noise, 2, 2.0)
        module.on_validation_batch_end(None, swapped, 0)

    assert not torch.equal(as_bound, as_swapped)


def test_finetune_predict_step_samples_with_a_bound_observation(tmp_path: Path) -> None:
    """The predict lane must bind its own observation rather than raising.

    ``cli/predict_capture.py`` calls ``predict_step`` directly with no Lightning hooks, and
    ``synth-setter-eval mode=predict`` goes through ``on_predict_batch_start`` — both reach
    the same guard, so both need the binding.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    module = _finetune(_base_checkpoint(tmp_path), overrides={"test_sample_steps": 2})
    batch = _batch()

    module.on_predict_batch_start(batch, 0)
    with torch.no_grad():
        predicted = module._sample(batch["audio"], torch.randn(_BATCH, _WIDTH), 2, 2.0)
    module.on_predict_batch_end(None, batch, 0)

    assert predicted.shape == (_BATCH, _WIDTH)
    assert torch.isfinite(predicted).all()
    assert module._sampling_target is None


def test_finetune_gradient_control_logs_positive_whole_batch_telemetry(tmp_path: Path) -> None:
    """Active gradient feedback reports signal, cost, and gradient magnitudes.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    metrics = _logged_control_metrics(_finetune(_base_checkpoint(tmp_path)), _batch())
    options = {
        "on_step": True,
        "on_epoch": False,
        "sync_dist": True,
        "batch_size": _BATCH,
    }

    assert set(metrics) == {
        "train/control_active_fraction",
        "train/control_cost",
        "train/control_grad_norm",
        "train/control_signal_norm",
    }
    assert metrics["train/control_active_fraction"] == (1.0, options)
    assert metrics["train/control_signal_norm"][0] > 0.0
    assert metrics["train/control_cost"][0] > 0.0
    assert metrics["train/control_grad_norm"][0] > 0.0
    assert metrics["train/control_signal_norm"][1] == options
    assert metrics["train/control_cost"][1] == options
    assert metrics["train/control_grad_norm"][1] == options


def test_finetune_null_control_logs_active_gate_and_zero_signal(tmp_path: Path) -> None:
    """The null arm distinguishes an active gate from its intentionally zero signal.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    metrics = _logged_control_metrics(
        _finetune(_base_checkpoint(tmp_path), control_mode="null"), _batch()
    )

    assert set(metrics) == {
        "train/control_active_fraction",
        "train/control_signal_norm",
    }
    assert metrics["train/control_active_fraction"][0] == 1.0
    assert metrics["train/control_signal_norm"][0] == 0.0


def test_finetune_learned_control_logs_common_metric_key_set(tmp_path: Path) -> None:
    """The learned arm emits only the telemetry shared by every control mode.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    module = _finetune(
        _base_checkpoint(tmp_path),
        control_mode="learned_audio",
        overrides={"control_encoder": _WaveformEncoder(out_dim=12)},
    )

    metrics = _logged_control_metrics(module, _batch())

    assert set(metrics) == {
        "train/control_active_fraction",
        "train/control_signal_norm",
    }
    assert metrics["train/control_active_fraction"][0] == 1.0
    assert metrics["train/control_signal_norm"][0] > 0.0


def test_finetune_inactive_batch_logs_finite_zero_control_telemetry(tmp_path: Path) -> None:
    """An entirely inactive gradient batch emits every metric as a finite zero.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    module = _finetune(_base_checkpoint(tmp_path), overrides={"cfg_dropout_rate": 1.0})

    metrics = _logged_control_metrics(module, _batch())

    assert set(metrics) == {
        "train/control_active_fraction",
        "train/control_cost",
        "train/control_grad_norm",
        "train/control_signal_norm",
    }
    assert metrics["train/control_active_fraction"][0] == 0.0
    assert metrics["train/control_signal_norm"][0] == 0.0
    assert metrics["train/control_cost"][0] == 0.0
    assert metrics["train/control_grad_norm"][0] == 0.0


def test_finetune_mixed_control_mask_logs_whole_batch_means(tmp_path: Path) -> None:
    """Inactive zeros remain in each telemetry denominator for mixed batches.

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    module = _finetune(_base_checkpoint(tmp_path))
    module.trainer = _trainer()  # pyright: ignore[reportAttributeAccessIssue]
    metrics: dict[str, torch.Tensor] = {}
    logging_options: dict[str, object] = {}

    def capture(
        values: dict[str, torch.Tensor],
        *,
        on_step: bool,
        on_epoch: bool,
        sync_dist: bool,
        batch_size: int,
    ) -> None:
        """Retain telemetry emitted at the Lightning logging boundary.

        :param values: Metric mapping emitted by the module.
        :param on_step: Whether Lightning emits per-step values.
        :param on_epoch: Whether Lightning aggregates over the epoch.
        :param sync_dist: Whether Lightning synchronizes distributed values.
        :param batch_size: Rows represented by each metric.
        """
        metrics.update(values)
        logging_options.update(
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=sync_dist,
            batch_size=batch_size,
        )

    module.log_dict = capture  # pyright: ignore[reportAttributeAccessIssue]
    module._log_control_telemetry(
        torch.tensor([[3.0, 4.0], [0.0, 0.0]]), torch.tensor([True, False])
    )

    assert float(metrics["train/control_active_fraction"]) == 0.5
    assert float(metrics["train/control_signal_norm"]) == 2.5
    assert float(metrics["train/control_cost"]) == 1.5
    assert float(metrics["train/control_grad_norm"]) == 2.0
    assert logging_options == {
        "on_step": True,
        "on_epoch": False,
        "sync_dist": True,
        "batch_size": 2,
    }


def test_controlled_sampling_scores_the_conditional_velocity(tmp_path: Path) -> None:
    """The control sees the velocity it was trained on, not the CFG-extrapolated one.

    Training scores ``flow(x_t, t, z)``; the sampler integrates
    ``(1 - w) * v_uncond + w * v_cond``, which at the shipped ``w`` is a different vector.
    Feeding that to the control would put it out of distribution, so evaluation metrics
    could move for reasons unrelated to the learned correction (#2782).

    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    module = _finetune(_base_checkpoint(tmp_path), overrides={"control_t_min": 0.0})
    batch = _batch()
    noise = torch.randn(_BATCH, _WIDTH)
    seen: list[torch.Tensor] = []
    original = module.vector_field.control.forward

    def _spy(t: torch.Tensor, velocity: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """Record the velocity feature the control is handed.

        :param t: Flow time.
        :param velocity: Velocity feature under test.
        :param control: Control signal.
        :returns: The unmodified correction.
        """
        seen.append(velocity.clone())
        return original(t, velocity, control)

    module.vector_field.control.forward = _spy  # pyright: ignore[reportAttributeAccessIssue]
    with torch.no_grad():
        module.on_validation_batch_start(batch, 0)
        module._sample(batch["audio"], noise, 1, 2.0)
        module.on_validation_batch_end(None, batch, 0)

        # The sampler's first RK4 evaluation is at (noise, t=0), so both candidate
        # velocities can be reconstructed exactly.
        z = module.encoder(batch["audio"])
        t0 = torch.zeros(_BATCH, 1)
        conditional = module.vector_field.flow(noise, t0, z)
        unconditional = module.vector_field.flow(noise, t0, None)
    guided = (1 - 2.0) * unconditional + 2.0 * conditional

    assert seen, "the control was never evaluated"
    assert not torch.allclose(conditional, guided), "fixture cannot distinguish the two"
    assert torch.allclose(seen[0], conditional, atol=1e-6)
    assert not torch.allclose(seen[0], guided, atol=1e-6)
