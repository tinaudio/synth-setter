"""Exercise load compatibility for checkpoints pickling a fused log-mel encoder.

``VSTFlowMatchingModule`` pickles the encoder *instance* into
``hyper_parameters``, so the class named there must resolve at unpickle time.
The fixture writes a real Lightning checkpoint naming the fused class and
carrying its flat attribute layout — the union of what ``LogMelFrontend`` and
``MelCNN`` hold — so the tests drive the real load paths.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest
import torch
from lightning import Trainer
from torch import nn

from synth_setter.models.components import cnn
from synth_setter.models.components.cnn import MelCNN
from synth_setter.models.components.spec_encoder import LogMelFrontend
from synth_setter.models.components.transformer import (
    ApproxEquivTransformer,
    LearntProjection,
)
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule

_PRED_WIDTH = 300
_SAMPLE_RATE = 16_000
_IN_DIM = 16_384
_OUT_DIM = 32


class _PreSplitLogMelEncoder(nn.Module):
    """Provide the flat fused-encoder layout the compatibility fixture pickles.

    Submodules come from the real front end and backbone, so the pickled state is the shipped one
    rather than a hand-built guess.
    """

    def __init__(self, frontend: LogMelFrontend, backbone: MelCNN) -> None:
        """Flatten a front end and a backbone into one fused module.

        :param frontend: Source of the mel transform and decibel scaling.
        :param backbone: Source of the convolutional stack and projection.
        """
        super().__init__()
        self.in_dim = frontend.in_dim
        self.mel = frontend.mel
        self.amin = frontend.amin
        self.db_multiplier = frontend.db_multiplier
        self.top_db = frontend.top_db
        self.conv_net = backbone.conv_net
        self.pool = backbone.pool
        self.projection = backbone.projection


# Pickle records a class by module and qualname, so the saved bytes must name
# the fused class rather than this stand-in.
_PreSplitLogMelEncoder.__module__ = "synth_setter.models.components.cnn"
_PreSplitLogMelEncoder.__qualname__ = "LogMelEncoder"


@pytest.fixture
def legacy_parts() -> tuple[LogMelFrontend, MelCNN]:
    """Build the front end and backbone the fused encoder holds.

    Batch-norm running statistics are moved off their defaults so an eval-mode comparison
    distinguishes restored buffers from freshly initialized ones.

    :returns: Seeded front end and backbone sharing one waveform contract.
    """
    torch.manual_seed(0)
    frontend = LogMelFrontend(_IN_DIM, sample_rate=_SAMPLE_RATE, n_mels=16)
    backbone = MelCNN(4, _OUT_DIM, input_channels=1, num_blocks=2)
    for module in backbone.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.running_mean = torch.randn(module.num_features)
            module.running_var = torch.rand(module.num_features) + 0.5
    return frontend, backbone


@pytest.fixture
def legacy_checkpoint(
    legacy_parts: tuple[LogMelFrontend, MelCNN],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Write a real checkpoint whose encoder is pickled as ``LogMelEncoder``.

    :param legacy_parts: Front end and backbone fused into the pickled encoder.
    :param tmp_path: Per-test directory receiving the checkpoint.
    :param monkeypatch: Makes the fused-class name resolvable while pickling.
    :returns: Path to the saved Lightning checkpoint.
    """
    frontend, backbone = legacy_parts
    encoder = _PreSplitLogMelEncoder(frontend, backbone)
    vector_field = ApproxEquivTransformer(
        projection=LearntProjection(
            d_model=_OUT_DIM,
            d_token=_OUT_DIM,
            num_params=_PRED_WIDTH,
            num_tokens=8,
            initial_ffn=True,
            final_ffn=False,
        ),
        num_layers=1,
        d_model=_OUT_DIM,
        conditioning_dim=_OUT_DIM,
        num_heads=2,
        d_ff=_OUT_DIM,
        num_tokens=8,
        learn_projection=True,
        time_encoding="sinusoidal",
        zero_init=False,
    )
    model = VSTFlowMatchingModule(
        encoder=encoder,
        vector_field=vector_field,
        # The modules take Hydra _partial_ optimizer factories despite the annotation.
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=_PRED_WIDTH,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.strategy.connect(model)
    path = tmp_path / "pre_split_encoder.ckpt"
    monkeypatch.setattr(cnn, "LogMelEncoder", _PreSplitLogMelEncoder, raising=False)
    trainer.save_checkpoint(path)
    monkeypatch.undo()
    return path


def test_load_from_checkpoint_pre_split_encoder_reproduces_its_encoding(
    legacy_checkpoint: Path, legacy_parts: tuple[LogMelFrontend, MelCNN]
) -> None:
    """A checkpoint naming ``LogMelEncoder`` loads and produces the same encoding.

    Runs in eval mode so batch norm reads the checkpoint's running statistics
    rather than the batch's: restoring the weights but not the buffers would
    pass in train mode and still encode wrongly at inference.

    :param legacy_checkpoint: Checkpoint pickling the encoder as ``LogMelEncoder``.
    :param legacy_parts: The front end and backbone whose weights it carries.
    """
    frontend, backbone = legacy_parts
    waveform = torch.zeros(2, _IN_DIM)
    waveform[:, ::64] = 0.5
    expected = backbone.eval()(frontend.eval()(waveform))

    model = VSTFlowMatchingModule.load_from_checkpoint(
        legacy_checkpoint, map_location="cpu", weights_only=False
    )

    torch.testing.assert_close(model.encoder.eval()(waveform), expected)


def test_torch_load_pre_split_checkpoint_resolves_the_renamed_encoder(
    legacy_checkpoint: Path,
) -> None:
    """The raw ``torch.load`` path resolves the checkpoint's ``LogMelEncoder``.

    :param legacy_checkpoint: Checkpoint pickling the encoder as ``LogMelEncoder``.
    """
    checkpoint = torch.load(legacy_checkpoint, map_location="cpu", weights_only=False)

    assert isinstance(checkpoint["hyper_parameters"]["encoder"], nn.Module)


def test_unknown_cnn_module_attribute_still_raises_attribute_error() -> None:
    """Compatibility resolution stays scoped to ``LogMelEncoder``."""
    with pytest.raises(AttributeError):
        cnn.NotAnEncoder  # noqa: B018
