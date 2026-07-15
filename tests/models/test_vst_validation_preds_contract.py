"""Pins the ``preds`` key every VST module's ``validation_step`` must return.

``ValAudioProbe`` stages ``outputs["preds"]`` to render predicted audio, so the
key is a contract across the VST module family rather than an implementation
detail of any one of them. Each module is built for real at tiny sizes and its
``validation_step`` driven directly — no mocks — so a module that silently stops
returning its predictions fails here.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import pytest
import torch

from synth_setter.models.components.transformer import (
    ApproxEquivTransformer,
    LearntProjection,
)
from synth_setter.models.vst_fake_oracle_module import FakeOracleNet, VSTFakeOracleModule
from synth_setter.models.vst_ff_module import VSTFeedForwardModule
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule

_VstModule = VSTFeedForwardModule | VSTFakeOracleModule | VSTFlowMatchingModule

_NUM_PARAMS = 6
_BATCH = 3
_MEL_CHANNELS = 2
_MEL_N_MELS = 4
_MEL_N_FRAMES = 5


def _batch() -> dict[str, torch.Tensor]:
    """Return a batch carrying the keys every VST module's step functions read.

    :returns: Batch dict with ``params`` and ``mel_spec``.
    """
    return {
        "params": torch.rand(_BATCH, _NUM_PARAMS),
        "mel_spec": torch.rand(_BATCH, _MEL_CHANNELS, _MEL_N_MELS, _MEL_N_FRAMES),
    }


class _TinyNet(torch.nn.Module):
    """Flattening linear net mapping a mel spec to ``_NUM_PARAMS`` predictions."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(_MEL_CHANNELS * _MEL_N_MELS * _MEL_N_FRAMES, _NUM_PARAMS)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Map ``mel_spec`` to a ``(B, _NUM_PARAMS)`` prediction.

        :param mel_spec: Batch of mel spectrograms.
        :returns: Predicted parameter tensor.
        """
        return self.linear(mel_spec.flatten(start_dim=1))


def _feed_forward_module() -> VSTFeedForwardModule:
    """Build a tiny real feed-forward module.

    :returns: Module wired for the test batch shapes.
    """
    return VSTFeedForwardModule(
        net=_TinyNet(),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
    )


def _fake_oracle_module() -> VSTFakeOracleModule:
    """Build a tiny real fake-oracle module.

    :returns: Module wired for the test batch shapes.
    """
    return VSTFakeOracleModule(
        net=FakeOracleNet(d_out=_NUM_PARAMS),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
    )


def _flow_matching_module() -> VSTFlowMatchingModule:
    """Build a tiny real flow-matching module with a 2-step validation sampler.

    :returns: Module wired for the test batch shapes.
    """
    vector_field = ApproxEquivTransformer(
        projection=LearntProjection(
            d_model=16,
            d_token=16,
            num_params=_NUM_PARAMS,
            num_tokens=4,
            initial_ffn=True,
            final_ffn=False,
        ),
        num_layers=1,
        d_model=16,
        conditioning_dim=16,
        num_heads=2,
        d_ff=16,
        num_tokens=4,
        learn_projection=True,
        time_encoding="sinusoidal",
        zero_init=False,
    )
    return VSTFlowMatchingModule(
        encoder=_TinyEncoder(),
        vector_field=vector_field,
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=_NUM_PARAMS,
        validation_sample_steps=2,
        validation_cfg_strength=1.0,
    )


class _TinyEncoder(torch.nn.Module):
    """Conditioning encoder mapping a mel spec to a ``(B, 1, 16)`` conditioning token."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(_MEL_CHANNELS * _MEL_N_MELS * _MEL_N_FRAMES, 16)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Map ``mel_spec`` to a single conditioning token per sample.

        :param mel_spec: Batch of mel spectrograms.
        :returns: Conditioning tensor of shape ``(B, 1, 16)``.
        """
        return self.linear(mel_spec.flatten(start_dim=1)).unsqueeze(1)


@pytest.mark.parametrize(
    "build_module",
    [_feed_forward_module, _fake_oracle_module, _flow_matching_module],
    ids=["feed_forward", "fake_oracle", "flow_matching"],
)
def test_validation_step_returns_preds_shaped_like_target_params(
    build_module: Callable[[], _VstModule],
) -> None:
    """Every VST module returns its predictions under ``preds``, shaped like the targets.

    :param build_module: Factory for the module under test.
    """
    module = build_module()
    batch = _batch()

    outputs = module.validation_step(batch, batch_idx=0)

    assert "preds" in outputs
    assert outputs["preds"].shape == batch["params"].shape
    # Finiteness, not range: raw predictions are unbounded by design (linear/flow
    # outputs); decode_model_output owns the mapping into parameter space.
    assert torch.isfinite(outputs["preds"]).all()


def test_validation_step_preds_are_the_feed_forward_nets_predictions() -> None:
    """The feed-forward module's ``preds`` is its net's output, not a placeholder."""
    module = _feed_forward_module()
    batch = _batch()

    outputs = module.validation_step(batch, batch_idx=0)

    expected = module.net(batch["mel_spec"])
    assert torch.allclose(outputs["preds"], expected)


def test_validation_step_preds_are_the_oracle_targets() -> None:
    """The oracle predicts ``batch["params"]`` verbatim, so ``preds`` equals the targets."""
    module = _fake_oracle_module()
    batch = _batch()

    outputs = module.validation_step(batch, batch_idx=0)

    assert torch.equal(outputs["preds"], batch["params"])
