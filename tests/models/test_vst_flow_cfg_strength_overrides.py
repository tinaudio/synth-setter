"""Hydra composes a bare ``0``/``1``/``2`` CFG strength as ``int``; sampling requires ``float``."""

from __future__ import annotations

import pytest
import torch

from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule

_BATCH = 3
_WIDTH = 6
_SIGNAL_LENGTH = 16
_CONDITIONING_DIM = 4


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


class _ZeroField(torch.nn.Module):
    """Network whose velocity is zero regardless of state, time, or conditioning."""

    def __init__(self) -> None:
        super().__init__()
        self.row = torch.nn.Parameter(torch.zeros(_WIDTH), requires_grad=False)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, conditioning: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Broadcast the zero row over the batch.

        :param x: Parameter state shaped ``(batch, _WIDTH)``.
        :param t: Flow time shaped ``(batch, 1)``.
        :param conditioning: Ignored.
        :returns: Zeros shaped ``(batch, _WIDTH)``.
        """
        return self.row.expand(x.shape[0], -1)

    def apply_dropout(
        self, z: torch.Tensor, rate: float = 0.1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep every conditioning row; this field ignores conditioning anyway.

        :param z: Conditioning rows.
        :param rate: Ignored.
        :returns: ``z`` unchanged and an all-True keep mask.
        """
        return z, torch.ones(z.shape[0], dtype=torch.bool, device=z.device)


def _module(**overrides: object) -> VSTFlowMatchingModule:
    r"""Build a CPU flow module with the supplied hyperparameter overrides.

    :param \*\*overrides: Hyperparameters forwarded to the module constructor.
    :returns: Constructed module.
    """
    return VSTFlowMatchingModule(
        encoder=_WaveformEncoder(),
        vector_field=_ZeroField(),
        optimizer=torch.optim.Adam,  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=_WIDTH,
        conditioning="audio",
        cfg_dropout_rate=0.0,
        compile=False,
        **overrides,  # pyright: ignore[reportArgumentType]
    )


def _batch() -> dict[str, torch.Tensor]:
    """Build one validation batch keyed as the flow module consumes it.

    :returns: Batch with audio conditioning and clean parameters.
    """
    torch.manual_seed(0)
    return {
        "audio": torch.randn(_BATCH, _SIGNAL_LENGTH),
        "params": torch.randn(_BATCH, _WIDTH),
    }


@pytest.mark.parametrize("strength", [0, 1, 2])
def test_validation_step_with_integer_cfg_strength_overrides_samples(strength: int) -> None:
    """A CLI-composed integer guidance scale reaches the sampling boundary as a float.

    :param strength: Guidance scale as Hydra composes a bare CLI integer.
    """
    module = _module(
        validation_cfg_strength=strength,
        validation_sketch_cfg_strength=strength,
        validation_sample_steps=2,
    )

    module.validation_step(_batch(), 0)


def test_test_step_with_integer_cfg_strength_override_samples() -> None:
    """The test-stage guidance scale accepts a CLI-composed integer too."""
    module = _module(test_cfg_strength=2, test_sketch_cfg_strength=2, test_sample_steps=2)

    module.test_step(_batch(), 0)


@pytest.mark.parametrize(
    "name",
    [
        "validation_cfg_strength",
        "validation_sketch_cfg_strength",
        "test_cfg_strength",
        "test_sketch_cfg_strength",
    ],
)
def test_boolean_cfg_strength_override_is_rejected(name: str) -> None:
    """``true`` must not silently become a guidance scale of 1.0.

    :param name: Guidance-scale hyperparameter under test.
    """
    with pytest.raises(TypeError, match=name):
        _module(**{name: True})
