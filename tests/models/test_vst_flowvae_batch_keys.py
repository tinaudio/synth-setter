"""``VSTFlowVAEModule`` reads its conditioning from the ``mel`` model-batch key.

The module receives batches straight from ``prepare_batch``, so the key it indexes
must track the model-batch contract rather than the stored ``mel_spec`` column name.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from synth_setter.models.vst_flowvae_module import VSTFlowVAEModule

_BATCH = 2
_MEL_SHAPE = (1, 4, 5)
_NUM_PARAMS = 3


class _RecordingNet(torch.nn.Module):
    """Net that records the tensor it was conditioned on."""

    def __init__(self) -> None:
        """Start with no recorded conditioning."""
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.seen: torch.Tensor | None = None

    def forward(self, mel: torch.Tensor) -> SimpleNamespace:
        """Record the conditioning and return a minimal VAE output.

        :param mel: Conditioning tensor taken from the batch.
        :returns: Object exposing the ``x_hat`` reconstruction ``predict_step`` returns.
        """
        self.seen = mel
        return SimpleNamespace(x_hat=torch.zeros(len(mel), _NUM_PARAMS))


def _module() -> VSTFlowVAEModule:
    """Build a Flow-VAE module around the recording net.

    :returns: Module whose ``net`` captures the conditioning it receives.
    """
    # Cast: the module annotates these as instances, but Hydra supplies factories
    # (``_partial_: true``) and neither is invoked on the step paths under test.
    return VSTFlowVAEModule(
        net=_RecordingNet(),
        optimizer=cast(torch.optim.Optimizer, torch.optim.Adam),
        scheduler=cast(Any, None),
        param_spec="surge_simple",
    )


def _batch() -> dict[str, torch.Tensor]:
    """Build a model batch carrying conditioning under the ``mel`` key.

    :returns: Batch with ``params`` and ``mel``.
    """
    return {
        "params": torch.zeros(_BATCH, _NUM_PARAMS),
        "mel": torch.rand(_BATCH, *_MEL_SHAPE),
    }


def test_model_step_conditions_the_net_on_the_mel_batch_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``model_step`` passes the batch's ``mel`` tensor to the net unchanged.

    :param monkeypatch: Patcher used to stub the loss, whose ``nflows`` dependency
        is optional (#1664) and irrelevant to which batch key is read.
    """
    import synth_setter.models.components.vae as vae

    monkeypatch.setattr(
        vae, "compute_flowvae_loss", lambda *args, **kwargs: {"loss": torch.zeros(())}
    )
    module = _module()
    batch = _batch()

    _, mel, target_params, _ = module.model_step(batch)

    assert module.net.seen is batch["mel"]
    assert mel is batch["mel"]
    assert target_params is batch["params"]


def test_predict_step_conditions_the_net_on_the_mel_batch_entry() -> None:
    """``predict_step`` reads the same ``mel`` entry and echoes the batch back."""
    module = _module()
    batch = _batch()

    predictions, returned_batch = module.predict_step(batch, 0)

    assert module.net.seen is batch["mel"]
    assert predictions.shape == (_BATCH, _NUM_PARAMS)
    assert returned_batch is batch


def test_model_step_without_a_mel_entry_raises_key_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch still using the stored column name is a contract violation, not a silent default.

    :param monkeypatch: Patcher used to stub the optional-dependency loss.
    """
    import synth_setter.models.components.vae as vae

    monkeypatch.setattr(
        vae, "compute_flowvae_loss", lambda *args, **kwargs: {"loss": torch.zeros(())}
    )
    batch = _batch()
    batch["mel_spec"] = batch.pop("mel")

    with pytest.raises(KeyError, match="mel"):
        _module().model_step(batch)
