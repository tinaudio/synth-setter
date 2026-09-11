"""Slot-collapse diagnostics for layerwise conditioning."""

from functools import partial

import pytest
import torch

from synth_setter.conditioning import EmbeddingConditioningSpec
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule


class _PassthroughField(torch.nn.Module):
    """Field stub that leaves conditioning untouched so the diagnostic reads the encoder."""

    def apply_dropout(
        self, conditioning: torch.Tensor, dropout_rate: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return conditioning unchanged with an all-keep mask.

        :param conditioning: Encoded conditioning.
        :param dropout_rate: Unused dropout probability.
        :returns: The unchanged conditioning and an all-keep mask.
        """
        return conditioning, torch.ones(conditioning.shape[0], dtype=torch.bool)


def _module_logging(*, on_cadence: bool) -> tuple[VSTFlowMatchingModule, dict[str, float]]:
    """Build a flow module whose encoder forwards the batch's conditioning verbatim.

    :param on_cadence: Whether the step reports as one of Lightning's logging steps.
    :returns: The module and the dict its ``log`` calls accumulate into.
    """
    module = VSTFlowMatchingModule(
        encoder=torch.nn.Identity(),
        vector_field=_PassthroughField(),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=2,
        conditioning=EmbeddingConditioningSpec(column="cached", input_shape=(4, 3)),
    )
    logged: dict[str, float] = {}

    def record(name: str, value: torch.Tensor, *, on_step: bool, on_epoch: bool) -> None:
        """Capture one logged scalar.

        :param name: Metric name.
        :param value: Logged scalar tensor.
        :param on_step: Lightning per-step flag, unused here.
        :param on_epoch: Lightning per-epoch flag, unused here.
        """
        logged[name] = float(value.item())

    module.log = record  # pyright: ignore[reportAttributeAccessIssue]
    module._is_trainer_logging_step = lambda: on_cadence  # pyright: ignore[reportAttributeAccessIssue]
    return module, logged


@pytest.mark.parametrize(
    ("slots", "expected"),
    [
        (torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]), 1.0),
        (torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]), -1.0 / 3.0),
    ],
    ids=["collapsed", "spread"],
)
def test_slot_cosine_diagnostic_reports_the_mean_off_diagonal_similarity(
    slots: torch.Tensor, expected: float
) -> None:
    """The metric averages every ordered slot pair, excluding each slot against itself.

    :param slots: Per-slot conditioning rows shared by both batch elements.
    :param expected: Mean pairwise cosine the metric should report.
    """
    module, logged = _module_logging(on_cadence=True)

    module._prepare_conditioning({"conditioning": slots.expand(2, -1, -1)})

    assert logged["train/slot_cosine"] == pytest.approx(expected)


def test_slot_cosine_diagnostic_with_pooled_conditioning_logs_nothing() -> None:
    """Rank-2 conditioning has no slot axis, so there is no pair to compare."""
    module, logged = _module_logging(on_cadence=True)

    module._prepare_conditioning({"conditioning": torch.randn(2, 4)})

    assert "train/slot_cosine" not in logged


def test_slot_cosine_diagnostic_off_the_logging_cadence_logs_nothing() -> None:
    """The diagnostic costs a gram matrix per step, so it rides Lightning's cadence."""
    module, logged = _module_logging(on_cadence=False)

    module._prepare_conditioning({"conditioning": torch.randn(2, 3, 4)})

    assert "train/slot_cosine" not in logged
