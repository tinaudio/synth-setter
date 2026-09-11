"""Research controls preserve common weights and exercise shipped model constructors."""

from pathlib import Path

import hydra
import pytest
import torch

from scripts.dev.param_language_ablation import build_config, summarize_parameters
from synth_setter.data.vst.param_spec_registry import param_specs


def test_parameter_summary_without_onehot_fields_has_finite_mse() -> None:
    """Scalar-only synths do not emit an undefined mean categorical accuracy."""
    width = param_specs["surge_simple"].encoded_width
    metrics = summarize_parameters(torch.ones(2, width), torch.zeros(2, width), "surge_simple")
    assert metrics
    assert all(key.startswith("field_mse/") and value == 1.0 for key, value in metrics.items())


@pytest.mark.parametrize("consumer", ["flow", "slap"])
def test_ablation_common_initial_weights_match_grouped(consumer: str, tmp_path: Path) -> None:
    """Residual controls differ in metadata capacity, not initial shared backbone weights.

    :param consumer: Shipped model family under comparison.
    :param tmp_path: Dataset and run root; initialization does not load artifacts.
    """
    baseline_cfg = build_config(tmp_path, consumer, "grouped", seed=101, steps=100)
    control_cfg = build_config(tmp_path, consumer, "random", seed=101, steps=100)
    torch.manual_seed(101)
    baseline = hydra.utils.instantiate(baseline_cfg.model)
    torch.manual_seed(101)
    control = hydra.utils.instantiate(control_cfg.model)
    control_parameters = dict(control.named_parameters())
    for name, parameter in baseline.named_parameters():
        torch.testing.assert_close(parameter, control_parameters[name], msg=name)
