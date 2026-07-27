"""Tests for the renderer-to-launcher rejection metrics contract."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from synth_setter.pipeline.schemas.render_metrics import (
    RenderRejectionMetrics,
    render_metrics_path,
)


def test_render_rejection_metrics_defaults_to_zero_counts() -> None:
    """An empty report represents a render with no rejected draws."""
    assert RenderRejectionMetrics() == RenderRejectionMetrics(clipped=0, silent=0)


@pytest.mark.parametrize(
    "payload",
    [
        {"clipped": -1, "silent": 0},
        {"clipped": 1.0, "silent": 0},
        {"clipped": "1", "silent": 0},
        {"clipped": 1, "silent": 0, "unexpected": 2},
    ],
)
def test_render_rejection_metrics_invalid_boundary_payload_rejected(
    payload: dict[str, object],
) -> None:
    """Negative, coerced, and extra report fields fail strict validation.

    :param payload: Invalid worker-report payload under test.
    """
    with pytest.raises(ValidationError):
        RenderRejectionMetrics.model_validate(payload)


@pytest.mark.parametrize("lance_path", [Path("shard-000001.lance"), "shard-000001.lance"])
def test_render_metrics_path_path_and_string_inputs_return_sibling_report(
    lance_path: Path | str,
) -> None:
    """Path and string shard inputs resolve to the same adjacent sidecar.

    :param lance_path: One of the API-supported representations for the same shard.
    """
    assert render_metrics_path(lance_path) == Path("shard-000001.lance.render-metrics.json")
