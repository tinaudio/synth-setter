"""Validate browser prediction transport before native rendering."""

import pytest
from pydantic import ValidationError

from synth_setter.evaluation.browser_flow import BrowserPrediction


@pytest.mark.parametrize("params", [[float("nan")], [float("inf")], ["0.5"]])
def test_browser_prediction_invalid_numbers_rejected(params: list[object]) -> None:
    """Nonfinite and coerced values cannot reach the renderer.

    :param params: Untrusted prediction values.
    """
    with pytest.raises(ValidationError):
        BrowserPrediction.model_validate({"token": "session", "params": params})
